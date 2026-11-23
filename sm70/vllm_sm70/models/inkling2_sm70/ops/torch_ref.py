# SPDX-License-Identifier: Apache-2.0
"""Dense torch reference ops of Inkling 2.

Each function is one module of the Hugging Face reference
(transformers `models/inkling2_loop/modeling_inkling2_loop.py`) as a pure
function of tensors: no cache, no paging, whole sequences. Casts sit where the
reference has them, so a result has the dtype the reference returns for the
same input dtypes. Masks are boolean.

Shapes: B batch, T tokens, H hidden, C conv channels, K conv taps, D head dim.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """RMS norm over the last dim in fp32, unweighted when `weight` is None."""
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = weight * y
    return y.to(x.dtype)


def l2_clip(
    x: torch.Tensor,
    threshold: float,
    scale: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shrink vectors of the last dim longer than `threshold` to it.

    `scale` ([heads, D]) multiplies `x` ([..., heads, D]) first, in fp32.
    The result, in the dtype of `x`, is written to `out` when given.
    """
    # The product promotes x to fp32 inside the kernel: no separate cast.
    y = x.float() if scale is None else x * scale.float()
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    y = y * (threshold / (norm + 1e-6)).clamp(max=1.0)
    if out is not None:
        return out.copy_(y)
    return y.to(x.dtype)


def log_scale_tau(
    position_ids: torch.Tensor, alpha: float, n_floor: float
) -> torch.Tensor:
    """tau = 1 + alpha * log(max((position + 1) / n_floor, 1)), fp32."""
    return 1.0 + alpha * torch.log(
        ((position_ids + 1).float() / n_floor).clamp(min=1.0)
    )


def log_scale(
    x: torch.Tensor, position_ids: torch.Tensor, alpha: float, n_floor: float
) -> torch.Tensor:
    """`x` [B, T, heads, D] times tau of `position_ids` [B, T]."""
    tau = log_scale_tau(position_ids, alpha, n_floor)
    return (x.float() * tau[..., None, None]).to(x.dtype)


def _conv_context(
    x: torch.Tensor,
    taps: int,
    valid: torch.Tensor | None,
    history: torch.Tensor | None,
) -> torch.Tensor:
    """[B, T, C, K] windows; entry j is the token K - 1 - j steps back."""
    context = x if valid is None else x * valid[..., None]
    if history is None:
        history = context.new_zeros(x.shape[0], taps - 1, x.shape[-1])
    context = torch.cat([history.float(), context], dim=1)
    return context.unfold(1, taps, 1)


def static_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Static causal conv: (conv(x) + x) / sqrt(K + 1), in the dtype of `x`.

    x [B, T, C]. weight [C, K] or [C, 1, K], taps oldest first: tap K - 1
    weights the current token. valid [B, T] bool zeroes conv inputs, not the
    residual. history [B, K - 1, C] is the conv input before `x`, oldest
    first; zeros when None.
    """
    weight = weight.float().reshape(weight.shape[0], -1)
    taps = weight.shape[-1]
    y = x.float()
    out = (_conv_context(y, taps, valid, history) * weight).sum(-1)
    return ((out + y) / math.sqrt(taps + 1)).to(x.dtype)


def dynamic_conv_taps(
    dynamic_input: torch.Tensor, weight_proj: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Per-token taps [B, T, heads, K] in fp32 from `dynamic_input` [B, T, H]."""
    taps = F.linear(dynamic_input, weight_proj).float()
    return taps.unflatten(-1, (num_heads, -1))


def dynamic_conv(
    x: torch.Tensor,
    static_weight: torch.Tensor,
    taps: torch.Tensor,
    valid: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dynamic causal conv: (conv(x) + x) / sqrt(2K + 1), in the dtype of `x`.

    x [B, T, C]. static_weight [K, C] and taps [B, T, heads, K]
    (`dynamic_conv_taps`), newest first: tap 0 weights the current token.
    A head's taps apply to its C / heads channels. valid and history as in
    `static_conv`.
    """
    width = static_weight.shape[0]
    heads = taps.shape[2]
    y = x.float()
    weights = static_weight.float().T.unflatten(0, (heads, -1)) + taps[:, :, :, None]
    windows = _conv_context(y, width, valid, history).flip(-1)
    out = (windows.unflatten(2, (heads, -1)) * weights).sum(-1).flatten(2)
    return ((out + y) / math.sqrt(2 * width + 1)).to(x.dtype)


def attn_res(
    block_states: torch.Tensor,
    partial_block_states: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Learned-query attention over the block states and the current state.

    block_states [B, T, N, H], partial_block_states [B, T, H], query [H] or
    [Q, H]. Keys and values are the unweighted RMS norm of the N + 1 states.
    Returns [B, T, Q, H] in fp32.
    """
    states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2)
    return attn_res_mix(rms_norm(states.float()), query)


def attn_res_mix(states: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """The mix of `attn_res` over states [..., N, H] that are already
    RMS-normed in fp32. query [H] or [Q, H]. Returns [..., Q, H] in fp32."""
    hidden = states.shape[-1]
    query = query.float().view(-1, hidden)
    scores = torch.matmul(query, states.transpose(-1, -2)) * hidden**-0.5
    return torch.matmul(F.softmax(scores, dim=-1), states)


def relative_position_bias(
    relative_states: torch.Tensor,
    proj: torch.Tensor,
    relative_distance: torch.Tensor,
) -> torch.Tensor:
    """Bias [B, heads, Tq, Tk]: zero outside 0 <= distance < extent.

    relative_states [B, Tq, heads, d_rel] (clipped, and log-scaled in full
    layers), proj [d_rel, extent], relative_distance [Tq, Tk] = query position
    minus key position.
    """
    extent = proj.shape[-1]
    logits = (relative_states @ proj).transpose(1, 2)
    index = relative_distance.clamp(0, extent - 1).expand(*logits.shape[:2], -1, -1)
    outside = (relative_distance < 0) | (relative_distance >= extent)
    return logits.gather(-1, index).masked_fill(outside, 0.0)


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor | None,
    scaling: float,
    position_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention with an fp32 softmax over the keys and one sink column.

    query [B, heads, Tq, D]; key, value [B, kv_heads, Tk, D]; allowed bool,
    broadcastable to [B, heads, Tq, Tk], true where the query may attend;
    position_bias [B, heads, Tq, Tk]; sinks [B, heads, Tq], a logit whose
    value is zero. Returns the output [B, Tq, heads * D] and the weights
    [B, heads, Tq, Tk], both in the dtype of `query`.
    """
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if position_bias is not None:
        scores = scores + position_bias
    scores = scores.float()
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    if sinks is not None:
        scores = torch.cat([scores, sinks.float().unsqueeze(-1)], dim=-1)
        weights = F.softmax(scores, dim=-1)[..., :-1]
    else:
        weights = F.softmax(scores, dim=-1)
    weights = weights.to(query.dtype)
    out = torch.matmul(weights, value).transpose(1, 2)
    return out.flatten(2), weights


def gated_output(
    attn_output: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    out_weight: torch.Tensor,
) -> torch.Tensor:
    """Output projection of `attn_output` times sigmoid(gate of the layer input)."""
    gate = torch.sigmoid(F.linear(hidden_states, gate_weight).float())
    return F.linear((attn_output.float() * gate).to(attn_output.dtype), out_weight)


@dataclasses.dataclass(frozen=True)
class AttentionWeights:
    """Weights of one attention layer. Dynamic convs when `k_conv_proj` is set."""

    q: torch.Tensor  # [heads * D, H]
    k: torch.Tensor  # [kv_heads * D, H]
    v: torch.Tensor  # [kv_heads * D, H]
    rel: torch.Tensor  # [heads * d_rel, H]
    out: torch.Tensor  # [H, heads * D]
    out_gate: torch.Tensor  # [heads * D, H]
    sink: torch.Tensor  # [heads, H]
    q_scale: torch.Tensor  # [heads, D]
    rel_scale: torch.Tensor  # [heads, d_rel]
    rel_proj: torch.Tensor  # [d_rel, extent]
    k_conv: torch.Tensor  # static [C, K] or [C, 1, K]; dynamic [K, C]
    v_conv: torch.Tensor
    k_conv_proj: torch.Tensor | None = None  # [kv_heads * K, H]
    v_conv_proj: torch.Tensor | None = None


def query_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    threshold: float,
    position_ids: torch.Tensor | None = None,
    log_scaling: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Scaled, clipped per-head states [B, T, heads, D] of a linear projection.

    `log_scaling` (alpha, n_floor) multiplies by tau of `position_ids` after
    the clip. Serves the queries and the relative states.
    """
    states = F.linear(hidden_states, weight).unflatten(-1, scale.shape)
    states = l2_clip(states, threshold, scale)
    if log_scaling is not None:
        states = log_scale(states, position_ids, *log_scaling)
    return states


def key_value_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    conv: torch.Tensor,
    num_heads: int,
    conv_proj: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    eps: float = 1e-6,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Key or value states [B, heads, T, D]: linear, conv, then the norm.

    The conv is dynamic when `conv_proj` is given. The norm is the L2 clip of
    a key when `clip_threshold` is given, else the unweighted RMS norm of a
    value.
    """
    states = F.linear(hidden_states, weight)
    if conv_proj is None:
        states = static_conv(states, conv, valid)
    else:
        taps = dynamic_conv_taps(hidden_states, conv_proj, num_heads)
        states = dynamic_conv(states, conv, taps, valid)
    states = states.unflatten(-1, (num_heads, -1))
    if clip_threshold is None:
        states = rms_norm(states, eps=eps)
    else:
        states = l2_clip(states, clip_threshold)
    return states.transpose(1, 2)


def attention(
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    value_input: torch.Tensor,
    w: AttentionWeights,
    *,
    num_kv_heads: int,
    allowed: torch.Tensor | None,
    relative_distance: torch.Tensor,
    position_ids: torch.Tensor,
    qk_clip: float,
    rel_clip: float,
    eps: float = 1e-6,
    log_scaling: tuple[float, float] | None = None,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One attention module over whole sequences: [B, T, H] in and out.

    Sliding layers pass the same normed state three times and
    `log_scaling=None`; full layers pass (alpha, n_floor). The bias, the sink
    and the output gate read `query_input`. Returns the output and the
    attention weights.
    """
    head_dim = w.q_scale.shape[-1]
    query = query_projection(
        query_input, w.q, w.q_scale, qk_clip, position_ids, log_scaling
    ).transpose(1, 2)
    key = key_value_projection(
        key_input, w.k, w.k_conv, num_kv_heads, w.k_conv_proj, qk_clip, valid=valid
    )
    value = key_value_projection(
        value_input, w.v, w.v_conv, num_kv_heads, w.v_conv_proj, eps=eps, valid=valid
    )
    relative_states = query_projection(
        query_input, w.rel, w.rel_scale, rel_clip, position_ids, log_scaling
    )
    bias = relative_position_bias(relative_states, w.rel_proj, relative_distance)
    sinks = F.linear(query_input, w.sink).transpose(1, 2)
    out, weights = dense_attention(
        query, key, value, allowed, head_dim**-0.5, bias, sinks
    )
    return gated_output(out, query_input, w.out_gate, w.out), weights


def router(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 routing of `x` [..., H]: (logits [N, E], weights [N, k], indices [N, k]).

    Experts are the top k of sigmoid(logits) + bias. Weights are the sigmoid
    scores without the bias, normalised over the chosen k, times `route_scale`.
    """
    logits = F.linear(x.float().reshape(-1, weight.shape[-1]), weight.float())
    scores = torch.sigmoid(logits)
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * route_scale, indices


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """cap * tanh(x / cap)."""
    return torch.tanh(x / cap) * cap


def softcapped_swiglu(
    gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    """softcap(gate) * sigmoid(gate) * softcap(up) in fp32, in the dtype of `gate`."""
    g, u = gate.float(), up.float()
    return (softcap(g, gate_cap) * torch.sigmoid(g) * softcap(u, up_cap)).to(gate.dtype)


def experts(
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """Weighted sum of the chosen experts, summed in fp32, in the dtype of `x`.

    x [N, H]; top_k_index, top_k_weights [N, k] (`router`); gate_up
    [E, 2 * I, H] with the gate rows first; down [E, H, I].
    """
    total = torch.zeros_like(x, dtype=torch.float32)
    for expert in top_k_index.unique().tolist():
        token, slot = torch.where(top_k_index == expert)
        gate, up = F.linear(x[token], gate_up[expert]).chunk(2, dim=-1)
        out = F.linear(softcapped_swiglu(gate, up, gate_cap, up_cap), down[expert])
        out = out * top_k_weights[token, slot, None]
        total.index_add_(0, token, out.float())
    return total.to(x.dtype)


def message_gate(
    hidden_states: torch.Tensor,
    message: torch.Tensor,
    down_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    gain_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """(1 - tanh(forget)) * hidden_states + tanh(gain) * message, in fp32.

    Both gates are projections of down_weight [rank, 2 * H] applied to the
    unweighted RMS norms of the two inputs, concatenated. forget_weight and
    gain_weight are [H, rank].
    """
    h, m = hidden_states.float(), message.float()
    z = torch.cat([rms_norm(h, eps=eps), rms_norm(m, eps=eps)], dim=-1)
    z = F.linear(z.to(down_weight.dtype), down_weight).float()
    forget = torch.tanh(F.linear(z, forget_weight.float()))
    gain = torch.tanh(F.linear(z, gain_weight.float()))
    return (1 - forget) * h + gain * m


def logit_cap(logits: torch.Tensor, cap: float = 30.0) -> torch.Tensor:
    """fp32 logits bounded to (-cap, cap)."""
    return softcap(logits.float(), cap)
