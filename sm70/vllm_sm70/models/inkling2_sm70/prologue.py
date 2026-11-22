# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the attention prologue of one layer visit, in torch.

From the projections of the layer input it produces what the attention op
reads, and stores this batch's keys and values:

    q_out    clip(q * q_scale), times tau in a global layer    [T, heads, D]
    rel_out  clip(r * rel_scale) (times tau) @ rel_proj        [T, heads, extent]
    K cache  clip(conv(k)) per KV head, at the tokens' attention slots
    V cache  rms_norm(conv(v)) per KV head, same slots

Clips, norms and tau run in fp32 and round to the input dtype where the
reference does (transformers modeling_inkling2_loop.py: scaled L2 clip, log
scale, key/value projection). The K and V convs read and store streams
STREAM_K and STREAM_V of the visit's conv state (sconv.py). rel_out[t, h, d]
is the bias of query t on the key d tokens back.

The K/V cache is vLLM's [blocks, kv_heads, tokens, 2 * D]: keys in the first
D values of a head, values in the rest.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .backend import cache_metadata
from .cache import STREAM_K, STREAM_V
from .ops.torch_ref import l2_clip, log_scale_tau, rms_norm
from .sconv import dynamic_conv, static_conv

__all__ = [
    "attention_prologue",
    "key_value_states",
    "project_qkvr",
    "query_states",
    "store_key_value",
]


def project_qkvr(
    x_q: torch.Tensor,
    x_k: torch.Tensor | None,
    x_v: torch.Tensor | None,
    weight: torch.Tensor,
    sizes: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    """(q, r, gate, sink, k, v) of the fused ``attn.qkvr.weight``; `sizes` are
    its row counts (``weights.qkvr_sizes``). A local layer passes one input
    (`x_k` and `x_v` None) and runs one GEMM."""
    if x_k is None and x_v is None:
        return F.linear(x_q, weight).split(sizes, dim=-1)
    query_rows = sum(sizes[:4])
    w_query, w_k, w_v = weight.split((query_rows, sizes[4], sizes[5]), dim=0)
    q, r, gate, sink = F.linear(x_q, w_query).split(sizes[:4], dim=-1)
    return q, r, gate, sink, F.linear(x_k, w_k), F.linear(x_v, w_v)


def query_states(
    x: torch.Tensor,
    scale: torch.Tensor,
    threshold: float,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled, clipped per-head states [T, heads, D] of `x` [T, heads * D];
    `tau` [T] fp32 multiplies the rounded clip. Serves q and r."""
    states = l2_clip(x.unflatten(-1, scale.shape), threshold, scale)
    if tau is not None:
        states = (states.float() * tau[:, None, None]).to(states.dtype)
    return states


def key_value_states(
    k: torch.Tensor,
    v: torch.Tensor,
    k_conv: torch.Tensor,
    v_conv: torch.Tensor,
    conv_cache: torch.Tensor,
    conv_metadata: Any,
    k_stream: tuple[int, int],
    v_stream: tuple[int, int],
    *,
    qk_clip: float,
    eps: float,
    k_taps: torch.Tensor | None = None,
    v_taps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys and values [T, kv_heads, D] of the projections `k`, `v`
    [T, kv_heads * D]: conv over the conv-state cache (dynamic when the taps
    are given), then the L2 clip of a key and the RMS norm of a value."""
    heads = conv_cache.shape[1]
    if k_taps is None:
        k = static_conv(k, k_conv, conv_cache, conv_metadata, k_stream)
        v = static_conv(v, v_conv, conv_cache, conv_metadata, v_stream)
    else:
        k = dynamic_conv(k, k_conv, k_taps, conv_cache, conv_metadata, k_stream)
        v = dynamic_conv(v, v_conv, v_taps, conv_cache, conv_metadata, v_stream)
    key = l2_clip(k.unflatten(-1, (heads, -1)), qk_clip)
    value = rms_norm(v.unflatten(-1, (heads, -1)), eps=eps)
    return key, value


def store_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store `key`, `value` [T, kv_heads, D] at the tokens' slots of the K/V
    cache; tokens with a negative slot are skipped."""
    block_tokens = kv_cache.shape[2]
    head_dim = key.shape[-1]
    slots = slot_mapping[: key.shape[0]]
    rows = (slots >= 0).nonzero().squeeze(1)
    slots = slots[rows]
    block, token = slots // block_tokens, slots % block_tokens
    kv_cache[block, :, token, :head_dim] = key[rows].to(kv_cache.dtype)
    kv_cache[block, :, token, head_dim:] = value[rows].to(kv_cache.dtype)


@eager_break_during_capture
def attention_prologue(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    rel_scale: torch.Tensor,
    rel_proj: torch.Tensor,
    k_conv: torch.Tensor,
    v_conv: torch.Tensor,
    k_taps: torch.Tensor | None,
    v_taps: torch.Tensor | None,
    attn_owner: Any,
    conv_owner: Any,
    q_out: torch.Tensor,
    rel_out: torch.Tensor,
    *,
    qk_clip: float,
    rel_clip: float,
    eps: float,
    log_scaling: tuple[float, float] | None = None,
) -> None:
    """Prologue of one layer visit over the caches of `attn_owner`
    (``Inkling2AttentionCache``) and `conv_owner` (``Inkling2ConvState``).

    q [T, heads * D], r [T, heads * d_rel], k and v [T, kv_heads * D] are the
    projections of ``project_qkvr``; q_scale [heads, D], rel_scale
    [heads, d_rel], rel_proj [d_rel, extent]. A sliding layer passes static
    conv weights [C, K] or [C, 1, K] and no taps; a global layer passes the
    static parts [K, C], the per-token taps [T, kv_heads * K] and
    `log_scaling` (alpha, n_floor). Writes `q_out` [T, heads, D] and `rel_out`
    [T, heads, extent] and stores K, V and the conv inputs.

    The metadata is the current forward's. A forward without it stores
    nothing and applies no tau.
    """
    attn_metadata = cache_metadata(attn_owner)
    conv_metadata = cache_metadata(conv_owner)
    tau = None
    if log_scaling is not None and attn_metadata is not None:
        positions = attn_metadata.positions[: q.shape[0]]
        tau = log_scale_tau(positions, *log_scaling)
    q_out.copy_(query_states(q, q_scale, qk_clip, tau))
    torch.matmul(query_states(r, rel_scale, rel_clip, tau), rel_proj, out=rel_out)
    if attn_metadata is None or conv_metadata is None:
        return
    key, value = key_value_states(
        k,
        v,
        k_conv,
        v_conv,
        conv_owner.kv_cache,
        conv_metadata,
        conv_owner.stream_ranges[STREAM_K],
        conv_owner.stream_ranges[STREAM_V],
        qk_clip=qk_clip,
        eps=eps,
        k_taps=k_taps,
        v_taps=v_taps,
    )
    store_key_value(key, value, attn_owner.kv_cache, attn_metadata.slot_mapping)
