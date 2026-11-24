# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the attention prologue of one layer visit, in torch.

It produces what the attention op reads and stores the batch's keys and values:

    q_out    clip(q * q_scale), times tau in a global layer    [T, heads, D]
    rel_out  clip(r * rel_scale) (times tau) @ rel_proj        [T, heads, extent]
    K cache  clip(conv(k)) per KV head, at the tokens' attention slots
    V cache  rms_norm(conv(v)) per KV head, same slots

Clips, norms and tau run in fp32 and round to the input dtype where the
reference does. The K and V convs use streams STREAM_K and STREAM_V of the
visit's conv state. rel_out[t, h, d] is the bias of query t on the key d
tokens back. The K/V cache is [blocks, kv_heads, tokens, 2 * D]: keys, then
values per head. Kernel routes: sconv.py, fused_small.py and visit_fusion.py.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as F
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from . import gemv
from .backend import cache_metadata, is_decode_only
from .cache import STREAM_K, STREAM_V
from .fused_small import on_fused
from .ops import paged_conv, small_ops
from .ops import visit as visit_ops
from .ops.torch_ref import l2_clip, log_scale_tau, rms_norm
from .sconv import (
    _block_tokens,
    _real_rows,
    _store_rows,
    dynamic_conv,
    on_triton,
    static_conv,
)
from .visit_fusion import on_decode
from .weights import rows_view

__all__ = [
    "STACKED_MAX_TOKENS",
    "attention_prologue",
    "key_value_states",
    "project_packed",
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
    (`x_k` and `x_v` None) and runs one GEMM; with three inputs, up to 8
    tokens run one GEMV launch on the route of gemv.py."""
    if x_k is None and x_v is None:
        return gemv.linear(x_q, weight).split(sizes, dim=-1)
    query_rows = sum(sizes[:4])
    y = gemv.segments((x_q, x_k, x_v), weight, (query_rows, query_rows + sizes[4]))
    if y is not None:
        return y.split(sizes, dim=-1)
    w_query, w_k, w_v = weight.split((query_rows, sizes[4], sizes[5]), dim=0)
    q, r, gate, sink = F.linear(x_q, w_query).split(sizes[:4], dim=-1)
    return q, r, gate, sink, F.linear(x_k, w_k), F.linear(x_v, w_v)


# Three inputs run as one GEMM up to this many tokens (``project_packed``).
STACKED_MAX_TOKENS = 8


def _stacked(*xs: torch.Tensor) -> torch.Tensor:
    """`xs` [T, H] stacked along dim 0: a view when they lie back to back."""
    view = rows_view(*xs)
    return torch.cat(xs) if view is None else view


def project_packed(
    x_q: torch.Tensor,
    x_k: torch.Tensor | None,
    x_v: torch.Tensor | None,
    weight: torch.Tensor,
    sizes: tuple[int, ...],
    tap_rows: int = 0,
) -> tuple[torch.Tensor | None, ...]:
    """(q, r, gate, sink, k, v, k_taps, v_taps) of a packed `weight`: the
    rows of ``attn.qkvr.weight`` (`sizes`), then `tap_rows` rows each of the
    key and value tap projections of a global layer (taps None when 0).

    One GEMM: with three inputs over [x_q; x_k; x_v], keeping each input's
    own columns (q, r, gate and sink of x_q; k and its taps of x_k; v and its
    taps of x_v). Each output is the GEMM of its own input and rows. On the
    route of gemv.py, one GEMV launch reads each row segment's own input."""
    splits = (*sizes, tap_rows, tap_rows)
    bounds = tuple(itertools.accumulate(splits))[3 : 5 if not tap_rows else 7]
    if x_k is None and x_v is None:
        parts = gemv.linear(x_q, weight).split(splits, dim=-1)
    elif (y := gemv.segments((x_q, x_k, x_v), weight, bounds)) is not None:
        parts = y.split(splits, dim=-1)
    else:
        assert x_k is not None and x_v is not None
        y = F.linear(_stacked(x_q, x_k, x_v), weight)
        by_q, by_k, by_v = (
            rows.split(splits, dim=-1) for rows in y.split(x_q.shape[0])
        )
        parts = (*by_q[:4], by_k[4], by_v[5], by_k[6], by_v[7])
    if not tap_rows:
        return (*parts[:6], None, None)
    return tuple(parts)


def query_states(
    x: torch.Tensor,
    scale: torch.Tensor,
    threshold: float,
    tau: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled, clipped per-head states [T, heads, D] of `x` [T, heads * D],
    in the dtype of `x`; `tau` [T] fp32 multiplies the rounded clip. Written
    to `out` when given. Serves q and r."""
    if tau is None:
        return l2_clip(x.unflatten(-1, scale.shape), threshold, scale, out=out)
    states = l2_clip(x.unflatten(-1, scale.shape), threshold, scale)
    scaled = states * tau[:, None, None]
    if out is not None:
        return out.copy_(scaled)
    return scaled.to(states.dtype)


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
    decode_only: bool = False,
) -> None:
    """Store `key`, `value` [n, kv_heads, D] of the real tokens at their
    slots of the K/V cache; tokens with a negative slot are skipped, without
    a host sync in a decode-only batch (sconv.py)."""
    block_tokens = kv_cache.shape[2]
    head_dim = key.shape[-1]
    slots = slot_mapping[: key.shape[0]]
    if decode_only:
        rows = torch.cat([key, value], dim=-1).to(kv_cache.dtype)
        _store_rows(kv_cache, slots, rows)
        return
    rows = (slots >= 0).nonzero().squeeze(1)
    slots = slots[rows]
    block, token = slots // block_tokens, slots % block_tokens
    kv_cache[block, :, token, :head_dim] = key[rows].to(kv_cache.dtype)
    kv_cache[block, :, token, head_dim:] = value[rows].to(kv_cache.dtype)


def _query_rel_torch(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    rel_scale: torch.Tensor,
    rel_proj: torch.Tensor,
    attn_metadata: Any,
    q_out: torch.Tensor,
    rel_out: torch.Tensor,
    n: int,
    qk_clip: float,
    rel_clip: float,
    log_scaling: tuple[float, float] | None,
) -> None:
    """`q_out` and `rel_out` of `attention_prologue` on the torch ops."""
    tau = None
    if log_scaling is not None:
        tau = log_scale_tau(attn_metadata.positions[:n], *log_scaling)
    query_states(q[:n], q_scale, qk_clip, tau, out=q_out[:n])
    q_out[n:] = 0
    torch.matmul(
        query_states(r[:n], rel_scale, rel_clip, tau), rel_proj, out=rel_out[:n]
    )
    rel_out[n:] = 0
    if is_decode_only(attn_metadata):
        padding = (attn_metadata.slot_mapping[:n] < 0)[:, None, None]
        q_out[:n].masked_fill_(padding, 0.0)
        rel_out[:n].masked_fill_(padding, 0.0)


def _check_conv_rows(k: torch.Tensor, conv_metadata: Any, n: int) -> None:
    if _real_rows(k, conv_metadata) != n:
        raise ValueError(
            f"vllm_sm70: attention metadata of {n} tokens, conv-state metadata "
            f"of {_real_rows(k, conv_metadata)}."
        )


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

    q [T, heads * D], r [T, heads * d_rel], k and v [T, kv_heads * D] come
    from ``project_qkvr``; q_scale [heads, D], rel_scale [heads, d_rel],
    rel_proj [d_rel, extent]. A sliding layer passes static conv weights
    [C, K] or [C, 1, K]; a global layer passes static parts [K, C], per-token
    taps [T, kv_heads * K] and `log_scaling` (alpha, n_floor). Writes `q_out`
    [T, heads, D] and `rel_out` [T, heads, extent] and stores K, V and the conv
    inputs. Rows from n = ``num_actual_tokens`` on (the slot-mapping rows of a
    decode-only batch) are zero in the outputs and not stored. A forward
    without metadata stores nothing and applies no tau.
    """
    attn_metadata = cache_metadata(attn_owner)
    conv_metadata = cache_metadata(conv_owner)
    if attn_metadata is None:
        query_states(q, q_scale, qk_clip, out=q_out)
        torch.matmul(query_states(r, rel_scale, rel_clip), rel_proj, out=rel_out)
        return
    n = _real_rows(q, attn_metadata)
    if n > attn_metadata.positions.shape[0]:
        raise ValueError(
            f"vllm_sm70: metadata of {n} tokens "
            f"({attn_metadata.positions.shape[0]} positions) for {q.shape[0]} rows."
        )
    if attn_owner.kv_cache.shape[2] != attn_metadata.block_size:
        raise ValueError(
            f"vllm_sm70: K/V cache blocks hold {attn_owner.kv_cache.shape[2]} "
            f"tokens, the metadata has block size {attn_metadata.block_size}."
        )
    decode_only = is_decode_only(attn_metadata)
    fused = on_fused(q, decode_only)
    if (
        fused
        and conv_metadata is not None
        and on_triton(k)
        and on_decode(q, decode_only)
        and conv_owner.kv_cache.shape[1] * q_scale.shape[1] == k.shape[1]
    ):
        _check_conv_rows(k, conv_metadata, n)
        _block_tokens(conv_owner.kv_cache, conv_metadata)
        visit_ops.prologue(
            q, r, q_scale, rel_scale, rel_proj, attn_metadata, q_out, rel_out,
            k, v, k_conv, v_conv, k_taps, v_taps, conv_owner.kv_cache,
            conv_metadata, conv_owner.stream_ranges[STREAM_K],
            conv_owner.stream_ranges[STREAM_V], attn_owner.kv_cache, n,
            qk_clip=qk_clip, rel_clip=rel_clip, eps=eps, log_scaling=log_scaling,
        )  # fmt: skip
        return
    if fused:
        small_ops.query_rel(
            q, r, q_scale, rel_scale, rel_proj, attn_metadata.slot_mapping,
            attn_metadata.positions, q_out, rel_out, n,
            q_clip=qk_clip, r_clip=rel_clip, log_scaling=log_scaling,
        )  # fmt: skip
    else:
        _query_rel_torch(
            q, r, q_scale, rel_scale, rel_proj, attn_metadata, q_out, rel_out, n,
            qk_clip, rel_clip, log_scaling,
        )  # fmt: skip
    if conv_metadata is None:
        return
    _check_conv_rows(k, conv_metadata, n)
    k_stream = conv_owner.stream_ranges[STREAM_K]
    v_stream = conv_owner.stream_ranges[STREAM_V]
    if on_triton(k):
        _block_tokens(conv_owner.kv_cache, conv_metadata)
        paged_conv.key_value_conv(
            k,
            v,
            k_conv,
            v_conv,
            k_taps,
            v_taps,
            conv_owner.kv_cache,
            conv_metadata,
            k_stream,
            v_stream,
            attn_owner.kv_cache,
            attn_metadata.slot_mapping,
            n,
            qk_clip=qk_clip,
            eps=eps,
        )
        return
    key, value = key_value_states(
        k[:n],
        v[:n],
        k_conv,
        v_conv,
        conv_owner.kv_cache,
        conv_metadata,
        k_stream,
        v_stream,
        qk_clip=qk_clip,
        eps=eps,
        k_taps=None if k_taps is None else k_taps[:n],
        v_taps=None if v_taps is None else v_taps[:n],
    )
    store_key_value(
        key,
        value,
        attn_owner.kv_cache,
        attn_metadata.slot_mapping,
        is_decode_only(attn_metadata),
    )
