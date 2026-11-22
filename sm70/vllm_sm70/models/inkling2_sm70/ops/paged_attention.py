# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 paged attention of one-query rows as a single Triton kernel
(sm_70: no tl.dot, no PDL).

    decode_attention   one launch: every one-query row of a batch over its
                       own keys, global or sliding    attention_torch.py

One program per (row, query head). It walks the row's keys one cache block
at a time from the first key it may see to its own position, both read on
the device, so a step costs the row's real context and the grid depends on
the tensor shapes alone: no host sync. Dot products and the value sum are
GEMV-style ``tl.sum`` reductions. Math and rounding as attention_torch.py
``_Softmax``, which stays the reference: q . k rounded to the query dtype,
times D^-0.5 and plus the relative bias with a rounding each, then an fp32
running softmax that starts at the sink logit.
"""

from __future__ import annotations

import torch

from ._compat import tl, triton

__all__ = ["decode_attention"]


@triton.jit
def _decode_attention_kernel(
    q_ptr,
    rel_ptr,
    sink_ptr,
    cache_ptr,
    out_ptr,
    rows_ptr,
    pos_ptr,
    seq_ptr,
    slot_ptr,
    table_ptr,
    stride_q0,
    stride_q1,
    stride_r0,
    stride_r1,
    stride_s0,
    stride_o0,
    stride_o1,
    stride_cb,
    stride_ch,
    stride_cn,
    stride_bt,
    extent,
    window,
    scale,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INDEXED: tl.constexpr,
    SLOTS: tl.constexpr,
):
    if INDEXED:
        row = tl.load(rows_ptr + tl.program_id(0)).to(tl.int64)
    else:
        row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dtype = q_ptr.dtype.element_ty
    pos = tl.load(pos_ptr + row).to(tl.int64)
    req = tl.load(seq_ptr + row).to(tl.int64)
    lo = tl.where(window > 0, tl.maximum(pos - window + 1, 0), 0)
    hi = pos + 1
    if SLOTS:
        hi = tl.where(tl.load(slot_ptr + row) < 0, lo, hi)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    n = tl.arange(0, BLOCK_N)
    q = tl.load(q_ptr + row * stride_q0 + head * stride_q1 + d, mask=d_mask, other=0.0)
    q = q.to(tl.float32)
    rel_row = rel_ptr + row * stride_r0 + head * stride_r1
    peak = tl.load(sink_ptr + row * stride_s0 + head).to(tl.float32)
    total = peak * 0.0 + 1.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    kv_head = cache_ptr + (head // GROUPS).to(tl.int64) * stride_ch
    for start in range(lo // BLOCK_N * BLOCK_N, hi, BLOCK_N):
        key_pos = start + n
        valid = (key_pos >= lo) & (key_pos < hi)
        block = tl.load(table_ptr + req * stride_bt + start // BLOCK_N).to(tl.int64)
        keys = kv_head + block * stride_cb + n[:, None] * stride_cn + d[None, :]
        kv_mask = valid[:, None] & d_mask[None, :]
        k = tl.load(keys, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(keys + D, mask=kv_mask, other=0.0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1).to(dtype).to(tl.float32)
        score = (score * scale).to(dtype).to(tl.float32)
        distance = pos - key_pos
        bias = tl.load(rel_row + distance, mask=valid & (distance < extent), other=0.0)
        score = (score + bias.to(tl.float32)).to(dtype).to(tl.float32)
        score = tl.where(valid, score, float("-inf"))
        top = tl.maximum(peak, tl.max(score, axis=0))
        probs = tl.exp(score - top)
        carry = tl.exp(peak - top)
        acc = acc * carry + tl.sum(probs[:, None] * v, axis=0)
        total = total * carry + tl.sum(probs, axis=0)
        peak = top
    out = acc / total
    out_row = out_ptr + row * stride_o0 + head * stride_o1
    tl.store(out_row + d, out.to(out_ptr.dtype.element_ty), mask=d_mask)


def decode_attention(
    query: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata,
    out: torch.Tensor,
    *,
    rows: torch.Tensor | int,
    padding: bool,
) -> torch.Tensor:
    """Attention before the gate of one-query rows into `out` [P, heads, D].

    rows: the batch rows (an int tensor on the device), or an int n for rows
    0..n-1. Each row reads the keys of request ``seq_idx[row]`` up to
    ``positions[row]``, the last ``sliding_window`` of them in a sliding
    layer. padding: rows with a negative slot are zero. Arguments otherwise
    as attention_torch.py ``gather_attention``; other rows of `out` are not
    written."""
    heads, head_dim = query.shape[1], query.shape[2]
    block_size = kv_cache.shape[2]
    count = rows if isinstance(rows, int) else rows.shape[0]
    if count == 0:
        return out
    assert query.stride(-1) == rel_logits.stride(-1) == sinks.stride(-1) == 1
    assert kv_cache.stride(-1) == 1 and out.stride(-1) == 1
    assert block_size == triton.next_power_of_2(block_size)
    indexed = not isinstance(rows, int)
    window = metadata.sliding_window or 0
    grid = (count, heads)
    _decode_attention_kernel[grid](
        query, rel_logits, sinks, kv_cache, out,
        rows if indexed else metadata.positions,
        metadata.positions, metadata.seq_idx, metadata.slot_mapping,
        metadata.block_table,
        query.stride(0), query.stride(1), rel_logits.stride(0),
        rel_logits.stride(1), sinks.stride(0), out.stride(0), out.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        metadata.block_table.stride(0), rel_logits.shape[-1], window,
        head_dim**-0.5,
        D=head_dim, GROUPS=heads // kv_cache.shape[1], BLOCK_N=block_size,
        BLOCK_D=triton.next_power_of_2(head_dim), INDEXED=indexed,
        SLOTS=padding, num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out
