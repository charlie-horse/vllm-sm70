# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 paged attention of one-query rows in Triton (sm_70: no tl.dot,
no PDL). ``decode_attention``: every one-query row of a batch over its own
keys, global or sliding (the torch form is attention_torch.py).

Each row walks its keys one cache block at a time from the first key it may
see to its own position, both read on the device: no host sync, and the grid
depends on the tensor shapes alone. Math and rounding as attention_torch.py
``_Softmax``, the reference: q . k rounded to the query dtype, times D^-0.5
and plus the relative bias with a rounding each, then an fp32 running softmax
over the sink logit and the keys. With ``splits`` > 1 (split-KV) the grid is
(rows, heads, splits): a row cuts its key range into at most ``splits``
block-aligned chunks of at least ``min_keys`` keys, each folded into an fp32
partial, and a second launch merges them with the sink.
"""

from __future__ import annotations

import functools

import torch

from ._compat import exp, tl, triton

__all__ = ["MIN_SPLIT_KEYS", "decode_attention", "split_count"]

# Fewest keys a split-KV chunk holds.
MIN_SPLIT_KEYS = 128
# Most chunks per row and head: one tile of partials in the merge.
MAX_SPLITS = 64
# Split-KV grids aim at about this many programs per streaming multiprocessor.
_PROGRAMS_PER_SM = 8


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


@triton.jit
def _split_range(
    rows_ptr,
    pos_ptr,
    slot_ptr,
    index,
    window,
    splits,
    min_keys,
    BLOCK_N: tl.constexpr,
    INDEXED: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """A row's keys [lo, hi) and its chunks: `active` chunks of `chunk` keys
    from the block-aligned `start`; at least one, even with no key."""
    if INDEXED:
        row = tl.load(rows_ptr + index).to(tl.int64)
    else:
        row = index.to(tl.int64)
    pos = tl.load(pos_ptr + row).to(tl.int64)
    lo = tl.where(window > 0, tl.maximum(pos - window + 1, 0), 0)
    hi = pos + 1
    if SLOTS:
        hi = tl.where(tl.load(slot_ptr + row) < 0, lo, hi)
    start = lo // BLOCK_N * BLOCK_N
    span = hi - start
    chunk = (span + splits - 1) // splits
    chunk = (chunk + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    chunk = tl.maximum(chunk, min_keys)
    active = tl.maximum((span + chunk - 1) // chunk, 1)
    return row, pos, lo, hi, start, chunk, active


@triton.jit
def _split_kv_kernel(
    q_ptr,
    rel_ptr,
    sink_ptr,
    cache_ptr,
    out_ptr,
    part_ptr,
    stat_ptr,
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
    splits,
    min_keys,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INDEXED: tl.constexpr,
    SLOTS: tl.constexpr,
):
    index = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    row, pos, lo, hi, start, chunk, active = _split_range(
        rows_ptr, pos_ptr, slot_ptr, index, window, splits, min_keys,
        BLOCK_N, INDEXED, SLOTS,
    )  # fmt: skip
    if split < active:
        dtype = q_ptr.dtype.element_ty
        req = tl.load(seq_ptr + row).to(tl.int64)
        begin = start + split * chunk
        end = tl.minimum(begin + chunk, hi)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < D
        n = tl.arange(0, BLOCK_N)
        q = tl.load(
            q_ptr + row * stride_q0 + head * stride_q1 + d, mask=d_mask, other=0.0
        ).to(tl.float32)
        rel_row = rel_ptr + row * stride_r0 + head * stride_r1
        sink = tl.load(sink_ptr + row * stride_s0 + head).to(tl.float32)
        # The sink is only the starting max here; the merge adds its weight.
        peak = sink
        total = sink * 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        kv_head = cache_ptr + (head // GROUPS).to(tl.int64) * stride_ch
        for key0 in range(begin, end, BLOCK_N):
            key_pos = key0 + n
            valid = (key_pos >= lo) & (key_pos < hi)
            block = tl.load(table_ptr + req * stride_bt + key0 // BLOCK_N).to(tl.int64)
            keys = kv_head + block * stride_cb + n[:, None] * stride_cn + d[None, :]
            kv_mask = valid[:, None] & d_mask[None, :]
            k = tl.load(keys, mask=kv_mask, other=0.0).to(tl.float32)
            v = tl.load(keys + D, mask=kv_mask, other=0.0).to(tl.float32)
            score = tl.sum(k * q[None, :], axis=1).to(dtype).to(tl.float32)
            score = (score * scale).to(dtype).to(tl.float32)
            distance = pos - key_pos
            bias = tl.load(
                rel_row + distance, mask=valid & (distance < extent), other=0.0
            )
            score = (score + bias.to(tl.float32)).to(dtype).to(tl.float32)
            score = tl.where(valid, score, float("-inf"))
            top = tl.maximum(peak, tl.max(score, axis=0))
            probs = exp(score - top)
            carry = exp(peak - top)
            acc = acc * carry + tl.sum(probs[:, None] * v, axis=0)
            total = total * carry + tl.sum(probs, axis=0)
            peak = top
        if active == 1:
            out = acc / (total + exp(sink - peak))
            out_row = out_ptr + row * stride_o0 + head * stride_o1
            tl.store(out_row + d, out.to(out_ptr.dtype.element_ty), mask=d_mask)
        else:
            slot = (index.to(tl.int64) * tl.num_programs(1) + head) * splits + split
            tl.store(part_ptr + slot * D + d, acc, mask=d_mask)
            tl.store(stat_ptr + slot * 2, peak)
            tl.store(stat_ptr + slot * 2 + 1, total)


@triton.jit
def _split_kv_merge_kernel(
    sink_ptr,
    out_ptr,
    part_ptr,
    stat_ptr,
    rows_ptr,
    pos_ptr,
    slot_ptr,
    stride_s0,
    stride_o0,
    stride_o1,
    window,
    splits,
    min_keys,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    INDEXED: tl.constexpr,
    SLOTS: tl.constexpr,
):
    index = tl.program_id(0)
    head = tl.program_id(1)
    row, _pos, _lo, _hi, _start, _chunk, active = _split_range(
        rows_ptr, pos_ptr, slot_ptr, index, window, splits, min_keys,
        BLOCK_N, INDEXED, SLOTS,
    )  # fmt: skip
    if active > 1:
        s = tl.arange(0, BLOCK_S)
        s_mask = s < active
        d = tl.arange(0, BLOCK_D)
        d_mask = d < D
        first = (index.to(tl.int64) * tl.num_programs(1) + head) * splits
        peaks = tl.load(stat_ptr + (first + s) * 2, mask=s_mask, other=float("-inf"))
        totals = tl.load(stat_ptr + (first + s) * 2 + 1, mask=s_mask, other=0.0)
        parts = tl.load(
            part_ptr + (first + s)[:, None] * D + d[None, :],
            mask=s_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        sink = tl.load(sink_ptr + row * stride_s0 + head).to(tl.float32)
        # Every partial's max starts at the sink: top is finite.
        top = tl.maximum(sink, tl.max(peaks, axis=0))
        weights = exp(peaks - top)
        total = exp(sink - top) + tl.sum(totals * weights, axis=0)
        out = tl.sum(parts * weights[:, None], axis=0) / total
        out_row = out_ptr + row * stride_o0 + head * stride_o1
        tl.store(out_row + d, out.to(out_ptr.dtype.element_ty), mask=d_mask)


@functools.cache
def _programs(device: torch.device) -> int:
    """Split-KV program budget of `device`."""
    sms = 80
    if device.type == "cuda":
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _PROGRAMS_PER_SM * sms


def split_count(
    count: int, heads: int, max_keys: int, device: torch.device, block_size: int
) -> int:
    """Chunks per row and head of a split-KV launch: a static count, from
    shapes and a bound on the keys a row reads, never from device values. 1
    when the rows and heads fill the GPU or no row passes one chunk."""
    by_keys = -(-max_keys // max(MIN_SPLIT_KEYS, block_size))
    by_grid = _programs(device) // max(count * heads, 1)
    return max(1, min(by_keys, by_grid, MAX_SPLITS))


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
    split_kv: bool = False,
    max_keys: int | None = None,
) -> torch.Tensor:
    """Attention before the gate of one-query rows into `out` [P, heads, D].

    rows: the batch rows (an int tensor on the device), or an int n for rows
    0..n-1. Each row reads the keys of request ``seq_idx[row]`` up to
    ``positions[row]``, the last ``sliding_window`` of them in a sliding
    layer. padding: rows with a negative slot are zero. split_kv: allow the
    split-KV launches; max_keys bounds the keys of a row (default: the
    block-table width), so a captured graph needs a static bound. Arguments
    otherwise as attention_torch.py ``gather_attention``; other rows of
    `out` are not written."""
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
    splits = 1
    if split_kv:
        if max_keys is None:
            max_keys = metadata.block_table.shape[1] * block_size
        if window:
            max_keys = min(max_keys, window + block_size - 1)
        splits = split_count(count, heads, max_keys, query.device, block_size)
    rows_arg = rows if indexed else metadata.positions
    common = {
        "D": head_dim,
        "BLOCK_N": block_size,
        "BLOCK_D": triton.next_power_of_2(head_dim),
        "INDEXED": indexed,
        "SLOTS": padding,
    }
    if splits == 1:
        _decode_attention_kernel[(count, heads)](
            query, rel_logits, sinks, kv_cache, out, rows_arg,
            metadata.positions, metadata.seq_idx, metadata.slot_mapping,
            metadata.block_table,
            query.stride(0), query.stride(1), rel_logits.stride(0),
            rel_logits.stride(1), sinks.stride(0), out.stride(0), out.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            metadata.block_table.stride(0), rel_logits.shape[-1], window,
            head_dim**-0.5,
            GROUPS=heads // kv_cache.shape[1], **common,
            num_warps=4, enable_fp_fusion=False,
        )  # fmt: skip
        return out
    min_keys = max(MIN_SPLIT_KEYS, block_size)
    # Fixed by the shapes: the same scratch size on every replay.
    part = torch.empty(
        count * heads * splits * head_dim, dtype=torch.float32, device=out.device
    )
    stat = torch.empty(
        count * heads * splits * 2, dtype=torch.float32, device=out.device
    )
    _split_kv_kernel[(count, heads, splits)](
        query, rel_logits, sinks, kv_cache, out, part, stat, rows_arg,
        metadata.positions, metadata.seq_idx, metadata.slot_mapping,
        metadata.block_table,
        query.stride(0), query.stride(1), rel_logits.stride(0),
        rel_logits.stride(1), sinks.stride(0), out.stride(0), out.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        metadata.block_table.stride(0), rel_logits.shape[-1], window,
        head_dim**-0.5, splits, min_keys,
        GROUPS=heads // kv_cache.shape[1], **common,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    _split_kv_merge_kernel[(count, heads)](
        sinks, out, part, stat, rows_arg, metadata.positions,
        metadata.slot_mapping, sinks.stride(0), out.stride(0), out.stride(1),
        window, splits, min_keys,
        BLOCK_S=triton.next_power_of_2(splits), **common,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out
