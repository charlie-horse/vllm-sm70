# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 paged short convs as single Triton kernels (sm_70: no tl.dot,
no PDL, no inline tanh).

    short_conv       one launch: cache read, taps, conv, residual, cache
                     write, zero padding rows                    sconv.py
    key_value_conv   one launch: the K and V convs of a prologue, the L2 clip
                     of K, the RMS norm of V and the K/V cache store
                                                                 prologue.py

Math, tap rule and rounding as the torch ops of sconv.py and prologue.py,
which stay the reference. A program reads its taps before it stores its
rows; a tap from the cache is a position before its request's first batch
row, which no program of the launch writes. Metadata is read on the device
only: no host sync, and grids and buffers depend on the tensor shapes alone.
`num_tokens` (n) is a kernel argument; rows from n on are zero in the
output and stored nowhere.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ._compat import tl, triton

__all__ = ["key_value_conv", "short_conv"]


@triton.jit
def _conv_tile(
    x_ptr,
    stride_x,
    w_ptr,
    stride_w0,
    stride_w1,
    dyn_ptr,
    stride_dyn,
    cache_ptr,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    offset,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    stride_bt,
    max_blocks,
    num_tokens,
    rows,
    cols,
    col_mask,
    divisor,
    store,
    WS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DYNAMIC: tl.constexpr,
):
    """fp32 conv of `rows` x `cols`, before the cast to the dtype of x;
    stores the conv inputs of the tile's real rows with a slot when the
    scalar `store` holds."""
    real = rows < num_tokens
    slot = tl.load(slot_ptr + rows, mask=real, other=-1)
    valid = real & (slot >= 0)
    pos = tl.load(pos_ptr + rows, mask=real, other=0)
    req = tl.load(seq_ptr + rows, mask=real, other=0).to(tl.int64)
    qstart = tl.load(qstart_ptr + rows, mask=real, other=0)
    head = tl.minimum(cols // WS, H - 1)
    cache_col = head * stride_ch + offset + cols % WS
    dtype = cache_ptr.dtype.element_ty
    row_col = real[:, None] & col_mask[None, :]
    x_rows = x_ptr + rows.to(tl.int64)[:, None] * stride_x + cols[None, :]
    cur = tl.load(x_rows, mask=row_col, other=0.0).to(dtype)
    acc = tl.zeros(cur.shape, dtype=tl.float32)
    for k in tl.static_range(W):
        lag = k if DYNAMIC else W - 1 - k
        if lag == 0:
            tap = tl.where(valid[:, None], cur.to(tl.float32), 0.0)
        else:
            src = pos - lag
            prev = rows - lag
            inside = valid & (src >= 0)
            intra = inside & (prev >= qstart)
            cached = inside & (prev < qstart)
            tap = (
                tl.load(
                    x_ptr
                    + tl.maximum(prev, 0).to(tl.int64)[:, None] * stride_x
                    + cols[None, :],
                    mask=intra[:, None] & col_mask[None, :],
                    other=0.0,
                )
                .to(dtype)
                .to(tl.float32)
            )
            safe = tl.maximum(src, 0)
            logical = tl.minimum(safe // cache_block, max_blocks - 1)
            block = tl.load(table_ptr + req * stride_bt + logical, mask=cached, other=0)
            tap += tl.load(
                cache_ptr
                + block.to(tl.int64)[:, None] * stride_cb
                + (safe % cache_block)[:, None] * stride_cn
                + cache_col[None, :],
                mask=cached[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        if DYNAMIC:
            weight = tl.load(
                w_ptr + lag * stride_w0 + cols * stride_w1, mask=col_mask, other=0.0
            ).to(tl.float32)
            weight = weight[None, :] + tl.load(
                dyn_ptr
                + rows.to(tl.int64)[:, None] * stride_dyn
                + (head * W + lag)[None, :],
                mask=row_col,
                other=0.0,
            ).to(tl.float32)
        else:
            weight = tl.load(
                w_ptr + cols * stride_w0 + k * stride_w1, mask=col_mask, other=0.0
            ).to(tl.float32)[None, :]
        if k == 0:
            acc = tap * weight
        else:
            acc = acc + tap * weight
    y = tl.div_rn(acc + cur.to(tl.float32), divisor)
    safe_slot = tl.maximum(slot, 0)
    tl.store(
        cache_ptr
        + (safe_slot // cache_block)[:, None] * stride_cb
        + (safe_slot % cache_block)[:, None] * stride_cn
        + cache_col[None, :],
        cur,
        mask=tl.where(store, valid, False)[:, None] & col_mask[None, :],
    )
    return y


@triton.jit
def _short_conv_kernel(
    x_ptr,
    w_ptr,
    dyn_ptr,
    cache_ptr,
    out_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    T,
    num_tokens,
    stride_x,
    stride_out,
    stride_w0,
    stride_w1,
    stride_dyn,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    offset,
    stride_bt,
    max_blocks,
    divisor,
    C: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    y = _conv_tile(
        x_ptr, stride_x, w_ptr, stride_w0, stride_w1, dyn_ptr, stride_dyn,
        cache_ptr, stride_cb, stride_ch, stride_cn, cache_block, offset,
        pos_ptr, seq_ptr, qstart_ptr, slot_ptr, table_ptr, stride_bt, max_blocks,
        num_tokens, rows, cols, col_mask, divisor, True, WS, H, W, DYNAMIC,
    )  # fmt: skip
    y = tl.where((rows < num_tokens)[:, None], y, 0.0)
    tl.store(
        out_ptr + rows.to(tl.int64)[:, None] * stride_out + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=(rows < T)[:, None] & col_mask[None, :],
    )


@triton.jit
def _key_value_tile(
    rows,
    head,
    x_ptr,
    stride_x,
    w_ptr,
    stride_w0,
    stride_w1,
    dyn_ptr,
    stride_dyn,
    offset,
    cache_ptr,
    kv_ptr,
    kv_slot_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    num_tokens,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    stride_bt,
    max_blocks,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    kv_block,
    divisor,
    clip,
    eps,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_V: tl.constexpr,
):
    d = tl.arange(0, BLOCK_D)
    col_mask = d < D
    y = _conv_tile(
        x_ptr, stride_x, w_ptr, stride_w0, stride_w1, dyn_ptr, stride_dyn,
        cache_ptr, stride_cb, stride_ch, stride_cn, cache_block, offset,
        pos_ptr, seq_ptr, qstart_ptr, slot_ptr, table_ptr, stride_bt, max_blocks,
        num_tokens, rows, head * D + d, col_mask, divisor, True, D, H, W, DYNAMIC,
    )  # fmt: skip
    # The conv output in the projection dtype, then clip (K) or norm (V).
    y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
    sum_sq = tl.sum(y * y, axis=1)
    if IS_V:
        factor = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(sum_sq, D) + eps))
    else:
        factor = tl.minimum(tl.div_rn(clip, tl.sqrt_rn(sum_sq) + 1e-6), 1.0)
    real = rows < num_tokens
    kv_slot = tl.load(kv_slot_ptr + rows, mask=real, other=-1)
    safe = tl.maximum(kv_slot, 0)
    tl.store(
        kv_ptr
        + (safe // kv_block)[:, None] * stride_kvb
        + head * stride_kvh
        + (safe % kv_block)[:, None] * stride_kvn
        + ((D if IS_V else 0) + d)[None, :],
        (y * factor[:, None]).to(kv_ptr.dtype.element_ty),
        mask=(real & (kv_slot >= 0))[:, None] & col_mask[None, :],
    )


@triton.jit
def _key_value_conv_kernel(
    k_ptr,
    v_ptr,
    wk_ptr,
    wv_ptr,
    dk_ptr,
    dv_ptr,
    cache_ptr,
    kv_ptr,
    kv_slot_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    num_tokens,
    stride_k,
    stride_v,
    stride_wk0,
    stride_wk1,
    stride_wv0,
    stride_wv1,
    stride_dk,
    stride_dv,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    offset_k,
    offset_v,
    stride_bt,
    max_blocks,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    kv_block,
    divisor,
    clip,
    eps,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Programs (tile, head, 0 | 1): the K | V conv of one KV head; constexprs
    # are passed by name because a tuple splat drops them.
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    head = tl.program_id(1)
    if tl.program_id(2) == 0:
        _key_value_tile(
            rows, head, k_ptr, stride_k, wk_ptr, stride_wk0, stride_wk1, dk_ptr, stride_dk,
            offset_k,
            cache_ptr, kv_ptr, kv_slot_ptr, pos_ptr, seq_ptr, qstart_ptr, slot_ptr,
            table_ptr, num_tokens, stride_cb, stride_ch, stride_cn, cache_block,
            stride_bt, max_blocks, stride_kvb, stride_kvh, stride_kvn, kv_block,
            divisor, clip, eps, D, H, W, DYNAMIC, BT, BLOCK_D, False,
        )  # fmt: skip
    else:
        _key_value_tile(
            rows, head, v_ptr, stride_v, wv_ptr, stride_wv0, stride_wv1, dv_ptr, stride_dv,
            offset_v,
            cache_ptr, kv_ptr, kv_slot_ptr, pos_ptr, seq_ptr, qstart_ptr, slot_ptr,
            table_ptr, num_tokens, stride_cb, stride_ch, stride_cn, cache_block,
            stride_bt, max_blocks, stride_kvb, stride_kvh, stride_kvn, kv_block,
            divisor, clip, eps, D, H, W, DYNAMIC, BT, BLOCK_D, True,
        )  # fmt: skip


def _tiles(num_tokens: int, channels: int) -> tuple[int, int]:
    """(rows, channels) per program, from the static batch size."""
    if num_tokens <= 8:
        block_t, block_c = 1, 1024
    elif num_tokens <= 256:
        block_t, block_c = 2, 512
    else:
        block_t, block_c = 4, 256
    return block_t, min(triton.next_power_of_2(channels), block_c)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """`t` with unit stride along its last dim (a view when it has one)."""
    return t if t.stride(-1) == 1 else t.contiguous()


def _metadata_args(metadata: Any) -> tuple:
    return (
        metadata.positions,
        metadata.seq_idx,
        metadata.query_start,
        metadata.slot_mapping,
        metadata.block_table,
    )


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    dynamic_taps: torch.Tensor | None,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
    num_tokens: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """sconv.static_conv (`dynamic_taps` None) or sconv.dynamic_conv of
    x [T, C] into `out` [T, C], in one launch. Rows from `num_tokens` on
    are zero."""
    total, channels = x.shape
    offset, width = stream
    heads = cache.shape[1]
    x = _rows(x)
    assert cache.stride(-1) == 1 and out.stride(-1) == 1 and out.shape == x.shape
    dynamic = dynamic_taps is not None
    if dynamic:
        taps = weight.shape[0]
        weight = _rows(weight)
        dynamic_taps = _rows(dynamic_taps.reshape(dynamic_taps.shape[0], -1))
        dyn_stride = dynamic_taps.stride(0)
    else:
        weight = weight.reshape(weight.shape[0], -1)
        taps = weight.shape[1]
        dynamic_taps, dyn_stride = weight, 0
    if total == 0:
        return out
    block_t, block_c = _tiles(total, channels)
    grid = (triton.cdiv(total, block_t), triton.cdiv(channels, block_c))
    block_table = metadata.block_table
    _short_conv_kernel[grid](
        x, weight, dynamic_taps, cache, out, *_metadata_args(metadata),
        total, num_tokens, x.stride(0), out.stride(0),
        weight.stride(0), weight.stride(1), dyn_stride,
        cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[2], offset,
        block_table.stride(0), block_table.shape[1],
        math.sqrt((2 if dynamic else 1) * taps + 1),
        C=channels, WS=width, H=heads, W=taps, DYNAMIC=dynamic,
        BT=block_t, BLOCK_C=block_c, num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out


def key_value_conv(
    k: torch.Tensor,
    v: torch.Tensor,
    k_conv: torch.Tensor,
    v_conv: torch.Tensor,
    k_taps: torch.Tensor | None,
    v_taps: torch.Tensor | None,
    conv_cache: torch.Tensor,
    conv_metadata: Any,
    k_stream: tuple[int, int],
    v_stream: tuple[int, int],
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    num_tokens: int,
    *,
    qk_clip: float,
    eps: float,
) -> None:
    """prologue.key_value_states and store_key_value of the first
    `num_tokens` rows of k, v [T, kv_heads * D], in one launch."""
    heads = conv_cache.shape[1]
    head_dim = k.shape[1] // heads
    assert k_stream[1] == v_stream[1] == head_dim
    assert kv_cache.shape[1] == heads and kv_cache.shape[3] == 2 * head_dim
    assert conv_cache.stride(-1) == 1 and kv_cache.stride(-1) == 1
    assert k.dtype == v.dtype
    k, v = _rows(k), _rows(v)
    dynamic = k_taps is not None
    if dynamic:
        assert v_taps is not None
        taps = k_conv.shape[0]
        k_conv, v_conv = _rows(k_conv), _rows(v_conv)
        k_taps = _rows(k_taps.reshape(k_taps.shape[0], -1))
        v_taps = _rows(v_taps.reshape(v_taps.shape[0], -1))
        dyn_strides = (k_taps.stride(0), v_taps.stride(0))
    else:
        k_conv = k_conv.reshape(k_conv.shape[0], -1)
        v_conv = v_conv.reshape(v_conv.shape[0], -1)
        taps = k_conv.shape[1]
        k_taps, v_taps, dyn_strides = k_conv, v_conv, (0, 0)
    total = k.shape[0]
    if total == 0:
        return
    block_t = 1 if total <= 8 else 4
    grid = (triton.cdiv(total, block_t), heads, 2)
    block_table = conv_metadata.block_table
    _key_value_conv_kernel[grid](
        k, v, k_conv, v_conv, k_taps, v_taps, conv_cache, kv_cache, kv_slot_mapping,
        *_metadata_args(conv_metadata),
        num_tokens, k.stride(0), v.stride(0), *k_conv.stride(), *v_conv.stride(),
        *dyn_strides,
        conv_cache.stride(0), conv_cache.stride(1), conv_cache.stride(2),
        conv_cache.shape[2], k_stream[0], v_stream[0],
        block_table.stride(0), block_table.shape[1],
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.shape[2],
        math.sqrt((2 if dynamic else 1) * taps + 1), qk_clip, eps,
        D=head_dim, H=heads, W=taps, DYNAMIC=dynamic,
        BT=block_t, BLOCK_D=triton.next_power_of_2(head_dim), num_warps=4,
        enable_fp_fusion=False,
    )  # fmt: skip
