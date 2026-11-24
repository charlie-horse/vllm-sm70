# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 decode kernels that merge adjacent launches of a layer visit
(sm_70: no tl.dot, no PDL, libdevice tanh). The route is visit.py's.

    conv_mix        a short conv (sconv.py), the residual stream it feeds
                    and the AttnRes mix of the next sublayer input, with or
                    without its RMS norm; one program per (query, token)
                    convolves the token's whole row, the first one stores
    conv_residual   a short conv and the residual add after it
    prologue        small_ops.query_rel and paged_conv.key_value_conv:
                    programs (token, query head), then (token, K/V head)
    gate_blend      fused.gate_down, fused.gate_up and loop.blend_rms_norm
                    of the message gates, in one launch (tickets.py)

Math and rounding as the kernels they merge, which stay the reference: the
conv output rounds to the dtype of its input before it is summed; the
residual sums ((base + update) + conv) in fp32 and rounds once; the mixes
are those of attn_res.py's row kernel. gate_blend sums each row's squares
per block of outputs, then over the blocks in order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from ._compat import exp, tanh, tl, triton
from .fused import _pick, _row_rstd
from .norm import rsqrt
from .paged_conv import _conv_tile, _key_value_tile, _metadata_args, _rows
from .small_ops import _query_rel_head
from .tickets import GATE_ROWS, GATE_TICKET, ticket_counter

__all__ = [
    "GATE_MAX_ROWS",
    "MIX_MAX_PROGRAMS",
    "conv_mix",
    "conv_residual",
    "gate_blend",
    "prologue",
]

# (query, token) programs up to which conv_mix runs (attn_res.py's row
# kernel serves the same bound).
MIX_MAX_PROGRAMS = 256
# (stream, token) rows up to which gate_blend runs; two counters each.
GATE_MAX_ROWS = 48
GATE_BLOCK_R = 4
GATE_BLOCK_K = 512
GATE_BLOCK_D = 32
GATE_BLOCK_N = 1024


# --- conv_mix and conv_residual ---------------------------------------------


@triton.jit
def _conv_row(
    x_ptr,
    w_ptr,
    cache_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    t,
    cols,
    mask,
    store,
    num_tokens,
    stride_x,
    stride_w0,
    stride_w1,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    offset,
    stride_bt,
    max_blocks,
    divisor,
    WS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    """fp32 static conv of `cols` of row `t`, rounded to the dtype of x and
    zero from `num_tokens` on; stores the conv inputs when `store`."""
    rows = t + tl.arange(0, 1)
    y = _conv_tile(
        x_ptr, stride_x, w_ptr, stride_w0, stride_w1, w_ptr, 0,
        cache_ptr, stride_cb, stride_ch, stride_cn, cache_block, offset,
        pos_ptr, seq_ptr, qstart_ptr, slot_ptr, table_ptr, stride_bt, max_blocks,
        num_tokens, rows, cols, mask, divisor, store, WS, H, W, False,
    )  # fmt: skip
    y = tl.where((rows < num_tokens)[:, None], y, 0.0)
    return tl.sum(y, axis=0).to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _state_part(
    y,
    base_ptr,
    update_ptr,
    conv_out_ptr,
    state_ptr,
    off,
    mask,
    store,
    HAS_BASE: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    """The fp32 current state of a mix from the rounded conv output `y`.

    RESIDUAL: the layer output ((base + update) + y), rounded and stored
    in `state_ptr`. Otherwise `y` is stored in `conv_out_ptr` and the state
    is base + y (y alone without a base)."""
    if RESIDUAL:
        x = tl.load(update_ptr + off, mask, other=0).to(tl.float32)
        if HAS_BASE:
            x = tl.load(base_ptr + off, mask, other=0).to(tl.float32) + x
        x = (x + y).to(state_ptr.dtype.element_ty)
        tl.store(state_ptr + off, x, tl.where(store, mask, False))
        return x.to(tl.float32)
    y_out = y.to(conv_out_ptr.dtype.element_ty)
    tl.store(conv_out_ptr + off, y_out, tl.where(store, mask, False))
    if HAS_BASE:
        return tl.load(base_ptr + off, mask, other=0).to(tl.float32) + y
    return y


@triton.jit(do_not_specialize=["T", "num_tokens"])
def _conv_mix_kernel(
    x_ptr,
    w_ptr,
    cache_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    base_ptr,
    update_ptr,
    conv_out_ptr,
    state_ptr,
    stack_ptr,
    query_ptr,
    norm_w_ptr,
    out_ptr,
    T,
    num_tokens,
    stride_x,
    stride_w0,
    stride_w1,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    offset,
    stride_bt,
    max_blocks,
    divisor,
    stride_s,
    stride_t,
    eps,
    norm_eps,
    q_scale,
    D: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_BASE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    q_id = tl.program_id(0)
    t = tl.program_id(1)
    store = q_id == 0
    t64 = t.to(tl.int64)
    row_off = t64 * D
    stack_off = t64 * stride_t
    q_off = q_id * D
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D
    conv = (
        x_ptr, w_ptr, cache_ptr, pos_ptr, seq_ptr, qstart_ptr, slot_ptr, table_ptr, t,
    )  # fmt: skip
    conv_args = (
        num_tokens, stride_x, stride_w0, stride_w1, stride_cb, stride_ch, stride_cn,
        cache_block, offset, stride_bt, max_blocks, divisor,
    )  # fmt: skip
    ya = _conv_row(*conv, da, mask_a, store, *conv_args, WS=WS, H=H, W=W)
    yb = _conv_row(*conv, db, mask_b, store, *conv_args, WS=WS, H=H, W=W)
    parts = (base_ptr, update_ptr, conv_out_ptr, state_ptr)
    last_a = _state_part(
        ya, *parts, row_off + da, mask_a, store, HAS_BASE=HAS_BASE, RESIDUAL=RESIDUAL
    )
    last_b = _state_part(
        yb, *parts, row_off + db, mask_b, store, HAS_BASE=HAS_BASE, RESIDUAL=RESIDUAL
    )

    # attn_res.py _attn_res_row_kernel with the current state in registers;
    # the stacked states are normed already.
    n_idx = tl.arange(0, BLOCK_N)
    qa = tl.load(query_ptr + q_off + da, mask_a, other=0)
    qb = tl.load(query_ptr + q_off + db, mask_b, other=0)
    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        if n < N_STACK:
            src = stack_ptr + n * stride_s + stack_off
            xa = tl.load(src + da, mask_a, other=0).to(tl.float32)
            xb = tl.load(src + db, mask_b, other=0).to(tl.float32)
            rstd = 1.0
        else:
            xa, xb = last_a, last_b
            rstd = rsqrt(tl.fma(tl.sum(xa * xa) + tl.sum(xb * xb), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        dot = tl.sum(qa * xa) + tl.sum(qb * xb)
        logits = tl.where(n_idx == n, dot * rstd * q_scale, logits)

    probs = exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    probs = probs * inv_rms

    acc_a = tl.zeros((BLOCK_A,), tl.float32)
    acc_b = tl.zeros((BLOCK_B,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        p = tl.sum(tl.where(n_idx == n, probs, 0.0))
        if n < N_STACK:
            src = stack_ptr + n * stride_s + stack_off
            xa = tl.load(src + da, mask_a, other=0).to(tl.float32)
            xb = tl.load(src + db, mask_b, other=0).to(tl.float32)
        else:
            xa, xb = last_a, last_b
        acc_a = tl.fma(p, xa, acc_a)
        acc_b = tl.fma(p, xb, acc_b)
    acc_a = acc_a * inv_sum
    acc_b = acc_b * inv_sum
    out = out_ptr + q_id.to(tl.int64) * T * D + row_off
    if APPLY_NORM:
        sum_sq = tl.sum(acc_a * acc_a) + tl.sum(acc_b * acc_b)
        out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
        wa = tl.load(norm_w_ptr + q_off + da, mask_a, other=0).to(tl.float32)
        wb = tl.load(norm_w_ptr + q_off + db, mask_b, other=0).to(tl.float32)
        acc_a = acc_a * out_rstd * wa
        acc_b = acc_b * out_rstd * wb
    tl.store(out + da, acc_a.to(out_ptr.dtype.element_ty), mask_a)
    tl.store(out + db, acc_b.to(out_ptr.dtype.element_ty), mask_b)


@triton.jit(do_not_specialize=["num_tokens"])
def _conv_residual_kernel(
    x_ptr,
    w_ptr,
    cache_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    base_ptr,
    update_ptr,
    state_ptr,
    num_tokens,
    stride_x,
    stride_w0,
    stride_w1,
    stride_cb,
    stride_ch,
    stride_cn,
    cache_block,
    offset,
    stride_bt,
    max_blocks,
    divisor,
    D: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HAS_BASE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < D
    y = _conv_row(
        x_ptr, w_ptr, cache_ptr, pos_ptr, seq_ptr, qstart_ptr, slot_ptr, table_ptr,
        t, cols, mask, True,
        num_tokens, stride_x, stride_w0, stride_w1, stride_cb, stride_ch, stride_cn,
        cache_block, offset, stride_bt, max_blocks, divisor, WS=WS, H=H, W=W,
    )  # fmt: skip
    _state_part(
        y, base_ptr, update_ptr, state_ptr, state_ptr, t.to(tl.int64) * D + cols,
        mask, True, HAS_BASE=HAS_BASE, RESIDUAL=True,
    )  # fmt: skip


def _conv_args(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
    num_tokens: int,
) -> tuple[tuple, dict]:
    """Runtime arguments and constexprs of a static conv of x [T, C] over
    `stream` of `cache`, from the static weight [C, K] or [C, 1, K]."""
    channels = x.shape[1]
    offset, width = stream
    weight = weight.reshape(weight.shape[0], -1)
    assert x.stride(-1) == 1 and cache.stride(-1) == 1
    assert weight.shape[0] == channels == cache.shape[1] * width
    block_table = metadata.block_table
    args = (
        num_tokens, x.stride(0), weight.stride(0), weight.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[2], offset,
        block_table.stride(0), block_table.shape[1],
        math.sqrt(weight.shape[1] + 1),
    )  # fmt: skip
    return (
        (x, weight, cache, *_metadata_args(metadata)),
        args,
        {"WS": width, "H": cache.shape[1], "W": weight.shape[1]},
    )


def conv_mix(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
    num_tokens: int,
    base: torch.Tensor | None,
    update: torch.Tensor | None,
    conv_out: torch.Tensor | None,
    state_out: torch.Tensor | None,
    stack: torch.Tensor,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """The static conv y of x [T, D] over `stream` (sconv.short_conv on a
    decode-only batch whose x is zero in padding rows), then the AttnRes
    mixes of the normed states `stack` [N, T, D] fp32 (any strides over N
    and T) and the current state, as norms.py's mixes, into `out` [Q, T, D]:
    normed with `norm_weight` [Q, D] in the dtype of `out`, or fp32 without.

    With `update` [T, D] the current state is the layer output
    ((base + update) + y), stored in `state_out` [T, D]; otherwise y is
    stored in `conv_out` [T, D] and the state is base + y. `base` may be
    None. Rows from `num_tokens` on convolve to zero."""
    total, d = x.shape
    num_q = queries.shape[0]
    residual = update is not None
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    assert stack.shape[1:] == x.shape and stack.stride(2) == 1
    assert out.shape == (num_q, total, d) and out.is_contiguous()
    assert num_q * total <= MIX_MAX_PROGRAMS
    parts = [t for t in (base, update, conv_out, state_out) if t is not None]
    assert all(t.shape == x.shape and t.is_contiguous() for t in parts)
    assert (state_out if residual else conv_out) is not None
    apply_norm = norm_weight is not None
    if apply_norm:
        assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    else:
        assert out.dtype == torch.float32
    if total == 0:
        return out
    tensors, args, meta = _conv_args(x, weight, cache, metadata, stream, num_tokens)
    first = parts[0]
    block_a = triton.next_power_of_2(d + 1) // 2
    _conv_mix_kernel[(num_q, total)](
        *tensors,
        base if base is not None else first,
        update if update is not None else first,
        conv_out if conv_out is not None else first,
        state_out if state_out is not None else first,
        stack, queries, norm_weight if apply_norm else queries, out,
        total, *args, stack.stride(0), stack.stride(1), 1e-6, norm_eps, d**-0.5,
        D=d, **meta, N_STACK=stack.shape[0], HAS_BASE=base is not None,
        RESIDUAL=residual, APPLY_NORM=apply_norm,
        BLOCK_N=triton.next_power_of_2(stack.shape[0] + 1),
        BLOCK_A=block_a, BLOCK_B=max(triton.next_power_of_2(d - block_a), 16),
        num_warps=8, enable_fp_fusion=False,
    )  # fmt: skip
    return out


def conv_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
    num_tokens: int,
    base: torch.Tensor | None,
    update: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """`out` [T, D] = ((base + update) + y) of the static conv y of x [T, D]
    (``conv_mix``), summed in fp32 and rounded once."""
    total, d = x.shape
    parts = [t for t in (base, update, out) if t is not None]
    assert all(t.shape == x.shape and t.is_contiguous() for t in parts)
    if total == 0:
        return out
    tensors, args, meta = _conv_args(x, weight, cache, metadata, stream, num_tokens)
    block = min(1024, triton.next_power_of_2(d))
    _conv_residual_kernel[(total, triton.cdiv(d, block))](
        *tensors, base if base is not None else update, update, out, *args,
        D=d, **meta, HAS_BASE=base is not None, BLOCK_C=block,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out


# --- prologue -----------------------------------------------------------------


@triton.jit(do_not_specialize=["num_tokens"])
def _prologue_kernel(
    q_ptr,
    r_ptr,
    q_scale_ptr,
    r_scale_ptr,
    proj_ptr,
    attn_slot_ptr,
    attn_pos_ptr,
    q_out_ptr,
    rel_out_ptr,
    k_ptr,
    v_ptr,
    wk_ptr,
    wv_ptr,
    dk_ptr,
    dv_ptr,
    cache_ptr,
    kv_ptr,
    pos_ptr,
    seq_ptr,
    qstart_ptr,
    slot_ptr,
    table_ptr,
    num_tokens,
    stride_q,
    stride_r,
    stride_qo0,
    stride_qo1,
    stride_ro0,
    stride_ro1,
    stride_p,
    extent,
    q_clip,
    r_clip,
    alpha,
    n_floor,
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
    eps,
    QH: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    LOG_SCALE: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DYNAMIC: tl.constexpr,
):
    # Programs (token, j): query head j < QH, then the K conv of KV head
    # j - QH and the V conv of KV head j - QH - H; constexprs by name.
    row = tl.program_id(0)
    j = tl.program_id(1)
    rows = row + tl.arange(0, 1)
    if j < QH:
        _query_rel_head(
            row, j,
            q_ptr, r_ptr, q_scale_ptr, r_scale_ptr, proj_ptr, attn_slot_ptr,
            attn_pos_ptr, q_out_ptr, rel_out_ptr, num_tokens, stride_q, stride_r,
            stride_qo0, stride_qo1, stride_ro0, stride_ro1, stride_p, extent,
            q_clip, r_clip, alpha, n_floor,
            D=D, DR=DR, BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R, BLOCK_E=BLOCK_E,
            LOG_SCALE=LOG_SCALE,
        )  # fmt: skip
    elif j < QH + H:
        _key_value_tile(
            rows, j - QH, k_ptr, stride_k, wk_ptr, stride_wk0, stride_wk1, dk_ptr,
            stride_dk, offset_k,
            cache_ptr, kv_ptr, attn_slot_ptr, pos_ptr, seq_ptr, qstart_ptr, slot_ptr,
            table_ptr, num_tokens, stride_cb, stride_ch, stride_cn, cache_block,
            stride_bt, max_blocks, stride_kvb, stride_kvh, stride_kvn, kv_block,
            divisor, q_clip, eps, D, H, W, DYNAMIC, 1, BLOCK_D, False,
        )  # fmt: skip
    else:
        _key_value_tile(
            rows, j - QH - H, v_ptr, stride_v, wv_ptr, stride_wv0, stride_wv1, dv_ptr,
            stride_dv, offset_v,
            cache_ptr, kv_ptr, attn_slot_ptr, pos_ptr, seq_ptr, qstart_ptr, slot_ptr,
            table_ptr, num_tokens, stride_cb, stride_ch, stride_cn, cache_block,
            stride_bt, max_blocks, stride_kvb, stride_kvh, stride_kvn, kv_block,
            divisor, q_clip, eps, D, H, W, DYNAMIC, 1, BLOCK_D, True,
        )  # fmt: skip


def prologue(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    rel_proj: torch.Tensor,
    attn_metadata: Any,
    q_out: torch.Tensor,
    rel_out: torch.Tensor,
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
    num_tokens: int,
    *,
    qk_clip: float,
    rel_clip: float,
    eps: float,
    log_scaling: tuple[float, float] | None,
) -> None:
    """small_ops.query_rel and paged_conv.key_value_conv of one prologue in
    one launch; arguments as theirs. The query heads' head dim is the KV
    heads' one."""
    total = q_out.shape[0]
    heads, head_dim = q_scale.shape
    d_rel = r_scale.shape[1]
    extent = rel_proj.shape[1]
    kv_heads = conv_cache.shape[1]
    assert k.shape[1] == kv_heads * head_dim and k.dtype == v.dtype
    assert k_stream[1] == v_stream[1] == head_dim
    assert kv_cache.shape[1] == kv_heads and kv_cache.shape[3] == 2 * head_dim
    assert conv_cache.stride(-1) == 1 and kv_cache.stride(-1) == 1
    assert q.stride(-1) == r.stride(-1) == rel_proj.stride(-1) == 1
    assert q_out.stride(-1) == rel_out.stride(-1) == 1 and q_out.dtype == rel_out.dtype
    assert q_scale.is_contiguous() and r_scale.is_contiguous()
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
    if total == 0:
        return
    alpha, n_floor = log_scaling if log_scaling is not None else (0.0, 1.0)
    block_table = conv_metadata.block_table
    _prologue_kernel[(total, heads + 2 * kv_heads)](
        q, r, q_scale, r_scale, rel_proj, attn_metadata.slot_mapping,
        attn_metadata.positions, q_out, rel_out,
        k, v, k_conv, v_conv, k_taps, v_taps, conv_cache, kv_cache,
        *_metadata_args(conv_metadata),
        num_tokens, q.stride(0), r.stride(0), q_out.stride(0), q_out.stride(1),
        rel_out.stride(0), rel_out.stride(1), rel_proj.stride(0), extent,
        float(qk_clip), float(rel_clip), float(alpha), float(n_floor),
        k.stride(0), v.stride(0), *k_conv.stride(), *v_conv.stride(), *dyn_strides,
        conv_cache.stride(0), conv_cache.stride(1), conv_cache.stride(2),
        conv_cache.shape[2], k_stream[0], v_stream[0],
        block_table.stride(0), block_table.shape[1],
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.shape[2],
        math.sqrt((2 if dynamic else 1) * taps + 1), eps,
        QH=heads, D=head_dim, DR=d_rel,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=max(2, triton.next_power_of_2(d_rel)),
        BLOCK_E=min(256, max(16, triton.next_power_of_2(extent))),
        LOG_SCALE=log_scaling is not None, H=kv_heads, W=taps, DYNAMIC=dynamic,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip


# --- gate_blend -----------------------------------------------------------------


@triton.jit
def _down_dot(acc, Row, rstd, W, r_mask, D: tl.constexpr, BLOCK_K: tl.constexpr):
    # fused._half_dot with its products accumulated by FMA, as that kernel
    # compiles them.
    cols = tl.arange(0, BLOCK_K)
    for k0 in tl.range(0, D, BLOCK_K):
        k = k0 + cols
        k_mask = k < D
        z = tl.load(Row + k, k_mask, other=0) * rstd
        z = z.to(W.dtype.element_ty).to(tl.float32)
        mask = r_mask[:, None] & k_mask[None, :]
        w = tl.load(W + k[None, :], mask, other=0).to(tl.float32)
        acc = tl.fma(w, z[None, :], acc)
    return acc


@triton.jit
def _gate_low(
    ticket,
    Dest,
    Message,
    Down0,
    Down1,
    Down2,
    Low,
    rows_ptr,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKS_R: tl.constexpr,
):
    """fused._gate_down_kernel of the program `ticket`, then one count on
    its row."""
    row = ticket // BLOCKS_R
    s = row // T
    t = (row % T).to(tl.int64)
    r = (ticket % BLOCKS_R) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r < R
    dest = Dest + s.to(tl.int64) * stride_dest_s + t * D
    message = Message + row.to(tl.int64) * D
    down = _pick(s, Down0, Down1, Down2) + r[:, None].to(tl.int64) * (2 * D)
    acc = tl.zeros((BLOCK_R, BLOCK_K), tl.float32)
    dest_rstd = _row_rstd(dest, eps, D, BLOCK_K)
    message_rstd = _row_rstd(message, eps, D, BLOCK_K)
    acc = _down_dot(acc, dest, dest_rstd, down, r_mask, D, BLOCK_K)
    acc = _down_dot(acc, message, message_rstd, down + D, r_mask, D, BLOCK_K)
    low = tl.sum(acc, axis=1).to(Down0.dtype.element_ty).to(tl.float32)
    tl.store(Low + row.to(tl.int64) * R + r, low, r_mask)
    tl.debug_barrier()
    tl.atomic_add(rows_ptr + 2 * row, 1, sem="acq_rel", scope="gpu")


@triton.jit
def _gate_rows(
    block,
    Dest,
    Message,
    Gain0,
    Gain1,
    Gain2,
    Forget0,
    Forget1,
    Forget2,
    NormW,
    Low,
    Blend,
    Partial,
    Out,
    rows_ptr,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCKS_R: tl.constexpr,
    BLOCKS_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Wait for the `low` of the row, then the blend of outputs block
    `block` (fused._gate_up_kernel, loop._blend_rms_norm_kernel); the last
    block of a row norms the row and puts its counters back to 0."""
    row = block // BLOCKS_D
    s = row // T
    t = (row % T).to(tl.int64)
    row64 = row.to(tl.int64)
    done = tl.atomic_add(rows_ptr + 2 * row, 0, sem="acq_rel", scope="gpu")
    while done < BLOCKS_R:
        done = tl.atomic_add(rows_ptr + 2 * row, 0, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    d = (block % BLOCKS_D) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    k = tl.arange(0, R)
    low = tl.load(Low + row64 * R + k, cache_modifier=".cg")
    offs = d[:, None].to(tl.int64) * R + k[None, :]
    gain = tl.load(_pick(s, Gain0, Gain1, Gain2) + offs, d_mask[:, None], other=0)
    forget = tl.load(
        _pick(s, Forget0, Forget1, Forget2) + offs, d_mask[:, None], other=0
    )
    gain = tanh(tl.sum(gain.to(tl.float32) * low[None, :], axis=1))
    forget = tanh(tl.sum(forget.to(tl.float32) * low[None, :], axis=1))
    x_dest = tl.load(Dest + s.to(tl.int64) * stride_dest_s + t * D + d, d_mask, other=0)
    x_msg = tl.load(Message + row64 * D + d, d_mask, other=0)
    x = (1.0 - forget) * x_dest + gain * x_msg
    x = tl.where(d_mask, x, 0.0)
    tl.store(Blend + row64 * D + d, x, d_mask)
    tl.store(Partial + row64 * BLOCKS_D + block % BLOCKS_D, tl.sum(x * x))
    tl.debug_barrier()
    old = tl.atomic_add(rows_ptr + 2 * row + 1, 1, sem="acq_rel", scope="gpu")
    if old == BLOCKS_D - 1:
        tl.debug_barrier()
        p = tl.arange(0, BLOCK_P)
        parts = tl.load(
            Partial + row64 * BLOCKS_D + p,
            p < BLOCKS_D,
            other=0.0,
            cache_modifier=".cg",
        )
        rstd = rsqrt(tl.div_rn(tl.sum(parts), D) + norm_eps)
        cols = tl.arange(0, BLOCK_N)
        for c0 in tl.range(0, D, BLOCK_N):
            c = c0 + cols
            c_mask = c < D
            y = tl.load(Blend + row64 * D + c, c_mask, other=0, cache_modifier=".cg")
            y = y * rstd
            w = tl.load(NormW + s * stride_w_s + c, c_mask, other=0)
            y = y * w.to(tl.float32)
            tl.store(Out + row64 * D + c, y.to(Out.dtype.element_ty), c_mask)
        tl.atomic_xchg(rows_ptr + 2 * row, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(rows_ptr + 2 * row + 1, 0, sem="relaxed", scope="gpu")


@triton.jit(do_not_specialize=["T"])
def _gate_blend_kernel(
    Dest,
    Message,
    Down0,
    Down1,
    Down2,
    Gain0,
    Gain1,
    Gain2,
    Forget0,
    Forget1,
    Forget2,
    NormW,
    Low,
    Blend,
    Partial,
    Out,
    ticket_ptr,
    rows_ptr,
    stride_dest_s,
    stride_w_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCKS_R: tl.constexpr,
    BLOCKS_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Tickets in launch order: the first S * T * BLOCKS_R programs compute
    # `low`, the others blend. A waiting program holds a later ticket than
    # every program it waits for, so they have all started.
    num_down = tl.num_programs(0) // (BLOCKS_R + BLOCKS_D) * BLOCKS_R
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == tl.num_programs(0) - 1:
        tl.atomic_xchg(ticket_ptr, 0, sem="relaxed", scope="gpu")
    if ticket < num_down:
        _gate_low(
            ticket, Dest, Message, Down0, Down1, Down2, Low, rows_ptr,
            stride_dest_s, T, eps,
            D=D, R=R, BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_K, BLOCKS_R=BLOCKS_R,
        )  # fmt: skip
    else:
        _gate_rows(
            ticket - num_down, Dest, Message, Gain0, Gain1, Gain2, Forget0,
            Forget1, Forget2, NormW, Low, Blend, Partial, Out, rows_ptr,
            stride_dest_s, stride_w_s, T, norm_eps,
            D=D, R=R, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N, BLOCKS_R=BLOCKS_R,
            BLOCKS_D=BLOCKS_D, BLOCK_P=BLOCK_P,
        )  # fmt: skip


def _three(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    assert 1 <= len(tensors) <= 3
    return [*tensors, *[tensors[0]] * (3 - len(tensors))]


def gate_blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: Sequence[torch.Tensor],
    gain: Sequence[torch.Tensor],
    forget: Sequence[torch.Tensor],
    eps: float,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``loop.blend_rms_norm`` of the gains of ``fused.gate_up`` of
    ``fused.gate_down``, in one launch: `out_dtype` [S, T, D]. fp32 `dest`
    [S, T, D] (may broadcast over streams) and `message` [S, T, D]; per
    stream `down[s]` [R, 2D], `gain[s]`, `forget[s]` [D, R] (R a power of
    2); `norm_weight` [S, D] (may broadcast). S * T <= GATE_MAX_ROWS."""
    num_streams, num_tokens, d = message.shape
    rank = down[0].shape[0]
    rows = num_streams * num_tokens
    assert len(down) == len(gain) == len(forget) == num_streams
    assert 0 < rows <= GATE_MAX_ROWS
    assert message.is_contiguous() and dest.shape == message.shape
    assert dest.stride()[1:] == (d, 1)
    assert dest.dtype == message.dtype == torch.float32
    assert rank == triton.next_power_of_2(rank)
    assert all(w.shape == (rank, 2 * d) and w.is_contiguous() for w in down)
    assert all(w.shape == (d, rank) and w.is_contiguous() for w in (*gain, *forget))
    assert len({w.dtype for w in down}) == len({w.dtype for w in (*gain, *forget)}) == 1
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    blocks_r = triton.cdiv(rank, GATE_BLOCK_R)
    blocks_d = triton.cdiv(d, GATE_BLOCK_D)
    low = message.new_empty((rows, rank))
    blend = message.new_empty((rows, d))
    partial = message.new_empty((rows, blocks_d))
    out = message.new_empty((num_streams, num_tokens, d), dtype=out_dtype)
    counters = ticket_counter(message.device, GATE_TICKET)
    _gate_blend_kernel[(rows * (blocks_r + blocks_d),)](
        dest, message, *_three(down), *_three(gain), *_three(forget), norm_weight,
        low, blend, partial, out, counters,
        ticket_counter(message.device, GATE_ROWS),
        dest.stride(0), norm_weight.stride(0), num_tokens, eps, norm_eps,
        D=d, R=rank, BLOCK_R=GATE_BLOCK_R, BLOCK_K=GATE_BLOCK_K,
        BLOCK_D=GATE_BLOCK_D, BLOCK_N=min(GATE_BLOCK_N, triton.next_power_of_2(d)),
        BLOCKS_R=blocks_r, BLOCKS_D=blocks_d,
        BLOCK_P=triton.next_power_of_2(blocks_d),
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out
