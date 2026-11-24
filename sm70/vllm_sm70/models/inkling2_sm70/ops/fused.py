# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 decode kernels around the vendored norm kernels (norm.py,
attn_res.py, loop.py): the message-gate projections and the residual add.

    gate_down      low = down(rms([dest, message])) of each stream: pair norm,
                   model-dtype cast and down GEMV in one kernel; [S, T, R] fp32
                   rounded once to the down weight's dtype
    gate_up        [gain, forget] = low @ [gain, forget]^T in fp32, the input
                   of loop.blend_rms_norm; [S, T, 2D]
    residual_add   ((base + update) + addend) in fp32, rounded once to `out`

A gate kernel serves up to three streams, each with its own weights. Programs
run per (stream, token, block of outputs), so the weights are read once per
token: these kernels are for decode batches (`MAX_TOKENS`). Products and sums
run in fp32 (GEMVs as elementwise products and row sums).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ._compat import tl, triton
from .norm import rsqrt

__all__ = ["MAX_TOKENS", "gate_down", "gate_up", "residual_add"]

# Tokens up to which the gate kernels serve a batch.
MAX_TOKENS = 16
MAX_STREAMS = 3
BLOCK_R = 4
BLOCK_K = 512
BLOCK_D = 32
BLOCK_ADD = 1024


@triton.jit
def _pick(s, p0, p1, p2):
    p = p0
    if s == 1:
        p = p1
    if s == 2:
        p = p2
    return p


@triton.jit
def _row_rstd(Row, eps, D: tl.constexpr, BLOCK_K: tl.constexpr):
    cols = tl.arange(0, BLOCK_K)
    sum_sq = tl.zeros((BLOCK_K,), tl.float32)
    for k0 in tl.range(0, D, BLOCK_K):
        x = tl.load(Row + k0 + cols, k0 + cols < D, other=0)
        sum_sq += x * x
    return rsqrt(tl.div_rn(tl.sum(sum_sq), D) + eps)


@triton.jit
def _half_dot(acc, Row, rstd, W, r_mask, D: tl.constexpr, BLOCK_K: tl.constexpr):
    # acc[r, :] += W[r, k] * cast(rstd * Row[k]) over k in [0, D), where W
    # points at column 0 of this half and rows are 2D apart.
    cols = tl.arange(0, BLOCK_K)
    for k0 in tl.range(0, D, BLOCK_K):
        k = k0 + cols
        k_mask = k < D
        z = tl.load(Row + k, k_mask, other=0) * rstd
        z = z.to(W.dtype.element_ty).to(tl.float32)
        mask = r_mask[:, None] & k_mask[None, :]
        w = tl.load(W + k[None, :], mask, other=0).to(tl.float32)
        acc += w * z[None, :]
    return acc


@triton.jit(do_not_specialize=["T"])
def _gate_down_kernel(
    Dest,
    Message,
    Down0,
    Down1,
    Down2,
    Low,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    s = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r < R
    row = s.to(tl.int64) * T + t
    dest = Dest + s.to(tl.int64) * stride_dest_s + t * D
    message = Message + row * D
    down = _pick(s, Down0, Down1, Down2) + r[:, None].to(tl.int64) * (2 * D)
    acc = tl.zeros((BLOCK_R, BLOCK_K), tl.float32)
    dest_rstd = _row_rstd(dest, eps, D, BLOCK_K)
    message_rstd = _row_rstd(message, eps, D, BLOCK_K)
    acc = _half_dot(acc, dest, dest_rstd, down, r_mask, D, BLOCK_K)
    acc = _half_dot(acc, message, message_rstd, down + D, r_mask, D, BLOCK_K)
    low = tl.sum(acc, axis=1).to(Down0.dtype.element_ty).to(tl.float32)
    tl.store(Low + row * R + r, low, r_mask)


@triton.jit(do_not_specialize=["T"])
def _gate_up_kernel(
    Low,
    Gain0,
    Gain1,
    Gain2,
    Forget0,
    Forget1,
    Forget2,
    Out,
    T,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    s = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    rows = tl.arange(0, R)
    row = s.to(tl.int64) * T + t
    low = tl.load(Low + row * R + rows)
    offs = d[:, None].to(tl.int64) * R + rows[None, :]
    mask = d_mask[:, None]
    gain = tl.load(_pick(s, Gain0, Gain1, Gain2) + offs, mask, other=0)
    forget = tl.load(_pick(s, Forget0, Forget1, Forget2) + offs, mask, other=0)
    out = Out + row * (2 * D) + d
    tl.store(out, tl.sum(gain.to(tl.float32) * low[None, :], axis=1), d_mask)
    tl.store(out + D, tl.sum(forget.to(tl.float32) * low[None, :], axis=1), d_mask)


@triton.jit
def _residual_add_kernel(
    Base,
    Update,
    Addend,
    Out,
    N,
    HAS_BASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(Update + offs, mask, other=0).to(tl.float32)
    if HAS_BASE:
        x = tl.load(Base + offs, mask, other=0).to(tl.float32) + x
    x = x + tl.load(Addend + offs, mask, other=0).to(tl.float32)
    tl.store(Out + offs, x.to(Out.dtype.element_ty), mask)


def _three(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    assert 1 <= len(tensors) <= MAX_STREAMS
    return [*tensors, *[tensors[0]] * (MAX_STREAMS - len(tensors))]


def gate_down(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: Sequence[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """fp32 [S, T, R]: `down[s]` [R, 2D] applied to `[rms(dest[s]),
    rms(message[s])]` cast to its dtype, rounded to its dtype. fp32
    `dest` [S, T, D] (may broadcast over streams) and `message` [S, T, D];
    S = len(down) <= 3."""
    num_streams, num_tokens, d = message.shape
    rank = down[0].shape[0]
    assert len(down) == num_streams and message.is_contiguous()
    assert dest.shape == message.shape and dest.stride()[1:] == (d, 1)
    assert dest.dtype == message.dtype == torch.float32
    assert all(w.shape == (rank, 2 * d) and w.is_contiguous() for w in down)
    assert len({w.dtype for w in down}) == 1
    low = message.new_empty((num_streams, num_tokens, rank))
    if num_tokens:
        _gate_down_kernel[(num_streams, num_tokens, triton.cdiv(rank, BLOCK_R))](
            dest, message, *_three(down), low, dest.stride(0), num_tokens, eps,
            D=d, R=rank, BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_K, num_warps=4,
        )  # fmt: skip
    return low


def gate_up(
    low: torch.Tensor,
    gain: Sequence[torch.Tensor],
    forget: Sequence[torch.Tensor],
) -> torch.Tensor:
    """fp32 [S, T, 2D]: `[low @ gain[s]^T, low @ forget[s]^T]` of fp32 `low`
    [S, T, R] with `gain[s]`, `forget[s]` [D, R]; R a power of 2."""
    num_streams, num_tokens, rank = low.shape
    d = gain[0].shape[0]
    assert low.is_contiguous() and low.dtype == torch.float32
    assert rank == triton.next_power_of_2(rank)
    assert len(gain) == len(forget) == num_streams
    assert all(w.shape == (d, rank) and w.is_contiguous() for w in (*gain, *forget))
    assert len({w.dtype for w in (*gain, *forget)}) == 1
    out = low.new_empty((num_streams, num_tokens, 2 * d))
    if num_tokens:
        _gate_up_kernel[(num_streams, num_tokens, triton.cdiv(d, BLOCK_D))](
            low, *_three(gain), *_three(forget), out, num_tokens,
            D=d, R=rank, BLOCK_D=BLOCK_D, num_warps=4,
        )  # fmt: skip
    return out


def residual_add(
    base: torch.Tensor | None,
    update: torch.Tensor,
    addend: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """`dtype` ((base + update) + addend), summed in fp32; `update` alone
    without `base`. Contiguous tensors of one shape."""
    tensors = [t for t in (base, update, addend) if t is not None]
    assert all(t.shape == update.shape and t.is_contiguous() for t in tensors)
    out = torch.empty(update.shape, dtype=dtype, device=update.device)
    n = update.numel()
    if n:
        _residual_add_kernel[(triton.cdiv(n, BLOCK_ADD),)](
            update if base is None else base, update, addend, out, n,
            HAS_BASE=base is not None, BLOCK=BLOCK_ADD, num_warps=4,
        )  # fmt: skip
    return out
