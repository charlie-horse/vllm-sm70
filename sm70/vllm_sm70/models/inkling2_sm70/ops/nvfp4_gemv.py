# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 NVFP4 expert slots as a Triton GEMV over the TurboMind banks
(sm_70: no tl.dot, libdevice exp and tanh, fp32 FMA).

    nvfp4_slot_gemv   out[s] = rows[s] @ W[expert[s]] for one-row slots,
                      W13 optionally with the soft-capped SwiGLU in its
                      epilogue

The banks are moe_tm.py's, read in place: codes int32 [E, K, N / 8] and
fp16 scales [E, K / 16, N] as ``tm.nvfp4_sm70_prepare`` writes them with
k_ld = 32 * K and q_ld = N. Codes are words [N / 32, K / 8, 32]: word
(t, j, c) holds the eight e2m1 codes of column 32 t + c at K rows 8 j +
``NIBBLE_K`` (nibble p, low bits first). A code's fp16 bit pattern placed
as sign | e1 e0 m << 9 is its value times 2^-14; the 2^14 applies to the
fp32 sum.

One program covers one tile of 32 output columns of one slot over all of K
(with the SwiGLU: a gate tile and the matching up tile), so every code byte
of a slot's expert is read once. Products and sums run in fp32;
a W13 output rounds to fp16 before the SwiGLU, as the TurboMind stage
output does, and the SwiGLU's fp32 result rounds once (ops/activation.py).
Expert ids are clamped to 0..E-1, as moe_tm.py clamps padding slots.
"""

from __future__ import annotations

from typing import Final

import torch

from ._compat import exp, tanh, tl, triton

__all__ = ["CODES_PER_WORD", "NIBBLE_K", "config", "nvfp4_slot_gemv"]

CODES_PER_WORD: Final = 8
# K row (within a word's 8) of nibble p.
NIBBLE_K: Final = (0, 2, 4, 6, 1, 3, 5, 7)
TILE_N: Final = 32
# (BLOCK_K8, num_warps) per stage; BLOCK_K8 counts K rows of 8.
CONFIGS: Final = {
    "w13": (16, 4),
    "w2": (8, 4),
}


@triton.jit
def _decode(word, nibble: tl.constexpr):
    """fp32 e2m1 value times 2^-14 of nibble `nibble` of the int32 `word`."""
    code = word >> (4 * nibble)
    bits = ((code & 8) << 12) | ((code & 7) << 9)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _tile_dot(codes, scales, tile, k8, k_mask, k8_rows, n_total, xs):
    """fp32 [BLOCK_K8, 32] scaled partial sums of the 32 columns of `tile`
    over the K rows of 8 `k8`; `xs` are the activations of nibbles 0..7."""
    lane = tl.max_contiguous(tl.arange(0, 32), 32)
    base = tl.multiple_of(tile * (k8_rows * 32), 32)
    rows = tl.multiple_of(k8 * 32, 32)
    word = tl.load(
        codes + base + rows[:, None] + lane[None, :],
        mask=k_mask[:, None],
        other=0,
    )
    x0, x1, x2, x3, x4, x5, x6, x7 = xs
    part = _decode(word, 0) * x0[:, None]
    part += _decode(word, 1) * x1[:, None]
    part += _decode(word, 2) * x2[:, None]
    part += _decode(word, 3) * x3[:, None]
    part += _decode(word, 4) * x4[:, None]
    part += _decode(word, 5) * x5[:, None]
    part += _decode(word, 6) * x6[:, None]
    part += _decode(word, 7) * x7[:, None]
    groups = tl.multiple_of((k8 // 2) * n_total, 32)
    scale = tl.load(
        scales + groups[:, None] + tile * 32 + lane[None, :],
        mask=k_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    return part * scale


@triton.jit
def _nvfp4_slot_gemv_kernel(
    in_ptr,
    codes_ptr,
    scales_ptr,
    out_ptr,
    expert_ptr,
    stride_in,
    stride_ce,
    stride_se,
    stride_out,
    n_out,
    k_in,
    experts,
    gate_cap,
    up_cap,
    SWIGLU: tl.constexpr,
    BLOCK_K8: tl.constexpr,
):
    """Program (slot, tile): the 32 output columns of `tile` (module doc)."""
    slot = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    # With the SwiGLU the bank holds the gate then the up columns.
    n_total = n_out
    if SWIGLU:
        n_total = n_out * 2
    expert = tl.minimum(tl.maximum(tl.load(expert_ptr + slot), 0), experts - 1)
    # Banks are whole 32-column tiles: every expert and tile starts 128-byte
    # aligned.
    codes = codes_ptr + tl.multiple_of(expert.to(tl.int64) * stride_ce, 32)
    scales = scales_ptr + tl.multiple_of(expert.to(tl.int64) * stride_se, 32)
    x_row = in_ptr + slot * stride_in
    k8_rows = k_in // 8
    acc = tl.zeros([BLOCK_K8, 32], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_K8, 32], dtype=tl.float32)
    for j0 in range(0, k8_rows, BLOCK_K8):
        k8 = j0 + tl.arange(0, BLOCK_K8)
        k_mask = k8 < k8_rows
        xk = x_row + k8 * 8
        # Nibble p of a word holds K row NIBBLE_K[p] of its 8.
        xs = (
            tl.load(xk + 0, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 2, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 4, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 6, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 1, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 3, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 5, mask=k_mask, other=0.0).to(tl.float32),
            tl.load(xk + 7, mask=k_mask, other=0.0).to(tl.float32),
        )
        acc += _tile_dot(codes, scales, tile, k8, k_mask, k8_rows, n_total, xs)
        if SWIGLU:
            up_tile = tile + n_out // 32
            acc_up += _tile_dot(
                codes, scales, up_tile, k8, k_mask, k8_rows, n_total, xs
            )
    y = tl.sum(acc, axis=0) * 16384.0
    if SWIGLU:
        g = y.to(tl.float16).to(tl.float32)
        u = (tl.sum(acc_up, axis=0) * 16384.0).to(tl.float16).to(tl.float32)
        gate = tanh(tl.div_rn(g, gate_cap)) * gate_cap
        sig = tl.div_rn(1.0, 1.0 + exp(-g))
        y = gate * sig * (tanh(tl.div_rn(u, up_cap)) * up_cap)
    cols = tile * 32 + tl.arange(0, 32)
    tl.store(out_ptr + slot * stride_out + cols, y.to(out_ptr.dtype.element_ty))


def config(stage: str) -> tuple[int, int]:
    """(BLOCK_K8, num_warps) of ``"w13"`` or ``"w2"``."""
    return CONFIGS[stage]


def nvfp4_slot_gemv(
    rows: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
    swiglu: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Write `out` [S, N] (with `swiglu` = (gate_cap, up_cap): [S, N / 2],
    the soft-capped SwiGLU of the gate and up halves) for fp16 `rows`
    [S, K], one slot each, on the bank `codes` int32 [E, K, N / 8] and
    `scales` fp16 [E, K / 16, N] of experts `expert_ids` int32 [S]."""
    slots, k_in = rows.shape
    experts, n_total = scales.shape[0], scales.shape[2]
    n_out = n_total // 2 if swiglu else n_total
    assert rows.dtype == out.dtype == scales.dtype == torch.float16
    assert codes.dtype == expert_ids.dtype == torch.int32
    assert codes.shape == (experts, k_in, n_total // CODES_PER_WORD), codes.shape
    assert scales.shape == (experts, k_in // 16, n_total) and k_in % 16 == 0
    assert codes.is_contiguous() and scales.is_contiguous()
    assert n_total % TILE_N == 0 and n_out % TILE_N == 0, n_total
    assert out.shape == (slots, n_out) and out.stride(1) == 1, out.shape
    assert rows.stride(1) == 1 and expert_ids.shape == (slots,)
    if not slots:
        return out
    block_k8, warps = config("w13" if swiglu else "w2")
    gate_cap, up_cap = swiglu if swiglu else (1.0, 1.0)
    grid = (slots, n_out // TILE_N)
    _nvfp4_slot_gemv_kernel[grid](
        rows, codes, scales, out, expert_ids,
        rows.stride(0), codes.stride(0), scales.stride(0), out.stride(0),
        n_out, k_in, experts, float(gate_cap), float(up_cap),
        SWIGLU=swiglu is not None, BLOCK_K8=block_k8,
        num_warps=warps,
    )  # fmt: skip
    return out
