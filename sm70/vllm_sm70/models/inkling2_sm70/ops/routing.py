# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 expert routing and slot dispatch of a small batch as Triton
kernels (sm_70: no tl.dot, libdevice exp).

    router_logits   fp32 [M, E] = x [M, H] @ gate.weight [E, H]^T, x read in
                    its own dtype and widened to fp32
    route           one launch: the top-k of sigmoid(logits) + bias, the
                    routing weights, and with ``dispatch`` the expert-sorted
                    slot order, the TurboMind group tables and the gathered
                    input rows
    combine         one launch: each token's weighted expert outputs summed
                    in fp32, rounded once into `out`

Router math as torch_ref.router (all fp32): weights are sigmoid(logits)
over the chosen experts, divided by their sum + 1e-20, times route_scale.
A token's chosen experts are written in ascending expert id. Top-k ties
keep the lowest expert id, as torch.topk does on CUDA (the CPU kernel of
torch.topk breaks ties another way); the logits must be finite. A row
flagged in `is_padding` gets `pad_id` and weight 0 in every slot.

``dispatch`` sorts the S = M * k slots by expert id, stable in token
order, padding slots last: what moe_tm.py builds with torch.sort.
``pos`` [M, k] is the sorted row of each slot; the sorted ids (padding
clamped to E - 1), the per-expert offsets of those ids and the rows x[t]
at their sorted places are written into the caller's buffers.

Every launch has a grid of static size for static M, reads nothing back to
the host and allocates only its outputs, so it captures in a CUDA graph.
One program holds all M tokens: callers keep M * k <= ``MAX_SLOTS``.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from ._compat import exp, tl, triton

__all__ = ["MAX_SLOTS", "Routed", "combine", "route", "router_logits"]

# Routed slots (M * top_k) up to which the launchers are meant to run.
MAX_SLOTS: Final = 64
GEMV_BLOCK_E: Final = 4
GEMV_BLOCK_K: Final = 512
BLOCK_H: Final = 128


class Routed(NamedTuple):
    """``route`` output: int32 ids and fp32 weights [M, k], ascending ids per
    token; ``pos`` int32 [M, k] with ``dispatch``, else None."""

    ids: torch.Tensor
    weights: torch.Tensor
    pos: torch.Tensor | None


@triton.jit
def _router_logits_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x,
    stride_w,
    stride_out,
    hidden,
    experts,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    e = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e < experts
    acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    for k0 in range(0, hidden, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < hidden
        v = tl.load(x_ptr + row * stride_x + k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + e[:, None].to(tl.int64) * stride_w + k[None, :],
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(w * v[None, :], axis=1)
    tl.store(out_ptr + row * stride_out + e, acc, mask=e_mask)


@triton.jit
def _route_kernel(
    logits_ptr,
    bias_ptr,
    pad_ptr,
    ids_ptr,
    weights_ptr,
    pos_ptr,
    sorted_ptr,
    offsets_ptr,
    src_ptr,
    rows_ptr,
    tokens,
    experts,
    hidden,
    stride_l,
    stride_src,
    stride_rows,
    pad_id,
    route_scale,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_TOP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_PAD: tl.constexpr,
    DISPATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.arange(0, BLOCK_M)
    e = tl.arange(0, BLOCK_E)
    r = tl.arange(0, BLOCK_TOP)
    t_ok = t < tokens
    e_ok = e < experts
    live = t_ok
    if HAS_PAD:
        flag = tl.load(pad_ptr + t, mask=t_ok, other=1).to(tl.int32)
        live = live & (flag == 0)
    logit = tl.load(
        logits_ptr + t[:, None] * stride_l + e[None, :],
        mask=t_ok[:, None] & e_ok[None, :],
        other=0.0,
    )
    score = tl.div_rn(1.0, 1.0 + exp(-logit))
    bias = tl.load(bias_ptr + e, mask=e_ok, other=0.0)
    choice = score + bias[None, :]

    # Top-k: k rounds of (max, lowest id holding it).
    avail = live[:, None] & e_ok[None, :]
    chosen = avail & (e[None, :] < 0)
    for _ in tl.static_range(TOP_K):
        best = tl.max(tl.where(avail, choice, float("-inf")), axis=1)
        tie = avail & (choice == best[:, None])
        pick = tl.min(tl.where(tie, e[None, :], BLOCK_E), axis=1)
        hit = e[None, :] == pick[:, None]
        chosen = chosen | hit
        avail = avail & ~hit

    picked = tl.where(chosen, score, 0.0)
    total = tl.sum(picked, axis=1)
    norm = tl.div_rn(picked, total[:, None] + 1e-20) * route_scale

    count = chosen.to(tl.int32)
    n_valid = tl.sum(count, axis=1)
    per_expert = tl.sum(count, axis=0)
    upto = tl.cumsum(per_expert, axis=0)
    # Sorted row of chosen (t, e): slots of lower experts, then those of
    # e in earlier tokens.
    place = (upto - per_expert)[None, :] + tl.cumsum(count, axis=0) - count
    valid_slots = tl.sum(per_expert, axis=0)
    invalid = tl.where(t_ok, TOP_K - n_valid, 0)
    invalid_before = tl.cumsum(invalid, axis=0) - invalid

    if DISPATCH:
        h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = t_ok[:, None] & (h < hidden)[None, :]
        x = tl.load(
            src_ptr + t[:, None].to(tl.int64) * stride_src + h[None, :],
            mask=h_mask,
            other=0.0,
        )

    ids = tl.zeros([BLOCK_M, BLOCK_TOP], dtype=tl.int32) + pad_id
    weights = tl.zeros([BLOCK_M, BLOCK_TOP], dtype=tl.float32)
    pos = tl.zeros([BLOCK_M, BLOCK_TOP], dtype=tl.int32)
    rest = chosen
    for j in tl.static_range(TOP_K):
        cur = tl.min(tl.where(rest, e[None, :], BLOCK_E), axis=1)
        here = e[None, :] == cur[:, None]
        rest = rest & ~here
        ok = cur < experts
        w_j = tl.sum(tl.where(here, norm, 0.0), axis=1)
        p_j = tl.where(
            ok,
            tl.sum(tl.where(here, place, 0), axis=1),
            valid_slots + invalid_before + j - n_valid,
        )
        col = r[None, :] == j
        ids = tl.where(col & ok[:, None], cur[:, None], ids)
        weights = tl.where(col & ok[:, None], w_j[:, None], weights)
        pos = tl.where(col, p_j[:, None], pos)
        if DISPATCH:
            tl.store(
                rows_ptr + p_j[:, None].to(tl.int64) * stride_rows + h[None, :],
                x,
                mask=h_mask,
            )

    if pid == 0:
        out = t[:, None] * TOP_K + r[None, :]
        out_mask = t_ok[:, None] & (r < TOP_K)[None, :]
        tl.store(ids_ptr + out, ids, mask=out_mask)
        tl.store(weights_ptr + out, weights, mask=out_mask)
        if DISPATCH:
            tl.store(pos_ptr + out, pos, mask=out_mask)
            clamped = tl.where((ids >= 0) & (ids < experts), ids, experts - 1)
            tl.store(sorted_ptr + pos, clamped, mask=out_mask)
            # offsets[0] = 0, offsets[e + 1] = slots of experts <= e; the
            # padding slots count as expert E - 1.
            bound = tl.where(e == experts - 1, tokens * TOP_K, upto)
            tl.store(offsets_ptr + 1 + e, bound, mask=e_ok)
            tl.store(offsets_ptr + e, tl.zeros_like(e), mask=e == 0)


@triton.jit
def _combine_kernel(
    y_ptr,
    ids_ptr,
    weights_ptr,
    pos_ptr,
    out_ptr,
    stride_y,
    stride_out,
    hidden,
    experts,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_POS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h < hidden
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        expert = tl.load(ids_ptr + t * TOP_K + j)
        weight = tl.load(weights_ptr + t * TOP_K + j)
        if HAS_POS:
            row = tl.load(pos_ptr + t * TOP_K + j).to(tl.int64)
        else:
            row = t * TOP_K + j
        y = tl.load(y_ptr + row * stride_y + h, mask=h_mask, other=0.0)
        part = y.to(tl.float32) * weight
        acc += tl.where((expert >= 0) & (expert < experts), part, 0.0)
    tl.store(
        out_ptr + t * stride_out + h, acc.to(out_ptr.dtype.element_ty), mask=h_mask
    )


def router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 logits [M, E] of x [M, H] (any float dtype) and fp32 weight [E, H]."""
    tokens, hidden = x.shape
    experts = weight.shape[0]
    assert weight.dtype == torch.float32 and weight.shape[1] == hidden
    assert x.stride(-1) == 1 and weight.stride(-1) == 1
    out = torch.empty(tokens, experts, dtype=torch.float32, device=x.device)
    if tokens == 0:
        return out
    grid = (tokens, triton.cdiv(experts, GEMV_BLOCK_E))
    _router_logits_kernel[grid](
        x, weight, out,
        x.stride(0), weight.stride(0), out.stride(0), hidden, experts,
        BLOCK_E=GEMV_BLOCK_E, BLOCK_K=GEMV_BLOCK_K,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out


def route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    route_scale: float,
    is_padding: torch.Tensor | None,
    *,
    pad_id: int,
    dispatch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
) -> Routed:
    """Route the rows of fp32 `logits` [M, E] (see the module doc).

    `dispatch` = (x [M, H], rows [>= S, H], sorted_ids int32 [>= S],
    offsets int32 [E + 1]): the buffers the sorted slots go to.
    """
    tokens, experts = logits.shape
    assert logits.dtype == torch.float32 and logits.stride(-1) == 1
    assert bias.dtype == torch.float32 and bias.shape == (experts,)
    assert 0 < top_k <= experts and not 0 <= pad_id < experts
    device = logits.device
    ids = torch.empty(tokens, top_k, dtype=torch.int32, device=device)
    weights = torch.empty(tokens, top_k, dtype=torch.float32, device=device)
    pos = None
    if dispatch is not None:
        x, rows, sorted_ids, offsets = dispatch
        slots = tokens * top_k
        assert x.shape[0] == tokens and rows.shape[0] >= slots
        assert rows.shape[1] == x.shape[1] and rows.dtype == x.dtype
        assert x.stride(-1) == 1 and rows.stride(-1) == 1
        assert sorted_ids.dtype == offsets.dtype == torch.int32
        assert sorted_ids.numel() >= slots and offsets.numel() >= experts + 1
        pos = torch.empty(tokens, top_k, dtype=torch.int32, device=device)
        hidden = x.shape[1]
        grid = (triton.cdiv(hidden, BLOCK_H),)
        src, src_stride, rows_stride = x, x.stride(0), rows.stride(0)
    else:
        sorted_ids = offsets = rows = src = ids
        hidden = src_stride = rows_stride = 0
        grid = (1,)
    if tokens == 0:
        return Routed(ids, weights, pos)
    _route_kernel[grid](
        logits, bias, is_padding if is_padding is not None else ids,
        ids, weights, pos if pos is not None else ids, sorted_ids, offsets,
        src, rows,
        tokens, experts, hidden, logits.stride(0), src_stride, rows_stride,
        pad_id, float(route_scale),
        TOP_K=top_k,
        BLOCK_M=max(2, triton.next_power_of_2(tokens)),
        BLOCK_E=max(2, triton.next_power_of_2(experts)),
        BLOCK_TOP=max(2, triton.next_power_of_2(top_k)),
        BLOCK_H=BLOCK_H,
        HAS_PAD=is_padding is not None,
        DISPATCH=dispatch is not None,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return Routed(ids, weights, pos)


def combine(
    y: torch.Tensor,
    routed: Routed,
    experts: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """out[t] = sum over j of weights[t, j] * y[row] in fp32 for the slots j
    whose id is in 0..E-1, the slots in ascending j; row = pos[t, j], or
    t * k + j without ``pos``. One rounding into the dtype of `out` [M, H]."""
    tokens, top_k = routed.ids.shape
    hidden = out.shape[1]
    assert y.shape[1] == hidden and y.stride(-1) == 1 and out.stride(-1) == 1
    if tokens == 0:
        return out
    pos = routed.pos
    grid = (tokens, triton.cdiv(hidden, BLOCK_H))
    _combine_kernel[grid](
        y, routed.ids, routed.weights, pos if pos is not None else routed.ids, out,
        y.stride(0), out.stride(0), hidden, experts,
        TOP_K=top_k, BLOCK_H=BLOCK_H, HAS_POS=pos is not None,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out
