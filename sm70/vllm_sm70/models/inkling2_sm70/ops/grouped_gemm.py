# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts held in the model dtype, without a host sync.

    grouped_expert_sum   this rank's weighted expert sum of x [M, H] into
                         `output`, the torch expert sum of moe.py as static
                         shapes: the ops depend on M and the weight shapes
                         alone, so one call captures in a CUDA graph

A routed slot is a (token, expert) pair, S = M * top_k of them, each
token's slots in ascending expert order. Tokens run in chunks of at most
``CHUNK_TOKENS``. Per chunk, the slots go two ways:

- capacity: each expert holds ``capacity(S, E)`` rows; a slot whose rank
  among its expert's slots is below it is gathered there, and one cuBLAS
  ``bmm`` per projection runs all experts (zero rows for empty places).
  Capacity 0 (S <= ``SLOT_MAX``) skips this way.
- slot: every other routed slot runs as a GEMV on its own expert's weights,
  read in place: ``_slot_gemv_kernel`` once for W13 and once for W2, one
  program per (slot, block of outputs); a program of a slot that runs
  elsewhere writes zeros.

Between the projections the soft-capped SwiGLU (activation.py) rounds the
fp32 caps once to the model dtype. Each slot's W2 output is in the model
dtype, as F.linear gives it; the routing weights apply in fp32 and a
token's slots are summed in fp32 and rounded once into `output`. Slots
with an expert id outside 0..E-1 add nothing; a row whose slots all are
such is zero.

``kernels=False`` runs the slot GEMVs and the SwiGLU as torch ops
(per-slot weight gathers and ``bmm``) for tensors off the GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch

from ._compat import tl, triton
from .activation import softcapped_swiglu_out

__all__ = ["CHUNK_TOKENS", "SLOT_MAX", "capacity", "grouped_expert_sum"]

# Routed slots up to which every slot takes the GEMV kernel.
SLOT_MAX: Final = 64
# Capacity per expert: CAP_FACTOR times the mean slots per expert, rounded
# up to CAP_ALIGN rows.
CAP_FACTOR: Final = 2
CAP_ALIGN: Final = 8
# Tokens per chunk: bounds the gathered rows (E * capacity) and temporaries.
CHUNK_TOKENS: Final = 512
BLOCK_N: Final = 16
BLOCK_K: Final = 128


def capacity(slots: int, experts: int) -> int:
    """Rows per expert of the capacity way for `slots` routed slots."""
    if slots <= SLOT_MAX:
        return 0
    rows = -(-CAP_FACTOR * slots // experts)
    return -(-rows // CAP_ALIGN) * CAP_ALIGN


@triton.jit
def _slot_gemv_kernel(
    in_ptr,
    w_ptr,
    out_ptr,
    expert_ptr,
    stride_in,
    stride_we,
    stride_wn,
    stride_out,
    n_out,
    k_in,
    ROW_DIV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out[s, n] = sum_k in[s // ROW_DIV, k] * w[expert[s], n, k] in fp32,
    rounded to the dtype of out; zeros where expert[s] < 0."""
    slot = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < n_out
    expert = tl.load(expert_ptr + slot)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    if expert >= 0:
        row = in_ptr + (slot // ROW_DIV) * stride_in
        w_rows = (
            w_ptr
            + expert.to(tl.int64) * stride_we
            + n[:, None].to(tl.int64) * stride_wn
        )
        for k0 in range(0, k_in, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < k_in
            v = tl.load(row + k, mask=k_mask, other=0.0).to(tl.float32)
            w = tl.load(
                w_rows + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
            )
            acc += tl.sum(w.to(tl.float32) * v[None, :], axis=1)
    tl.store(
        out_ptr + slot * stride_out + n, acc.to(out_ptr.dtype.element_ty), mask=n_mask
    )


def _slot_gemv(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    experts: torch.Tensor,
    row_div: int,
    *,
    kernels: bool,
) -> torch.Tensor:
    """[S, N]: row s is inputs[s // row_div] times weight[experts[s]]^T
    (weight [E, N, K]), zeros where experts[s] < 0."""
    slots = experts.shape[0]
    n_out, k_in = weight.shape[1], weight.shape[2]
    if not kernels:
        rows = (
            inputs
            if row_div == 1
            else inputs.repeat_interleave(row_div, 0, output_size=slots)
        )
        w = weight.index_select(0, experts.clamp(min=0))
        out = torch.bmm(w, rows.unsqueeze(-1)).squeeze(-1)
        return out.masked_fill_((experts < 0).unsqueeze(-1), 0.0)
    out = inputs.new_empty(slots, n_out)
    if slots == 0:
        return out
    assert inputs.stride(-1) == 1 and weight.stride(-1) == 1
    grid = (slots, triton.cdiv(n_out, BLOCK_N))
    _slot_gemv_kernel[grid](
        inputs, weight, out, experts,
        inputs.stride(0), weight.stride(0), weight.stride(1), out.stride(0),
        n_out, k_in,
        ROW_DIV=row_div, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out


def _swiglu(gate_up: torch.Tensor, gate_cap: float, up_cap: float, kernels: bool):
    out = gate_up.new_empty(gate_up.shape[0], gate_up.shape[1] // 2)
    return softcapped_swiglu_out(gate_up, out, gate_cap, up_cap, kernel=kernels)


def grouped_expert_sum(
    x: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    output: torch.Tensor,
    *,
    kernels: bool = True,
    act_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Write the weighted expert sum of x [M, H] into `output` [M, H].

    top_k_index, top_k_weights [M, k]; w13 [E, 2 * I, H] (gate rows first),
    w2 [E, H, I]. `act_hook` maps each SwiGLU output (capacity rows, slot
    rows) before its W2 projection. See the module doc."""
    for start in range(0, x.shape[0], CHUNK_TOKENS):
        stop = min(start + CHUNK_TOKENS, x.shape[0])
        _chunk(
            x[start:stop], top_k_index[start:stop], top_k_weights[start:stop],
            w13, w2, gate_cap, up_cap, output[start:stop], kernels, act_hook,
        )  # fmt: skip


def _chunk(
    x, top_k_index, top_k_weights, w13, w2, gate_cap, up_cap, output, kernels, act_hook
):
    tokens, top_k = top_k_index.shape
    experts, hidden = w13.shape[0], x.shape[1]
    slots = tokens * top_k
    ids = top_k_index.long()
    ids = ids.masked_fill((ids < 0) | (ids > experts), experts)
    ids, order = torch.sort(ids, dim=1, stable=True)
    valid = ids < experts
    weights = top_k_weights.gather(1, order).float().masked_fill(~valid, 0.0)
    flat, valid = ids.reshape(-1), valid.reshape(-1)
    x = x.contiguous()
    cap = capacity(slots, experts)
    if cap:
        # Rank of each slot among its expert's slots, from a stable sort.
        by_expert = torch.argsort(flat, stable=True)
        counts = flat.new_zeros(experts + 1).scatter_add_(
            0, flat, torch.ones_like(flat)
        )
        starts = counts.cumsum(0) - counts
        arange = torch.arange(slots, device=x.device)
        rank = torch.empty_like(flat).scatter_(
            0,
            by_expert,
            arange - starts.index_select(0, flat.index_select(0, by_expert)),
        )
        held = valid & (rank < cap)
        places = experts * cap
        dest = torch.where(held, flat * cap + rank, places + arange)
        source = torch.full(
            (places + slots,), tokens, dtype=torch.long, device=x.device
        )
        source.scatter_(0, dest, arange // top_k)
        padded = torch.cat([x, x.new_zeros(1, hidden)])
        held_x = padded.index_select(0, source[:places]).view(experts, cap, hidden)
        gate_up = torch.bmm(held_x, w13.transpose(1, 2))
        act = _swiglu(gate_up.view(places, -1), gate_cap, up_cap, kernels)
        if act_hook is not None:
            act = act_hook(act)
        held_y = torch.bmm(act.view(experts, cap, -1), w2.transpose(1, 2))
        held_y = held_y.view(places, hidden).index_select(0, torch.where(held, dest, 0))
        slot_experts = torch.where(valid & ~held, flat, -1)
    else:
        slot_experts = torch.where(valid, flat, -1)
    slot_experts = slot_experts.to(torch.int32)
    gate_up = _slot_gemv(x, w13, slot_experts, top_k, kernels=kernels)
    act = _swiglu(gate_up, gate_cap, up_cap, kernels)
    if act_hook is not None:
        act = act_hook(act)
    y = _slot_gemv(act, w2, slot_experts, 1, kernels=kernels)
    if cap:
        y = torch.where(held.unsqueeze(-1), held_y, y)
    total = torch.bmm(weights.unsqueeze(1), y.view(tokens, top_k, hidden).float())
    output.copy_(total.view(tokens, hidden))
