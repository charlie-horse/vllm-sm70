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

A weight is a tensor in the model dtype or an ``Mxfp8Weight`` (fp16
model): e4m3 codes with one exponent byte per 32 values along K. The slot
GEMVs decode the codes in registers (``_slot_gemv_mx_kernel``); the
capacity way decodes ``DEQUANT_EXPERTS`` experts at a time into a
temporary for its ``bmm``. Every decoded value is exact in fp16, so the
products and sums are those of the dequantized weight.

    routed_expert_sum    the same sum for a batch that routing.py routed
                         (S <= SLOT_MAX): the slot way only, without the
                         sort and masks, the sum in ``routing.combine``
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch

from . import routing
from ._compat import tl, triton
from .activation import softcapped_swiglu_out

__all__ = [
    "CHUNK_TOKENS",
    "DEQUANT_EXPERTS",
    "SLOT_MAX",
    "Mxfp8Weight",
    "capacity",
    "dequantize",
    "expert_weights",
    "grouped_expert_sum",
    "routed_expert_sum",
]

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
# MXFP8: values per exponent byte; experts decoded per capacity-way bmm.
MX_BLOCK: Final = 32
DEQUANT_EXPERTS: Final = 8
DEQUANT_BLOCK: Final = 1024


@dataclass(frozen=True)
class Mxfp8Weight:
    """Expert weight [E, N, K]: value[e, n, k] = e4m3(codes[e, n, k]) *
    2 ** (scale[e, n, (k + offset) // 32] - 127), exact in fp16.

    ``codes`` uint8 [E, N, K] contiguous; ``scale`` uint8 [E, N, B] with
    every byte at most 246 (expert_weights.MXFP8_MAX_SCALE)."""

    codes: torch.Tensor
    scale: torch.Tensor
    offset: int = 0

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape

    def select(self, experts: torch.Tensor | slice) -> Mxfp8Weight:
        if isinstance(experts, slice):
            return Mxfp8Weight(self.codes[experts], self.scale[experts], self.offset)
        return Mxfp8Weight(
            self.codes.index_select(0, experts),
            self.scale.index_select(0, experts),
            self.offset,
        )


ExpertWeight = torch.Tensor | Mxfp8Weight


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


@triton.jit
def _mx_decode(codes, scales):
    """fp32 values of uint8 e4m3 `codes` times 2 ** (`scales` - 127): the
    code's bits placed in an fp16 (value * 2 ** -8, subnormals included),
    times 2 ** (scale - 119) built as fp32 bits."""
    c = codes.to(tl.int32)
    bits = ((c & 0x80) << 8) | ((c & 0x7F) << 7)
    half = bits.to(tl.int16).to(tl.float16, bitcast=True)
    factor = ((scales.to(tl.int32) + 8) << 23).to(tl.float32, bitcast=True)
    return half.to(tl.float32) * factor


@triton.jit
def _slot_gemv_mx_kernel(
    in_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    expert_ptr,
    stride_in,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_out,
    n_out,
    k_in,
    offset,
    ROW_DIV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MX_BLOCK: tl.constexpr,
):
    """``_slot_gemv_kernel`` on an ``Mxfp8Weight``: codes w, scales s."""
    slot = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < n_out
    expert = tl.load(expert_ptr + slot)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    if expert >= 0:
        row = in_ptr + (slot // ROW_DIV) * stride_in
        e = expert.to(tl.int64)
        n64 = n[:, None].to(tl.int64)
        w_rows = w_ptr + e * stride_we + n64 * stride_wn
        s_rows = s_ptr + e * stride_se + n64 * stride_sn
        for k0 in range(0, k_in, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < k_in
            mask = n_mask[:, None] & k_mask[None, :]
            v = tl.load(row + k, mask=k_mask, other=0.0).to(tl.float32)
            codes = tl.load(w_rows + k[None, :], mask=mask, other=0)
            scales = tl.load(
                s_rows + ((k + offset) // MX_BLOCK)[None, :], mask=mask, other=0
            )
            acc += tl.sum(_mx_decode(codes, scales) * v[None, :], axis=1)
    tl.store(
        out_ptr + slot * stride_out + n, acc.to(out_ptr.dtype.element_ty), mask=n_mask
    )


@triton.jit
def _mx_dequant_kernel(
    w_ptr,
    s_ptr,
    out_ptr,
    k_in,
    stride_w,
    stride_s,
    stride_out,
    offset,
    BLOCK: tl.constexpr,
    MX_BLOCK: tl.constexpr,
):
    """out[r, k] = the fp16 value of codes w[r, k] with scales s[r, :]."""
    r = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = k < k_in
    codes = tl.load(w_ptr + r * stride_w + k, mask=mask, other=0)
    scales = tl.load(
        s_ptr + r * stride_s + (k + offset) // MX_BLOCK, mask=mask, other=0
    )
    value = _mx_decode(codes, scales)
    tl.store(
        out_ptr + r * stride_out + k, value.to(out_ptr.dtype.element_ty), mask=mask
    )


def dequantize(weight: Mxfp8Weight, *, kernels: bool) -> torch.Tensor:
    """[E, N, K] fp16 values of `weight`; torch ops unless `kernels`."""
    experts, n_out, k_in = weight.shape
    codes, scale = weight.codes, weight.scale
    if not kernels:
        blocks = (torch.arange(k_in, device=codes.device) + weight.offset) // MX_BLOCK
        factor = torch.exp2(scale.float() - 127).index_select(-1, blocks)
        values = codes.view(torch.float8_e4m3fn).float() * factor
        return values.to(torch.float16)
    out = torch.empty(weight.shape, dtype=torch.float16, device=codes.device)
    rows = experts * n_out
    if rows == 0:
        return out
    codes, scale = codes.reshape(rows, k_in), scale.reshape(rows, -1)
    assert codes.stride(-1) == 1 and scale.stride(-1) == 1
    grid = (rows, triton.cdiv(k_in, DEQUANT_BLOCK))
    _mx_dequant_kernel[grid](
        codes, scale, out, k_in, codes.stride(0), scale.stride(0), k_in,
        weight.offset, BLOCK=DEQUANT_BLOCK, MX_BLOCK=MX_BLOCK, num_warps=4,
    )  # fmt: skip
    return out


def expert_weights(
    weight: ExpertWeight, experts: torch.Tensor | int, *, kernels: bool = False
) -> torch.Tensor:
    """weight[experts] in the model dtype: [S, N, K] for an index tensor,
    [N, K] for an int."""
    if isinstance(experts, int):
        return expert_weights(weight, slice(experts, experts + 1), kernels=kernels)[0]
    if isinstance(weight, torch.Tensor):
        if isinstance(experts, slice):
            return weight[experts]
        return weight.index_select(0, experts)
    return dequantize(weight.select(experts), kernels=kernels)


def _expert_bmm(x: torch.Tensor, weight: ExpertWeight, kernels: bool) -> torch.Tensor:
    """x [E, C, K] times each expert's weight [N, K]^T -> [E, C, N]."""
    if isinstance(weight, torch.Tensor):
        return torch.bmm(x, weight.transpose(1, 2))
    experts = weight.shape[0]
    out = x.new_empty(experts, x.shape[1], weight.shape[1])
    for start in range(0, experts, DEQUANT_EXPERTS):
        part = slice(start, min(start + DEQUANT_EXPERTS, experts))
        w = expert_weights(weight, part, kernels=kernels)
        torch.bmm(x[part], w.transpose(1, 2), out=out[part])
    return out


def _slot_gemv(
    inputs: torch.Tensor,
    weight: ExpertWeight,
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
        w = expert_weights(weight, experts.clamp(min=0))
        out = torch.bmm(w, rows.unsqueeze(-1)).squeeze(-1)
        return out.masked_fill_((experts < 0).unsqueeze(-1), 0.0)
    out = inputs.new_empty(slots, n_out)
    if slots == 0:
        return out
    if isinstance(weight, Mxfp8Weight):
        codes, scale = weight.codes, weight.scale
        assert inputs.stride(-1) == 1 and codes.stride(-1) == 1
        assert scale.stride(-1) == 1
        grid = (slots, triton.cdiv(n_out, BLOCK_N))
        _slot_gemv_mx_kernel[grid](
            inputs, codes, scale, out, experts,
            inputs.stride(0), codes.stride(0), codes.stride(1),
            scale.stride(0), scale.stride(1), out.stride(0),
            n_out, k_in, weight.offset,
            ROW_DIV=row_div, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, MX_BLOCK=MX_BLOCK,
            num_warps=4, enable_fp_fusion=False,
        )  # fmt: skip
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
    w13: ExpertWeight,
    w2: ExpertWeight,
    gate_cap: float,
    up_cap: float,
    output: torch.Tensor,
    *,
    kernels: bool = True,
    act_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Write the weighted expert sum of x [M, H] into `output` [M, H].

    top_k_index, top_k_weights [M, k]; w13 [E, 2 * I, H] (gate rows first),
    w2 [E, H, I], tensors or ``Mxfp8Weight``. `act_hook` maps each SwiGLU output (capacity rows, slot
    rows) before its W2 projection. See the module doc."""
    for start in range(0, x.shape[0], CHUNK_TOKENS):
        stop = min(start + CHUNK_TOKENS, x.shape[0])
        _chunk(
            x[start:stop], top_k_index[start:stop], top_k_weights[start:stop],
            w13, w2, gate_cap, up_cap, output[start:stop], kernels, act_hook,
        )  # fmt: skip


def routed_expert_sum(
    x: torch.Tensor,
    routed: routing.Routed,
    w13: ExpertWeight,
    w2: ExpertWeight,
    gate_cap: float,
    up_cap: float,
    output: torch.Tensor,
    *,
    act_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """`grouped_expert_sum` with the kernels, for `routed` from
    ``routing.route(..., pad_id=-1)`` of x [M, H] with M * k <= SLOT_MAX."""
    tokens, top_k = routed.ids.shape
    assert tokens * top_k <= SLOT_MAX and x.shape[0] == tokens
    x = x.contiguous()
    slot_experts = routed.ids.reshape(-1)
    gate_up = _slot_gemv(x, w13, slot_experts, top_k, kernels=True)
    act = _swiglu(gate_up, gate_cap, up_cap, True)
    if act_hook is not None:
        act = act_hook(act)
    y = _slot_gemv(act, w2, slot_experts, 1, kernels=True)
    routing.combine(y, routed, w13.shape[0], output)


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
        gate_up = _expert_bmm(held_x, w13, kernels)
        act = _swiglu(gate_up.view(places, -1), gate_cap, up_cap, kernels)
        if act_hook is not None:
            act = act_hook(act)
        held_y = _expert_bmm(act.view(experts, cap, -1), w2, kernels)
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
