# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 NVFP4 experts on the TurboMind W4A16 dense stage.

One ``Nvfp4TmExperts`` per NVFP4 layer and TP rank holds the two packed banks
(``tm.nvfp4_sm70_prepare`` per expert, gate-first W13) and their strided
pointer tables. ``expert_sum`` sorts the routed slots by expert, runs W13
(k = H, n = 2 * I / tp), the soft-capped SwiGLU in fp32, W2 (k = I / tp,
n = H), and sums a token's experts in fp32, as the torch path of moe.py does.

The slots run in chunks of at most ``CHUNK_ROWS`` rows, so the stage scratch
and the fp32 temporaries do not grow with the batch. Stage call forms per
chunk: up to ``COMPACT_MAX_SLOTS`` slots each slot is its own one-row group
(one launch per stage; ``VLLM_SM70_FLINT_MOE_COMPACT=0`` turns this off); a
batch of up to ``COMPACT_BATCH_MAX_SLOTS`` slots runs in compact chunks of
``COMPACT_MAX_SLOTS``; above that the groups are the per-expert offsets over
all experts.

Every routed slot runs, so ``expert_sum`` reads nothing back to the host: a
slot with no expert (padding) is computed on expert E - 1 and zeroed.
``routed_expert_sum`` takes the router logits of a small batch instead and
routes, sorts and gathers the slots in one launch and sums them in another
(ops/routing.py); the stages and the SwiGLU are those of ``expert_sum``.

Switch: the route is on where ``rejection`` admits a layer;
``VLLM_SM70_INKLING_NVFP4_TM=0`` keeps every layer on the torch path of
moe.py, which holds all experts in fp16.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Final

import torch
from vllm.logger import init_logger

from vllm_sm70 import envs
from vllm_sm70.fused_moe.method import COMPACT_MAX_SLOTS

from .expert_weights import NVFP4_BLOCK, Nvfp4Codes, RangeCheck
from .ops import routing, torch_ref
from .ops.activation import softcapped_swiglu_out

logger = init_logger("vllm.sm70.models.inkling2.moe_tm")

SWITCH: Final = "VLLM_SM70_INKLING_NVFP4_TM"
SWITCHED_OFF: Final = f"{SWITCH}=0"
REQUIRED_OPS: Final = (
    "nvfp4_sm70_prepare",
    "awq_moe_build_strided_ptrs",
    "nvfp4_moe_dense_stage_sm70_out",
)
# (hidden, global intermediate, experts, top-k)
INKLING_CONTRACTS: Final = frozenset({(2304, 1152, 128, 8)})
# Only the TP2 bank shapes (W13 n=1152, W2 k=576) have run on a GPU.
INKLING_TP_SIZES: Final = (2,)
# tm.nvfp4_sm70_prepare packs 8 codes of the output axis into one int32.
CODES_PER_WORD: Final = 8
FP16_MIN_NORMAL: Final = 2.0**-14
FP16_MAX: Final = 65504.0
# Routed slots per stage call: bounds the workspace (TP2: 4096 x 6336 fp16,
# 52 MB per device) and the fp32 temporaries of one call.
CHUNK_ROWS: Final = 4096
# Batches of up to this many slots run as chunks of COMPACT_MAX_SLOTS compact
# slot groups, because the per-expert groups over all 128 experts cost about
# 128 launches per stage call at any batch size. The bound covers the largest
# captured verify step: 8 requests of 8 tokens at top-k 8 is 512 slots.
COMPACT_BATCH_MAX_SLOTS: Final = 8 * COMPACT_MAX_SLOTS

Contract = tuple[int, int, int, int]


class Rejected(ValueError):
    """A layer whose banks the route refuses at load; it stays on torch."""


def enabled() -> bool:
    """False only when VLLM_SM70_INKLING_NVFP4_TM=0."""
    return bool(envs.VLLM_SM70_INKLING_NVFP4_TM)


def turbomind() -> Any | None:
    """vllm_sm70.ops.tm when it has every required op, else None."""
    try:
        from vllm_sm70.ops import tm
    except ImportError:
        return None
    probe = getattr(tm, "has_op", None)
    for name in REQUIRED_OPS:
        present = probe(name) if callable(probe) else callable(getattr(tm, name, None))
        if not present:
            return None
    return tm


def _real_extension() -> bool:
    from vllm_sm70 import _ext

    return _ext.is_loaded()


def rejection(
    contract: Contract,
    tp_size: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> str | None:
    """Why a layer of this shape stays on the torch path, or None (admitted).

    ``contract`` is (hidden, global intermediate, experts, top-k). ``device``
    is checked when given (at load): the built ops take CUDA tensors only.
    """
    if not enabled():
        return SWITCHED_OFF
    if dtype != torch.float16:
        return f"model dtype {dtype} is not float16"
    if tp_size not in INKLING_TP_SIZES:
        return (
            f"TP size {tp_size} is not in {INKLING_TP_SIZES}, the GPU-checked "
            "bank shapes"
        )
    if contract not in INKLING_CONTRACTS:
        return f"shape {contract} is not in {sorted(INKLING_CONTRACTS)}"
    hidden, intermediate, _, _ = contract
    local = intermediate // tp_size
    if intermediate % tp_size or local % NVFP4_BLOCK or hidden % NVFP4_BLOCK:
        return (
            f"hidden {hidden} and local intermediate {intermediate}/{tp_size} "
            f"must be multiples of {NVFP4_BLOCK}"
        )
    if turbomind() is None:
        return "vllm_sm70.ops.tm lacks " + ", ".join(REQUIRED_OPS)
    if device is not None and device.type != "cuda" and _real_extension():
        return f"device {device} is not CUDA"
    return None


def bank_nbytes(experts: int, k: int, n: int) -> int:
    """Bytes of one prepared bank: int32 [E, K, N/8] codes + fp16 [E, K/16, N]."""
    return experts * (k * (n // CODES_PER_WORD) * 4 + (k // NVFP4_BLOCK) * n * 2)


def layer_nbytes(experts: int, hidden: int, local_intermediate: int) -> int:
    """Bytes of the W13 and W2 banks of one layer on one rank."""
    return bank_nbytes(experts, hidden, 2 * local_intermediate) + bank_nbytes(
        experts, local_intermediate, hidden
    )


def prepared_scales(
    codes: Nvfp4Codes, range_check: RangeCheck | None = None
) -> tuple[torch.Tensor, int]:
    """(fp16 [E, K/16, N] scales with the global folded in, pre-shift).

    The scales are block_scale * global_scale * 2**shift. ``shift`` is 0
    unless a nonzero scale would be subnormal in fp16; then it is the
    smallest power of two that makes every nonzero scale normal (``Bank``
    refuses it). Raises ValueError when a scale is not finite or no shift
    fits the fp16 range.
    ``range_check(name, fp32 scales, fp16 scales)`` sees the cast.
    """
    scales = codes.block_scale.float() * codes.global_scale.float().reshape(-1, 1, 1)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError(f"{codes.name}: NVFP4 scales are not finite")
    magnitude = scales.abs()
    nonzero = magnitude[magnitude > 0]
    shift = 0
    if nonzero.numel():
        low, high = float(nonzero.min()), float(magnitude.max())
        while low * 2.0**shift < FP16_MIN_NORMAL:
            shift += 1
        if high * 2.0**shift > FP16_MAX:
            raise ValueError(
                f"{codes.name}: NVFP4 scales span {low:.3e}..{high:.3e}, "
                "which no power-of-two shift fits into normal fp16"
            )
    if shift:
        scales = scales * 2.0**shift
    cast = scales.half()
    if range_check is not None:
        range_check(f"{codes.name} (NVFP4 scales)", scales, cast)
    return cast.transpose(1, 2).contiguous(), shift


class Bank:
    """One stage's prepared experts: ``weight`` int32 [E, K, N/8], ``scales``
    fp16 [E, K/16, N] and the pointer tables into them. Raises Rejected when
    the scales need a pre-shift: the fp16 stage output would carry the
    2**shift and could overflow before any compensation."""

    def __init__(
        self,
        tm: Any,
        codes: Nvfp4Codes,
        device: torch.device,
        range_check: RangeCheck | None = None,
    ) -> None:
        experts, n, k = codes.codes.shape
        if k % NVFP4_BLOCK or n % CODES_PER_WORD:
            raise ValueError(
                f"{codes.name}: [N {n}, K {k}] needs K % {NVFP4_BLOCK} == 0 "
                f"and N % {CODES_PER_WORD} == 0"
            )
        scales, shift = prepared_scales(codes, range_check)
        if shift:
            raise Rejected(
                f"{codes.name}: NVFP4 scales below the fp16 normal range need "
                f"a 2**{shift} pre-shift, which the fp16 stage output could "
                "overflow with"
            )
        self.name = codes.name
        self.k, self.n, self.num_experts = k, n, experts
        qweight = codes.codes.to(device)
        scales = scales.to(device)
        weights, prepared, meta = [], [], None
        for expert in range(experts):
            w, s, meta = tm.nvfp4_sm70_prepare(
                qweight[expert].t().contiguous(), scales[expert], NVFP4_BLOCK, False
            )
            weights.append(w)
            prepared.append(s)
        self.weight = torch.stack(weights)
        self.scales = torch.stack(prepared)
        k_ld, q_ld = (int(v) for v in meta.tolist())
        self.ptrs_w, self.ptrs_s = tm.awq_moe_build_strided_ptrs(
            self.weight, self.scales, k_ld, q_ld, experts
        )

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.weight, self.scales, self.ptrs_w, self.ptrs_s)
        )


class Workspace:
    """Stage scratch of one device and shape, shared by its layers (they run
    one after another). Rows grow on demand up to ``CHUNK_ROWS``; ``reserve``
    sizes them at load, for the engine's batch, so a captured decode step
    never grows them (growth inside a capture raises)."""

    _CACHE: ClassVar[dict[tuple[str, int, int, int], Workspace]] = {}

    def __init__(
        self, device: torch.device, experts: int, hidden: int, local: int
    ) -> None:
        i32 = {"dtype": torch.int32, "device": device}
        self.device = device
        self.widths = {
            "permuted_input": hidden,
            "gate_up": 2 * local,
            "intermediate": local,
            "sorted_output": hidden,
        }
        self.rows = 0
        self.buffers: dict[str, torch.Tensor] = {}
        self.expert_offsets = torch.zeros(experts + 1, **i32)
        self.dense_expert_ids = torch.arange(experts, **i32)
        self.compact_offsets = torch.arange(COMPACT_MAX_SLOTS + 1, **i32)
        self.compact_expert_ids = torch.zeros(COMPACT_MAX_SLOTS, **i32)
        self.expert_bounds = torch.arange(experts, dtype=torch.int64, device=device)
        self.reserve(COMPACT_MAX_SLOTS)

    @classmethod
    def get(
        cls, device: torch.device, experts: int, hidden: int, local: int
    ) -> Workspace:
        key = (str(device), experts, hidden, local)
        ws = cls._CACHE.get(key)
        if ws is None:
            ws = cls._CACHE[key] = cls(device, experts, hidden, local)
        return ws

    @classmethod
    def clear(cls) -> None:
        cls._CACHE.clear()

    def reserve(self, rows: int) -> None:
        rows = min(rows, CHUNK_ROWS)
        if rows <= self.rows:
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            # A graph would keep the replaced buffers' addresses.
            raise RuntimeError(
                f"Inkling 2 TurboMind workspace grows to {rows} rows inside a "
                f"CUDA graph capture; it holds {self.rows}."
            )
        self.buffers = {}
        for name, width in self.widths.items():
            self.buffers[name] = torch.empty(
                rows, width, dtype=torch.float16, device=self.device
            )
        self.rows = rows

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers.values())


class Nvfp4TmExperts:
    """The NVFP4 experts of one layer on one rank, as TurboMind banks."""

    def __init__(
        self,
        tm: Any,
        w13: Nvfp4Codes,
        w2: Nvfp4Codes,
        *,
        device: torch.device,
        gate_cap: float,
        up_cap: float,
        max_slots: int = 0,
        compact: bool = True,
        range_check: RangeCheck | None = None,
    ) -> None:
        self.tm = tm
        self.compact = compact
        self.w13 = Bank(tm, w13, device, range_check)
        self.w2 = Bank(tm, w2, device, range_check)
        hidden, local = self.w13.k, self.w2.k
        if (
            self.w13.n != 2 * local
            or self.w2.n != hidden
            or self.w13.num_experts != self.w2.num_experts
        ):
            raise ValueError(
                f"{w13.name}: W13 [N {self.w13.n}, K {hidden}] and W2 "
                f"[N {self.w2.n}, K {local}] are not one layer's pair"
            )
        self.num_experts = self.w13.num_experts
        self.hidden_size, self.local_intermediate = hidden, local
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        self.workspace = Workspace.get(device, self.num_experts, hidden, local)
        self.workspace.reserve(max_slots)

    def nbytes(self) -> int:
        return self.w13.nbytes() + self.w2.nbytes()

    def stage_groups(
        self, sorted_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """(offsets, expert ids, group count) for `sorted_ids`, the ascending
        expert ids 0..E-1 of the slots."""
        ws = self.workspace
        slots = sorted_ids.numel()
        if self.compact and slots <= COMPACT_MAX_SLOTS:
            ids = ws.compact_expert_ids[:slots]
            ids.copy_(sorted_ids)
            return ws.compact_offsets[: slots + 1], ids, slots
        # cumsum(bincount) without bincount's host sync on CUDA.
        ws.expert_offsets[1:] = torch.searchsorted(
            sorted_ids, ws.expert_bounds, right=True
        )
        return ws.expert_offsets, ws.dense_expert_ids, self.num_experts

    def expert_sum(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        output: torch.Tensor,
        act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fused_act: bool = False,
    ) -> None:
        """Write this rank's weighted expert sum of x [M, H] fp16 into `output`.

        top_k_index, top_k_weights [M, k]; ids outside 0..E-1 select no expert.
        `act_quant` (act_quant.py) rewrites the SwiGLU output before W2.
        `fused_act`: the SwiGLU runs as one kernel (fused_small.py).
        """
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise TypeError(
                f"Inkling 2 NVFP4 experts need fp16 [M, {self.hidden_size}] "
                f"input, got {x.dtype} {tuple(x.shape)}"
            )
        top_k = top_k_index.shape[-1]
        none = self.num_experts
        flat = top_k_index.reshape(-1)
        slots = flat.numel()
        if not slots:
            output.zero_()
            return
        flat = flat.masked_fill((flat < 0) | (flat > none), none)
        sorted_ids, order = torch.sort(flat, stable=True)
        skip = sorted_ids == none
        sorted_ids = sorted_ids.clamp(max=none - 1)
        tokens = order // top_k
        weights = top_k_weights.reshape(-1)[order].float()
        total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        self.workspace.reserve(slots)
        step = CHUNK_ROWS
        if self.compact and COMPACT_MAX_SLOTS < slots <= COMPACT_BATCH_MAX_SLOTS:
            step = min(step, COMPACT_MAX_SLOTS)
        for start in range(0, slots, step):
            part = slice(start, min(start + step, slots))
            self._chunk(
                x, tokens[part], sorted_ids[part], weights[part], skip[part], total,
                act_quant, fused_act,
            )  # fmt: skip
        output.copy_(total)

    def _chunk(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        sorted_ids: torch.Tensor,
        weights: torch.Tensor,
        skip: torch.Tensor,
        total: torch.Tensor,
        act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fused_act: bool = False,
    ) -> None:
        """Add the weighted expert outputs of up to CHUNK_ROWS slots, sorted
        by expert, into the fp32 `total` [M, H]; slots where the bool `skip`
        is set add zero."""
        ws = self.workspace
        slots = tokens.numel()
        rows, gate_up, act, sorted_out = (
            ws.buffers[name][:slots] for name in ws.widths
        )
        torch.index_select(x, 0, tokens, out=rows)
        offsets, expert_ids, groups = self.stage_groups(sorted_ids)
        self._stages(rows, gate_up, act, sorted_out, offsets, expert_ids, groups,
                     act_quant, fused_act)  # fmt: skip
        weighted = sorted_out.float() * weights[:, None]
        total.index_add_(0, tokens, weighted.masked_fill_(skip[:, None], 0.0))

    def _stages(
        self,
        rows: torch.Tensor,
        gate_up: torch.Tensor,
        act: torch.Tensor,
        sorted_out: torch.Tensor,
        offsets: torch.Tensor,
        expert_ids: torch.Tensor,
        groups: int,
        act_quant: Callable[[torch.Tensor], torch.Tensor] | None,
        fused_act: bool = False,
    ) -> None:
        """W13, the soft-capped SwiGLU in fp32 (one kernel with `fused_act`)
        and W2 of the expert-sorted `rows` into `sorted_out`, through the
        scratch `gate_up` and `act`."""
        stage = self.tm.nvfp4_moe_dense_stage_sm70_out
        stage(
            gate_up, rows, offsets, expert_ids,
            self.w13.ptrs_w, self.w13.ptrs_s,
            groups, self.hidden_size, 2 * self.local_intermediate, NVFP4_BLOCK,
        )  # fmt: skip
        if fused_act:
            softcapped_swiglu_out(gate_up, act, self.gate_cap, self.up_cap)
        else:
            gate, up = gate_up.float().chunk(2, dim=-1)
            act.copy_(
                torch_ref.softcap(gate, self.gate_cap)
                * torch.sigmoid(gate)
                * torch_ref.softcap(up, self.up_cap)
            )
        if act_quant is not None:
            act.copy_(act_quant(act))
        stage(
            sorted_out, act, offsets, expert_ids,
            self.w2.ptrs_w, self.w2.ptrs_s,
            groups, self.local_intermediate, self.hidden_size, NVFP4_BLOCK,
        )  # fmt: skip

    def routed_expert_sum(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        bias: torch.Tensor,
        top_k: int,
        route_scale: float,
        is_padding: torch.Tensor | None,
        output: torch.Tensor,
        act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fused_act: bool = False,
    ) -> None:
        """`expert_sum` of x [M, H] fp16 routed from the fp32 router `logits`
        [M, E] by ops/routing.py, for M * top_k <= routing.MAX_SLOTS; rows
        where the bool `is_padding` is set are zero."""
        tokens = x.shape[0]
        slots = tokens * top_k
        if x.dtype != torch.float16 or x.shape[1] != self.hidden_size:
            raise TypeError(
                f"Inkling 2 NVFP4 experts need fp16 [M, {self.hidden_size}] "
                f"input, got {x.dtype} {tuple(x.shape)}"
            )
        assert 0 < slots <= routing.MAX_SLOTS, slots
        ws = self.workspace
        ws.reserve(slots)
        rows, gate_up, act, sorted_out = (
            ws.buffers[name][:slots] for name in ws.widths
        )
        compact = self.compact and slots <= COMPACT_MAX_SLOTS
        if compact:
            sorted_ids = ws.compact_expert_ids[:slots]
        else:
            sorted_ids = torch.empty(slots, dtype=torch.int32, device=x.device)
        routed = routing.route(
            logits, bias, top_k, route_scale, is_padding,
            pad_id=self.num_experts,
            dispatch=(x.contiguous(), rows, sorted_ids, ws.expert_offsets),
        )  # fmt: skip
        if compact:
            offsets, expert_ids = ws.compact_offsets[: slots + 1], sorted_ids
            groups = slots
        else:
            offsets, expert_ids, groups = (
                ws.expert_offsets, ws.dense_expert_ids, self.num_experts,
            )  # fmt: skip
        self._stages(rows, gate_up, act, sorted_out, offsets, expert_ids, groups,
                     act_quant, fused_act)  # fmt: skip
        routing.combine(sorted_out, routed, self.num_experts, output)


def build(
    w13: Nvfp4Codes,
    w2: Nvfp4Codes,
    *,
    device: torch.device,
    gate_cap: float,
    up_cap: float,
    top_k: int,
    range_check: RangeCheck | None = None,
) -> Nvfp4TmExperts:
    """Banks of one layer on `device`, with scratch for the engine's batch.
    Raises Rejected for banks the route refuses."""
    from vllm_sm70.fused_moe.method import max_batched_tokens
    from vllm_sm70.quant.nvfp4_moe import compact_policy_enabled

    tm = turbomind()
    if tm is None:
        raise RuntimeError(
            "Inkling 2 NVFP4 experts need vllm_sm70.ops.tm with "
            + ", ".join(REQUIRED_OPS)
        )
    return Nvfp4TmExperts(
        tm, w13, w2,
        device=device, gate_cap=gate_cap, up_cap=up_cap,
        max_slots=max_batched_tokens() * top_k,
        compact=compact_policy_enabled(), range_check=range_check,
    )  # fmt: skip
