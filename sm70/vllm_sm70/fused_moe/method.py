# SPDX-License-Identifier: Apache-2.0
"""SM70 direct-apply MoE base and its shared routing runtime (M4).

What: ``Sm70MoEMethodBase`` adapts the fork's SM70 MoE methods to the UP
  4ba787e MoE interface in "direct apply" style: not monolithic and no
  modular kernel (``moe_kernel`` stays None). ``Sm70MoEWorkspace`` holds the
  persistent routing and activation scratch, allocated once per (device,
  shape) in process_weights_after_loading and shared by all layers (they run
  one after another). ``route()``, ``silu_and_mul()`` and ``unpermute()`` wrap
  UP's ``_moe_C`` / ``_C`` ops.
Why: with ``is_monolithic`` False and ``moe_kernel`` None, UP
  ``MoERunner._apply_quant_method`` (runner/moe_runner.py:620-645) routes with
  its own router (fp32 weights, int32 ids by default, router/base_router.py
  :156-165) and calls ``quant_method.apply(layer, x, topk_weights, topk_ids,
  shared_experts, shared_experts_input)`` through ``RoutedExperts.
  forward_modular`` (routed_experts.py:1394-1404). ``_fused_output_is_reduced``
  is then False (moe_runner.py:421-426), so the runner owns the single TP
  all-reduce (:483-515). Methods here must never all-reduce. Shared experts
  run in the runner (NO_OVERLAP, :616-618).
Kill-switch: none here; each method carries its own.

Routing follows the K03 contract agreed with Kai (2026-11-05):
metadata comes from UP ``_moe_C::moe_prepare_scatter`` (csrc/moe/
torch_bindings.cpp:122-126) with the fork -> UP mapping
inv_permuted_idx -> scatter_idx, permuted_idx -> sorted_rows,
permuted_experts_id -> sorted_experts, and input_row_indices = sorted_rows //
topk. ``moe_permute_with_scratch`` (:113-119) materializes the permuted input
when the indexed W13 op is absent. ``moe_unpermute`` (:129-131) does the
weighted top-k sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger("vllm.sm70.fused_moe")

# Validated topologies. TP4+ is out of scope and untested: refuse, never fall
# back to upstream (which has no cc 7.0 MoE kernel either).
SUPPORTED_TP_SIZES = (1, 2)
DEFAULT_MAX_TOKENS = 8192  # FORK nix/module.nix serves --max-num-batched-tokens 8192
# Largest routed-slot count (M * top_k) served as compact slot groups: the
# TurboMind compact grouped bound (tm_moe_sm70.cu kNvfp4MaxCompactGroups).
# ported from FORK nvfp4_sm70_moe.py:88 (_COMPACT_GROUPED_MAX_SLOTS)
COMPACT_MAX_SLOTS = 80


def use_compact_grouped(num_tokens: int, top_k: int) -> bool:
    """Bound compact dispatch by its routed-row workload, not batch size.

    ported from FORK nvfp4_sm70_moe.py:718-721 (_use_compact_grouped)
    """
    total_slots = num_tokens * top_k
    return 0 < total_slots <= COMPACT_MAX_SLOTS


# --------------------------------------------------------------------------
# UP ops (one indirection, so CPU tests can substitute pure-torch versions)
# --------------------------------------------------------------------------
class _UpstreamOps:
    """UP ``_moe_C`` / ``_C`` ops used by the SM70 MoE route."""

    @staticmethod
    def sort_workspace_size(rows: int, n_expert: int) -> int:
        return int(torch.ops._moe_C.moe_permute_sort_workspace_size(rows, n_expert))

    @staticmethod
    def prepare_scatter(*args) -> None:
        torch.ops._moe_C.moe_prepare_scatter(*args)

    @staticmethod
    def permute_with_scratch(*args) -> None:
        torch.ops._moe_C.moe_permute_with_scratch(*args)

    @staticmethod
    def unpermute(*args) -> None:
        torch.ops._moe_C.moe_unpermute(*args)

    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        torch.ops._C.silu_and_mul(out, x)

    @staticmethod
    def available() -> bool:
        try:
            moe = torch.ops._moe_C
            return all(
                hasattr(moe, name)
                for name in (
                    "moe_prepare_scatter",
                    "moe_permute_with_scratch",
                    "moe_unpermute",
                    "moe_permute_sort_workspace_size",
                )
            ) and hasattr(torch.ops._C, "silu_and_mul")
        except (AttributeError, RuntimeError):
            return False


OPS: Any = _UpstreamOps


def max_batched_tokens() -> int:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    sched = getattr(cfg, "scheduler_config", None)
    value = getattr(sched, "max_num_batched_tokens", None)
    return int(value) if value else DEFAULT_MAX_TOKENS


# --------------------------------------------------------------------------
# Persistent workspace
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkspaceKey:
    device: str
    experts: int
    hidden: int
    intermediate: int
    top_k: int
    max_tokens: int


class Sm70MoEWorkspace:
    """Routing + activation scratch for up to ``max_tokens`` tokens.

    ported from FORK fused_moe/sm70/fp4_workspace.py:20-97 (NvFp4MoEWorkspace.
    allocate), with two changes. It covers every M up to max_tokens (the fork
    allocated M>18 lazily in forward, :127-190, which K03 forbids), and it is
    shared by all layers with the same key instead of being allocated per layer.
    """

    _CACHE: ClassVar[dict[WorkspaceKey, Sm70MoEWorkspace]] = {}

    def __init__(self, key: WorkspaceKey) -> None:
        dev = torch.device(key.device)
        slots = key.max_tokens * key.top_k
        i32 = {"dtype": torch.int32, "device": dev}
        f16 = {"dtype": torch.float16, "device": dev}
        self.key = key
        self.permuted_input = torch.empty(slots, key.hidden, **f16)
        self.gate_up = torch.empty(slots, 2 * key.intermediate, **f16)
        self.intermediate = torch.empty(slots, key.intermediate, **f16)
        self.sorted_output = torch.empty(slots, key.hidden, **f16)
        self.input_row_indices = torch.empty(slots, **i32)
        self.expert_offsets64 = torch.empty(
            key.experts + 1, dtype=torch.int64, device=dev
        )
        self.expert_offsets = torch.empty(key.experts + 1, **i32)
        self.scatter_idx = torch.empty(slots, **i32)
        self.topk_ids = torch.empty(slots, **i32)
        self.token_expert_indices = torch.arange(slots, **i32)
        self.permuted_idx = torch.empty(slots, **i32)
        self.sorted_experts = torch.empty(slots, **i32)
        self.sorted_rows = torch.empty(slots, **i32)
        self.topk_ids_for_sort = torch.empty(slots, **i32)
        self.sort_workspace = torch.empty(
            OPS.sort_workspace_size(slots, key.experts), dtype=torch.int8, device=dev
        )
        self.dense_expert_ids = torch.arange(key.experts, **i32)
        # Compact slot groups: group g is routed slot g (one row each), so the
        # offsets are 0..slots. FORK nvfp4_sm70_moe.py:570-618 writes them per
        # call with a Triton kernel; they are constant, so they are built once.
        self.compact_offsets = torch.arange(COMPACT_MAX_SLOTS + 1, **i32)
        # Per-slot expert ids for the compact groups: the sorted ids clamped
        # to [0, experts) (see ModelOptNvFp4Sm70MoEMethod._stage_groups).
        self.compact_expert_ids = torch.zeros(COMPACT_MAX_SLOTS, **i32)

    @classmethod
    def get(cls, key: WorkspaceKey) -> Sm70MoEWorkspace:
        ws = cls._CACHE.get(key)
        if ws is None:
            ws = cls._CACHE[key] = cls(key)
        return ws

    @classmethod
    def clear(cls) -> None:
        cls._CACHE.clear()

    def views(self, num_tokens: int) -> dict[str, torch.Tensor]:
        """Slices for ``num_tokens``; no allocation (graph-safe)."""
        top_k = self.key.top_k
        slots = num_tokens * top_k
        if num_tokens > self.key.max_tokens:
            raise RuntimeError(
                f"SM70 MoE: {num_tokens} tokens exceed the persistent workspace "
                f"({self.key.max_tokens}); raise max_num_batched_tokens at load."
            )
        nk = (num_tokens, top_k)
        return {
            "permuted_input": self.permuted_input[:slots],
            "gate_up": self.gate_up[:slots],
            "intermediate": self.intermediate[:slots],
            "sorted_output": self.sorted_output[:slots],
            "input_row_indices": self.input_row_indices[:slots],
            "expert_offsets64": self.expert_offsets64,
            "expert_offsets": self.expert_offsets,
            "scatter_idx": self.scatter_idx[:slots].view(nk),
            "topk_ids": self.topk_ids[:slots].view(nk),
            "token_expert_indices": self.token_expert_indices[:slots].view(nk),
            "permuted_idx": self.permuted_idx[:slots],
            "sorted_experts": self.sorted_experts[:slots].view(nk),
            "sorted_rows": self.sorted_rows[:slots].view(nk),
            "topk_ids_for_sort": self.topk_ids_for_sort[:slots].view(nk),
            "sort_workspace": self.sort_workspace,
            "dense_expert_ids": self.dense_expert_ids,
            "compact_offsets": self.compact_offsets[
                : min(slots, COMPACT_MAX_SLOTS) + 1
            ],
            "compact_expert_ids": self.compact_expert_ids[
                : min(slots, COMPACT_MAX_SLOTS)
            ],
        }


# --------------------------------------------------------------------------
# Routing primitives
# --------------------------------------------------------------------------
def int32_topk_ids(buffers: dict[str, torch.Tensor], topk_ids: torch.Tensor):
    """UP moe_prepare_scatter requires int32 ids (moe_permute_unpermute_op.cu:73)."""
    # ported from FORK nvfp4_sm70_moe.py:1563-1564
    if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
        return topk_ids
    ids = buffers["topk_ids"]
    ids.copy_(topk_ids)
    return ids


def route_indexed(
    buffers: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    n_expert: int,
    top_k: int,
) -> None:
    """Expert offsets, scatter indices and W13 input rows; no permuted copy.

    ported from FORK nvfp4_sm70_moe.py:1566-1582 (moe_permute_metadata_with_
    scratch), rewritten onto UP moe_prepare_scatter per Kai's mapping.
    """
    OPS.prepare_scatter(
        int32_topk_ids(buffers, topk_ids),
        buffers["token_expert_indices"],
        None,  # expert_map: TP only (validated at load)
        n_expert,
        n_expert,
        buffers["expert_offsets64"],
        buffers["scatter_idx"],
        buffers["sort_workspace"],
        buffers["sorted_experts"],
        buffers["sorted_rows"],
        buffers["topk_ids_for_sort"],
    )
    # input_row_indices = floor(sorted_rows / topk) (fork
    # permute_unpermute_kernels:117), into the preallocated buffer.
    torch.floor_divide(
        buffers["sorted_rows"].view(-1), top_k, out=buffers["input_row_indices"]
    )
    buffers["expert_offsets"].copy_(buffers["expert_offsets64"])


def route_materialized(
    buffers: dict[str, torch.Tensor],
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expert: int,
    top_k: int,
) -> None:
    """Expert offsets, scatter indices and the permuted input copy.

    ported from FORK nvfp4_sm70_moe.py:1562-1565, 1583-1603.
    """
    slots = buffers["permuted_idx"].numel()
    buffers["permuted_idx"].fill_(slots)
    OPS.permute_with_scratch(
        x,
        int32_topk_ids(buffers, topk_ids),
        buffers["token_expert_indices"],
        None,
        n_expert,
        n_expert,
        top_k,
        buffers["permuted_input"],
        buffers["expert_offsets64"],
        buffers["scatter_idx"],
        buffers["permuted_idx"],
        buffers["sort_workspace"],
        buffers["sorted_experts"],
        buffers["sorted_rows"],
        buffers["topk_ids_for_sort"],
    )
    buffers["expert_offsets"].copy_(buffers["expert_offsets64"])


def unpermute(
    buffers: dict[str, torch.Tensor], topk_weights: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Weighted top-k sum of sorted_output into out [M, H].

    ported from FORK fused_moe/sm70/fp4_stages.py:99-106.
    """
    OPS.unpermute(
        buffers["sorted_output"],
        topk_weights,
        buffers["scatter_idx"],
        buffers["expert_offsets64"],
        topk_weights.shape[1],
        out,
    )
    return out


# --------------------------------------------------------------------------
# Method base
# --------------------------------------------------------------------------
class Sm70MoEMethodBase(FusedMoEMethodBase):
    """Direct-apply SM70 MoE lifecycle (R1 adapter).

    ported from FORK fused_moe/sm70/method.py:14-70, minus the env policy.
    """

    def __init__(self, moe: FusedMoEConfig) -> None:
        # UP fused_moe_method_base.py:29-33
        FusedMoEMethodBase.__init__(self, moe)
        self.validate_topology(moe)

    @staticmethod
    def validate_topology(moe: FusedMoEConfig) -> None:
        """TP 1 or 2 only: no TP4+, DP/EP/all2all or sequence parallel.

        ported from FORK nvfp4_sm70_moe.py:752-756.
        """
        if moe.tp_size not in SUPPORTED_TP_SIZES:
            raise NotImplementedError(
                f"SM70 MoE supports tensor_parallel_size in {SUPPORTED_TP_SIZES} "
                f"(validated on 2x V100); got {moe.tp_size}. TP4+ is untested."
            )
        par = moe.moe_parallel_config
        if par.use_all2all_kernels or par.use_ep or moe.dp_size > 1:
            raise NotImplementedError("SM70 MoE supports tensor parallel only.")
        if moe.is_sequence_parallel:
            raise NotImplementedError("SM70 MoE does not support sequence parallel.")

    # ---- UP FusedMoEMethodBase surface (fused_moe_method_base.py) ---------
    @property
    def supports_internal_mk(self) -> bool:  # UP :35-39
        return False

    @property
    def mk_can_overlap_shared_experts(self) -> bool:  # UP :41-47
        return False

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:  # UP :108-112
        return None  # router default: int32 ids

    @property
    def supports_eplb(self) -> bool:  # UP :126-128
        return False

    @property
    def is_monolithic(self) -> bool:  # UP :134-141
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        # UP :70-100. No rounding: TurboMind layouts use checkpoint shapes.
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None  # no modular kernel (UP routed_experts.py:244-249)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SM70 MoE methods are not monolithic.")

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError
