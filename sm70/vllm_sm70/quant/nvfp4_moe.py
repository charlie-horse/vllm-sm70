# SPDX-License-Identifier: Apache-2.0
"""M4 nvfp4-moe-turbomind: Flint's ModelOpt W4A16_NVFP4 routed experts on SM70.

What: ``ModelOptNvFp4Sm70MoEMethod``, a direct-apply MoE method (not
  monolithic, no modular kernel; see vllm_sm70/fused_moe/method.py). It
  inherits UP 4ba787e ``ModelOptNvFp4FusedMoE.create_weights``
  (modelopt.py:912-1037), so parameter names, shapes, dtypes and loaders are
  upstream's. ``process_weights_after_loading`` repacks every expert with
  ``tm.nvfp4_sm70_prepare`` and builds the strided pointer tables.
  ``apply`` runs the dense-stage route: route (UP _moe_C) -> W13 TurboMind
  dense stage -> SiLU*up (UP _C) -> W2 dense stage -> weighted unpermute
  (UP _moe_C). The output is the local, un-reduced [M, H] fp16.
  ``Sm70ModelOptNvFp4Config.get_quant_method`` (quant/configs.py) returns it
  for routed experts when ``vllm_sm70.platform.is_sm70_ready()``.
Why: on cc 7.0, UP's NVFP4 MoE backend selection has no kernel (Marlin and
  CUTLASS need SM75+/SM100). This ports FORK cf9492a
  quantization/nvfp4_sm70_moe.py:794-1731 (ModelOptNvFp4SM70MoEMethod), pruned
  to Flint's TP2 dense-stage route. Dropped from the fork: Qwen3.8 QPN /
  fused-SwiGLU / raw-scale / grouped lanes (:158-520), the GLM-5.3 lanes
  (:311-335, 685-717, 994-1037), the M=1 direct single-token slots
  (:527-568, :1550-1558; M=1 takes K12 or the compact slot groups below),
  the persistent single-launch decode (:293-310), and diagnostics
  (:1688-1716).
  M<=2 decode: at load, layers that ``vllm_sm70.ops.flint_moe.admit_layer``
  admits (Flint TP2 only: E256, H2048, local I256, top-6) get a DecodePlan
  and the device's shared split-K workspace; ``apply`` sends calls that
  ``flint_moe.can_decode`` accepts (M in {1, 2}) to the K12 two-kernel
  decode. Everything else stays on the dense-stage route.
  Compact slot groups (FORK :570-618, :718-721, :1619-1636): when the dense
  stage serves M*top_k <= 80 routed slots, both stages run with one row per
  slot (``_stage_groups``), one grouped TurboMind launch each. Above 80
  slots they use per-expert offsets, and TurboMind's grouped prefill takes
  rows > 64. Without compact groups, 18..64 slots (Flint M 3..10) fall
  into TurboMind's per-expert loop: 256 launches per stage.
Kill-switch: ``Sm70Config.flint_moe_qpn_decode`` /
  ``VLLM_SM70_FLINT_MOE_QPN_DECODE=0`` turns off the M<=2 decode fast path
  (K12 ops). The dense stage then serves every M.
  ``Sm70Config.flint_moe_compact`` / ``VLLM_SM70_FLINT_MOE_COMPACT=0`` turns
  off the compact slot groups (per-expert offsets for every M).
  ``VLLM_SM70_QUANT=0`` (M2) disables the whole route.

Ops: Kai's, only through ``vllm_sm70.ops.tm`` (K03): nvfp4_sm70_prepare,
awq_moe_build_strided_ptrs, nvfp4_moe_dense_stage_sm70_out (W13 always runs on
the permuted input: the indexed W13 op is a Qwen3.8-only contract, hidden 2560,
FORK nvfp4_sm70_moe.py:336-359, and its CUDA guard rejects Flint shapes).
The K12 decode (nvfp4_moe_decode_w13_swiglu_sm70_out,
nvfp4_moe_decode_w2_reduce_sm70_out) is used only through
``vllm_sm70.ops.flint_moe`` (admit_layer, allocate_workspace, can_decode,
flint_moe_decode_out; contract in sm70/csrc/flint_moe/PROVENANCE.md). Dtypes per the K03 contract: topk_weights float32, topk_ids
int32 (copied to int32 for moe_prepare_scatter when needed). Persistent
buffers are allocated in process_weights_after_loading, never in forward.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

from vllm_sm70.config import sm70_config
from vllm_sm70.fused_moe import method as moe_rt
from vllm_sm70.fused_moe.method import Sm70MoEMethodBase, Sm70MoEWorkspace, WorkspaceKey
from vllm_sm70.quant.nvfp4_linear import _tm, has_tm_op, unpack_fp4_weight

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger("vllm.sm70.quant.nvfp4_moe")

# K12 decode switch and the one contract it serves (flint_moe.FLINT_TP2;
# repeated here so the check works when vllm_sm70.ops.flint_moe is absent).
_DECODE_SWITCH: Final = "VLLM_SM70_FLINT_MOE_QPN_DECODE"
_FLINT_TP2_CONTRACT: Final = (256, 2048, 256, 6)
_warned_decode_ops_absent = False

NVFP4_GROUP_SIZE: Final = 16
REQUIRED_OPS: Final = (
    "nvfp4_sm70_prepare",
    "awq_moe_build_strided_ptrs",
    "nvfp4_moe_dense_stage_sm70_out",
)

# ported from FORK nvfp4_sm70_moe.py:79-86 (Flint rows only)
SUPPORTED_CONTRACTS: Final = frozenset(
    {
        # (hidden, global intermediate, experts, top-k)
        (2048, 512, 256, 6),  # Flint
        (2048, 512, 256, 8),  # Qwen3.6-35B-A3B shape (NVFP4 variant)
    }
)
SUPPORTED_TP_SIZES: Final = moe_rt.SUPPORTED_TP_SIZES  # (1, 2)
# K12 split-K workspaces, one per device, shared by every admitted layer on
# it (they run one after another on one stream). Filled only from
# process_weights_after_loading, never from apply or under graph capture.
_DECODE_WORKSPACES: dict[str, Any] = {}


def validate_contract(moe: FusedMoEConfig) -> None:
    """Reject topologies outside the validated SM70 NVFP4 contract.

    ported from FORK nvfp4_sm70_moe.py:724-757 (GLM TP8 row dropped).
    """
    local_intermediate = moe.intermediate_size_per_partition
    contract = (
        moe.hidden_dim,
        local_intermediate * max(moe.tp_size, 1),
        moe.num_experts,
        moe.experts_per_token,
    )
    if moe.tp_size not in SUPPORTED_TP_SIZES:
        raise NotImplementedError(
            f"SM70 NVFP4 MoE supports TP sizes {SUPPORTED_TP_SIZES}, got {moe.tp_size}."
        )
    if local_intermediate <= 0 or local_intermediate % NVFP4_GROUP_SIZE:
        raise NotImplementedError(
            "SM70 NVFP4 MoE needs a local intermediate size divisible by "
            f"{NVFP4_GROUP_SIZE}, got {local_intermediate}."
        )
    if contract not in SUPPORTED_CONTRACTS:
        raise NotImplementedError(
            f"SM70 NVFP4 MoE shape {contract} is not validated; "
            f"validated: {sorted(SUPPORTED_CONTRACTS)}."
        )


def _validate_weight_layout(layer: Any) -> None:
    # ported from FORK nvfp4_sm70_moe.py:758-791
    if layer.w13_weight.dtype != torch.uint8 or layer.w2_weight.dtype != torch.uint8:
        raise TypeError("SM70 NVFP4 MoE requires packed uint8 expert weights.")
    if (
        layer.w13_weight_scale.dtype != torch.float8_e4m3fn
        or layer.w2_weight_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("SM70 NVFP4 MoE requires FP8 E4M3 block scales.")


def expert_scales(
    block_scale: torch.Tensor, global_scale: torch.Tensor, split: int | None
) -> torch.Tensor:
    """One expert's [N, K/16] E4M3 block scales x FP32 global -> [K/16, N] FP16.

    ``split`` is the W13 gate/up boundary (two globals) or None (W2, one).
    ported from FORK nvfp4_sm70_moe.py:1051-1055 (W13), :1079-1080 (W2).
    """
    scales = block_scale.float().clone()
    g = global_scale.float().reshape(-1)
    if split is None:
        scales.mul_(g[0])
    else:
        scales[:split].mul_(g[0])
        scales[split:].mul_(g[-1])
    return scales.half().t().contiguous()


def decode_policy_enabled(vllm_config: Any = None) -> bool:
    """Sm70Config.flint_moe_qpn_decode after VLLM_SM70_FLINT_MOE_QPN_DECODE=0."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    return bool(sm70_config(vllm_config).flint_moe_qpn_decode)


def compact_policy_enabled(vllm_config: Any = None) -> bool:
    """Sm70Config.flint_moe_compact after VLLM_SM70_FLINT_MOE_COMPACT=0."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    return bool(sm70_config(vllm_config).flint_moe_compact)


def _flint_moe():
    """Kai's K12 wrappers (vllm_sm70.ops.flint_moe), or None if not importable."""
    try:
        from vllm_sm70.ops import flint_moe  # type: ignore[attr-defined]
    except ImportError:
        return None
    return flint_moe


def decode_workspace(flint_moe: Any, device: torch.device) -> Any:
    """The device's shared K12 workspace; allocates it on first use (load only).

    flint_moe.allocate_workspace refuses inside CUDA graph capture.
    """
    key = str(device)
    ws = _DECODE_WORKSPACES.get(key)
    if ws is None:
        ws = _DECODE_WORKSPACES[key] = flint_moe.allocate_workspace(device)
    return ws


def clear_decode_workspaces() -> None:
    _DECODE_WORKSPACES.clear()


class ModelOptNvFp4Sm70MoEMethod(Sm70MoEMethodBase, ModelOptNvFp4FusedMoE):
    """ModelOpt NVFP4 experts with FP16 activations on SM70 (TurboMind).

    MRO: Sm70MoEMethodBase supplies the runner-facing surface (not
    monolithic, no modular kernel, no EPLB). ModelOptNvFp4FusedMoE supplies
    create_weights and uses_weight_scale_2_pattern (UP modelopt.py:892).
    """

    # UP ModelOptNvFp4FusedMoE.__init__ signature (modelopt.py:850-854)
    def __init__(self, quant_config: Any, moe_config: FusedMoEConfig) -> None:
        # ported from FORK nvfp4_sm70_moe.py:797-813. UP's __init__ is skipped
        # on purpose: it selects an NVFP4 backend, and none exists at cc 7.0.
        Sm70MoEMethodBase.__init__(self, moe_config)
        if quant_config.quant_method not in {"NVFP4", "W4A16_NVFP4"}:
            raise NotImplementedError(
                "SM70 NVFP4 MoE needs NVFP4-family checkpoint weights, "
                f"got {quant_config.quant_method}."
            )
        self.quant_config = quant_config
        self.use_a16 = True
        self.use_global_sf = False  # read by UP create_weights (modelopt.py:1009)
        validate_contract(moe_config)
        self.decode_enabled = decode_policy_enabled()
        self.compact_enabled = compact_policy_enabled()
        logger.info_once(
            "SM70 NVFP4 MoE: compact slot groups %s for dense-stage calls with "
            "M*top_k <= %d (VLLM_SM70_FLINT_MOE_COMPACT=0 disables them).",
            "enabled" if self.compact_enabled else "DISABLED",
            moe_rt.COMPACT_MAX_SLOTS,
        )

    # create_weights: inherited from UP ModelOptNvFp4FusedMoE (modelopt.py:912).

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Repack experts for TurboMind and allocate persistent scratch.

        ported from FORK nvfp4_sm70_moe.py:828-1364 (dense-stage subset):
        checks :834-916, per-expert prepare :1045-1093, pointer tables
        :1150-1163 and :1203-1206, layer dims :1208-1219, decode admission
        :1268-1275, workspace :1288-1296, raw-weight release :1298-1309.
        """
        missing = [name for name in REQUIRED_OPS if not has_tm_op(name)]
        if missing:
            raise RuntimeError(
                "SM70 NVFP4 MoE needs vllm_sm70.ops.tm with " + ", ".join(missing)
            )
        if not moe_rt.OPS.available():
            raise RuntimeError(
                "SM70 NVFP4 MoE needs UP _moe_C (moe_prepare_scatter, "
                "moe_permute_with_scratch, moe_unpermute) and _C.silu_and_mul."
            )
        if self.moe.has_bias:
            raise NotImplementedError("SM70 NVFP4 MoE does not support expert bias.")
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if layer.activation != MoEActivation.SILU or layer.swiglu_limit is not None:
            raise NotImplementedError("SM70 NVFP4 MoE supports unclamped SwiGLU only.")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 NVFP4 MoE does not support router weights on input."
            )
        if layer.expert_map is not None or (
            layer.local_num_experts != layer.global_num_experts
        ):
            raise NotImplementedError("SM70 NVFP4 MoE requires replicated experts.")
        validate_contract(layer.moe_config)
        _validate_weight_layout(layer)

        tm = _tm()
        num_experts = int(layer.local_num_experts)
        hidden = int(layer.moe_config.hidden_dim)
        intermediate = int(layer.moe_config.intermediate_size_per_partition)
        top_k = int(layer.moe_config.experts_per_token)

        banks: dict[str, tuple[list, list, list]] = {
            "w13": ([], [], []),
            "w2": ([], [], []),
        }
        for e in range(num_experts):
            for stage, split in (("w13", intermediate), ("w2", None)):
                packed = unpack_fp4_weight(getattr(layer, stage + "_weight").data[e])
                scales = expert_scales(
                    getattr(layer, stage + "_weight_scale").data[e],
                    getattr(layer, stage + "_weight_scale_2").data[e],
                    split,
                )
                w, s, meta = tm.nvfp4_sm70_prepare(
                    packed, scales, NVFP4_GROUP_SIZE, False
                )
                for dst, src in zip(banks[stage], (w, s, meta)):
                    dst.append(src)

        for stage in ("w13", "w2"):
            weights, scales, metas = banks[stage]
            tm_w = torch.stack(weights)
            tm_s = torch.stack(scales)
            k_ld, q_ld = (int(v) for v in metas[0].tolist())
            ptrs_w, ptrs_s = tm.awq_moe_build_strided_ptrs(
                tm_w, tm_s, k_ld, q_ld, num_experts
            )
            setattr(layer, stage + "_tm_weight", Parameter(tm_w, requires_grad=False))
            setattr(layer, stage + "_tm_scales", Parameter(tm_s, requires_grad=False))
            setattr(
                layer, stage + "_strided_ptrs_w", Parameter(ptrs_w, requires_grad=False)
            )
            setattr(
                layer, stage + "_strided_ptrs_s", Parameter(ptrs_s, requires_grad=False)
            )

        layer.sm70_nvfp4_num_experts = num_experts
        layer.sm70_nvfp4_hidden_size = hidden
        layer.sm70_nvfp4_intermediate_size = intermediate
        layer.sm70_nvfp4_top_k = top_k

        self._prepare_decode(layer, (num_experts, hidden, intermediate, top_k))
        layer.sm70_moe_workspace = Sm70MoEWorkspace.get(
            WorkspaceKey(
                device=str(layer.w13_tm_weight.device),
                experts=num_experts,
                hidden=hidden,
                intermediate=intermediate,
                top_k=top_k,
                max_tokens=moe_rt.max_batched_tokens(),
            )
        )

        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_2",
            "w13_input_scale",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_2",
            "w2_input_scale",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        if layer.w13_tm_weight.is_cuda:
            torch.cuda.empty_cache()

    def _prepare_decode(self, layer: Any, contract: tuple[int, int, int, int]) -> None:
        """Admit the K12 M<=2 decode (flint_moe.admit_layer) after the banks
        are stacked, and give admitted layers the device's shared workspace.

        ``contract`` is (experts, hidden, local intermediate, top_k).
        admit_layer owns the contract, op presence, bank shape/dtype and
        scale pre-shift checks; with an explicit policy it does not read the
        env, so the kill switch is applied here too (it can only disable).
        """
        layer.sm70_nvfp4_decode_plan = None
        layer.sm70_nvfp4_decode_ws = None
        flint_moe = _flint_moe()
        ops_present = flint_moe is not None and flint_moe.available()
        if (
            not ops_present
            and self.decode_enabled
            and contract == _FLINT_TP2_CONTRACT
        ):
            # fork nvfp4_sm70_moe.py:240-251: an explicit request for the
            # decode with the ops missing is an error, not a silent fallback.
            # "1" never enables anything here; it only makes absence fatal.
            if os.environ.get(_DECODE_SWITCH) == "1":
                raise RuntimeError(
                    f"{_DECODE_SWITCH}=1 requires vllm_sm70._sm70_C built with "
                    "the K12 ops nvfp4_moe_decode_w13_swiglu_sm70_out and "
                    "nvfp4_moe_decode_w2_reduce_sm70_out."
                )
            global _warned_decode_ops_absent
            if not _warned_decode_ops_absent:
                _warned_decode_ops_absent = True
                logger.warning(
                    "The SM70 NVFP4 M<=2 decode ops are absent from the loaded "
                    "_sm70_C; keeping the dense-stage route for M<=2."
                )
        if flint_moe is None:
            return
        plan = flint_moe.admit_layer(
            contract,
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            policy_enabled=bool(self.decode_enabled and flint_moe.enabled()),
        )
        if plan is None:
            # Per layer (the layer name is part of the message key), so a
            # single rejected layer is visible among the admitted ones.
            logger.info_once(
                "SM70 NVFP4 MoE %s: K12 M<=2 decode not admitted for contract "
                "%s (policy %s, ops %s); dense-stage route serves M<=2.",
                getattr(layer, "layer_name", "?"), contract,
                bool(self.decode_enabled), ops_present,
            )
            return
        logger.info_once(
            "SM70 NVFP4 MoE: K12 M<=2 decode admitted for contract %s "
            "(VLLM_SM70_FLINT_MOE_QPN_DECODE=0 disables it).", contract,
        )
        layer.sm70_nvfp4_decode_plan = plan
        layer.sm70_nvfp4_decode_ws = decode_workspace(
            flint_moe, layer.w13_tm_weight.device
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Local (un-reduced) routed output [M, H] fp16; the runner all-reduces.

        ported from FORK nvfp4_sm70_moe.py:1375-1406 (checks), :1442-1450
        (decode), :1555-1563/1583-1607/1652-1686 (dense route) and
        fused_moe/sm70/fp4_stages.py:40-106 + fp4_codec.py:255-270 (stages).
        """
        del shared_experts, shared_experts_input  # run by the UP runner
        hidden = int(layer.sm70_nvfp4_hidden_size)
        inter = int(layer.sm70_nvfp4_intermediate_size)
        experts = int(layer.sm70_nvfp4_num_experts)
        top_k = int(layer.sm70_nvfp4_top_k)
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != hidden:
            raise TypeError(f"SM70 NVFP4 MoE needs FP16 [M, {hidden}] activations.")
        if tuple(topk_ids.shape) != (x.shape[0], top_k) or tuple(
            topk_weights.shape
        ) != tuple(topk_ids.shape):
            raise ValueError("SM70 NVFP4 MoE top-k shape mismatch.")
        if topk_weights.dtype != torch.float32:
            raise TypeError("SM70 NVFP4 MoE requires float32 top-k weights.")
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, hidden))
        x = x.contiguous()
        ws: Sm70MoEWorkspace = layer.sm70_moe_workspace
        buffers = ws.views(num_tokens)

        plan = getattr(layer, "sm70_nvfp4_decode_plan", None)
        if plan is not None:
            flint_moe = _flint_moe()
            if flint_moe.can_decode(x, topk_weights, topk_ids):
                # K12: int32 or int64 IDs; out is fully overwritten; the result
                # is the local un-reduced sum, like the dense route.
                out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=x.device)
                return flint_moe.flint_moe_decode_out(
                    out,
                    buffers["intermediate"],
                    x,
                    layer.w13_tm_weight,
                    layer.w13_tm_scales,
                    layer.w2_tm_weight,
                    layer.w2_tm_scales,
                    topk_weights,
                    topk_ids,
                    layer.sm70_nvfp4_decode_ws,
                    plan,
                )

            if num_tokens <= 2:
                # Admitted at load but refused for this decode call: name the
                # layer and the C++ guard's reason (once per layer and reason).
                logger.info_once(
                    "SM70 NVFP4 MoE %s: K12 decode skipped for an M=%d call "
                    "(%s); dense-stage route used.",
                    getattr(layer, "layer_name", "?"), num_tokens,
                    flint_moe.call_rejection(x, topk_weights, topk_ids),
                )

        tm = _tm()
        output = torch.zeros((num_tokens, hidden), dtype=x.dtype, device=x.device)
        moe_rt.route_materialized(buffers, x, topk_ids, experts, top_k)
        stage_offsets, stage_expert_ids, stage_experts = self._stage_groups(
            buffers, num_tokens, top_k, experts
        )
        tm.nvfp4_moe_dense_stage_sm70_out(
            buffers["gate_up"],
            buffers["permuted_input"],
            stage_offsets,
            stage_expert_ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            stage_experts,
            hidden,
            2 * inter,
            NVFP4_GROUP_SIZE,
        )
        moe_rt.OPS.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        tm.nvfp4_moe_dense_stage_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            stage_offsets,
            stage_expert_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            stage_experts,
            inter,
            hidden,
            NVFP4_GROUP_SIZE,
        )
        # Unpermute always uses the per-expert offsets (FORK :1683-1685).
        return moe_rt.unpermute(buffers, topk_weights, output)

    def _stage_groups(
        self,
        buffers: dict[str, torch.Tensor],
        num_tokens: int,
        top_k: int,
        experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """(offsets, expert ids, group count) for both dense stages.

        ported from FORK nvfp4_sm70_moe.py:1619-1640. With M*top_k <= 80 each
        routed slot is its own one-row group (offsets 0..slots, ids = the
        sorted expert id of each permuted row), which TurboMind's compact
        grouped dispatch serves in one launch per stage (tm_moe_sm70.cu:
        input rows == group count <= 80). Duplicate expert ids stay separate
        groups (FORK :606-609). Otherwise: per-expert offsets over all experts.
        FORK _prepare_compact_slot_groups copies the sorted ids into an
        active_expert_ids buffer with a Triton kernel and rewrites the
        constant offsets each call. Here one torch.clamp writes the sorted
        ids, clamped to [0, experts), into the persistent compact_expert_ids
        buffer (the clamp handles padding routes, see below), and the offsets
        are a persistent arange.
        """
        if self.compact_enabled and moe_rt.use_compact_grouped(num_tokens, top_k):
            slots = num_tokens * top_k
            # Padding routes: UP routers set padding tokens' ids to -1
            # (VLLM_MOE_SKIP_PADDING=1, router/fused_topk_router.py), and
            # every token is padding in CUDA-graph capture dummy runs
            # (gpu_model_runner.py:5753). moe_permute maps them to the
            # sentinel id `experts` (moe_permute_unpermute_op.cu:108-117).
            # Per-expert offsets skip those rows, but a compact group would
            # index the pointer table at `experts`, one past the end: the
            # illegal memory access at capture on c4130 (#97 A/B). Clamp the
            # group ids into a persistent buffer so those rows run on a real
            # expert. Their output is discarded: padding tokens have weight 0
            # and nothing reads their rows. The fork predates padding skip.
            ids = buffers["compact_expert_ids"][:slots]
            sorted_ids = buffers["sorted_experts"].reshape(slots)
            torch.clamp(sorted_ids, 0, experts - 1, out=ids)
            return buffers["compact_offsets"][: slots + 1], ids, slots
        return buffers["expert_offsets"], buffers["dense_expert_ids"], experts


def build_moe_method(quant_config: Any, moe_config: FusedMoEConfig):
    return ModelOptNvFp4Sm70MoEMethod(quant_config, moe_config)
