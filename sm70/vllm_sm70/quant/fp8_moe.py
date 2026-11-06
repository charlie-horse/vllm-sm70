# SPDX-License-Identifier: Apache-2.0
"""M6 qwen-fp8-turbomind (MoE): Qwen3.6-35B-A3B block-FP8 routed experts on SM70.

What: ``Fp8Sm70MoEMethod``, a direct-apply MoE method on M4's shared base
  (vllm_sm70/fused_moe/method.py): not monolithic, no modular kernel. It
  creates UP 4ba787e ``Fp8MoEMethod``'s block-quant parameters (fp8.py
  :530-668: w13/w2 e4m3, w13/w2_weight_scale_inv fp32 [E, N/128, K/128]).
  ``process_weights_after_loading`` repacks every expert with
  ``tm.fp8_sm70_prepare`` and builds the strided pointer tables.
  ``apply`` has three routes:
  - batched (any M): UP moe_permute_with_scratch, then TurboMind
    ``fp8_moe_gemm_sm70_out`` for W13, UP silu_and_mul, the W2 batched GEMM,
    and UP moe_unpermute;
  - compact decode (M == 1, the default): the fork's legacy compact route,
    one fused ``fp8_moe_single_token_sm70_out`` launch with exact_layout
    (FORK fp8_sm70_moe.py:547-554, fork defaults envs.py:8858/9138 and
    sm70_moe.py:344-347, frozen in M1's CONSTANTS);
  - decomposed decode (M == 1, compact killed): the active-expert W13 op
    (``fp8_moe_single_token_dense_w13_sm70_out``, which builds its own slot
    order), SiLU*up, the indexed W2 stage and moe_unpermute.
  The output is the local, un-reduced [M, H] fp16; the runner all-reduces.
  ``Sm70Fp8Config.get_quant_method`` (quant/configs.py) returns it for routed
  experts when ``is_sm70_ready()``.
Why: UP's FP8 MoE oracle has no cc 7.0 backend. Ported from FORK cf9492a
  quantization/fp8_sm70_moe.py:100-649 (Fp8SM70MoEMethod) with
  fused_moe/sm70/stages.py:18-196 and single_token.py:19-151. Dropped: the
  compact-compare diagnostics (:348-398), the compact route's decomposed
  and top-k-descriptor variants (:420-444; CONSTANTS FP8_MOE_COMPACT_
  DECOMPOSED is False and exact layout is True), the per-expert-dispatch and chunked W2 variants, the
  non-block (per-tensor) checkpoints, and the dequant-to-FP16 fallback.
Kill-switch: ``VLLM_SM70_FP8_MOE_DECODE=0`` (``Sm70Config.fp8_moe_decode``)
  turns off both single-token routes; the batched route then serves M == 1.
  ``VLLM_SM70_FP8_MOE_COMPACT_DECODE=0`` (``Sm70Config.fp8_moe_compact_
  decode``) swaps the compact op for the decomposed plan.
  ``VLLM_SM70_FP8_TURBOMIND=0`` refuses FP8 on SM70 altogether (M2 gate),
  and get_quant_method then never builds this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    validate_fp8_block_shape_moe,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_sm70.config import CONSTANTS, sm70_config
from vllm_sm70.fused_moe import method as moe_rt
from vllm_sm70.fused_moe.method import (
    Sm70MoEMethodBase,
    Sm70MoEWorkspace,
    WorkspaceKey,
)
from vllm_sm70.quant.fp8_linear import FP8_BLOCK, check_fp8_sm70_config
from vllm_sm70.quant.nvfp4_linear import _tm, has_tm_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger("vllm.sm70.quant.fp8_moe")

REQUIRED_OPS: Final = (
    "fp8_sm70_prepare",
    "awq_moe_build_strided_ptrs",
    "fp8_moe_gemm_sm70_out",
)
DECODE_W13_OP: Final = "fp8_moe_single_token_dense_w13_sm70_out"
DECODE_W2_OP: Final = "fp8_moe_single_token_indexed_dense_stage_sm70_out"
COMPACT_OP: Final = "fp8_moe_single_token_sm70_out"
# Frozen fork values (M1 CONSTANTS; FORK config/sm70_moe.py:344-349).
COMPACT_EXACT_LAYOUT: Final = bool(CONSTANTS["FP8_MOE_COMPACT_EXACT_LAYOUT"])
COMPACT_NATIVE_UNPERMUTE: Final = bool(CONSTANTS["FP8_MOE_COMPACT_NATIVE_UNPERMUTE"])
if not CONSTANTS["FP8_MOE_LEGACY_COMPACT"] or CONSTANTS["FP8_MOE_COMPACT_DECOMPOSED"]:
    raise AssertionError("M6 ports only the legacy compact op (not decomposed).")


def decode_policy_enabled(vllm_config: Any = None) -> bool:
    """Sm70Config.fp8_moe_decode after VLLM_SM70_FP8_MOE_DECODE=0."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    return bool(sm70_config(vllm_config).fp8_moe_decode)


def compact_policy_enabled(vllm_config: Any = None) -> bool:
    """Sm70Config.fp8_moe_compact_decode after VLLM_SM70_FP8_MOE_COMPACT_DECODE=0."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    return bool(sm70_config(vllm_config).fp8_moe_compact_decode)


class Fp8Sm70MoEMethod(Sm70MoEMethodBase):
    """TurboMind W8A16 block-FP8 experts (FP16 activations)."""

    # UP Fp8MoEMethod.__init__ signature (fp8.py:480)
    def __init__(self, quant_config: Any, layer: RoutedExperts) -> None:
        # ported from FORK fp8_sm70_moe.py:108-125
        check_fp8_sm70_config(quant_config)
        super().__init__(layer.moe_config)
        if self.moe.has_bias:
            raise NotImplementedError("SM70 FP8 MoE does not support expert bias.")
        if self.moe.intermediate_size_per_partition % FP8_BLOCK:
            raise NotImplementedError(
                "SM70 FP8 MoE needs a local intermediate size divisible by 128, "
                f"got {self.moe.intermediate_size_per_partition}."
            )
        self.quant_config = quant_config
        self.weight_block_size = list(quant_config.weight_block_size)
        self.block_quant = True
        self.weight_scale_name = "weight_scale_inv"
        self.decode_enabled = decode_policy_enabled()
        self.compact_enabled = compact_policy_enabled()

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # UP fp8.py:530-668, block branch (== FORK fp8_sm70_moe.py:147-214).
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size
        block_n, block_k = self.weight_block_size
        validate_fp8_block_shape_moe(
            intermediate_size_per_partition, self.weight_block_size
        )
        fp8 = torch.float8_e4m3fn
        shards = self.moe.w13_num_shards
        w13 = Parameter(
            torch.empty(
                num_experts,
                shards * intermediate_size_per_partition,
                hidden_size,
                dtype=fp8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra_weight_attrs)
        w2 = Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size_per_partition, dtype=fp8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra_weight_attrs)
        w13_s = Parameter(
            torch.ones(
                num_experts,
                shards * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_s = Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_s)
        layer.register_parameter("w2_weight_scale_inv", w2_s)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_s, extra_weight_attrs)
        set_weight_attrs(w2_s, extra_weight_attrs)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Repack experts for TurboMind and bind the persistent workspace.

        ported from FORK fp8_sm70_moe.py:216-311 (block branch), with
        fused_moe/sm70/weight_codec.py:33-60 (per-expert prepare) and
        workspace.py (persistent buffers, now M4's shared workspace).
        """
        missing = [name for name in REQUIRED_OPS if not has_tm_op(name)]
        if missing:
            raise RuntimeError(
                "SM70 FP8 MoE needs vllm_sm70.ops.tm with " + ", ".join(missing)
            )
        if not moe_rt.OPS.available():
            raise RuntimeError("SM70 FP8 MoE needs UP _moe_C and _C.silu_and_mul.")
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if layer.activation != MoEActivation.SILU or layer.swiglu_limit is not None:
            raise NotImplementedError("SM70 FP8 MoE supports unclamped SwiGLU only.")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 FP8 MoE does not support router weights on input."
            )
        if layer.expert_map is not None or (
            layer.local_num_experts != layer.global_num_experts
        ):
            raise NotImplementedError("SM70 FP8 MoE requires replicated experts.")
        if layer.orig_dtype != torch.float16:
            raise RuntimeError("SM70 FP8 MoE needs --dtype half.")

        tm = _tm()
        num_experts = int(layer.w13_weight.shape[0])
        hidden = int(layer.w2_weight.shape[1])
        intermediate = int(layer.w2_weight.shape[2])
        top_k = int(layer.moe_config.experts_per_token)
        for stage in ("w13", "w2"):
            weight = getattr(layer, stage + "_weight").data
            scale = getattr(layer, stage + "_weight_scale_inv").data.float()
            prepared = [
                tm.fp8_sm70_prepare(
                    weight[e].contiguous(), scale[e].contiguous(), FP8_BLOCK, False
                )
                for e in range(num_experts)
            ]
            tm_w = torch.stack([p[0] for p in prepared])
            tm_s = torch.stack([p[1] for p in prepared])
            k_ld, q_ld = (int(v) for v in prepared[0][2].tolist())
            ptrs_w, ptrs_s = tm.awq_moe_build_strided_ptrs(
                tm_w, tm_s, k_ld, q_ld, num_experts
            )
            for name, value in (
                ("_tm_weight", tm_w),
                ("_tm_scales", tm_s),
                ("_strided_ptrs_w", ptrs_w),
                ("_strided_ptrs_s", ptrs_s),
            ):
                setattr(layer, stage + name, Parameter(value, requires_grad=False))

        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_size = hidden
        layer.sm70_intermediate_size = intermediate
        layer.sm70_top_k = top_k
        # Precedence: fp8_moe_decode off -> batched only; else compact (if
        # its switch and op are on), else the decomposed plan (if its ops are).
        layer.sm70_fp8_moe_compact = bool(
            self.decode_enabled and self.compact_enabled and has_tm_op(COMPACT_OP)
        )
        layer.sm70_fp8_moe_decode = bool(
            self.decode_enabled and has_tm_op(DECODE_W13_OP) and has_tm_op(DECODE_W2_OP)
        )
        if layer.sm70_fp8_moe_compact:
            self._allocate_compact_buffers(layer, top_k)
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
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        ):
            delattr(layer, name)
        if layer.w13_tm_weight.is_cuda:
            torch.cuda.empty_cache()

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Local (un-reduced) routed output [M, H] fp16.

        ported from FORK fp8_sm70_moe.py:521-647 (minus legacy compact and
        compare), stages.py:107-196 (batched) and single_token.py:19-151.
        """
        del shared_experts, shared_experts_input  # run by the UP runner
        hidden = int(layer.sm70_hidden_size)
        inter = int(layer.sm70_intermediate_size)
        experts = int(layer.sm70_num_experts)
        top_k = int(layer.sm70_top_k)
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != hidden:
            raise TypeError(f"SM70 FP8 MoE needs FP16 [M, {hidden}] activations.")
        if tuple(topk_ids.shape) != (x.shape[0], top_k) or tuple(
            topk_weights.shape
        ) != tuple(topk_ids.shape):
            raise ValueError("SM70 FP8 MoE top-k shape mismatch.")
        if topk_weights.dtype != torch.float32:
            raise TypeError("SM70 FP8 MoE requires float32 top-k weights.")
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, hidden))
        x = x.contiguous()
        tm = _tm()
        buffers = layer.sm70_moe_workspace.views(num_tokens)
        output = torch.zeros((num_tokens, hidden), dtype=x.dtype, device=x.device)

        if num_tokens == 1 and layer.sm70_fp8_moe_compact:
            return self._apply_compact(
                layer, x, topk_weights, topk_ids, buffers, output
            )
        if num_tokens == 1 and layer.sm70_fp8_moe_decode:
            return self._apply_single_token(
                layer, x, topk_weights, topk_ids, buffers, output
            )

        moe_rt.route_materialized(buffers, x, topk_ids, experts, top_k)
        tm.fp8_moe_gemm_sm70_out(
            buffers["gate_up"],
            buffers["permuted_input"],
            buffers["expert_offsets"],
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            experts,
            hidden,
            2 * inter,
            FP8_BLOCK,
            False,
        )
        moe_rt.OPS.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        tm.fp8_moe_gemm_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            buffers["expert_offsets"],
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            experts,
            inter,
            hidden,
            FP8_BLOCK,
            False,
        )
        return moe_rt.unpermute(buffers, topk_weights, output)

    @staticmethod
    def _allocate_compact_buffers(layer: Any, top_k: int) -> None:
        """Per-layer scratch of the compact op (top_k rows of pointer descriptors).

        ported from FORK fused_moe/sm70/workspace.py:75-126 (single-token part)
        and fp8_sm70_moe.py:274-289 (pointer rows).
        """
        dev = layer.w13_tm_weight.device
        row = int(layer.w13_strided_ptrs_w.numel() // layer.sm70_num_experts)
        layer.sm70_fp8_ptr_rows = {
            name: getattr(layer, name).view(layer.sm70_num_experts, row)
            for name in (
                "w13_strided_ptrs_w",
                "w13_strided_ptrs_s",
                "w2_strided_ptrs_w",
                "w2_strided_ptrs_s",
            )
        }
        ptr_dtype = layer.w13_strided_ptrs_w.dtype
        layer.sm70_fp8_compact = {
            "sorted_weights": torch.empty(top_k, dtype=torch.float32, device=dev),
            "broadcast_input_indices": torch.empty(
                top_k, dtype=torch.int32, device=dev
            ),
            **{
                f"legacy_{s}_ptrs_{t}": torch.empty(
                    top_k, row, dtype=ptr_dtype, device=dev
                )
                for s in ("w13", "w2")
                for t in ("w", "s")
            },
            "empty_weight": torch.empty(0, dtype=torch.float8_e4m3fn, device=dev),
            "empty_scale": torch.empty(0, dtype=torch.float32, device=dev),
        }

    @staticmethod
    def _apply_compact(layer, x, topk_weights, topk_ids, buffers, output):
        """M == 1 default: one fused legacy compact launch, exact layout.

        ported from FORK fp8_sm70_moe.py:399-410 and :447-490 (exact layout,
        no fused SiLU/reduce/broadcast/direct-reduce/indexed pointers), with
        the optional native unpermute at :491-500 (off per CONSTANTS).
        """
        top_k = int(layer.sm70_top_k)
        rows = layer.sm70_fp8_ptr_rows
        c = layer.sm70_fp8_compact
        ids = moe_rt.int32_topk_ids(buffers, topk_ids)
        _tm().fp8_moe_single_token_sm70_out(
            output,
            x,
            topk_weights,
            ids,
            rows["w13_strided_ptrs_w"],
            rows["w13_strided_ptrs_s"],
            rows["w2_strided_ptrs_w"],
            rows["w2_strided_ptrs_s"],
            buffers["permuted_input"],
            buffers["gate_up"],
            buffers["intermediate"],
            buffers["sorted_output"],
            c["sorted_weights"],
            c["legacy_w13_ptrs_w"],
            c["legacy_w13_ptrs_s"],
            c["legacy_w2_ptrs_w"],
            c["legacy_w2_ptrs_s"],
            buffers["expert_offsets"],
            buffers["scatter_idx"],
            buffers["sorted_experts"].view(-1)[:top_k],
            c["broadcast_input_indices"],
            c["empty_weight"],
            c["empty_scale"],
            int(layer.sm70_hidden_size),
            2 * int(layer.sm70_intermediate_size),
            int(layer.sm70_intermediate_size),
            int(layer.sm70_hidden_size),
            FP8_BLOCK,
            int(layer.sm70_hidden_size),
            False,  # fused_gated_silu
            False,  # fused_weighted_reduce
            False,  # broadcast_input
            False,  # w2_direct_reduce
            False,  # indexed_expert_ptrs
            COMPACT_EXACT_LAYOUT,  # exact_per_route
        )
        if COMPACT_NATIVE_UNPERMUTE:
            output.zero_()
            moe_rt.OPS.unpermute(
                buffers["sorted_output"],
                topk_weights,
                buffers["scatter_idx"],
                None,
                top_k,
                output,
            )
        return output

    @staticmethod
    def _apply_single_token(layer, x, topk_weights, topk_ids, buffers, output):
        """M == 1 decomposed plan: active-expert W13 (builds its own slot
        order), SiLU*up, indexed W2, moe_unpermute.

        ported from FORK single_token.py:19-151 with plan w13="active_dense",
        w2="indexed". The fork's effective single-token reduce was the native
        weighted reduce (M1 CONSTANTS MOE_SINGLE_TOKEN_REDUCE="weighted",
        awq_moe_single_token_weighted_reduce_out); this fallback uses UP
        moe_unpermute for the same weighted sum. It only runs with
        VLLM_SM70_FP8_MOE_COMPACT_DECODE=0; the compact op is the default.
        """
        tm = _tm()
        top_k = int(layer.sm70_top_k)
        hidden = int(layer.sm70_hidden_size)
        inter = int(layer.sm70_intermediate_size)
        ids = moe_rt.int32_topk_ids(buffers, topk_ids)
        sorted_ids = buffers["sorted_experts"].view(-1)[:top_k]
        getattr(tm, DECODE_W13_OP)(
            buffers["gate_up"],
            buffers["permuted_input"],
            x,
            ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            buffers["expert_offsets"],
            buffers["expert_offsets64"],
            buffers["scatter_idx"],
            sorted_ids,
            hidden,
            2 * inter,
            FP8_BLOCK,
            hidden,
        )
        moe_rt.OPS.silu_and_mul(buffers["intermediate"], buffers["gate_up"])
        getattr(tm, DECODE_W2_OP)(
            buffers["sorted_output"],
            buffers["intermediate"],
            buffers["expert_offsets"],
            sorted_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            top_k,
            inter,
            hidden,
            FP8_BLOCK,
        )
        moe_rt.OPS.unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["scatter_idx"],
            buffers["expert_offsets64"][: top_k + 1],
            top_k,
            output,
        )
        return output
