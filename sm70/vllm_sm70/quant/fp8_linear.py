# SPDX-License-Identifier: Apache-2.0
"""M6 qwen-fp8-turbomind (dense): Qwen block-128 FP8 linear on SM70.

What: ``Fp8Sm70LinearMethod``, a subclass of UP 4ba787e ``Fp8LinearMethod``
  for serialized e4m3 checkpoints with weight_block_size [128, 128] and
  dynamic activations (Qwen3.6-35B-A3B-FP8, Qwen3.8-27B-FP8). It creates the
  same parameters as UP (weight [N, K] e4m3, weight_scale_inv [N/128, K/128]
  fp32 BlockQuantScaleParameter; fp8.py:282-343) without UP's kernel
  selection, which has no cc 7.0 kernel. After loading it repacks with
  ``tm.fp8_sm70_prepare`` and runs ``tm.fp8_gemm_sm70_out``: weight-only FP8
  (W8A16), FP16 activations, no activation quantization.
  ``Sm70Fp8Config.get_quant_method`` (quant/configs.py) returns it when
  ``is_sm70_ready()`` and the TurboMind FP8 switch is on.
Why: V100 has no FP8 tensor cores and UP refuses FP8 below cc 7.5. Ported
  from FORK cf9492a kernels/linear/qpn/fp8.py:448-1260
  (TurboMindFp8LinearKernel), keeping only the core TurboMind route
  (prepare :828-852, plain GEMM :1045-1052 and :1121-1130). Dropped: QPN8 (M8), the
  PP2/TP4 and grouped-BMM (DeepSeek) variants, prescaled M1/8K, exact-dense
  prefill, gated-SiLU, and the dequant-to-FP16 fallback.
Kill-switch: ``VLLM_SM70_FP8_TURBOMIND=0``. M2's gate then returns
  upstream's capability 75, so UP refuses FP8 on SM70, and get_quant_method
  never builds this class.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_weight_block_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter
from vllm.model_executor.utils import replace_parameter

from vllm_sm70.quant.nvfp4_linear import (
    _tm,
    flatten_linear_input,
    has_tm_op,
    restore_linear_output,
)

logger = logging.getLogger(__name__)

FP8_BLOCK = 128
PREPARE_OP = "fp8_sm70_prepare"
GEMM_OP = "fp8_gemm_sm70_out"


def check_fp8_sm70_config(quant_config: Any) -> None:
    """Serialized e4m3, block [128, 128], dynamic activations, no UE8M0 scales.

    ported from FORK qpn/fp8.py:464-475 (can_implement) and fp8.py:367-370
    (use_sm70_fp8_turbomind).
    """
    if list(quant_config.weight_block_size or ()) != [FP8_BLOCK, FP8_BLOCK]:
        raise NotImplementedError(
            "SM70 FP8 supports weight_block_size [128, 128] only, got "
            f"{quant_config.weight_block_size}."
        )
    if quant_config.activation_scheme != "dynamic":
        raise NotImplementedError("SM70 FP8 is weight-only; static activation scales.")
    if getattr(quant_config, "is_scale_e8m0", False):
        raise NotImplementedError("SM70 FP8 does not support UE8M0 block scales.")


@register_weight_loader_v2_supported_method
class Fp8Sm70LinearMethod(Fp8LinearMethod):
    """TurboMind W8A16 block-FP8 linear (FP16 activations)."""

    def __init__(self, quant_config: Any) -> None:
        check_fp8_sm70_config(quant_config)
        super().__init__(quant_config)  # UP fp8.py:241-280 (keys only)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # UP fp8.py:282-343 (block branch) without init_fp8_linear_kernel
        # (:347), which has no kernel for cc 7.0.
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size
        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)
        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            self.weight_block_size,
            weight_loader,
            scale_dtype=None,
        )
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack [N, K] e4m3 + [N/128, K/128] scales into TurboMind operands.

        ported from FORK qpn/fp8.py:492-517 (checks, block normalization) and
        :828-852 (prepare, replace, meta).
        """
        if layer.orig_dtype != torch.float16:
            raise RuntimeError(
                "SM70 TurboMind FP8 needs --dtype half (FP16 activations), "
                f"got {layer.orig_dtype}."
            )
        if not (has_tm_op(PREPARE_OP) and has_tm_op(GEMM_OP)):
            raise RuntimeError(
                f"SM70 FP8 needs vllm_sm70.ops.tm with {PREPARE_OP} and {GEMM_OP}."
            )
        weight, scale = process_fp8_weight_block_strategy(
            layer.weight, layer.weight_scale_inv
        )
        scale = scale.to(torch.float32)
        tm_weight, tm_scales, meta = _tm().fp8_sm70_prepare(
            weight.contiguous(), scale.contiguous(), FP8_BLOCK, False
        )
        replace_parameter(layer, "weight", tm_weight)
        replace_parameter(layer, "weight_scale_inv", tm_scales)
        layer.input_scale = None
        k_ld, q_ld = (int(v) for v in meta.tolist())
        layer.sm70_fp8_k_ld = k_ld
        layer.sm70_fp8_q_ld = q_ld
        layer.sm70_fp8_turbomind = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ported from FORK qpn/fp8.py:1045-1052, 1121-1141
        if x.dtype != torch.float16:
            raise TypeError(f"SM70 FP8 linear needs FP16 activations, got {x.dtype}.")
        n = layer.output_size_per_partition
        x_2d = flatten_linear_input(x)
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        out_2d = torch.empty((x_2d.shape[0], n), device=x.device, dtype=x.dtype)
        if x_2d.shape[0] > 0:
            _tm().fp8_gemm_sm70_out(
                out_2d,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                FP8_BLOCK,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
            )
        out = restore_linear_output(out_2d, x, n)
        if bias is not None:
            out.add_(bias)
        return out
