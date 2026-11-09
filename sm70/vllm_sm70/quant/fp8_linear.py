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

M8 (block QPN8 decode): when ``Sm70Config.fp8_block_qpn8`` is True (the
  default, as the fork's kernel_config.sm70_fp8.block_qpn8, FORK
  config/kernel.py:386; on for Qwen3.6 and Qwen3.8 alike),
  VLLM_SM70_FP8_BLOCK_QPN8 is not "0", TP is 1 or 2, K09's ops are present
  (vllm_sm70.ops.qpn8), ``qpn8_rejection(N, K)`` is None and the scale
  raster is [N/128, K/128] (FORK kernels/linear/qpn/fp8_block.py:160-195),
  the layer is repacked at load with ``qpn8_prepare`` instead of TurboMind
  and served by ``qpn8_linear``. M<=max_m runs the QPN8 GEMV, with max_m =
  K09's ``qpn8_max_m(block_scaled=True)``: 8 by default, 32 with
  VLLM_SM70_FP8_QPN8_M32=1 (K09's opt-in). Larger M uses the TurboMind copy when one is kept, else a
  dequantize into one shared per-device fp16 scratch (allocated at load)
  and torch.mm. ``fp8_block_qpn8_turbomind_copy`` decides (rows =
  max_num_seqs * (num_speculative_tokens + 1)): None (default) admits QPN8
  with no copy iff rows <= max_m and otherwise leaves the model on M6, so
  the default never holds two weight copies; True admits and keeps a
  TurboMind copy (the fork's two-copy behaviour, ~+12 GiB per GPU on
  Qwen3.8-27B at TP2); False admits with no copy. The M choice is inside
  K09's opaque op, so a graph traced at small M stays correct. ``Sm70Config.fp8_qpn8`` (the
  fork's sm70_native qpn8 selector) does not gate this. The fork's PP2/TP4
  QPN8 variant is not carried. Every other case is the M6 TurboMind path
  unchanged.
  Kill-switch: ``VLLM_SM70_FP8_BLOCK_QPN8=0``.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config_or_none
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

from vllm_sm70.config import sm70_config
from vllm_sm70.quant.nvfp4_linear import (
    _tm,
    flatten_linear_input,
    has_tm_op,
    restore_linear_output,
)

logger = init_logger("vllm.sm70.quant.fp8")
# M8 route lines go under "vllm" so vLLM's handler prints them in server.log
# (m8_qpn8_ab.sh greps "SM70 FP8 QPN8").
m8_logger = logging.getLogger("vllm.sm70")

FP8_BLOCK = 128
PREPARE_OP = "fp8_sm70_prepare"
GEMM_OP = "fp8_gemm_sm70_out"

# M8: K09 ops the QPN8 route needs (no TurboMind copy, so no K03 op).
QPN8_OPS = (
    "fp8_qpn8_prepare_sm70",
    "fp8_qpn8_gemm_sm70_out",
    "fp8_qpn8_dequantize_sm70_out",
)
QPN8_API = ("qpn8_prepare", "qpn8_linear", "qpn8_rejection", "Qpn8Prepared")
# TP2 is the target, TP1 is in K09's shape-only admission. The fork's TP4
# QPN8 meaning (PP2/TP4 tables) is not carried.
QPN8_TP_SIZES = (1, 2)
QPN8_SCRATCH = "sm70_qpn8_dense_scratch"
QPN8_DEFAULT_MAX_M = 8  # used only if K09 has no qpn8_max_m()


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


def qpn8_module():
    """K09's vllm_sm70.ops.qpn8 when its API and ops are present, else None.

    Missing module, wrapper or op means the plain M6 path, never an error.
    """
    try:
        from vllm_sm70.ops import qpn8
    except (ImportError, OSError, RuntimeError):  # unbuilt, or a stub ops pkg
        return None
    if not all(callable(getattr(qpn8, name, None)) for name in QPN8_API):
        return None
    has_op = getattr(qpn8, "has_op", None)
    if not callable(has_op) or not all(has_op(name) for name in QPN8_OPS):
        return None
    return qpn8


def qpn8_route_enabled(vllm_config: Any = None) -> bool:
    """M8 engine gate: Sm70Config.fp8_block_qpn8 (after the
    VLLM_SM70_FP8_BLOCK_QPN8 kill) and TP in QPN8_TP_SIZES. Layer admission
    and op presence are checked at load. No vllm_config: False."""
    if vllm_config is None:
        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or not sm70_config(vllm_config).fp8_block_qpn8:
        return False
    parallel = getattr(vllm_config, "parallel_config", None)
    tp = int(getattr(parallel, "tensor_parallel_size", 1) or 1)
    return tp in QPN8_TP_SIZES


def qpn8_max_rows(vllm_config: Any = None) -> int | None:
    """Largest decode/verify M: max_num_seqs * (num_speculative_tokens + 1).

    None when the scheduler config is not reachable (auto then keeps the
    TurboMind copy, as the fork did without an engine config).
    """
    if vllm_config is None:
        vllm_config = get_current_vllm_config_or_none()
    sched = getattr(vllm_config, "scheduler_config", None)
    seqs = getattr(sched, "max_num_seqs", None)
    if seqs is None:
        return None
    spec = getattr(vllm_config, "speculative_config", None)
    k = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
    return int(seqs) * (k + 1)


def qpn8_max_m(qpn8: Any) -> int:
    """K09's largest block-scaled QPN8 GEMV M: ``qpn8_max_m(block_scaled=True)``
    (8 by default, 32 with VLLM_SM70_FP8_QPN8_M32=1; QPN8_MAX_M is only the
    compiled kernel bound). 8 if the module has no such function."""
    fn = getattr(qpn8, "qpn8_max_m", None)
    if not callable(fn):
        return QPN8_DEFAULT_MAX_M
    return int(fn(block_scaled=True))


def qpn8_copy_decision(
    policy: bool | None, rows: int | None, max_m: int
) -> tuple[bool, bool]:
    """(admit QPN8, keep a TurboMind copy) from fp8_block_qpn8_turbomind_copy.

    None (default, safe): admit with no copy iff rows <= max_m; otherwise
      (or rows unknown) do not admit, so the layer stays M6 with one copy.
      This deliberately differs from the fork's auto, which kept the copy
      when rows > 8 (FORK config/kernel.py:388, fp8_block.py:204-213): the
      default must not double the weights on a 32 GB V100.
    True: admit and always keep the copy (the fork's two-copy behaviour).
    False: admit with no copy; rows above max_m use K09's dense fallback.
    """
    if policy is None:
        return (rows is not None and rows <= max_m), False
    return True, bool(policy)


class _Qpn8DenseScratch:
    """All of M8's shared dense-scratch plumbing (K09's ``dense_scratch``).

    One fp16 buffer per device, shared by every QPN8 layer there, grown at
    load to the largest K*N; layers already bound are re-pointed, so nothing
    is allocated after loading. Layers run one at a time on the model stream,
    as the fork's shared exact-dense workspace assumed (FORK
    qpn/fp8.py:318-352). If K09 drops the ``Tensor(a!)`` argument (Inductor
    cloning it), only this class changes: ``bind`` becomes a no-op and
    ``linear`` stops passing ``dense_scratch``.
    """

    def __init__(self) -> None:
        self.pools: dict[str, tuple[list[torch.Tensor], weakref.WeakSet]] = {}

    def bind(self, layer: torch.nn.Module, numel: int, device: torch.device) -> None:
        holder, layers = self.pools.setdefault(str(device), ([], weakref.WeakSet()))
        if not holder or holder[0].numel() < numel:
            holder[:] = [torch.empty((numel,), dtype=torch.float16, device=device)]
            for other in layers:
                other._buffers[QPN8_SCRATCH] = holder[0]
        layer.register_buffer(QPN8_SCRATCH, holder[0], persistent=False)
        layers.add(layer)

    @staticmethod
    def linear(qpn8, layer, x, prepared, bias):
        return qpn8.qpn8_linear(
            x, prepared, bias, dense_scratch=getattr(layer, QPN8_SCRATCH, None)
        )

    def clear(self) -> None:
        self.pools.clear()


_QPN8_DENSE_SCRATCH = _Qpn8DenseScratch()
_LOGGED: set[str] = set()


def _log_once(msg: str, level: int = logging.INFO) -> None:
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        m8_logger.log(level, msg)


M32_SLOW_WARNING = (
    "VLLM_SM70_FP8_QPN8_M32=1 is not recommended: block QPN8 at M=17..32 "
    "measured 27-33% slower than M6 TurboMind (Qwen3.8-27B TP2 DFlash2, "
    "c=3/4, M8 A/B 2026-11-08); unset it to stay on M6 at these rows."
)


_COPY_BYTES: dict[str, int] = {}


def _count_copy_bytes(device: torch.device, nbytes: int) -> None:
    """Running total of the TurboMind copy per GPU, logged once per GiB."""
    if not nbytes:
        return
    key = str(device)
    before = _COPY_BYTES.get(key, 0)
    _COPY_BYTES[key] = after = before + nbytes
    if after >> 30 > before >> 30 or before == 0:
        msg = (
            f"SM70 FP8 QPN8 TurboMind copy on {key}: {after / 2**30:.2f} GiB so far "
            "(logged at each GiB crossed; fp8_block_qpn8_turbomind_copy=True "
            "holds two copies of each dense FP8 weight)"
        )
        if msg not in _LOGGED:
            _LOGGED.add(msg)
            m8_logger.warning(msg)


def clear_qpn8_scratch() -> None:
    _QPN8_DENSE_SCRATCH.clear()
    _COPY_BYTES.clear()


@register_weight_loader_v2_supported_method
class Fp8Sm70LinearMethod(Fp8LinearMethod):
    """TurboMind W8A16 block-FP8 linear (FP16 activations); QPN8 for decode
    on admitted layers when M8 is on."""

    def __init__(self, quant_config: Any) -> None:
        check_fp8_sm70_config(quant_config)
        super().__init__(quant_config)  # UP fp8.py:241-280 (keys only)
        self.qpn8_enabled = qpn8_route_enabled()
        self.qpn8_max_rows = qpn8_max_rows() if self.qpn8_enabled else None
        self.qpn8_copy_policy = (
            sm70_config(get_current_vllm_config_or_none()).fp8_block_qpn8_turbomind_copy
            if self.qpn8_enabled
            else None
        )
        self._qpn8: Any = None

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
        layer.sm70_fp8_qpn8 = False
        if self.qpn8_enabled and self._prepare_qpn8(layer, weight, scale):
            return
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

    def _prepare_qpn8(
        self, layer: torch.nn.Module, weight: torch.Tensor, scale: torch.Tensor
    ) -> bool:
        """M8: repack into QPN8 at load (K09 contract), or False for M6.

        ported from FORK qpn/fp8.py:686-787 (generic, non-PP2/TP4 branch) with
        K09's shape-only admission (qpn8_rejection) instead of the fork's
        layer-suffix list.
        """
        qpn8 = qpn8_module()
        if qpn8 is None:
            _log_once(
                "SM70 FP8 QPN8 requested, but vllm_sm70.ops.qpn8 or its ops are "
                "missing; using TurboMind FP8."
            )
            return False
        max_m = qpn8_max_m(qpn8)
        rows = self.qpn8_max_rows
        admit, keep_copy = qpn8_copy_decision(self.qpn8_copy_policy, rows, max_m)
        if not admit:
            _log_once(
                "SM70 FP8 QPN8 not used: decode/verify can reach "
                f"rows={rows if rows is not None else 'unknown'} (max_num_seqs x "
                f"(num_speculative_tokens + 1)) > max_m={max_m} and "
                "fp8_block_qpn8_turbomind_copy is None (safe default: no second "
                "weight copy); using TurboMind FP8. Set VLLM_SM70_FP8_QPN8_M32=1, "
                "lower max_num_seqs, or set the copy policy explicitly."
            )
            return False
        if weight.dim() != 2:
            return False
        n, k = (int(d) for d in weight.shape)
        if qpn8.qpn8_rejection(n, k) is not None:
            return False
        if tuple(scale.shape) != (n // FP8_BLOCK, k // FP8_BLOCK):
            return False  # e.g. a merged layer with an unaligned partition
        prepared = qpn8.qpn8_prepare(
            weight.contiguous(), scale.contiguous(), turbomind_fallback=keep_copy
        )
        layer.register_buffer("sm70_qpn8_codes", prepared.codes, persistent=False)
        layer.register_buffer(
            "sm70_qpn8_group_scales", prepared.group_scales, persistent=False
        )
        copy_bytes = 0
        if prepared.tm_weight is not None:
            layer.register_buffer(
                "sm70_qpn8_tm_weight", prepared.tm_weight, persistent=False
            )
            layer.register_buffer(
                "sm70_qpn8_tm_scales", prepared.tm_scales, persistent=False
            )
            copy_bytes = sum(
                t.numel() * t.element_size()
                for t in (prepared.tm_weight, prepared.tm_scales)
            )
        layer.sm70_qpn8_meta = (
            prepared.n,
            prepared.k,
            prepared.split_k,
            prepared.accumulator_chains,
            prepared.prefetch_codes,
            prepared.tm_k_ld,
            prepared.tm_q_ld,
        )
        device = prepared.codes.device
        if prepared.tm_weight is None:
            _QPN8_DENSE_SCRATCH.bind(layer, n * k, device)
        # Free the checkpoint layout (K09: "the caller frees the checkpoint
        # weight afterwards"); the parameters stay for weight reloading.
        replace_parameter(
            layer, "weight", torch.empty(0, dtype=torch.uint8, device=device)
        )
        replace_parameter(
            layer,
            "weight_scale_inv",
            torch.empty(0, dtype=torch.float32, device=device),
        )
        layer.input_scale = None
        layer.sm70_fp8_qpn8 = True
        layer.sm70_fp8_turbomind = False
        self._qpn8 = qpn8
        _log_once(
            f"SM70 FP8 QPN8 decode (M<={max_m}) enabled for block-FP8 linears: "
            f"rows={rows} (max_num_seqs x (num_speculative_tokens + 1)), "
            f"max_m={max_m}, keep_copy={keep_copy} (policy "
            f"{self.qpn8_copy_policy}); larger M "
            + (
                "runs a second, TurboMind-packed copy of each weight."
                if keep_copy
                else "dequantizes into a shared FP16 scratch."
            )
        )
        if max_m > QPN8_DEFAULT_MAX_M:
            _log_once(M32_SLOW_WARNING, logging.WARNING)
        _count_copy_bytes(device, copy_bytes)
        return True

    def _apply_qpn8(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        qpn8 = self._qpn8
        n, k, split_k, chains, prefetch, k_ld, q_ld = layer.sm70_qpn8_meta
        prepared = qpn8.Qpn8Prepared(
            layer.sm70_qpn8_codes,
            layer.sm70_qpn8_group_scales,
            n,
            k,
            split_k,
            chains,
            prefetch,
            getattr(layer, "sm70_qpn8_tm_weight", None),
            getattr(layer, "sm70_qpn8_tm_scales", None),
            k_ld,
            q_ld,
        )
        return _QPN8_DENSE_SCRATCH.linear(qpn8, layer, x, prepared, bias)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ported from FORK qpn/fp8.py:1045-1052, 1121-1141
        if x.dtype != torch.float16:
            raise TypeError(f"SM70 FP8 linear needs FP16 activations, got {x.dtype}.")
        if getattr(layer, "sm70_fp8_qpn8", False):
            return self._apply_qpn8(layer, x, bias)
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
