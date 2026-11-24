# SPDX-License-Identifier: Apache-2.0
"""Qwen block-128 FP8 linear on SM70 (TurboMind, block QPN8 for decode).

``Fp8Sm70LinearMethod`` subclasses upstream ``Fp8LinearMethod`` for e4m3
checkpoints with weight_block_size [128, 128] and dynamic activations. It
creates upstream's parameters without upstream's kernel selection (Volta
has no FP8 tensor cores, and upstream has no cc 7.0 kernel). After loading
it repacks with ``tm.fp8_sm70_prepare`` and runs ``tm.fp8_gemm_sm70_out``:
weight-only FP8 (W8A16), FP16 activations. Kill: VLLM_SM70_FP8_TURBOMIND=0.

Block QPN8 decode: when ``Sm70Config.fp8_block_qpn8`` is on, TP is 1 or 2,
the QPN8 ops exist and ``qpn8_rejection(N, K)`` is None, the layer is
repacked with ``qpn8_prepare`` and served by ``qpn8_linear`` for M <=
``qpn8_max_m(block_scaled=True)``. Larger M uses the TurboMind copy when
one is kept (``fp8_block_qpn8_turbomind_copy``), else a dequantize into one
shared per-device fp16 scratch. Kill: VLLM_SM70_FP8_BLOCK_QPN8=0."""

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
# QPN8 route lines go under "vllm" so vLLM's log handler prints them.
m8_logger = logging.getLogger("vllm.sm70")

FP8_BLOCK = 128
PREPARE_OP = "fp8_sm70_prepare"
GEMM_OP = "fp8_gemm_sm70_out"

# Ops the QPN8 route needs (no TurboMind copy, so no TurboMind op).
QPN8_OPS = (
    "fp8_qpn8_prepare_sm70",
    "fp8_qpn8_gemm_sm70_out",
    "fp8_qpn8_dequantize_sm70_out",
)
QPN8_API = ("qpn8_prepare", "qpn8_linear", "qpn8_rejection", "Qpn8Prepared")
# TP sizes the QPN8 route admits.
QPN8_TP_SIZES = (1, 2)
QPN8_SCRATCH = "sm70_qpn8_dense_scratch"
QPN8_DEFAULT_MAX_M = 8  # used only if ops.qpn8 has no qpn8_max_m()


def check_fp8_sm70_config(quant_config: Any) -> None:
    """Serialized e4m3, block [128, 128], dynamic activations, no UE8M0 scales."""
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
    """vllm_sm70.ops.qpn8 when its API and ops are present, else None.

    Missing module, wrapper or op means the plain TurboMind path, never an error.
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
    """QPN8 engine gate: Sm70Config.fp8_block_qpn8 (after the
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
    TurboMind copy).
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
    """Largest block-scaled QPN8 GEMV M: ``qpn8_max_m(block_scaled=True)``
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

    None (default): admit with no copy iff rows <= max_m; otherwise (or rows
      unknown) do not admit, so the layer stays on TurboMind with one copy.
    True: admit and always keep the copy (two weight copies).
    False: admit with no copy; rows above max_m use the QPN8 dense fallback.
    """
    if policy is None:
        return (rows is not None and rows <= max_m), False
    return True, bool(policy)


class _Qpn8DenseScratch:
    """All of the QPN8 shared dense-scratch plumbing (``dense_scratch``).

    One fp16 buffer per device, shared by every QPN8 layer there, grown at
    load to the largest K*N; layers already bound are re-pointed, so nothing
    is allocated after loading. Layers run one at a time on the model
    stream, so sharing is safe. Only this class passes ``dense_scratch``.
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
    "VLLM_SM70_FP8_QPN8_M32=1 is not recommended: block QPN8 at M=24/32 "
    "is slower than the TurboMind FP8 path; unset it to stay on TurboMind "
    "at these rows."
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
    on admitted layers when block QPN8 is on."""

    def __init__(self, quant_config: Any) -> None:
        check_fp8_sm70_config(quant_config)
        super().__init__(quant_config)  # keys only
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
        # Upstream's block branch without init_fp8_linear_kernel, which has
        # no kernel for cc 7.0.
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
        """Repack [N, K] e4m3 + [N/128, K/128] scales into TurboMind operands."""
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
        """Repack into QPN8 at load, or False for the TurboMind path.

        Admission is by shape only (qpn8_rejection).
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
        # Free the checkpoint layout (the caller of qpn8_prepare frees it);
        # the parameters stay for weight reloading.
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
