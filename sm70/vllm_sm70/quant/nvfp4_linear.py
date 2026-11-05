# SPDX-License-Identifier: Apache-2.0
"""M3 nvfp4-dense-turbomind: ModelOpt W4A16_NVFP4 dense linear on SM70.

What: ``ModelOptNvFp4Sm70LinearMethod``, plus a ``LINEAR_METHOD_BUILDERS
  ["W4A16_NVFP4"]`` builder (UP 4ba787e modelopt.py:2763-2790) that returns it
  only when ``vllm_sm70.platform.is_sm70_ready()`` (K01). On any other
  device the builder falls through to upstream's own ``resolve()`` +
  ``ModelOptLinearMethod`` (modelopt.py:2710-2760 and :2777-2790), so
  non-SM70 behaviour is unchanged.
Why: upstream's W4A16 path selects its kernel through ``init_nvfp4_linear_
  kernel(use_a16=True)`` (kernels/linear/__init__.py:1191). On cc 7.0 the
  ``auto`` branch forces Marlin (:1236-1245), which needs SM75+, so it raises.
  The builder seam is the documented escape hatch (modelopt.py:2763-2774), so
  this needs no UP edit and no kernel-selector change.
  Parameters are created by upstream's own ``KNvfp4Static`` scheme and
  ``_DROP_INPUT_SCALE`` format (modelopt.py:2018-2096, :2532-2557). Names,
  dtypes, shapes, parameter classes and the NaN sentinel are therefore
  identical to what UP's W4A16 loader produces. Only kernel selection,
  post-load preparation and apply differ.
Kill-switch: ``Sm70Config.nvfp4_gemv`` / ``VLLM_SM70_MODELOPT_NVFP4_GEMV=0``
  turns off the M<=2 GEMV only; TurboMind GEMM then serves every M.
  ``VLLM_SM70_QUANT=0`` (M2) disables the whole plugin route, including this
  builder.

Kernels: Kai's ops, reached only through ``vllm_sm70.ops.tm`` and never
``torch.ops._sm70_C`` directly. Contract assumed here (unverified until Kai
lands ``vllm_sm70.ops``):

    tm.nvfp4_sm70_prepare(qweight u8 [K,N], scales f16 [K/16,N],
                          group_size=16, interleave_gated_silu=False)
        -> [tm_weight i32 [K,N/8], tm_scales f16 [K/16,N], meta i64[2]=(k_ld,q_ld)]
    tm.nvfp4_gemm_sm70_out(out, x, tm_weight, tm_scales, group_size, k_ld,
                          q_ld, gated_silu=False) -> None (mutates out)
    tm.nvfp4_tm_gemv_dispatch_sm70_out(out, x, tm_weight, tm_scales,
                          group_size, k_ld, q_ld, scale_shift=14) -> None
    tm.has_op(name) -> bool        (checks torch.ops._sm70_C)

K01 (main a17c3b0) ships ``vllm_sm70/ops/__init__.py`` with only the area
``register()`` hook; ``vllm_sm70.ops.tm`` arrives with K03. Until then
``has_tm_op`` returns False, and process_weights_after_loading raises a clear
error instead of picking another kernel. The CPU tests mock the module.

Kai confirmed this against FORK cf9492a csrc/torch_bindings.cpp:654-660,
948-961 and vllm/_sm70/linear.py:350, 1763, 1805, minus the ``native_policy``
tuple. No op takes a global scale: the caller folds weight_scale_2 into the
fp16 scales once, before prepare (``format_nvfp4_scales``). The GEMV is
feature-detected: without K11's op, the GEMM serves every M.

Not M3: online ``--quantization fp8`` is admitted by M2's capability gate
(Sm70Fp8Config) but has no SM70 linear path here. The M6 owner must refuse it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Parameter
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import modelopt as up_modelopt
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

from vllm_sm70.config import sm70_config

logger = init_logger(__name__)

ALGO = "W4A16_NVFP4"
NVFP4_GROUP_SIZE = 16
# SM70 packed NVFP4 GEMM needs complete 32-column tiles (FORK sm70_turbomind.py:26-30).
NVFP4_OUTPUT_ALIGNMENT = 32
STATE_ATTR = "_sm70_turbomind_linear"
GEMM_OP = "nvfp4_gemm_sm70_out"
GEMV_OP = "nvfp4_tm_gemv_dispatch_sm70_out"
PREPARE_OP = "nvfp4_sm70_prepare"

# Per-rank (K, N) of the Flint ModelOpt W4A16 dense layers at TP2: qkv, o,
# shared-expert gate/up and down, dense-MLP gate/up and down. The GEMV launch
# table in K11's kernel is tuned for exactly these.
# ported from FORK sm70_turbomind.py:567-579
MODELOPT_NVFP4_GEMV_SHAPES = frozenset(
    {
        (2048, 2560),
        (2048, 2048),
        (2048, 1024),
        (512, 2048),
        (2048, 6144),
        (3072, 2048),
    }
)
_FP16_MAX = 65504.0  # FORK sm70_turbomind.py:580


# --------------------------------------------------------------------------
# Kai's op surface
# --------------------------------------------------------------------------
def _tm():
    """Kai's TurboMind op wrappers (import is lazy: the module may not exist yet)."""
    from vllm_sm70.ops import tm  # type: ignore[import-not-found]

    return tm


def has_tm_op(name: str) -> bool:
    # ported from FORK sm70_turbomind.py:597-598 (has_tm_gemv_op), generalized
    try:
        tm = _tm()
    except ImportError:
        return False
    probe = getattr(tm, "has_op", None)
    if callable(probe):
        return bool(probe(name))
    return callable(getattr(tm, name, None))


# --------------------------------------------------------------------------
# Layer state
# --------------------------------------------------------------------------
@dataclass
class SM70TurboMindLinearState:
    """Prepared TurboMind operands of one linear.

    ported from FORK sm70_turbomind.py:36-73. NVFP4 only: the uint4/fp8/mxfp4/
    QPN op kinds, the gated-SiLU and prescaled variants, and ``native_ops``
    (NativeBindings, Kai's boundary) are dropped. ``refresh_provider`` and
    FORK sm70_provider.py:106-166 (``bind_prepared_provider``) collapse into
    ``provider``: "gemm" or "tm_gemv".
    """

    weight: torch.Tensor
    scales: torch.Tensor
    group_size: int
    k_ld: int
    q_ld: int
    output_size: int
    padded_output_size: int = 0
    tm_gemv: bool = False
    tm_gemv_scale_shift: int = 14

    @property
    def provider(self) -> str:
        return GEMV_OP if self.tm_gemv else GEMM_OP


def has_prepared_linear(layer: torch.nn.Module) -> bool:
    # ported from FORK sm70_turbomind.py:383-384
    return getattr(layer, STATE_ATTR, None) is not None


def _store_state(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor | None,
    group_size: int,
    output_size: int,
    padded_output_size: int = 0,
) -> SM70TurboMindLinearState:
    # ported from FORK sm70_turbomind.py:346-380 (nvfp4 op_kind only)
    state = SM70TurboMindLinearState(
        weight=weight,
        scales=scales,
        group_size=group_size,
        k_ld=0 if meta is None else int(meta[0]),
        q_ld=0 if meta is None else int(meta[1]),
        output_size=output_size,
        padded_output_size=padded_output_size,
    )
    setattr(layer, STATE_ATTR, state)
    return state


# --------------------------------------------------------------------------
# Unpack / format
# --------------------------------------------------------------------------
def _get_u4_slices(x: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    # ported from FORK sm70_turbomind.py:294-305
    if x.dtype == torch.int32:
        count = 8
    elif x.dtype == torch.uint8:
        count = 2
    else:
        raise TypeError(f"expected int32 or uint8 packed int4 tensor, got {x.dtype}")
    xs = []
    for _ in range(count):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def unpack_fp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    """[N, K/2] u8 (low nibble = even k) -> [K, N] u8 E2M1 codes.

    ported from FORK sm70_turbomind.py:331-339 (``unpack_mxfp4_weight``; the
    fork uses it for NVFP4 as well, :507).
    """
    if weight_packed.dim() > 2:
        weight_packed = torch.flatten(weight_packed, start_dim=-2)
    xs = _get_u4_slices(weight_packed, torch.uint8)
    weight = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
    return weight.t().contiguous()


def format_nvfp4_scales(
    weight_scale: torch.Tensor, weight_global_scale: torch.Tensor
) -> torch.Tensor:
    """[N, K/16] E4M3 block scales x FP32 global -> [K/16, N] FP16.

    ported from FORK sm70_turbomind.py:508-515. ModelOpt stores the global
    scale as amax / (6 * 448), and TurboMind combines block * global (FORK
    modelopt.py:1129-1135), so there is no reciprocal.
    """
    return (
        (weight_scale.t().to(torch.float32) * weight_global_scale.to(torch.float32))
        .to(torch.float16)
        .contiguous()
    )


def _pad_output(
    qweight: torch.Tensor, scales: torch.Tensor, output_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # ported from FORK sm70_turbomind.py:516-536 (gated-SiLU check dropped:
    # M3 never interleaves)
    padded = (
        (output_size + NVFP4_OUTPUT_ALIGNMENT - 1) // NVFP4_OUTPUT_ALIGNMENT
    ) * NVFP4_OUTPUT_ALIGNMENT
    if padded == output_size:
        return qweight, scales, padded
    padded_qweight = qweight.new_zeros((qweight.size(0), padded))
    padded_scales = scales.new_zeros((scales.size(0), padded))
    padded_qweight[:, :output_size].copy_(qweight)
    padded_scales[:, :output_size].copy_(scales)
    return padded_qweight, padded_scales, padded


def prepare_nvfp4_linear(layer: torch.nn.Module) -> SM70TurboMindLinearState:
    """Repack a loaded ModelOpt NVFP4 linear into TurboMind operands.

    ported from FORK sm70_turbomind.py:493-564 (interleave_gated_silu=False
    and prescale_for_batch=False, which is how the ModelOpt W4A16 path calls
    it: FORK modelopt.py:1142, :1588). Expects ``layer.weight`` [N, K/2] u8,
    ``layer.weight_scale`` [N, K/16] E4M3 and ``layer.weight_global_scale``
    (scalar FP32), all produced by upstream's ``KNvfp4Static.process``.
    """
    if not has_tm_op(PREPARE_OP) or not has_tm_op(GEMM_OP):
        raise RuntimeError(
            "SM70 ModelOpt NVFP4 needs vllm_sm70.ops.tm with "
            f"{PREPARE_OP} and {GEMM_OP} (Kai's TurboMind extension)."
        )
    tm = _tm()
    qweight = unpack_fp4_weight(layer.weight.data)
    scales = format_nvfp4_scales(layer.weight_scale.data, layer.weight_global_scale)
    output_size = qweight.size(1)
    qweight, scales, padded = _pad_output(qweight, scales, output_size)
    tm_weight, tm_scales, meta = tm.nvfp4_sm70_prepare(
        qweight,
        scales,
        NVFP4_GROUP_SIZE,
        False,  # group_size, interleave_gated_silu
    )
    return _store_state(
        layer,
        tm_weight,
        tm_scales,
        meta,
        NVFP4_GROUP_SIZE,
        output_size,
        padded_output_size=padded,
    )


# --------------------------------------------------------------------------
# M<=2 GEMV admission
# --------------------------------------------------------------------------
def _tm_gemv_scale_shift(scales: torch.Tensor) -> int:
    """Largest power-of-two prescale (<= 2^14) that keeps FP16 scales finite.

    The GEMV multiplies each FP16 scale by 2^shift and each E2M1 code (an FP16
    pattern worth code * 2^-14) by that, then applies 2^(14 - shift) in FP32.
    With shift 14 every weight equals TurboMind's fp16(code * scale).

    ported from FORK sm70_turbomind.py:583-594
    """
    peak = float(scales.abs().amax())
    shift = 14
    while shift > 0 and peak * 2.0**shift > _FP16_MAX:
        shift -= 1
    return shift


def admit_modelopt_nvfp4_gemv(
    state: SM70TurboMindLinearState, *, enabled: bool
) -> bool:
    """Route M <= 2 of a prepared ModelOpt W4A16 linear to the TM GEMV.

    Admitted only for the exact Flint per-rank shapes, plain (non-padded)
    TurboMind NVFP4 operands, and an extension that has the operator. The
    operator keeps TurboMind for M > 2, so CUDA graphs and torch.compile see a
    single opaque call either way.

    ported from FORK sm70_turbomind.py:601-631. The fork's gated/prescaled/
    scale-code exclusions are always true here. ``enabled`` replaces
    ``format_option("nvfp4", "modelopt_gemv")`` (FORK :99-132, :624), and the
    exact-SM70 tensor check (FORK :198-201) is the builder's job.
    """
    k = int(state.weight.shape[0])
    n = int(state.weight.shape[1]) * 8
    if (
        not enabled
        or state.group_size != NVFP4_GROUP_SIZE
        or state.padded_output_size not in (0, state.output_size)
        or (k, n) not in MODELOPT_NVFP4_GEMV_SHAPES
        or state.k_ld != 32 * k
        or state.q_ld != n
        or state.scales.dtype != torch.float16
        or not has_tm_op(GEMV_OP)
    ):
        return False
    state.tm_gemv_scale_shift = _tm_gemv_scale_shift(state.scales)
    state.tm_gemv = True
    return True


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------
def flatten_linear_input(x: torch.Tensor, features: int | None = None) -> torch.Tensor:
    """Keep existing matrix views; flatten only additional batch dimensions.

    ported from FORK vllm/model_executor/kernels/linear_io.py:9-15
    """
    features = x.shape[-1] if features is None else features
    if x.ndim == 2 and features != 0 and x.shape[-1] == features:
        return x
    return x.reshape(-1, features)


def restore_linear_output(
    output: torch.Tensor, x: torch.Tensor, features: int | None = None
) -> torch.Tensor:
    # ported from FORK vllm/model_executor/kernels/linear_io.py:18-24
    features = output.shape[-1] if features is None else features
    if x.ndim == 2 and output.shape == (x.shape[0], features):
        return output
    return output.reshape(*x.shape[:-1], features)


def _run(state: SM70TurboMindLinearState, x: torch.Tensor, out: torch.Tensor) -> None:
    # ported from FORK sm70_provider.py:24-53 (_gemm non-gated, _tm_gemv)
    tm = _tm()
    args = (
        out,
        x,
        state.weight,
        state.scales,
        state.group_size,
        state.k_ld,
        state.q_ld,
    )
    if state.tm_gemv:
        # The operator branches on M itself: GEMV for M <= 2, TurboMind above.
        tm.nvfp4_tm_gemv_dispatch_sm70_out(*args, state.tm_gemv_scale_shift)
    else:
        tm.nvfp4_gemm_sm70_out(*args)


def apply_prepared_linear(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Flatten, dispatch, crop the alignment padding, add bias, reshape.

    ported from FORK sm70_turbomind.py:911-920 + sm70_provider.py:169-203
    (non-gated, non-interleaved branch)
    """
    state: SM70TurboMindLinearState = getattr(layer, STATE_ATTR)
    if x.dtype != torch.float16:
        # TurboMind W4A16 on SM70 is FP16-only (V100 has no BF16).
        raise RuntimeError(
            f"SM70 NVFP4 TurboMind needs float16 activations, got {x.dtype}"
        )
    x_2d = flatten_linear_input(x)
    logical_n = state.output_size
    physical_n = state.padded_output_size or logical_n
    out = torch.empty((x_2d.shape[0], physical_n), dtype=x.dtype, device=x.device)
    _run(state, x_2d, out)
    if physical_n != logical_n:
        out = out[:, :logical_n]
    if bias is not None:
        out.add_(bias)
    return restore_linear_output(out, x, logical_n)


# --------------------------------------------------------------------------
# Linear method
# --------------------------------------------------------------------------
@register_weight_loader_v2_supported_method
class ModelOptNvFp4Sm70LinearMethod(LinearMethodBase):
    """ModelOpt W4A16_NVFP4 dense linear on SM70 (TurboMind W4A16).

    Lifecycle mirrors UP ``ModelOptLinearMethod`` (modelopt.py:2560-2708) for
    ``resolve("W4A16_NVFP4")`` = (kNvfp4Static, no activation,
    _DROP_INPUT_SCALE) (:2754-2758), with the kernel replaced. FORK
    equivalent: ``ModelOptNvFp4W4A16LinearMethod`` (FORK modelopt.py:1453-1601)
    + ``_try_prepare_sm70_modelopt_nvfp4`` (:1126-1167).
    """

    # Pre-processed (repacked) weights would bypass the TurboMind repack.
    supports_pre_processed_weights = False

    def __init__(self, quant_config: Any, prefix: str = "") -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        group_size = getattr(quant_config, "group_size", NVFP4_GROUP_SIZE)
        if group_size != NVFP4_GROUP_SIZE:
            raise ValueError(
                f"SM70 NVFP4 TurboMind needs group_size {NVFP4_GROUP_SIZE}, "
                f"got {group_size}"
            )
        self.ctx = up_modelopt.CkptCtx(group_size=group_size)
        self.wkey = up_modelopt.SCHEME_FOR[kNvfp4Static]
        self.fmt = up_modelopt._DROP_INPUT_SCALE
        self.gemv_enabled = bool(
            sm70_config(get_current_vllm_config_or_none()).nvfp4_gemv
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Same as UP ModelOptLinearMethod.create_weights (modelopt.py:2612-2641)
        # up to, but excluding, select_linear_kernel (:2634), which would pick
        # Marlin and fail on cc 7.0.
        del input_size, output_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        shapes = up_modelopt.Shapes(
            output_partition_sizes, input_size_per_partition, params_dtype
        )
        self.wkey.create_weights(
            layer, up_modelopt.WEIGHT, self.ctx, shapes, weight_loader
        )
        self.fmt.extra_weights(layer, shapes, self.ctx, weight_loader)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # UP lifecycle (modelopt.py:2643-2661): NaN sentinel check, global
        # scale -> weight_global_scale, input_scale dropped.
        self.wkey.process(layer, up_modelopt.WEIGHT)
        self.fmt.post_process(layer)
        state = prepare_nvfp4_linear(layer)
        if admit_modelopt_nvfp4_gemv(state, enabled=self.gemv_enabled):
            logger.info_once(
                "SM70 ModelOpt NVFP4 W4A16 single-token GEMV enabled for the "
                "Flint dense shapes (VLLM_SM70_MODELOPT_NVFP4_GEMV=0 disables it)."
            )
        logger.info_once(
            "SM70 ModelOpt NVFP4 TurboMind dense path enabled "
            "(weight-only; activations remain half)."
        )
        # Free the checkpoint layout (ported from FORK modelopt.py:1158-1166).
        layer.weight = Parameter(
            torch.empty(0, dtype=torch.uint8, device=layer.weight.device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.empty(0, dtype=torch.float8_e4m3fn, device=layer.weight_scale.device),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_prepared_linear(layer, x, bias)


# --------------------------------------------------------------------------
# Builder (UP modelopt.py:2763-2790)
# --------------------------------------------------------------------------
_upstream_builder: Callable[..., LinearMethodBase] | None = None


def _fallthrough(config: Any, prefix: str) -> LinearMethodBase:
    """Exactly what UP build_linear_method does with no builder registered."""
    if _upstream_builder is not None:
        return _upstream_builder(config, prefix)
    spec, ctx, format_scheme = up_modelopt.resolve(ALGO, config, prefix)
    return up_modelopt.ModelOptLinearMethod(spec, ctx, format_scheme)


def build_w4a16_nvfp4(config: Any, prefix: str) -> LinearMethodBase:
    # K01 readiness (all visible GPUs cc 7.0 and _sm70_C loaded), looked up on
    # the module so tests can patch it, as M2 does.
    from vllm_sm70 import platform as sm70_platform

    if sm70_platform.is_sm70_ready():
        return ModelOptNvFp4Sm70LinearMethod(config, prefix)
    return _fallthrough(config, prefix)


def install_linear_builder() -> None:
    """Install the builder (idempotent; keeps any non-SM70 builder to chain to)."""
    global _upstream_builder
    current = up_modelopt.LINEAR_METHOD_BUILDERS.get(ALGO)
    if (
        current is not None
        and getattr(current, "__name__", "") != build_w4a16_nvfp4.__name__
    ):
        _upstream_builder = current
    up_modelopt.LINEAR_METHOD_BUILDERS[ALGO] = build_w4a16_nvfp4


def uninstall_linear_builder() -> None:
    if getattr(up_modelopt.LINEAR_METHOD_BUILDERS.get(ALGO), "__name__", "") == (
        build_w4a16_nvfp4.__name__
    ):
        if _upstream_builder is not None:
            up_modelopt.LINEAR_METHOD_BUILDERS[ALGO] = _upstream_builder
        else:
            del up_modelopt.LINEAR_METHOD_BUILDERS[ALGO]


def builder_installed() -> bool:
    b = up_modelopt.LINEAR_METHOD_BUILDERS.get(ALGO)
    return getattr(b, "__name__", "") == build_w4a16_nvfp4.__name__ and (
        getattr(b, "__module__", "") == __name__
    )
