# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboMind SM70 ops: Python wrappers and fake impls over torch.ops._sm70_C.

Tuning knobs are read in C++ (sm70/csrc/turbomind/ops/tm_config.h), not
passed from Python. Importing this module registers the fakes for every op
that the loaded `_sm70_C` provides (AWQ ops exist only in a
VLLM_SM70_TM_AWQ build). Wrappers raise RuntimeError when their op is
missing.
"""

from __future__ import annotations

import torch

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

NAMESPACE = "_sm70_C"

# Load _sm70_C before any has_op() check below: the fakes are registered at
# import, and an import that ran before the .so was loaded would silently
# register none of them.
from vllm_sm70 import _ext  # noqa: E402

_ext.load()


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def _op(name: str):
    if not has_op(name):
        raise RuntimeError(
            f"SM70 TurboMind op {NAMESPACE}::{name} is not available. "
            "Build vllm_sm70._sm70_C with CUDA 12.x for sm_70."
        )
    return getattr(getattr(torch.ops, NAMESPACE), name)


# ---------------------------------------------------------------------------
# fp16 helpers
# ---------------------------------------------------------------------------


def silu_and_mul_interleaved(out: torch.Tensor, input: torch.Tensor) -> None:
    _op("silu_and_mul_interleaved")(out, input)


if has_op("silu_and_mul_interleaved"):

    @register_fake("_sm70_C::silu_and_mul_interleaved")
    def _silu_and_mul_interleaved_fake(out: torch.Tensor, input: torch.Tensor) -> None:
        del out, input
        return None


def sm70_f16_gate_mul_out(
    out: torch.Tensor,
    input: torch.Tensor,
    gate_weight: torch.Tensor,
) -> None:
    _op("sm70_f16_gate_mul_out")(out, input, gate_weight)


if has_op("sm70_f16_gate_mul_out"):

    @register_fake("_sm70_C::sm70_f16_gate_mul_out")
    def _sm70_f16_gate_mul_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> None:
        return None

# ---------------------------------------------------------------------------
# dense GEMM
# ---------------------------------------------------------------------------


# registered only with VLLM_SM70_TM_AWQ
def awq_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("awq_sm70_prepare")(
        qweight,
        scales,
        qzeros,
        group_size,
        interleave_gated_silu,
    )


if has_op("awq_sm70_prepare"):

    @register_fake("_sm70_C::awq_sm70_prepare")
    def _awq_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del qzeros, group_size, interleave_gated_silu
        n = qweight.size(1) * 8
        num_groups = scales.size(0)
        tm_weight = torch.empty_like(qweight)
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.int32,
            device=qweight.device,
        )
        meta = torch.empty(
            (2,), dtype=torch.int64, device="cpu"
        )  # C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# registered only with VLLM_SM70_TM_AWQ
def awq_sm70_prepare_compact(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("awq_sm70_prepare_compact")(
        qweight,
        scales,
        qzeros,
        group_size,
        interleave_gated_silu,
    )


if has_op("awq_sm70_prepare_compact"):

    @register_fake("_sm70_C::awq_sm70_prepare_compact")
    def _awq_sm70_prepare_compact_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del qzeros, group_size, interleave_gated_silu
        n = qweight.size(1) * 8
        num_groups = scales.size(0)
        tm_weight = torch.empty_like(qweight)
        tm_scales = torch.empty(
            (num_groups, n, 3),
            dtype=torch.uint8,
            device=qweight.device,
        )
        meta = torch.empty(
            (2,), dtype=torch.int64, device="cpu"
        )  # C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# registered only with VLLM_SM70_TM_AWQ
def awq_sm70_dequantize_out(
    out: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("awq_sm70_dequantize_out")(out, qweight, scales, group_size)


if has_op("awq_sm70_dequantize_out"):

    @register_fake("_sm70_C::awq_sm70_dequantize_out")
    def _awq_sm70_dequantize_out_fake(
        out: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> None:
        del out, qweight, scales, group_size
        return None


# registered only with VLLM_SM70_TM_AWQ
def uint4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("uint4_sm70_prepare")(
        qweight,
        scales,
        zeros,
        group_size,
        interleave_gated_silu,
    )


if has_op("uint4_sm70_prepare"):

    @register_fake("_sm70_C::uint4_sm70_prepare")
    def _uint4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del zeros, group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.int32,
            device=qweight.device,
        )
        meta = torch.empty(
            (2,), dtype=torch.int64, device="cpu"
        )  # C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


def fp8_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("fp8_sm70_prepare")(
        qweight,
        scales,
        group_size,
        interleave_gated_silu,
    )


if has_op("fp8_sm70_prepare"):

    @register_fake("_sm70_C::fp8_sm70_prepare")
    def _fp8_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        n = qweight.size(0)
        k = qweight.size(1)
        num_groups = scales.size(1)
        tm_weight = torch.empty((k, n), dtype=torch.uint8, device=qweight.device)
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.float16,
            device=qweight.device,
        )
        meta = torch.empty(
            (2,), dtype=torch.int64, device="cpu"
        )  # C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


def fp8_sm70_dequantize_out(
    out: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("fp8_sm70_dequantize_out")(out, qweight, scales, group_size)


if has_op("fp8_sm70_dequantize_out"):

    @register_fake("_sm70_C::fp8_sm70_dequantize_out")
    def _fp8_sm70_dequantize_out_fake(
        out: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> None:
        del out, qweight, scales, group_size
        return None


def nvfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 16,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    """NVFP4 (W4A16, group 16) weight prepare for TurboMind.

    qweight: uint8 [K, N], one e2m1 code per byte. scales: fp16 [K/16, N],
    with the global scale already folded in by the caller. Returns
    [tm_weight int32 [K, N/8], tm_scales fp16 [K/16, N], meta int64[2] on CPU
    = (k_ld, q_ld)] for nvfp4_gemm_sm70_out.
    """
    return _op("nvfp4_sm70_prepare")(
        qweight,
        scales,
        group_size,
        interleave_gated_silu,
    )


if has_op("nvfp4_sm70_prepare"):

    @register_fake("_sm70_C::nvfp4_sm70_prepare")
    def _nvfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int = 16,
        interleave_gated_silu: bool = False,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.float16,
            device=qweight.device,
        )
        meta = torch.empty(
            (2,), dtype=torch.int64, device="cpu"
        )  # C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


def sm70_f16_prepare(weight: torch.Tensor) -> list[torch.Tensor]:
    return _op("sm70_f16_prepare")(weight)


if has_op("sm70_f16_prepare"):

    @register_fake("_sm70_C::sm70_f16_prepare")
    def _sm70_f16_prepare_fake(weight: torch.Tensor) -> list[torch.Tensor]:
        meta = torch.empty(
            (1,), dtype=torch.int64, device="cpu"
        )  # C++ returns a CPU tensor
        return [torch.empty_like(weight), meta]


# registered only with VLLM_SM70_TM_AWQ
def awq_gemm_sm70(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    return _op("awq_gemm_sm70")(
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
    )


if has_op("awq_gemm_sm70"):

    @register_fake("_sm70_C::awq_gemm_sm70")
    def _awq_gemm_sm70_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> torch.Tensor:
        del scales, group_size, k_ld, q_ld
        return torch.empty(
            (input.size(0), qweight.size(1) * 8),
            dtype=input.dtype,
            device=input.device,
        )


# registered only with VLLM_SM70_TM_AWQ
def awq_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_gemm_sm70_out")(
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
    )


if has_op("awq_gemm_sm70_out"):

    @register_fake("_sm70_C::awq_gemm_sm70_out")
    def _awq_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
    preserve_default_partition: bool = False,
) -> None:
    if preserve_default_partition:
        _op("fp8_gemm_sm70_out")(
            out,
            input,
            qweight,
            scales,
            group_size,
            k_ld,
            q_ld,
            gated_silu,
            True,
        )
        return
    _op("fp8_gemm_sm70_out")(
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
    )


if has_op("fp8_gemm_sm70_out"):

    @register_fake("_sm70_C::fp8_gemm_sm70_out")
    def _fp8_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
        preserve_default_partition: bool = False,
    ) -> None:
        return None


def fp8_gemm_sm70_prefill_prescaled_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    prescaled_factors: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> None:
    _op("fp8_gemm_sm70_prefill_prescaled_out")(
        out,
        input,
        qweight,
        prescaled_factors,
        group_size,
        k_ld,
        q_ld,
    )


if has_op("fp8_gemm_sm70_prefill_prescaled_out"):

    @register_fake("_sm70_C::fp8_gemm_sm70_prefill_prescaled_out")
    def _fp8_gemm_sm70_prefill_prescaled_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        prescaled_factors: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> None:
        return None


def fp8_gemm_sm70_prescaled_m1_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    prescaled_factors: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> None:
    _op("fp8_gemm_sm70_prescaled_m1_out")(
        out,
        input,
        qweight,
        prescaled_factors,
        group_size,
        k_ld,
        q_ld,
    )


if has_op("fp8_gemm_sm70_prescaled_m1_out"):

    @register_fake("_sm70_C::fp8_gemm_sm70_prescaled_m1_out")
    def _fp8_gemm_sm70_prescaled_m1_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        prescaled_factors: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> None:
        return None


def fp8_gemm_sm70_prefill_dispatch_out(
    out: torch.Tensor,
    dense_weight_ptr: int,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool,
    min_prefill_m: int,
) -> None:
    _op("fp8_gemm_sm70_prefill_dispatch_out")(
        out,
        dense_weight_ptr,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
    )


if has_op("fp8_gemm_sm70_prefill_dispatch_out"):

    @register_fake("_sm70_C::fp8_gemm_sm70_prefill_dispatch_out")
    def _fp8_gemm_sm70_prefill_dispatch_out_fake(
        out: torch.Tensor,
        dense_weight_ptr: int,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
        min_prefill_m: int,
    ) -> None:
        del dense_weight_ptr
        return None


def nvfp4_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("nvfp4_gemm_sm70_out")(
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
    )


if has_op("nvfp4_gemm_sm70_out"):

    @register_fake("_sm70_C::nvfp4_gemm_sm70_out")
    def _nvfp4_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool = False,
    ) -> None:
        return None


# Flint NVFP4 M<=2 GEMV (sm70/csrc/turbomind/ops/nvfp4_tm_gemv_sm70.cu).
# Per-rank (K, N) that the kernel's launch table covers: the Flint ModelOpt
# W4A16 dense layers at TP2 (qkv, o, shared-expert gate/up and down,
# dense-MLP gate/up and down). The op falls back to nvfp4_gemm_sm70_out for
# every other shape, for M > 2, and for operands the GEMV can't read
# (non-16-byte-aligned or badly strided activations). Admission (which
# layers call this op) is in vllm_sm70/quant/nvfp4_linear.py (
# MODELOPT_NVFP4_GEMV_SHAPES); tests/ops/test_k11_nvfp4_tm_gemv_cpu.py keeps
# the three sets equal.
NVFP4_TM_GEMV_SHAPES = frozenset(
    {
        (2048, 2560),
        (2048, 2048),
        (2048, 1024),
        (512, 2048),
        (2048, 6144),
        (3072, 2048),
    }
)
NVFP4_TM_GEMV_MAX_ROWS = 2


def nvfp4_tm_gemv_takes_gemv(
    m: int, k: int, n: int, group_size: int, k_ld: int, q_ld: int
) -> bool:
    """True when nvfp4_tm_gemv_dispatch_sm70_out runs the GEMV, not TurboMind.

    Mirrors the C++ shape gate (nvfp4_tm_gemv_dispatch_sm70_out). It leaves out
    the operand checks (dtype, contiguity, 16-byte input alignment), which
    the op also applies.
    """
    return (
        1 <= m <= NVFP4_TM_GEMV_MAX_ROWS
        and group_size == 16
        and k_ld == 32 * k
        and q_ld == n
        and (k, n) in NVFP4_TM_GEMV_SHAPES
    )


def nvfp4_tm_gemv_dispatch_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    scale_shift: int = 14,
) -> None:
    """M <= 2 GEMV on the TurboMind NVFP4 operands, TurboMind otherwise."""
    _op("nvfp4_tm_gemv_dispatch_sm70_out")(
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        scale_shift,
    )


if has_op("nvfp4_tm_gemv_dispatch_sm70_out"):

    @register_fake("_sm70_C::nvfp4_tm_gemv_dispatch_sm70_out")
    def _nvfp4_tm_gemv_dispatch_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        scale_shift: int = 14,
    ) -> None:
        return None


def nvfp4_gemm_sm70_prescaled_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("nvfp4_gemm_sm70_prescaled_out")(
        out,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
    )


if has_op("nvfp4_gemm_sm70_prescaled_out"):

    @register_fake("_sm70_C::nvfp4_gemm_sm70_prescaled_out")
    def _nvfp4_gemm_sm70_prescaled_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_gemm_sm70_out_auto(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    _op("fp8_gemm_sm70_out_auto")(out, input, qweight, scales)


def fp8_gemm_sm70_out_meta(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    gated_silu: bool = False,
) -> None:
    _op("fp8_gemm_sm70_out_meta")(
        out,
        input,
        qweight,
        scales,
        meta,
        gated_silu,
    )


def sm70_f16_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _op("sm70_f16_gemm")(input, weight)


if has_op("sm70_f16_gemm"):

    @register_fake("_sm70_C::sm70_f16_gemm")
    def _sm70_f16_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (input.size(0), weight.size(0)),
            dtype=input.dtype,
            device=input.device,
        )


def sm70_f16_gemm_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("sm70_f16_gemm_out")(out, input, weight, k_ld, gated_silu)


if has_op("sm70_f16_gemm_out"):

    @register_fake("_sm70_C::sm70_f16_gemm_out")
    def _sm70_f16_gemm_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def sm70_gemm_import_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_import_cache")(device_hint, path)


def sm70_gemm_export_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_export_cache")(device_hint, path)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def nvfp4_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("nvfp4_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if has_op("nvfp4_moe_dense_stage_sm70_out"):

    @register_fake("_sm70_C::nvfp4_moe_dense_stage_sm70_out")
    def _nvfp4_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


# Python mirror of the launch form nvfp4_moe_dense_stage_sm70_out picks
# (sm70/csrc/turbomind/ops/tm_moe_sm70.cu); tests/quant/
# test_m12_inkling2_grouped_prefill_cpu.py keeps it in step with the C++.
NVFP4_MOE_COMPACT_MAX_GROUPS = 80
NVFP4_MOE_GROUPED_MIN_ROWS = 65
# num_experts -> (k, n) stage shapes of the grouped prefill.
NVFP4_MOE_GROUPED_PREFILL_SHAPES = {
    128: frozenset({(2304, 1152), (576, 2304)}),  # Inkling 2 Mini, TP2
    256: frozenset(
        {(2048, 1024), (2048, 512), (2048, 256), (512, 2048), (256, 2048)}
        | {(128, 2048)}
    ),  # Qwen3.6 and Flint
    512: frozenset({(2560, 320), (160, 2560)}),  # Qwen4 experts
}
INKLING_GROUPED_PREFILL_SWITCH = "VLLM_SM70_INKLING_TM_GROUPED_PREFILL"


def nvfp4_moe_dense_stage_route(rows: int, num_experts: int, k: int, n: int) -> str:
    """Launch form of one nvfp4_moe_dense_stage_sm70_out call on `rows`
    input rows: "compact" or "grouped" (one launch each) or "per_expert"
    (num_experts launches). The knobs are read as the C++ reads them:
    VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL by atoi (default 1),
    VLLM_SM70_INKLING_TM_GROUPED_PREFILL off only at "0"."""
    import os

    from vllm_sm70.warmup.plan import knob

    if rows == num_experts and num_experts <= NVFP4_MOE_COMPACT_MAX_GROUPS:
        return "compact"
    inkling_on = os.environ.get(INKLING_GROUPED_PREFILL_SWITCH) != "0"
    if (
        rows >= NVFP4_MOE_GROUPED_MIN_ROWS
        and (k, n) in NVFP4_MOE_GROUPED_PREFILL_SHAPES.get(num_experts, ())
        and knob("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", 1) != 0
        and (num_experts != 128 or inkling_on)
    ):
        return "grouped"
    return "per_expert"


def awq_moe_build_strided_ptrs(
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
    num_experts: int,
) -> list[torch.Tensor]:
    return _op("awq_moe_build_strided_ptrs")(
        tm_weights,
        tm_scales,
        k_ld,
        q_ld,
        num_experts,
    )


if has_op("awq_moe_build_strided_ptrs"):

    @register_fake("_sm70_C::awq_moe_build_strided_ptrs")
    def _awq_moe_build_strided_ptrs_fake(
        tm_weights: torch.Tensor,
        tm_scales: torch.Tensor,
        k_ld: int,
        q_ld: int,
        num_experts: int,
    ) -> list[torch.Tensor]:
        del tm_scales, k_ld, q_ld
        buf = num_experts * 16
        opts = dict(dtype=torch.uint8, device=tm_weights.device)
        return [torch.empty(buf, **opts), torch.empty(buf, **opts)]


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_gemm_sm70_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_moe_gemm_sm70_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_gemm_sm70_per_expert_dispatch_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_moe_gemm_sm70_per_expert_dispatch_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if has_op("awq_moe_gemm_sm70_out"):

    @register_fake("_sm70_C::awq_moe_gemm_sm70_out")
    def _awq_moe_gemm_sm70_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


if has_op("awq_moe_gemm_sm70_per_expert_dispatch_out"):

    @register_fake("_sm70_C::awq_moe_gemm_sm70_per_expert_dispatch_out")
    def _awq_moe_gemm_sm70_per_expert_dispatch_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if has_op("awq_moe_dense_stage_sm70_out"):

    @register_fake("_sm70_C::awq_moe_dense_stage_sm70_out")
    def _awq_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_active_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    permuted_experts_id: torch.Tensor,
    active_expert_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    total_slots: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_active_dense_stage_sm70_out")(
        out,
        input,
        permuted_experts_id,
        active_expert_offsets,
        active_expert_ids,
        ptrs_w,
        ptrs_s,
        total_slots,
        k,
        n,
        group_size,
    )


if has_op("awq_moe_active_dense_stage_sm70_out"):

    @register_fake("_sm70_C::awq_moe_active_dense_stage_sm70_out")
    def _awq_moe_active_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        permuted_experts_id: torch.Tensor,
        active_expert_offsets: torch.Tensor,
        active_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        total_slots: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_chunked_w2_sm70_out(
    out: torch.Tensor,
    chunk_output: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    permuted_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    chunk_expert_offsets: torch.Tensor,
    chunk_range_begin: torch.Tensor,
    chunk_range_end: torch.Tensor,
    chunk_a_indices: torch.Tensor,
    chunk_inv_permuted_idx: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    k: int,
    n: int,
    hidden_logical_size: int,
    group_size: int,
    chunk_tokens: int,
) -> None:
    _op("awq_moe_chunked_w2_sm70_out")(
        out,
        chunk_output,
        input,
        expert_offsets,
        permuted_idx,
        topk_weights,
        chunk_expert_offsets,
        chunk_range_begin,
        chunk_range_end,
        chunk_a_indices,
        chunk_inv_permuted_idx,
        ptrs_w,
        ptrs_s,
        num_tokens,
        top_k,
        num_experts,
        k,
        n,
        hidden_logical_size,
        group_size,
        chunk_tokens,
    )


if has_op("awq_moe_chunked_w2_sm70_out"):

    @register_fake("_sm70_C::awq_moe_chunked_w2_sm70_out")
    def _awq_moe_chunked_w2_sm70_out_fake(
        out: torch.Tensor,
        chunk_output: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        permuted_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        chunk_expert_offsets: torch.Tensor,
        chunk_range_begin: torch.Tensor,
        chunk_range_end: torch.Tensor,
        chunk_a_indices: torch.Tensor,
        chunk_inv_permuted_idx: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        k: int,
        n: int,
        hidden_logical_size: int,
        group_size: int,
        chunk_tokens: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_single_token_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_single_token_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if has_op("awq_moe_single_token_dense_stage_sm70_out"):

    @register_fake("_sm70_C::awq_moe_single_token_dense_stage_sm70_out")
    def _awq_moe_single_token_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_single_token_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_single_token_indexed_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if has_op("awq_moe_single_token_indexed_dense_stage_sm70_out"):

    @register_fake("_sm70_C::awq_moe_single_token_indexed_dense_stage_sm70_out")
    def _awq_moe_single_token_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_single_token_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if has_op("awq_moe_single_token_dense_w13_sm70_out"):

    @register_fake("_sm70_C::awq_moe_single_token_dense_w13_sm70_out")
    def _awq_moe_single_token_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_single_token_indexed_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_indexed_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if has_op("awq_moe_single_token_indexed_dense_w13_sm70_out"):

    @register_fake("_sm70_C::awq_moe_single_token_indexed_dense_w13_sm70_out")
    def _awq_moe_single_token_indexed_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_single_token_compact_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    compact_w13_ptrs_w: torch.Tensor,
    compact_w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_compact_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        compact_w13_ptrs_w,
        compact_w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if has_op("awq_moe_single_token_compact_dense_w13_sm70_out"):

    @register_fake("_sm70_C::awq_moe_single_token_compact_dense_w13_sm70_out")
    def _awq_moe_single_token_compact_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        compact_w13_ptrs_w: torch.Tensor,
        compact_w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_single_token_exact_layout_prepare(
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    compact_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    num_experts: int,
) -> None:
    _op("awq_moe_single_token_exact_layout_prepare")(
        topk_ids,
        x,
        compact_input,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        num_experts,
    )


if has_op("awq_moe_single_token_exact_layout_prepare"):

    @register_fake("_sm70_C::awq_moe_single_token_exact_layout_prepare")
    def _awq_moe_single_token_exact_layout_prepare_fake(
        topk_ids: torch.Tensor,
        x: torch.Tensor,
        compact_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        num_experts: int,
    ) -> None:
        return None


def awq_moe_single_token_weighted_reduce_out(
    sorted_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_weighted_reduce_out")(
        sorted_output,
        topk_weights,
        inv_permuted_idx,
        out,
        top_k,
        hidden_logical_size,
    )


if has_op("awq_moe_single_token_weighted_reduce_out"):

    @register_fake("_sm70_C::awq_moe_single_token_weighted_reduce_out")
    def _awq_moe_single_token_weighted_reduce_out_fake(
        sorted_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        out: torch.Tensor,
        top_k: int,
        hidden_logical_size: int,
    ) -> None:
        return None


# registered only with VLLM_SM70_TM_AWQ
def awq_moe_single_token_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_w13_ptrs_w_rows: torch.Tensor,
    src_w13_ptrs_s_rows: torch.Tensor,
    src_w2_ptrs_w_rows: torch.Tensor,
    src_w2_ptrs_s_rows: torch.Tensor,
    compact_input: torch.Tensor,
    intermediate: torch.Tensor,
    sorted_output: torch.Tensor,
    sorted_weights: torch.Tensor,
    dst_w13_ptrs_w_rows: torch.Tensor,
    dst_w13_ptrs_s_rows: torch.Tensor,
    dst_w2_ptrs_w_rows: torch.Tensor,
    dst_w2_ptrs_s_rows: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    w13_k: int,
    w13_n: int,
    w2_k: int,
    w2_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_sm70_out")(
        out,
        x,
        topk_weights,
        topk_ids,
        src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows,
        src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows,
        compact_input,
        intermediate,
        sorted_output,
        sorted_weights,
        dst_w13_ptrs_w_rows,
        dst_w13_ptrs_s_rows,
        dst_w2_ptrs_w_rows,
        dst_w2_ptrs_s_rows,
        expert_offsets,
        inv_permuted_idx,
        w13_k,
        w13_n,
        w2_k,
        w2_n,
        group_size,
        hidden_logical_size,
    )


if has_op("awq_moe_single_token_sm70_out"):

    @register_fake("_sm70_C::awq_moe_single_token_sm70_out")
    def _awq_moe_single_token_sm70_out_fake(
        out: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        src_w13_ptrs_w_rows: torch.Tensor,
        src_w13_ptrs_s_rows: torch.Tensor,
        src_w2_ptrs_w_rows: torch.Tensor,
        src_w2_ptrs_s_rows: torch.Tensor,
        compact_input: torch.Tensor,
        intermediate: torch.Tensor,
        sorted_output: torch.Tensor,
        sorted_weights: torch.Tensor,
        dst_w13_ptrs_w_rows: torch.Tensor,
        dst_w13_ptrs_s_rows: torch.Tensor,
        dst_w2_ptrs_w_rows: torch.Tensor,
        dst_w2_ptrs_s_rows: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        w13_k: int,
        w13_n: int,
        w2_k: int,
        w2_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        del (
            out,
            x,
            topk_weights,
            topk_ids,
            src_w13_ptrs_w_rows,
            src_w13_ptrs_s_rows,
            src_w2_ptrs_w_rows,
            src_w2_ptrs_s_rows,
            compact_input,
            intermediate,
            sorted_output,
            dst_w13_ptrs_w_rows,
            dst_w13_ptrs_s_rows,
            dst_w2_ptrs_w_rows,
            dst_w2_ptrs_s_rows,
            expert_offsets,
            inv_permuted_idx,
            w13_k,
            w13_n,
            w2_k,
            w2_n,
            group_size,
            hidden_logical_size,
        )
        return None


def fp8_moe_gemm_sm70_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_moe_gemm_sm70_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


def fp8_moe_gemm_sm70_per_expert_dispatch_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_moe_gemm_sm70_per_expert_dispatch_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if has_op("fp8_moe_gemm_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_gemm_sm70_out")
    def _fp8_moe_gemm_sm70_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


if has_op("fp8_moe_gemm_sm70_per_expert_dispatch_out"):

    @register_fake("_sm70_C::fp8_moe_gemm_sm70_per_expert_dispatch_out")
    def _fp8_moe_gemm_sm70_per_expert_dispatch_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if has_op("fp8_moe_dense_stage_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_dense_stage_sm70_out")
    def _fp8_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_single_token_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if has_op("fp8_moe_single_token_dense_stage_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_single_token_dense_stage_sm70_out")
    def _fp8_moe_single_token_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_single_token_indexed_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if has_op("fp8_moe_single_token_indexed_dense_stage_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_single_token_indexed_dense_stage_sm70_out")
    def _fp8_moe_single_token_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if has_op("fp8_moe_single_token_dense_w13_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_single_token_dense_w13_sm70_out")
    def _fp8_moe_single_token_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_indexed_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_indexed_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if has_op("fp8_moe_single_token_indexed_dense_w13_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_single_token_indexed_dense_w13_sm70_out")
    def _fp8_moe_single_token_indexed_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_compact_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    compact_w13_ptrs_w: torch.Tensor,
    compact_w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_compact_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        compact_w13_ptrs_w,
        compact_w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if has_op("fp8_moe_single_token_compact_dense_w13_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_single_token_compact_dense_w13_sm70_out")
    def _fp8_moe_single_token_compact_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        compact_w13_ptrs_w: torch.Tensor,
        compact_w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_w13_ptrs_w_rows: torch.Tensor,
    src_w13_ptrs_s_rows: torch.Tensor,
    src_w2_ptrs_w_rows: torch.Tensor,
    src_w2_ptrs_s_rows: torch.Tensor,
    compact_input: torch.Tensor,
    gate_up: torch.Tensor,
    intermediate: torch.Tensor,
    sorted_output: torch.Tensor,
    sorted_weights: torch.Tensor,
    dst_w13_ptrs_w_rows: torch.Tensor,
    dst_w13_ptrs_s_rows: torch.Tensor,
    dst_w2_ptrs_w_rows: torch.Tensor,
    dst_w2_ptrs_s_rows: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    broadcast_input_indices: torch.Tensor,
    w2_raw_weight: torch.Tensor,
    w2_raw_scale_inv: torch.Tensor,
    w13_k: int,
    w13_n: int,
    w2_k: int,
    w2_n: int,
    group_size: int,
    hidden_logical_size: int,
    fused_gated_silu: bool,
    fused_weighted_reduce: bool,
    broadcast_input: bool,
    w2_direct_reduce: bool,
    indexed_expert_ptrs: bool,
    exact_per_route: bool,
) -> None:
    _op("fp8_moe_single_token_sm70_out")(
        out,
        x,
        topk_weights,
        topk_ids,
        src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows,
        src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows,
        compact_input,
        gate_up,
        intermediate,
        sorted_output,
        sorted_weights,
        dst_w13_ptrs_w_rows,
        dst_w13_ptrs_s_rows,
        dst_w2_ptrs_w_rows,
        dst_w2_ptrs_s_rows,
        expert_offsets,
        inv_permuted_idx,
        sorted_expert_ids,
        broadcast_input_indices,
        w2_raw_weight,
        w2_raw_scale_inv,
        w13_k,
        w13_n,
        w2_k,
        w2_n,
        group_size,
        hidden_logical_size,
        fused_gated_silu,
        fused_weighted_reduce,
        broadcast_input,
        w2_direct_reduce,
        indexed_expert_ptrs,
        exact_per_route,
    )


if has_op("fp8_moe_single_token_sm70_out"):

    @register_fake("_sm70_C::fp8_moe_single_token_sm70_out")
    def _fp8_moe_single_token_sm70_out_fake(
        out: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        src_w13_ptrs_w_rows: torch.Tensor,
        src_w13_ptrs_s_rows: torch.Tensor,
        src_w2_ptrs_w_rows: torch.Tensor,
        src_w2_ptrs_s_rows: torch.Tensor,
        compact_input: torch.Tensor,
        gate_up: torch.Tensor,
        intermediate: torch.Tensor,
        sorted_output: torch.Tensor,
        sorted_weights: torch.Tensor,
        dst_w13_ptrs_w_rows: torch.Tensor,
        dst_w13_ptrs_s_rows: torch.Tensor,
        dst_w2_ptrs_w_rows: torch.Tensor,
        dst_w2_ptrs_s_rows: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        broadcast_input_indices: torch.Tensor,
        w2_raw_weight: torch.Tensor,
        w2_raw_scale_inv: torch.Tensor,
        w13_k: int,
        w13_n: int,
        w2_k: int,
        w2_n: int,
        group_size: int,
        hidden_logical_size: int,
        fused_gated_silu: bool,
        fused_weighted_reduce: bool,
        broadcast_input: bool,
        w2_direct_reduce: bool,
        indexed_expert_ptrs: bool,
        exact_per_route: bool,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Further fakes, so that every _sm70_C op schema has one.
# ---------------------------------------------------------------------------

if has_op("fp8_gemm_sm70_out_auto"):

    @register_fake("_sm70_C::fp8_gemm_sm70_out_auto")
    def _fp8_gemm_sm70_out_auto_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        return None


if has_op("fp8_gemm_sm70_out_meta"):

    @register_fake("_sm70_C::fp8_gemm_sm70_out_meta")
    def _fp8_gemm_sm70_out_meta_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        meta: torch.Tensor,
        gated_silu: bool,
    ) -> None:
        return None


if has_op("sm70_gemm_import_cache"):

    @register_fake("_sm70_C::sm70_gemm_import_cache")
    def _sm70_gemm_import_cache_fake(device_hint: torch.Tensor, path: str) -> int:
        return 0


if has_op("sm70_gemm_export_cache"):

    @register_fake("_sm70_C::sm70_gemm_export_cache")
    def _sm70_gemm_export_cache_fake(device_hint: torch.Tensor, path: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# Runtime ops (gemm_runtime.cpp).
# ---------------------------------------------------------------------------


def sm70_gemm_configure_workspace(nbytes: int) -> None:
    """Set the TurboMind split-K scratch size (1..32 MiB, 256-aligned)."""
    _op("sm70_gemm_configure_workspace")(nbytes)


def sm70_gemm_workspace_storage() -> list[int]:
    """[number of stream workspaces, scratch bytes each, total bytes]."""
    return list(_op("sm70_gemm_workspace_storage")())


def tm_runtime_close() -> None:
    """Free all TurboMind state (see gemm_runtime.cpp TurboMindRuntime::release).

    Syncs every device that has TurboMind workspaces, Gemm objects or
    prepared fp16 weights, then frees them and forgets tuning results. Call
    it at shutdown, after every CUDA graph that replays TurboMind kernels is
    destroyed. Raises if called during CUDA graph capture. Safe to call more
    than once; later TurboMind calls start again from scratch.
    """
    _op("tm_runtime_close")()


# No fakes for ops without Tensor arguments (sm70_gemm_configure_workspace,
# sm70_gemm_workspace_storage, tm_runtime_close). They run eagerly at load,
# warmup or shutdown and are never traced, and torch refuses a meta kernel on
# an op whose C++ impl is CompositeImplicitAutograd (the default for
# m.def(schema, fn)). tests/ops checks this rule.
NO_FAKE_OPS = frozenset(
    {
        "sm70_gemm_configure_workspace",
        "sm70_gemm_workspace_storage",
        "tm_runtime_close",
    }
)
