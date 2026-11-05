# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboMind SM70 ops (K03): Python wrappers and fake impls over torch.ops._sm70_C.

Ported from fork cf9492a vllm/_sm70/{auxiliary,linear,moe}.py. The fork's
`native_policy` argument and `call_native()` indirection are gone (knobs are
read in C++, sm70/csrc/turbomind/ops/tm_config.h), and `_C::` became
`_sm70_C::`. Each fork function is marked with its source lines; see
sm70/csrc/turbomind/PROVENANCE.md for the full move list.

Importing this module registers the fakes for every op that the loaded
`_sm70_C` provides (AWQ ops exist only in a VLLM_SM70_TM_AWQ build).
Wrappers raise RuntimeError when their op is missing.
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
# fp16 helpers: from fork cf9492a vllm/_sm70/auxiliary.py
# ---------------------------------------------------------------------------


# fork vllm/_sm70/auxiliary.py:10-11
def silu_and_mul_interleaved(out: torch.Tensor, input: torch.Tensor) -> None:
    _op("silu_and_mul_interleaved")(out, input)


# fork vllm/_sm70/auxiliary.py:14-19
if has_op("silu_and_mul_interleaved"):

    @register_fake("_sm70_C::silu_and_mul_interleaved")
    def _silu_and_mul_interleaved_fake(out: torch.Tensor, input: torch.Tensor) -> None:
        del out, input
        return None


# fork vllm/_sm70/auxiliary.py:706-711
def sm70_f16_gate_mul_out(
    out: torch.Tensor,
    input: torch.Tensor,
    gate_weight: torch.Tensor,
) -> None:
    _op("sm70_f16_gate_mul_out")(out, input, gate_weight)


# fork vllm/_sm70/auxiliary.py:714-722
if has_op("sm70_f16_gate_mul_out"):

    @register_fake("_sm70_C::sm70_f16_gate_mul_out")
    def _sm70_f16_gate_mul_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> None:
        return None

# ---------------------------------------------------------------------------
# dense GEMM: from fork cf9492a vllm/_sm70/linear.py
# ---------------------------------------------------------------------------


# fork vllm/_sm70/linear.py:78-94  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/linear.py:97-118
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
        )  # K03: C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# fork vllm/_sm70/linear.py:121-137  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/linear.py:140-161
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
        )  # K03: C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# fork vllm/_sm70/linear.py:164-173  # K04: registered only with VLLM_SM70_TM_AWQ
def awq_sm70_dequantize_out(
    out: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("awq_sm70_dequantize_out")(out, qweight, scales, group_size)


# fork vllm/_sm70/linear.py:176-187
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


# fork vllm/_sm70/linear.py:190-206  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/linear.py:209-235
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
        )  # K03: C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# fork vllm/_sm70/linear.py:238-252
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


# fork vllm/_sm70/linear.py:255-276
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
        )  # K03: C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# fork vllm/_sm70/linear.py:279-288
def fp8_sm70_dequantize_out(
    out: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> None:
    _op("fp8_sm70_dequantize_out")(out, qweight, scales, group_size)


# fork vllm/_sm70/linear.py:291-302
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


# fork vllm/_sm70/linear.py:350-364
def nvfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 16,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    """NVFP4 (W4A16, group 16) weight prepare for TurboMind (K03 contract).

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


# fork vllm/_sm70/linear.py:367-392
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
        )  # K03: C++ returns a CPU tensor
        return [tm_weight, tm_scales, meta]


# fork vllm/_sm70/linear.py:395-398
def sm70_f16_prepare(weight: torch.Tensor) -> list[torch.Tensor]:
    return _op("sm70_f16_prepare")(weight)


# fork vllm/_sm70/linear.py:401-408
if has_op("sm70_f16_prepare"):

    @register_fake("_sm70_C::sm70_f16_prepare")
    def _sm70_f16_prepare_fake(weight: torch.Tensor) -> list[torch.Tensor]:
        meta = torch.empty(
            (1,), dtype=torch.int64, device="cpu"
        )  # K03: C++ returns a CPU tensor
        return [torch.empty_like(weight), meta]


# fork vllm/_sm70/linear.py:432-450  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/linear.py:453-470
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


# fork vllm/_sm70/linear.py:473-495  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/linear.py:498-512
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


# fork vllm/_sm70/linear.py:572-610
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


# fork vllm/_sm70/linear.py:613-628
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


# fork vllm/_sm70/linear.py:631-651
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


# fork vllm/_sm70/linear.py:654-667
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


# fork vllm/_sm70/linear.py:670-690
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


# fork vllm/_sm70/linear.py:693-706
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


# fork vllm/_sm70/linear.py:1672-1698
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


# fork vllm/_sm70/linear.py:1701-1718
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


# fork vllm/_sm70/linear.py:1763-1785
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


# fork vllm/_sm70/linear.py:1788-1802
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


# fork vllm/_sm70/linear.py:1848-1870
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


# fork vllm/_sm70/linear.py:1873-1887
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


# fork vllm/_sm70/linear.py:2001-2010
def fp8_gemm_sm70_out_auto(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    _op("fp8_gemm_sm70_out_auto")(out, input, qweight, scales)


# fork vllm/_sm70/linear.py:2013-2031
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


# fork vllm/_sm70/linear.py:2034-2037
def sm70_f16_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _op("sm70_f16_gemm")(input, weight)


# fork vllm/_sm70/linear.py:2040-2052
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


# fork vllm/_sm70/linear.py:2055-2065
def sm70_f16_gemm_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("sm70_f16_gemm_out")(out, input, weight, k_ld, gated_silu)


# fork vllm/_sm70/linear.py:2068-2079
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


# fork vllm/_sm70/linear.py:2103-2106
def sm70_gemm_import_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_import_cache")(device_hint, path)


# fork vllm/_sm70/linear.py:2109-2112
def sm70_gemm_export_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_export_cache")(device_hint, path)


# ---------------------------------------------------------------------------
# MoE: from fork cf9492a vllm/_sm70/moe.py
# ---------------------------------------------------------------------------


# fork vllm/_sm70/moe.py:837-863
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


# fork vllm/_sm70/moe.py:866-882
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


# fork vllm/_sm70/moe.py:1083-1099
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


# fork vllm/_sm70/moe.py:1102-1116
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


# fork vllm/_sm70/moe.py:1119-1145  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1148-1174  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1177-1193
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


# fork vllm/_sm70/moe.py:1196-1212
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


# fork vllm/_sm70/moe.py:1215-1242  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1245-1261
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


# fork vllm/_sm70/moe.py:1316-1345  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1348-1365
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


# fork vllm/_sm70/moe.py:1368-1417
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


# fork vllm/_sm70/moe.py:1420-1447
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


# fork vllm/_sm70/moe.py:1450-1477  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1480-1496
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


# fork vllm/_sm70/moe.py:1499-1526  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1529-1545
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


# fork vllm/_sm70/moe.py:1548-1583  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1586-1606
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


# fork vllm/_sm70/moe.py:1609-1644  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1647-1667
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


# fork vllm/_sm70/moe.py:1670-1709  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1712-1734
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


# fork vllm/_sm70/moe.py:1737-1758
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


# fork vllm/_sm70/moe.py:1761-1774
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


# fork vllm/_sm70/moe.py:1777-1796
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


# fork vllm/_sm70/moe.py:1799-1811
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


# fork vllm/_sm70/moe.py:1860-1915  # K04: registered only with VLLM_SM70_TM_AWQ
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


# fork vllm/_sm70/moe.py:1918-1973
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


# fork vllm/_sm70/moe.py:1976-2002
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


# fork vllm/_sm70/moe.py:2005-2031
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


# fork vllm/_sm70/moe.py:2034-2050
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


# fork vllm/_sm70/moe.py:2053-2069
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


# fork vllm/_sm70/moe.py:2072-2099
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


# fork vllm/_sm70/moe.py:2102-2118
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


# fork vllm/_sm70/moe.py:2121-2148
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


# fork vllm/_sm70/moe.py:2151-2167
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


# fork vllm/_sm70/moe.py:2170-2197
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


# fork vllm/_sm70/moe.py:2200-2216
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


# fork vllm/_sm70/moe.py:2219-2254
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


# fork vllm/_sm70/moe.py:2257-2277
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


# fork vllm/_sm70/moe.py:2280-2315
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


# fork vllm/_sm70/moe.py:2318-2338
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


# fork vllm/_sm70/moe.py:2341-2380
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


# fork vllm/_sm70/moe.py:2383-2405
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


# fork vllm/_sm70/moe.py:2408-2485
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


# fork vllm/_sm70/moe.py:2488-2529
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
# Fakes the fork did not have, so that every _sm70_C op schema has one.
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
# Runtime ops (K02 gemm_runtime.cpp; tm_runtime_close is new in K03).
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
