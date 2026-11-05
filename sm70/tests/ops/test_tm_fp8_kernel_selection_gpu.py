# SPDX-License-Identifier: Apache-2.0
"""Kernel-level FP8 route checks (the C++ side of fork test_sm70_fp8_kernel_selection).

The fork test (tests/quantization/test_sm70_fp8_kernel_selection.py,
c9d825d5c) is about Python kernel selection and KernelConfig, which is
Mara's M-side work. What K03 owns is the C++ routing inside the ops:
- the pre-scaled block-FP8 entry points accept only their exact shapes
  (fork awq_sm70_gemm.cu:4288-4300);
- fp8_gemm_sm70_prefill_dispatch_out below min_prefill_m is the plain
  TurboMind GEMM, bit for bit, and above it is the dequantize + dense GEMM
  path (CUTLASS for Qwen3.8 8K prefill, else torch.mm);
- fp8_gemm_sm70_out_meta reads (k_ld, q_ld) from the prepare meta.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _ref import fp8_block_dequant, fp8_block_weight, max_abs_rel  # noqa: E402

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")


def _prepared(tm, n, k, gated=False, seed=0):
    q, scales = fp8_block_weight(n, k, seed=seed)
    w, s, meta = tm.fp8_sm70_prepare(q, scales, 128, gated)
    return q, scales, w, s, int(meta[0]), int(meta[1]), meta


def test_prescaled_rejects_unlisted_shapes(tm_gpu):
    _, _, w, s, k_ld, q_ld, _ = _prepared(tm_gpu, 512, 512)
    x = torch.randn(1, 512, device="cuda", dtype=torch.float16)
    out = torch.empty(1, 512, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="pre-scaled block-FP8 requires"):
        tm_gpu.fp8_gemm_sm70_prescaled_m1_out(out, x, w, s, 128, k_ld, q_ld)
    x2 = torch.randn(2, 512, device="cuda", dtype=torch.float16)
    out2 = torch.empty(2, 512, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="M=1"):
        tm_gpu.fp8_gemm_sm70_prescaled_m1_out(out2, x2, w, s, 128, k_ld, q_ld)


@pytest.mark.parametrize("gated", [False, True])
def test_prefill_dispatch_routes(tm_gpu, gated):
    n, k, min_m = 1024, 2048, 64
    q, scales, w, s, k_ld, q_ld, _ = _prepared(tm_gpu, n, k, gated, seed=3)
    width = n // 2 if gated else n
    workspace = torch.empty(k, n, device="cuda", dtype=torch.float16)
    for m in (8, 63, 64, 300):
        x = (torch.randn(m, k, device="cuda") * 0.1).half()
        out = torch.empty(m, width, device="cuda", dtype=torch.float16)
        tm_gpu.fp8_gemm_sm70_prefill_dispatch_out(
            out, workspace.data_ptr(), x, w, s, 128, k_ld, q_ld, gated, min_m
        )
        if m < min_m:
            plain = torch.empty_like(out)
            tm_gpu.fp8_gemm_sm70_out(plain, x, w, s, 128, k_ld, q_ld, gated)
            assert torch.equal(out.view(torch.int16), plain.view(torch.int16))
        else:
            if not gated:
                # The dense path dequantizes into the caller's workspace [K, N].
                ref_w = fp8_block_dequant(q, scales).T
                assert max_abs_rel(workspace, ref_w) < 1e-2
            ref = x.float() @ fp8_block_dequant(q, scales).T
            if gated:
                ref = torch.nn.functional.silu(ref[:, : n // 2]) * ref[:, n // 2 :]
            assert max_abs_rel(out, ref) < 1e-2


def test_out_meta_matches_explicit_ld(tm_gpu):
    n, k = 1024, 2048
    _, _, w, s, k_ld, q_ld, meta = _prepared(tm_gpu, n, k, seed=5)
    x = (torch.randn(16, k, device="cuda") * 0.1).half()
    a = torch.empty(16, n, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    tm_gpu.fp8_gemm_sm70_out(a, x, w, s, 128, k_ld, q_ld, False)
    tm_gpu.fp8_gemm_sm70_out_meta(b, x, w, s, meta, False)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_dequantize_matches_reference(tm_gpu):
    n, k = 1024, 2048
    q, scales, w, s, _, _, _ = _prepared(tm_gpu, n, k, seed=9)
    out = torch.empty(k, n, device="cuda", dtype=torch.float16)
    tm_gpu.fp8_sm70_dequantize_out(out, w, s, 128)
    assert max_abs_rel(out, fp8_block_dequant(q, scales).T) < 1e-2
