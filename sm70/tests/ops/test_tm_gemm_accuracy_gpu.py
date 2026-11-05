# SPDX-License-Identifier: Apache-2.0
"""K03 GPU accuracy: TurboMind fp16 / FP8-block / NVFP4 GEMM vs torch.matmul.

The reference is fp32 torch.matmul on the dequantized weight. The pass bar is
max|out - ref| / max|ref| < 1e-2 at M in {1, 2, 8, 64, 512}, on Flint (NVFP4)
and Qwen3.6/3.8 TP2 (FP8 block-128, fp16 DFlash2 draft) projection shapes.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _ref import (  # noqa: E402
    fp8_block_dequant,
    fp8_block_weight,
    max_abs_rel,
    nvfp4_dequant,
    nvfp4_weight,
)

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")

MS = (1, 2, 8, 64, 512)
TOL = 1e-2

# (N, K). NVFP4: the Flint shapes that have exact-MNK TurboMind kernels
# (sm70_884_4.cu ExactMnkKernelImpl 8704x5120 and 5120x4352).
NVFP4_SHAPES = [(8704, 5120), (5120, 4352)]
# FP8 block-128: Qwen3.8-27B at TP2 (hidden 5120, intermediate 17408):
# gate_up, down, and a 4096-wide attention projection.
FP8_SHAPES = [(17408, 5120), (5120, 8704), (4096, 5120)]
# fp16: DFlash2 draft projections (hidden 5120).
F16_SHAPES = [(5120, 5120), (4096, 5120)]


def _x(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, k, device="cuda", generator=g) * 0.1).half()


@pytest.mark.parametrize("n,k", NVFP4_SHAPES)
@pytest.mark.parametrize("m", MS)
def test_nvfp4_gemm_matches_matmul(tm_gpu, n, k, m):
    codes, scales = nvfp4_weight(k, n, seed=n + k)
    w, s, meta = tm_gpu.nvfp4_sm70_prepare(codes, scales)
    assert w.shape == (k, n // 8) and w.dtype == torch.int32
    assert meta.device.type == "cpu" and meta.shape == (2,)
    x = _x(m, k, seed=m)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    tm_gpu.nvfp4_gemm_sm70_out(out, x, w, s, 16, int(meta[0]), int(meta[1]))
    ref = x.float() @ nvfp4_dequant(codes, scales)
    assert max_abs_rel(out, ref) < TOL


def test_nvfp4_prepare_meta_contract(tm_gpu):
    """Mara's K03 contract: meta = (k_ld = 32 * K, q_ld = N)."""
    k, n = 5120, 4352
    codes, scales = nvfp4_weight(k, n)
    _, _, meta = tm_gpu.nvfp4_sm70_prepare(codes, scales)
    assert meta.tolist() == [32 * k, n]


@pytest.mark.parametrize("m", (1, 8, 512))
def test_nvfp4_gated_silu(tm_gpu, m):
    n, k = 2 * 4352, 5120
    codes, scales = nvfp4_weight(k, n, seed=7)
    w, s, meta = tm_gpu.nvfp4_sm70_prepare(codes, scales, 16, True)
    x = _x(m, k, seed=m + 1)
    out = torch.empty(m, n // 2, device="cuda", dtype=torch.float16)
    tm_gpu.nvfp4_gemm_sm70_out(out, x, w, s, 16, int(meta[0]), int(meta[1]), True)
    gate_up = x.float() @ nvfp4_dequant(codes, scales)
    gate, up = gate_up[:, : n // 2], gate_up[:, n // 2 :]
    ref = torch.nn.functional.silu(gate) * up
    assert max_abs_rel(out, ref) < TOL


@pytest.mark.parametrize("n,k", FP8_SHAPES)
@pytest.mark.parametrize("m", MS)
def test_fp8_block_gemm_matches_matmul(tm_gpu, n, k, m):
    q, scales = fp8_block_weight(n, k, seed=n + k)
    w, s, meta = tm_gpu.fp8_sm70_prepare(q, scales, 128, False)
    assert w.shape == (k, n) and w.dtype == torch.uint8
    x = _x(m, k, seed=m)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    tm_gpu.fp8_gemm_sm70_out(out, x, w, s, 128, int(meta[0]), int(meta[1]), False)
    ref = x.float() @ fp8_block_dequant(q, scales).T
    assert max_abs_rel(out, ref) < TOL


@pytest.mark.parametrize("n,k", F16_SHAPES)
@pytest.mark.parametrize("m", MS)
def test_f16_gemm_matches_matmul(tm_gpu, n, k, m):
    g = torch.Generator(device="cuda").manual_seed(n + k)
    weight = (torch.randn(n, k, device="cuda", generator=g) * 0.05).half()
    tm_weight, meta = tm_gpu.sm70_f16_prepare(weight)
    x = _x(m, k, seed=m)
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    tm_gpu.sm70_f16_gemm_out(out, x, tm_weight, int(meta[0]), False)
    ref = x.float() @ weight.float().T
    assert max_abs_rel(out, ref) < TOL
    # The allocating variant gives the same answer.
    assert max_abs_rel(tm_gpu.sm70_f16_gemm(x, weight), ref) < TOL
