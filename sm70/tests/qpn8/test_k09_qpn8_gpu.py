# SPDX-License-Identifier: Apache-2.0
"""K09 GPU checks for the QPN8 block-FP8 GEMV on SM70 (V100).

What:
- Port of fork tests/kernels/quantization/test_sm70_qpn8_block_fp8.py on
  vllm_sm70.ops.qpn8 (qpn8_prepare + qpn8_linear instead of the fork's
  QPN8Fp8BlockScaledMMLinearKernel): output vs the dequantized reference
  (rel-L2 < 1e-3) for M in {0, 1, 6, 8, 9, 20, 64, 512}, with and without
  the TurboMind FP8 copy; M>32 with the copy equals TurboMind FP8 bit for
  bit; without it, the dequantize + mm path, with or without a caller
  scratch; two streams with their own per-call scratch.
- With VLLM_SM70_FP8_QPN8_M32 unset (default) and =1 (opt-in QPN8 M9..32):
  accuracy at the Qwen3.6 / Qwen3.8 TP2 dense shapes (qkv, o, GDN in/out,
  MLP / shared expert), M in {1, 2, 4, 8, 9, 16, 17, 24, 32} (M 9..32 is the
  block-scaled two-row-tile path, DFlash2 k=7 verify for 2-4 requests): max|y - ref| / max|ref| < 1e-2
  and rel-L2 < 1e-3 against x @ dequant(W)^T in float32.
- Kernel-level parts of fork tests/quantization/
  test_block_fp8_qpn8_capabilities.py: unaligned shapes are refused,
  prepare/dequantize round-trip, opcheck on the three ops, and CUDA graph
  capture of the GEMV at M=1, 16 and 32.

Run from the repo root: pytest -m gpu sm70/tests/qpn8
"""

from __future__ import annotations

import pytest
from _qpn8_helpers import TP2_SHAPES

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def _weight(n: int, k: int, scale_dtype=torch.float32, seed=None):
    """Random e4m3 weight + block scales and the float32 [N, K] reference
    (fork test_sm70_qpn8_block_fp8.py:39-55)."""
    gen = torch.Generator(device="cuda").manual_seed(n + k if seed is None else seed)
    raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=gen)
    # 0x7F/0xFF are e4m3 NaN codes; a checkpoint never stores them.
    raw[raw == 0x7F] = 0x7E
    raw[raw == 0xFF] = 0xFE
    weight = raw.view(torch.float8_e4m3fn)
    scales = (
        2.0 ** (torch.rand(n // 128, k // 128, device="cuda", generator=gen) * 3 - 12)
    ).float()
    scales = scales.to(scale_dtype)
    full = scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight, scales, weight.float() * full


def rel_l2(y, ref):
    return ((y.float() - ref).norm() / ref.norm()).item()


def max_abs_rel(y, ref):
    return ((y.float() - ref).abs().max() / ref.abs().max()).item()


_SCALE_DTYPES = [torch.float32]
if hasattr(torch, "float8_e8m0fnu"):
    _SCALE_DTYPES.append(torch.float8_e8m0fnu)


@pytest.mark.parametrize(
    ("n", "k"), [(128, 128), (768, 1408), (1536, 4096), (8192, 1024), (4096, 8192)]
)
@pytest.mark.parametrize("m", [0, 1, 6, 8, 9, 20, 64, 512])
@pytest.mark.parametrize("turbomind", [True, False], ids=["tm-copy", "one-layout"])
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_matches_dequantized_reference(qpn8_gpu, n, k, m, turbomind, scale_dtype):
    weight, scales, reference = _weight(n, k, scale_dtype)
    prepared = qpn8_gpu.qpn8_prepare(weight, scales, turbomind_fallback=turbomind)
    assert (prepared.tm_weight is not None) is turbomind
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    y = qpn8_gpu.qpn8_linear(x, prepared)
    assert y.shape == (m, n) and y.dtype == torch.float16
    if m:
        assert rel_l2(y, x.float() @ reference.t()) < 1e-3


@pytest.fixture
def m32(monkeypatch):
    def apply(on: bool):
        if on:
            monkeypatch.setenv("VLLM_SM70_FP8_QPN8_M32", "1")
        else:
            monkeypatch.delenv("VLLM_SM70_FP8_QPN8_M32", raising=False)

    apply(False)
    return apply


@pytest.mark.parametrize("route", ["default", "m32"])
@pytest.mark.parametrize("name", sorted(TP2_SHAPES))
@pytest.mark.parametrize("m", [1, 2, 4, 8, 9, 16, 17, 24, 32])
def test_tp2_dense_shapes_accuracy(qpn8_gpu, m32, route, name, m):
    # default: M>8 takes dequantize + mm; m32: the opt-in QPN8 M9..32 path.
    m32(route == "m32")
    assert qpn8_gpu.qpn8_max_m() == (32 if route == "m32" else 8)
    n, k = TP2_SHAPES[name]
    weight, scales, reference = _weight(n, k)
    prepared = qpn8_gpu.qpn8_prepare(weight, scales)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    y = qpn8_gpu.qpn8_linear(x, prepared)
    ref = x.float() @ reference.t()
    errors = {"max_abs_rel": max_abs_rel(y, ref), "rel_l2": rel_l2(y, ref)}
    assert errors["max_abs_rel"] < 1e-2, errors
    assert errors["rel_l2"] < 1e-3, errors
    # 3-D input and bias, as a linear layer sees it.
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    y3 = qpn8_gpu.qpn8_linear(x.view(1, m, k), prepared, bias=bias)
    torch.testing.assert_close(y3.view(m, n), y + bias, rtol=0, atol=0)


@pytest.mark.parametrize("m", [33, 64])
def test_m_gt_32_with_copy_is_turbomind(qpn8_gpu, m):
    from vllm_sm70.ops import tm

    weight, scales, _ = _weight(1536, 4096)
    prepared = qpn8_gpu.qpn8_prepare(weight, scales, turbomind_fallback=True)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.float16)
    expected = torch.empty((m, 1536), device="cuda", dtype=torch.float16)
    tm.fp8_gemm_sm70_out(
        expected,
        x,
        prepared.tm_weight,
        prepared.tm_scales,
        128,
        prepared.tm_k_ld,
        prepared.tm_q_ld,
        False,
    )
    torch.testing.assert_close(
        qpn8_gpu.qpn8_linear(x, prepared), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("m", [33, 64, 512])
def test_one_layout_uses_caller_scratch(qpn8_gpu, m):
    weight, scales, reference = _weight(1536, 4096)
    prepared = qpn8_gpu.qpn8_prepare(weight, scales)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.float16) * 0.1
    scratch = torch.empty(1536 * 4096, device="cuda", dtype=torch.float16)
    y = qpn8_gpu.qpn8_linear(x, prepared, dense_scratch=scratch)
    assert rel_l2(y, x.float() @ reference.t()) < 1e-3
    torch.testing.assert_close(
        scratch.view(4096, 1536).float(), reference.t(), rtol=1e-3, atol=1e-6
    )
    torch.testing.assert_close(qpn8_gpu.qpn8_linear(x, prepared), y, rtol=0, atol=0)


def test_concurrent_streams_keep_their_own_dense_buffer(qpn8_gpu):
    # fork test_concurrent_streams_keep_their_own_prefill_buffer (dense_only).
    wa, sa, ref_a = _weight(8192, 1024)
    wb, sb, ref_b = _weight(4096, 1024)
    pa = qpn8_gpu.qpn8_prepare(wa, sa)
    pb = qpn8_gpu.qpn8_prepare(wb, sb)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.float16) * 0.1
    aux = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    aux.wait_stream(main)
    for _ in range(20):
        with torch.cuda.stream(aux):
            y_b = qpn8_gpu.qpn8_linear(x, pb)
        y_a = qpn8_gpu.qpn8_linear(x, pa)
    main.wait_stream(aux)
    torch.cuda.synchronize()
    assert rel_l2(y_a, x.float() @ ref_a.t()) < 1e-3
    assert rel_l2(y_b, x.float() @ ref_b.t()) < 1e-3


@pytest.mark.parametrize("n, k", [(127, 128), (128, 129), (32, 2048)])
def test_unaligned_weights_are_refused(qpn8_gpu, n, k):
    weight = torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(max(n // 128, 1), max(k // 128, 1), device="cuda")
    with pytest.raises(ValueError, match="divisible by 128"):
        qpn8_gpu.qpn8_prepare(weight, scales)


def test_prepare_dequantize_round_trip(qpn8_gpu):
    weight, scales, reference = _weight(4608, 2048)
    codes, group_scales = qpn8_gpu.fp8_qpn8_prepare_sm70(weight, scales)
    assert codes.shape == (2048, 4608) and codes.dtype == torch.uint8
    assert group_scales.shape == (16, 144) and group_scales.dtype == torch.float16
    dense = torch.empty(2048, 4608, device="cuda", dtype=torch.float16)
    qpn8_gpu.fp8_qpn8_dequantize_sm70_out(dense, codes, group_scales)
    torch.testing.assert_close(dense.float(), reference.t(), rtol=1e-3, atol=1e-6)


def test_gemm_rejects_m_above_cap(qpn8_gpu):
    cap = qpn8_gpu.QPN8_MAX_M
    weight, scales, _ = _weight(256, 512)
    p = qpn8_gpu.qpn8_prepare(weight, scales)
    x = torch.zeros(cap + 1, 512, device="cuda", dtype=torch.float16)
    out = torch.empty(cap + 1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match=rf"M must be in \[1, {cap}\]"):
        qpn8_gpu.fp8_qpn8_gemm_sm70_out(out, x, p.codes, p.group_scales, 16, 2)


def test_opcheck(qpn8_gpu):
    weight, scales, _ = _weight(512, 1024)
    # prepare takes the e4m3 weight itself (its TORCH_CHECK rejects a uint8
    # view), and opcheck's test_schema compares inputs with torch.allclose,
    # which has no Float8_e4m3fn kernel ("mul_cuda" not implemented). So run
    # test_faketensor only, and check test_schema's point (no input is
    # mutated) bitwise on uint8 views.
    weight_before = weight.view(torch.uint8).clone()
    scales_before = scales.clone()
    torch.library.opcheck(
        torch.ops._sm70_C.fp8_qpn8_prepare_sm70.default,
        (weight, scales),
        test_utils=("test_faketensor",),
    )
    torch.ops._sm70_C.fp8_qpn8_prepare_sm70(weight, scales)
    assert torch.equal(weight.view(torch.uint8), weight_before)
    assert torch.equal(scales.view(torch.int32), scales_before.view(torch.int32))
    p = qpn8_gpu.qpn8_prepare(weight, scales)
    x = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
    out = torch.empty(4, 512, device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        torch.ops._sm70_C.fp8_qpn8_gemm_sm70_out.default,
        (out, x, p.codes, p.group_scales, p.split_k, p.accumulator_chains, True, False),
        test_utils=("test_schema", "test_faketensor"),
    )
    dense = torch.empty(1024, 512, device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        torch.ops._sm70_C.fp8_qpn8_dequantize_sm70_out.default,
        (dense, p.codes, p.group_scales),
        test_utils=("test_schema", "test_faketensor"),
    )
    torch.library.opcheck(
        torch.ops.vllm_sm70.qpn8_native_linear.default,
        (x, p.codes, p.group_scales, p.split_k, p.accumulator_chains, False),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize("m", [1, 16, 32])
def test_decode_replays_in_a_cuda_graph(qpn8_gpu, m32, m):
    m32(True)
    n, k = TP2_SHAPES["qwen3.8-mlp.down_proj"]
    weight, scales, reference = _weight(n, k)
    p = qpn8_gpu.qpn8_prepare(weight, scales)
    x = torch.zeros(m, k, device="cuda", dtype=torch.float16)
    qpn8_gpu.qpn8_linear(x, p)  # warm up outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = qpn8_gpu.qpn8_linear(x, p)
    x.copy_(torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1)
    graph.replay()
    torch.cuda.synchronize()
    assert rel_l2(y, x.float() @ reference.t()) < 1e-3
