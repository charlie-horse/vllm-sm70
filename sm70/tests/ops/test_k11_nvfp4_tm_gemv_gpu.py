# SPDX-License-Identifier: Apache-2.0
"""K11 GPU: the Flint NVFP4 M<=2 GEMV on the TurboMind operands.

Ported from fork cf9492a tests/kernels/quantization/test_sm70_nvfp4_tm_gemv.py
(0c9f3ee73). The fork prepared ModelOpt layers through its sm70_turbomind
provider. This port calls the ops directly (tm.nvfp4_sm70_prepare +
tm.nvfp4_tm_gemv_dispatch_sm70_out), with operands built the way M3 builds
them: global scale folded into the fp16 scales, one e2m1 code per byte.
The layer-level path is M3's GPU test. The reference dequantizes the codes
with the effective fp16 scales TurboMind uses: fp16(e4m3 * global).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FLINT_SHAPES = {
    "qkv": (2048, 2560),
    "o": (2048, 2048),
    "shared_gate_up": (2048, 1024),
    "shared_down": (512, 2048),
    "dense_gate_up": (2048, 6144),
    "dense_down": (3072, 2048),
}
GLOBAL_SCALE = 0.05 / (6.0 * 448.0)
_FP16_MAX = 65504.0


def _scale_shift(scales):
    # Same rule as M3's _tm_gemv_scale_shift (fork sm70_turbomind.py:583-594).
    peak = float(scales.abs().amax())
    shift = 14
    while shift > 0 and peak * 2.0**shift > _FP16_MAX:
        shift -= 1
    return shift


class Prepared:
    def __init__(self, tm, k, n, seed, global_scale=GLOBAL_SCALE):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        codes = torch.randint(
            0, 16, (k, n), generator=gen, device="cuda", dtype=torch.uint8
        )
        e4m3 = (
            torch.rand((k // 16, n), generator=gen, device="cuda") * 384.0 + 64.0
        ).to(torch.float8_e4m3fn)
        self.scales_f16 = (e4m3.float() * global_scale).half()
        lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device="cuda")
        self.reference = lut[codes.long()] * self.scales_f16.float().repeat_interleave(
            16, 0
        )
        self.weight, self.scales, meta = tm.nvfp4_sm70_prepare(codes, self.scales_f16)
        self.k_ld, self.q_ld = (int(v) for v in meta.tolist())
        self.k, self.n = k, n
        self.shift = _scale_shift(self.scales)
        self.tm = tm

    def args(self):
        return (self.weight, self.scales, 16, self.k_ld, self.q_ld)

    def gemv(self, x):
        out = torch.empty(x.shape[0], self.n, device="cuda", dtype=torch.float16)
        self.tm.nvfp4_tm_gemv_dispatch_sm70_out(out, x, *self.args(), self.shift)
        return out

    def turbomind(self, x):
        out = torch.empty(x.shape[0], self.n, device="cuda", dtype=torch.float16)
        self.tm.nvfp4_gemm_sm70_out(out, x, *self.args())
        return out


def _max_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


@pytest.fixture
def tm(tm_gpu):
    if not tm_gpu.has_op("nvfp4_tm_gemv_dispatch_sm70_out"):
        pytest.fail(
            "_sm70_C is loaded but has no nvfp4_tm_gemv_dispatch_sm70_out (K11)"
        )
    return tm_gpu


@pytest.fixture
def layers(tm):
    return {
        name: Prepared(tm, k, n, seed)
        for seed, (name, (k, n)) in enumerate(FLINT_SHAPES.items())
    }


def test_prepare_meta_matches_the_gemv_gate(layers, tm):
    for p in layers.values():
        assert (p.k_ld, p.q_ld) == (32 * p.k, p.n)
        assert p.shift == 14
        assert tm.nvfp4_tm_gemv_takes_gemv(1, p.k, p.n, 16, p.k_ld, p.q_ld)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("name", list(FLINT_SHAPES))
@torch.inference_mode()
def test_gemv_matches_fp32_reference_and_turbomind(layers, name, m):
    p = layers[name]
    x = torch.randn(m, p.k, device="cuda").half()
    out = p.gemv(x)
    reference = x.float() @ p.reference
    turbomind = p.turbomind(x)
    magnitude = reference.abs().max().item()
    # FP32 accumulation of exact FP16 products, well inside FP16 output
    # rounding. TurboMind accumulates in a different order.
    assert _max_error(out, reference) <= 2e-3 * magnitude
    assert _max_error(out, turbomind) <= 2e-3 * magnitude
    # A different kernel ran (its summation order differs from TurboMind's).
    assert not torch.equal(out, turbomind)


@pytest.mark.parametrize("name", list(FLINT_SHAPES))
@torch.inference_mode()
def test_larger_batches_keep_turbomind(layers, name):
    p = layers[name]
    for m in (3, 8):
        x = torch.randn(m, p.k, device="cuda").half()
        assert torch.equal(p.gemv(x), p.turbomind(x))


@pytest.mark.parametrize("name", ["qkv", "shared_down", "dense_gate_up"])
@torch.inference_mode()
def test_cuda_graph_replay_is_deterministic(layers, name):
    p = layers[name]
    inputs = {m: torch.zeros(m, p.k, device="cuda", dtype=torch.half) for m in (1, 2)}
    outputs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x in inputs.values():  # warm up outside capture
            p.gemv(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for m, x in inputs.items():
            outputs[m] = p.gemv(x)
    for _ in range(3):
        for x in inputs.values():
            x.normal_()
        graph.replay()
        for m, x in inputs.items():
            assert torch.equal(outputs[m], p.gemv(x))
            reference = x.float() @ p.reference
            magnitude = reference.abs().max().item()
            assert _max_error(outputs[m], reference) <= 2e-3 * magnitude


@torch.inference_mode()
def test_strided_activation_rows(layers):
    p = layers["o"]
    wide = torch.randn(2, p.k + 64, device="cuda").half()
    strided = wide[:, : p.k]
    assert not strided.is_contiguous()
    assert torch.equal(p.gemv(strided), p.gemv(strided.clone()))


@torch.inference_mode()
def test_misaligned_rows_fall_back_to_turbomind(layers):
    """Row stride not a multiple of 8 halves: the GEMV can't read 16-byte
    rows, so the op must run TurboMind (and still be correct)."""
    p = layers["o"]
    wide = torch.randn(2, p.k + 4, device="cuda").half()
    x = wide[:, : p.k]
    assert x.stride(0) % 8 != 0
    out = p.gemv(x)
    assert torch.equal(out, p.turbomind(x.contiguous()))


@torch.inference_mode()
def test_scales_above_four_use_a_smaller_shift(tm):
    # global 0.05 * e4m3 up to 448 -> effective scales up to ~22.
    p = Prepared(tm, 512, 2048, seed=11, global_scale=0.05)
    assert p.shift < 14
    for m in (1, 2):
        x = torch.randn(m, 512, device="cuda").half() * 1e-2
        out = p.gemv(x)
        reference = x.float() @ p.reference
        assert torch.isfinite(out).all()
        assert _max_error(out, reference) <= 2e-3 * reference.abs().max().item()


@pytest.mark.parametrize("k,n", [(1024, 2048), (5120, 8704), (2560, 2048)])
@torch.inference_mode()
def test_non_flint_shapes_run_turbomind(tm, k, n):
    p = Prepared(tm, k, n, seed=3)
    assert not tm.nvfp4_tm_gemv_takes_gemv(1, k, n, 16, p.k_ld, p.q_ld)
    for m in (1, 2):
        x = torch.randn(m, k, device="cuda").half()
        assert torch.equal(p.gemv(x), p.turbomind(x))
