# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K18 GPU tests: DFlash2 FP16 M<=8 GEMM and compact FP16 top-64 on SM70.

Ported from fork cf9492a tests/kernels/quantization/test_sm70_dflash2_fp16_m8.py
and tests/kernels/core/test_sm70_compact_half_topk.py. Changes: the shape set
is the TP2 admitted set (DRAFT_SHAPES_TP2) instead of the fork's TP4 shards;
new cases for M 1..7 (padded), misaligned inputs, the U1 GEMM callable and the
kill switch. The fork's test_target_probe_preserves_dense_token_mask is not
ported: it needs the fork-only DFlash2 sparse-rejection sampler.

    pytest -m gpu sm70/tests/dflash2
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _opt_in(monkeypatch):
    """K18 is opt-in (VLLM_SM70_DFLASH2_FP16=1); these tests exercise it."""
    from vllm_sm70.ops import dflash2

    monkeypatch.setenv("VLLM_SM70_DFLASH2_FP16", "1")
    dflash2.reset_selection_cache()
    yield
    dflash2.reset_selection_cache()


def _input_scale(n: int, k: int) -> float:
    # The fork's per-shape scales (FORK test :17-25) by projection role: the
    # M7 emulation feeds gate_up x/32, o_proj attention/256, down a
    # transport-scaled SwiGLU product.
    if (n, k) in {(17408, 5120), (6144, 2048)}:  # gate_up
        return 1 / 32
    if (n, k) in {(5120, 2048), (2048, 2048)}:  # o_proj
        return 1 / 256
    if (n, k) in {(5120, 8704), (2048, 3072)}:  # down
        return 0.1
    return 1.0


def _shapes():
    from vllm_sm70.ops import dflash2

    return sorted(dflash2.GEOMETRY)


def _layer(mod, n, k, seed=20261004):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.float16)
    # Use BF16 weight values within FP16 transport's range.
    layer.weight.copy_(layer.weight.bfloat16().half())
    assert mod.mark_and_prepare(layer), (n, k)
    return layer


@pytest.mark.parametrize("n,k", _shapes())
@torch.inference_mode()
def test_fp16_m8_replays_live_inputs(dflash2_gpu, n, k):
    mod = dflash2_gpu
    input_scale = _input_scale(n, k)
    layer = _layer(mod, n, k)
    original = layer.weight.clone()
    assert torch.equal(layer.weight, original)
    assert mod.PACKED_ATTR not in layer.state_dict()
    assert getattr(layer, mod.GEOMETRY_ATTR) == mod.GEOMETRY[(n, k)]
    x = torch.randn(8, k, device="cuda", dtype=torch.float16) * input_scale
    mod.apply_dflash2_fp16_m8(layer, x, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mod.apply_dflash2_fp16_m8(layer, x, None)
    for amplitude in [1.0, -0.5, 0.0, 2.0]:
        x.copy_(torch.randn_like(x) * (input_scale * amplitude))
        graph.replay()
        oracle = F.linear(x.float(), original.float())
        torch.testing.assert_close(output.float(), oracle, atol=2e-3, rtol=2e-3)
        assert torch.isfinite(output).all()
    # Larger batches and strided/3D operands retain the existing GEMM route.
    assert mod.apply_dflash2_fp16_m8(layer, x.repeat(4, 1), None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x.T.contiguous().T, None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x.unsqueeze(0), None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x, torch.zeros(n, device=x.device)) is None


@pytest.mark.parametrize("n,k", [(1280, 5120), (3072, 5120), (2048, 3072)])
@pytest.mark.parametrize("m", [1, 2, 5, 7])
@torch.inference_mode()
def test_m_below_8_is_padded_m8(dflash2_gpu, n, k, m):
    """M 1..7: rows are bitwise those of the M8 kernel on zero-padded input."""
    mod = dflash2_gpu
    layer = _layer(mod, n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    out = mod.apply_dflash2_fp16_m8(layer, x, None)
    assert out is not None and out.shape == (m, n)
    padded = torch.zeros(8, k, device="cuda", dtype=torch.float16)
    padded[:m] = x
    ref = torch.empty(8, n, device="cuda", dtype=torch.float16)
    tile, warps = getattr(layer, mod.GEOMETRY_ATTR)
    torch.ops._sm70_C.sm70_dflash2_fp16_m8_out(
        ref, padded, getattr(layer, mod.PACKED_ATTR), tile, warps
    )
    assert torch.equal(out.view(torch.int16), ref[:m].view(torch.int16))
    oracle = F.linear(x.float(), layer.weight.float())
    torch.testing.assert_close(out.float(), oracle, atol=2e-3, rtol=2e-3)
    # Under graph capture too.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mod.apply_dflash2_fp16_m8(layer, x, None)
    x.copy_(torch.randn_like(x) * 0.1)
    graph.replay()
    torch.testing.assert_close(
        captured.float(),
        F.linear(x.float(), layer.weight.float()),
        atol=2e-3,
        rtol=2e-3,
    )


@torch.inference_mode()
def test_misaligned_input_keeps_matrix_product(dflash2_gpu):
    mod = dflash2_gpu
    n, k = 1280, 5120
    layer = _layer(mod, n, k)
    storage = torch.randn(8 * k + 1, device="cuda", dtype=torch.float16) * 0.1
    x = storage[1:].view(8, k)  # contiguous, 2-byte offset
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    out = torch.empty(8, n, device="cuda", dtype=torch.float16)
    tile, warps = getattr(layer, mod.GEOMETRY_ATTR)
    packed = getattr(layer, mod.PACKED_ATTR)
    torch.ops._sm70_C.sm70_dflash2_fp16_dispatch_out(
        out, x, packed, layer.weight, tile, warps
    )
    assert torch.equal(
        out.view(torch.int16), F.linear(x, layer.weight).view(torch.int16)
    )
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        torch.ops._sm70_C.sm70_dflash2_fp16_m8_out(out, x, packed, tile, warps)


@torch.inference_mode()
def test_u1_gemm_callable(dflash2_gpu, monkeypatch):
    mod = dflash2_gpu
    gemm = mod.select_unquantized_gemm("auto")
    assert gemm is mod.dflash2_unquantized_gemm
    layer = _layer(mod, 3072, 5120)
    plain = torch.nn.Linear(5120, 3072, bias=False, device="cuda", dtype=torch.float16)
    for m in (1, 8, 16, 128):
        x = torch.randn(m, 5120, device="cuda", dtype=torch.float16) * 0.1
        out = gemm(layer, x, layer.weight, None)
        torch.testing.assert_close(
            out.float(), F.linear(x.float(), layer.weight.float()), atol=2e-3, rtol=2e-3
        )
        if m > 8:
            assert torch.equal(out, F.linear(x, layer.weight))
        # Unmarked layers are exactly F.linear.
        assert torch.equal(
            gemm(plain, x, plain.weight, None), F.linear(x, plain.weight)
        )
    monkeypatch.delenv(mod.KILL_SWITCH)  # default: off
    mod.reset_selection_cache()
    assert mod.select_unquantized_gemm("auto") is None
    fresh = torch.nn.Linear(5120, 3072, bias=False, device="cuda", dtype=torch.float16)
    assert not mod.mark_and_prepare(fresh)


@torch.inference_mode()
def test_dynamic_prefill_trace_retains_m8_decode_dispatch(dflash2_gpu):
    mod = dflash2_gpu
    layer = _layer(mod, 1280, 5120)
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return torch._inductor.compile(graph, inputs)

    compiled = torch.compile(
        lambda x: mod.dflash2_unquantized_gemm(layer, x, layer.weight, None),
        backend=backend,
        fullgraph=True,
        dynamic=True,
    )
    tile, warps = getattr(layer, mod.GEOMETRY_ATTR)

    def check(rows):
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16) * 0.1
        actual = compiled(x)
        if rows <= 8:
            expected = torch.empty_like(actual)
            torch.ops._sm70_C.sm70_dflash2_fp16_dispatch_out(
                expected, x, getattr(layer, mod.PACKED_ATTR), layer.weight, tile, warps
            )
        else:
            expected = F.linear(x, layer.weight)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    # One dynamic graph serves every M >= 2: prefill sizes and the M<=8 decode.
    for rows in [12, 8, 7, 32, 8, 128, 2]:
        check(rows)
    assert len(compilations) == 1
    # M=1 (the commonest decode size) through the compiled route. Dynamo's
    # backed dynamic shapes specialize sizes 0 and 1, so M=1 fails the M>=2
    # guard and compiles exactly one more graph, by design (the earlier
    # "assert 2 == 1"). It must still dispatch to the M8 kernel bitwise.
    check(1)
    assert len(compilations) == 2
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(x)
    for amplitude in [0.0, -0.1, 0.2]:
        x.copy_(torch.randn_like(x) * amplitude)
        graph.replay()
        expected = mod.apply_dflash2_fp16_m8(layer, x, None)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


# ------------------------------------------------------------ compact top-k
# 62080 = the 248320-token vocab over TP4; 124160 over TP2 exceeds the op's
# 65535 limit, so at TP2 the guard returns None and the caller uses topk.
@pytest.mark.parametrize("rows,width", [(8, 62080), (32, 65535), (8, 1025)])
@torch.inference_mode()
def test_half_source_values_and_live_graph(dflash2_gpu, rows, width):
    mod = dflash2_gpu
    torch.manual_seed(1530)
    logits = torch.randn(rows, width, dtype=torch.float16, device="cuda")
    assert mod.compact_half_topk(logits) is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = mod.compact_half_topk(logits)
    for case in ("random", "ties", "zeros", "nonfinite"):
        logits.normal_()
        if case == "ties":
            logits.round_()
        elif case == "zeros":
            logits.zero_()
            logits[:, ::2] = -0.0
        elif case == "nonfinite":
            logits[:, 0] = float("nan")
            logits[:, 1] = float("inf")
            logits[:, 2] = -float("inf")
        graph.replay()
        values, ids = selected
        expected = logits.float().topk(64, dim=-1).values
        torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
        original = logits.gather(1, ids).float()
        torch.testing.assert_close(values, original, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(values)
        assert torch.equal(
            values.view(torch.int32)[finite], original.view(torch.int32)[finite]
        )
        assert all(len(set(row)) == 64 for row in ids.cpu().tolist())


@torch.inference_mode()
def test_fp32_source_never_qualifies(dflash2_gpu):
    logits = torch.randn(8, 62080, device="cuda", dtype=torch.float32)
    assert dflash2_gpu.compact_half_topk(logits) is None


@torch.inference_mode()
def test_tp2_vocab_shard_falls_back(dflash2_gpu):
    logits = torch.randn(8, 124160, device="cuda", dtype=torch.float16)
    assert dflash2_gpu.compact_half_topk(logits) is None
