# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 weight-streaming GEMVs on an SM70 GPU.

- ops/small_m_gemv.py against cuBLAS at the Inkling 2 Mini TP2 shapes (gated
  qkvr with its k and v inputs, wo, the sharded LM head), 1 to 8 rows;
- ops/nvfp4_gemv.py on banks prepared by ``tm.nvfp4_sm70_prepare`` against
  the TurboMind compact stage (``nvfp4_moe_dense_stage_sm70_out``, one
  one-row group per slot) and the soft-capped SwiGLU kernel;
- both bitwise equal under CUDA-graph replay.

Timing is sm70/tools/bench_inkling2_gemv.py.

    pytest -m gpu sm70/tests/ops/test_m12_inkling2_small_m_gemv_gpu.py
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")

HIDDEN = 2304
QKVR_SIZES = (1536, 192, 1536, 12, 512, 512)
HEAD_ROWS = 50256
LOCAL = 576
TOL = 2e-3
GATE_CAP, UP_CAP = 4.0, 25.0


def max_abs_rel(got, want):
    return float((got.float() - want.float()).abs().max() / want.float().abs().max())


@pytest.fixture
def cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    return torch.device("cuda")


def _rand(*shape, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=g, device="cuda") * scale).half()


@torch.inference_mode()
@pytest.mark.parametrize("rows", range(1, 9))
def test_dense_shapes_match_cublas(cuda, rows):
    from vllm_sm70.models.inkling2_sm70.ops.small_m_gemv import segment_gemv

    qkvr = _rand(sum(QKVR_SIZES), HIDDEN, scale=0.02, seed=1)
    wo = _rand(HIDDEN, 1536, scale=0.02, seed=2)
    head = _rand(HEAD_ROWS, HIDDEN, scale=0.02, seed=3)
    xs = tuple(_rand(rows, HIDDEN, seed=4 + i) for i in range(3))
    query = sum(QKVR_SIZES[:4])
    bounds = (query, query + QKVR_SIZES[4])
    got = segment_gemv(xs, qkvr, bounds, xs[0].new_empty(rows, qkvr.shape[0]))
    w_q, w_k, w_v = qkvr.split((query, *QKVR_SIZES[4:]))
    want = torch.cat([xs[0] @ w_q.t(), xs[1] @ w_k.t(), xs[2] @ w_v.t()], dim=1)
    assert max_abs_rel(got, want) <= TOL
    gated = _rand(rows, 1536, seed=8)
    got = segment_gemv((gated,), wo, (), gated.new_empty(rows, HIDDEN))
    assert max_abs_rel(got, gated @ wo.t()) <= TOL
    got = segment_gemv((xs[0],), head, (), xs[0].new_empty(rows, HEAD_ROWS))
    assert max_abs_rel(got, xs[0] @ head.t()) <= TOL


def _bank(tm, experts, k, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 16, (experts, k, n), generator=g, device="cuda")
    e4m3 = (torch.rand(experts, k // 16, n, generator=g, device="cuda") * 3 + 1).to(
        torch.float8_e4m3fn
    )
    scales = (e4m3.float() * 0.002).half()
    words, prepared = [], []
    for e in range(experts):
        w, s, meta = tm.nvfp4_sm70_prepare(codes[e].to(torch.uint8), scales[e])
        assert tuple(meta.tolist()) == (32 * k, n)
        words.append(w)
        prepared.append(s)
    words, prepared = torch.stack(words), torch.stack(prepared)
    meta = (32 * k, n)
    ptrs = tm.awq_moe_build_strided_ptrs(words, prepared, *meta, experts)
    return words, prepared, ptrs


def _tm_stage(tm, out, rows, ids, ptrs, k, n):
    slots = rows.shape[0]
    offsets = torch.arange(slots + 1, dtype=torch.int32, device="cuda")
    tm.nvfp4_moe_dense_stage_sm70_out(out, rows, offsets, ids, *ptrs, slots, k, n, 16)
    return out


@torch.inference_mode()
@pytest.mark.parametrize("slots", [8, 16, 32, 64, 80])
def test_nvfp4_slots_match_turbomind(cuda, tm_gpu, slots):
    from vllm_sm70.models.inkling2_sm70.ops.activation import softcapped_swiglu_out
    from vllm_sm70.models.inkling2_sm70.ops.nvfp4_gemv import nvfp4_slot_gemv

    tm = tm_gpu
    w13 = _bank(tm, 16, HIDDEN, 2 * LOCAL, seed=1)
    w2 = _bank(tm, 16, LOCAL, HIDDEN, seed=2)
    g = torch.Generator(device="cuda").manual_seed(slots)
    ids = torch.randint(0, 16, (slots,), generator=g, device="cuda")
    ids = ids.sort().values.to(torch.int32)
    rows = _rand(slots, HIDDEN, seed=slots)
    gate_up = _tm_stage(
        tm, rows.new_empty(slots, 2 * LOCAL), rows, ids, w13[2], HIDDEN, 2 * LOCAL
    )
    want_act = softcapped_swiglu_out(
        gate_up, rows.new_empty(slots, LOCAL), GATE_CAP, UP_CAP
    )
    act = nvfp4_slot_gemv(
        rows, w13[0], w13[1], ids, rows.new_empty(slots, LOCAL), (GATE_CAP, UP_CAP)
    )
    assert max_abs_rel(act, want_act) <= TOL
    want = _tm_stage(
        tm, rows.new_empty(slots, HIDDEN), want_act, ids, w2[2], LOCAL, HIDDEN
    )
    got = nvfp4_slot_gemv(want_act, w2[0], w2[1], ids, rows.new_empty(slots, HIDDEN))
    assert max_abs_rel(got, want) <= TOL


@torch.inference_mode()
def test_graph_replay_is_bitwise(cuda, tm_gpu):
    from vllm_sm70.models.inkling2_sm70.ops.nvfp4_gemv import nvfp4_slot_gemv
    from vllm_sm70.models.inkling2_sm70.ops.small_m_gemv import segment_gemv

    w13 = _bank(tm_gpu, 8, HIDDEN, 2 * LOCAL, seed=3)
    wo = _rand(HIDDEN, 1536, scale=0.02, seed=4)
    rows, gated = _rand(8, HIDDEN, seed=5), _rand(4, 1536, seed=6)
    ids = torch.arange(8, dtype=torch.int32, device="cuda")
    act, out = rows.new_empty(8, LOCAL), gated.new_empty(4, HIDDEN)

    def step():
        nvfp4_slot_gemv(rows, w13[0], w13[1], ids, act, (GATE_CAP, UP_CAP))
        segment_gemv((gated,), wo, (), out)

    step()
    eager = (act.clone(), out.clone())
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        step()
        with torch.cuda.graph(graph, stream=stream):
            step()
    act.zero_()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(act, eager[0]) and torch.equal(out, eager[1])
