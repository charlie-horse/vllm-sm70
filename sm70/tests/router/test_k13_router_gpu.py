# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SM70 E256/K6 sigmoid router against the generic FP32 route (GPU).

Ported from fork cf9492a tests/kernels/moe/test_sm70_flint_router.py
(192 lines, 3629e29f6). The generic route is what the MoE runner runs for
this layer: F.linear with the FP32 gate on FP32-cast hidden states, then
topk_sigmoid (renormalized). The fused kernel only sums the logits in another
FP32 order, so ids must match except at genuine near-ties, and weights must
agree to FP32 rounding.

New here: the descending-capture regression (risk R4, the C4 pattern):
graphs captured largest batch first on a fresh stream and a poisoned shared
pool, the smallest replayed first, with the workspace admitted eagerly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import make_gate, make_runner

pytestmark = pytest.mark.gpu

FP32_EPS = 2.0**-24
# A flip is a near-tie when the two experts' exact logits differ by at most
# this many FP32 rounding units of the dot product's magnitude, or when their
# exact sigmoid scores are within a few FP32 ulps of 1 (saturation ties).
LOGIT_TIE_UNITS = 64
SCORE_TIE = 4 * 2.0**-24
WEIGHT_ATOL = 2e-6


def reference(x, w, top_k):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    logits = F.linear(x.to(w.dtype), w)
    weights, ids, _ = fused_topk(
        x, logits, top_k, renormalize=True, scoring_func="sigmoid"
    )
    return weights, ids


def state_for(router, w):
    runner = make_runner(gate=make_gate(device=w.device))
    runner.gate.weight.data = w
    return router.FlintFusedRouter(router._flint_route_op(), w, runner)


def compare(router, x, w, weights, ids, ref_weights, ref_ids):
    """Tie-aware comparison; returns the number of rows with flipped ids."""
    exact = x.double() @ w.double().T
    magnitude = x.double().abs() @ w.double().abs().T
    tol = LOGIT_TIE_UNITS * FP32_EPS * magnitude
    score = torch.sigmoid(exact)
    flips = 0
    for row in range(x.shape[0]):
        got = dict(zip(ids[row].tolist(), weights[row].tolist(), strict=True))
        want = dict(zip(ref_ids[row].tolist(), ref_weights[row].tolist(), strict=True))
        assert len(got) == router.TOP_K, f"row {row}: duplicate ids {ids[row]}"
        extra = sorted(set(got) - set(want))
        missing = sorted(set(want) - set(got))
        if extra:
            flips += 1
            for e_in, e_out in zip(extra, missing, strict=True):
                logit_gap = (exact[row, e_in] - exact[row, e_out]).abs()
                score_gap = (score[row, e_in] - score[row, e_out]).abs()
                assert (
                    logit_gap <= tol[row, e_in] + tol[row, e_out]
                    or score_gap <= SCORE_TIE
                ), (
                    f"row {row}: expert {e_in} replaced {e_out} with exact "
                    f"logit gap {float(logit_gap):.3e} and score gap "
                    f"{float(score_gap):.3e}"
                )
        for expert in set(got) & set(want):
            assert abs(got[expert] - want[expert]) <= WEIGHT_ATOL, (
                f"row {row} expert {expert}: {got[expert]} vs {want[expert]}"
            )
        total = sum(got.values())
        assert total == 0.0 or abs(total - 1.0) <= 1e-5
    return flips


def gate_weight(router, seed: int, scale: float = 0.02) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(router.NUM_EXPERTS, router.HIDDEN, device="cuda", generator=g)
    # Real router weights are FP16 values held in FP32.
    return (w * scale).half().float()


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("x_scale", [0.05, 1.0, 8.0])
def test_random_matches_generic_route(router_gpu, tokens, x_scale):
    r = router_gpu
    torch.manual_seed(tokens * 7 + int(x_scale * 10))
    w = gate_weight(r, 1234)
    state = state_for(r, w)
    flips = rows = 0
    for _ in range(150):
        x = (torch.randn(tokens, r.HIDDEN, device="cuda") * x_scale).half()
        # Massive-activation channels, as in real residual-normed inputs.
        x[:, torch.randint(0, r.HIDDEN, (4,))] *= 30
        weights, ids = state.route(x)
        ref_weights, ref_ids = reference(x, w, r.TOP_K)
        assert ids.dtype == ref_ids.dtype == torch.int32
        assert weights.dtype == ref_weights.dtype == torch.float32
        flips += compare(r, x, w, weights, ids, ref_weights, ref_ids)
        rows += tokens
    # Near-ties in FP32 are rare on continuous data.
    assert flips <= max(2, rows // 100)


def test_selection_order_matches_generic_route(router_gpu):
    """Ids come out in descending score order, like topk_sigmoid."""
    r = router_gpu
    torch.manual_seed(3)
    w = gate_weight(r, 77)
    x = torch.randn(2, r.HIDDEN, device="cuda").half()
    weights, ids = state_for(r, w).route(x)
    ref_weights, ref_ids = reference(x, w, r.TOP_K)
    if torch.equal(ids, ref_ids):
        torch.testing.assert_close(weights, ref_weights, atol=WEIGHT_ATOL, rtol=0)
    scores = torch.sigmoid(F.linear(x.float(), w)).gather(1, ids.long())
    assert torch.all(scores[:, :-1] >= scores[:, 1:])


def test_saturated_ties_take_lowest_ids(router_gpu):
    r = router_gpu
    w = torch.zeros(r.NUM_EXPERTS, r.HIDDEN, device="cuda")
    hot = [200, 5, 77, 3, 150, 9, 31, 64]
    w[hot] = 0.01
    x = torch.ones(2, r.HIDDEN, device="cuda", dtype=torch.float16)
    weights, ids = state_for(r, w).route(x)
    ref_weights, ref_ids = reference(x, w, r.TOP_K)
    assert ids.tolist() == [sorted(hot)[: r.TOP_K]] * 2
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_weights)


def test_zero_and_nan_rows_match_exactly(router_gpu):
    r = router_gpu
    w = gate_weight(r, 5)
    x = torch.zeros(2, r.HIDDEN, device="cuda", dtype=torch.float16)
    x[1, 17] = float("nan")
    weights, ids = state_for(r, w).route(x)
    ref_weights, ref_ids = reference(x, w, r.TOP_K)
    assert ids.tolist() == [list(range(r.TOP_K))] * 2
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_weights)
    assert torch.all(weights[1] == 0)


def test_strided_rows(router_gpu):
    r = router_gpu
    torch.manual_seed(11)
    w = gate_weight(r, 9)
    base = torch.randn(2, r.HIDDEN + 64, device="cuda").half()
    x = base[:, 32 : 32 + r.HIDDEN]
    assert not x.is_contiguous()
    assert r.FlintFusedRouter.accepts(x)
    weights, ids = state_for(r, w).route(x)
    ref_weights, ref_ids = reference(x.contiguous(), w, r.TOP_K)
    compare(r, x, w, weights, ids, ref_weights, ref_ids)


def test_repeated_launches_and_graph_replay(router_gpu):
    """The arrival counter re-arms: every launch and replay sees new input."""
    r = router_gpu
    torch.manual_seed(21)
    w = gate_weight(r, 42)
    state = state_for(r, w)
    for tokens in (1, 2):
        static_x = torch.randn(tokens, r.HIDDEN, device="cuda").half()
        for _ in range(3):
            state.route(static_x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_weights, out_ids = state.route(static_x)
        for replay in range(40):
            static_x.copy_(
                torch.randn(tokens, r.HIDDEN, device="cuda")
                * (0.1, 1.0, 4.0)[replay % 3]
            )
            graph.replay()
            eager_weights, eager_ids = state.route(static_x)
            # Fixed-order reductions: replay is bitwise equal to eager.
            assert torch.equal(out_ids, eager_ids)
            assert torch.equal(out_weights, eager_weights)
            ref_weights, ref_ids = reference(static_x, w, r.TOP_K)
            compare(r, static_x, w, out_weights, out_ids, ref_weights, ref_ids)
        assert int(state.counter.item()) == 0


def _poison_pool(pool, stream, device):
    """Fill the shared graph pool with garbage that later captures reuse."""
    poison = torch.cuda.CUDAGraph()
    with torch.cuda.graph(poison, pool=pool, stream=stream):
        blocks = [torch.empty(16 << 20, dtype=torch.int32, device=device)]
        blocks += [
            torch.empty(1 << 17, dtype=torch.int32, device=device) for _ in range(64)
        ]
        for block in blocks:
            block.fill_(0x7FFFFFFF)
    poison.replay()
    torch.cuda.synchronize()
    del blocks, block
    return poison


@torch.inference_mode()
def test_descending_capture_regression(router_gpu, monkeypatch):
    """Black box (R4): admission through the runner hook, capture 2 then 1
    on a fresh stream and a poisoned shared pool with no eager route call,
    then replay 1 before 2. A workspace or counter created lazily under
    capture would come from the poisoned pool (or only be zeroed by the
    first graph) and corrupt the batch-1 graph."""
    r = router_gpu
    device = torch.device("cuda")
    w = gate_weight(r, 314)
    runner = make_runner(gate=make_gate(device=device))
    runner.gate.weight.data = w
    model = torch.nn.Module()
    holder = torch.nn.Module()
    for k, v in dict(
        router_provider=None,
        _precomputed_route=None,
        _precomputed_route_decided=False,
        **vars(runner),
    ).items():
        object.__setattr__(holder, k, v)
    model.layer = holder
    # Warmup (U1 kernel_warmup): the only place the workspace is created.
    assert r.prepare_model(model) == 1
    state = holder._precomputed_route
    assert isinstance(state, r.FlintFusedRouter)
    counter_ptr, logits_ptr = state.counter.data_ptr(), state.logits.data_ptr()

    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    poison = _poison_pool(pool, stream, device)
    x = torch.empty(2, r.HIDDEN, device=device, dtype=torch.float16)
    graphs, outs = {}, {}
    for batch in (2, 1):  # descending, as vLLM captures
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool, stream=stream):
            outs[batch] = state.try_route(x[:batch])
        assert outs[batch] is not None
        graphs[batch] = g
    # Capture created nothing new.
    assert holder._precomputed_route is state
    assert (state.counter.data_ptr(), state.logits.data_ptr()) == (
        counter_ptr,
        logits_ptr,
    )

    torch.manual_seed(5)
    for batch in (1, 2, 1, 1, 2):  # smallest first
        x.copy_(torch.randn(2, r.HIDDEN, device=device) * 2.0)
        graphs[batch].replay()
        weights, ids = outs[batch]
        ref_weights, ref_ids = reference(x[:batch].clone(), w, r.TOP_K)
        compare(r, x[:batch], w, weights, ids, ref_weights, ref_ids)
    torch.cuda.synchronize()
    assert int(state.counter.item()) == 0
    del poison
