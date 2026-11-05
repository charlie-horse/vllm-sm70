# SPDX-License-Identifier: Apache-2.0
"""K12 GPU: Flint NVFP4 MoE M<=2 two-kernel decode on Flint TP2 shapes.

Port of the two-kernel GPU decode cases of fork cf9492a
tests/kernels/moe/test_sm70_nvfp4_contract_flint.py:
  * test_flint_decode_qpn_matches_reference_and_turbomind (:1183)
  * test_flint_decode_qpn_skips_invalid_routes (:1223)
  * test_flint_decode_qpn_cuda_graph_replay (:1242)
Dropped: the persistent single-launch cases (:1296-1447; the op is not
ported) and test_flint_decode_qpn_kill_switch_keeps_turbomind (:1276; the
switch acts at load, covered on CPU in test_k12_flint_moe_cpu.py, and the
TurboMind fallback route is M4's, tested in tests/quant/test_m4_nvfp4_moe.py).

New for R4 and the shape audit:
  * test_descending_capture_regression: the C4 `--max-num-seqs 20` pattern
    (fork test_decode_sinks_descending_capture_regression, sinks test :763)
    applied to the split-K counters: graphs captured M=2 then M=1 on a
    poisoned shared pool with no eager call first, M=1 replayed first.
  * test_shared_workspace_two_layers_in_one_graph.
  * test_admitted_shape_set / test_guards_reject_other_shapes: the C++
    guards admit exactly Flint TP2 at M in {1, 2} and refuse TP1/TP4 banks,
    M=3, top-8 and misaligned input.

Run from the repo root against the built package:
    pytest -m gpu sm70/tests/flint_moe
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _flint_moe_ref import (  # noqa: E402
    EXPERTS,
    HIDDEN,
    INTER,
    TOP_K,
    dirty_graph_pool,
    flint_expert_weights,
    prepare_banks,
    reference_moe,
    relative_error,
    routing,
    turbomind_moe,
)

pytestmark = pytest.mark.gpu

_BANKS: dict = {}


def _layer(flint_gpu, w2_global: float = 0.04, seed: int = 20261029):
    """(raw, banks, plan) for one Flint TP2 layer; cached per process."""
    device, flint_moe, tm = flint_gpu
    key = (str(device), w2_global, seed)
    if key not in _BANKS:
        raw = flint_expert_weights(device, seed, w2_global)
        banks = prepare_banks(tm, raw)
        plan = flint_moe.admit_layer(
            (EXPERTS, HIDDEN, INTER, TOP_K),
            banks.w13_weight,
            banks.w13_scales,
            banks.w2_weight,
            banks.w2_scales,
            policy_enabled=True,
        )
        assert plan is not None, "Flint TP2 banks must be admitted"
        _BANKS[key] = (raw, banks, plan)
    return _BANKS[key]


def _buffers(device, tokens):
    out = torch.empty(tokens, HIDDEN, dtype=torch.float16, device=device)
    act = torch.empty(tokens * TOP_K, INTER, dtype=torch.float16, device=device)
    return out, act


def _decode(flint_moe, banks, plan, ws, x, weights, ids, out=None, act=None):
    if out is None:
        out, act = _buffers(x.device, x.shape[0])
    assert flint_moe.call_rejection(x, weights, ids) is None
    return flint_moe.flint_moe_decode_out(
        out,
        act,
        x,
        banks.w13_weight,
        banks.w13_scales,
        banks.w2_weight,
        banks.w2_scales,
        weights,
        ids,
        ws,
        plan,
    )


@pytest.mark.parametrize("w2_global", (0.04, 7.0), ids=("checkpoint", "x16_w2"))
def test_decode_matches_reference_and_turbomind(flint_gpu, w2_global):
    """M=1/2 against the FP32 dequantized oracle and TurboMind GEMMs.

    The second case makes the prepared W2 scales reach 28 (E4M3 up to 2 times
    a global of 7), beyond the FP16 limit of prepared_scale * 2^14, so the
    load-time power-of-two pre-shift must engage.
    """
    device, flint_moe, tm = flint_gpu
    raw, banks, plan = _layer(flint_gpu, w2_global)
    assert plan.w13_scale_shift == 0
    peak = banks.w2_scales.float().max().item()
    shift = plan.w2_scale_shift
    assert peak * 2.0 ** (14 - shift) <= 65504.0
    assert (shift > 0) == (peak * 2.0**14 > 65504.0)
    if w2_global > 1:
        assert shift > 0
    ws = flint_moe.allocate_workspace(device)
    gen = torch.Generator().manual_seed(20261029)
    for tokens in (1, 2, 1, 2):
        x, weights, ids = routing(tokens, gen, device)
        if tokens == 2 and torch.rand(1, generator=gen) < 0.5:
            ids[1] = ids[0]  # both tokens on the same six experts
        actual = _decode(flint_moe, banks, plan, ws, x, weights, ids).clone()
        again = _decode(flint_moe, banks, plan, ws, x, weights, ids).clone()
        turbomind = turbomind_moe(tm, banks, x, weights, ids)
        expected = reference_moe(x, weights, ids, raw)
        torch.cuda.synchronize(device)
        assert torch.isfinite(actual).all(), tokens
        assert torch.equal(actual, again), "decode must be deterministic"
        assert relative_error(actual, expected) < 1e-2, tokens
        # TurboMind rounds the same FP16 boundaries in another K order.
        assert relative_error(actual, turbomind) < 2e-3, tokens
        assert not ws.counters.any(), "split-K counters must re-arm"


def test_int64_ids_match_int32(flint_gpu):
    """The runner may hand int64 IDs; they go through workspace.ids."""
    device, flint_moe, _ = flint_gpu
    _, banks, plan = _layer(flint_gpu)
    ws = flint_moe.allocate_workspace(device)
    x, weights, ids = routing(2, torch.Generator().manual_seed(5), device)
    a = _decode(flint_moe, banks, plan, ws, x, weights, ids).clone()
    b = _decode(flint_moe, banks, plan, ws, x, weights, ids.long()).clone()
    assert torch.equal(a, b)
    assert torch.equal(ws.ids, ids.view(-1))


def test_decode_skips_invalid_routes(flint_gpu):
    device, flint_moe, _ = flint_gpu
    raw, banks, plan = _layer(flint_gpu)
    ws = flint_moe.allocate_workspace(device)
    x, weights, ids = routing(2, torch.Generator().manual_seed(7), device)
    ids[0, 3] = -1
    ids[1, 5] = EXPERTS
    actual = _decode(flint_moe, banks, plan, ws, x, weights, ids).clone()
    valid = (ids >= 0) & (ids < EXPERTS)
    expected = reference_moe(x, weights * valid, torch.where(valid, ids, 0), raw)
    assert relative_error(actual, expected) < 1e-2
    assert not ws.counters.any()


def test_decode_cuda_graph_replay(flint_gpu):
    """FULL-graph style capture at M=1 and M=2: replays track new inputs.

    Replaying many times also checks that the split-K arrival counters re-arm
    inside the kernel, with no reset launch and no host work.
    """
    device, flint_moe, _ = flint_gpu
    raw, banks, plan = _layer(flint_gpu)
    ws = flint_moe.allocate_workspace(device)
    gen = torch.Generator().manual_seed(11)
    for tokens in (1, 2):
        static = [t.clone() for t in routing(tokens, gen, device)]
        out, act = _buffers(device, tokens)
        _decode(flint_moe, banks, plan, ws, *static, out, act)  # warm up
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _decode(flint_moe, banks, plan, ws, *static, out, act)
        for _ in range(4):
            case = routing(tokens, gen, device)
            for buffer, value in zip(static, case):
                buffer.copy_(value)
            for _ in range(3):
                graph.replay()
            replayed = out.clone()
            eager = _decode(flint_moe, banks, plan, ws, *case).clone()
            expected = reference_moe(*case, raw)
            torch.cuda.synchronize(device)
            assert torch.equal(replayed, eager), tokens
            assert relative_error(replayed, expected) < 1e-2, tokens
        assert not ws.counters.any()


def test_descending_capture_regression(flint_gpu):
    """Black box (R4): graphs captured largest first, smallest replayed first.

    Captures M=2 then M=1 on a fresh stream and a shared, poisoned graph pool
    with no eager decode call, then replays M=1 for several routings before
    M=2 ever replays, then M=2, then M=1 again. A workspace created or zeroed
    lazily inside the first capture would leave the M=1 graph on garbage
    counters (the fork's C4 `--max-num-seqs 20` gibberish).
    """
    device, flint_moe, _ = flint_gpu
    raw, banks, plan = _layer(flint_gpu)
    ws = flint_moe.allocate_workspace(device)  # load time, before any capture
    x = torch.zeros(2, HIDDEN, dtype=torch.float16, device=device)
    weights = torch.zeros(2, TOP_K, dtype=torch.float32, device=device)
    ids = torch.zeros(2, TOP_K, dtype=torch.int32, device=device)
    out = torch.full((2, HIDDEN), float("nan"), dtype=torch.float16, device=device)
    act = torch.empty(2 * TOP_K, INTER, dtype=torch.float16, device=device)

    def run(m):
        _decode(
            flint_moe,
            banks,
            plan,
            ws,
            x[:m],
            weights[:m],
            ids[:m],
            out[:m],
            act[: m * TOP_K],
        )

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.stream(stream):
        poison = dirty_graph_pool(device, pool, stream)
        graphs = {}
        for m in (2, 1):
            graphs[m] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[m], pool=pool, stream=stream):
                run(m)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    del poison
    assert not ws.counters.any(), "capture must not touch the counters"

    gen = torch.Generator().manual_seed(150)

    def step(m):
        case = routing(m, gen, device)
        x[:m].copy_(case[0])
        weights[:m].copy_(case[1])
        ids[:m].copy_(case[2])
        out.fill_(float("nan"))
        graphs[m].replay()
        torch.cuda.synchronize(device)
        assert torch.isfinite(out[:m]).all(), f"M={m} graph left rows unset"
        eager = _decode(flint_moe, banks, plan, ws, *case).clone()
        assert torch.equal(out[:m], eager), m
        assert relative_error(out[:m], reference_moe(*case, raw)) < 1e-2, m
        assert not ws.counters.any(), m

    for m in (1, 1, 1, 2, 2, 1, 2, 1):
        step(m)


def test_shared_workspace_two_layers_in_one_graph(flint_gpu):
    """Two layers in one stream share one workspace (counters re-arm per
    launch), as M4 may do; replays match per-layer eager results."""
    device, flint_moe, _ = flint_gpu
    _, banks_a, plan_a = _layer(flint_gpu)
    _, banks_b, plan_b = _layer(flint_gpu, w2_global=7.0)
    ws = flint_moe.allocate_workspace(device)
    gen = torch.Generator().manual_seed(21)
    x, weights, ids = (t.clone() for t in routing(2, gen, device))
    out_a, act_a = _buffers(device, 2)
    out_b, act_b = _buffers(device, 2)

    def run():
        _decode(flint_moe, banks_a, plan_a, ws, x, weights, ids, out_a, act_a)
        _decode(flint_moe, banks_b, plan_b, ws, out_a, weights, ids, out_b, act_b)

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        case = routing(2, gen, device)
        x.copy_(case[0])
        weights.copy_(case[1])
        ids.copy_(case[2])
        graph.replay()
        torch.cuda.synchronize(device)
        a = _decode(flint_moe, banks_a, plan_a, ws, *case).clone()
        b = _decode(flint_moe, banks_b, plan_b, ws, a, case[1], case[2]).clone()
        assert torch.equal(out_a, a)
        assert torch.equal(out_b, b)
        assert not ws.counters.any()


@pytest.mark.parametrize("tokens", (1, 2))
def test_admitted_shape_set(flint_gpu, tokens):
    """The admitted set, end to end: Flint TP2 banks, M in {1, 2}, top-6."""
    device, flint_moe, _ = flint_gpu
    raw, banks, plan = _layer(flint_gpu)
    ws = flint_moe.allocate_workspace(device)
    x, weights, ids = routing(tokens, torch.Generator().manual_seed(tokens), device)
    actual = _decode(flint_moe, banks, plan, ws, x, weights, ids)
    assert relative_error(actual, reference_moe(x, weights, ids, raw)) < 1e-2


def _raw_bank(device, inter, hidden=HIDDEN, experts=EXPERTS):
    i32 = {"dtype": torch.int32, "device": device}
    f16 = {"dtype": torch.float16, "device": device}
    return (
        torch.zeros(experts, hidden, 2 * inter // 8, **i32),
        torch.zeros(experts, hidden // 16, 2 * inter, **f16),
        torch.zeros(experts, inter, hidden // 8, **i32),
        torch.zeros(experts, inter // 16, hidden, **f16),
    )


@pytest.mark.parametrize(
    ("case", "inter", "tokens", "top_k"),
    [
        ("tp1_bank", 512, 1, TOP_K),  # Flint TP1: local intermediate 512
        ("tp4_bank", 128, 1, TOP_K),  # TP4: local intermediate 128
        ("m3", INTER, 3, TOP_K),
        ("top8", INTER, 1, 8),  # Qwen3.6 routing on Flint banks
    ],
)
def test_guards_reject_other_shapes(flint_gpu, case, inter, tokens, top_k):
    """The C++ guards refuse every shape but Flint TP2 (the TP4-only op
    lesson): a wrong shape must fail loudly, never run out of bounds."""
    device, flint_moe, _ = flint_gpu
    w13_w, w13_s, w2_w, w2_s = _raw_bank(device, inter)
    ws = flint_moe.allocate_workspace(device)
    x = torch.zeros(tokens, HIDDEN, dtype=torch.float16, device=device)
    ids = torch.zeros(tokens * top_k, dtype=torch.int32, device=device)
    weights = torch.zeros(tokens, top_k, dtype=torch.float32, device=device)
    act = torch.empty(tokens * top_k, inter, dtype=torch.float16, device=device)
    out = torch.empty(tokens, HIDDEN, dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError, match="expected Flint TP2 W13"):
        flint_moe.nvfp4_moe_decode_w13_swiglu_sm70_out(
            act, x, w13_w, w13_s, ids, ws.partials, ws.counters, 0
        )
    with pytest.raises(RuntimeError, match="expected Flint TP2 W2"):
        flint_moe.nvfp4_moe_decode_w2_reduce_sm70_out(
            out, act, w2_w, w2_s, ids, weights, 0
        )
    # Load-time admission refuses the other banks before any call is made.
    bank_reason = flint_moe.bank_rejection(w13_w, w13_s, w2_w, w2_s)
    assert (bank_reason is None) == (inter == INTER), bank_reason
    torch.cuda.synchronize(device)


def test_guards_reject_misaligned_input_and_short_workspace(flint_gpu):
    device, flint_moe, _ = flint_gpu
    w13_w, w13_s, _, _ = _raw_bank(device, INTER)
    ws = flint_moe.allocate_workspace(device)
    act = torch.empty(TOP_K, INTER, dtype=torch.float16, device=device)
    ids = torch.zeros(TOP_K, dtype=torch.int32, device=device)
    storage = torch.zeros(HIDDEN + 1, dtype=torch.float16, device=device)
    x = storage[1:].view(1, HIDDEN)  # 2-byte offset
    assert (
        flint_moe.call_rejection(
            x, torch.zeros(1, TOP_K, device=device), ids.view(1, TOP_K)
        )
        == "x is not 16-byte aligned"
    )
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        flint_moe.nvfp4_moe_decode_w13_swiglu_sm70_out(
            act, x, w13_w, w13_s, ids, ws.partials, ws.counters, 0
        )
    good = torch.zeros(1, HIDDEN, dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError, match="split-K workspace"):
        flint_moe.nvfp4_moe_decode_w13_swiglu_sm70_out(
            act, good, w13_w, w13_s, ids, ws.partials, ws.counters[:95], 0
        )


def test_allocate_workspace_refuses_under_capture(flint_gpu):
    device, flint_moe, _ = flint_gpu
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="graph capture"), torch.cuda.graph(graph):
        flint_moe.allocate_workspace(device)
