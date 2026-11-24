# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routing kernels (inkling2_sm70/ops/routing.py) on an SM70 GPU.

At the Inkling 2 Mini sizes (hidden 2304, 128 experts, top-8, route scale 8)
and decode batches of 1 to 8 tokens:
- the router GEMV against cuBLAS F.linear in fp32;
- ``route`` against the torch routing on CUDA (torch.topk's CUDA kernel),
  identical ids and weights within 1e-6, also on tied logits;
- the slot sort, offsets, row gather and ``combine`` against moe_tm.py;
- the grouped and the TurboMind module routes against the switch-off
  forward, and a CUDA graph of the fused routing captured with
  ``torch.cuda.set_sync_debug_mode("error")`` and replayed on new inputs.
CPU cases: test_m12_inkling2_routing_cpu.py.

    pytest -m gpu sm70/tests/ops/test_m12_inkling2_routing_gpu.py
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.gpu

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _inkling2_routing as cases

HIDDEN, EXPERTS, TOP_K, SCALE = 2304, 128, 8, 8.0
INTERMEDIATE = 1152
TOKENS = [1, 2, 4, 8]
SWITCH = "VLLM_SM70_INKLING_FUSED_ROUTING"


@pytest.fixture(autouse=True)
def _gpu(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    monkeypatch.delenv(SWITCH, raising=False)
    torch.cuda.set_device(0)


def _routing():
    from vllm_sm70.models.inkling2_sm70.ops import routing

    return routing


def inputs(tokens, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=g).half().cuda()
    weight = (torch.randn(EXPERTS, HIDDEN, generator=g) / HIDDEN**0.5).cuda()
    bias = (torch.randn(EXPERTS, generator=g) * 0.05).cuda()
    return x, weight, bias


def padding(tokens, rows):
    flags = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    flags[list(rows)] = True
    return flags


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


@pytest.mark.parametrize("tokens", TOKENS)
def test_router_logits_match_cublas(tokens):
    x, weight, _ = inputs(tokens, tokens)
    got = _routing().router_logits(x, weight)
    want = torch.nn.functional.linear(x.float(), weight)
    assert float((got - want).abs().max()) <= 1e-5 * float(want.abs().max())


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("pad_id", [-1, EXPERTS])
def test_route_matches_the_torch_routing(tokens, pad_id):
    x, weight, bias = inputs(tokens, 10 + tokens)
    logits = torch.nn.functional.linear(x.float(), weight)
    flags = padding(tokens, [tokens - 1] if tokens > 1 else [])
    routed = _routing().route(logits, bias, TOP_K, SCALE, flags, pad_id=pad_id)
    want = cases.torch_route(logits, bias, TOP_K, SCALE, flags, pad_id)
    assert cases.check_route(routed, *want) <= cases.WEIGHT_TOL


@pytest.mark.parametrize("seed", range(6))
def test_ties_match_torch_topk_on_cuda(seed):
    logits = cases.tie_logits(8, EXPERTS, seed, "cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    if seed % 2:
        bias = cases.tie_logits(1, EXPERTS, seed + 10, "cuda")[0] * 0.5
    routed = _routing().route(logits, bias, TOP_K, SCALE, None, pad_id=EXPERTS)
    torch_ids = cases.torch_route(logits, bias, TOP_K, SCALE)
    lowest = cases.lowest_id_route(logits, bias, TOP_K, SCALE)
    assert torch.equal(torch_ids[0], lowest[0])  # torch.topk on CUDA: lowest id
    assert cases.check_route(routed, *torch_ids) <= cases.WEIGHT_TOL


@pytest.mark.parametrize("tokens", TOKENS)
def test_dispatch_and_combine_match_the_torch_sort(tokens):
    x, weight, bias = inputs(tokens, 20 + tokens)
    logits = _routing().router_logits(x, weight)
    flags = padding(tokens, [0] if tokens > 1 else [])
    routed, rows, sorted_ids, offsets = cases.run_dispatch(
        logits, bias, TOP_K, SCALE, flags, x, pad_id=EXPERTS
    )
    cases.check_dispatch(routed, x, rows, sorted_ids, offsets, EXPERTS)
    y = torch.randn(tokens * TOP_K, HIDDEN, device="cuda").half()
    out = torch.empty_like(x)
    _routing().combine(y, routed, EXPERTS, out)
    want = cases.combine_reference(y, routed, EXPERTS, torch.float32)
    assert float((out.float() - want).abs().max()) <= 2e-3 * float(want.abs().max())


def grouped_module():
    from vllm_sm70.models.inkling2_sm70 import moe

    config = SimpleNamespace(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_per_tok=TOP_K,
        route_scale=SCALE,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        module = moe.Inkling2Sm70Experts(config, tp_size=2).cuda()
    g = torch.Generator().manual_seed(0)
    module.w13.copy_(torch.randn(module.w13.shape, generator=g) * 0.05)
    module.w2.copy_(torch.randn(module.w2.shape, generator=g) * 0.05)
    module.gate.weight.data.copy_(torch.randn(EXPERTS, HIDDEN, generator=g) / 48)
    module.gate.bias.data.copy_(torch.randn(EXPERTS, generator=g) * 0.05)
    return module


def tm_module():
    sys.path.insert(0, str(HERE.parent / "quant"))
    import test_m12_inkling2_nvfp4_tm_gpu as tm_test

    from vllm_sm70.models.inkling2_sm70 import moe, moe_tm

    if moe_tm.turbomind() is None:
        pytest.skip("_sm70_C lacks the TurboMind ops")
    module = grouped_module()
    module.tm_route = moe_tm.build(
        tm_test.random_codes("w13", INTERMEDIATE, HIDDEN, 1),
        tm_test.random_codes("w2", HIDDEN, INTERMEDIATE // 2, 2),
        device=torch.device("cuda"), gate_cap=4.0, up_cap=25.0, top_k=TOP_K,
    )  # fmt: skip
    assert isinstance(module, moe.Inkling2Sm70Experts)
    return module


def forward(module, x, flags):
    out = torch.full_like(x, float("nan"))
    module(x, out, flags)
    return out


@pytest.mark.parametrize("route", ["grouped", "turbomind"])
@pytest.mark.parametrize("tokens", TOKENS)
def test_module_forward_matches_the_switch_off_forward(monkeypatch, route, tokens):
    module = grouped_module() if route == "grouped" else tm_module()
    x = inputs(tokens, 30 + tokens)[0]
    flags = padding(tokens, [tokens - 1] if tokens > 1 else [])
    assert module._fused_routing(x)
    got = forward(module, x, flags)
    monkeypatch.setenv(SWITCH, "0")
    assert not module._fused_routing(x)
    want = forward(module, x, flags)
    assert max_abs_rel(got, want) <= 2e-3
    if tokens > 1:
        assert bool((got[-1] == 0).all())


@pytest.mark.parametrize("route", ["grouped", "turbomind"])
def test_fused_routing_captures_and_replays(route):
    module = grouped_module() if route == "grouped" else tm_module()
    tokens = 4
    x = inputs(tokens, 40)[0]
    flags = padding(tokens, [])
    out = torch.empty_like(x)
    module(x, out, flags)  # eager warmup (Triton compiles, tuning measures)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.graph(graph):
            module(x, out, flags)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for seed in (41, 42):
        x.copy_(inputs(tokens, seed)[0])
        flags.copy_(padding(tokens, [seed % tokens]))
        graph.replay()
        want = forward(module, x, flags)
        assert torch.equal(out, want)
