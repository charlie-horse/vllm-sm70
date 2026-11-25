# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routing kernels (inkling2_sm70/ops/routing.py) and the fused
routing of moe.py, on CPU.

- the kill switch VLLM_SM70_INKLING_FUSED_ROUTING and the route choice;
- through Triton's interpreter: the router GEMV against F.linear in fp32;
  ``route`` against the torch routing (identical ids, weights within 1e-6,
  padding rows, the lowest-id tie rule, the tiny checkpoint's recorded
  router); ``route_blocks`` and the module route of a large batch that is
  not decode-only against the torch routing (VLLM_SM70_INKLING_MIXED_SMALL_OPS);
  the slot sort, group offsets, row gather and ``combine`` against
  moe_tm.py; the module forward on the grouped and the TurboMind route
  against the switch-off forward, padding rows zero, without a host sync.
The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). The AOT compile check is
test_m12_inkling2_triton_sm70_compile_cpu.py; the GPU cases are
test_m12_inkling2_routing_gpu.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
import _inkling2_routing as cases
from inkling2_sync_guard import allowed, no_host_sync

OPS = HERE.parents[1] / "vllm_sm70" / "models" / "inkling2_sm70" / "ops"
TINY = HERE.parent / "models" / "fixtures" / "inkling2_tiny"
SWITCH = "VLLM_SM70_INKLING_FUSED_ROUTING"
MIXED_SWITCH = "VLLM_SM70_INKLING_MIXED_SMALL_OPS"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
# Inkling 2 Mini: hidden 2304, 128 experts, top-8, route scale 8.
MINI = (2304, 128, 8, 8.0)
SMALL = (64, 8, 2, 2.0)


def _moe():
    from vllm_sm70.models.inkling2_sm70 import moe

    return moe


def _routing():
    from vllm_sm70.models.inkling2_sm70.ops import routing

    return routing


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


def inputs(tokens, size, seed, dtype=torch.float16):
    hidden, experts, _, _ = size
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g).to(dtype)
    weight = torch.randn(experts, hidden, generator=g) / hidden**0.5
    bias = torch.randn(experts, generator=g) * 0.05
    return x, weight, bias


def padding(tokens, rows):
    flags = torch.zeros(tokens, dtype=torch.bool)
    flags[list(rows)] = True
    return flags


# --- source, switch and route choice --------------------------------------------------


def test_kernel_source_holds_nothing_sm70_rejects():
    text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / "routing.py").read_text())
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    for banned in ("tl.dot", "tanh.approx", "inline_asm", "gdc_", "bfloat16"):
        assert banned not in code, banned


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_route_off(monkeypatch, value, enabled):
    moe = _moe()
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert moe.FUSED_SWITCH == SWITCH
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["models"]
    gpu = SimpleNamespace(is_cuda=True, shape=(8, 2304))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(moe._compat, "AVAILABLE", True)
        assert moe.on_fused_routing(gpu, 8) is enabled
        # Above the slot bound, and an empty batch: the torch router.
        assert not moe.on_fused_routing(SimpleNamespace(is_cuda=True, shape=(9, 1)), 8)
        assert not moe.on_fused_routing(SimpleNamespace(is_cuda=True, shape=(0, 1)), 8)
    if not INTERPRETED:
        assert moe.on_fused_routing(torch.zeros(1, 4), 8) is False


def test_slot_bound_is_the_grouped_slot_way_and_the_decode_graph_cap():
    from vllm_sm70.models.inkling2_sm70 import config
    from vllm_sm70.models.inkling2_sm70.ops import grouped_gemm

    routing = _routing()
    assert routing.MAX_SLOTS == grouped_gemm.SLOT_MAX
    assert routing.MAX_SLOTS == config.DECODE_GRAPH_MAX_BATCH * MINI[2]


def test_a_layer_on_the_torch_expert_sum_keeps_the_torch_router(monkeypatch):
    moe = _moe()
    module = grouped_module(torch.float16)
    monkeypatch.setattr(moe, "on_fused_routing", lambda x, k: True)
    monkeypatch.setattr(moe, "on_grouped", lambda x: False)
    assert module.tm_route is None and not module._fused_routing(torch.zeros(1, 64))
    monkeypatch.setattr(moe, "on_grouped", lambda x: True)
    assert module._fused_routing(torch.zeros(1, 64))


# --- interpreted: router --------------------------------------------------------------


@interpreted
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("size", [SMALL, MINI], ids=["small", "mini"])
def test_interpreted_router_logits_match_f_linear(dtype, size):
    x, weight, _ = inputs(3, size, 0, dtype)
    got = _routing().router_logits(x, weight)
    want = torch.nn.functional.linear(x.float(), weight)
    assert got.dtype == torch.float32
    assert float((got - want).abs().max()) <= 2e-6 * max(1.0, float(want.abs().max()))


@interpreted
@pytest.mark.parametrize("tokens", [1, 2, 5, 8])
@pytest.mark.parametrize("size", [SMALL, MINI], ids=["small", "mini"])
@pytest.mark.parametrize("pad_id", [-1, None], ids=["pad-1", "padE"])
def test_interpreted_route_matches_the_torch_routing(tokens, size, pad_id):
    _, experts, top_k, scale = size
    pad_id = experts if pad_id is None else pad_id
    x, weight, bias = inputs(tokens, size, tokens)
    logits = torch.nn.functional.linear(x.float(), weight)
    flags = padding(tokens, [tokens // 2] if tokens > 1 else [])
    routed = _routing().route(logits, bias, top_k, scale, flags, pad_id=pad_id)
    want = cases.torch_route(logits, bias, top_k, scale, flags, pad_id)
    assert cases.check_route(routed, *want) <= cases.WEIGHT_TOL
    if tokens > 1:
        assert bool((routed.ids[tokens // 2] == pad_id).all())
        assert bool((routed.weights[tokens // 2] == 0).all())
    assert routed.pos is None
    unpadded = _routing().route(logits, bias, top_k, scale, None, pad_id=pad_id)
    want = cases.torch_route(logits, bias, top_k, scale)
    assert cases.check_route(unpadded, *want) <= cases.WEIGHT_TOL


@interpreted
@pytest.mark.parametrize("size", [SMALL, MINI], ids=["small", "mini"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interpreted_ties_keep_the_lowest_expert_id(size, seed):
    _, experts, top_k, scale = size
    tokens = 8 if top_k == 8 else 16
    logits = cases.tie_logits(tokens, experts, seed)
    bias = torch.zeros(experts)
    if seed == 2:  # ties made of sigmoid + bias
        bias = cases.tie_logits(1, experts, seed + 10)[0] * 0.5
    flags = padding(tokens, [3])
    routed = _routing().route(logits, bias, top_k, scale, flags, pad_id=experts)
    want = cases.lowest_id_route(logits, bias, top_k, scale, flags, experts)
    assert cases.check_route(routed, *want) <= cases.WEIGHT_TOL
    # Every row has a tie across the top-k edge, so this checks the rule.
    choice = (torch.sigmoid(logits) + bias).sort(dim=1, descending=True).values
    live = ~flags
    assert bool((choice[live, top_k - 1] == choice[live, top_k]).any())


@interpreted
def test_interpreted_all_equal_logits_choose_the_first_experts():
    logits = torch.zeros(2, 128)
    routed = _routing().route(logits, torch.zeros(128), 8, 8.0, None, pad_id=128)
    assert routed.ids.tolist() == [list(range(8))] * 2
    assert torch.allclose(routed.weights, torch.ones(2, 8), atol=1e-6, rtol=0)


@interpreted
def test_interpreted_tiny_checkpoint_router_end_to_end():
    """Layer 1's recorded router (the reference's fp32 routing of 20 real
    hidden states): same experts, weights within 1e-6, and a strictly
    positive gap at every top-k edge, so no tie rule decides a pick."""
    from safetensors.torch import load_file

    native = load_file(TINY / "model.safetensors")
    ops = load_file(TINY / "ops.safetensors")
    config = json.loads((TINY / "config.json").read_text())["text_config"]
    top_k, scale = config["num_experts_per_tok"], config["route_scale"]
    weight = native["model.llm.layers.1.mlp.gate.weight"].float()
    bias = native["model.llm.layers.1.mlp.gate.bias"].float()
    x = ops["router.0.in.hidden_states"]
    logits = _routing().router_logits(x, weight)
    assert float((logits - ops["router.0.out.0"]).abs().max()) <= 1e-5
    routed = _routing().route(logits, bias, top_k, scale, None, pad_id=-1)
    want = cases.by_id(ops["router.0.out.2"], ops["router.0.out.1"])
    assert cases.check_route(routed, *want) <= cases.WEIGHT_TOL
    choice = (torch.sigmoid(ops["router.0.out.0"]) + bias).sort(1, descending=True)
    gap = choice.values[:, top_k - 1] - choice.values[:, top_k]
    assert float(gap.min()) > 1e-4


# --- interpreted: route_blocks ---------------------------------------------------------


@interpreted
@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 40])
@pytest.mark.parametrize("size", [SMALL, MINI], ids=["small", "mini"])
def test_interpreted_route_blocks_match_the_torch_routing(tokens, size):
    _, experts, top_k, scale = size
    x, weight, bias = inputs(tokens, size, 20 + tokens)
    logits = torch.nn.functional.linear(x.float(), weight)
    rows = [0, tokens // 2, tokens - 1] if tokens > 2 else []
    flags = padding(tokens, rows)
    for pad in (flags, None):
        weights, ids = _routing().route_blocks(
            logits, bias, top_k, scale, pad, pad_id=experts
        )
        assert ids.dtype == torch.int64 and weights.dtype == torch.float32
        want_ids, want_weights = cases.torch_route(logits, bias, top_k, scale, pad)
        assert torch.equal(ids, want_ids)
        assert float((weights - want_weights).abs().max()) <= cases.WEIGHT_TOL
    weights, ids = _routing().route_blocks(
        logits, bias, top_k, scale, flags, pad_id=experts
    )
    assert bool((ids[rows] == experts).all()) and bool((weights[rows] == 0).all())


@interpreted
def test_interpreted_route_blocks_ties_keep_the_lowest_expert_id():
    _, experts, top_k, scale = SMALL
    logits = cases.tie_logits(33, experts, 3)
    bias = torch.zeros(experts)
    weights, ids = _routing().route_blocks(logits, bias, top_k, scale, None, pad_id=-1)
    want_ids, want_weights = cases.lowest_id_route(logits, bias, top_k, scale)
    assert torch.equal(ids, want_ids)
    assert float((weights - want_weights).abs().max()) <= cases.WEIGHT_TOL


@interpreted
@pytest.mark.parametrize("decode_only", [False, True], ids=["mixed", "decode_only"])
def test_interpreted_module_route_of_a_large_batch(monkeypatch, decode_only):
    """Above the slot bound, a batch that is not decode-only routes through
    route_blocks; VLLM_SM70_INKLING_MIXED_SMALL_OPS=0 and a decode-only
    batch keep the torch router. Same ids, weights within the tolerance."""
    module = grouped_module(torch.float32)
    tokens = 40
    x = inputs(tokens, (HIDDEN, EXPERTS, TOP_K, 2.0), 7)[0]
    flags = padding(tokens, [3, tokens - 1])
    calls = []
    launch = _routing().route_blocks
    monkeypatch.setattr(
        _routing(), "route_blocks", lambda *a, **k: calls.append(1) or launch(*a, **k)
    )
    monkeypatch.delenv(MIXED_SWITCH, raising=False)
    logits, weights, ids = module.route(x, flags, decode_only)
    assert len(calls) == (0 if decode_only else 1)
    monkeypatch.setenv(MIXED_SWITCH, "0")
    want_logits, want_weights, want_ids = module.route(x, flags, decode_only)
    assert len(calls) == (0 if decode_only else 1)
    assert torch.equal(logits, want_logits)
    ids, weights = cases.by_id(ids, weights)
    assert torch.equal(ids, cases.by_id(want_ids, want_weights)[0])
    want = cases.by_id(want_ids, want_weights)[1]
    assert float((weights - want).abs().max()) <= cases.WEIGHT_TOL
    assert bool((ids[[3, tokens - 1]] == EXPERTS).all())


# --- interpreted: dispatch and combine ------------------------------------------------


@interpreted
@pytest.mark.parametrize(
    "tokens, pad", [(1, ()), (3, (1,)), (8, (0, 7)), (4, (0, 1, 2, 3))]
)
@pytest.mark.parametrize("size", [SMALL, MINI], ids=["small", "mini"])
def test_interpreted_dispatch_matches_the_torch_sort(tokens, pad, size):
    _, experts, top_k, scale = size
    x, weight, bias = inputs(tokens, size, 10 + tokens)
    logits = torch.nn.functional.linear(x.float(), weight)
    if tokens == 8 and size is SMALL:  # ties: experts shared across tokens
        logits = cases.tie_logits(tokens, experts, 5)
    flags = padding(tokens, pad)
    routed, rows, sorted_ids, offsets = cases.run_dispatch(
        logits, bias, top_k, scale, flags, x, pad_id=experts
    )
    want = cases.torch_route(logits, bias, top_k, scale, flags, experts)
    if tokens == 8 and size is SMALL:
        want = cases.lowest_id_route(logits, bias, top_k, scale, flags, experts)
    assert cases.check_route(routed, *want) <= cases.WEIGHT_TOL
    cases.check_dispatch(routed, x, rows, sorted_ids, offsets, experts)
    slots = tokens * top_k
    assert bool(torch.isnan(rows[slots:].float()).all())
    assert bool((sorted_ids[slots:] == -7).all())


@interpreted
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dispatch", [False, True], ids=["by_token", "sorted"])
def test_interpreted_combine_matches_the_index_add_sum(dtype, dispatch):
    hidden, experts, top_k, scale = SMALL
    tokens = 6
    x, weight, bias = inputs(tokens, SMALL, 3)
    logits = torch.nn.functional.linear(x.float(), weight)
    flags = padding(tokens, [2])
    if dispatch:
        routed = cases.run_dispatch(logits, bias, top_k, scale, flags, x, experts)[0]
    else:
        routed = _routing().route(logits, bias, top_k, scale, flags, pad_id=-1)
    g = torch.Generator().manual_seed(4)
    y = (torch.randn(tokens * top_k, hidden, generator=g) * 3).to(dtype)
    # A padding slot's row is never added, whatever it holds.
    y[routed.pos[2] if dispatch else torch.arange(2 * top_k, 3 * top_k)] = float("inf")
    out = torch.full((tokens, hidden), float("nan"), dtype=dtype)
    got = _routing().combine(y, routed, experts, out)
    assert got is out
    want = cases.combine_reference(y, routed, experts, dtype)
    assert torch.equal(out, want)
    assert bool((out[2] == 0).all())


# --- interpreted: the module ----------------------------------------------------------


EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 64, 32, 2


def grouped_module(dtype, seed=0):
    config = SimpleNamespace(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_per_tok=TOP_K,
        route_scale=2.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        module = _moe().Inkling2Sm70Experts(config, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    module.w13.copy_(torch.randn(module.w13.shape, generator=g) * 0.3)
    module.w2.copy_(torch.randn(module.w2.shape, generator=g) * 0.3)
    module.gate.weight.data.copy_(torch.randn(EXPERTS, HIDDEN, generator=g))
    module.gate.bias.data.copy_(torch.randn(EXPERTS, generator=g) * 0.1)
    return module


@pytest.fixture
def tm_module(monkeypatch):
    """Layer 1 (NVFP4) of the tiny checkpoint on the TurboMind route of
    moe_tm.py, with the pure-torch tm of the NVFP4 TurboMind test."""
    sys.path.insert(0, str(HERE.parent / "quant"))
    import test_m12_inkling2_nvfp4_tm_cpu as tm_test

    from vllm_sm70.models.inkling2_sm70 import moe_tm

    monkeypatch.setenv(tm_test.SWITCH, "1")
    mock = tm_test.make_mock_tm()
    ops = types.ModuleType("vllm_sm70.ops")
    ops.tm = mock
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", mock)
    monkeypatch.setattr(
        moe_tm, "INKLING_CONTRACTS", moe_tm.INKLING_CONTRACTS | {tm_test.TINY_CONTRACT}
    )
    monkeypatch.setattr(moe_tm, "INKLING_TP_SIZES", (1, 2))
    moe_tm.Workspace.clear()
    from vllm_sm70.models.inkling2_sm70 import sm70_configs

    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    module = tm_test.build(config)
    assert module.tm_route is not None
    module.tm_route.tm = types.SimpleNamespace(
        nvfp4_moe_dense_stage_sm70_out=allowed(mock.nvfp4_moe_dense_stage_sm70_out)
    )
    yield module, mock
    moe_tm.Workspace.clear()


def forward(module, x, flags):
    out = torch.full_like(x, float("nan"))
    module(x, out, flags)
    return out


@interpreted
@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_interpreted_grouped_module_forward(monkeypatch, tokens):
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.delenv("VLLM_SM70_INKLING_GROUPED_MOE", raising=False)
    module = grouped_module(torch.float16)
    x = inputs(tokens, (HIDDEN, EXPERTS, TOP_K, 2.0), tokens)[0]
    flags = padding(tokens, [tokens - 1] if tokens > 1 else [])
    assert module._fused_routing(x)
    calls = []
    monkeypatch.setattr(module, "route", lambda *a: calls.append(a))
    with no_host_sync():
        got = forward(module, x, flags)
    assert calls == []
    monkeypatch.undo()
    monkeypatch.setenv(SWITCH, "0")
    assert not module._fused_routing(x)
    want = forward(module, x, flags)
    assert max_abs_rel(got, want) <= 2e-3
    if tokens > 1:
        assert bool((got[-1] == 0).all())


@interpreted
@pytest.mark.parametrize("tokens", [1, 2, 8, 20, 40])
@pytest.mark.parametrize("compact", [True, False])
def test_interpreted_turbomind_module_forward(monkeypatch, tm_module, tokens, compact):
    module, mock = tm_module
    module.tm_route.compact = compact
    x = cases_hidden(tokens)
    flags = padding(tokens, [0] if tokens > 1 else [])
    monkeypatch.delenv(SWITCH, raising=False)
    fused = module._fused_routing(x)
    assert fused is (tokens * TOP_K <= _routing().MAX_SLOTS)
    mock.stages.clear()
    with no_host_sync():
        got = forward(module, x, flags)
    stages = list(mock.stages)
    monkeypatch.setenv(SWITCH, "0")
    assert not module._fused_routing(x)
    mock.stages.clear()
    want = forward(module, x, flags)
    # The same stage calls: groups, offsets and expert ids.
    assert len(stages) == len(mock.stages) == 2
    for a, b in zip(stages, mock.stages, strict=True):
        assert a[:4] == b[:4]
        assert torch.equal(a[4], b[4]) and torch.equal(a[5], b[5])
    assert torch.equal(got, want) or max_abs_rel(got, want) <= 1e-3
    if tokens > 1:
        assert bool((got[0] == 0).all())


def cases_hidden(tokens):
    from safetensors.torch import load_file

    x = load_file(TINY / "ops.safetensors")["router.0.in.hidden_states"]
    return x[torch.arange(tokens) % x.shape[0]].half()


@interpreted
@pytest.mark.parametrize("route", ["grouped", "turbomind"])
def test_interpreted_fused_forward_ops_do_not_depend_on_the_routing(
    monkeypatch, request, route
):
    monkeypatch.delenv(SWITCH, raising=False)
    if route == "grouped":
        module = grouped_module(torch.float16)
    else:
        module = request.getfixturevalue("tm_module")[0]
        # A stage that runs the same ops for any routing (the mock's do not).
        module.tm_route.tm = types.SimpleNamespace(
            nvfp4_moe_dense_stage_sm70_out=lambda out, *args: out.zero_()
        )
    traces = []
    for seed in (5, 6):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(4, HIDDEN, generator=g).half()
        trace: list = []
        out = torch.empty_like(x)
        with no_host_sync(trace):
            module(x, out, padding(4, [seed - 5]))
        traces.append(trace)
        assert bool(torch.isfinite(out).all())
    assert traces[0] == traces[1] and len(traces[0]) > 0


# --- interpreted child run ------------------------------------------------------------


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    for name in (SWITCH, MIXED_SWITCH, "VLLM_SM70_INKLING_GROUPED_MOE"):
        env.pop(name, None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", __file__, "-q", "-p",
            "no:cacheprovider", "-m", "not gpu", "-k",
            "interpreted and not test_interpreted_suite",
        ],
        env=env, capture_output=True, text=True, timeout=3600, check=False,
    )  # fmt: skip
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 53, summary
