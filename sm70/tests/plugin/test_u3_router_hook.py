# SPDX-License-Identifier: Apache-2.0
"""U3 (patches/0004): MoE router provider hook at the runner gate."""

import contextlib
from types import SimpleNamespace

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def runner_mod():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_moe_router_provider"):
        pytest.skip("vLLM without patches/0004 (U3)")
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    return moe_runner


def _platform(provider):
    return SimpleNamespace(get_moe_router_provider=lambda: provider)


def _fake_runner(runner_mod, calls):
    """Just enough MoERunner state to run the real _forward_impl."""
    cls = runner_mod.MoERunner

    def gate(x):
        calls.append("gate")
        return "gate_logits", None

    def apply(**kw):
        calls.append(("apply", kw["router_logits"], kw["precomputed_topk"]))
        return None, "out"

    return SimpleNamespace(
        routed_experts=SimpleNamespace(_ensure_moe_quant_config_init=lambda: None),
        _shared_experts=None,
        gate=gate,
        _fse_fuse_gate=False,
        _sequence_parallel_context=contextlib.nullcontext,
        _maybe_dispatch=lambda h, r: (h, r),
        _apply_quant_method=apply,
        _maybe_combine=lambda shared, h: h,
        _forward_impl=cls._forward_impl,
    )


def _forward(runner_mod, monkeypatch, provider):
    calls = []
    monkeypatch.setattr(runner_mod, "current_platform", _platform(provider))
    runner = _fake_runner(runner_mod, calls)
    out = runner_mod.MoERunner._forward_impl(runner, "hidden", "caller_logits", None)
    assert out == "out"
    return runner, calls


def test_default_is_none():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_moe_router_provider"):
        pytest.skip("vLLM without patches/0004 (U3)")
    assert interface.Platform.get_moe_router_provider() is None


def test_no_provider_runs_upstream_gate(runner_mod, monkeypatch):
    _, calls = _forward(runner_mod, monkeypatch, None)
    assert calls == ["gate", ("apply", "gate_logits", None)]


def test_route_none_runs_upstream_gate(runner_mod, monkeypatch):
    route = SimpleNamespace(try_route=lambda h: None)
    _, calls = _forward(runner_mod, monkeypatch, lambda runner: route)
    assert calls == ["gate", ("apply", "gate_logits", None)]


def test_provider_none_for_layer_runs_upstream_gate(runner_mod, monkeypatch):
    _, calls = _forward(runner_mod, monkeypatch, lambda runner: None)
    assert calls == ["gate", ("apply", "gate_logits", None)]


def test_route_skips_gate_and_passes_topk(runner_mod, monkeypatch):
    seen = []

    def provider(runner):
        seen.append(runner)
        return SimpleNamespace(try_route=lambda h: ("w", "ids"))

    runner, calls = _forward(runner_mod, monkeypatch, provider)
    assert seen == [runner]
    assert calls == [("apply", "caller_logits", ("w", "ids"))]


def _apply_runner(runner_mod, monolithic, calls):
    def select_experts(**kw):
        calls.append("select_experts")
        return "sel_w", "sel_ids"

    def forward_modular(**kw):
        calls.append(("modular", kw["topk_weights"], kw["topk_ids"]))
        return "fused"

    return SimpleNamespace(
        _maybe_apply_shared_experts=lambda *a: None,
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=monolithic),
            forward_modular=forward_modular,
        ),
        router=SimpleNamespace(select_experts=select_experts),
        _quant_method=SimpleNamespace(topk_indices_dtype=None),
        _shared_experts=None,
    )


def _apply(runner_mod, runner, topk):
    return runner_mod.MoERunner._apply_quant_method(
        runner, "hidden", "logits", None, precomputed_topk=topk
    )


def test_precomputed_topk_skips_select_experts(runner_mod):
    calls = []
    out = _apply(runner_mod, _apply_runner(runner_mod, False, calls), ("w", "ids"))
    assert out == (None, "fused")
    assert calls == [("modular", "w", "ids")]


def test_no_precomputed_topk_selects_experts(runner_mod):
    calls = []
    _apply(runner_mod, _apply_runner(runner_mod, False, calls), None)
    assert calls == ["select_experts", ("modular", "sel_w", "sel_ids")]


def test_monolithic_rejects_precomputed_topk(runner_mod):
    with pytest.raises(AssertionError, match="modular kernel"):
        _apply(runner_mod, _apply_runner(runner_mod, True, []), ("w", "ids"))
