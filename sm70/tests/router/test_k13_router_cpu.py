# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the K13 fused router: admission, metadata gate, kill
switch, the U3 provider contract and capture safety (R4).

Admission cases ported from fork cf9492a
tests/kernels/moe/test_sm70_flint_router_admission.py (227 lines, 3629e29f6);
the fork's policy-object test is replaced by the env kill switch test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from conftest import make_gate, make_router, make_runner


def test_kill_switch_is_registered():
    from vllm_sm70 import envs

    assert "VLLM_SM70_FLINT_FUSED_ROUTER" in envs._SECTIONS["fused_moe"]


@pytest.mark.parametrize(
    "raw,on", [(None, True), ("1", True), ("", True), ("0", False)]
)
def test_kill_switch_only_zero_disables(router_mod, monkeypatch, raw, on):
    if raw is not None:
        monkeypatch.setenv("VLLM_SM70_FLINT_FUSED_ROUTER", raw)
    assert router_mod.enabled() is on
    assert (router_mod.provider() is router_mod.admit_flint_fused_router) is on


def test_admits_flint_router(admitted, router_mod):
    state = router_mod.admit_flint_fused_router(make_runner())
    assert isinstance(state, router_mod.FlintFusedRouter)
    assert state.logits.numel() == 2 * 256 and state.logits.dtype == torch.float32
    assert state.counter.tolist() == [0] and state.counter.dtype == torch.int32


def test_kill_switch_blocks_admission(admitted, router_mod, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLINT_FUSED_ROUTER", "0")
    assert router_mod.admit_flint_fused_router(make_runner()) is None


def test_missing_operator_keeps_generic_route(admitted, router_mod, monkeypatch):
    monkeypatch.setattr(router_mod, "_flint_route_op", lambda: None)
    assert router_mod.admit_flint_fused_router(make_runner()) is None


def test_non_sm70_keeps_generic_route(admitted, router_mod, monkeypatch):
    monkeypatch.setattr(router_mod, "_is_exact_sm70", lambda device: False)
    assert router_mod.admit_flint_fused_router(make_runner()) is None


def test_no_admission_under_capture(admitted, router_mod, monkeypatch):
    """R4: the workspace never comes from a graph pool."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert router_mod.admit_flint_fused_router(make_runner()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(gate=None),
        dict(gate=torch.nn.Linear(2048, 256, bias=False)),
        dict(gate=make_gate(shape=(512, 2048))),
        dict(gate=make_gate(shape=(256, 2560))),
        dict(gate=make_gate(dtype=torch.float16)),  # upstream Flint at --dtype half
        dict(gate=make_gate(out_dtype=torch.float16)),
        dict(gate=make_gate(out_dtype=None)),
        dict(router=make_router(scoring_func="softmax")),
        dict(router=make_router(top_k=8)),
        dict(router=make_router(renormalize=False)),
        dict(router=make_router(global_num_experts=128)),
        dict(router=make_router(eplb_state=object())),
        dict(_fse_fuse_gate=True),
        dict(routed_input_transform=torch.nn.Identity()),
        dict(enable_dbo=True),
        dict(
            _quant_method=SimpleNamespace(is_monolithic=True, topk_indices_dtype=None)
        ),
        dict(moe_config=SimpleNamespace(dp_size=2, pcp_size=1, use_ep=False)),
        dict(moe_config=SimpleNamespace(dp_size=1, pcp_size=2, use_ep=False)),
        dict(moe_config=SimpleNamespace(dp_size=1, pcp_size=1, use_ep=True)),
    ],
)
def test_other_layers_keep_generic_route(admitted, router_mod, overrides):
    assert router_mod.admit_flint_fused_router(make_runner(**overrides)) is None


def test_biased_or_quantized_gate_is_rejected(admitted, router_mod):
    gate = make_gate()
    gate.bias = torch.nn.Parameter(torch.zeros(256))
    assert router_mod.admit_flint_fused_router(make_runner(gate=gate)) is None
    gate = make_gate()
    gate.is_unquantized = False
    assert router_mod.admit_flint_fused_router(make_runner(gate=gate)) is None


@pytest.mark.parametrize(
    ("shape", "dtype", "accepted"),
    [
        ((1, 2048), torch.float16, True),
        ((2, 2048), torch.float16, True),
        ((3, 2048), torch.float16, False),
        ((0, 2048), torch.float16, False),
        ((1, 2048), torch.bfloat16, False),
        ((1, 2048), torch.float32, False),
        ((2, 2560), torch.float16, False),
    ],
)
def test_metadata_gate(router_mod, shape, dtype, accepted):
    x = torch.zeros(shape, dtype=dtype)
    assert router_mod.FlintFusedRouter.accepts(x) is accepted


def test_metadata_gate_rejects_odd_strides(router_mod):
    base = torch.zeros(2, 2050, dtype=torch.float16)
    accepts = router_mod.FlintFusedRouter.accepts
    assert not accepts(base[:, :2048])
    assert not accepts(base[:, 1:2049])
    assert not accepts(torch.zeros(2048, 2, dtype=torch.float16).T)


def test_try_route_defers_to_generic_route(admitted, router_mod):
    runner = make_runner()
    state = router_mod.admit_flint_fused_router(runner)
    x = torch.zeros(2, 2048, dtype=torch.float16)

    weights, ids = state.try_route(x)
    assert weights.shape == ids.shape == (2, 6)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    (op_args,) = admitted
    assert op_args[2] is x and op_args[3] is runner.gate.weight
    assert op_args[4] is state.logits and op_args[5] is state.counter

    assert state.try_route(torch.zeros(3, 2048, dtype=torch.float16)) is None
    runner.router.capture_fn = lambda ids: None
    assert state.try_route(x) is None
    runner.router.capture_fn = None
    runner.router._routing_replay_out = torch.zeros(4, 6, dtype=torch.int16)
    assert state.try_route(x) is None
    runner.router._routing_replay_out = None
    runner._quant_method.topk_indices_dtype = torch.int64
    assert state.try_route(x) is None
    runner._quant_method.topk_indices_dtype = torch.int32
    assert state.try_route(x) is not None
    assert len(admitted) == 2


def _u3_hook():
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    hook = getattr(MoERunner, "_maybe_precomputed_route", None)
    if hook is None:
        pytest.skip("vLLM without the U3 router-provider hook (Penny's patch)")
    return hook


def _bindable_runner(**kw):
    return make_runner(
        router_provider=None,
        _precomputed_route=None,
        _precomputed_route_decided=False,
        **kw,
    )


def test_u3_runner_hook_admits_once_outside_capture(admitted, router_mod, monkeypatch):
    hook = _u3_hook()
    x = torch.zeros(1, 2048, dtype=torch.float16)
    runner = _bindable_runner()
    # Unbound layers (every other model) keep the gate route untouched.
    assert hook(runner, x) is None and not runner._precomputed_route_decided

    calls = []

    def provider(layer):
        calls.append(layer)
        return router_mod.admit_flint_fused_router(layer)

    runner.router_provider = provider
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert hook(runner, x) is None and not calls
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    weights, ids = hook(runner, x)
    assert weights.shape == ids.shape == (1, 6)
    assert hook(runner, x) is not None
    assert calls == [runner] and len(admitted) == 2


class _Model(torch.nn.Module):
    def __init__(self, *runners):
        super().__init__()
        self.layers = torch.nn.ModuleList(runners)


class _Runner(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        for k, v in vars(_bindable_runner(**kw)).items():
            object.__setattr__(self, k, v)


def test_prepare_model_admits_before_capture(admitted, router_mod):
    """R4: kernel_warmup allocates every workspace eagerly, once."""
    good, other = _Runner(), _Runner(router=make_router(top_k=8))
    model = _Model(good, other, torch.nn.Linear(2, 2))
    assert router_mod.prepare_model(model) == 1
    assert isinstance(good._precomputed_route, router_mod.FlintFusedRouter)
    assert good._precomputed_route_decided and other._precomputed_route_decided
    assert other._precomputed_route is None
    first = good._precomputed_route
    assert router_mod.prepare_model(model) == 1
    assert good._precomputed_route is first  # no re-allocation


def test_prepare_model_respects_foreign_provider_and_capture(
    admitted, router_mod, monkeypatch
):
    foreign = _Runner()
    foreign.router_provider = lambda layer: None
    assert router_mod.prepare_model(_Model(foreign)) == 0
    assert not foreign._precomputed_route_decided

    fresh = _Runner()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert router_mod.prepare_model(_Model(fresh)) == 0
    assert not fresh._precomputed_route_decided


def test_prepare_model_noop_with_kill_switch_or_without_u3(
    admitted, router_mod, monkeypatch
):
    plain = torch.nn.Module()  # no router_provider attribute: no U3
    assert router_mod.prepare_model(_Model(plain)) == 0
    runner = _Runner()
    monkeypatch.setenv("VLLM_SM70_FLINT_FUSED_ROUTER", "0")
    assert router_mod.prepare_model(_Model(runner)) == 0
    assert runner.router_provider is None


def test_platform_hooks(router_mod, monkeypatch):
    import importlib
    import sys
    import types

    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:  # CPU host: vllm.platforms.cuda imports it (UP :23)
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    from vllm_sm70._cuda_platform import Sm70CudaPlatform

    assert Sm70CudaPlatform.get_moe_router_provider() is (
        router_mod.admit_flint_fused_router
    )
    monkeypatch.setenv("VLLM_SM70_FLINT_FUSED_ROUTER", "0")
    assert Sm70CudaPlatform.get_moe_router_provider() is None
    seen = []
    monkeypatch.setattr(router_mod, "prepare_model", seen.append)
    model = torch.nn.Linear(2, 2)
    Sm70CudaPlatform.kernel_warmup(SimpleNamespace(get_model=lambda: model))
    assert seen == [model]
    assert Sm70CudaPlatform.kernel_warmup(object()) is None
