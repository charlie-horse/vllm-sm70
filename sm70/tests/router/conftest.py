# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K13 fused-router tests.

CPU tests need torch and vLLM's Python source. GPU tests are marked `gpu`
and need an SM70 device, a built vllm_sm70._sm70_C with the K13 op, and a
built vLLM (the reference route uses its topk_sigmoid). From the repo root:

    pytest -m gpu sm70/tests/router
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm.model_executor.layers.fused_moe.router.gate_linear")


def make_gate(
    shape=(256, 2048), dtype=torch.float32, out_dtype=torch.float32, device="cpu"
):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.zeros(shape, dtype=dtype, device=device), requires_grad=False
    )
    gate.bias = None
    gate.out_dtype = out_dtype
    gate.is_unquantized = True
    return gate


def make_router(**kw):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )

    args = dict(
        top_k=6, global_num_experts=256, scoring_func="sigmoid", renormalize=True
    )
    args.update(kw)
    return FusedTopKRouter(**args)


def make_runner(**overrides):
    fields = dict(
        layer_name="model.layers.2.mlp.experts",
        gate=make_gate(),
        router=make_router(),
        _fse_fuse_gate=False,
        routed_input_transform=None,
        enable_dbo=False,
        _quant_method=SimpleNamespace(is_monolithic=False, topk_indices_dtype=None),
        moe_config=SimpleNamespace(dp_size=1, pcp_size=1, use_ep=False),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def router_mod(monkeypatch):
    from vllm_sm70 import router

    monkeypatch.delenv("VLLM_SM70_FLINT_FUSED_ROUTER", raising=False)
    router._logged.clear()
    return router


@pytest.fixture
def admitted(router_mod, monkeypatch):
    """Admission on the CPU: SM70 check and the op are faked; returns calls."""
    calls = []

    def op(*args):
        calls.append(args)

    monkeypatch.setattr(router_mod, "_is_exact_sm70", lambda device: True)
    monkeypatch.setattr(router_mod, "_flint_route_op", lambda: op)
    return calls


@pytest.fixture
def router_gpu(router_mod):
    """vllm_sm70.router on a real SM70 GPU with flint_route built, else skip."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    # On a real SM70 GPU a missing op is a broken build or a shadowing
    # PYTHONPATH, not a reason to skip.
    if not _ext.is_loaded():
        pytest.fail(
            "SM70 GPU present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    if not router_mod.has_op():
        pytest.fail("_sm70_C is loaded but has no flint_route (built without K13)")
    return router_mod
