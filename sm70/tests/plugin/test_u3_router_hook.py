# SPDX-License-Identifier: Apache-2.0
"""U3 (patches/0004): fused-MoE router platform hook in upstream vLLM."""

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def factory():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_fused_moe_router"):
        pytest.skip("vLLM without patches/0003 (U3)")
    from vllm.model_executor.layers.fused_moe.router import router_factory

    return router_factory


def _platform_cls(factory):
    return type(factory.current_platform)


def test_default_is_none():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_fused_moe_router"):
        pytest.skip("vLLM without patches/0003 (U3)")
    assert interface.Platform.get_fused_moe_router({"top_k": 2}) is None


def test_platform_router_replaces_upstream(factory, monkeypatch):
    sentinel = object()
    seen = []

    def hook(cls, params):
        seen.append(params)
        return sentinel

    monkeypatch.setattr(_platform_cls(factory), "get_fused_moe_router", classmethod(hook))
    assert factory.create_fused_moe_router(top_k=2, global_num_experts=8) is sentinel
    (params,) = seen
    assert params["top_k"] == 2
    assert params["global_num_experts"] == 8
    assert params["renormalize"] is True
    assert params["scoring_func"] == "softmax"


def test_none_keeps_upstream_router(factory, monkeypatch):
    upstream = factory.create_fused_moe_router(top_k=2, global_num_experts=8)
    monkeypatch.setattr(
        _platform_cls(factory), "get_fused_moe_router", classmethod(lambda cls, p: None)
    )
    hooked = factory.create_fused_moe_router(top_k=2, global_num_experts=8)
    assert type(hooked) is type(upstream)
    assert vars(hooked).keys() == vars(upstream).keys()
