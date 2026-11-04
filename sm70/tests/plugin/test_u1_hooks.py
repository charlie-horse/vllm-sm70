# SPDX-License-Identifier: Apache-2.0
"""U1 (patches/0001): platform GEMM and warmup hooks in upstream vLLM."""

import inspect

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def interface():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_unquantized_gemm"):
        pytest.skip("vLLM without patches/0001 (U1)")
    return interface


def test_defaults(interface):
    assert interface.Platform.get_unquantized_gemm("auto") is None
    assert interface.Platform.kernel_warmup(object()) is None


def test_dispatch_uses_platform_gemm(interface, monkeypatch):
    from vllm.model_executor.layers import utils

    def gemm(layer, x, weight, bias=None):
        return "sm70"

    seen = []

    class FakePlatform:
        @staticmethod
        def get_unquantized_gemm(linear_backend):
            seen.append(linear_backend)
            return gemm

    monkeypatch.setattr(utils, "current_platform", FakePlatform)
    assert utils.dispatch_unquantized_gemm() is gemm
    assert utils.dispatch_unquantized_gemm("flashinfer") is gemm
    assert seen == ["auto", "flashinfer"]


def test_kernel_warmup_calls_platform_before_capture_only_return(interface):
    from vllm.model_executor.warmup import kernel_warmup

    src = inspect.getsource(kernel_warmup.kernel_warmup)
    hook = src.index("current_platform.kernel_warmup(worker)")
    assert hook < src.index("if process_local_only:")
