# SPDX-License-Identifier: Apache-2.0
"""U4 (patches/0004): V1 sampler platform hook in upstream vLLM."""

import inspect

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def interface():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_sampler"):
        pytest.skip("vLLM without patches/0004 (U4)")
    return interface


def test_default_is_none(interface):
    assert (
        interface.Platform.get_sampler(logprobs_mode="raw_logprobs", use_fp64_gumbel=False)
        is None
    )


def test_runner_asks_platform_then_falls_back(interface):
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner.__init__)
    hook = src.index("self.sampler = current_platform.get_sampler(")
    fallback = src.index(") or Sampler(", hook)
    # Both calls get the same arguments, so None gives the upstream sampler.
    hook_args = src[hook:fallback]
    fallback_args = src[fallback : src.index(")", fallback + len(") or Sampler("))]
    for arg in ("logprobs_mode=", "use_fp64_gumbel="):
        assert arg in hook_args and arg in fallback_args
