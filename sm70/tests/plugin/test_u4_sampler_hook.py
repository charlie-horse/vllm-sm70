# SPDX-License-Identifier: Apache-2.0
"""U4 (patches/0005): V1 hidden-state sampler hook before compute_logits."""

import inspect

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def interface():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_hidden_state_sampler"):
        pytest.skip("vLLM without patches/0005 (U4)")
    return interface


@pytest.fixture
def runner_cls(interface):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    return GPUModelRunner


def test_default_is_none(interface):
    assert interface.Platform.get_hidden_state_sampler() is None


def test_init_asks_platform_once(runner_cls):
    src = inspect.getsource(runner_cls.__init__)
    assert "current_platform.get_hidden_state_sampler()" in src
    assert "self._platform_sampler_output: SamplerOutput | None = None" in src
    # The upstream Sampler is still built, for the None path.
    assert "self.sampler = Sampler(" in src


def test_try_sample_runs_before_compute_logits(runner_cls):
    src = inspect.getsource(runner_cls.execute_model)
    try_sample = src.index("self._hidden_state_sampler.try_sample(")
    guard = src.index("if self._platform_sampler_output is None:", try_sample)
    logits = src.index("logits = self.model.compute_logits(sample_hidden_states)")
    assert try_sample < guard < logits
    # Only the common (last-PP-rank) case is hooked; the broadcast path is not.
    assert src.count("try_sample(") == 1


def test_sample_tokens_uses_then_clears_platform_output(runner_cls):
    src = inspect.getsource(runner_cls.sample_tokens)
    use = src.index("self._platform_sampler_output or self._sample(")
    clear = src.index("self._platform_sampler_output = None", use)
    bookkeep = src.index("self._bookkeeping_sync(")
    assert use < clear < bookkeep
