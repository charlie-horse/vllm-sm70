# SPDX-License-Identifier: Apache-2.0
"""patches/0006: V2 speculator class hook and the inkling2_dspark draft
kept out of the DeepSeek DSpark fallback.

The patched-source checks skip on a vLLM without patches/0006; the plugin
side (Sm70CudaPlatform.get_speculator_cls) runs everywhere.
"""

import inspect
from types import SimpleNamespace

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def interface():
    from vllm.platforms import interface

    if not hasattr(interface.Platform, "get_speculator_cls"):
        pytest.skip("vLLM without patches/0006 (U6)")
    return interface


def test_default_is_none(interface):
    assert interface.Platform.get_speculator_cls(object()) is None


def test_init_speculator_asks_platform_first(interface):
    from vllm.v1.worker.gpu import spec_decode

    src = inspect.getsource(spec_decode.init_speculator)
    ask = src.index("current_platform.get_speculator_cls(vllm_config)")
    build = src.index("return speculator_cls(vllm_config, device)", ask)
    assert build < src.index('if speculative_config.method == "extract_hidden_states"')
    assert src.count("get_speculator_cls(") == 1


def test_dspark_fallback_skips_inkling2_dspark(interface):
    from vllm.config.speculative import SpeculativeConfig

    src = inspect.getsource(SpeculativeConfig.__post_init__)
    guard = src.index('!= "inkling2_dspark"')
    fallback = src.index('"DSparkV41DraftModel" if is_v41 else "DSparkDraftModel"')
    assert src.rindex('elif self.method == "dspark" and (', 0, guard) < guard < fallback


def test_platform_hook_delegates_to_the_draft_module(stub_stable_libtorch, monkeypatch):
    from vllm_sm70 import platform
    from vllm_sm70.models.inkling2_sm70 import draft

    seen = []
    monkeypatch.setattr(draft, "speculator_cls", lambda cfg: seen.append(cfg) or "cls")
    cfg = SimpleNamespace(speculative_config=None)
    assert platform.Sm70CudaPlatform.get_speculator_cls(cfg) == "cls"
    assert seen == [cfg]


def test_platform_hook_is_none_without_an_inkling_draft(stub_stable_libtorch):
    from vllm_sm70 import platform

    hook = platform.Sm70CudaPlatform.get_speculator_cls
    assert hook(SimpleNamespace(speculative_config=None)) is None
    qwen = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3")
        ),
    )
    assert hook(SimpleNamespace(speculative_config=qwen)) is None
