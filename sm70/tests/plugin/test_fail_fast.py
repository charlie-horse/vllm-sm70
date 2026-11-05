# SPDX-License-Identifier: Apache-2.0
"""registered_areas(), and fail-fast on SM70 when an area fails (Astro)."""

from types import SimpleNamespace

import pytest
from conftest import requires_vllm


def _break(monkeypatch, area_name):
    import importlib

    mod = importlib.import_module(f"vllm_sm70.{area_name}")

    def boom():
        raise RuntimeError("broken area")

    monkeypatch.setattr(mod, "register", boom)
    return mod


@requires_vllm
def test_registered_areas_reports_each_area(fresh_plugin, no_ext, monkeypatch):
    import vllm_sm70

    _break(monkeypatch, "attention")
    vllm_sm70.register()
    status = vllm_sm70.registered_areas()
    assert list(status) == list(vllm_sm70.AREAS)
    assert status["attention"] == "RuntimeError: broken area"
    assert all(s == "ok" for a, s in status.items() if a != "attention")
    assert vllm_sm70.failed_areas() == {"attention": "RuntimeError: broken area"}


@requires_vllm
def test_retry_recovers_area(fresh_plugin, no_ext, monkeypatch):
    import vllm_sm70

    mod = _break(monkeypatch, "gdn")
    vllm_sm70.register()
    assert "gdn" in vllm_sm70.failed_areas()
    monkeypatch.setattr(mod, "register", lambda: None)
    vllm_sm70.retry_failed_areas()
    assert vllm_sm70.failed_areas() == {}


@pytest.fixture
def platform_cls(stub_stable_libtorch, monkeypatch):
    from vllm.platforms.cuda import CudaPlatform

    from vllm_sm70 import platform

    monkeypatch.setattr(
        CudaPlatform, "check_and_update_config", classmethod(lambda cls, cfg: None)
    )
    return platform.Sm70CudaPlatform


def _cfg():
    from vllm.config import CompilationConfig

    return SimpleNamespace(compilation_config=CompilationConfig())


@requires_vllm
def test_sm70_platform_fails_fast(
    fresh_plugin, no_ext, platform_cls, monkeypatch
):
    import vllm_sm70

    monkeypatch.delenv("VLLM_SM70_PLUGIN_STRICT", raising=False)
    _break(monkeypatch, "quant")
    vllm_sm70.register()
    with pytest.raises(RuntimeError, match="quant: RuntimeError: broken area"):
        platform_cls.check_and_update_config(_cfg())


@requires_vllm
def test_strict_off_downgrades_to_warning(
    fresh_plugin, no_ext, platform_cls, monkeypatch, caplog
):
    import vllm_sm70

    monkeypatch.setenv("VLLM_SM70_PLUGIN_STRICT", "0")
    _break(monkeypatch, "quant")
    vllm_sm70.register()
    platform_cls.check_and_update_config(_cfg())  # no raise
    assert "strict off" in caplog.text


@requires_vllm
def test_all_ok_does_not_raise(fresh_plugin, no_ext, platform_cls, monkeypatch):
    import vllm_sm70

    monkeypatch.delenv("VLLM_SM70_PLUGIN_STRICT", raising=False)
    vllm_sm70.register()
    platform_cls.check_and_update_config(_cfg())


@requires_vllm
def test_shutdown_hook_is_safe_without_ext(stub_stable_libtorch):
    from vllm_sm70 import _cuda_platform

    _cuda_platform._tm_runtime_close()  # no _sm70_C, no CUDA: a no-op
