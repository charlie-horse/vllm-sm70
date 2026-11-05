# SPDX-License-Identifier: Apache-2.0
"""register(): idempotent, works without _sm70_C, merges env vars into vllm."""

import os

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


def test_register_idempotent_without_ext(fresh_plugin, no_ext, monkeypatch):
    import vllm_sm70
    from vllm_sm70 import _ext

    calls = []
    for area in vllm_sm70.AREAS:
        mod = __import__(f"vllm_sm70.{area}", fromlist=["register"])
        orig = mod.register
        monkeypatch.setattr(
            mod, "register", lambda a=area, o=orig: (calls.append(a), o())
        )

    vllm_sm70.register()
    vllm_sm70.register()
    assert calls == list(vllm_sm70.AREAS)  # once each, in order
    assert calls == [
        "ops",
        "attention",
        "gdn",
        "fused_moe",
        "quant",
        "models",
        "passes",
    ]
    assert _ext.load() is None
    assert _ext.fingerprint() == "none"


def test_register_survives_a_broken_area(fresh_plugin, no_ext, monkeypatch):
    import vllm_sm70
    from vllm_sm70 import attention, passes

    def boom():
        raise RuntimeError("broken area")

    seen = []
    monkeypatch.setattr(attention, "register", boom)
    monkeypatch.setattr(passes, "register", lambda: seen.append("passes"))
    vllm_sm70.register()
    assert seen == ["passes"]


def test_env_merge_and_validate_environ(fresh_plugin, no_ext, monkeypatch):
    import vllm.envs as vllm_envs
    from vllm.platforms.interface import Platform

    import vllm_sm70
    from vllm_sm70 import envs

    # Hermetic: CI and dev shells export build-time VLLM_SM70_* vars
    # (e.g. VLLM_SM70_SKIP_EXT=1). Clear them so the only unknown var is ours.
    for key in [k for k in os.environ if k.startswith("VLLM_SM70_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("VLLM_SM70_PLUGIN_DEBUG", "1")
    for name in envs.environment_variables():
        monkeypatch.delitem(vllm_envs.environment_variables, name, raising=False)

    with pytest.raises(ValueError, match="VLLM_SM70_PLUGIN_DEBUG"):
        Platform.validate_environ(hard_fail=True)

    vllm_sm70.register()
    for name in envs.environment_variables():
        assert name in vllm_envs.environment_variables
    Platform.validate_environ(hard_fail=True)  # no longer unknown
    assert vllm_envs.VLLM_SM70_PLUGIN_DEBUG is True
    assert envs.merge_into_vllm() == []  # idempotent


def test_fingerprint_enters_compile_factors(fresh_plugin, no_ext, monkeypatch):
    import vllm.envs as vllm_envs

    from vllm_sm70 import envs

    envs.merge_into_vllm()
    factors = vllm_envs.compile_factors()
    assert factors["VLLM_SM70_EXT_FINGERPRINT"] == "none"
    assert "VLLM_SM70_PLUGIN_DEBUG" in factors


def test_merge_never_overrides_upstream(monkeypatch):
    import vllm.envs as vllm_envs

    from vllm_sm70 import envs

    sentinel = lambda: "upstream"  # noqa: E731
    monkeypatch.setitem(
        vllm_envs.environment_variables, "VLLM_SM70_PLUGIN_DEBUG", sentinel
    )
    envs.merge_into_vllm()
    assert vllm_envs.environment_variables["VLLM_SM70_PLUGIN_DEBUG"] is sentinel


def test_env_lookup_after_envs_cache(fresh_plugin, no_ext, monkeypatch):
    import vllm.envs as vllm_envs

    from vllm_sm70 import envs

    for name in envs.environment_variables():
        monkeypatch.delitem(vllm_envs.environment_variables, name, raising=False)
    vllm_envs.enable_envs_cache()
    try:
        envs.merge_into_vllm()  # plugin loaded after the cache was enabled
        assert vllm_envs.VLLM_SM70_EXT_FINGERPRINT == "none"
    finally:
        vllm_envs.disable_envs_cache()


def test_vllm_loads_our_entry_points(fresh_plugin, no_ext, monkeypatch):
    import importlib.metadata as md

    try:
        md.distribution("vllm-sm70")
    except md.PackageNotFoundError:
        pytest.skip("vllm-sm70 is not installed")
    from vllm import plugins

    import vllm_sm70

    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    general = plugins.load_plugins_by_group(plugins.DEFAULT_PLUGINS_GROUP)
    assert general["sm70"] is vllm_sm70.register
    platforms = plugins.load_plugins_by_group(plugins.PLATFORM_PLUGINS_GROUP)
    assert platforms["sm70"]() is None  # no SM70 GPU on a CPU host
    monkeypatch.setattr(plugins, "plugins_loaded", False)
    plugins.load_general_plugins()
    assert vllm_sm70._registered
