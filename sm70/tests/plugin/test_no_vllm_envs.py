# SPDX-License-Identifier: Apache-2.0
"""env registry semantics; no vllm needed."""

import subprocess
import sys

import pytest

from vllm_sm70 import envs


def test_names_and_sections():
    names = envs.environment_variables()
    assert "VLLM_SM70_PLUGIN_DEBUG" in names
    assert all(n.startswith("VLLM_SM70_") for n in names)
    assert list(envs._SECTIONS) == [
        "core",
        "ops",
        "attention",
        "gdn",
        "fused_moe",
        "quant",
        "models",
        "passes",
    ]


@pytest.mark.parametrize(
    "value,enabled",
    [
        (None, True),
        ("", True),
        ("1", True),
        ("true", True),
        ("off", True),
        ("0", False),
    ],
)
def test_kill_switch_only_zero_acts(monkeypatch, value, enabled):
    name = "VLLM_SM70_TEST_SWITCH"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert envs.kill_switch(name)() is enabled


def test_debug_flag(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_PLUGIN_DEBUG", raising=False)
    assert envs.VLLM_SM70_PLUGIN_DEBUG is False
    monkeypatch.setenv("VLLM_SM70_PLUGIN_DEBUG", "1")
    assert envs.VLLM_SM70_PLUGIN_DEBUG is True


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        envs.VLLM_SM70_DOES_NOT_EXIST  # noqa: B018


def test_import_pulls_in_no_vllm():
    code = (
        "import sys, vllm_sm70, vllm_sm70.envs, vllm_sm70.platform; "
        "bad = [m for m in sys.modules if m == 'vllm' or m.startswith('vllm.')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
