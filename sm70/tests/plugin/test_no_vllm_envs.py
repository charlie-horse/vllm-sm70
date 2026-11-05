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
        "build",
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


def test_every_referenced_var_is_registered():
    """Every VLLM_SM70_* name used by the build or the package is registered.

    vLLM's validate_environ rejects unregistered VLLM_* vars that are set at
    runtime, so a var read only by setup.py or CMake must still be registered
    (as an inert build-time entry).
    """
    import pathlib
    import re

    from vllm_sm70 import envs

    root = pathlib.Path(__file__).resolve().parents[2]  # sm70/
    sources = [root / "setup.py", root / "CMakeLists.txt"]
    sources += sorted((root / "vllm_sm70").rglob("*.py"))
    sources += sorted((root / "csrc").rglob("*.cmake"))
    pattern = re.compile(r"VLLM_SM70_[A-Z0-9_]+")
    referenced = set()
    for path in sources:
        if path.is_file():
            referenced |= set(pattern.findall(path.read_text()))
    missing = sorted(referenced - set(envs.environment_variables()))
    assert not missing, f"unregistered VLLM_SM70_* vars: {missing}"
