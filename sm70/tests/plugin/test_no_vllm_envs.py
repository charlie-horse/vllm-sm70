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
        "sampler",
        "passes",
        "warmup",
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
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


def test_every_referenced_var_is_registered():
    """Every VLLM_SM70_* env var the build, Python or C++ reads is registered.

    vLLM's validate_environ rejects unregistered VLLM_* vars that are set at
    runtime, so a var read only by setup.py or CMake must still be registered
    (as an inert build-time entry).
    """
    import pathlib
    import re

    from vllm_sm70 import envs

    root = pathlib.Path(__file__).resolve().parents[2]  # sm70/
    quoted = re.compile(r"[\"'](VLLM_SM70_[A-Z0-9_]+)[\"']")  # env names in strings
    cmake_env = re.compile(r"ENV\{(VLLM_SM70_[A-Z0-9_]+)\}")  # CMake $ENV{...}
    referenced = set()
    py_and_cpp = [root / "setup.py"]
    py_and_cpp += sorted((root / "vllm_sm70").rglob("*.py"))
    for ext in ("*.h", "*.cuh", "*.cc", "*.cpp", "*.cu"):
        py_and_cpp += sorted((root / "csrc").rglob(ext))
    for path in py_and_cpp:
        referenced |= set(quoted.findall(path.read_text(errors="replace")))
    for path in [root / "CMakeLists.txt", *sorted((root / "csrc").rglob("*.cmake"))]:
        if path.is_file():
            referenced |= set(cmake_env.findall(path.read_text()))
    missing = sorted(referenced - set(envs.environment_variables()))
    assert not missing, f"unregistered VLLM_SM70_* vars: {missing}"
