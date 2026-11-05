# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the plugin tests (CPU only, no GPU, no CUDA).

test_no_vllm_*.py need only this package. The other files need vLLM's
Python source importable (an installed vllm, or the source tree on
PYTHONPATH); they are skipped otherwise. No compiled vLLM extension is
needed: `vllm.platforms.cuda` imports `vllm._C_stable_libtorch` at module
level (UP platforms/cuda.py:23), so `stub_stable_libtorch` puts an empty
module in its place when the real one is absent.
"""

import importlib
import importlib.util
import sys
import types

import pytest


class FakeNvml:
    """Minimal pynvml stand-in: one compute capability per device index."""

    def __init__(self, caps, uuids=None):
        self.caps = list(caps)
        self.uuids = list(uuids or [f"GPU-{i}" for i in range(len(self.caps))])
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return len(self.caps)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return self.uuids.index(uuid)

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return self.caps[handle]


@pytest.fixture
def fake_nvml(monkeypatch):
    """Returns install(caps) -> FakeNvml; clears the platform caches."""
    from vllm_sm70 import platform

    def install(caps, uuids=None):
        nvml = FakeNvml(caps, uuids)
        monkeypatch.setattr(platform, "_import_pynvml", lambda: nvml)
        platform.all_visible_gpus_are_sm70.cache_clear()
        platform.is_sm70_ready.cache_clear()
        return nvml

    yield install
    platform.all_visible_gpus_are_sm70.cache_clear()
    platform.is_sm70_ready.cache_clear()


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


requires_vllm = pytest.mark.skipif(
    not _have_vllm(), reason="vLLM Python source not importable"
)


@pytest.fixture
def stub_stable_libtorch(monkeypatch):
    """Allow `import vllm.platforms.cuda` without vLLM's compiled ops."""
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


@pytest.fixture
def fresh_plugin(monkeypatch):
    """Reset vllm_sm70's once-only state around a test."""
    import vllm_sm70
    from vllm_sm70 import _ext

    monkeypatch.setattr(vllm_sm70, "_registered", False)
    _ext.load.cache_clear()
    _ext.abi_version.cache_clear()
    _ext.so_sha256.cache_clear()
    yield
    _ext.load.cache_clear()
    _ext.abi_version.cache_clear()
    _ext.so_sha256.cache_clear()


@pytest.fixture
def no_ext(monkeypatch):
    """Make `import vllm_sm70._sm70_C` fail, as on a build without it."""
    from vllm_sm70 import _ext

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "vllm_sm70._sm70_C":
            raise ImportError("No module named 'vllm_sm70._sm70_C' (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_ext.importlib, "import_module", fake_import)
    _ext.load.cache_clear()
    _ext.abi_version.cache_clear()
    _ext.so_sha256.cache_clear()
    yield
    _ext.load.cache_clear()
    _ext.abi_version.cache_clear()
    _ext.so_sha256.cache_clear()
