# SPDX-License-Identifier: Apache-2.0
"""Platform plugin detection through NVML (faked); no CUDA init, no vllm."""

import sys

import pytest

from vllm_sm70 import platform


def test_activate_none_without_gpu(monkeypatch):
    def no_nvml():
        raise ImportError("no NVML in this test")

    monkeypatch.setattr(platform, "_import_pynvml", no_nvml)
    platform.all_visible_gpus_are_sm70.cache_clear()
    try:
        assert platform.activate() is None
        assert platform.is_sm70_ready() is False
    finally:
        platform.all_visible_gpus_are_sm70.cache_clear()
        platform.is_sm70_ready.cache_clear()


def test_activate_none_on_this_cpu_host():
    """Real NVML probe on the CI host (expected: no GPU)."""
    platform.all_visible_gpus_are_sm70.cache_clear()
    try:
        if platform.all_visible_gpus_are_sm70():
            pytest.skip("this host has SM70 GPUs")
        assert platform.activate() is None
    finally:
        platform.all_visible_gpus_are_sm70.cache_clear()


@pytest.mark.parametrize(
    "caps,visible,expected",
    [
        ([(7, 0), (7, 0)], None, True),
        ([(7, 0)], None, True),
        ([], None, False),
        ([(7, 0), (8, 0)], None, False),
        ([(7, 5)], None, False),
        ([(7, 0), (8, 0)], "0", True),
        ([(8, 0), (7, 0)], "1", True),
        ([(7, 0), (8, 0)], "0,1", False),
        ([(7, 0), (8, 0)], "", False),
        ([(7, 0), (8, 0)], "0,7,1", True),  # CUDA stops at the bad index
        ([(8, 0), (7, 0)], "GPU-1", True),
    ],
)
def test_activate_capabilities(fake_nvml, monkeypatch, caps, visible, expected):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    nvml = fake_nvml(caps)
    want = platform.PLATFORM_QUALNAME if expected else None
    assert platform.activate() == want
    assert nvml.init_calls == nvml.shutdown_calls == 1
    assert "torch.cuda" not in sys.modules or not _cuda_initialized()


def _cuda_initialized() -> bool:
    import torch

    return torch.cuda.is_initialized()


def test_is_sm70_ready_needs_ext(fake_nvml, no_ext):
    fake_nvml([(7, 0), (7, 0)])
    assert platform.activate() == platform.PLATFORM_QUALNAME
    assert platform.is_sm70_ready() is False


def test_is_sm70_ready_cached(fake_nvml, no_ext):
    nvml = fake_nvml([(7, 0)])
    platform.is_sm70_ready()
    platform.is_sm70_ready()
    assert nvml.init_calls == 1
