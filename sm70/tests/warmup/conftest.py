# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the TurboMind pre-capture warmup tests (vllm_sm70/warmup).

CPU tests use fake layers, fake configs and a threaded 2-rank group. GPU
tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C. Run them from the repo root with the built package first
on PYTHONPATH:

    pytest -m gpu sm70/tests/warmup
"""

from __future__ import annotations

import pytest


@pytest.fixture
def warmup_gpu():
    """An SM70 GPU with _sm70_C's TurboMind ops loaded; skip without the GPU,
    fail (not skip) when the GPU is there but the extension is not."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail(
            "SM70 GPU present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    from vllm_sm70.ops import tm

    for op in (
        "nvfp4_gemm_sm70_out",
        "sm70_gemm_export_cache",
        "sm70_gemm_import_cache",
    ):
        if not tm.has_op(op):
            pytest.fail(f"_sm70_C is loaded but has no {op} (built without K03)")
    return torch
