# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K03 op tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C. Run them with:

    pytest -m gpu sm70/tests/ops
"""

from __future__ import annotations

import pytest


@pytest.fixture
def tm_gpu():
    """vllm_sm70.ops.tm on a real SM70 GPU with _sm70_C loaded, else skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.skip("vllm_sm70._sm70_C is not built")
    from vllm_sm70.ops import tm

    if not tm.has_op("nvfp4_gemm_sm70_out"):
        pytest.skip("_sm70_C has no TurboMind ops (built without K03)")
    return tm
