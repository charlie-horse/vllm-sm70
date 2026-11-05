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

    # On a real SM70 GPU a missing extension is a broken build or a wrong
    # PYTHONPATH (e.g. the source tree shadowing the built package), not a
    # reason to skip: fail so the nightly run cannot pass on stubs.
    if not _ext.is_loaded():
        pytest.fail(
            "SM70 GPU present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    from vllm_sm70.ops import tm

    if not tm.has_op("nvfp4_gemm_sm70_out"):
        pytest.fail("_sm70_C is loaded but has no TurboMind ops (built without K03)")
    return tm
