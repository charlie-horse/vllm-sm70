# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K05 Flash-V100 op tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C with the K05 ops. Run them from the repo root with the
built package first on PYTHONPATH (see test_fv100_ops_cpu.py for the CPU
side):

    pytest -m gpu sm70/tests/attention
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fv100_gpu():
    """(device, vllm_sm70.ops.fv100) on a real SM70 GPU, else skip.

    Like tm_gpu (tests/ops/conftest.py): with an SM70 GPU present, a missing
    or K05-less _sm70_C is a broken build or a wrong PYTHONPATH, so the test
    fails instead of skipping and the nightly run cannot pass on stubs.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an SM70 GPU")
    device = None
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (7, 0):
            device = torch.device(f"cuda:{index}")
            break
    if device is None:
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail(
            "SM70 GPU present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    from vllm_sm70.ops import fv100

    missing = [name for name in fv100.OPS if not fv100.has_op(name)]
    if missing:
        pytest.fail(f"_sm70_C is loaded but lacks the K05 Flash-V100 ops {missing}")
    torch.cuda.set_device(device)
    fv100.clear_workspaces()
    return device, fv100
