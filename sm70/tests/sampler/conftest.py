# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K16 top-k candidate sampler tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C with the K16 ops. Run them from the repo root with the
built package first on PYTHONPATH:

    pytest -m gpu sm70/tests/sampler

The parity tests import upstream vLLM's reference sampler
(vllm.v1.sample.ops.topk_topp_sampler), so vLLM must be importable too.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sampler_gpu():
    """(device, vllm_sm70.sample) on a real SM70 GPU, with the overflow
    counter prepared (as kernel_warmup does before capture), else skip.

    Like tm_gpu (tests/ops/conftest.py): with an SM70 GPU present, a missing
    or K16-less _sm70_C is a broken build or a wrong PYTHONPATH, so the test
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
    from vllm_sm70 import sample

    if not sample.ops_available():
        pytest.fail("_sm70_C is loaded but has no K16 sampler ops")
    torch.cuda.set_device(device)
    sample.prepare(device)
    return device, sample
