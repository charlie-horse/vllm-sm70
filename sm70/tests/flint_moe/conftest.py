# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K12 Flint NVFP4 MoE decode tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C with the K03 TurboMind ops (to prepare the expert banks)
and the K12 decode ops. Run them from the repo root with the built package
first on PYTHONPATH:

    pytest -m gpu sm70/tests/flint_moe
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ops"))
from _schemas import sm70_schemas  # noqa: E402

AREA = "flint_moe"


@pytest.fixture(scope="module")
def flint_stub():
    """vllm_sm70.ops.flint_moe with its fakes registered, against stub schemas
    defined from the C++ sources when _sm70_C is not built."""
    torch = pytest.importorskip("torch")
    from vllm_sm70.ops import flint_moe as _probe  # noqa: F401

    lib = None
    if not hasattr(torch.ops._sm70_C, "nvfp4_moe_decode_w13_swiglu_sm70_out"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in sm70_schemas(area=AREA).items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.ops.flint_moe as mod

    mod = importlib.reload(mod)
    yield mod
    if lib is not None:
        lib._destroy()


@pytest.fixture
def flint_gpu():
    """(device, flint_moe, tm) on a real SM70 GPU, else skip.

    Like tm_gpu (tests/ops/conftest.py): with an SM70 GPU present, a missing
    or K12-less _sm70_C is a broken build or a wrong PYTHONPATH, so the test
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
    from vllm_sm70.ops import flint_moe, tm

    missing = [name for name in flint_moe.OPS if not flint_moe.has_op(name)]
    if missing:
        pytest.fail(f"_sm70_C is loaded but lacks the K12 decode ops {missing}")
    if not tm.has_op("nvfp4_sm70_prepare"):
        pytest.fail("_sm70_C is loaded but has no TurboMind ops (built without K03)")
    torch.cuda.set_device(device)
    return device, flint_moe, tm
