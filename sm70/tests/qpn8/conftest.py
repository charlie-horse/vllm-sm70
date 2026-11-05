# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K09 QPN8 tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C with the K09 QPN8 ops (and K03 TurboMind FP8 for the
M>8 fallback). Run them from the repo root with:

    pytest -m gpu sm70/tests/qpn8
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _qpn8_helpers import qpn8_schemas  # noqa: E402


@pytest.fixture(scope="module")
def qpn8_stub():
    """vllm_sm70.ops.qpn8 with its fakes registered, against stub schemas
    defined from the C++ sources when _sm70_C is not built."""
    torch = pytest.importorskip("torch")
    from vllm_sm70.ops import qpn8 as _probe  # noqa: F401

    lib = None
    if not hasattr(torch.ops._sm70_C, "fp8_qpn8_gemm_sm70_out"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in qpn8_schemas().items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.ops.qpn8 as qpn8_mod

    qpn8_mod = importlib.reload(qpn8_mod)
    yield qpn8_mod
    if lib is not None:
        lib._destroy()


@pytest.fixture
def qpn8_gpu():
    """vllm_sm70.ops.qpn8 on a real SM70 GPU with _sm70_C loaded, else skip.

    On an SM70 GPU a missing extension or missing QPN8 ops fails the test,
    so a nightly run cannot pass on stubs.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail(
            "SM70 GPU present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    from vllm_sm70.ops import qpn8

    for name in qpn8_schemas():
        if not qpn8.has_op(name):
            pytest.fail(f"_sm70_C is loaded but has no {name} (built without K09)")
    from vllm_sm70.ops import tm

    if not tm.has_op("fp8_gemm_sm70_out"):
        pytest.fail("_sm70_C has no TurboMind FP8 ops (built without K03)")
    return qpn8
