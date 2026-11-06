# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K18 DFlash2 FP16 M<=8 GEMM / compact top-k tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C with the K18 ops. Run them from the repo root with:

    pytest -m gpu sm70/tests/dflash2
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _dflash2_helpers import dflash2_schemas  # noqa: E402


@pytest.fixture
def dflash2_stub():
    """vllm_sm70.ops.dflash2 with its fakes registered, against stub schemas
    read from the C++ sources when _sm70_C is not built. The fakes double as
    Meta kernels, so the ops run on meta tensors."""
    torch = pytest.importorskip("torch")
    from vllm_sm70.ops import dflash2 as _probe  # noqa: F401

    lib = None
    if not hasattr(torch.ops._sm70_C, "sm70_dflash2_fp16_dispatch_out"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in dflash2_schemas().items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.ops.dflash2 as mod

    if lib is not None:
        # Re-run the has_op()-guarded register_fake calls against the stubs.
        # With the real ops loaded the fakes are already registered.
        mod = importlib.reload(mod)
    yield mod
    if lib is not None:
        lib._destroy()
        importlib.reload(mod)


@pytest.fixture
def dflash2_gpu():
    """vllm_sm70.ops.dflash2 on a real SM70 GPU with _sm70_C loaded, else skip.

    On an SM70 GPU a missing extension or missing K18 ops fails the test, so
    a nightly run cannot pass on stubs.
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
    from vllm_sm70.ops import dflash2

    missing = [name for name in dflash2.OPS if not dflash2.has_op(name)]
    if missing:
        pytest.fail(f"_sm70_C is loaded but lacks the K18 ops {missing}")
    return dflash2
