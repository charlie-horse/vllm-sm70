# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K08 GDN tests.

GPU tests are marked `gpu` and need an SM70 device plus a built
vllm_sm70._sm70_C (with the K08 GDN ops) and vLLM's Python source.
Run them from the repo root with:

    pytest -m gpu sm70/tests/gdn
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gdn_helpers import gdn_schemas  # noqa: E402


@pytest.fixture(scope="module")
def gdn_stub():
    """vllm_sm70.gdn.kernels with its fakes registered, against stub schemas
    defined from the C++ sources when _sm70_C is not built."""
    torch = pytest.importorskip("torch")
    from vllm_sm70.gdn import kernels as _probe  # noqa: F401

    lib = None
    if not hasattr(torch.ops._sm70_C, "gdn_forward_vlk_varlen_out"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in gdn_schemas().items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.gdn.kernels as kernels_mod

    kernels_mod = importlib.reload(kernels_mod)
    yield kernels_mod
    if lib is not None:
        lib._destroy()


@pytest.fixture
def gdn_gpu():
    """vllm_sm70.gdn.kernels on a real SM70 GPU with _sm70_C loaded, else skip.

    On an SM70 GPU a missing extension or missing GDN ops fails the test, so
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
    from vllm_sm70.gdn import kernels

    for name in gdn_schemas():
        if not kernels.has_op(name):
            pytest.fail(f"_sm70_C is loaded but has no {name} (built without K08)")
    return kernels
