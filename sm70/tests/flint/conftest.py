# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the K15 Flint q/k-norm + RoPE + KV-write tests.

``interp`` runs the Triton kernels on the CPU through Triton's interpreter,
so the kernel numerics and the KV-cache addressing are checked without a GPU.
Triton must see TRITON_INTERPRET=1 when it is first imported, so the tests
using ``interp`` run in a child pytest started with it
(test_k15_qk_norm_rope_kv_cpu.py::test_interpreted_suite) and skip in the
parent. vllm.triton_utils hands out placeholders when no GPU driver is
active, so the fixture loads private copies of the two kernel modules with
the real ``triton`` bound in, and leaves vllm.triton_utils as it was.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

INTERPRET_ENV = "TRITON_INTERPRET"


def _load_copy(module_name: str, alias: str) -> ModuleType:
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None, module_name
    copy_spec = importlib.util.spec_from_file_location(alias, spec.origin)
    module = importlib.util.module_from_spec(copy_spec)
    sys.modules[alias] = module
    copy_spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def interp():
    """SimpleNamespace(qk=<interpreted qk_norm_rope_kv module>,
    reshape_and_cache=<interpreted upstream triton_reshape_and_cache_flash>)."""
    import os

    if os.environ.get(INTERPRET_ENV) != "1":
        pytest.skip(f"runs in the {INTERPRET_ENV}=1 child (test_interpreted_suite)")
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")

    import vllm.triton_utils as vt

    # The regular module registers the custom op; the copy then skips it.
    importlib.import_module("vllm_sm70.flint.qk_norm_rope_kv")
    saved = (vt.triton, vt.tl)
    vt.triton, vt.tl = triton, tl
    try:
        qk = _load_copy("vllm_sm70.flint.qk_norm_rope_kv", "_k15_interp_qk")
        writer = _load_copy(
            "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
            "_k15_interp_reshape",
        )
    finally:
        vt.triton, vt.tl = saved
    # Upstream's writer sizes its tile from the CUDA capability; answer as
    # a V100 (sm_70) would. Only the interpreted child process sees this.
    import torch

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
        yield SimpleNamespace(
            qk=qk, reshape_and_cache=writer.triton_reshape_and_cache_flash
        )
    sys.modules.pop("_k15_interp_qk", None)
    sys.modules.pop("_k15_interp_reshape", None)
