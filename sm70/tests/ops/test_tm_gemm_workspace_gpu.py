# SPDX-License-Identifier: Apache-2.0
"""Bounded TurboMind scratch, cache export/import and tm_runtime_close.

Port of fork cf9492a tests/kernels/quantization/test_sm70_gemm_workspace.py
(419c1b414). The fork test ran GGUF formats under two engine-owned native
runtimes, each with its own KernelConfig.sm70_turbomind_workspace_bytes. Here
the runtime is per process, so the two budgets run one after the other,
separated by `_sm70_C::tm_runtime_close` (new in K03). NVFP4 and FP8 replace
GGUF.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")  # before _ref, which imports torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _ref import (  # noqa: E402
    fp8_block_dequant,
    fp8_block_weight,
    max_abs_rel,
    nvfp4_dequant,
    nvfp4_weight,
)

pytestmark = pytest.mark.gpu

MiB = 1024**2
# Barriers + tensormaps + flags per stream workspace (fork test, same layout).
FIXED_BYTES = 2 * MiB + 4


def _nvfp4(tm, n, k):
    codes, scales = nvfp4_weight(k, n, seed=n)
    w, s, meta = tm.nvfp4_sm70_prepare(codes, scales)
    ref_w = nvfp4_dequant(codes, scales)
    k_ld, q_ld = meta.tolist()

    def run(out, x):
        tm.nvfp4_gemm_sm70_out(out, x, w, s, 16, k_ld, q_ld)

    return run, lambda x: x.float() @ ref_w


def _fp8(tm, n, k):
    q, scales = fp8_block_weight(n, k, seed=n)
    w, s, meta = tm.fp8_sm70_prepare(q, scales, 128, False)
    ref_w = fp8_block_dequant(q, scales).T
    k_ld, q_ld = meta.tolist()

    def run(out, x):
        tm.fp8_gemm_sm70_out(out, x, w, s, 128, k_ld, q_ld, False)

    return run, lambda x: x.float() @ ref_w


@pytest.mark.parametrize("fmt,n,k", [("nvfp4", 4352, 5120), ("fp8", 5120, 8704)])
def test_bounded_workspace_cache_and_close(tm_gpu, tmp_path, fmt, n, k):
    tm = tm_gpu
    tm.tm_runtime_close()  # start from a clean runtime
    budgets = (32 * MiB, 8 * MiB)
    cache = str(tmp_path / f"{fmt}.cache")
    results = {}
    for budget in budgets:
        tm.tm_runtime_close()
        assert tm.sm70_gemm_workspace_storage()[0] == 0
        tm.sm70_gemm_configure_workspace(budget)
        # Prepare after close(): close() frees TurboMind's own state (scratch,
        # Gemm objects, tuning, fp16 weight cache), not caller-owned weights.
        run, reference = (_nvfp4 if fmt == "nvfp4" else _fp8)(tm, n, k)
        x0 = torch.randn(1, k, device="cuda", dtype=torch.float16)
        if budget == budgets[1]:
            assert tm.sm70_gemm_import_cache(x0, cache) >= 0
        for m in (1, 5, 20, 512):
            torch.manual_seed(1167 + m)
            x = (torch.randn(m, k, device="cuda") * 0.125).half()
            original = x.clone()
            out = torch.empty(m, n, device="cuda", dtype=torch.float16)
            run(out, x)
            assert max_abs_rel(out, reference(x)) < 1e-2
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(out, x)
            x.copy_(x * 0.5)
            graph.replay()
            captured = out.clone()
            run(out, x)
            assert torch.equal(captured, out)
            x.copy_(original)
            run(out, x)
            results[(budget, m)] = out.clone()
            del graph
        if budget == budgets[0]:
            assert tm.sm70_gemm_export_cache(x0, cache) >= 0
        count, actual_budget, total = tm.sm70_gemm_workspace_storage()
        assert count >= 1 and actual_budget == budget
        assert total == count * (budget + FIXED_BYTES)
        with pytest.raises(RuntimeError, match="cannot change"):
            tm.sm70_gemm_configure_workspace(4 * MiB)
    for m in (1, 5, 20, 512):
        a, b = results[(budgets[0], m)], results[(budgets[1], m)]
        assert (a.float() - b.float()).norm() <= a.float().norm() * 0.003 + 1e-6
    tm.tm_runtime_close()


def test_runtime_close_is_idempotent_and_ops_recover(tm_gpu):
    tm = tm_gpu
    run, reference = _nvfp4(tm, 1024, 2048)
    x = (torch.randn(8, 2048, device="cuda") * 0.1).half()
    out = torch.empty(8, 1024, device="cuda", dtype=torch.float16)
    run(out, x)
    assert tm.sm70_gemm_workspace_storage()[0] >= 1
    tm.tm_runtime_close()
    tm.tm_runtime_close()
    count, _, total = tm.sm70_gemm_workspace_storage()
    assert count == 0 and total == 0
    run(out, x)  # recreates the workspace and the Gemm lazily
    assert max_abs_rel(out, reference(x)) < 1e-2


# close() raises before anything is captured, so torch warns the graph is empty.
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_runtime_close_refuses_capture(tm_gpu):
    tm = tm_gpu
    run, _ = _nvfp4(tm, 1024, 2048)
    x = (torch.randn(8, 2048, device="cuda") * 0.1).half()
    out = torch.empty(8, 1024, device="cuda", dtype=torch.float16)
    run(out, x)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError):
        with torch.cuda.graph(graph):
            tm.tm_runtime_close()
    del graph
    torch.cuda.synchronize()
