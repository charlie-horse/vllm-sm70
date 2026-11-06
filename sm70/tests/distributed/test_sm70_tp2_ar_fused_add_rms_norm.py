# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K17: _sm70_C TP2 all-reduce + fused_add_rms_norm against the unfused ops.

Ported from fork cf9492a tests/distributed/test_sm70_tp2_ar_fused_add_rms_norm.py.
Two V100s. The reference is what the model runs without the fusion:
upstream's custom all-reduce (cross_device_reduce_1stage), the elementwise
scale or partial-sum add where present, then vLLM's C fused_add_rms_norm
kernel. Every output must be bitwise equal to it, on both ranks.

Run under the server's conditions as well: the allocator with
expandable_segments, operands allocated before graph capture (static
buffers the peer cannot open over IPC), graphs captured inside and outside
the communicator's capture context (FULL and PIECEWISE capture, and
capture-time warmup that runs eagerly), and ranks entering the collective at
different times. With expandable segments upstream's own capture path must
copy (register_graph_buffers=False, as CudaCommunicator does for a cuMem
pool); the fused op never registers graph buffers in either mode (R5).

Differences from the fork: the communicator is upstream's CustomAllreduce
and the kernel is _sm70_C's, called with its handle (no vLLM patch); an
op-level section drives torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm
through a group stand-in, fused (decode rows) and unfused (prefill rows).

    pytest -m gpu sm70/tests/distributed/test_sm70_tp2_ar_fused_add_rms_norm.py
"""

from __future__ import annotations

import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

H = 2048
EPS = 1e-6
ROWS = (1, 2, 3, 4, 5, 8, 16)


def _fused(ca, x, x_b, residual, weight, eps, scale):
    from vllm_sm70 import ar_rms_fusion

    out = torch.empty_like(x)
    res = torch.empty_like(residual)
    ar_rms_fusion._kernel()(
        ca._ptr,
        x,
        x_b,
        residual,
        weight,
        out,
        res,
        ca.buffer_ptrs[ca.rank],
        ca.max_size,
        eps,
        scale,
    )
    return out, res


def _reference(ca, x, x_b, residual, weight, scale):
    """Unfused ops; under capture the collective input is a pool buffer."""
    total = x.clone() if x_b is None else x + x_b
    reduced = ca.custom_all_reduce(total)
    if scale != 1.0:
        reduced = reduced * scale
    out = reduced.clone()
    res = residual.clone()
    torch.ops._C.fused_add_rms_norm(out, res, weight, EPS)
    return out, res


def _fill(rank, seed, x, x_b, residual, weight, magnitude):
    # Peer inputs differ; the residual stream and the weight are replicated.
    gen = torch.Generator(device="cuda").manual_seed(seed * 7 + rank)
    x.copy_(torch.randn(x.shape, generator=gen, device="cuda") * magnitude)
    if x_b is not None:
        x_b.copy_(torch.randn(x.shape, generator=gen, device="cuda") * magnitude)
    gen = torch.Generator(device="cuda").manual_seed(seed * 7 + 5)
    residual.copy_(torch.randn(residual.shape, generator=gen, device="cuda") * 8)
    weight.copy_(1 + 0.1 * torch.randn(weight.shape, generator=gen, device="cuda"))
    if seed % 4 == 1:
        # Exact cancellation across ranks and against the residual.
        x.copy_(-x if rank else x)
        residual[0].zero_()


def _assert_same_on_both_ranks(*tensors):
    import torch.distributed as dist

    for t in tensors:
        gathered = [torch.empty_like(t, device="cpu") for _ in range(2)]
        dist.all_gather(gathered, t.cpu())
        assert torch.equal(gathered[0], gathered[1])


def _assert_equal(actual, expected, context):
    for a, e in zip(actual, expected):
        assert torch.equal(a, e), context


def _op_level(rank, ca):
    """torch.ops.vllm_sm70 op: fused for decode rows, unfused for prefill."""
    import vllm.ir.ops
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers import layernorm  # noqa: F401
    from vllm_sm70 import ar_rms_fusion

    ar_rms_fusion.register_op()
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=ca),
        _all_reduce_out_place=lambda t: ca.custom_all_reduce(t),
    )
    parallel_state._groups["k17:gpu"] = lambda: group
    op = torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm
    weight = torch.empty(H, device="cuda", dtype=torch.float16)
    for rows, fused in ((2, True), (16, True), (17, False), (64, False)):
        x = torch.empty(rows, H, device="cuda", dtype=torch.float16)
        residual = torch.empty_like(x)
        _fill(rank, 3, x, None, residual, weight, 1.0)
        assert ar_rms_fusion.can_fuse(ca, x, None, residual, weight) is fused
        out, res = op(x, None, residual, weight, EPS, 1.0, "k17:gpu")
        if fused:
            expected = _reference(ca, x, None, residual, weight, 1.0)
        else:
            expected = vllm.ir.ops.fused_add_rms_norm(
                ca.custom_all_reduce(x), residual, weight, EPS
            )
        torch.accelerator.synchronize()
        _assert_equal((out, res), expected, ("op", rows))
        _assert_same_on_both_ranks(out, res)
    del parallel_state._groups["k17:gpu"]


def _check(rank: int, rendezvous: str, expandable: bool) -> None:
    import torch.distributed as dist
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm_sm70 import ar_rms_fusion

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    ca = CustomAllreduce(
        dist.group.WORLD,
        torch.device("cuda", rank),
        max_size=8 * 1024 * 1024,
        register_graph_buffers=not expandable,
    )
    assert not ca.disabled
    try:
        weight = torch.empty(H, device="cuda", dtype=torch.float16)

        # Admission: decode-sized FP16 [1..16, 2048] only.
        def admitted(rows, hidden=H, dtype=torch.float16):
            t = torch.zeros(rows, hidden, device="cuda", dtype=dtype)
            w = torch.zeros(hidden, device="cuda", dtype=dtype)
            return ar_rms_fusion.can_fuse(ca, t, None, t, w)

        assert admitted(1) and admitted(16)
        assert not admitted(17) and not admitted(2, hidden=4096)
        assert not admitted(2, dtype=torch.bfloat16)

        # Eager, and eagerly inside the capture context (graph warmup).
        for rows in ROWS:
            for sum2 in (False, True):
                for scale in (1.0, 0.125):
                    for seed, magnitude in enumerate((1.0, 64.0, 1.0, 2048.0)):
                        x = torch.empty(rows, H, device="cuda", dtype=torch.float16)
                        x_b = torch.empty_like(x) if sum2 else None
                        residual = torch.empty_like(x)
                        _fill(rank, seed, x, x_b, residual, weight, magnitude)
                        assert ar_rms_fusion.can_fuse(ca, x, x_b, residual, weight)
                        context = (rows, sum2, scale, seed)
                        expected = _reference(ca, x, x_b, residual, weight, scale)
                        actual = _fused(ca, x, x_b, residual, weight, EPS, scale)
                        with ca.capture():
                            warmup = _fused(ca, x, x_b, residual, weight, EPS, scale)
                        torch.accelerator.synchronize()
                        _assert_equal(actual, expected, context)
                        _assert_equal(warmup, expected, context)
                        _assert_same_on_both_ranks(*actual)

        # CUDA graphs over operands allocated before capture (static
        # buffers; with expandable segments they are not IPC-exportable).
        for rows in (1, 2, 3, 5):
            x = torch.zeros(rows, H, device="cuda", dtype=torch.float16)
            x_b = torch.zeros_like(x)
            residual = torch.zeros_like(x)
            for sum2 in (False, True):
                for scale in (1.0, 0.125):
                    b = x_b if sum2 else None
                    args = (ca, x, b, residual, weight, EPS, scale)
                    _fused(*args)
                    _reference(ca, x, b, residual, weight, scale)
                    torch.accelerator.synchronize()
                    dist.barrier()
                    # FULL-style capture: interleaved with ordinary
                    # all-reduces, then the communicator registers the
                    # captured collective buffers (none from the fused op).
                    graph = torch.cuda.CUDAGraph()
                    with ca.capture(), torch.cuda.graph(graph):
                        for _ in range(12):
                            out, res = _fused(*args)
                            ref = _reference(ca, x, b, residual, weight, scale)
                    # A graph captured without the communicator's context.
                    bare = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(bare):
                        for _ in range(12):
                            bare_out = _fused(*args)
                    for cycle in range(6):
                        context = (rows, sum2, scale, cycle)
                        _fill(rank, 100 + cycle, x, b, residual, weight, 4.0)
                        dist.barrier()
                        if rank == cycle % 2:
                            torch.cuda._sleep(20000)
                        for _ in range(3):
                            graph.replay()
                        torch.accelerator.synchronize()
                        _assert_equal((out, res), ref, context)
                        _assert_same_on_both_ranks(out, res)
                        dist.barrier()
                        if rank != cycle % 2:
                            torch.cuda._sleep(20000)
                        for _ in range(3):
                            bare.replay()
                        torch.accelerator.synchronize()
                        _assert_equal(bare_out, ref, context)
                        eager = _fused(*args)
                        torch.accelerator.synchronize()
                        _assert_equal(eager, ref, context)
                    del graph, bare

        _op_level(rank, ca)
    finally:
        ca.close()
        dist.destroy_process_group()


def _sm70_pair() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return all(torch.cuda.get_device_capability(i) == (7, 0) for i in range(2))


@pytest.mark.parametrize("expandable", [False, True])
def test_sm70_tp2_all_reduce_fused_add_rms_norm(monkeypatch, tmp_path, expandable):
    import torch.multiprocessing as mp

    if not _sm70_pair():
        pytest.skip("requires two peer-capable V100 GPUs")
    from vllm_sm70 import _ext, ar_rms_fusion

    # Two SM70 GPUs: a missing kernel is a broken build or a wrong
    # PYTHONPATH, so fail instead of skipping.
    if not _ext.is_loaded():
        pytest.fail(
            "two SM70 GPUs present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    if not ar_rms_fusion.kernel_available():
        pytest.fail(
            "_sm70_C lacks tp2_all_reduce_fused_add_rms_norm: built without "
            "VLLM_SM70_VLLM_CSRC_DIR (csrc/custom_ar skipped)"
        )
    import vllm._custom_ops  # noqa: F401  (loads vLLM's _C)

    if not hasattr(torch.ops._C, "fused_add_rms_norm"):
        pytest.fail("vLLM's _C (the reference fused_add_rms_norm) is not loaded")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    if expandable:
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    else:
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    # spawn (never fork after CUDA init) and a bounded wait, so a stuck
    # rank fails the test instead of hanging the run.
    ctx = mp.spawn(
        _check,
        args=((tmp_path / "rendezvous").as_uri(), expandable),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 600
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.kill()
            pytest.fail("TP2 all-reduce ranks did not finish within 600 s")
