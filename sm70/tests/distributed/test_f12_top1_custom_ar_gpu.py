# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F12: _sm70_C custom-AR greedy top-1 exchange against the all-gather.

Two V100s. Each rank builds its [rows, 2] FP32 (value, global id) pairs as
get_top_tokens does (ids increasing by rank); the reference is upstream's
reduction after an NCCL all-gather of the pairs (UP logits_processor.py:
241-247). ``sample_greedy.exchange_custom_ar`` must return the same int64
token on both ranks, for 1-128 rows, ties across ranks, NaN and -inf, with
the allocator in default and expandable-segments mode, and interleaved with
upstream's own custom all-reduce on the same buffers and flags (so the
staged barrier keeps the signal protocol in step). Before each exchange the
staging buffer holds stale pairs that would win with a rank-0 id, and one
case puts every row's winner on rank 1 at a local id that is also valid on
rank 0: a stale staging read or a missing shard offset returns that id.

    pytest -m gpu sm70/tests/distributed/test_f12_top1_custom_ar_gpu.py
"""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

ROWS = (1, 2, 3, 7, 16, 64, 127, 128)
SHARD = 75072  # Flint TP2 vocab shard


def _pairs(rank, rows, seed, mode):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # The same draw on both ranks, then a per-rank part: ties are common.
    base = torch.randint(-4, 4, (rows, 2), generator=gen, device="cuda").float()
    values = base[:, rank].clone()
    if mode == "random":
        values += torch.rand(rows, generator=gen, device="cuda") * (rank + 1)
    elif mode == "nan":
        values[torch.arange(rows, device="cuda") % 3 == rank] = float("nan")
    elif mode == "inf":
        values[torch.arange(rows, device="cuda") % 2 == 0] = -float("inf")
    ids = rank * SHARD + torch.randint(0, SHARD, (rows,), generator=gen, device="cuda")
    return torch.stack([values, ids.float()], dim=-1)


# Stale staging content: a huge value naming a rank-0 id (fp16 hidden states
# read back as FP32 pairs look like this).
POISON = (1.0e30, 11749.0)


def _poison_staging(ca, rows=128):
    """Leave POISON pairs in this rank's eager (staging) buffer: upstream's
    eager custom all-reduce copies its input there first."""
    poison = torch.tensor(POISON, device="cuda").repeat(rows, 1)
    assert ca.should_custom_ar(poison.view(torch.float16))
    ca.custom_all_reduce(poison.view(torch.float16))


def _rank1_winner_pairs(rank, rows):
    """Every row: rank 0 (1.0, 3426), rank 1 (2.0, SHARD + 11749)."""
    value, local = (1.0, 3426) if rank == 0 else (2.0, 11749)
    pair = torch.tensor([value, rank * SHARD + local], device="cuda")
    return pair.repeat(rows, 1).contiguous()


def _reference(pair, world):
    import torch.distributed as dist

    parts = [torch.empty_like(pair) for _ in range(2)]
    dist.all_gather(parts, pair, group=world)
    gathered = torch.cat(parts, dim=-1).view(pair.shape[0], 2, 2)
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
    return top.squeeze(-1).to(torch.int64)


def _check(rank: int, rendezvous: str, expandable: bool) -> None:
    import torch.distributed as dist
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    from vllm_sm70 import sample_greedy

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=120),
    )
    nccl = dist.new_group(backend="nccl")
    ca = CustomAllreduce(
        dist.group.WORLD,
        torch.device("cuda", rank),
        max_size=8 * 1024 * 1024,
        register_graph_buffers=not expandable,
    )
    assert not ca.disabled
    try:
        assert (
            sample_greedy.custom_ar_reject_reason(ca, 1, 2, torch.device("cuda", rank))
            is None
        )
        for rows in ROWS:
            pair = _rank1_winner_pairs(rank, rows)
            want = torch.full((rows,), SHARD + 11749, dtype=torch.int64, device="cuda")
            assert torch.equal(_reference(pair, nccl), want)
            _poison_staging(ca)
            got = sample_greedy.exchange_custom_ar(ca, pair)
            assert torch.equal(got, want), (rows, got)
        seed = 0
        for mode in ("random", "nan", "inf"):
            for rows in ROWS:
                seed += 1
                pair = _pairs(rank, rows, seed, mode)
                want = _reference(pair, nccl)
                _poison_staging(ca)
                got = sample_greedy.exchange_custom_ar(ca, pair)
                assert got.dtype == torch.int64 and got.shape == (rows,)
                assert torch.equal(got, want), (mode, rows, got, want)
                # Same token on both ranks.
                other = [torch.empty_like(got) for _ in range(2)]
                dist.all_gather(other, got, group=nccl)
                assert torch.equal(other[0], other[1])
                # Upstream's custom all-reduce on the same buffer and flags.
                x = torch.full(
                    (rows, 2048), float(rank + 1), device="cuda", dtype=torch.float16
                )
                if ca.should_custom_ar(x):
                    reduced = ca.custom_all_reduce(x)
                    assert torch.equal(reduced, torch.full_like(x, 3.0))
        # Many back-to-back exchanges, ranks entering at different times.
        for step in range(200):
            if rank == step % 2:
                time.sleep(0.001)
            pair = _pairs(rank, 1 + step % 128, 1000 + step, "random")
            want = _reference(pair, nccl)
            if step % 2:
                _poison_staging(ca)
            assert torch.equal(sample_greedy.exchange_custom_ar(ca, pair), want)
        torch.cuda.synchronize()
    finally:
        ca.close()
        dist.destroy_process_group()


def _sm70_pair() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return all(torch.cuda.get_device_capability(i) == (7, 0) for i in range(2))


@pytest.mark.parametrize("expandable", [False, True])
def test_f12_top1_custom_ar_equals_all_gather(monkeypatch, tmp_path, expandable):
    import torch.multiprocessing as mp

    if not _sm70_pair():
        pytest.skip("requires two peer-capable V100 GPUs")
    from vllm_sm70 import _ext, sample_greedy

    # Two SM70 GPUs: a missing kernel is a broken build or a wrong
    # PYTHONPATH, so fail instead of skipping.
    if not _ext.is_loaded():
        pytest.fail(
            "two SM70 GPUs present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    if not sample_greedy.custom_ar_kernel_available():
        pytest.fail(
            "_sm70_C lacks custom_ar_top1_argmax: built without "
            "VLLM_SM70_VLLM_CSRC_DIR (csrc/custom_ar skipped)"
        )
    monkeypatch.delenv("VLLM_SM70_TOP1_CUSTOM_AR", raising=False)
    # The test builds CustomAllreduce on a bare torch.distributed group, so
    # vLLM's P2P check (which needs the world group) must stay skipped,
    # whatever the caller's env says.
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    if expandable:
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    else:
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
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
            pytest.fail("top-1 exchange ranks did not finish within 600 s")
