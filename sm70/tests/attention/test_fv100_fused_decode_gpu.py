# SPDX-License-Identifier: Apache-2.0
"""K14 (C4) GPU: the one-launch sink decode and its capture-time counters.

Port of the fused GPU cases of fork cf9492a
tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py :400-860:

* fused vs two launches vs the FP32 reference, eager and in the server's
  262K capture envelope (:411-451);
* CUDA-graph replay in the server envelope at changing lengths (:454-514);
* descending capture, largest batch first, with and without the eager
  prepare (:517-659);
* capture on a fresh stream after 0 or 1 warmups (:662-735);
* **the gibberish regression** `test_decode_sinks_descending_capture_regression`
  (:738-858): batches 8/4/2/1 captured on a poisoned shared graph pool with no
  eager call, batch 1 replayed first at 65/300/5000 tokens. Before fork
  7f12d21b8 the counters were created inside the first capture, so only the
  batch-8 graph zeroed them and every other graph decoded garbage above 64
  tokens (the `--max-num-seqs 20` bug).

Changes from the fork: `vllm_sm70.ops.fv100` instead of the pybind interface
(the fixture FAILS on SM70 when the fused op is missing, since the default
build has it); caches are reset with `fv100.clear_workspaces()`; the shared
decode workspace is not optional any more; an LBNHC (upstream v0.33.0 KV
layout) case for the TP2 shape is added.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _fv100_ref import FULL, SWA513, decode_reference, paged_kv, sinks_for  # noqa: E402

pytestmark = pytest.mark.gpu

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
WINDOWS = pytest.mark.parametrize("window", [SWA513, FULL], ids=["swa513", "full"])
# Query/KV heads per rank: TP1 32q/4kv and TP2 16q/2kv (2xV100), GQA 8:1.
HEADS = pytest.mark.parametrize("heads", [(32, 4), (16, 2)], ids=["tp1", "tp2"])
ATOL = 5e-4
RTOL = 3e-2
GROUPED_ATOL = 2e-3
GROUPED_RTOL = 1e-2
FUSED_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE"
CAPTURE_BATCHES = (8, 4, 2, 1)
CAPTURE_CAPACITY = 8192


@pytest.fixture
def fv100_fused(fv100_gpu, monkeypatch):
    """(device, fv100) with the fused op, fresh caches and the switch on."""
    device, fv100 = fv100_gpu
    if not fv100.has_op(fv100.FUSED_DECODE_OP):
        pytest.fail(
            "_sm70_C lacks fv100_decode_paged_sinks_fused: built with "
            "-DVLLM_SM70_FV100_FUSED_DECODE=OFF? (the default build has it)"
        )
    monkeypatch.setenv(FUSED_ENV, "1")
    monkeypatch.delenv("VLLM_SM70_FLINT_ATTN_FUSED_DECODE_ZERO_COUNTERS", raising=False)
    fv100.clear_workspaces()
    yield device, fv100
    fv100.clear_workspaces()


def _query(batch, num_heads, device, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((batch, num_heads, HEAD_DIM), generator=gen).to(
        device, torch.float16
    )


def _decode(fv100, q, k_cache, v_cache, block_table, seq_lens, window, **kw):
    return fv100.flash_attn_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=SCALE,
        window_size=window,
        **kw,
    )


def _ref(q, k_cache, v_cache, block_table, lens, window, sinks):
    return decode_reference(
        q, k_cache, v_cache, block_table, lens, scale=SCALE, window=window, sinks=sinks
    )


def _kv(seq_lens, num_kv_heads, device, seed, layout="separate"):
    k, v, table, _ = paged_kv(
        seq_lens, num_kv_heads, HEAD_DIM, 16, device, seed, layout=layout
    )
    return k, v, table


@WINDOWS
@HEADS
@pytest.mark.parametrize(
    "seq_lens",
    [[1], [63, 64, 65], [255, 256, 257, 513, 514], [3000, 70], [4096], [65536]],
    ids=["one", "min-chunk", "edges", "mixed", "4k", "64k"],
)
@pytest.mark.parametrize("capacity", [None, 262144], ids=["eager", "envelope"])
@torch.inference_mode()
def test_decode_sinks_fused_matches_two_launches(
    fv100_fused, monkeypatch, window, heads, seq_lens, capacity
):
    """fork :411-451. One launch per layer against partition kernel plus
    reducer; ``capacity`` is the server's 262K capture envelope (the fused
    grid fills the device and the last CTA reduces many partials)."""
    device, fv100 = fv100_fused
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table = _kv(
        seq_lens, num_kv_heads, device, seed=91 + len(seq_lens)
    )
    q = _query(len(seq_lens), num_heads, device, seed=92)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sinks = sinks_for(num_heads, device, seed=93)
    kw = {"max_seq_len_hint": max(seq_lens), "sinks": sinks}
    if capacity is not None:
        kw["workspace_seq_capacity_hint"] = capacity
        kw["partition_size_hint"] = 256
    monkeypatch.setenv(FUSED_ENV, "0")
    two = _decode(fv100, q, k_cache, v_cache, block_table, lens, window, **kw)
    assert fv100._fused_decode_counter_cache == {}
    monkeypatch.setenv(FUSED_ENV, "1")
    fused = _decode(fv100, q, k_cache, v_cache, block_table, lens, window, **kw)
    assert fv100._fused_decode_counter_cache  # the fused route really ran
    # Repeat: the arrival counters must be back at zero after each launch.
    again = _decode(fv100, q, k_cache, v_cache, block_table, lens, window, **kw)
    torch.testing.assert_close(fused, two, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)
    assert torch.equal(fused, again)
    expected = _ref(q, k_cache, v_cache, block_table, lens, window, sinks)
    torch.testing.assert_close(fused.float(), expected, rtol=RTOL, atol=ATOL)
    for entry in fv100._fused_decode_counter_cache.values():
        assert torch.count_nonzero(entry.buffer) == 0


@WINDOWS
@pytest.mark.parametrize("seq_lens", [[1, 600], [4096, 65, 513]], ids=["b2", "b3"])
@torch.inference_mode()
def test_decode_sinks_fused_lbnhc_tp2(fv100_fused, window, seq_lens):
    """New: TP2 16q/2kv on upstream's LBNHC cache (K and V interleaved per
    head, stride(2) = 2D), as the K06 backend passes it."""
    device, fv100 = fv100_fused
    k_cache, v_cache, block_table = _kv(seq_lens, 2, device, seed=61, layout="lbnhc")
    assert k_cache.stride(2) == 2 * HEAD_DIM
    q = _query(len(seq_lens), 16, device, seed=62)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, seed=63)
    assert fv100.fused_decode_eligible(q, k_cache, sinks)
    out = _decode(
        fv100,
        q,
        k_cache,
        v_cache,
        block_table,
        lens,
        window,
        max_seq_len_hint=max(seq_lens),
        sinks=sinks,
    )
    assert fv100._fused_decode_counter_cache
    expected = _ref(q, k_cache, v_cache, block_table, lens, window, sinks)
    torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_fused_cuda_graph_server_envelope(fv100_fused, window):
    """fork :454-514. The server's capture: hint 1, a 262K envelope; replays
    follow new lengths both ways, are bit-stable, keep the empty row zero."""
    device, fv100 = fv100_fused
    capacity = 262144
    k_cache, v_cache, block_table = _kv([40000, 40000], 2, device, seed=95)
    q = _query(2, 16, device, seed=96)
    lens = torch.ones(2, dtype=torch.int32, device=device)
    active = torch.ones(1, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, seed=97)
    out = torch.empty_like(q)

    def run():
        _decode(
            fv100,
            q,
            k_cache,
            v_cache,
            block_table,
            lens,
            window,
            out=out,
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=capacity,
            active_num_partitions=active,
            partition_size_hint=256,
            sinks=sinks,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        run()

    for step, new_lens in enumerate(
        ([4096, 513], [40000, 1], [600, 0], [32768, 4097], [65, 40000])
    ):
        q.copy_(_query(2, 16, device, seed=100 + step))
        lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        active.fill_(max(1, math.ceil(max(new_lens) / 256)))
        graph.replay()
        torch.accelerator.synchronize()
        first = out.clone()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, first)
        expected = _ref(q, k_cache, v_cache, block_table, lens, window, sinks)
        torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


def _descending_capture(fv100, device, window, stream, *, eager_first):
    """fork :534-590. Graphs for batches 8, 4, 2, 1 on one stream, largest
    first, rows being prefixes of shared persistent buffers (as the runner's
    are); the decode workspace grows during the first (largest) capture."""
    rows = CAPTURE_BATCHES[0]
    k_cache, v_cache, block_table = _kv([CAPTURE_CAPACITY] * rows, 2, device, seed=111)
    q = _query(rows, 16, device, seed=112)
    lens = torch.ones(rows, dtype=torch.int32, device=device)
    active = torch.ones(1, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, seed=113)
    out = torch.zeros_like(q)
    fused_calls = []
    counters_of = fv100._fused_decode_counters

    def recording_counters(*args, **kwargs):
        counters = counters_of(*args, **kwargs)
        fused_calls.append(counters is not None)
        return counters

    def run(batch):
        _decode(
            fv100,
            q[:batch],
            k_cache,
            v_cache,
            block_table[:batch],
            lens[:batch],
            window,
            out=out[:batch],
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=CAPTURE_CAPACITY,
            active_num_partitions=active,
            partition_size_hint=256,
            sinks=sinks,
        )

    stream.wait_stream(torch.cuda.current_stream(device))
    graphs = {}
    with torch.cuda.stream(stream):
        if eager_first:
            assert fv100.flash_attn_decode_paged_sinks_prepare(device)
        fv100._fused_decode_counters = recording_counters
        try:
            for batch in CAPTURE_BATCHES:
                graphs[batch] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[batch], stream=stream):
                    run(batch)
        finally:
            fv100._fused_decode_counters = counters_of
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.accelerator.synchronize()
    state = (q, k_cache, v_cache, block_table, lens, active, sinks, out)
    return graphs, state, fused_calls


def _two_launch(fv100, monkeypatch, state, window, batch):
    q, k_cache, v_cache, block_table, lens, _, sinks, _ = state
    monkeypatch.setenv(FUSED_ENV, "0")
    try:
        return _decode(
            fv100,
            q[:batch],
            k_cache,
            v_cache,
            block_table[:batch],
            lens[:batch],
            window,
            max_seq_len_hint=int(lens[:batch].max()),
            sinks=sinks,
        )
    finally:
        monkeypatch.setenv(FUSED_ENV, "1")


@WINDOWS
@pytest.mark.parametrize("eager_first", [False, True], ids=["capture-only", "eager"])
@torch.inference_mode()
def test_decode_sinks_fused_descending_capture(
    fv100_fused, monkeypatch, window, eager_first
):
    """fork :612-659. Without an eager call first no counters may be created
    under capture (every graph takes two launches); with the eager prepare,
    every graph takes the fused path. Either way the smallest graph decodes
    65, 300 and 5000 tokens as the two-launch kernels do, and the counters
    are back at zero after each replay."""
    device, fv100 = fv100_fused
    stream = torch.cuda.Stream(device)
    graphs, state, fused_calls = _descending_capture(
        fv100, device, window, stream, eager_first=eager_first
    )
    q, k_cache, v_cache, block_table, lens, active, sinks, out = state
    assert fused_calls == [eager_first] * len(CAPTURE_BATCHES)
    cache = fv100._fused_decode_counter_cache
    assert bool(cache) == eager_first
    for step, length in enumerate((65, 300, 5000)):
        q.copy_(_query(q.shape[0], 16, device, seed=120 + step))
        lens.fill_(0)
        lens[0] = length
        active.fill_(max(1, math.ceil(length / 256)))
        graphs[1].replay()
        torch.accelerator.synchronize()
        two = _two_launch(fv100, monkeypatch, state, window, 1)
        torch.testing.assert_close(out[:1], two, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)
        expected = _ref(
            q[:1], k_cache, v_cache, block_table[:1], lens[:1], window, sinks
        )
        torch.testing.assert_close(out[:1].float(), expected, rtol=RTOL, atol=ATOL)
        for entry in cache.values():
            assert torch.count_nonzero(entry.buffer) == 0
    # The largest graph, captured while the workspace grew, still decodes.
    lens.copy_(torch.tensor([65, 300, 5000, 1, 64, 513, 4096, 8192]))
    active.fill_(CAPTURE_CAPACITY // 256)
    graphs[8].replay()
    torch.accelerator.synchronize()
    two = _two_launch(fv100, monkeypatch, state, window, 8)
    torch.testing.assert_close(out, two, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)
    for entry in cache.values():
        assert torch.count_nonzero(entry.buffer) == 0


@WINDOWS
@pytest.mark.parametrize("warmups", [0, 1])
@torch.inference_mode()
def test_decode_sinks_fused_capture_on_fresh_stream(fv100_fused, window, warmups):
    """fork :662-735. vLLM's order: eager on the default stream, then capture
    on a fresh side stream after 0 or 1 eager warmups there. Either way the
    graph takes the one-launch path with the device's counters."""
    device, fv100 = fv100_fused
    k_cache, v_cache, block_table = _kv([5000], 2, device, seed=131)
    q = _query(1, 16, device, seed=132)
    lens = torch.tensor([5000], dtype=torch.int32, device=device)
    active = torch.ones(1, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, seed=133)
    out = torch.zeros_like(q)

    def run():
        _decode(
            fv100,
            q,
            k_cache,
            v_cache,
            block_table,
            lens,
            window,
            out=out,
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=8192,
            active_num_partitions=active,
            partition_size_hint=256,
            sinks=sinks,
        )

    run()  # eager, on the default stream
    capture = torch.cuda.Stream(device)
    capture.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture):
        for _ in range(warmups):
            run()
    recorded = []
    counters_of = fv100._fused_decode_counters

    def recording_counters(*args, **kwargs):
        counters = counters_of(*args, **kwargs)
        recorded.append(counters)
        return counters

    before = fv100.flash_attn_decode_paged_sinks_capture_counts()
    graph = torch.cuda.CUDAGraph()
    fv100._fused_decode_counters = recording_counters
    try:
        with torch.cuda.graph(graph, stream=capture):
            run()
    finally:
        fv100._fused_decode_counters = counters_of
    torch.cuda.current_stream(device).wait_stream(capture)
    torch.accelerator.synchronize()
    fused, two_launch = fv100.flash_attn_decode_paged_sinks_capture_counts()
    assert (fused - before[0], two_launch - before[1]) == (1, 0)
    (buffer,) = [entry.buffer for entry in fv100._fused_decode_counter_cache.values()]
    assert len(recorded) == 1 and recorded[0] is buffer
    assert fv100._fused_decode_warned == set()
    for length in (65, 300, 5000):
        lens.fill_(length)
        active.fill_(max(1, math.ceil(length / 256)))
        graph.replay()
        torch.accelerator.synchronize()
        expected = _ref(q, k_cache, v_cache, block_table, lens, window, sinks)
        torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)
        assert torch.count_nonzero(buffer) == 0


def _dirty_graph_pool(device, pool, stream):
    """fork :738-759. Leave freed memory of the shared graph pool full of
    0x7fffffff, so later captures on the pool reuse poisoned memory."""
    numel = 64 << 20  # 256 MiB of int32
    poison = torch.cuda.CUDAGraph()
    with torch.cuda.graph(poison, pool=pool, stream=stream):
        # One large-pool block, and small-pool (<1 MiB) blocks filling 2 MiB
        # segments: small workspaces (counters, statistics) come from those.
        blocks = [torch.empty(numel, dtype=torch.int32, device=device)]
        blocks += [
            torch.empty(1 << 17, dtype=torch.int32, device=device) for _ in range(64)
        ]
        for block in blocks:
            block.fill_(0x7FFFFFFF)
    poison.replay()
    torch.accelerator.synchronize()
    del blocks, block
    return poison


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_descending_capture_regression(fv100_fused, monkeypatch, window):
    """fork :762-858, the --max-num-seqs 20 gibberish regression. Black box:
    graphs captured largest first on a fresh stream and a shared, poisoned
    graph pool with no eager decode call; batch 1 replays first at 65, 300
    and 5000 tokens before batch 8 ever replays. Uses only the public
    decode entry point."""
    device, fv100 = fv100_fused
    rows, capacity = 8, 8192
    k_cache, v_cache, block_table = _kv([capacity] * rows, 2, device, 141)
    q = _query(rows, 16, device, seed=142)
    lens = torch.ones(rows, dtype=torch.int32, device=device)
    active = torch.ones(1, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, seed=143)
    out = torch.full_like(q, float("nan"))

    def run(batch):
        _decode(
            fv100,
            q[:batch],
            k_cache,
            v_cache,
            block_table[:batch],
            lens[:batch],
            window,
            out=out[:batch],
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=capacity,
            active_num_partitions=active,
            partition_size_hint=256,
            sinks=sinks,
        )

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.stream(stream):
        poison = _dirty_graph_pool(device, pool, stream)
        graphs = {}
        for batch in CAPTURE_BATCHES:
            graphs[batch] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[batch], pool=pool, stream=stream):
                run(batch)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.accelerator.synchronize()
    del poison

    def two_launch(batch):
        monkeypatch.setenv(FUSED_ENV, "0")
        try:
            return _decode(
                fv100,
                q[:batch],
                k_cache,
                v_cache,
                block_table[:batch],
                lens[:batch],
                window,
                max_seq_len_hint=int(lens[:batch].max()),
                sinks=sinks,
            )
        finally:
            monkeypatch.setenv(FUSED_ENV, "1")

    for step, length in enumerate((65, 300, 5000)):
        q.copy_(_query(rows, 16, device, seed=150 + step))
        lens.fill_(0)
        lens[0] = length
        active.fill_(max(1, math.ceil(length / 256)))
        out.fill_(float("nan"))
        graphs[1].replay()
        torch.accelerator.synchronize()
        assert torch.isfinite(out[:1]).all(), f"batch-1 graph left {length} unset"
        torch.testing.assert_close(
            out[:1], two_launch(1), rtol=GROUPED_RTOL, atol=GROUPED_ATOL
        )
        expected = _ref(
            q[:1], k_cache, v_cache, block_table[:1], lens[:1], window, sinks
        )
        torch.testing.assert_close(out[:1].float(), expected, rtol=RTOL, atol=ATOL)
    lens.copy_(torch.tensor([65, 300, 5000, 1, 64, 513, 4096, 8192]))
    active.fill_(capacity // 256)
    graphs[8].replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, two_launch(8), rtol=GROUPED_RTOL, atol=GROUPED_ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_descending_capture_after_impl_prepare(
    fv100_fused, monkeypatch, window
):
    """New (R4): the K14 production order. The impl prepared the counters at
    layer construction; then batches 20, 16, 8, 4, 2, 1 (--max-num-seqs 20)
    are captured largest first on a poisoned pool, all on the fused path,
    and every graph replays correctly at lengths above 64."""
    device, fv100 = fv100_fused
    assert fv100.flash_attn_decode_paged_sinks_prepare(device)
    rows, capacity = 20, 8192
    k_cache, v_cache, block_table = _kv([capacity] * rows, 2, device, 171)
    q = _query(rows, 16, device, seed=172)
    lens = torch.ones(rows, dtype=torch.int32, device=device)
    active = torch.ones(1, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, seed=173)
    out = torch.full_like(q, float("nan"))

    def run(batch):
        _decode(
            fv100,
            q[:batch],
            k_cache,
            v_cache,
            block_table[:batch],
            lens[:batch],
            window,
            out=out[:batch],
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=capacity,
            active_num_partitions=active,
            partition_size_hint=256,
            sinks=sinks,
        )

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    pool = torch.cuda.graph_pool_handle()
    batches = (20, 16, 8, 4, 2, 1)
    with torch.cuda.stream(stream):
        poison = _dirty_graph_pool(device, pool, stream)
        graphs = {}
        for batch in batches:
            graphs[batch] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[batch], pool=pool, stream=stream):
                run(batch)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.accelerator.synchronize()
    del poison
    assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (len(batches), 0)
    (buffer,) = [e.buffer for e in fv100._fused_decode_counter_cache.values()]
    for batch in reversed(batches):  # smallest first, as the gibberish needed
        for step, length in enumerate((65, 300, 5000)):
            q.copy_(_query(rows, 16, device, seed=180 + step))
            lens.fill_(0)
            lens[:batch] = torch.tensor(
                [length + 7 * i for i in range(batch)], dtype=torch.int32
            )
            active.fill_(max(1, math.ceil(int(lens.max()) / 256)))
            out.fill_(float("nan"))
            graphs[batch].replay()
            torch.accelerator.synchronize()
            assert torch.isfinite(out[:batch]).all(), (batch, length)
            expected = _ref(
                q[:batch],
                k_cache,
                v_cache,
                block_table[:batch],
                lens[:batch],
                window,
                sinks,
            )
            torch.testing.assert_close(
                out[:batch].float(), expected, rtol=RTOL, atol=ATOL
            )
            assert torch.count_nonzero(buffer) == 0
