# SPDX-License-Identifier: Apache-2.0
"""K05 GPU: learned sinks in the Flash-V100 paged decoder (Flint shapes).

Port of the non-fused GPU cases of fork cf9492a
tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py (:188-420 and
the dominant-sink case :862-882). The fused one-launch cases (:420-860,
incl. the gibberish regression test_decode_sinks_descending_capture_regression)
and the CPU dispatch cases (:942-1591) belong to K14 (C4) and K06.

D128, GQA 8:1 (TP1 32q/4kv, TP2 16q/2kv), sliding window 513 and full
attention, fp16 paged KV, P256/P512/P1024, batches 1-8, verifier rows and
CUDA-graph replay. With sinks, these shapes take the one-launch fused kernel
since K14 (VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0: the grouped two-launch
kernel); the grouped-vs-per-head case runs both routes. The fused-only cases
are in test_fv100_fused_decode_gpu.py.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _fv100_ref import (  # noqa: E402
    FULL,
    SWA513,
    decode_reference,
    paged_kv,
    sinks_for,
)

pytestmark = pytest.mark.gpu

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
WINDOWS = pytest.mark.parametrize("window", [SWA513, FULL], ids=["swa513", "full"])
HEADS = pytest.mark.parametrize("heads", [(32, 4), (16, 2)], ids=["tp1", "tp2"])
ATOL = 5e-4
RTOL = 3e-2
# The grouped kernel against the per-head kernel: both round partials to
# fp16, so they agree to a couple of fp16 ulps.
GROUPED_ATOL = 2e-3
GROUPED_RTOL = 1e-2


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


def _check_case(fv100_gpu, seq_lens, heads, window, *, block_size=16, seed=0, **kw):
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table, _ = paged_kv(
        seq_lens, num_kv_heads, HEAD_DIM, block_size, device, seed
    )
    q = _query(len(seq_lens), num_heads, device, seed + 1)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sinks = sinks_for(num_heads, device, seed + 2)
    kw.setdefault("max_seq_len_hint", max(seq_lens))

    out = _decode(fv100, q, k_cache, v_cache, block_table, lens, window, **kw)
    expected = _ref(q, k_cache, v_cache, block_table, lens, window, None)
    torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)

    out_sinks = _decode(
        fv100, q, k_cache, v_cache, block_table, lens, window, sinks=sinks, **kw
    )
    expected_sinks = _ref(q, k_cache, v_cache, block_table, lens, window, sinks)
    torch.testing.assert_close(out_sinks.float(), expected_sinks, rtol=RTOL, atol=ATOL)
    # The dominant sinks must visibly shrink the output, or the case is vacuous.
    effect = (expected_sinks - expected).abs().max()
    assert effect > max(4 * ATOL, 0.05 * expected.abs().max())


@WINDOWS
@HEADS
@pytest.mark.parametrize("seq_len", [1, 100, 256, 257, 513, 514, 600, 8192])
@torch.inference_mode()
def test_decode_sinks_single_sequence(fv100_gpu, window, heads, seq_len):
    """One partition (<=256) through many partitions at the default P256."""
    _check_case(fv100_gpu, [seq_len], heads, window, seed=seq_len)


@WINDOWS
@HEADS
@pytest.mark.parametrize("partition_size", [256, 512, 1024])
@torch.inference_mode()
def test_decode_sinks_every_partition_instantiation(
    fv100_gpu, window, heads, partition_size
):
    _check_case(
        fv100_gpu,
        [3000, 513, 1025],
        heads,
        window,
        partition_size_hint=partition_size,
        seed=partition_size,
    )


@WINDOWS
@HEADS
@pytest.mark.parametrize(
    "seq_lens",
    [
        [100],
        [600, 100],
        [1, 255, 256, 257],
        [100, 513, 600, 1024, 2049, 4096, 8192, 777],
    ],
    ids=["b1", "b2", "b4", "b8"],
)
@torch.inference_mode()
def test_decode_sinks_batches(fv100_gpu, window, heads, seq_lens):
    _check_case(fv100_gpu, seq_lens, heads, window, seed=len(seq_lens))


@WINDOWS
@pytest.mark.parametrize("block_size", [32, 64])
@torch.inference_mode()
def test_decode_sinks_other_page_sizes(fv100_gpu, window, block_size):
    _check_case(fv100_gpu, [700, 2500], (16, 2), window, block_size=block_size)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_64k_context(fv100_gpu, window):
    """The >=32K default plan switches to P1024 split-KV partitions."""
    _check_case(fv100_gpu, [65536], (16, 2), window, seed=7)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_static_graph_envelope(fv100_gpu, window):
    """A capacity hint launches idle partitions, as CUDA-graph decode does."""
    _check_case(
        fv100_gpu,
        [600, 70],
        (32, 4),
        window,
        workspace_seq_capacity_hint=16384,
        seed=11,
    )


@WINDOWS
@HEADS
@torch.inference_mode()
def test_decode_sinks_expanded_verifier_rows(fv100_gpu, window, heads):
    """Multi-token verification runs one decode row per query token."""
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    context, num_tokens = 1500, 4
    k_cache, v_cache, block_table, _ = paged_kv(
        [context], num_kv_heads, HEAD_DIM, 16, device, seed=21
    )
    rows = block_table.expand(num_tokens, -1).contiguous()
    lens = torch.arange(
        context - num_tokens + 1, context + 1, dtype=torch.int32, device=device
    )
    q = _query(num_tokens, num_heads, device, seed=22)
    sinks = sinks_for(num_heads, device, seed=23)
    out = _decode(
        fv100,
        q,
        k_cache,
        v_cache,
        rows,
        lens,
        window,
        max_seq_len_hint=context,
        sinks=sinks,
    )
    expected = _ref(q, k_cache, v_cache, rows, lens, window, sinks)
    torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_none_is_bit_identical(fv100_gpu, window):
    """`sinks=None` is the sink-free kernel; -inf sinks add an exact zero.

    With sinks, GQA 8:1 D128 layers take the grouped partition kernel, whose
    fp32 sums run in another order than the per-head kernel's, so -inf sinks
    match the sink-free output to rounding rather than bit for bit.
    """
    device, fv100 = fv100_gpu
    seq_lens = [100, 600, 8192]
    k_cache, v_cache, block_table, _ = paged_kv(seq_lens, 4, HEAD_DIM, 16, device, 31)
    q = _query(len(seq_lens), 32, device, seed=32)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    kw = {"max_seq_len_hint": max(seq_lens)}
    plain = _decode(fv100, q, k_cache, v_cache, block_table, lens, window, **kw)
    none = _decode(
        fv100, q, k_cache, v_cache, block_table, lens, window, sinks=None, **kw
    )
    neg_inf = torch.full((32,), -math.inf, dtype=torch.float32, device=device)
    masked = _decode(
        fv100, q, k_cache, v_cache, block_table, lens, window, sinks=neg_inf, **kw
    )
    assert torch.equal(none, plain)
    torch.testing.assert_close(masked, plain, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)


@WINDOWS
@HEADS
@pytest.mark.parametrize("partition_size", [256, 512, 1024])
@pytest.mark.parametrize(
    "seq_lens",
    [[1], [255, 256, 257], [513, 3000, 70], [8192]],
    ids=["one", "edges", "mixed", "long"],
)
@pytest.mark.parametrize("fused", ["0", "1"], ids=["two-launch", "fused"])
@torch.inference_mode()
def test_decode_sinks_grouped_kernel_matches_per_head_kernel(
    fv100_gpu, monkeypatch, window, heads, partition_size, seq_lens, fused
):
    """A sink of -1e4 adds an exact zero, so grouped == per-head sink-free."""
    monkeypatch.setenv("VLLM_SM70_FLINT_ATTN_FUSED_DECODE", fused)
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table, _ = paged_kv(
        seq_lens, num_kv_heads, HEAD_DIM, 16, device, seed=81 + partition_size
    )
    q = _query(len(seq_lens), num_heads, device, seed=82)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    kw = {"max_seq_len_hint": max(seq_lens), "partition_size_hint": partition_size}
    per_head = _decode(fv100, q, k_cache, v_cache, block_table, lens, window, **kw)
    silent = torch.full((num_heads,), -1.0e4, dtype=torch.float32, device=device)
    grouped = _decode(
        fv100, q, k_cache, v_cache, block_table, lens, window, sinks=silent, **kw
    )
    torch.testing.assert_close(grouped, per_head, rtol=GROUPED_RTOL, atol=GROUPED_ATOL)


@WINDOWS
@torch.inference_mode()
def test_decode_sinks_cuda_graph_replay(fv100_gpu, window):
    """Captured once; replays follow new queries, lengths and sink values."""
    device, fv100 = fv100_gpu
    capacity = 4096
    k_cache, v_cache, block_table, _ = paged_kv(
        [capacity, capacity], 4, HEAD_DIM, 16, device, seed=41
    )
    q = _query(2, 32, device, seed=42)
    lens = torch.tensor([600, 50], dtype=torch.int32, device=device)
    sinks = sinks_for(32, device, seed=43)
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
            max_seq_len_hint=capacity,
            workspace_seq_capacity_hint=capacity,
            sinks=sinks,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for step, new_lens in enumerate(([600, 50], [3000, 1], [4096, 2049])):
        q.copy_(_query(2, 32, device, seed=50 + step))
        sinks.copy_(sinks_for(32, device, seed=60 + step))
        lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        expected = _ref(q, k_cache, v_cache, block_table, lens, window, sinks)
        torch.testing.assert_close(out.float(), expected, rtol=RTOL, atol=ATOL)


@torch.inference_mode()
def test_decode_sinks_dominant_sink_stays_finite(fv100_gpu):
    """A sink far above every score must not overflow the denominator."""
    device, fv100 = fv100_gpu
    k_cache, v_cache, block_table, _ = paged_kv([2000], 2, HEAD_DIM, 16, device, 71)
    q = _query(1, 16, device, seed=72)
    lens = torch.tensor([2000], dtype=torch.int32, device=device)
    sinks = torch.full((16,), 120.0, dtype=torch.float32, device=device)
    out = _decode(
        fv100,
        q,
        k_cache,
        v_cache,
        block_table,
        lens,
        FULL,
        max_seq_len_hint=2000,
        sinks=sinks,
    )
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e-3
