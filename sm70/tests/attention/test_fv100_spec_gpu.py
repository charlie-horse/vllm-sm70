# SPDX-License-Identifier: Apache-2.0
"""K07 GPU: Flash-V100 spec-decode verify (smallq -> XQA) and DFlash drafts.

Ports (fork cf9492a):
* tests/kernels/attention/test_sm70_flash_v100_mtp_smallq_exactness.py:
  the token-wise decode expansion equals the causal paged prefill (now for
  both the scalar and the XQA decode).
* tests/kernels/attention/test_sm70_dflash_batched_prefill.py: a batched
  non-causal draft step through the backend matches serial per-request
  prefill, and a CUDA graph replays it after each step's metadata rebuild
  (v0.33 rebuilds metadata eagerly every step; fp16 only, pages 16/832/1648).
* tests/kernels/attention/test_sm70_flash_v100_prefix_decode_rows.py: in a
  mixed batch the verify rows taken by the decode route match the prefill
  route (kill switch off), and the chunk row is bit-identical either way.
* tests/v1/spec_decode/test_dflash2.py (attention parts): causal verify and
  non-causal draft attention against upstream TRITON_ATTN.
Plus: verify output vs TRITON_ATTN on the same inputs (max |diff| < 2e-2,
D256, GQA 6/8, 8 query rows, fp16 KV), eager and graph-replayed.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytestmark = pytest.mark.gpu

SPEC_TOKENS = 7
L = SPEC_TOKENS + 1
MAX_ABS = 2e-2


@pytest.fixture(autouse=True)
def _upstream():
    pytest.importorskip("vllm.v1.attention.backends.triton_attn")


def _cfg(heads, head_size, max_model_len, max_num_seqs=16, spec=SPEC_TOKENS):
    from vllm.config import CUDAGraphMode

    num_heads, num_kv_heads = heads
    model = SimpleNamespace(
        get_num_attention_heads=lambda p: num_heads,
        get_num_kv_heads=lambda p: num_kv_heads,
        get_head_size=lambda: head_size,
        rswa_window=None,
        max_model_len=max_model_len,
    )
    return SimpleNamespace(
        model_config=model,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=[]
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=8192
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=spec, parallel_drafting=False
        ),
        use_v2_model_runner=False,
    )


def _builders(monkeypatch, heads, head_size, block_size, device, max_model_len):
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.kv_cache_interface import KVQuantMode

    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100MetadataBuilder

    monkeypatch.setattr(
        triton_attn, "get_num_attention_heads_from_layers", lambda *a: heads[0]
    )
    cfg = _cfg(heads, head_size, max_model_len)
    spec = SimpleNamespace(block_size=block_size, kv_quant_mode=KVQuantMode.NONE)
    return (
        FlashAttnV100MetadataBuilder(spec, ["l0"], cfg, device),
        triton_attn.TritonAttentionMetadataBuilder(spec, ["l0"], cfg, device),
    )


def _impls(heads, head_size, window=None, sinks=None):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

    args = (heads[0], head_size, head_size**-0.5, heads[1], None, window, "auto")
    return FlashAttnV100Impl(*args, sinks=sinks), TritonAttentionImpl(
        *args, sinks=sinks
    )


def _layer(device):
    one = torch.ones(1, dtype=torch.float32, device=device)
    return SimpleNamespace(_k_scale=one, _v_scale=one, _q_scale=one)


def _cm(query_lens, seq_lens, table, device, causal=True, num_tokens=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens).cumsum(0)
    n = num_tokens or int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(query_lens),
        num_actual_tokens=n,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=table,
        slot_mapping=torch.zeros(n, dtype=torch.int64, device=device),
        causal=causal,
    )


def _kv(num_blocks, block_size, h_kv, d, device, seed):
    gen = torch.Generator().manual_seed(seed)
    storage = torch.randn(num_blocks, block_size, h_kv, 2 * d, generator=gen)
    return storage.to(device, torch.float16).permute(0, 2, 1, 3)


def _table(batch, width, pool, device, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, pool, (batch, width), generator=gen, dtype=torch.int32).to(
        device
    )


def _q(n, heads, d, device, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, heads[0], d, generator=gen).to(device, torch.float16)


# ----------------------------------------------- smallq exactness (port)


@pytest.mark.parametrize("query_len", [5, L, 16])
@pytest.mark.parametrize("heads", [(8, 1), (12, 2)], ids=["gqa8", "gqa6"])
@pytest.mark.parametrize("decode", ["scalar", "xqa"])
@torch.inference_mode()
def test_smallq_decode_matches_causal_prefill(fv100_gpu, query_len, heads, decode):
    device, fv100 = fv100_gpu
    torch.manual_seed(1234)
    seq_len, block_size, d = 4097, 16, 256
    num_blocks = -(-seq_len // block_size)
    k = torch.randn(
        num_blocks, block_size, heads[1], d, dtype=torch.float16, device=device
    )
    v = torch.randn_like(k)
    table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    q = torch.randn(query_len, heads[0], d, dtype=torch.float16, device=device)
    lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    prefill = fv100.flash_attn_prefill_paged(q[None], k, v, table, lens, causal=True)[0]
    rows_table = table.repeat(query_len, 1).contiguous()
    rows_lens = torch.arange(
        seq_len - query_len + 1, seq_len + 1, dtype=torch.int32, device=device
    )
    fn = (
        fv100.flash_attn_decode_paged_xqa
        if decode == "xqa"
        else fv100.flash_attn_decode_paged
    )
    smallq = fn(
        q,
        k,
        v,
        rows_table,
        rows_lens,
        max_seq_len_hint=seq_len,
        workspace_seq_capacity_hint=seq_len,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(smallq, prefill, atol=1.5e-2, rtol=1e-2)


# ------------------------------- A1: one-launch expansion == torch expansion


@pytest.mark.parametrize("width", [3, 129, 300])
@pytest.mark.parametrize("row_stride_pad", [0, 37], ids=["dense", "strided"])
@torch.inference_mode()
def test_smallq_kernel_matches_torch_expansion(
    fv100_gpu, monkeypatch, width, row_stride_pad
):
    """Bit-identical to the torch path: mixed lengths, empty rows, padding.

    When strided, the block table is a column slice of a wider buffer (row
    stride > width). Padding tokens get seq_len 0; their table
    rows are not part of the contract (neither path's decode reads them).
    """
    from vllm_sm70.attention.flash_v100.spec.verify import (
        VerifyExpansion,
        expand_reference,
    )

    device, _ = fv100_gpu
    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SMALLQ_METADATA", raising=False)
    max_reqs = 16
    cap = max_reqs * L + 8  # room for padding tokens after a full batch
    kern = VerifyExpansion(cap, device, max_reqs=max_reqs)
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_SMALLQ_METADATA", "0")
    ref = VerifyExpansion(cap, device, max_reqs=max_reqs)
    assert kern.uses_kernel and not ref.uses_kernel
    gen = torch.Generator().manual_seed(width + row_stride_pad)
    cases = [
        [L] * 16,
        [L],
        [L, 0, L, 3, 1, L],  # an empty row (equal boundaries) and short rows
        [1] * 7,
        [5, L, L, 2],
    ]
    for query_lens in cases:
        rows = len(query_lens)
        qsl = torch.zeros(rows + 1, dtype=torch.int32)
        qsl[1:] = torch.tensor(query_lens).cumsum(0)
        mapped = int(qsl[-1])
        seq = [int(torch.randint(q, 5000, (1,), generator=gen)) + 1 for q in query_lens]
        wide = torch.randint(
            0, 10**6, (rows, width + row_stride_pad), generator=gen, dtype=torch.int32
        ).to(device)
        table = wide[:, :width]
        for pad in (0, 5):
            args = (
                qsl.to(device),
                torch.tensor(seq, dtype=torch.int32, device=device),
                table,
                rows,
                mapped,
                mapped + pad,
            )
            ks, kt = kern.expand(*args)
            rs, rt = ref.expand(*args)
            torch.cuda.synchronize()
            assert torch.equal(ks, rs), (query_lens, pad)
            assert torch.equal(kt[:mapped], rt[:mapped]), (query_lens, pad)
            want_rows, want_lens = expand_reference(query_lens, seq)
            assert ks[:mapped].tolist() == want_lens
            assert ks[mapped:].tolist() == [0] * pad
            assert torch.equal(kt[:mapped], table[torch.tensor(want_rows)])


@torch.inference_mode()
def test_smallq_kernel_rewrites_the_same_buffers(fv100_gpu, monkeypatch):
    # Graph safety: each step's kernel launch writes the builder's
    # persistent buffers in place, like the torch path.
    from vllm_sm70.attention.flash_v100.spec.verify import VerifyExpansion

    device, _ = fv100_gpu
    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SMALLQ_METADATA", raising=False)
    exp = VerifyExpansion(4 * L, device, max_reqs=4)
    assert exp.uses_kernel
    qsl = torch.arange(0, 5 * L, L, dtype=torch.int32, device=device)
    table = torch.arange(4 * 8, dtype=torch.int32, device=device).view(4, 8)
    ptrs = set()
    for step in range(3):
        lens = torch.tensor([100 + step, 9, 40, 2000], dtype=torch.int32, device=device)
        s, t = exp.expand(qsl, lens, table, 4, 4 * L, 4 * L)
        ptrs.add((s.data_ptr(), t.data_ptr()))
        assert s[L - 1].item() == 100 + step
    assert len(ptrs) == 1


# ----------------------------------------- verify vs TRITON_ATTN (+ graph)

# B2's switch. The K07 tests below set it to "0": B2 takes 8-token verify by
# default and has its own tests (B2 section), and K07 stays the fallback.
VERIFY = "VLLM_SM70_FLASH_V100_VERIFY_KERNEL"


@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["gqa6", "gqa8"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("block_size", [16, 784])
@torch.inference_mode()
def test_verify_matches_triton(fv100_gpu, monkeypatch, heads, batch, block_size):
    device, _ = fv100_gpu
    monkeypatch.setenv(VERIFY, "0")
    d, context = 256, 4096
    ours_b, tri_b = _builders(monkeypatch, heads, d, block_size, device, 8192)
    ours, tri = _impls(heads, d)
    width = -(-context // block_size)
    table = _table(batch, width, 64, device, seed=1)
    kv = _kv(64, block_size, heads[1], d, device, seed=2)
    lens = [context - 37 * i for i in range(batch)]
    cm = _cm([L] * batch, lens, table, device)
    md = ours_b.build(0, cm)
    assert md.spec_verify and md.decode_rows == L * batch
    q = _q(L * batch, heads, d, device, seed=3)
    layer = _layer(device)
    out, ref = torch.empty_like(q), torch.empty_like(q)
    ours.forward(layer, q, None, None, kv, md, out)
    tri.forward(layer, q, None, None, kv, tri_b.build(0, cm), ref)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < MAX_ABS, diff


@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["gqa6", "gqa8"])
@torch.inference_mode()
def test_verify_graph_replays_after_rebuild(fv100_gpu, monkeypatch, heads):
    """UNIFORM_BATCH: capture a verify batch, rebuild metadata, replay."""
    device, _ = fv100_gpu
    monkeypatch.setenv(VERIFY, "0")
    d, block_size, batch, capacity = 256, 16, 4, 4096
    ours_b, tri_b = _builders(monkeypatch, heads, d, block_size, device, capacity)
    ours, tri = _impls(heads, d)
    width = -(-capacity // block_size)
    table = _table(batch, width, 64, device, seed=4)
    kv = _kv(64, block_size, heads[1], d, device, seed=5)
    layer = _layer(device)
    q = _q(L * batch, heads, d, device, seed=6)
    out = torch.empty_like(q)
    ours_b.build(0, _cm([L] * batch, [capacity] * batch, table, device))  # sizes
    md = ours_b.build_for_cudagraph_capture(
        _cm([L] * batch, [capacity] * batch, table, device)
    )
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        ours.forward(layer, q, None, None, kv, md, out)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.forward(layer, q, None, None, kv, md, out)
    for step, lens in enumerate(([600, 50, 4096, L], [3000, 9, 2049, 700])):
        q.copy_(_q(L * batch, heads, d, device, seed=10 + step))
        table.random_(0, 64)
        cm = _cm([L] * batch, lens, table, device)
        ours_b.build(0, cm)  # the step's eager rebuild refreshes the buffers
        graph.replay()
        ref = torch.empty_like(q)
        tri.forward(layer, q, None, None, kv, tri_b.build(0, cm), ref)
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < MAX_ABS, (step, diff)


# B1 envelopes: (max_model_len, capture lengths, replay steps). The 150000
# envelope replays contexts above 32K (40000, 100000).
B1_CASES = {
    "env32k": (32768, 4096, ([1000, 4096], [5000, 300])),
    "env150k": (150000, 40000, ([40000, 100000], [100000, 2000])),
}


@pytest.mark.parametrize("case", sorted(B1_CASES))
@pytest.mark.parametrize("query_len", [L, 1], ids=["verify", "decode"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("small", ["1", "0"], ids=["b1", "b1-off"])
@torch.inference_mode()
def test_small_batch_graph_at_long_envelopes(
    fv100_gpu, monkeypatch, case, query_len, batch, small
):
    """B1: a captured XQA decode of <= 16 rows (verify rows of one or two
    requests, or single-token rows) at a 32K or 150K envelope (256-token
    partitions; 1024 with the switch off) replays to TRITON_ATTN's output,
    including contexts above 32K. B2 is off (VERIFY): the verify rows are
    K07's XQA expansion; B2's envelope test is
    test_verify_kernel_graph_replays_after_rebuild."""
    device, fv100 = fv100_gpu
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_SMALL_BATCH_PARTITION", small)
    monkeypatch.setenv(VERIFY, "0")
    capacity, cap_len, steps = B1_CASES[case]
    heads, d, block_size = (12, 2), 256, 16
    fv100.clear_workspaces()
    ours_b, tri_b = _builders(monkeypatch, heads, d, block_size, device, capacity)
    ours, tri = _impls(heads, d)
    width = -(-capacity // block_size)
    pool = 512
    table = _table(batch, width, pool, device, seed=14)
    kv = _kv(pool, block_size, heads[1], d, device, seed=15)
    layer = _layer(device)
    tokens = query_len * batch
    q = _q(tokens, heads, d, device, seed=16)
    out = torch.empty_like(q)
    qlens = [query_len] * batch
    lens0 = [cap_len] * batch
    ours_b.build(0, _cm(qlens, lens0, table, device))
    md = ours_b.build_for_cudagraph_capture(_cm(qlens, lens0, table, device))
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        ours.forward(layer, q, None, None, kv, md, out)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.forward(layer, q, None, None, kv, md, out)
    partitions = {key[4] for key in fv100._decode_workspace_cache}
    assert (256 in partitions) == (small == "1"), partitions
    for step, lens in enumerate(steps):
        q.copy_(_q(tokens, heads, d, device, seed=20 + step))
        table.random_(0, pool)
        cm = _cm(qlens, lens[:batch], table, device)
        ours_b.build(0, cm)
        graph.replay()
        ref = torch.empty_like(q)
        tri.forward(layer, q, None, None, kv, tri_b.build(0, cm), ref)
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < MAX_ABS, (step, lens[:batch], diff)


# ------------------------- B2: verify kernel (KV read once per request)

# Lengths around the 128-key partition and 32-key tile edges, and long ones.
B2_LENS = [8, 31, 127, 128, 129, 161, 300, 4096, 4097, 40000]


def _verify_forward(ours_b, ours, layer, q, kv, cm, out):
    md = ours_b.build(0, cm)
    assert md.verify_query_len == L
    ours.forward(layer, q, None, None, kv, md, out)


@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["gqa6", "gqa8"])
@pytest.mark.parametrize("block_size", [16, 832])
@pytest.mark.parametrize("batch", [1, 3])
@torch.inference_mode()
def test_verify_kernel_matches_triton_and_k07(
    fv100_gpu, monkeypatch, heads, block_size, batch
):
    """Eager: the verify kernel against TRITON_ATTN and against K07's XQA
    expansion (switch off), over lengths at partition and tile edges."""
    device, fv100 = fv100_gpu
    d, capacity = 256, 65536
    ours_b, tri_b = _builders(monkeypatch, heads, d, block_size, device, capacity)
    ours, tri = _impls(heads, d)
    width = -(-capacity // block_size)
    pool = -(-max(B2_LENS) // block_size) + 8
    kv = _kv(pool, block_size, heads[1], d, device, seed=31)
    layer = _layer(device)
    for i in range(0, len(B2_LENS), batch):
        lens = (B2_LENS[i : i + batch] + B2_LENS[:batch])[:batch]
        table = _table(batch, width, pool, device, seed=32 + i)
        q = _q(L * batch, heads, d, device, seed=33 + i)
        cm = _cm([L] * batch, lens, table, device)
        monkeypatch.delenv(VERIFY, raising=False)
        fv100.clear_workspaces()
        out = torch.empty_like(q)
        _verify_forward(ours_b, ours, layer, q, kv, cm, out)
        assert any(
            k[4] == fv100.VERIFY_PARTITION_SIZE for k in fv100._decode_workspace_cache
        )
        ref = torch.empty_like(q)
        tri.forward(layer, q, None, None, kv, tri_b.build(0, cm), ref)
        monkeypatch.setenv(VERIFY, "0")
        k07 = torch.empty_like(q)
        ours.forward(layer, q, None, None, kv, ours_b.build(0, cm), k07)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), lens
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < MAX_ABS, (lens, diff)
        diff_k07 = (out.float() - k07.float()).abs().max().item()
        assert diff_k07 < MAX_ABS, (lens, diff_k07)


@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["gqa6", "gqa8"])
@pytest.mark.parametrize("capacity", [32768, 150000], ids=["env32k", "env150k"])
@pytest.mark.parametrize("batch", [1, 2, 4])
@torch.inference_mode()
def test_verify_kernel_graph_replays_after_rebuild(
    fv100_gpu, monkeypatch, heads, capacity, batch
):
    """A captured verify batch (grid sized for the envelope) replays fresh
    lengths, tables and queries, including contexts above 32K."""
    device, fv100 = fv100_gpu
    monkeypatch.delenv(VERIFY, raising=False)
    d, block_size = 256, 16
    fv100.clear_workspaces()
    ours_b, tri_b = _builders(monkeypatch, heads, d, block_size, device, capacity)
    ours, tri = _impls(heads, d)
    width = -(-capacity // block_size)
    pool = 512
    table = _table(batch, width, pool, device, seed=41)
    kv = _kv(pool, block_size, heads[1], d, device, seed=42)
    layer = _layer(device)
    q = _q(L * batch, heads, d, device, seed=43)
    out = torch.empty_like(q)
    lens0 = [4096] * batch
    ours_b.build(0, _cm([L] * batch, lens0, table, device))
    md = ours_b.build_for_cudagraph_capture(_cm([L] * batch, lens0, table, device))
    assert md.verify_query_len == L
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        ours.forward(layer, q, None, None, kv, md, out)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.forward(layer, q, None, None, kv, md, out)
    partitions = {key[4] for key in fv100._decode_workspace_cache}
    assert partitions == {fv100.VERIFY_PARTITION_SIZE}, partitions
    steps = [[129, 4096, 8, 1000], [min(capacity, 40000), 300, 31000, 128]]
    for step, lens in enumerate(steps):
        lens = lens[:batch]
        q.copy_(_q(L * batch, heads, d, device, seed=50 + step))
        table.random_(0, pool)
        cm = _cm([L] * batch, lens, table, device)
        ours_b.build(0, cm)
        graph.replay()
        ref = torch.empty_like(q)
        tri.forward(layer, q, None, None, kv, tri_b.build(0, cm), ref)
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < MAX_ABS, (step, lens, diff)


@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["gqa6", "gqa8"])
@pytest.mark.parametrize("batch", [2, 4])
@torch.inference_mode()
def test_verify_kernel_graph_with_padding_requests(
    fv100_gpu, monkeypatch, heads, batch
):
    """Capture `batch` verify requests, then replay with batch - 1 live
    ones and a zero-token padding request at the tail (the V2 runner's FULL
    graph padding: seq_len 0). Live rows match TRITON_ATTN on the live
    requests; the padding rows come out zero, through builder and graph."""
    device, fv100 = fv100_gpu
    monkeypatch.delenv(VERIFY, raising=False)
    d, block_size, capacity = 256, 16, 32768
    fv100.clear_workspaces()
    ours_b, tri_b = _builders(monkeypatch, heads, d, block_size, device, capacity)
    ours, tri = _impls(heads, d)
    width = -(-capacity // block_size)
    pool = 512
    table = _table(batch, width, pool, device, seed=71)
    kv = _kv(pool, block_size, heads[1], d, device, seed=72)
    layer = _layer(device)
    q = _q(L * batch, heads, d, device, seed=73)
    out = torch.empty_like(q)
    full = [4096] * batch
    ours_b.build(0, _cm([L] * batch, full, table, device))
    md = ours_b.build_for_cudagraph_capture(_cm([L] * batch, full, table, device))
    assert md.verify_query_len == L
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        ours.forward(layer, q, None, None, kv, md, out)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.forward(layer, q, None, None, kv, md, out)
    live = batch - 1
    lens = [300 + 1000 * i for i in range(live)]
    q.copy_(_q(L * batch, heads, d, device, seed=74))
    table.random_(0, pool)
    padded = _cm([L] * live + [0], lens + [0], table, device, num_tokens=L * batch)
    md_live = ours_b.build(0, padded)
    assert md_live.decode_rows == L * batch and md_live.verify_query_len == L
    out.fill_(7.0)
    graph.replay()
    ref = torch.empty_like(q[: L * live])
    tri.forward(
        layer,
        q[: L * live],
        None,
        None,
        kv,
        tri_b.build(0, _cm([L] * live, lens, table[:live], device)),
        ref,
    )
    torch.cuda.synchronize()
    diff = (out[: L * live].float() - ref.float()).abs().max().item()
    assert diff < MAX_ABS, diff
    assert torch.count_nonzero(out[L * live :]) == 0


@torch.inference_mode()
def test_verify_kernel_padding_requests_write_zeros(fv100_gpu):
    """Rows whose request has seq_len 0 (CUDA-graph padding) come out zero."""
    device, fv100 = fv100_gpu
    heads, d, block_size = (12, 2), 256, 16
    pool, width = 64, 8
    kv = _kv(pool, block_size, heads[1], d, device, seed=61)
    k, v = fv100.split_lbnhc_kv_cache(kv, d)
    q = _q(2 * L, heads, d, device, seed=62)
    table = _table(2 * L, width, pool, device, seed=63)
    table[L:] = table[L : L + 1]  # each request's rows share one table row
    table[:L] = table[:1]
    lens = torch.tensor(
        [100 - L + 1 + j for j in range(L)] + [0] * L, dtype=torch.int32, device=device
    )
    out = torch.full_like(q, 7.0)
    fv100.flash_attn_decode_verify(q, k, v, table, lens, out=out, max_seq_len_hint=128)
    torch.cuda.synchronize()
    assert torch.isfinite(out[:L]).all()
    assert torch.count_nonzero(out[L:]) == 0


# -------------------------------------- mixed batch: verify rows vs prefill


@pytest.mark.parametrize("block_size", [16, 784])
@torch.inference_mode()
def test_prefix_decode_rows_match_prefill_route(fv100_gpu, monkeypatch, block_size):
    device, _ = fv100_gpu
    heads, d = (12, 2), 256
    query_lens = [1, 4, L, 300]  # reordered: decode, verify rows, then a chunk
    lens = [800, 1203, 3001, 1700]
    width = -(-max(lens) // block_size)
    table = _table(len(lens), width, 64, device, seed=7)
    kv = _kv(64, block_size, heads[1], d, device, seed=8)
    q = _q(sum(query_lens), heads, d, device, seed=9)
    layer = _layer(device)
    outs = {}
    for env in ("1", "0"):
        monkeypatch.setenv("VLLM_SM70_FLASH_V100_SPEC", env)
        b, _ = _builders(monkeypatch, heads, d, block_size, device, 4096)
        ours, _ = _impls(heads, d)
        md = b.build(0, _cm(query_lens, lens, table, device))
        assert md.spec_verify == (env == "1")
        outs[env] = torch.empty_like(q)
        ours.forward(layer, q, None, None, kv, md, outs[env])
    torch.cuda.synchronize()
    chunk = slice(sum(query_lens[:3]), None)
    assert torch.equal(outs["1"][chunk], outs["0"][chunk])
    small = slice(0, sum(query_lens[:3]))
    diff = (outs["1"][small].float() - outs["0"][small].float()).abs().max().item()
    assert diff < 1.5e-2, diff


# ------------------------------------------ DFlash non-causal draft (port)


@pytest.mark.parametrize("block", [16, 832, 1648])
@pytest.mark.parametrize(
    "batch,q_len,context", [(1, 8, 2048), (4, 8, 2048), (8, 8, 32768), (4, 4, 4096)]
)
@torch.inference_mode()
def test_dflash_batch_matches_serial_with_graph(
    fv100_gpu, monkeypatch, batch, q_len, context, block
):
    from vllm_sm70.attention.flash_v100.impl import symmetric_window

    device, fv100 = fv100_gpu
    heads, d = (8, 2), 128
    b, tri_b = _builders(monkeypatch, heads, d, block, device, 65536)
    ours, tri = _impls(heads, d, window=2048)
    tokens = batch * q_len
    storage = torch.randn(tokens, heads[0] * d + 32, device=device).half()
    query = storage[:, : heads[0] * d].view(tokens, heads[0], d)  # strided view
    output = torch.full((tokens + 3, heads[0], d), 0.125, device=device).half()
    kv = _kv(64, block, heads[1], d, device, seed=11)
    table = _table(batch, -(-context // block), 64, device, seed=12)
    layer = _layer(device)

    def lens_for(step):
        return [max(q_len, context - i * 17 - step) for i in range(batch)]

    def build(step):
        return b.build(
            0, _cm([q_len] * batch, lens_for(step), table, device, causal=False)
        )

    md = build(0)
    assert md.num_prefills == batch and not md.spec_verify
    ours.forward(layer, query, None, None, kv, md, output)  # warm-up / sizes
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.forward(layer, query, None, None, kv, md, output)
    for step in range(1, 4):
        storage.normal_()
        table.random_(0, 64)
        cm = _cm([q_len] * batch, lens_for(step), table, device, causal=False)
        b.build(0, cm)  # eager per-step rebuild (persistent seq_lens copy)
        graph.replay()
        torch.cuda.synchronize()
        for i in range(batch):
            expected = fv100.flash_attn_prefill_paged(
                query[i * q_len : (i + 1) * q_len].unsqueeze(0).contiguous(),
                *fv100.split_lbnhc_kv_cache(kv, d),
                table[i : i + 1],
                cm.seq_lens[i : i + 1],
                softmax_scale=ours.scale,
                causal=False,
                # Non-causal windows are two-sided (Triton |dist| < w,
                # upstream FA _maybe_symmetrize_window): (2047, 2047).
                window_size=symmetric_window(tuple(ours.sliding_window), False),
            )[0]
            torch.testing.assert_close(
                output[i * q_len : (i + 1) * q_len], expected, atol=2e-3, rtol=1e-2
            )
        assert torch.all(output[tokens:] == 0.125)
        ref = torch.empty_like(query)
        tri.forward(layer, query, None, None, kv, tri_b.build(0, cm), ref)
        diff = (output[:tokens].float() - ref.float()).abs().max().item()
        assert diff < MAX_ABS, (step, diff)
