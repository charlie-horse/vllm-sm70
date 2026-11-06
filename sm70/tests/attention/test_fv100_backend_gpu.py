# SPDX-License-Identifier: Apache-2.0
"""K06 GPU: the Flash-V100 backend end to end against upstream TRITON_ATTN.

Both backends get the same inputs on the same upstream (LBNHC) KV cache:
the new tokens' K/V are written with upstream's do_kv_cache_update
(triton_reshape_and_cache_flash), then FlashAttnV100Impl and
TritonAttentionImpl run the same mixed batch (decode rows first, then
prefill rows of several query lengths, fresh and chunked). Max |diff| must
stay below 2e-2. A second test captures a uniform decode batch in a CUDA
graph (UNIFORM_BATCH) and replays it with new queries and sequence lengths,
rebuilding the metadata before each replay as the V1 runner does.

Shapes: Flint D128 GQA8 (16q/2kv) with sliding window 513 and sinks, and full
attention with sinks; Qwen3.8/3.6 D256 GQA6 (12q/2kv) and GQA8 (8q/1kv).
Needs upstream v0.33.0 Python importable and an SM70 GPU with _sm70_C.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _upstream():
    """Upstream v0.33.0 Python must be importable (checked at run time, so
    CPU collection never imports vLLM)."""
    pytest.importorskip("vllm.v1.attention.backends.triton_attn")


MAX_ABS = 2e-2

CASES = {
    "flint-swa513": dict(heads=(16, 2), head_size=128, window=513, sinks=True),
    "flint-full": dict(heads=(16, 2), head_size=128, window=None, sinks=True),
    "qwen38-gqa6": dict(heads=(12, 2), head_size=256, window=None, sinks=False),
    "qwen36-gqa8": dict(heads=(8, 1), head_size=256, window=None, sinks=False),
}


def _vllm_config(heads, head_size, max_model_len, max_num_seqs=16):
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
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=None,
    )


def _builders(monkeypatch, case, block_size, device, max_model_len=8192):
    from vllm.v1.attention.backends import triton_attn

    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100MetadataBuilder

    monkeypatch.setattr(
        triton_attn,
        "get_num_attention_heads_from_layers",
        lambda *a: case["heads"][0],
    )
    cfg = _vllm_config(case["heads"], case["head_size"], max_model_len)
    spec = SimpleNamespace(block_size=block_size)
    return (
        FlashAttnV100MetadataBuilder(spec, ["l0"], cfg, device),
        triton_attn.TritonAttentionMetadataBuilder(spec, ["l0"], cfg, device),
    )


def _impls(case, device):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

    num_heads, num_kv_heads = case["heads"]
    d = case["head_size"]
    sinks = None
    if case["sinks"]:
        gen = torch.Generator().manual_seed(3)
        sinks = (torch.randn(num_heads, generator=gen) * 2).to(device)
    args = (num_heads, d, d**-0.5, num_kv_heads, None, case["window"], "auto")
    ours = FlashAttnV100Impl(*args, sinks=sinks)
    assert ours._fv100_reason is None
    return ours, TritonAttentionImpl(*args, sinks=sinks)


def _layer(device):
    one = torch.ones(1, dtype=torch.float32, device=device)
    return SimpleNamespace(_k_scale=one, _v_scale=one, _q_scale=one)


def _batch(query_lens, seq_lens, block_size, device, seed):
    """CommonAttentionMetadata over a shuffled block table, plus slots."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    gen = torch.Generator().manual_seed(seed)
    blocks = [-(-s // block_size) for s in seq_lens]
    width = max(blocks)
    num_blocks = sum(blocks) + 4
    perm = torch.randperm(num_blocks, generator=gen)
    table = torch.zeros(len(seq_lens), width, dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(blocks):
        table[row, :count] = perm[cursor : cursor + count]
        cursor += count
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens).cumsum(0)
    slots = []
    for row, (q, s) in enumerate(zip(query_lens, seq_lens)):
        for pos in range(s - q, s):
            slots.append(
                int(table[row, pos // block_size]) * block_size + pos % block_size
            )
    n = int(qsl[-1])
    cm = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(query_lens),
        num_actual_tokens=n,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=table.to(device),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )
    return cm, num_blocks


def _kv_cache(num_blocks, block_size, num_kv_heads, d, device, seed):
    gen = torch.Generator().manual_seed(seed)
    storage = torch.randn(num_blocks, block_size, num_kv_heads, 2 * d, generator=gen)
    return storage.to(device, torch.float16).permute(0, 2, 1, 3)


def _qkv(n, case, device, seed):
    gen = torch.Generator().manual_seed(seed)
    num_heads, num_kv_heads = case["heads"]
    d = case["head_size"]

    def r(*shape):
        return torch.randn(*shape, generator=gen).to(device, torch.float16)

    return r(n, num_heads, d), r(n, num_kv_heads, d), r(n, num_kv_heads, d)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("block_size", [16, 784])
@torch.inference_mode()
def test_mixed_batch_matches_triton(fv100_gpu, monkeypatch, name, block_size):
    device, _ = fv100_gpu
    case = CASES[name]
    ours_builder, triton_builder = _builders(monkeypatch, case, block_size, device)
    ours, triton = _impls(case, device)
    # Reordered batch: 3 decodes, then prefill rows (fresh, chunked, chunked).
    query_lens = [1, 1, 1, 37, 128, 37]
    seq_lens = [700, 64, 2000, 37, 1500, 600]
    cm, num_blocks = _batch(query_lens, seq_lens, block_size, device, seed=1)
    kv = _kv_cache(
        num_blocks, block_size, case["heads"][1], case["head_size"], device, seed=2
    )
    query, key, value = _qkv(cm.num_actual_tokens, case, device, seed=4)
    layer = _layer(device)
    ours.do_kv_cache_update(layer, key, value, kv, cm.slot_mapping)

    md = ours_builder.build(0, cm)
    assert md.num_decodes == 3 and {g.query_len for g in md.prefill_groups} == {37, 128}
    out = torch.empty_like(query)
    ours.forward(layer, query, key, value, kv, md, out)
    ref = torch.empty_like(query)
    triton.forward(layer, query, key, value, kv, triton_builder.build(0, cm), ref)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < MAX_ABS, diff


@pytest.mark.parametrize("name", sorted(CASES))
@torch.inference_mode()
def test_uniform_decode_cuda_graph_replay(fv100_gpu, monkeypatch, name):
    device, _ = fv100_gpu
    case = CASES[name]
    block_size, batch, capacity = 16, 4, 4096
    ours_builder, triton_builder = _builders(
        monkeypatch, case, block_size, device, max_model_len=capacity
    )
    ours, triton = _impls(case, device)
    cm, num_blocks = _batch([1] * batch, [capacity] * batch, block_size, device, 5)
    kv = _kv_cache(
        num_blocks, block_size, case["heads"][1], case["head_size"], device, seed=6
    )
    layer = _layer(device)
    query, _, _ = _qkv(batch, case, device, seed=7)
    out = torch.empty_like(query)

    md = ours_builder.build_for_cudagraph_capture(cm)  # seq_lens -> 1
    assert md.cudagraph_capture and md.num_decodes == batch
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        ours.forward(layer, query, None, None, kv, md, out)  # warm-up
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ours.forward(layer, query, None, None, kv, md, out)

    for step, lens in enumerate(([600, 50, 4096, 1], [3000, 1, 2049, 700])):
        new_q, _, _ = _qkv(batch, case, device, seed=10 + step)
        query.copy_(new_q)
        cm.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        tcm = cm.replace(max_seq_len=max(lens))
        # The V1 runner rebuilds attention metadata eagerly every step before
        # replaying a FULL graph; the rebuild refreshes the persistent
        # fv_seq_lens / decode buffers the captured kernels read.
        step_md = ours_builder.build(0, tcm)
        assert step_md.decode_seq_lens.data_ptr() == md.decode_seq_lens.data_ptr()
        graph.replay()
        ref = torch.empty_like(query)
        triton.forward(layer, query, None, None, kv, triton_builder.build(0, tcm), ref)
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < MAX_ABS, (step, diff)
