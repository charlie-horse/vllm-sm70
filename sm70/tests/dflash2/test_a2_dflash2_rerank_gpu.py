# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2 GPU tests: DFlash2 QPN8 support search + exact FP32 rerank on SM70.

The real Qwen3.8 TP2 head shard (124160 x 5120, two 62080-row chunks) with
K09's channel-FP8 GEMM and the Triton exact dots. Checks: the Triton dots
against an FP32 reference; the route's top-16 equals the FP32 dense top-16
and its values match it; against upstream's FP16 dense head the ids agree
except where FP16 rounding reorders near-ties; a captured graph replays
fresh hidden states; the E4M3 copy's size.

    pytest -m gpu sm70/tests/dflash2/test_a2_dflash2_rerank_gpu.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

N, K, TOP_K = 124160, 5120, 16


@pytest.fixture(scope="module")
def rerank():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail("SM70 GPU present but vllm_sm70._sm70_C is not loaded")
    from vllm_sm70.ops import dflash2_rerank

    if not dflash2_rerank.ops_available():
        pytest.fail(f"_sm70_C lacks the K09 ops {dflash2_rerank.OPS}")
    return dflash2_rerank


@pytest.fixture(scope="module")
def head(rerank):
    torch.manual_seed(20261109)
    module = torch.nn.Module()
    # LM-head-like rows: unit-scale logits for unit-scale hidden states.
    weight = (torch.randn(N, K, device="cuda") / K**0.5).half()
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.bias = None
    module.tp_size = 1
    module.shard_indices = SimpleNamespace(
        num_org_vocab_padding=0, num_added_elements=0, org_vocab_start_index=0
    )
    yield module
    module._sm70_dflash2_rerank = None
    torch.cuda.empty_cache()


def _processor():
    return SimpleNamespace(
        scale=1.0, soft_cap=None, logits_as_input=False, head_dtype=None
    )


def _fp32_dense(x, weight, chunk=16384):
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device)
    xf = x.float()
    for b in range(0, weight.shape[0], chunk):
        out[:, b : b + chunk] = xf @ weight[b : b + chunk].float().t()
    return out


@torch.inference_mode()
def test_triton_dots_match_fp32_reference(rerank, head):
    from vllm_sm70.ops.dflash2_rerank_kernel import exact_logits

    x = torch.randn(7, K, device="cuda").half()
    ids = torch.randint(0, N, (7, 128), device="cuda")
    out = torch.empty((7, 128), device="cuda")
    assert exact_logits(x, head.weight, ids, out)
    ref = torch.empty_like(out)
    rerank.exact_logits_reference(x, head.weight, ids, ref)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m", [1, 7, 8])
@torch.inference_mode()
def test_route_equals_fp32_dense_topk(rerank, head, m):
    torch.manual_seed(m)
    x = torch.randn(m, K, device="cuda").half()
    ids, values = rerank.maybe_candidates(head, _processor(), x, TOP_K)
    state = getattr(head, rerank.STATE_ATTR)
    assert len(state.groups) == 2 and state.cand_ids.shape[1] == 128
    ref_values, ref_ids = torch.topk(_fp32_dense(x, head.weight), TOP_K, dim=-1)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(values, ref_values, rtol=1e-5, atol=1e-4)


@torch.inference_mode()
def test_route_agrees_with_upstream_fp16_head(rerank, head):
    # Upstream ranks FP16-rounded logits, so near-ties may swap; the sets
    # must agree up to tokens within FP16 rounding of the 16th value.
    torch.manual_seed(7)
    x = torch.randn(7, K, device="cuda").half()
    ids, values = rerank.maybe_candidates(head, _processor(), x, TOP_K)
    up_values, up_ids = torch.topk(x @ head.weight.t(), TOP_K, dim=-1)
    for r in range(7):
        ours, theirs = set(ids[r].tolist()), set(up_ids[r].tolist())
        for token in ours ^ theirs:
            exact = _fp32_dense(x[r : r + 1], head.weight[token : token + 1])[0, 0]
            assert abs(exact.item() - values[r, -1].item()) < 2e-2, (r, token)
    torch.testing.assert_close(values, up_values.float(), rtol=0, atol=2e-2)


@torch.inference_mode()
def test_graph_replays_fresh_hidden_states(rerank, head):
    x = torch.randn(7, K, device="cuda").half()
    rerank.maybe_candidates(head, _processor(), x, TOP_K)  # prepared, warm
    state = getattr(head, rerank.STATE_ATTR)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        rerank.shard_topk(state, x, head.weight, TOP_K)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        values, ids = rerank.shard_topk(state, x, head.weight, TOP_K)
    for seed in range(3):
        torch.manual_seed(100 + seed)
        x.copy_(torch.randn(7, K, device="cuda").half())
        graph.replay()
        torch.cuda.synchronize()
        ref_values, ref_ids = torch.topk(_fp32_dense(x, head.weight), TOP_K, dim=-1)
        assert torch.equal(ids, ref_ids), seed
        torch.testing.assert_close(values, ref_values, rtol=1e-5, atol=1e-4)


@torch.inference_mode()
def test_fp8_copy_is_half_the_head(rerank, head):
    rerank.maybe_candidates(
        head, _processor(), torch.randn(7, K, device="cuda").half(), TOP_K
    )
    state = getattr(head, rerank.STATE_ATTR)
    assert state.codes.dtype == torch.uint8 and state.codes.numel() == N * K
    assert state.scales.numel() == N
