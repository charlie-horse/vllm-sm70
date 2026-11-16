# SPDX-License-Identifier: Apache-2.0
"""F12 on a V100: greedy identity of the vocab-parallel top-1.

The fake 2-rank split of test_f12_greedy_fastpath_cpu.py, on CUDA: real
ParallelLMHead shards in FP16, the real LogitsProcessor (its LM-head GEMM is
whatever the platform dispatches, the same call on both sides), and a
stand-in pair all-gather. Each rank's token must equal ``argmax`` over the
gathered FP32 logits (upstream ``Sampler.greedy_sample`` on CUDA) bit for
bit, including CUDA's ``max``/``argmax`` tie order. Engine-level greedy
identity (F12 vs upstream inside one Flint engine) is
tests/models/test_m10_flint_sampler.py G5/G6.

    pytest -m gpu sm70/tests/sampler/test_f12_greedy_fastpath_gpu.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_f12_greedy_fastpath_cpu import (  # noqa: E402
    FLINT_VOCAB,
    _fast_path,
    _logits_processor,
    _reference,
    _shards,
    vllm_env,  # noqa: F401  (fixture)
)

pytestmark = pytest.mark.gpu


@pytest.fixture
def cuda_sm70():
    if not torch.cuda.is_available():
        pytest.skip("requires an SM70 GPU")
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (7, 0):
            device = torch.device(f"cuda:{index}")
            torch.cuda.set_device(device)
            return device
    pytest.skip("requires an SM70 GPU")


def _identity(monkeypatch, vllm_env, weight, hidden, tp_size=2):
    heads = [h.to(weight.device) for h in _shards(vllm_env, weight, tp_size)]
    lp = _logits_processor(weight.shape[0])
    want = _reference(lp, heads, hidden)
    for got in _fast_path(monkeypatch, vllm_env, lp, heads, hidden):
        assert got.device == hidden.device
        assert torch.equal(got, want), (got.tolist(), want.tolist())
    return want


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 20])
@pytest.mark.parametrize("seed", range(4))
def test_flint_tp2_greedy_identity(monkeypatch, vllm_env, cuda_sm70, rows, seed):
    """Flint's vocabulary and hidden size at TP2, FP16, decode batch sizes up
    to the prod max-num-seqs."""
    g = torch.Generator(device=cuda_sm70).manual_seed(seed)
    weight = (
        torch.randn(FLINT_VOCAB, 2048, generator=g, device=cuda_sm70) * 0.02
    ).half()
    hidden = torch.randn(rows, 2048, generator=g, device=cuda_sm70).half()
    _identity(monkeypatch, vllm_env, weight, hidden)


def test_ties_and_padding_on_cuda(monkeypatch, vllm_env, cuda_sm70):
    """Exact FP16 ties across ranks and inside a rank, and a padded last
    shard (509 -> 512) whose padding rows would win if not masked."""
    vocab, hidden_size = 509, 64
    weight = torch.randn(vocab, hidden_size, device=cuda_sm70) * 0.01
    for t in (17, 300, 301, 508):
        weight[t] = 0.5
    weight = weight.half()
    hidden = torch.ones(3, hidden_size, device=cuda_sm70).half()
    heads = [h.to(cuda_sm70) for h in _shards(vllm_env, weight, 2)]
    pad = heads[1].shard_indices.num_org_vocab_padding
    assert pad == 3
    heads[1].weight.data[-pad:] = 4.0
    lp = _logits_processor(vocab)
    want = _reference(lp, heads, hidden)
    assert want.tolist() == [17, 17, 17]
    for got in _fast_path(monkeypatch, vllm_env, lp, heads, hidden):
        assert torch.equal(got, want)


@pytest.mark.parametrize("tp_size", [1, 4])
def test_other_tp_sizes(monkeypatch, vllm_env, cuda_sm70, tp_size):
    g = torch.Generator(device=cuda_sm70).manual_seed(tp_size)
    weight = (torch.randn(4099, 256, generator=g, device=cuda_sm70) * 0.05).half()
    hidden = torch.randn(3, 256, generator=g, device=cuda_sm70).half()
    _identity(monkeypatch, vllm_env, weight, hidden, tp_size=tp_size)
