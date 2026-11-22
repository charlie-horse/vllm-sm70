# SPDX-License-Identifier: Apache-2.0
"""M12 CPU checks: the paged torch attention of Inkling 2
(models/inkling2_sm70/attention_torch.py) against the dense reference
(models/inkling2_sm70/ops/torch_ref.py), fp32.

Needs upstream v0.33.0's Python importable (skipped otherwise). A sliding
layer (window 8, 4 heads on 2 KV heads) and a global layer (6 heads on 2 KV
heads, bias extent 8, log scaling from position 16) run over a fake paged
cache with 4 tokens per block. Sequence lengths 7, 8 and 9 put the last
query at one key under, at and over the window; length 40 passes the
log-scaling floor. In sliding layers every key that left the window is
overwritten with NaN before it could be read.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inkling2_paged import FakePagedKV

HIDDEN = 16
HEAD_DIM = 4
KV_HEADS = 2
D_REL = 4
TAPS = 4
BLOCK = 4
WINDOW = 8
EXTENT = 8
QK_CLIP = 2.0
REL_CLIP = 1.5
LOG_SCALING = (0.1, 16.0)
TOL = 1e-5
LENGTHS = [7, 8, 9, 40]


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


class Layer:
    """Random weights of one attention layer and its dense reference."""

    def __init__(self, kind: str, seed: int = 0) -> None:
        from vllm_sm70.models.inkling2_sm70.ops import torch_ref

        self.ref = torch_ref
        self.is_local = kind == "sliding"
        self.heads = 4 if self.is_local else 6
        self.window = WINDOW if self.is_local else None
        self.log_scaling = None if self.is_local else LOG_SCALING
        g = torch.Generator().manual_seed(seed)

        def rand(*shape, scale=1.0):
            return torch.randn(*shape, generator=g) * scale

        width, kv_width = self.heads * HEAD_DIM, KV_HEADS * HEAD_DIM
        conv = {"k_conv": rand(kv_width, TAPS, scale=0.5)}
        conv["v_conv"] = rand(kv_width, TAPS, scale=0.5)
        if not self.is_local:
            conv = {
                "k_conv": rand(TAPS, kv_width, scale=0.5),
                "v_conv": rand(TAPS, kv_width, scale=0.5),
                "k_conv_proj": rand(KV_HEADS * TAPS, HIDDEN, scale=0.2),
                "v_conv_proj": rand(KV_HEADS * TAPS, HIDDEN, scale=0.2),
            }
        self.w = torch_ref.AttentionWeights(
            q=rand(width, HIDDEN),
            k=rand(kv_width, HIDDEN),
            v=rand(kv_width, HIDDEN),
            rel=rand(self.heads * D_REL, HIDDEN),
            out=rand(HIDDEN, width, scale=0.3),
            out_gate=rand(width, HIDDEN, scale=0.3),
            sink=rand(self.heads, HIDDEN, scale=0.3),
            q_scale=rand(self.heads, HEAD_DIM, scale=0.5),
            rel_scale=rand(self.heads, D_REL, scale=0.5),
            rel_proj=rand(D_REL, EXTENT),
            **conv,
        )

    def sequence(self, length: int, seed: int) -> SimpleNamespace:
        """The dense output of one sequence and the per-token inputs of the
        paged op: what the prologue hands over."""
        ref, w = self.ref, self.w
        g = torch.Generator().manual_seed(1000 + seed)
        x_q = torch.randn(1, length, HIDDEN, generator=g)
        x_k, x_v = x_q, x_q
        if not self.is_local:
            x_k = torch.randn(1, length, HIDDEN, generator=g)
            x_v = torch.randn(1, length, HIDDEN, generator=g)
        positions = torch.arange(length)
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        if self.window is not None:
            allowed &= distance < self.window
        expected, _ = ref.attention(
            x_q,
            x_k,
            x_v,
            w,
            num_kv_heads=KV_HEADS,
            allowed=allowed,
            relative_distance=distance,
            position_ids=positions[None],
            qk_clip=QK_CLIP,
            rel_clip=REL_CLIP,
            log_scaling=self.log_scaling,
        )
        ids = positions[None]
        query = ref.query_projection(
            x_q, w.q, w.q_scale, QK_CLIP, ids, self.log_scaling
        )
        rel = ref.query_projection(
            x_q, w.rel, w.rel_scale, REL_CLIP, ids, self.log_scaling
        )
        key = ref.key_value_projection(
            x_k, w.k, w.k_conv, KV_HEADS, w.k_conv_proj, QK_CLIP
        )
        value = ref.key_value_projection(x_v, w.v, w.v_conv, KV_HEADS, w.v_conv_proj)
        return SimpleNamespace(
            length=length,
            expected=expected[0],
            query=query[0],
            rel_logits=(rel @ w.rel_proj)[0],
            sinks=F.linear(x_q, w.sink)[0],
            gate=F.linear(x_q, w.out_gate)[0],
            key=key[0].transpose(0, 1),
            value=value[0].transpose(0, 1),
        )

    def cache(self, sequences, dtype=torch.float32) -> FakePagedKV:
        return FakePagedKV(
            num_requests=len(sequences),
            max_len=max(s.length for s in sequences),
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            block_size=BLOCK,
            sliding_window=self.window,
            dtype=dtype,
        )


def run_batch(layer, cache, sequences, batch, *, pad_tokens=0, dtype=None, **chunks):
    """Write the batch's K/V, run the paged op, and return (output, expected)
    over the real tokens. `batch` holds (request, first position, tokens)."""
    from vllm_sm70.models.inkling2_sm70.attention_torch import attention_with_metadata

    dtype = dtype or torch.float32
    rows = {name: [] for name in ("query", "rel_logits", "sinks", "gate", "expected")}
    for request, first, n in batch:
        seq = sequences[request]
        span = slice(first, first + n)
        cache.write(request, first, seq.key[span], seq.value[span])
        if layer.window is not None:
            cache.poison(request, first - layer.window + 1)
        for name, parts in rows.items():
            parts.append(getattr(seq, name)[span])
    inputs = {}
    for name, parts in rows.items():
        tensor = torch.cat(parts)
        pad = torch.full((pad_tokens, *tensor.shape[1:]), 7.0)
        inputs[name] = torch.cat([tensor, pad])
    num_tokens = inputs["query"].shape[0] - pad_tokens
    output = torch.full((num_tokens + pad_tokens, HIDDEN), float("nan"), dtype=dtype)
    returned = attention_with_metadata(
        inputs["query"].to(dtype),
        inputs["rel_logits"].to(dtype),
        inputs["sinks"].to(dtype),
        inputs["gate"].to(dtype),
        cache.kv_cache,
        cache.metadata(batch, pad_tokens=pad_tokens),
        output,
        layer.w.out.to(dtype),
        **chunks,
    )
    assert returned is output
    assert not output[num_tokens:].any()
    return output[:num_tokens], inputs["expected"][:num_tokens]


def run_steps(layer, length, sizes, **chunks):
    """One request fed in steps of `sizes` (cycled) tokens; the max error."""
    seq = layer.sequence(length, seed=length)
    cache = layer.cache([seq])
    first, step, worst = 0, 0, 0.0
    while first < length:
        n = min(sizes[step % len(sizes)], length - first)
        output, expected = run_batch(layer, cache, [seq], [(0, first, n)], **chunks)
        worst = max(worst, (output - expected).abs().max().item())
        first += n
        step += 1
    return worst


KINDS = ["sliding", "global"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("length", LENGTHS)
def test_prefill_equals_dense(kind, length):
    assert run_steps(Layer(kind), length, [length]) < TOL


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("length", LENGTHS)
def test_prefill_then_decode_equals_dense(kind, length):
    assert run_steps(Layer(kind), length, [3, 1]) < TOL
    assert run_steps(Layer(kind), length, [1]) < TOL


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("length", LENGTHS)
def test_chunked_prefill_equals_dense(kind, length):
    assert run_steps(Layer(kind), length, [1, 3, 7]) < TOL
    assert run_steps(Layer(kind), length, [7, 3]) < TOL


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "chunks",
    [
        {"query_chunk": 1, "key_chunk": 1},
        {"query_chunk": 2, "key_chunk": 3},
        {"query_chunk": 5, "key_chunk": 1000},
        {"query_chunk": 1000, "key_chunk": 5},
    ],
)
def test_chunked_gather_equals_unchunked(kind, chunks):
    layer = Layer(kind)
    seq = layer.sequence(40, seed=3)
    outputs = []
    for kwargs in ({}, chunks):
        cache = layer.cache([seq])
        prefix, _ = run_batch(layer, cache, [seq], [(0, 0, 29)], **kwargs)
        chunk, expected = run_batch(layer, cache, [seq], [(0, 29, 10)], **kwargs)
        decode, last = run_batch(layer, cache, [seq], [(0, 39, 1)], **kwargs)
        assert (chunk - expected).abs().max() < TOL
        assert (decode - last).abs().max() < TOL
        outputs.append(torch.cat([prefix, chunk, decode]))
    assert (outputs[0] - outputs[1]).abs().max() < 2e-6


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("chunks", [{}, {"query_chunk": 2, "key_chunk": 5}])
def test_mixed_batch_decode_first_with_padding(kind, chunks):
    """Two decodes at different depths, a short and a long prefill chunk and
    two padding tokens in one batch; every request keeps to its own blocks."""
    layer = Layer(kind)
    sequences = [layer.sequence(n, seed=n) for n in (40, 9, 23, 12)]
    cache = layer.cache(sequences)
    run_batch(layer, cache, sequences, [(0, 0, 39)])
    run_batch(layer, cache, sequences, [(1, 0, 2)])
    run_batch(layer, cache, sequences, [(2, 0, 6)])
    batch = [(0, 39, 1), (1, 2, 1), (2, 6, 17), (3, 0, 12)]
    output, expected = run_batch(layer, cache, sequences, batch, pad_tokens=2, **chunks)
    assert (output - expected).abs().max() < TOL
    # Batch rows need not be in request order.
    batch = [(3, 0, 12), (1, 3, 1)]
    output, expected = run_batch(layer, cache, sequences, batch)
    assert (output - expected).abs().max() < TOL


def test_gather_output_is_the_dense_attention_before_the_gate():
    """`gather_attention` alone equals `dense_attention` of the reference."""
    from vllm_sm70.models.inkling2_sm70.attention_torch import gather_attention

    layer = Layer("global")
    seq = layer.sequence(21, seed=5)
    cache = layer.cache([seq])
    cache.write(0, 0, seq.key, seq.value)
    positions = torch.arange(21)
    distance = positions[:, None] - positions[None, :]
    index = distance.clamp(0, EXTENT - 1).expand(layer.heads, -1, -1)
    bias = seq.rel_logits.transpose(0, 1).gather(-1, index)
    bias = bias.masked_fill((distance < 0) | (distance >= EXTENT), 0.0)
    expected, _ = layer.ref.dense_attention(
        seq.query.transpose(0, 1)[None],
        seq.key.transpose(0, 1)[None],
        seq.value.transpose(0, 1)[None],
        distance >= 0,
        HEAD_DIM**-0.5,
        bias[None],
        seq.sinks.t()[None],
    )
    out = gather_attention(
        seq.query,
        seq.rel_logits,
        seq.sinks,
        cache.kv_cache,
        cache.metadata([(0, 0, 21)]),
        torch.empty(21, layer.heads, HEAD_DIM),
        query_chunk=4,
        key_chunk=6,
    )
    assert (out.flatten(1) - expected[0]).abs().max() < TOL


def test_gated_output_without_projection():
    from vllm_sm70.models.inkling2_sm70.attention_torch import (
        attention_with_metadata,
        gather_attention,
    )

    layer = Layer("sliding")
    seq = layer.sequence(9, seed=2)
    cache = layer.cache([seq])
    cache.write(0, 0, seq.key, seq.value)
    md = cache.metadata([(0, 0, 9)])
    width = layer.heads * HEAD_DIM
    gated = attention_with_metadata(
        seq.query,
        seq.rel_logits,
        seq.sinks,
        seq.gate,
        cache.kv_cache,
        md,
        torch.empty(9, width),
    )
    plain = gather_attention(
        seq.query,
        seq.rel_logits,
        seq.sinks,
        cache.kv_cache,
        md,
        torch.empty(9, layer.heads, HEAD_DIM),
    )
    assert torch.allclose(gated, plain.flatten(1) * torch.sigmoid(seq.gate), atol=1e-6)
    assert torch.allclose(F.linear(gated, layer.w.out), seq.expected, atol=TOL)


def test_window_edge_drops_exactly_the_key_at_distance_window():
    """With 9 keys the last query of a sliding layer must not see key 0, and
    must see key 1: poisoning either one shows which were read."""
    layer = Layer("sliding")
    seq = layer.sequence(9, seed=9)
    for poisoned, clean in ((0, True), (1, False)):
        cache = layer.cache([seq])
        cache.write(0, 0, seq.key, seq.value)
        slot = cache.slots(0, torch.tensor([poisoned]))
        cache.kv_cache.transpose(1, 2).flatten(0, 1)[slot] = float("nan")
        from vllm_sm70.models.inkling2_sm70.attention_torch import (
            attention_with_metadata,
        )

        output = attention_with_metadata(
            seq.query[8:],
            seq.rel_logits[8:],
            seq.sinks[8:],
            seq.gate[8:],
            cache.kv_cache,
            cache.metadata([(0, 8, 1)]),
            torch.empty(1, HIDDEN),
            layer.w.out,
        )
        assert bool(torch.isfinite(output).all()) is clean
        if clean:
            assert (output[0] - seq.expected[8]).abs().max() < TOL


def test_fp16_runs_and_stays_near_the_fp32_reference():
    layer = Layer("global")
    seq = layer.sequence(40, seed=4)
    cache = layer.cache([seq], dtype=torch.float16)
    prefix, expected = run_batch(
        layer, cache, [seq], [(0, 0, 33)], dtype=torch.float16, key_chunk=16
    )
    assert prefix.dtype == torch.float16
    assert (prefix.float() - expected).abs().max() < 3e-2
    decode, expected = run_batch(layer, cache, [seq], [(0, 33, 1)], dtype=torch.float16)
    assert (decode.float() - expected).abs().max() < 3e-2


def test_rejects_bad_arguments():
    from vllm_sm70.models.inkling2_sm70.attention_torch import (
        gather_attention,
        kv_cache_views,
    )

    layer = Layer("sliding")
    seq = layer.sequence(4, seed=1)
    cache = layer.cache([seq])
    md = cache.metadata([(0, 0, 4)])
    out = torch.empty(4, layer.heads, HEAD_DIM)
    args = (seq.query, seq.rel_logits, seq.sinks)
    with pytest.raises(ValueError, match="chunk sizes"):
        gather_attention(*args, cache.kv_cache, md, out, key_chunk=0)
    with pytest.raises(ValueError, match="does not fit"):
        gather_attention(*args, cache.kv_cache[..., :6], md, out)
    with pytest.raises(ValueError, match="expected a K/V cache"):
        kv_cache_views(cache.kv_cache, HEAD_DIM + 1)
    key_cache, value_cache = kv_cache_views(cache.kv_cache, HEAD_DIM)
    assert key_cache.shape == value_cache.shape == (2, BLOCK, KV_HEADS, HEAD_DIM)


def test_eager_op_reads_the_owner_metadata_of_the_current_forward(monkeypatch):
    """`paged_attention` looks its metadata up on every call (a replayed
    break has only the tensors it was recorded with) and writes zeros while
    no cache is bound."""
    from vllm import forward_context

    from vllm_sm70.models.inkling2_sm70.attention_torch import paged_attention

    layer = Layer("sliding")
    seq = layer.sequence(12, seed=6)
    cache = layer.cache([seq])
    owner = SimpleNamespace(prefix="model.layers.0.attn", kv_cache=torch.tensor([]))
    context = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)

    def call(first, n):
        cache.write(0, first, seq.key[first : first + n], seq.value[first : first + n])
        span = slice(first, first + n)
        output = torch.full((n, HIDDEN), float("nan"))
        result = paged_attention(
            seq.query[span],
            seq.rel_logits[span],
            seq.sinks[span],
            seq.gate[span],
            owner,
            output,
            layer.w.out,
        )
        assert result is None
        return output

    assert not call(0, 5).any()
    owner.kv_cache = cache.kv_cache
    for first, n in ((0, 5), (5, 6), (11, 1)):
        context.attn_metadata = {owner.prefix: cache.metadata([(0, first, n)])}
        assert (call(first, n) - seq.expected[first : first + n]).abs().max() < TOL
