# SPDX-License-Identifier: Apache-2.0
"""CPU checks: the paged torch short convs and attention prologue of
Inkling 2 (models/inkling2_sm70/sconv.py, prologue.py).

The ops run over fake paged caches (_inkling2_paged_conv.py) with metadata from
the real builders and are compared in fp32 with the dense reference ops
(ops/torch_ref.py) on whole sequences, for a sliding layer and a global
layer of the tiny fixture's config and random weights.

Needs upstream vLLM's Python importable (skipped otherwise).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inkling2_paged_conv import PagedVisit, tiny_text_config

SLIDING, GLOBAL = 0, 5
LAYERS = pytest.mark.parametrize("layer", [SLIDING, GLOBAL], ids=["sliding", "global"])
HIDDEN = 64
LENGTHS = {0: 30, 1: 21}
PREFIX = 16
TOL = 1e-5

# A step is ("batch", [(request, new tokens), ...]) or
# ("fork", source, target, cached prefix).
SCENARIOS = {
    "one_shot": [("batch", [(0, 30)])],
    "prefill_decode": [("batch", [(0, 24)])] + [("batch", [(0, 1)])] * 6,
    "chunks_1": [("batch", [(0, 1)])] * 30,
    "chunks_3": [("batch", [(0, 3)])] * 10,
    "chunks_7": [("batch", [(0, 7)])] * 4 + [("batch", [(0, 2)])],
    "prefix_resume": [
        ("batch", [(0, 30)]),
        ("fork", 0, 1, PREFIX),
        ("batch", [(1, 2)]),
        ("batch", [(1, 1)]),
        ("batch", [(1, 2)]),
    ],
    "interleaved": [
        ("batch", [(0, 5), (1, 3)]),
        ("batch", [(1, 1), (0, 1)]),
        ("batch", [(0, 7), (1, 4)]),
        ("batch", [(1, 2), (0, 3)]),
        ("batch", [(0, 1), (1, 1)]),
        ("batch", [(1, 10), (0, 13)]),
    ],
}
SCENARIO = pytest.mark.parametrize("scenario", list(SCENARIOS))


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


def _weights(layer: int, seed: int = 0):
    """Random AttentionWeights of `layer` of the tiny config, all ranks."""
    from vllm_sm70.models.inkling2_sm70.ops import torch_ref

    config = tiny_text_config()
    local = layer in config.local_layer_ids
    heads = config.swa_num_attention_heads if local else config.num_attention_heads
    kv_heads = config.swa_num_key_value_heads if local else config.num_key_value_heads
    dim = config.swa_head_dim if local else config.head_dim
    taps = config.sconv_kernel_size
    gen = torch.Generator().manual_seed(100 + layer + seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen) * scale

    conv = (kv_heads * dim, 1, taps) if local else (taps, kv_heads * dim)
    return torch_ref.AttentionWeights(
        q=rand(heads * dim, HIDDEN, scale=0.125),
        k=rand(kv_heads * dim, HIDDEN, scale=0.125),
        v=rand(kv_heads * dim, HIDDEN, scale=0.125),
        rel=rand(heads * config.d_rel, HIDDEN, scale=0.0625),
        out=rand(HIDDEN, heads * dim),
        out_gate=rand(heads * dim, HIDDEN),
        sink=rand(heads, HIDDEN),
        q_scale=1 + rand(heads, dim, scale=0.2),
        rel_scale=1 + rand(heads, config.d_rel, scale=0.2),
        rel_proj=rand(config.d_rel, config.rel_extent),
        k_conv=rand(*conv, scale=0.5),
        v_conv=rand(*conv, scale=0.5),
        k_conv_proj=None if local else rand(kv_heads * taps, HIDDEN, scale=0.0625),
        v_conv_proj=None if local else rand(kv_heads * taps, HIDDEN, scale=0.0625),
    )


def _shard(w, rank: int, tp: int):
    """The contiguous head slice of rank `rank` of `tp`."""
    import dataclasses

    def rows(t):
        return None if t is None else t.chunk(tp, dim=0)[rank]

    dynamic = w.k_conv_proj is not None
    return dataclasses.replace(
        w,
        **{
            name: rows(getattr(w, name))
            for name in (
                "q",
                "k",
                "v",
                "rel",
                "out_gate",
                "sink",
                "q_scale",
                "rel_scale",
                "k_conv_proj",
                "v_conv_proj",
            )
        },
        out=w.out.chunk(tp, dim=1)[rank],
        k_conv=w.k_conv.chunk(tp, dim=1 if dynamic else 0)[rank],
        v_conv=w.v_conv.chunk(tp, dim=1 if dynamic else 0)[rank],
    )


def _inputs(layer: int, scale: float = 1.0, seed: int = 0):
    """Per request the (q, k, v) inputs [L, HIDDEN]: one normed state three
    times in a sliding layer, three in a global one. Request 1 shares its
    first PREFIX tokens with request 0; `seed` and `scale` change the rest
    of request 1 only."""
    streams = 1 if layer == SLIDING else 3
    inputs = {}
    for seq, length in LENGTHS.items():
        gen = torch.Generator().manual_seed(7 + seq * (1 + seed))
        x = [torch.randn(length, HIDDEN, generator=gen) for _ in range(streams)]
        if seq == 1:
            x = [
                torch.cat([a[:PREFIX], b[PREFIX:] * scale])
                for a, b in zip(inputs[0], x)
            ]
        inputs[seq] = tuple(x * (3 // streams))
    return inputs


def _reference(layer: int, w, x, num_kv_heads: int):
    """Dense q [L, heads, D], rel [L, heads, extent], key and value
    [L, kv_heads, D] of one request."""
    from vllm_sm70.models.inkling2_sm70.ops import torch_ref

    config = tiny_text_config()
    x_q, x_k, x_v = (t[None] for t in x)
    positions = torch.arange(x_q.shape[1])[None]
    log_scaling = None
    if layer == GLOBAL:
        log_scaling = (config.log_scaling_alpha, config.log_scaling_n_floor)
    qk_clip, rel_clip = config.qk_l2_clip_threshold, config.rel_l2_clip_threshold
    q = torch_ref.query_projection(x_q, w.q, w.q_scale, qk_clip, positions, log_scaling)
    rel = torch_ref.query_projection(
        x_q, w.rel, w.rel_scale, rel_clip, positions, log_scaling
    )
    key = torch_ref.key_value_projection(
        x_k, w.k, w.k_conv, num_kv_heads, w.k_conv_proj, qk_clip
    )
    value = torch_ref.key_value_projection(
        x_v, w.v, w.v_conv, num_kv_heads, w.v_conv_proj, eps=config.rms_norm_eps
    )
    return q[0], (rel @ w.rel_proj)[0], key[0].transpose(0, 1), value[0].transpose(0, 1)


def _rows(x: torch.Tensor, rows, pad: int) -> torch.Tensor:
    """Batch rows of the per-request tensors `x` plus `pad` padding rows."""
    parts = [x[seq][start : start + count] for seq, start, count in rows]
    if pad:
        parts.append(torch.full((pad, parts[0].shape[1]), 1e3))
    return torch.cat(parts)


def _run_prologue(
    layer,
    steps,
    inputs,
    *,
    tp=1,
    rank=0,
    pad=0,
    pad_metadata=True,
    weights=None,
    whole=False,
):
    """Run the prologue over `steps`. Returns the visit and per request the
    q_out and rel_out rows by position. `whole` projects each request's whole
    input once, so a row's projection does not depend on its batch."""
    import torch.nn.functional as F

    from vllm_sm70.models.inkling2_sm70 import prologue
    from vllm_sm70.models.inkling2_sm70.weights import qkvr_sizes

    config = tiny_text_config()
    visit = PagedVisit(layer, tp=tp)
    w = _shard(weights or _weights(layer), rank, tp)
    sizes = qkvr_sizes(config, layer == SLIDING, tp)
    qkvr = torch.cat([w.q, w.rel, w.out_gate, w.sink, w.k, w.v])
    heads, dim = w.q_scale.shape
    out = {seq: ({}, {}) for seq in inputs}
    log_scaling = None
    if layer == GLOBAL:
        log_scaling = (config.log_scaling_alpha, config.log_scaling_n_floor)
    projected = {}
    if whole:
        for seq, (x_q, x_k, x_v) in inputs.items():
            taps_k = w.k if layer == SLIDING else w.k_conv_proj
            taps_v = w.v if layer == SLIDING else w.v_conv_proj
            projected[seq] = (
                x_q @ w.q.T, x_q @ w.rel.T, x_k @ w.k.T, x_v @ w.v.T,
                x_k @ taps_k.T, x_v @ taps_v.T,
            )  # fmt: skip
    for step in steps:
        if step[0] == "fork":
            visit.fork(*step[1:])
            continue
        metadata, rows = visit.batch(step[1], pad_tokens=pad, pad_metadata=pad_metadata)
        x_q, x_k, x_v = (
            _rows({s: x[i] for s, x in inputs.items()}, rows, pad) for i in range(3)
        )
        if whole:
            q, r, k, v, k_taps, v_taps = (
                _rows({s: p[i] for s, p in projected.items()}, rows, pad)
                for i in range(6)
            )
            if layer == SLIDING:
                k_taps = v_taps = None
        elif layer == SLIDING:
            q, r, _, _, k, v = prologue.project_qkvr(x_q, None, None, qkvr, sizes)
            k_taps = v_taps = None
        else:
            q, r, _, _, k, v = prologue.project_qkvr(x_q, x_k, x_v, qkvr, sizes)
            k_taps = F.linear(x_k, w.k_conv_proj)
            v_taps = F.linear(x_v, w.v_conv_proj)
        q_out = torch.full((x_q.shape[0], heads, dim), float("nan"))
        rel_out = torch.full((x_q.shape[0], heads, config.rel_extent), float("nan"))
        with visit.forward(metadata):
            prologue.attention_prologue(
                q, r, k, v, w.q_scale, w.rel_scale, w.rel_proj, w.k_conv, w.v_conv,
                k_taps, v_taps, visit.attn, visit.conv, q_out, rel_out,
                qk_clip=config.qk_l2_clip_threshold,
                rel_clip=config.rel_l2_clip_threshold,
                eps=config.rms_norm_eps,
                log_scaling=log_scaling,
            )  # fmt: skip
        if pad:
            assert not q_out[-pad:].any() and not rel_out[-pad:].any()
        row = 0
        for seq, start, count in rows:
            for i in range(count):
                out[seq][0][start + i] = q_out[row]
                out[seq][1][start + i] = rel_out[row]
                row += 1
    return visit, out


def _scheduled(steps) -> dict[int, int]:
    """First position each request computes itself (a fork starts later)."""
    first = {}
    for step in steps:
        if step[0] == "fork":
            first[step[2]] = step[3]
        else:
            for seq, _ in step[1]:
                first.setdefault(seq, 0)
    return first


def _check_prologue(layer, steps, *, tp=1, rank=0, pad=0, pad_metadata=True):
    from vllm_sm70.models.inkling2_sm70.cache import STREAM_K, STREAM_V

    inputs = _inputs(layer)
    weights = _weights(layer)
    visit, out = _run_prologue(
        layer,
        steps,
        inputs,
        tp=tp,
        rank=rank,
        pad=pad,
        pad_metadata=pad_metadata,
        weights=weights,
    )
    kv_heads = visit.attn.num_kv_heads * tp
    for seq, first in _scheduled(steps).items():
        length = visit.lengths[seq]
        x = tuple(t[:length] for t in inputs[seq])
        q, rel, key, value = (
            t.chunk(tp, dim=1)[rank] for t in _reference(layer, weights, x, kv_heads)
        )
        q_out, rel_out = out[seq]
        assert sorted(q_out) == list(range(first, length))
        got_q = torch.stack([q_out[p] for p in range(first, length)])
        got_rel = torch.stack([rel_out[p] for p in range(first, length)])
        torch.testing.assert_close(got_q, q[first:], rtol=0, atol=TOL)
        torch.testing.assert_close(got_rel, rel[first:], rtol=0, atol=TOL)
        # Every position's K and V, the cached prefix included.
        got_key, got_value = visit.stored_kv(seq)
        torch.testing.assert_close(got_key, key, rtol=0, atol=TOL)
        torch.testing.assert_close(got_value, value, rtol=0, atol=TOL)
        # The conv state holds the conv inputs: the K and V projections.
        w = _shard(weights, rank, tp)
        for stream, x_in, proj in ((STREAM_K, x[1], w.k), (STREAM_V, x[2], w.v)):
            torch.testing.assert_close(
                visit.stored_conv(seq, stream, first),
                (x_in @ proj.T)[first:],
                rtol=0,
                atol=TOL,
            )
    # Nothing was stored outside the scheduled slots.
    stored = sum(visit.lengths[seq] - first for seq, first in _scheduled(steps).items())
    kv = visit.attn.kv_cache
    assert int((~kv.isnan()).sum()) == stored * kv.shape[1] * kv.shape[3]


@LAYERS
@SCENARIO
def test_prologue_matches_dense_reference(layer, scenario):
    _check_prologue(layer, SCENARIOS[scenario])


@LAYERS
@pytest.mark.parametrize("pad_metadata", [True, False])
def test_prologue_skips_padding_tokens(layer, pad_metadata):
    _check_prologue(layer, SCENARIOS["interleaved"], pad=2, pad_metadata=pad_metadata)


@LAYERS
@pytest.mark.parametrize("rank", [0, 1])
def test_prologue_tp2_rank_is_a_head_slice(layer, rank):
    _check_prologue(layer, SCENARIOS["interleaved"], tp=2, rank=rank)
    _check_prologue(layer, SCENARIOS["prefix_resume"], tp=2, rank=rank)


@SCENARIO
@pytest.mark.parametrize("pad_metadata", [True, False])
@pytest.mark.parametrize("stream_name", ["STREAM_ATTN", "STREAM_MLP"])
def test_short_conv_matches_dense_reference(scenario, stream_name, pad_metadata):
    """The post-attention and post-MLP convs over the hidden-wide streams."""
    from vllm_sm70.models.inkling2_sm70 import cache, sconv
    from vllm_sm70.models.inkling2_sm70.ops import torch_ref

    stream = getattr(cache, stream_name)
    gen = torch.Generator().manual_seed(3 + stream)
    weight = torch.randn(HIDDEN, 1, 4, generator=gen) * 0.5
    inputs = {seq: x[0] for seq, x in _inputs(SLIDING, seed=stream).items()}
    steps = SCENARIOS[scenario]
    visit = PagedVisit(SLIDING)
    got = {seq: {} for seq in inputs}
    for step in steps:
        if step[0] == "fork":
            visit.fork(*step[1:])
            continue
        metadata, rows = visit.batch(step[1], pad_tokens=1, pad_metadata=pad_metadata)
        x = _rows(inputs, rows, 1)
        out = torch.full_like(x, float("nan"))
        with visit.forward(metadata):
            sconv.short_conv(x, weight, visit.conv, stream, out)
        row = 0
        for seq, start, count in rows:
            for i in range(count):
                got[seq][start + i] = out[row]
                row += 1
        # A row after the scheduled tokens is zero.
        assert not out[-1].any()
    for seq, first in _scheduled(steps).items():
        length = visit.lengths[seq]
        want = torch_ref.static_conv(inputs[seq][None, :length], weight)[0]
        have = torch.stack([got[seq][p] for p in range(first, length)])
        torch.testing.assert_close(have, want[first:], rtol=0, atol=TOL)
        torch.testing.assert_close(
            visit.stored_conv(seq, stream, first), inputs[seq][first:length]
        )
    # The other three streams of the conv state are untouched.
    offset, width = visit.conv.stream_ranges[stream]
    state = visit.conv.kv_cache
    assert state[..., :offset].isnan().all()
    assert state[..., offset + width :].isnan().all()


def test_taps_never_cross_a_sequence_boundary():
    """Each request's conv input encodes (request, position); every tap of
    every token must be the value `lag` positions back in its own request, or
    zero before position 0, whatever its neighbours in the batch are."""
    from vllm_sm70.models.inkling2_sm70 import cache, sconv

    visit = PagedVisit(SLIDING)
    conv = visit.conv
    stream = conv.stream_ranges[cache.STREAM_ATTN]
    channel = torch.arange(HIDDEN) / 1024

    def encode(seq, position):
        return 100.0 * (seq + 1) + position + channel

    steps = [
        [(0, 5)],
        [(0, 2), (1, 3)],  # request 1 starts right after rows of request 0
        [(1, 1), (0, 1), (2, 1)],
        [(2, 6), (1, 1), (0, 1)],
        [(0, 1), (1, 1), (2, 1)],
        [(1, 4), (2, 2), (0, 3), (3, 2)],
    ]
    checked = 0
    for parts in steps:
        metadata, rows = visit.batch(parts, pad_tokens=1)
        tokens = [(seq, start + i) for seq, start, count in rows for i in range(count)]
        x = torch.stack([encode(*t) for t in tokens] + [torch.full((HIDDEN,), 9e3)])
        md = metadata[conv.prefix]
        taps = sconv.conv_taps(x, 4, conv.kv_cache, md, stream)
        sconv.store_taps(x, conv.kv_cache, md, stream)
        for row, (seq, position) in enumerate(tokens):
            for lag in range(4):
                want = torch.zeros(HIDDEN)
                if position - lag >= 0:
                    want = encode(seq, position - lag)
                assert torch.equal(taps[lag][row], want), (seq, position, lag)
                checked += 1
        for lag in range(4):
            assert not taps[lag][-1].any()  # the padding row
    assert checked == 4 * sum(visit.lengths.values())


@LAYERS
def test_other_request_does_not_change_outputs(layer):
    """Request 0's outputs and stored K/V are bit-identical whether request 1
    shares its batches or not, and whatever request 1 holds."""
    steps = SCENARIOS["interleaved"]
    alone = [("batch", [part for part in step[1] if part[0] == 0]) for step in steps]
    runs = [
        _run_prologue(layer, steps, _inputs(layer), whole=True),
        _run_prologue(layer, steps, _inputs(layer, scale=1e4, seed=1), whole=True),
        _run_prologue(layer, alone, {0: _inputs(layer)[0]}, whole=True),
    ]
    visit, base = runs[0]
    for other, out in runs[1:]:
        for kind in (0, 1):
            for position, want in base[0][kind].items():
                assert torch.equal(out[0][kind][position], want)
        for want, got in zip(visit.stored_kv(0), other.stored_kv(0)):
            assert torch.equal(got, want)
    # Request 1's own outputs did change with its inputs.
    assert not torch.equal(runs[1][1][1][0][PREFIX], base[1][0][PREFIX])


@LAYERS
def test_forward_without_cache_metadata(layer):
    """A forward with no per-layer metadata (profile run) computes q and rel
    without tau and leaves the caches alone; the conv copies its input."""
    from types import SimpleNamespace

    from vllm.forward_context import override_forward_context

    from vllm_sm70.models.inkling2_sm70 import prologue, sconv
    from vllm_sm70.models.inkling2_sm70.cache import STREAM_ATTN

    config = tiny_text_config()
    visit = PagedVisit(layer)
    w = _weights(layer)
    heads, dim = w.q_scale.shape
    x = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(1))
    q, r, k, v = (x @ t.T for t in (w.q, w.rel, w.k, w.v))
    q_out = torch.empty(6, heads, dim)
    rel_out = torch.empty(6, heads, config.rel_extent)
    out = torch.empty_like(x)
    with override_forward_context(SimpleNamespace(attn_metadata=None)):
        prologue.attention_prologue(
            q, r, k, v, w.q_scale, w.rel_scale, w.rel_proj, w.k_conv, w.v_conv,
            None, None, visit.attn, visit.conv, q_out, rel_out,
            qk_clip=config.qk_l2_clip_threshold,
            rel_clip=config.rel_l2_clip_threshold,
            eps=config.rms_norm_eps,
            log_scaling=(0.1, 1.0) if layer == GLOBAL else None,
        )  # fmt: skip
        sconv.short_conv(x, torch.ones(HIDDEN, 1, 4), visit.conv, STREAM_ATTN, out)
    want_q = prologue.query_states(q, w.q_scale, config.qk_l2_clip_threshold)
    want_rel = prologue.query_states(r, w.rel_scale, config.rel_l2_clip_threshold)
    assert torch.equal(q_out, want_q)
    torch.testing.assert_close(rel_out, want_rel @ w.rel_proj)
    assert torch.equal(out, x)
    assert visit.attn.kv_cache.isnan().all()
    assert visit.conv.kv_cache.isnan().all()


@LAYERS
def test_fp16_caches_round_like_the_reference(layer):
    """fp16 inputs and caches: outputs are fp16 and the stored K/V equal the
    dense reference run in fp16 up to fp16 rounding of the fp32 sums."""
    import torch.nn.functional as F

    from vllm_sm70.models.inkling2_sm70 import prologue

    config = tiny_text_config()
    visit = PagedVisit(layer, dtype=torch.float16)
    assert visit.attn.kv_cache.dtype == visit.conv.kv_cache.dtype == torch.float16
    w = _weights(layer)
    heads, dim = w.q_scale.shape
    x = tuple(t.half() for t in _inputs(layer)[0])
    half = {
        name: getattr(w, name).half()
        for name in ("q", "k", "v", "rel", "q_scale", "rel_scale", "rel_proj")
    }
    log_scaling = k_taps = v_taps = None
    k_conv, v_conv = w.k_conv, w.v_conv
    if layer == GLOBAL:
        log_scaling = (config.log_scaling_alpha, config.log_scaling_n_floor)
        k_conv, v_conv = k_conv.half(), v_conv.half()
    for count in (11, 1, 1, 17):
        metadata, rows = visit.batch([(0, count)])
        x_q, x_k, x_v = (_rows({0: t}, rows, 0) for t in x)
        if layer == GLOBAL:
            k_taps = F.linear(x_k, w.k_conv_proj.half())
            v_taps = F.linear(x_v, w.v_conv_proj.half())
        q_out = torch.empty(count, heads, dim, dtype=torch.float16)
        rel_out = torch.empty(count, heads, config.rel_extent, dtype=torch.float16)
        with visit.forward(metadata):
            prologue.attention_prologue(
                F.linear(x_q, half["q"]), F.linear(x_q, half["rel"]),
                F.linear(x_k, half["k"]), F.linear(x_v, half["v"]),
                half["q_scale"], half["rel_scale"], half["rel_proj"],
                k_conv, v_conv, k_taps, v_taps, visit.attn, visit.conv,
                q_out, rel_out,
                qk_clip=config.qk_l2_clip_threshold,
                rel_clip=config.rel_l2_clip_threshold,
                eps=config.rms_norm_eps,
                log_scaling=log_scaling,
            )  # fmt: skip
        assert q_out.isfinite().all() and rel_out.isfinite().all()
    import dataclasses

    w16 = dataclasses.replace(
        w,
        **half,
        k_conv=k_conv,
        v_conv=v_conv,
        k_conv_proj=None if layer == SLIDING else w.k_conv_proj.half(),
        v_conv_proj=None if layer == SLIDING else w.v_conv_proj.half(),
    )
    _, _, key, value = _reference(layer, w16, x, visit.attn.num_kv_heads)
    got_key, got_value = visit.stored_kv(0)
    assert got_key.dtype == torch.float16
    torch.testing.assert_close(got_key, key, rtol=0, atol=2e-3)
    torch.testing.assert_close(got_value, value, rtol=0, atol=2e-3)


@pytest.mark.parametrize("dynamic", [False, True])
def test_non_finite_row_of_another_request_stays_out_of_the_taps(dynamic):
    """Request 1 starts in the batch row after request 0's row, which holds
    inf: its first token has no taps, and its output stays finite."""
    from vllm_sm70.models.inkling2_sm70 import cache, sconv

    visit = PagedVisit(SLIDING)
    conv = visit.conv
    stream = conv.stream_ranges[cache.STREAM_ATTN]
    metadata, _ = visit.batch([(0, 1), (1, 1)])
    md = metadata[conv.prefix]
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(2, HIDDEN, generator=gen)
    x[0, 3] = float("inf")
    if dynamic:
        heads = conv.kv_cache.shape[1]
        weight = torch.randn(4, HIDDEN, generator=gen)
        taps = torch.randn(2, heads * 4, generator=gen)
        y = sconv.dynamic_conv(x, weight, taps, conv.kv_cache, md, stream)
        want = (
            x[1]
            / 9**0.5
            * (
                1
                + weight[0]
                + taps[1].view(heads, 4)[:, 0].repeat_interleave(HIDDEN // heads)
            )
        )
    else:
        weight = torch.randn(HIDDEN, 4, generator=gen)
        y = sconv.static_conv(x, weight, conv.kv_cache, md, stream)
        want = x[1] * (1 + weight[:, 3]) / 5**0.5
    assert torch.isfinite(y[1]).all()
    torch.testing.assert_close(y[1], want, rtol=0, atol=TOL)


def test_ops_refuse_a_cache_with_another_block_size(monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import cache, sconv

    visit = PagedVisit(SLIDING)
    conv = visit.conv
    stream = conv.stream_ranges[cache.STREAM_ATTN]
    metadata, _ = visit.batch([(0, 3)])
    md = metadata[conv.prefix]
    md.block_size += 1
    x = torch.zeros(3, HIDDEN)
    with pytest.raises(ValueError, match="conv-state blocks hold"):
        sconv.static_conv(x, torch.zeros(HIDDEN, 4), conv.kv_cache, md, stream)

    batch = PagedVisit.batch

    def tampered(self, *args, **kwargs):
        metadata, rows = batch(self, *args, **kwargs)
        metadata[self.attn.prefix].block_size += 1
        return metadata, rows

    monkeypatch.setattr(PagedVisit, "batch", tampered)
    with pytest.raises(ValueError, match="K/V cache blocks hold"):
        _run_prologue(SLIDING, [("batch", [(0, 3)])], _inputs(SLIDING))
