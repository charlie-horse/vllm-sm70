# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 vocab-parallel top-k candidate sampler against the reference sampler.

Port of FORK tests/kernels/core/test_sm70_topk_candidates_sampler.py
(cf9492a, 230 lines). The reference is upstream's PyTorch-native path:
FP32 logits divided by the temperature, ``apply_top_k_top_p_pytorch`` and
``random_sample`` with FP32 exponential noise. Upstream v0.33 added
``use_fp64_gumbel`` (UP topk_topp_sampler.py:366); the kernel reproduces the
FP32 draw only, so every reference call pins ``use_fp64_gumbel=False`` (I16).
TP ranks are simulated by splitting the vocabulary into shards.

Added for the port: the Flint TP2 shape set (k=20, 75072-logit shards, a
padded last shard) and the R4 descending-capture regression (graphs captured
largest batch first on a poisoned pool, smallest replayed first).
"""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
topk_topp = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")

pytestmark = pytest.mark.gpu

VOCAB = 150144  # Flint vocabulary; TP2 shards are 75072 logits


def _metadata(rows, temperature, top_k, top_p, generators=None):
    return SimpleNamespace(
        all_random=True,
        all_greedy=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=torch.full((rows,), temperature, device="cuda"),
        top_k=torch.full((rows,), top_k, dtype=torch.int32, device="cuda"),
        top_p=(
            None
            if top_p is None
            else torch.full((rows,), top_p, dtype=torch.float32, device="cuda")
        ),
        generators=generators or {},
    )


def _gathered_wire(candidates, logits, tp_size):
    shard = logits.shape[1] // tp_size
    wires = [
        candidates.local_topk_candidates(
            logits[:, rank * shard : (rank + 1) * shard],
            vocab_start=rank * shard,
            num_valid=shard,
        )
        for rank in range(tp_size)
    ]
    assert all(wire is not None for wire in wires)
    return torch.cat(wires, dim=0)


def _reference(logits, metadata):
    scaled = logits.float().div_(metadata.temperature.unsqueeze(1))
    masked = topk_topp.apply_top_k_top_p_pytorch(scaled, metadata.top_k, metadata.top_p)
    probs = masked.softmax(dim=-1, dtype=torch.float32)
    # Pinned FP32 Gumbel/exponential noise (I16).
    return topk_topp.random_sample(probs, metadata.generators, False), masked


def _logits(rows, kind, seed, vocab=VOCAB):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, vocab, device="cuda", generator=gen) * 2.5
    if kind == "peaked":
        hot = torch.randint(0, vocab, (rows, 4), device="cuda", generator=gen)
        x.scatter_add_(1, hot, torch.full((rows, 4), 6.0, device="cuda"))
    elif kind == "ties":
        x = (x * 16).round() / 16
    return x.half()


def _offset():
    return torch.cuda.default_generators[torch.cuda.current_device()].get_offset()


def test_reference_noise_is_fp32():
    probs = torch.ones(1, 8)
    assert topk_topp.empty_exponential_noise_like(probs, False).dtype == torch.float32


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("cols", [75072, 37536, 1000, 131072, 150144])
@pytest.mark.parametrize("vocab_start", [0, 75072])
@torch.inference_mode()
def test_local_top32_matches_topk(sampler_gpu, rows, cols, vocab_start):
    _, candidates = sampler_gpu
    torch.manual_seed(cols + rows)
    full = (torch.randn(rows, cols + 3, device="cuda") * 3).half()
    # An odd row stride with two rows exercises the unaligned scalar loads.
    logits = full[:, :cols]
    num_valid = cols - 5
    wire = candidates.local_topk_candidates(
        logits, vocab_start=vocab_start, num_valid=num_valid
    )
    assert wire is not None and wire.shape == (rows, 64)
    expected = torch.topk(logits[:, :num_valid].float(), 32, dim=-1).values
    values = wire[:, :32]
    ids = wire[:, 32:].long() - vocab_start
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    assert bool(((ids >= 0) & (ids < num_valid)).all())
    torch.testing.assert_close(logits.float().gather(1, ids), values, rtol=0, atol=0)
    for row in range(rows):
        assert ids[row].unique().numel() == 32


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize(
    "temperature,top_k,top_p,kind",
    [
        (1.0, 20, 0.95, "flat"),
        (1.0, 20, 0.95, "peaked"),
        (0.6, 20, 0.8, "peaked"),
        (1.3, 5, None, "flat"),
        (1.0, 1, 0.95, "flat"),
        (1.0, 20, 0.95, "ties"),
    ],
)
@torch.inference_mode()
def test_same_token_as_reference(
    sampler_gpu, tp_size, rows, temperature, top_k, top_p, kind
):
    """Same Philox state -> same token and same generator offset."""
    _, candidates = sampler_gpu
    trials = 64
    matches = 0
    for trial in range(trials):
        logits = _logits(rows, kind, seed=1000 * trial + rows)
        metadata = _metadata(rows, temperature, top_k, top_p)
        wire = _gathered_wire(candidates, logits, tp_size)

        torch.manual_seed(trial)
        sampled = candidates.sample_topk_candidates(
            wire, metadata, tp_size=tp_size, vocab_size=VOCAB
        )
        offset = _offset()
        torch.manual_seed(trial)
        expected, masked = _reference(logits, metadata)
        assert _offset() == offset
        assert sampled is not None and sampled.dtype == torch.int32
        kept = masked.gather(1, sampled.long().unsqueeze(1))
        assert bool(torch.isfinite(kept).all()), "sampled a masked token"
        matches += int(torch.equal(sampled.long(), expected))
    # Mismatches need an FP32 rounding tie in a cumulative probability or a
    # race score; ties beyond the 12 spare candidates of a rank are rarer.
    assert matches >= trials - (2 if kind == "ties" else 1)


@torch.inference_mode()
def test_request_generator_matches_reference(sampler_gpu):
    _, candidates = sampler_gpu
    for trial in range(32):
        logits = _logits(1, "peaked", seed=trial)
        wire = _gathered_wire(candidates, logits, 2)
        mine = torch.Generator(device="cuda").manual_seed(trial)
        ref = torch.Generator(device="cuda").manual_seed(trial)
        default_offset = _offset()
        sampled = candidates.sample_topk_candidates(
            wire,
            _metadata(1, 1.0, 20, 0.95, {0: mine}),
            tp_size=2,
            vocab_size=VOCAB,
        )
        assert _offset() == default_offset
        expected, _ = _reference(logits, _metadata(1, 1.0, 20, 0.95, {0: ref}))
        assert torch.equal(sampled.long(), expected)
        assert mine.get_offset() == ref.get_offset()


@torch.inference_mode()
def test_distribution_chi_square(sampler_gpu):
    """Draw frequencies follow the reference's masked softmax."""
    _, candidates = sampler_gpu
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = (torch.randn(1, VOCAB, device="cuda", generator=gen) * 1.2).half()
    metadata = _metadata(1, 1.0, 20, 0.95)
    wire = _gathered_wire(candidates, logits, 2)
    _, masked = _reference(logits, metadata)
    keep = torch.isfinite(masked[0]).nonzero().flatten()
    probs = torch.softmax(masked[0, keep].double(), dim=0)
    draws = 6000
    torch.manual_seed(123)
    counts = torch.zeros(VOCAB, dtype=torch.float64, device="cuda")
    for _ in range(draws):
        token = candidates.sample_topk_candidates(
            wire, metadata, tp_size=2, vocab_size=VOCAB
        )
        counts[token.long()] += 1
    assert counts.sum().item() == draws
    assert counts[keep].sum().item() == draws, "sampled outside the kept set"
    expected = probs * draws
    big = expected >= 5
    observed = counts[keep][big]
    chi2 = (((observed - expected[big]) ** 2) / expected[big]).sum().item()
    dof = int(big.sum().item()) - 1
    assert dof >= 5
    assert chi2 < dof + 6 * math.sqrt(2 * dof) + 10, (chi2, dof)


@pytest.mark.parametrize("ties_on_rank", [0, 40])
@torch.inference_mode()
def test_tie_overflow_is_counted(sampler_gpu, ties_on_rank):
    """Rows whose k-th value ties past a rank's 32 candidates are counted."""
    device, candidates = sampler_gpu
    logits = _logits(1, "flat", seed=5).float()
    if ties_on_rank:
        logits[0, 100 : 100 + ties_on_rank] = 20.0
    logits = logits.half()
    metadata = _metadata(1, 1.0, 20, 0.95)
    before = candidates.overflow_count(device)
    sampled = candidates.sample_topk_candidates(
        _gathered_wire(candidates, logits, 2), metadata, tp_size=2, vocab_size=VOCAB
    )
    assert sampled is not None
    after = candidates.overflow_count(device)
    assert after - before == (1 if ties_on_rank else 0)
    if ties_on_rank:
        # The reference keeps all 40 ties; the sampled token is one of them.
        assert 100 <= int(sampled.item()) < 100 + ties_on_rank


# ----------------------------------------------------------- Flint TP2 shapes


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("org_vocab", [VOCAB, 150000])
@torch.inference_mode()
def test_flint_tp2_k20_shards(sampler_gpu, rows, org_vocab):
    """TP2 as ParallelLMHead shards it: padded vocab split in two equal
    shards, the last one holding fewer valid logits when the vocabulary is
    not a multiple of 64 (columns past num_valid are padding)."""
    _, candidates = sampler_gpu
    padded = -(-org_vocab // 64) * 64
    per_rank = padded // 2
    full = _logits(rows, "peaked", seed=rows + org_vocab, vocab=padded)
    full[:, org_vocab:] = 60000.0  # padding must never be selected
    metadata = _metadata(rows, 1.0, 20, 0.95)
    wires = []
    for rank in range(2):
        start = rank * per_rank
        num_valid = min(org_vocab - start, per_rank)
        wire = candidates.local_topk_candidates(
            full[:, start : start + per_rank], vocab_start=start, num_valid=num_valid
        )
        assert wire is not None
        wires.append(wire)
    wire = torch.cat(wires, dim=0)
    logits = full[:, :org_vocab]
    matches = 0
    for trial in range(16):
        torch.manual_seed(trial)
        sampled = candidates.sample_topk_candidates(
            wire, metadata, tp_size=2, vocab_size=org_vocab
        )
        offset = _offset()
        torch.manual_seed(trial)
        expected, _ = _reference(logits, metadata)
        assert _offset() == offset
        assert bool((sampled.long() < org_vocab).all())
        matches += int(torch.equal(sampled.long(), expected))
    assert matches >= 15  # as above: an FP32 rounding tie may flip one draw


# --------------------------------------------- R4: descending-capture pattern


def _dirty_graph_pool(device, pool, stream):
    """Fill a graph pool with garbage before the real captures (pattern of
    FORK tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py:735-758):
    tensors allocated in a throwaway capture on the pool are filled by one
    replay and freed, so later captures on the pool reuse poisoned memory."""
    poison = torch.cuda.CUDAGraph()
    with torch.cuda.graph(poison, pool=pool, stream=stream):
        blocks = [torch.empty(16 << 20, dtype=torch.int32, device=device)]
        blocks += [
            torch.empty(1 << 17, dtype=torch.int32, device=device) for _ in range(64)
        ]
        for block in blocks:
            block.fill_(0x7FFFFFFF)
    poison.replay()
    torch.cuda.synchronize(device)
    del blocks, block
    return poison


@torch.inference_mode()
def test_descending_capture_regression(sampler_gpu):
    """Black box, as C4's test_decode_sinks_descending_capture_regression.

    Captures TP2 candidate sampling at batches 8, 4, 2, 1 on a fresh stream
    and a shared, poisoned graph pool, with no eager sampler call first; then
    replays batch 1 before any larger graph. Each replay must equal an eager
    call from the same RNG state, and capture must not create the overflow
    counter (or any other per-device state) lazily: it exists from prepare()
    and keeps its storage.
    """
    device, candidates = sampler_gpu
    monitor = candidates._overflow_monitor(device)
    assert monitor is not None, "prepare() must create the counter"
    counter_ptr = monitor.counter.data_ptr()
    monitors_before = dict(candidates._OVERFLOW_MONITORS)

    rows = 8
    shard = VOCAB // 2
    logits = torch.empty(rows, VOCAB, dtype=torch.float16, device=device)
    temperature = torch.ones(rows, device=device)
    top_k = torch.full((rows,), 20, dtype=torch.int32, device=device)
    top_p = torch.full((rows,), 0.95, device=device)
    outputs = {}

    def run(batch):
        meta = SimpleNamespace(
            temperature=temperature[:batch],
            top_k=top_k[:batch],
            top_p=top_p[:batch],
            generators={},
        )
        wire = torch.cat(
            [
                candidates.local_topk_candidates(
                    logits[:batch, r * shard : (r + 1) * shard],
                    vocab_start=r * shard,
                    num_valid=shard,
                )
                for r in range(2)
            ],
            dim=0,
        )
        return candidates.sample_topk_candidates(
            wire, meta, tp_size=2, vocab_size=VOCAB
        )

    logits.copy_(_logits(rows, "peaked", seed=11))
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    with torch.cuda.stream(stream):
        poison = _dirty_graph_pool(device, pool, stream)
        for batch in (8, 4, 2, 1):
            graphs[batch] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[batch], pool=pool, stream=stream):
                outputs[batch] = run(batch)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    del poison

    assert candidates._OVERFLOW_MONITORS == monitors_before
    assert candidates._overflow_monitor(device).counter.data_ptr() == counter_ptr

    for step, batch in enumerate((1, 1, 2, 4, 8)):
        logits.copy_(_logits(rows, "peaked" if step % 2 else "flat", seed=20 + step))
        state = torch.cuda.get_rng_state(device)
        graphs[batch].replay()
        torch.cuda.synchronize(device)
        replayed = outputs[batch].clone()
        torch.cuda.set_rng_state(state, device)
        eager = run(batch)
        assert torch.equal(replayed, eager), (batch, replayed, eager)
        assert bool(((replayed >= 0) & (replayed < VOCAB)).all())
