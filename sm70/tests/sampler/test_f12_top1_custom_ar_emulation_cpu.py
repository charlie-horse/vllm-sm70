# SPDX-License-Identifier: Apache-2.0
"""F12 custom-AR exchange: CPU emulation of the kernel's per-rank data flow.

Each emulated rank builds its pair with the real ``sample_greedy.local_pair``
(padding mask, shard offset, FP32 packing) from a shard of one full-vocabulary
logit matrix, stores it into its own staging buffer, and reduces every rank's
staging rows as ``staged_cross_device_top1_argmax`` does (ranks in order,
``llrintf`` id decode, NaN first, larger value, lowest id). Every rank must
return the full-vocabulary argmax, with global ids.

Staging visibility follows the kernel source: a peer sees a row's fresh pair
only if the thread that stored it executes a system fence before the start
barrier publishes the flag (``membar`` orders the executing thread's own
stores). Rows without one read the buffer's previous bytes, which are
poisoned here with a pair that wins and names a rank-0 id.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_sm70 import sample_greedy

CU = Path(__file__).resolve().parents[2] / "csrc" / "custom_ar" / "top1_custom_ar.cu"
SHARD = 75072  # Flint TP2 vocab shard
STAGING_FLOATS = 2 * sample_greedy.MAX_CUSTOM_AR_ROWS
# Stale staging bytes: a huge value naming a rank-0 id (fp16 hidden states
# read as FP32 pairs look like this: the high half sets a large exponent).
POISON = (1.0e30, 11749.0)


def _fenced_rows(src: str, ngpus: int, rows: int) -> set[int]:
    """Rows whose staging store is fenced by its own thread before the start
    barrier: all rows if the store block runs membar_sys(), else only the
    threads that run the barrier's fence (threadIdx.x < ngpus)."""
    store = re.search(
        r"if \(row < rows\) \{\s*own\[2 \* row\] = pair\[2 \* row\];"
        r"\s*own\[2 \* row \+ 1\] = pair\[2 \* row \+ 1\];(?P<tail>.*?)\}",
        src,
        re.S,
    )
    assert store, "staging store block not found in top1_custom_ar.cu"
    if "membar_sys();" in store.group("tail"):
        return set(range(rows))
    return set(range(min(ngpus, rows)))


def _better(v, i, bv, bi):
    vn, bn = math.isnan(v), math.isnan(bv)
    return (vn and (not bn or i < bi)) or (
        not vn and not bn and (v > bv or (v == bv and i < bi))
    )


def _emulate(pairs, fenced):
    """pairs: per rank [rows, 2] FP32. Returns per rank int64 [rows]."""
    ngpus, rows = len(pairs), pairs[0].shape[0]
    staging = []
    for pair in pairs:
        buf = torch.tensor(POISON * (STAGING_FLOATS // 2), dtype=torch.float32)
        buf[: 2 * rows] = pair.reshape(-1)
        staging.append(buf)
    stale = torch.tensor(POISON * (STAGING_FLOATS // 2), dtype=torch.float32)
    outs = []
    for rank in range(ngpus):
        out = []
        for row in range(rows):
            bv, bi = -math.inf, 2**63 - 1
            for i in range(ngpus):
                # A rank reads its own rows fresh; a peer's unfenced row stale.
                src = staging[i] if (i == rank or row in fenced) else stale
                v = src[2 * row].item()
                idx = round(src[2 * row + 1].item())  # llrintf: half to even
                if _better(v, idx, bv, bi):
                    bv, bi = v, idx
            out.append(bi)
        outs.append(torch.tensor(out, dtype=torch.int64))
    return outs


def _rank_pairs(logits, ngpus, shard, *, offset=True):
    """Real local_pair per rank on its padded shard of `logits` [rows, V]."""
    rows, vocab = logits.shape
    pairs = []
    lp = SimpleNamespace(soft_cap=None, scale=1.0)
    for rank in range(ngpus):
        start = rank * shard
        part = logits[:, start : start + shard]
        pad = shard - part.shape[1]
        if pad:
            part = torch.cat([part, torch.full((rows, pad), 64.0).half()], dim=-1)
        lp._apply_head = lambda head, h, bias, part=part: part.clone()
        head = SimpleNamespace(
            shard_indices=SimpleNamespace(
                num_org_vocab_padding=pad,
                org_vocab_start_index=start if offset else 0,
            )
        )
        pairs.append(sample_greedy.local_pair(lp, head, torch.empty(rows, 1)))
    return pairs


def _logits(rows, vocab, winners, seed=0):
    """FP16 [rows, vocab] in [-4, 4) with a clear maximum at winners[r]."""
    g = torch.Generator().manual_seed(seed)
    logits = (torch.rand(rows, vocab, generator=g) * 8 - 4).half()
    for r, w in enumerate(winners):
        logits[r, w] = 30.0
    return logits


def _want(logits):
    return logits.float().argmax(dim=-1)


# Row 2 onwards is written by threads that do not run the start barrier's
# fence; the winners sit on rank 1 at local ids that are valid on rank 0.
WINNERS = [3426, SHARD + 11749, SHARD + 11749, 7, SHARD + 3426, 2 * SHARD - 1]


def test_kernel_source_fences_every_staging_store():
    src = CU.read_text()
    for ngpus in sample_greedy.CUSTOM_AR_WORLD_SIZES:
        assert _fenced_rows(src, ngpus, 128) == set(range(128))


@pytest.mark.parametrize("rows", [1, 2, 3, 6, 128])
def test_every_rank_returns_the_global_argmax(rows):
    winners = (WINNERS * 22)[:rows]
    logits = _logits(rows, 2 * SHARD, winners)
    pairs = _rank_pairs(logits, 2, SHARD)
    # The pair carries global ids: rank 1's winner is past rank 0's shard.
    for r, w in enumerate(winners):
        if w >= SHARD:
            assert pairs[1][r, 1].item() == w
    fenced = _fenced_rows(CU.read_text(), 2, rows)
    want = _want(logits)
    for out in _emulate(pairs, fenced):
        assert torch.equal(out, want), (out, want)


@pytest.mark.parametrize("ngpus", [2, 4, 8])
def test_padded_last_shard_and_ties(ngpus):
    shard = 1024
    vocab = ngpus * shard - 5  # last shard padded; padding poisoned to 64
    rows = 8
    g = torch.Generator().manual_seed(ngpus)
    logits = torch.randint(-3, 3, (rows, vocab), generator=g).half()
    logits[0, :] = 2.0  # all tied: lowest id
    logits[1, shard + 5] = 9.0
    logits[1, 5] = 9.0  # tie across ranks: rank 0's id
    logits[2, vocab - 1] = 9.0  # last real id of the padded shard
    pairs = _rank_pairs(logits, ngpus, shard)
    fenced = _fenced_rows(CU.read_text(), ngpus, rows)
    want = _want(logits)
    for out in _emulate(pairs, fenced):
        assert torch.equal(out, want)


# The emulation catches each way the exchange can return a local or stale id.


def test_unfenced_rows_would_read_stale_staging():
    rows = 3
    logits = _logits(rows, 2 * SHARD, WINNERS[:rows])
    pairs = _rank_pairs(logits, 2, SHARD)
    old_kernel_fence = set(range(min(2, rows)))  # barrier threads only
    outs = _emulate(pairs, old_kernel_fence)
    assert outs[0][2].item() == 11749 != _want(logits)[2].item()


def test_missing_shard_offset_would_return_a_local_id():
    rows = 3
    logits = _logits(rows, 2 * SHARD, WINNERS[:rows])
    pairs = _rank_pairs(logits, 2, SHARD, offset=False)
    out = _emulate(pairs, set(range(rows)))[0]
    assert out[1].item() == 11749 != _want(logits)[1].item()
