# SPDX-License-Identifier: Apache-2.0
"""Cases shared by the Inkling 2 paged decode-attention tests (CPU
interpreter and GPU).

``Case`` holds one layer visit's cache, inputs and metadata; requests own
shuffled 32-token blocks and every slot no row may read holds NaN.
``both_routes`` runs it on the torch route (switch off) and on the kernel
and returns (kernel, torch). The shape constants are module globals, so a
test may monkeypatch them before building a case.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch

SWITCH = "VLLM_SM70_INKLING_TRITON_DECODE_ATTN"
BLOCK = 32
KV_HEADS = 2
HEADS = 6
HEAD_DIM = 24
NUM_BLOCKS = 48


def attention_module():
    from vllm_sm70.models.inkling2_sm70 import attention_torch

    return attention_torch


class Case:
    """One layer visit's cache, inputs and metadata. Requests own shuffled
    blocks; every slot no row may read holds NaN."""

    def __init__(
        self,
        dtype,
        window,
        extent,
        seq_lens,
        *,
        pad=0,
        decode_only=True,
        query_lens=None,
        seed=0,
    ):
        g = torch.Generator().manual_seed(seed)
        query_lens = query_lens or [1] * len(seq_lens)
        pool = list(range(1, NUM_BLOCKS))
        random.Random(seed).shuffle(pool)
        width = max(-(-n // BLOCK) for n in seq_lens) + 2
        table = torch.zeros(len(seq_lens), width, dtype=torch.int32)
        cache = torch.randn(NUM_BLOCKS, KV_HEADS, BLOCK, 2 * HEAD_DIM, generator=g)
        cache = cache.to(dtype)
        written = torch.zeros(NUM_BLOCKS, BLOCK, dtype=torch.bool)
        positions, seq_idx, slots = [], [], []
        for req, (n, q) in enumerate(zip(seq_lens, query_lens)):
            blocks = [pool.pop() for _ in range(-(-n // BLOCK))]
            table[req, : len(blocks)] = torch.tensor(blocks)
            # Keys before the window of the request's first query hold NaN.
            first = 0 if window is None else max(0, n - q - window + 1)
            for p in range(first, n):
                written[blocks[p // BLOCK], p % BLOCK] = True
            for p in range(n - q, n):
                positions.append(p)
                seq_idx.append(req)
                slots.append(blocks[p // BLOCK] * BLOCK + p % BLOCK)
        cache.masked_fill_(~written[:, None, :, None], float("nan"))
        rows = len(positions) + pad
        positions += [0] * pad
        seq_idx += [0] * pad
        slots += [-1] * pad
        qsl = [0]
        for q in query_lens:
            qsl.append(qsl[-1] + q)
        self.cache = cache
        self.query = torch.randn(rows, HEADS, HEAD_DIM, generator=g).to(dtype)
        self.rel = torch.randn(rows, HEADS, extent, generator=g).to(dtype)
        self.sinks = torch.randn(rows, HEADS, generator=g).to(dtype)
        self.md = SimpleNamespace(
            block_size=BLOCK,
            sliding_window=window,
            decode_only=decode_only,
            positions=torch.tensor(positions, dtype=torch.int64),
            seq_idx=torch.tensor(seq_idx, dtype=torch.int32),
            slot_mapping=torch.tensor(slots, dtype=torch.int64),
            block_table=table,
            num_actual_tokens=rows - pad,
            num_reqs=len(seq_lens),
            query_start_loc=torch.tensor(qsl, dtype=torch.int32),
            query_lens_cpu=torch.tensor(query_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        )

    def run(self, attention, **kwargs):
        out = torch.full_like(self.query, float("nan"))
        return attention.gather_attention(
            self.query, self.rel, self.sinks, self.cache, self.md, out, **kwargs
        )


def both_routes(case, monkeypatch):
    attention = attention_module()
    calls = []
    launch = attention.decode_attention

    def counted(*args, **kwargs):
        calls.append(1)
        return launch(*args, **kwargs)

    monkeypatch.setattr(attention, "decode_attention", counted)
    monkeypatch.setenv(SWITCH, "0")
    want = case.run(attention)
    assert calls == []
    monkeypatch.delenv(SWITCH)
    got = case.run(attention)
    assert calls == [1]
    return got, want


def rel_err(got, want):
    assert torch.equal(got.isnan(), want.isnan())
    assert not want.isnan().any()
    return float((got.float() - want.float()).abs().max() / want.float().abs().max())
