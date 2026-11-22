# SPDX-License-Identifier: Apache-2.0
"""Cases shared by the Inkling 2 paged short-conv kernel tests (CPU
interpreter and GPU).

``Paged`` holds the conv-state cache and the K/V cache of one layer visit
with block size 32, hands out shuffled physical blocks and builds the
metadata of a batch of (request, new tokens). Unwritten slots and the null
block 0 hold NaN, so a read of a slot nothing stored shows. ``run`` drives
one scenario through the ops of sconv.py and prologue.py on the route the
switch selects and returns every output and the final caches; the tests
compare the two routes.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

SWITCH = "VLLM_SM70_INKLING_TRITON_SCONV"
BLOCK = 32
TAPS = 4
KV_HEADS = 2
HEAD_DIM = 8
HIDDEN = 48  # 24 channels per KV head in the output streams: not a power of 2
Q_HEADS = 4
D_REL = 4
EXTENT = 8
QK_CLIP = 2.0
REL_CLIP = 1.0
EPS = 1e-6

# A step is a batch [(request, new tokens), ...]; ("pad", rows) pads the
# next batch. Each batch lists decodes first, as the builders sort them.
SCENARIOS = {
    "prefill": [[(0, 40)], [(1, 3), (2, 33)]],
    "decode": [[(0, 1), (1, 1), (2, 1)]] * 6,
    "boundary": [[(0, 29)]] + [[(0, 1)]] * 5 + [[(0, 31)], [(0, 1)], [(0, 2)]],
    "short": [[(0, 2)], [(0, 1)], [(1, 1)], [(1, 1)], [(0, 1), (1, 1)]],
    "mixed": [
        [(0, 30), (1, 2)],
        [(0, 1), (1, 1), (2, 35)],
        [(0, 1), (1, 1), (2, 1), (3, 3)],
        [(2, 1), (0, 1), (3, 1), (1, 30)],
        [(1, 1), (0, 1), (2, 1), (3, 1)],
    ],
}


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest error relative to the largest reference value; NaN must sit
    at the same places."""
    assert got.shape == want.shape, (got.shape, want.shape)
    got, want = got.float().cpu(), want.float().cpu()
    nan = want.isnan()
    assert torch.equal(got.isnan(), nan)
    got, want = got[~nan], want[~nan]
    if not want.numel():
        return 0.0
    scale = max(float(want.abs().max()), 1e-6)
    return float((got - want).abs().max() / scale)


class Paged:
    """The caches of one visit and the block tables of its requests.

    `lbnhc` stores the conv cache with tokens before heads and binds the
    logical (blocks, heads, tokens, width) view, as the runner does for that
    layout."""

    def __init__(
        self, device, dtype: torch.dtype, blocks: int = 48, *, lbnhc: bool = False
    ):
        self.device = torch.device(device)
        self.head_size = 64  # next power of 2 of 2 * HEAD_DIM + 2 * 24
        if lbnhc:
            storage = torch.full(
                (blocks, BLOCK, KV_HEADS, self.head_size), float("nan"), dtype=dtype
            )
            self.conv_cache = storage.to(self.device).transpose(1, 2)
        else:
            self.conv_cache = torch.full(
                (blocks, KV_HEADS, BLOCK, self.head_size), float("nan"), dtype=dtype
            ).to(self.device)
        self.kv_cache = torch.full(
            (blocks, KV_HEADS, BLOCK, 2 * HEAD_DIM), float("nan"), dtype=dtype
        ).to(self.device)
        gen = torch.Generator().manual_seed(11)
        self.free = {
            "conv": (torch.randperm(blocks - 1, generator=gen) + 1).tolist(),
            "attn": (torch.randperm(blocks - 1, generator=gen) + 1).tolist(),
        }
        self.tables: dict[str, dict[int, list[int]]] = {"conv": {}, "attn": {}}
        self.lengths: dict[int, int] = {}
        self.stream_ranges = (
            (0, HEAD_DIM),
            (HEAD_DIM, HEAD_DIM),
            (2 * HEAD_DIM, 24),
            (2 * HEAD_DIM + 24, 24),
        )
        self.conv = SimpleNamespace(
            prefix="conv", kv_cache=self.conv_cache, stream_ranges=self.stream_ranges
        )
        self.attn = SimpleNamespace(prefix="attn", kv_cache=self.kv_cache)

    def _slot(self, kind: str, seq: int, position: int) -> int:
        table = self.tables[kind].setdefault(seq, [])
        while len(table) * BLOCK <= position:
            table.append(self.free[kind].pop())
        return table[position // BLOCK] * BLOCK + position % BLOCK

    def batch(self, parts, pad: int = 0, pad_metadata: bool = True, hole=None):
        """Metadata of one batch per owner name. `pad` rows follow the
        scheduled ones; the token tensors cover them (slot -1) with
        `pad_metadata`. `hole` is a scheduled row whose slots are -1."""
        rows = []
        for seq, count in parts:
            start = self.lengths.get(seq, 0)
            rows.append((seq, start, count))
            self.lengths[seq] = start + count
        num_tokens = sum(count for _, _, count in rows)
        covered = num_tokens + (pad if pad_metadata else 0)
        metadata = {}
        for kind in ("conv", "attn"):
            positions, seq_idx, query_start, slots = [], [], [], []
            first = 0
            for index, (seq, start, count) in enumerate(rows):
                for i in range(count):
                    positions.append(start + i)
                    seq_idx.append(index)
                    query_start.append(first)
                    slots.append(self._slot(kind, seq, start + i))
                first += count
            if hole is not None:
                slots[hole] = -1
            extra = covered - num_tokens
            width = max(len(self.tables[kind][seq]) for seq, _, _ in rows)
            table = [
                self.tables[kind][seq] + [0] * (width - len(self.tables[kind][seq]))
                for seq, _, _ in rows
            ]

            def tensor(values, dtype):
                return torch.tensor(values, dtype=dtype, device=self.device)

            metadata[kind] = SimpleNamespace(
                num_actual_tokens=num_tokens,
                block_size=BLOCK,
                block_table=tensor(table, torch.int32),
                slot_mapping=tensor(slots + [-1] * extra, torch.int64),
                seq_idx=tensor(seq_idx + [0] * extra, torch.int32),
                query_start=tensor(query_start + [0] * extra, torch.int32),
                positions=tensor(positions + [0] * extra, torch.int64),
            )
        return metadata, num_tokens

    @contextlib.contextmanager
    def forward(self, metadata):
        from vllm.forward_context import override_forward_context

        with override_forward_context(SimpleNamespace(attn_metadata=metadata)):
            yield


class Weights:
    """Seeded random weights and inputs of one visit in `dtype`."""

    def __init__(self, device, dtype: torch.dtype, dynamic: bool, seed: int = 0):
        self.device, self.dtype, self.dynamic = torch.device(device), dtype, dynamic
        self.gen = torch.Generator().manual_seed(seed)
        kv = KV_HEADS * HEAD_DIM
        conv = (TAPS, kv) if dynamic else (kv, 1, TAPS)
        self.k_conv = self.randn(*conv, scale=0.5)
        self.v_conv = self.randn(*conv, scale=0.5)
        self.out_conv = {
            False: self.randn(HIDDEN, 1, TAPS, scale=0.5),
            True: self.randn(TAPS, HIDDEN, scale=0.5),
        }
        self.q_scale = 1 + self.randn(Q_HEADS, HEAD_DIM, scale=0.2)
        self.rel_scale = 1 + self.randn(Q_HEADS, D_REL, scale=0.2)
        self.rel_proj = self.randn(D_REL, EXTENT)

    def randn(self, *shape, scale=1.0):
        x = torch.randn(shape, generator=self.gen) * scale
        return x.to(self.device, self.dtype)


def run(
    weights: Weights,
    steps,
    *,
    pad: int = 0,
    pad_metadata: bool = True,
    hole: bool = False,
    lbnhc: bool = False,
    paged: Paged | None = None,
):
    """Every output of `steps` on the route the switch selects, then the
    final conv-state and K/V caches. A batch runs the prologue, the static
    output conv of stream 2 and the dynamic one of stream 3."""
    from vllm_sm70.models.inkling2_sm70 import prologue, sconv

    paged = paged or Paged(weights.device, weights.dtype, lbnhc=lbnhc)
    outputs = []
    log_scaling = (0.1, 16.0) if weights.dynamic else None
    for step in steps:
        metadata, n = paged.batch(
            step, pad, pad_metadata, hole=(n_hole(step) if hole else None)
        )
        total = n + pad
        q = weights.randn(total, Q_HEADS * HEAD_DIM)
        r = weights.randn(total, Q_HEADS * D_REL)
        k = weights.randn(total, KV_HEADS * HEAD_DIM)
        v = weights.randn(total, KV_HEADS * HEAD_DIM, scale=3.0)
        taps = None
        if weights.dynamic:
            taps = (
                weights.randn(total, KV_HEADS * TAPS, scale=0.3),
                weights.randn(total, KV_HEADS * TAPS, scale=0.3),
            )
        q_out = torch.full(
            (total, Q_HEADS, HEAD_DIM), float("nan"), dtype=weights.dtype
        ).to(weights.device)
        rel_out = torch.full(
            (total, Q_HEADS, EXTENT), float("nan"), dtype=weights.dtype
        ).to(weights.device)
        x_attn = weights.randn(total, HIDDEN)
        x_mlp = weights.randn(total, HIDDEN)
        dyn = weights.randn(total, KV_HEADS, TAPS, scale=0.3)
        attn_out = torch.full_like(x_attn, float("nan"))
        mlp_out = torch.full_like(x_mlp, float("nan"))
        with paged.forward(metadata):
            prologue.attention_prologue(
                q, r, k, v, weights.q_scale, weights.rel_scale, weights.rel_proj,
                weights.k_conv, weights.v_conv,
                None if taps is None else taps[0], None if taps is None else taps[1],
                paged.attn, paged.conv, q_out, rel_out,
                qk_clip=QK_CLIP, rel_clip=REL_CLIP, eps=EPS, log_scaling=log_scaling,
            )  # fmt: skip
            sconv.short_conv(x_attn, weights.out_conv[False], paged.conv, 2, attn_out)
            sconv.short_conv(
                x_mlp, weights.out_conv[True], paged.conv, 3, mlp_out, dynamic_taps=dyn
            )
        outputs += [q_out, rel_out, attn_out, mlp_out]
    return outputs + [paged.conv_cache, paged.kv_cache]


def n_hole(step) -> int:
    """A scheduled row in the middle of the batch."""
    return sum(count for _, count in step) // 2
