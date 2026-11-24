# SPDX-License-Identifier: Apache-2.0
"""Fake paged caches of one Inkling 2 layer visit for the CPU tests.

``PagedVisit`` owns the K/V cache and the conv-state cache of one visit of
the tiny fixture's config, hands out physical blocks in a shuffled order and
builds each batch's metadata with the real builders (backend.py). Unwritten
cache slots and block 0 hold NaN, so a read of a slot nothing stored shows
in every comparison.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inkling2_owners import build

TINY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models"
    / "fixtures"
    / "inkling2_tiny"
    / "config.json"
)


def tiny_text_config():
    from vllm_sm70.models.inkling2_sm70.configs import Inkling2ModelConfig

    return Inkling2ModelConfig(**json.loads(TINY.read_text())["text_config"])


def _builder_config(block_size: int, tp: int):
    from vllm.config import CacheConfig

    cache_config = CacheConfig(block_size=block_size, enable_prefix_caching=True)
    cache_config.kv_cache_layout = "LBNHC"
    return SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_batched_tokens=64
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, tensor_parallel_size=tp
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        kv_transfer_config=None,
        speculative_config=None,
        max_in_flight_tokens=64,
    )


class _Pages:
    """One paged cache: its tensor, the builder and each request's blocks."""

    def __init__(self, owner, owner_config, builder_config, width, blocks, gen):
        spec = owner.get_kv_cache_spec(owner_config)
        self.owner = owner
        self.block_size = spec.block_size
        owner.kv_cache = torch.full(
            (blocks, owner.num_kv_heads, spec.block_size, width),
            float("nan"),
            dtype=spec.dtype,
        )
        self.builder = owner.get_attn_backend().get_builder_cls()(
            spec, [owner.prefix], builder_config, torch.device("cpu")
        )
        self.free = (torch.randperm(blocks - 1, generator=gen) + 1).tolist()
        self.table: dict[int, list[int]] = {}

    def grow(self, seq: int, length: int) -> None:
        table = self.table.setdefault(seq, [])
        while len(table) * self.block_size < length:
            table.append(self.free.pop())

    def slot(self, seq: int, position: int) -> int:
        block = self.table[seq][position // self.block_size]
        return block * self.block_size + position % self.block_size


class PagedVisit:
    """The caches of visit (`layer`, `pass_idx`) of the tiny config."""

    def __init__(
        self,
        layer: int,
        pass_idx: int = 0,
        *,
        tp: int = 1,
        attn_block: int = 8,
        blocks: int = 48,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.config = tiny_text_config()
        owners, owner_config = build(attn_block, tp, dtype, config=self.config)
        self.attn = owners.attn[layer, pass_idx]
        self.conv = owners.conv[layer, pass_idx]
        gen = torch.Generator().manual_seed(seed)
        builder_config = _builder_config(attn_block, tp)
        self.attn_pages = _Pages(
            self.attn, owner_config, builder_config, 2 * self.attn.head_dim, blocks, gen
        )
        self.conv_pages = _Pages(
            self.conv, owner_config, builder_config, self.conv.head_size, blocks, gen
        )
        self.lengths: dict[int, int] = {}

    def fork(self, source: int, target: int, prefix: int) -> None:
        """Start request `target` on a prefix-cache hit of `prefix` tokens of
        `source`. It shares the K/V blocks of the prefix and only the last
        conv block; earlier conv blocks are the null block."""
        for pages, keep in ((self.attn_pages, None), (self.conv_pages, 1)):
            if prefix % pages.block_size:
                raise ValueError("a cached prefix ends on a block boundary")
            shared = list(pages.table[source][: prefix // pages.block_size])
            if keep is not None:
                shared[:-keep] = [0] * (len(shared) - keep)
            pages.table[target] = shared
        self.lengths[target] = prefix

    def batch(
        self,
        parts: list[tuple[int, int]],
        pad_tokens: int = 0,
        pad_metadata: bool = True,
    ):
        """Schedule `count` new tokens of each (request, count). Returns the
        metadata per owner name and the (request, first position, count) of
        each batch row range. `pad_tokens` rows follow the batch; the slot
        mapping covers them (-1) with `pad_metadata`, as in a FULL graph, and
        ends at the scheduled tokens without it."""
        from vllm.v1.attention.backend import CommonAttentionMetadata

        rows = []
        for seq, count in parts:
            start = self.lengths.get(seq, 0)
            rows.append((seq, start, count))
            self.lengths[seq] = start + count
        starts = [0]
        for _, _, count in rows:
            starts.append(starts[-1] + count)
        qsl = torch.tensor(starts, dtype=torch.int32)
        seq_lens = torch.tensor([s + c for _, s, c in rows], dtype=torch.int32)
        metadata = {}
        seen_pad = pad_tokens if pad_metadata else 0
        for pages in (self.attn_pages, self.conv_pages):
            slots = []
            for seq, start, count in rows:
                pages.grow(seq, start + count)
                slots += [pages.slot(seq, start + i) for i in range(count)]
            width = max(len(pages.table[seq]) for seq, _, _ in rows)
            table = [
                pages.table[seq] + [0] * (width - len(pages.table[seq]))
                for seq, _, _ in rows
            ]
            common = CommonAttentionMetadata(
                query_start_loc=qsl.clone(),
                query_start_loc_cpu=qsl,
                seq_lens=seq_lens,
                num_reqs=len(rows),
                num_actual_tokens=starts[-1] + seen_pad,
                max_query_len=max(c for _, _, c in rows),
                max_seq_len=int(seq_lens.max()),
                block_table_tensor=torch.tensor(table, dtype=torch.int32),
                slot_mapping=torch.tensor(slots + [-1] * seen_pad, dtype=torch.int64),
                seq_lens_cpu_upper_bound=seq_lens.clone(),
            )
            metadata[pages.owner.prefix] = pages.builder.build(0, common)
        return metadata, rows

    @contextlib.contextmanager
    def forward(self, metadata):
        """The forward context the ops read their metadata from."""
        from vllm.forward_context import override_forward_context

        with override_forward_context(SimpleNamespace(attn_metadata=metadata)):
            yield

    def stored_kv(self, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values [L, kv_heads, D] the K/V cache holds for `seq`."""
        pages, cache = self.attn_pages, self.attn.kv_cache
        slots = torch.tensor(
            [pages.slot(seq, p) for p in range(self.lengths[seq])], dtype=torch.int64
        )
        rows = cache[slots // pages.block_size, :, slots % pages.block_size]
        return rows[..., : self.attn.head_dim], rows[..., self.attn.head_dim :]

    def stored_conv(self, seq: int, stream: int, first: int = 0) -> torch.Tensor:
        """Conv inputs [L - first, C] the conv state holds for `seq`."""
        pages, cache = self.conv_pages, self.conv.kv_cache
        offset, width = self.conv.stream_ranges[stream]
        slots = torch.tensor(
            [pages.slot(seq, p) for p in range(first, self.lengths[seq])],
            dtype=torch.int64,
        )
        rows = cache[slots // pages.block_size, :, slots % pages.block_size]
        return rows[..., offset : offset + width].flatten(1)
