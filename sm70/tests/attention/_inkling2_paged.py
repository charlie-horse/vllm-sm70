# SPDX-License-Identifier: Apache-2.0
"""A fake paged K/V cache for the M12 attention CPU tests.

One cache tensor in the logical per-layer shape vLLM binds
([num_blocks, kv_heads, block_size, 2 * D], here a view of block-compact
LBNHC storage), a block table per request with shuffled physical blocks, and
`Inkling2AttentionMetadata` for a batch of (request, first position, tokens)
entries. Unwritten slots hold NaN, so a read of a key that was never written
or was poisoned shows in the output.
"""

from __future__ import annotations

import numpy as np
import torch


class FakePagedKV:
    def __init__(
        self,
        *,
        num_requests: int,
        max_len: int,
        kv_heads: int,
        head_dim: int,
        block_size: int,
        sliding_window: int | None,
        dtype=torch.float32,
        seed: int = 0,
    ) -> None:
        self.block_size = block_size
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        blocks_per_request = -(-max_len // block_size)
        num_blocks = 1 + num_requests * blocks_per_request
        storage = torch.full(
            (num_blocks, block_size, kv_heads, 2 * head_dim), float("nan"), dtype=dtype
        )
        self.kv_cache = storage.transpose(1, 2)
        order = torch.randperm(
            num_blocks - 1, generator=torch.Generator().manual_seed(seed)
        )
        # Block 0 is the null block; no request owns it.
        self.block_table = (
            (order + 1).view(num_requests, blocks_per_request).to(torch.int32)
        )

    def slots(self, request: int, positions: torch.Tensor) -> torch.Tensor:
        block = self.block_table[request, positions // self.block_size].long()
        return block * self.block_size + positions % self.block_size

    def write(
        self, request: int, first: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """key, value [tokens, kv_heads, D] of positions from `first` on."""
        from vllm_sm70.models.inkling2_sm70.attention_torch import kv_cache_views

        key_cache, value_cache = kv_cache_views(self.kv_cache, self.head_dim)
        slots = self.slots(request, first + torch.arange(key.shape[0]))
        key_cache.flatten(0, 1)[slots] = key.to(key_cache.dtype)
        value_cache.flatten(0, 1)[slots] = value.to(value_cache.dtype)

    def poison(self, request: int, stop: int) -> None:
        """NaN in every slot of positions below `stop`."""
        if stop > 0:
            slots = self.slots(request, torch.arange(stop))
            self.kv_cache.transpose(1, 2).flatten(0, 1)[slots] = float("nan")

    def metadata(
        self,
        batch: list[tuple[int, int, int]],
        pad_tokens: int = 0,
        pad_metadata: bool = True,
    ):
        """Metadata of a batch of (request, first position, tokens) entries,
        in batch order (decode-first is the caller's order). Batch row r uses
        the block table row of its request. The token tensors cover
        `pad_tokens` padding rows with `pad_metadata` only."""
        if not pad_metadata:
            pad_tokens = 0
        from vllm_sm70.models.inkling2_sm70.backend import Inkling2AttentionMetadata

        query_lens = [n for _, _, n in batch]
        seq_lens = [first + n for _, first, n in batch]
        starts = np.concatenate(([0], np.cumsum(query_lens))).astype(np.int64)
        slots, seq_idx, query_start, positions = [], [], [], []
        for row, (request, first, n) in enumerate(batch):
            pos = first + torch.arange(n)
            slots.append(self.slots(request, pos))
            positions.append(pos)
            seq_idx += [row] * n
            query_start += [int(starts[row])] * n
        pad = [0] * pad_tokens
        num_tokens = int(starts[-1])
        decodes = sum(1 for n in query_lens if n == 1)
        return Inkling2AttentionMetadata(
            num_actual_tokens=num_tokens,
            block_size=self.block_size,
            block_table=self.block_table[[request for request, _, _ in batch]],
            slot_mapping=torch.cat(
                [*slots, torch.full((pad_tokens,), -1, dtype=torch.int64)]
            ),
            seq_idx=torch.tensor(seq_idx + pad, dtype=torch.int32),
            query_start=torch.tensor(query_start + pad, dtype=torch.int32),
            positions=torch.cat(
                [*positions, torch.zeros(pad_tokens, dtype=torch.int64)]
            ),
            num_reqs=len(batch),
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            sliding_window=self.sliding_window,
            causal=True,
            num_decodes=decodes,
            num_prefills=len(batch) - decodes,
            num_decode_tokens=decodes,
            num_prefill_tokens=num_tokens - decodes,
            query_lens_cpu=np.asarray(query_lens, dtype=np.int64),
            seq_lens_cpu=np.asarray(seq_lens, dtype=np.int64),
        )
