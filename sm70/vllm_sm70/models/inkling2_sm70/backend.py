# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the backends of the model-owned caches.

The model runs attention and the short convs itself, so neither backend has
an attention impl. Each one only names the builder that turns the runner's
per-batch ``CommonAttentionMetadata`` into the metadata the model's ops read:

    Inkling2AttentionBackend -> Inkling2AttentionMetadata  (K/V caches)
    Inkling2SconvBackend     -> Inkling2SconvMetadata      (conv-state caches)

The runner stores one metadata object per cache owner under the owner's
forward-context name; ``cache_metadata(owner)`` reads it back.

Token tensors cover the padded batch. A padding token has slot -1 and
zeros elsewhere. The builders sort decode requests first.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

__all__ = [
    "ATTENTION_BACKEND_NAME",
    "SCONV_BACKEND_NAME",
    "Inkling2AttentionBackend",
    "Inkling2AttentionMetadata",
    "Inkling2AttentionMetadataBuilder",
    "Inkling2SconvBackend",
    "Inkling2SconvMetadata",
    "Inkling2SconvMetadataBuilder",
    "cache_metadata",
]

ATTENTION_BACKEND_NAME = "INKLING2_SM70_ATTN"
SCONV_BACKEND_NAME = "INKLING2_SM70_SCONV"


@dataclass
class Inkling2SconvMetadata(AttentionMetadata):
    """Metadata of one conv-state cache group. R requests, P padded tokens."""

    num_actual_tokens: int  # tokens without padding
    block_size: int  # tokens per block of this group's block table
    block_table: torch.Tensor  # [R, max_blocks] int32 physical block per request
    slot_mapping: torch.Tensor  # [P] int64 cache slot of each token, -1 = skip
    seq_idx: torch.Tensor  # [P] int32 request (batch row) of each token
    query_start: torch.Tensor  # [P] int32 first batch token of that request
    positions: torch.Tensor  # [P] int64 absolute position in the sequence


@dataclass
class Inkling2AttentionMetadata(Inkling2SconvMetadata):
    """Metadata of one K/V cache group; requests are decode-first."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor  # [R + 1] int32 first batch token per request
    seq_lens: torch.Tensor  # [R] int32 context + query tokens per request
    sliding_window: int | None  # None for a full-attention group
    causal: bool | torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    # Host copies, [R] int64. seq_lens_cpu is an upper bound on decode rows.
    query_lens_cpu: np.ndarray
    seq_lens_cpu: np.ndarray


class _TokenIndex:
    """Per-token request index, query start and position of a batch, in
    buffers every builder on a device shares."""

    def __init__(self, device: torch.device, size: int) -> None:
        self.device = device
        self._allocate(size)

    def _allocate(self, size: int) -> None:
        device = self.device
        self.size = size
        self.arange = torch.arange(size, dtype=torch.int32, device=device)
        self.seq_idx = torch.zeros(size, dtype=torch.int32, device=device)
        self.query_start = torch.zeros(size, dtype=torch.int32, device=device)
        self.positions = torch.zeros(size, dtype=torch.int64, device=device)

    def fill(self, cm: CommonAttentionMetadata) -> dict[str, Any]:
        num_reqs = cm.num_reqs
        num_actual = int(cm.query_start_loc_cpu[num_reqs])
        num_padded = cm.slot_mapping.shape[0]
        if num_padded < num_actual:
            raise ValueError(
                f"vllm_sm70: slot mapping covers {num_padded} tokens, the "
                f"batch has {num_actual}."
            )
        if num_padded > self.size:
            self._allocate(num_padded)
        seq_idx = self.seq_idx[:num_padded]
        query_start = self.query_start[:num_padded]
        positions = self.positions[:num_padded]
        if num_reqs and num_padded:
            starts = cm.query_start_loc[: num_reqs + 1]
            ends = starts[1:].contiguous()
            token = self.arange[:num_padded]
            # Request of a token: the number of requests that end at or
            # before it. Zero-token requests are skipped.
            req = torch.searchsorted(ends, token, right=True).clamp_(max=num_reqs - 1)
            real = token < num_actual
            start = starts[req]
            context = cm.seq_lens[:num_reqs] - (ends - starts[:-1])
            seq_idx.copy_(torch.where(real, req, 0))
            query_start.copy_(torch.where(real, start, 0))
            positions.copy_(torch.where(real, context[req] + token - start, 0))
        if cm.positions is not None:
            positions = cm.positions[:num_padded]
        return {
            "num_actual_tokens": num_actual,
            "slot_mapping": cm.slot_mapping[:num_padded],
            "seq_idx": seq_idx,
            "query_start": query_start,
            "positions": positions,
        }


_TOKEN_INDEX: dict[torch.device, _TokenIndex] = {}


def _token_index(device: torch.device, vllm_config: Any) -> _TokenIndex:
    size = int(vllm_config.scheduler_config.max_num_batched_tokens)
    index = _TOKEN_INDEX.get(device)
    if index is None:
        index = _TOKEN_INDEX[device] = _TokenIndex(device, size)
    elif index.size < size:
        index._allocate(size)
    return index


class _Builder(AttentionMetadataBuilder):
    """Common part of both builders. The metadata of two groups with equal
    specs differs in the block table and slot mapping only."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = int(kv_cache_spec.block_size)
        self.token_index = _token_index(torch.device(device), vllm_config)
        self._init_reorder_batch_threshold(1)

    def update_block_table(
        self, metadata: Any, blk_table: torch.Tensor, slot_mapping: torch.Tensor
    ) -> Any:
        updated = copy.copy(metadata)
        updated.block_table = blk_table
        updated.slot_mapping = slot_mapping[: metadata.slot_mapping.shape[0]]
        return updated


class Inkling2SconvMetadataBuilder(_Builder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2SconvMetadata:
        cm = common_attn_metadata
        return Inkling2SconvMetadata(
            block_size=self.block_size,
            block_table=cm.block_table_tensor,
            **self.token_index.fill(cm),
        )


class Inkling2AttentionMetadataBuilder(_Builder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        window = getattr(kv_cache_spec, "sliding_window", None)
        self.sliding_window = None if window is None else int(window)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2AttentionMetadata:
        cm = common_attn_metadata
        num_reqs = cm.num_reqs
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cm, decode_threshold=self.reorder_batch_threshold or 1
            )
        )
        seq_lens_cpu = cm.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            seq_lens_cpu = cm.seq_lens.cpu()
        return Inkling2AttentionMetadata(
            block_size=self.block_size,
            block_table=cm.block_table_tensor,
            **self.token_index.fill(cm),
            num_reqs=num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            query_start_loc=cm.query_start_loc,
            seq_lens=cm.seq_lens,
            sliding_window=self.sliding_window,
            causal=cm.causal,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            query_lens_cpu=cm.query_start_loc_cpu[: num_reqs + 1]
            .diff()
            .numpy()
            .astype(np.int64),
            seq_lens_cpu=seq_lens_cpu[:num_reqs].numpy().astype(np.int64),
        )


class _OwnedBackend(AttentionBackend):
    """A backend with a metadata builder and no impl. The spec and the kernel
    block sizes are the owner's, unchanged."""

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "vllm_sm70: Inkling 2 cache backends have no attention impl; the "
            "model runs attention and the short convs itself."
        )


class Inkling2AttentionBackend(_OwnedBackend):
    @staticmethod
    def get_name() -> str:
        return ATTENTION_BACKEND_NAME

    @staticmethod
    def get_builder_cls() -> type[Inkling2AttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder


class Inkling2SconvBackend(_OwnedBackend):
    @staticmethod
    def get_name() -> str:
        return SCONV_BACKEND_NAME

    @staticmethod
    def get_builder_cls() -> type[Inkling2SconvMetadataBuilder]:
        return Inkling2SconvMetadataBuilder


def cache_metadata(owner: Any) -> Any:
    """The current forward's metadata of a cache owner
    (``Inkling2AttentionCache`` or ``Inkling2ConvState``), or None when the
    forward has no per-layer metadata or the owner's cache is not bound."""
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return attn_metadata.get(owner.prefix)
