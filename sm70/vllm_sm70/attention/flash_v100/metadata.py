# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention metadata and its builder (K06).

Rebuilt from fork cf9492a vllm/v1/attention/backends/flash_v100/metadata.py
(455 lines) for v0.33.0. The builder extends TritonAttentionMetadataBuilder,
so the metadata is a TritonAttentionMetadata (the impl can always fall back
to upstream's kernel) plus the decode/prefill split:

* reorder_batch_threshold = 1: decode rows (query length 1) come first.
* Decode rows run the paged decode kernels on [0, num_decode_tokens).
* Prefill rows are grouped by query length (the K05 prefill op takes one
  query length M per launch); each group carries device row and token
  indices built once per step and shared by every layer.

v0.33.0 changes handled here (sec. 5.4-5.7 of section B):
* No CommonAttentionMetadata.seq_lens_cpu: grouping uses
  query_start_loc_cpu (exact for prefill rows); kernels read the device
  seq_lens. compute_num_computed_tokens() is not needed: the paged kernels
  place a row's queries at the end of its seq_len.
* `causal` may be a tensor: recorded, and the impl falls back to Triton.
* No cascade metadata, no build_for_drafting: draft-decode metadata update
  is off (supports_draft_decode_metadata_update = False; K07 changes that).
* CUDA graphs: UNIFORM_BATCH (single-token decode batches without a
  speculative config, see get_cudagraph_support).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec


@dataclass(frozen=True)
class PrefillGroup:
    """Prefill rows that share one query length."""

    query_len: int
    rows: torch.Tensor  # int64 [G], device: row indices into the batch
    token_index: torch.Tensor  # int64 [G * query_len], device: query tokens
    num_rows: int


@dataclass
class FlashAttnV100Metadata(TritonAttentionMetadata):
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    prefill_groups: tuple[PrefillGroup, ...] = ()
    # Launch envelope for captured decode (graphs replay any seq_len).
    max_model_len: int = 0
    cudagraph_capture: bool = False


def group_prefill_rows(
    query_start_loc_cpu: torch.Tensor,
    first_row: int,
    num_reqs: int,
    device: torch.device,
) -> tuple[PrefillGroup, ...]:
    """Group rows [first_row, num_reqs) by query length (CPU planning)."""
    starts = query_start_loc_cpu[first_row : num_reqs + 1].tolist()
    by_len: dict[int, list[int]] = {}
    for offset, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        if b > a:  # padding rows have no query tokens
            by_len.setdefault(b - a, []).append(first_row + offset)
    groups = []
    all_starts = query_start_loc_cpu.tolist()
    for query_len in sorted(by_len):
        rows = by_len[query_len]
        token_index = torch.cat(
            [
                torch.arange(
                    all_starts[r], all_starts[r] + query_len, dtype=torch.int64
                )
                for r in rows
            ]
        )
        groups.append(
            PrefillGroup(
                query_len=query_len,
                rows=torch.tensor(rows, dtype=torch.int64).to(
                    device, non_blocking=True
                ),
                token_index=token_index.to(device, non_blocking=True),
                num_rows=len(rows),
            )
        )
    return tuple(groups)


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # K07 (spec decode) implements update_draft_decode_metadata.
    supports_draft_decode_metadata_update = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Triton's __init__ sets an instance value; keep ours.
        self.supports_draft_decode_metadata_update = False
        self._init_reorder_batch_threshold(1)
        self.max_model_len = int(vllm_config.model_config.max_model_len)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        # Verify batches (query length 1 + k) run the prefill path, whose
        # groups are planned on the host each step; until K07 makes that
        # capture-safe, graphs cover single-token decode only.
        if getattr(vllm_config, "speculative_config", None) is not None:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return cls._cudagraph_support

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttnV100Metadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cm = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cm, decode_threshold=self.reorder_batch_threshold or 1
        )
        groups: tuple[PrefillGroup, ...] = ()
        if num_prefills:
            groups = group_prefill_rows(
                cm.query_start_loc_cpu, num_decodes, cm.num_reqs, self.device
            )
        return FlashAttnV100Metadata(
            **{f.name: getattr(base, f.name) for f in fields(TritonAttentionMetadata)},
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill_groups=groups,
            max_model_len=self.max_model_len,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashAttnV100Metadata:
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        metadata.cudagraph_capture = True
        return metadata
