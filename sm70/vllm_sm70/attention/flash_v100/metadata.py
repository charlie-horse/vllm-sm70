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
  stays off (supports_draft_decode_metadata_update = False, see K07 below).
* CUDA graphs: UNIFORM_BATCH. Everything the captured forward reads is a
  persistent buffer refreshed in build(): the runner's block table, a
  builder-owned copy of seq_lens, the K07 verify expansion, and arange
  slices for a uniform prefill group.

K07 (speculative decoding, spec/): with a speculative config the reorder
threshold is 1 + num_speculative_tokens, and causal decode-region rows
(verify rows) are expanded token-wise into decode rows (spec/verify.py) so
the D256 XQA / scalar decode verifies them. Non-causal draft batches
(DFlash) take the paged prefill path. VLLM_SM70_FLASH_V100_SPEC=0 restores
K06: reorder threshold 1, verify rows as prefill groups, and graphs for
single-token decode only.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from vllm_sm70.attention.flash_v100.spec import spec_verify_enabled
from vllm_sm70.attention.flash_v100.spec.verify import VerifyExpansion

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
    # Decode route inputs: `decode_rows` rows, one query token each.
    decode_rows: int = 0
    decode_seq_lens: torch.Tensor | None = None
    decode_block_table: torch.Tensor | None = None
    # Builder-owned (persistent) copy of seq_lens for the prefill route.
    fv_seq_lens: torch.Tensor | None = None
    # K07: decode rows are the token-wise expansion of verify rows.
    spec_verify: bool = False
    # B2: every decode request has exactly verify_query_len tokens (so decode
    # row = request * verify_query_len + token); 0 when not uniform.
    verify_query_len: int = 0


def group_prefill_rows(
    query_start_loc_cpu: torch.Tensor,
    first_row: int,
    num_reqs: int,
    device: torch.device,
    persistent_arange: torch.Tensor | None = None,
) -> tuple[PrefillGroup, ...]:
    """Group rows [first_row, num_reqs) by query length (CPU planning).

    When every row has the same length (a uniform batch, e.g. a DFlash draft
    or a capturable verify batch), the indices are slices of
    `persistent_arange`, so a captured forward keeps valid addresses.
    """
    starts = query_start_loc_cpu[first_row : num_reqs + 1].tolist()
    by_len: dict[int, list[int]] = {}
    for offset, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        if b > a:  # padding rows have no query tokens
            by_len.setdefault(b - a, []).append(first_row + offset)
    if (
        persistent_arange is not None
        and len(by_len) == 1
        and len(next(iter(by_len.values()))) == num_reqs - first_row
        and starts[-1] <= persistent_arange.numel()
        and num_reqs <= persistent_arange.numel()
    ):
        ((query_len, rows),) = by_len.items()
        return (
            PrefillGroup(
                query_len=query_len,
                rows=persistent_arange[first_row:num_reqs],
                token_index=persistent_arange[starts[0] : starts[-1]],
                num_rows=len(rows),
            ),
        )
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
    # Off on purpose (K07): a drafter recording update_draft_decode_metadata
    # inside its graph would need the seq_lens copy and the verify expansion
    # to run in the captured region against the drafter's buffers; build()
    # does them eagerly instead (see the module docstring).
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
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.spec_query_len = max_decode_query_len(vllm_config)
        self.spec_verify = self.spec_query_len > 1 and spec_verify_enabled()
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=self.spec_verify)
        sched = vllm_config.scheduler_config
        max_num_seqs = max(int(getattr(sched, "max_num_seqs", 1) or 1), 1)
        max_tokens = int(getattr(sched, "max_num_batched_tokens", 0) or 0)
        self._seq_lens_buf = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        self._arange = torch.arange(
            max(max_tokens, max_num_seqs * self.spec_query_len, max_num_seqs),
            dtype=torch.int64,
            device=device,
        )
        self.verify = (
            VerifyExpansion(max_num_seqs * self.spec_query_len, device)
            if self.spec_verify
            else None
        )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        # Verify batches are capture-safe through the K07 expansion; with the
        # route off they run host-planned prefill groups, so graphs then
        # cover single-token decode only.
        if max_decode_query_len(vllm_config) > 1 and not spec_verify_enabled():
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
        causal = cm.causal
        # Token-wise expansion is exact for causal rows only; a non-causal
        # draft (DFlash) keeps multi-token rows on the paged prefill.
        expand = self.verify is not None and causal is True
        threshold = (self.reorder_batch_threshold or 1) if expand else 1
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cm, decode_threshold=threshold
        )
        num_reqs = cm.num_reqs
        if num_reqs > self._seq_lens_buf.numel():
            self._seq_lens_buf = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        fv_seq_lens = self._seq_lens_buf[:num_reqs]
        fv_seq_lens.copy_(cm.seq_lens[:num_reqs], non_blocking=True)
        verify_query_len = 0
        if expand:
            decode_rows = num_decode_tokens
            if num_decodes and num_decode_tokens == num_decodes * self.spec_query_len:
                qsl = cm.query_start_loc_cpu[: num_decodes + 1]
                if bool(((qsl[1:] - qsl[:-1]) == self.spec_query_len).all()):
                    verify_query_len = self.spec_query_len
            decode_seq_lens, decode_block_table = self.verify.expand(
                cm.query_start_loc,
                fv_seq_lens,
                cm.block_table_tensor,
                num_rows=num_decodes,
                num_mapped_tokens=int(cm.query_start_loc_cpu[num_decodes]),
                num_tokens=num_decode_tokens,
            )
        else:
            decode_rows = min(num_decodes, num_decode_tokens)
            decode_seq_lens = fv_seq_lens[:decode_rows]
            decode_block_table = cm.block_table_tensor[:decode_rows]
        groups: tuple[PrefillGroup, ...] = ()
        if num_prefills:
            groups = group_prefill_rows(
                cm.query_start_loc_cpu,
                num_decodes,
                num_reqs,
                self.device,
                persistent_arange=self._arange,
            )
        return FlashAttnV100Metadata(
            **{f.name: getattr(base, f.name) for f in fields(TritonAttentionMetadata)},
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill_groups=groups,
            max_model_len=self.max_model_len,
            decode_rows=decode_rows,
            decode_seq_lens=decode_seq_lens,
            decode_block_table=decode_block_table,
            fv_seq_lens=fv_seq_lens,
            spec_verify=expand,
            verify_query_len=verify_query_len,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashAttnV100Metadata:
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        metadata.cudagraph_capture = True
        return metadata
