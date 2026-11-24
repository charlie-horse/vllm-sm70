# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the backends of the model-owned caches.

The model runs attention and the short convs itself, so neither backend has
an impl; each names the builder of its metadata (Inkling2AttentionMetadata,
Inkling2SconvMetadata), stored per cache owner (``cache_metadata(owner)``).
``num_actual_tokens`` is n = ``query_start_loc_cpu[num_reqs]``, not the
runner's padded count. Token tensors have P >= n rows: ops read rows [0, n),
and a row with a negative slot is padding. Builders sort decodes first.
A step is decode-only (``decode_only``) when it runs as a FULL decode graph:
q tokens per request (q in ``decode_query_lens``), zero-token requests last.
Its ops read no host data and their shapes depend on the token tensors and
block-table width alone. VLLM_SM70_INKLING_DECODE_GRAPH=0, or a per-visit
dump, marks no step decode-only. On the V2 runner a step builds once per
(builder class, spec). The backends declare the LBNHC layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_sm70 import envs

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
    "decode_graph_enabled",
    "decode_query_len",
    "decode_query_lens",
    "is_decode_only",
    "replays_decode_graph",
]

ATTENTION_BACKEND_NAME = "INKLING2_SM70_ATTN"
SCONV_BACKEND_NAME = "INKLING2_SM70_SCONV"
SWITCH = "VLLM_SM70_INKLING_DECODE_GRAPH"
_enabled = envs.environment_variables()[SWITCH]


def decode_graph_enabled(vllm_config: Any = None) -> bool:
    """False when VLLM_SM70_INKLING_DECODE_GRAPH=0 or the engine dumps the
    per-visit states (``sm70.inkling2.visit_dump_dir``)."""
    if not _enabled():
        return False
    if vllm_config is None:
        return True
    from vllm_sm70.config import sm70_config

    return not sm70_config(vllm_config).inkling2.visit_dump_dir


def decode_query_len(vllm_config: Any) -> int:
    """q: query tokens per request of a decode step, 1 + num_speculative_tokens
    with a speculative config (UP v1/attention/backend.py
    ``max_decode_query_len``), else 1."""
    spec = getattr(vllm_config, "speculative_config", None)
    k = int(getattr(spec, "num_speculative_tokens", None) or 0)
    # The V1 runner's parallel drafter appends k mask slots per request.
    v1_parallel = getattr(spec, "parallel_drafting", False) and not getattr(
        vllm_config, "use_v2_model_runner", False
    )
    return 1 + (2 * k if v1_parallel else k)


def decode_query_lens(vllm_config: Any) -> tuple[int, ...]:
    """Every q of a decode step: ``decode_query_len``, and with a dynamic
    speculative-decoding schedule (spec_policy.py) 1 + each of its draft
    counts, capped at num_speculative_tokens."""
    q = decode_query_len(vllm_config)
    spec = getattr(vllm_config, "speculative_config", None)
    schedule = getattr(spec, "num_speculative_tokens_per_batch_size", None)
    if not schedule:
        return (q,)
    return tuple(sorted({1 + min(int(n), q - 1) for _, _, n in schedule}))


def _uniform_decode(cm: CommonAttentionMetadata, query_lens: tuple[int, ...]) -> bool:
    """Whether every request of `cm` has q tokens, q one of `query_lens`,
    but for zero-token requests after them (host data only)."""
    if cm.max_query_len <= 1:
        return 1 in query_lens
    query_len = cm.max_query_len
    if query_len not in query_lens or not cm.num_reqs:
        return False
    lens = cm.query_start_loc_cpu[: cm.num_reqs + 1].diff()
    real, rest = divmod(int(lens.sum()), query_len)
    if rest or not real:
        return False
    return bool((lens[:real] == query_len).all()) and not bool(lens[real:].any())


def replays_decode_graph(vllm_config: Any, cm: CommonAttentionMetadata) -> bool:
    """Whether the runner runs the step of `cm` as a FULL decode graph
    (module docstring)."""
    from vllm.config.compilation import CUDAGraphMode

    if not decode_graph_enabled(vllm_config):
        return False
    if not _uniform_decode(cm, decode_query_lens(vllm_config)):
        return False
    compilation = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    if not isinstance(mode, CUDAGraphMode) or mode.decode_mode() != CUDAGraphMode.FULL:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "enforce_eager", False):
        return False
    cap = getattr(compilation, "max_cudagraph_capture_size", None) or 0
    return cm.num_actual_tokens <= cap


def is_decode_only(metadata: Any) -> bool:
    """Whether `metadata` is of a decode-only batch (module docstring)."""
    return bool(getattr(metadata, "decode_only", False))


@dataclass
class Inkling2SconvMetadata(AttentionMetadata):
    """Metadata of one conv-state cache group. R requests, P >= n tokens."""

    num_actual_tokens: int  # n: scheduled tokens, without padding
    block_size: int  # tokens per block of this group's block table
    block_table: torch.Tensor  # [R, max_blocks] int32 physical block per request
    slot_mapping: torch.Tensor  # [P] int64 cache slot of each token, -1 = skip
    seq_idx: torch.Tensor  # [P] int32 request (batch row) of each token
    query_start: torch.Tensor  # [P] int32 first batch token of that request
    positions: torch.Tensor  # [P] int64 absolute position in the sequence
    # Every request has at most one query token (module docstring).
    decode_only: bool = field(default=False, kw_only=True)


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
    buffers one builder owns for its lifetime (CUDA graphs capture them).

    A batch whose first n requests hold one token each and the rest none
    (every decode batch, padded or not) has seq_idx = query_start = row on
    rows [0, n) and 0 after them. The buffers keep that pattern between such
    batches and are written only when n or P changes."""

    def __init__(self, device: torch.device, size: int) -> None:
        self.size = size
        self.arange = torch.arange(size, dtype=torch.int32, device=device)
        self.seq_idx = torch.zeros(size, dtype=torch.int32, device=device)
        self.query_start = torch.zeros(size, dtype=torch.int32, device=device)
        self.positions = torch.zeros(size, dtype=torch.int64, device=device)
        # (n, P) of the decode pattern rows [0, P) hold, or None.
        self._pattern: tuple[int, int] | None = None

    @staticmethod
    def _decode_rows(cm: CommonAttentionMetadata, num_actual: int) -> bool:
        """Whether request i < num_actual has token i alone and the rest no
        token: with at most one token per request, the first num_actual
        requests hold num_actual tokens (host data only)."""
        if cm.max_query_len > 1 or num_actual > cm.num_reqs:
            return False
        return int(cm.query_start_loc_cpu[num_actual]) == num_actual

    def _fill_decode(self, num_actual: int, num_padded: int) -> None:
        if self._pattern == (num_actual, num_padded):
            return
        for buffer in (self.seq_idx, self.query_start):
            buffer[:num_actual].copy_(self.arange[:num_actual])
            buffer[num_actual:num_padded].zero_()
        self._pattern = (num_actual, num_padded)

    def fill(self, cm: CommonAttentionMetadata, decode_only: bool) -> dict[str, Any]:
        num_reqs = cm.num_reqs
        num_actual = int(cm.query_start_loc_cpu[num_reqs])
        num_padded = cm.slot_mapping.shape[0]
        if num_padded < num_actual:
            raise ValueError(
                f"vllm_sm70: slot mapping covers {num_padded} tokens, the "
                f"batch has {num_actual}."
            )
        if num_padded > self.size:
            raise ValueError(
                f"vllm_sm70: a batch of {num_padded} tokens exceeds "
                f"max_num_batched_tokens {self.size}."
            )
        seq_idx = self.seq_idx[:num_padded]
        query_start = self.query_start[:num_padded]
        positions = self.positions[:num_padded]
        own_positions = cm.positions is None
        if not num_reqs or not num_padded:
            pass
        elif not own_positions and self._decode_rows(cm, num_actual):
            self._fill_decode(num_actual, num_padded)
        else:
            self._pattern = None
            starts = cm.query_start_loc[: num_reqs + 1]
            ends = starts[1:].contiguous()
            token = self.arange[:num_padded]
            # Request of a token: the number of requests that end at or
            # before it. Zero-token requests are skipped.
            req = torch.searchsorted(ends, token, right=True).clamp_(max=num_reqs - 1)
            real = token < num_actual
            start = starts[req]
            seq_idx.copy_(torch.where(real, req, 0))
            query_start.copy_(torch.where(real, start, 0))
            if own_positions:
                context = cm.seq_lens[:num_reqs] - (ends - starts[:-1])
                positions.copy_(torch.where(real, context[req] + token - start, 0))
        if not own_positions:
            positions = cm.positions[:num_padded]
        return {
            "num_actual_tokens": num_actual,
            "slot_mapping": cm.slot_mapping[:num_padded],
            "seq_idx": seq_idx,
            "query_start": query_start,
            "positions": positions,
            "decode_only": decode_only,
        }


class _StepBuilds:
    """What the builders made in the current V2 step, by key: the metadata
    per (builder class, spec) and the token index (``_TokenIndex``)."""

    def __init__(self) -> None:
        self._step: tuple[Any, ...] = ()
        self._built: dict[Any, tuple[int, Any]] = {}

    def _select(self, cm: CommonAttentionMetadata) -> None:
        tensors = (cm.query_start_loc_cpu, cm.seq_lens, cm.positions)
        step = (*tensors, cm.num_reqs, cm.num_actual_tokens, cm.max_query_len)
        same = len(step) == len(self._step) and all(
            a is b for a, b in zip(tensors, self._step[:3], strict=True)
        ) and step[3:] == self._step[3:]
        if not same:
            self._step, self._built = step, {}

    def get(self, cm: CommonAttentionMetadata, key: Any) -> Any:
        self._select(cm)
        rows, value = self._built.get(key, (-1, None))
        return value if rows == cm.slot_mapping.shape[0] else None

    def put(self, cm: CommonAttentionMetadata, key: Any, value: Any) -> None:
        self._select(cm)
        self._built[key] = (cm.slot_mapping.shape[0], value)


_step_builds = _StepBuilds()


def _uses_v2_runner(vllm_config: Any) -> bool:
    return bool(getattr(vllm_config, "use_v2_model_runner", False))


class _Builder(AttentionMetadataBuilder):
    """Common part of both builders. The metadata of two groups with equal
    specs differs in the block table and slot mapping only."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport:
        if not decode_graph_enabled(vllm_config):
            return AttentionCGSupport.NEVER
        if decode_query_len(vllm_config) > 1:
            return AttentionCGSupport.UNIFORM_BATCH
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = int(kv_cache_spec.block_size)
        self.token_index = _TokenIndex(
            torch.device(device),
            int(vllm_config.scheduler_config.max_num_batched_tokens),
        )
        # Verify requests (q tokens) are ordered as decodes.
        self._init_reorder_batch_threshold(decode_query_len(vllm_config))
        self._reuse_step_builds = _uses_v2_runner(vllm_config)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Any:
        cm = common_attn_metadata
        if not self._reuse_step_builds:
            return self._build(cm)
        key = (type(self), self.kv_cache_spec)
        hit = _step_builds.get(cm, key)
        if hit is not None:
            return self.update_block_table(hit, cm.block_table_tensor, cm.slot_mapping)
        metadata = self._build(cm)
        _step_builds.put(cm, key, metadata)
        return metadata

    def _build(self, cm: CommonAttentionMetadata) -> Any:
        raise NotImplementedError

    def _token_fields(self, cm: CommonAttentionMetadata) -> dict[str, Any]:
        """``_TokenIndex.fill`` of `cm`; on V2 once per step for all builders."""
        if not self._reuse_step_builds:
            return self.token_index.fill(cm, replays_decode_graph(self.vllm_config, cm))
        fields = _step_builds.get(cm, _TokenIndex)
        if fields is None:
            fields = self.token_index.fill(
                cm, replays_decode_graph(self.vllm_config, cm)
            )
            _step_builds.put(cm, _TokenIndex, fields)
        return {**fields, "slot_mapping": cm.slot_mapping}

    def update_block_table(
        self, metadata: Any, blk_table: torch.Tensor, slot_mapping: torch.Tensor
    ) -> Any:
        updated = copy.copy(metadata)
        updated.block_table = blk_table
        updated.slot_mapping = slot_mapping[: metadata.slot_mapping.shape[0]]
        return updated


class Inkling2SconvMetadataBuilder(_Builder):
    def _build(self, cm: CommonAttentionMetadata) -> Inkling2SconvMetadata:
        return Inkling2SconvMetadata(
            block_size=self.block_size,
            block_table=cm.block_table_tensor,
            **self._token_fields(cm),
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

    def _build(self, cm: CommonAttentionMetadata) -> Inkling2AttentionMetadata:
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
            **self._token_fields(cm),
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

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)


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
