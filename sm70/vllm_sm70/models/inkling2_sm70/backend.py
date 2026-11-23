# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the backends of the model-owned caches.

The model runs attention and the short convs itself, so neither backend has
an attention impl. Each one only names the builder that turns the runner's
per-batch ``CommonAttentionMetadata`` into the metadata the model's ops read:

    Inkling2AttentionBackend -> Inkling2AttentionMetadata  (K/V caches)
    Inkling2SconvBackend     -> Inkling2SconvMetadata      (conv-state caches)

The runner stores one metadata object per cache owner under the owner's
forward-context name; ``cache_metadata(owner)`` reads it back.

``num_actual_tokens`` is n, the scheduled tokens of the batch:
``query_start_loc_cpu[num_reqs]``, not the runner's
``CommonAttentionMetadata.num_actual_tokens``, which is the padded count in a
FULL graph (UP v1/worker/gpu_model_runner.py ``_build_attention_metadata``).
Token tensors have P >= n rows. The runner gives P = n except in a FULL
graph, where P is the padded batch, while the model's tensors are always
padded to the capture size. So an op reads rows [0, n) only and treats the
model rows after them as padding; inside [0, n) a token with a negative slot
is padding too. The builders sort decode requests first.

A step that runs as a FULL decode graph is decode-only (``decode_only``):
every request has at most one query token, the engine's cudagraph_mode has
FULL decode graphs and the step's (padded) token count is at most
``max_cudagraph_capture_size``, so the runner captures or replays it (UP
v1/cudagraph_dispatcher.py). The builders read this from the engine's
compilation config at build time. The ops of such a step read no host data
and their shapes depend on the token tensors and the block-table width
alone; an op covers the rows of the slot mapping (P) and a row with a
negative slot is padding. Every other step, eager decodes included, takes
the ops that branch in Python on the batch (numpy request lengths,
nonzero): they run eagerly or as eager breaks of a PIECEWISE graph. The
builders declare UNIFORM_SINGLE_TOKEN_DECODE graph support.
VLLM_SM70_INKLING_DECODE_GRAPH=0, or a per-visit dump, marks no step
decode-only and declares no graph support.

On the V2 runner a step builds once per (builder class, spec) and fills the
token index once: the other groups of that step reuse the first build
through ``update_block_table``, as the V1 runner does itself (UP
v1/worker/gpu_model_runner.py ``_build_attention_metadata``); V2 reuses only
Mamba builds (UP v1/worker/gpu/attn_utils.py ``build_attn_metadata``). A step is the set of
builds that share the runner's ``query_start_loc_cpu``, ``seq_lens`` and
``positions`` tensors.

Cache views are logical (blocks, heads, tokens, width) in every layout; the
backends declare LBNHC, the layout Flash-V100 uses.
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


def replays_decode_graph(vllm_config: Any, cm: CommonAttentionMetadata) -> bool:
    """Whether the runner runs the step of `cm` as a FULL decode graph
    (module docstring)."""
    from vllm.config.compilation import CUDAGraphMode

    if cm.max_query_len > 1 or not decode_graph_enabled(vllm_config):
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
        step = (cm.query_start_loc_cpu, cm.seq_lens, cm.positions)
        same = len(step) == len(self._step) and all(
            a is b for a, b in zip(step, self._step, strict=True)
        )
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
        self._init_reorder_batch_threshold(1)
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
