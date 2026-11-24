# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode/prefill dispatch onto the Flash-V100 ops.

  decode   D256, no sinks, no window, GQA 4/6/8 -> tensor-core XQA
           otherwise (Flint D128 GQA8 sinks/window, other GQA) -> paged
           decode, with sinks when the layer has them
  prefill  rows grouped by query length -> paged prefill (generic, low-SMEM
           or BM32 phase chosen in C++), with sinks when present

CUDA graphs: a captured decode replays for any sequence length, so under
capture the launch grid covers max_model_len (the "workspace envelope");
eager decode launches only what the batch's max_seq_len needs. A captured
XQA decode of at most SMALL_BATCH_ROWS rows uses 256-token partitions
instead of the envelope's default (1024 when max_model_len >= 32768), so
small batches use more of the GPU. That workspace is its own cache entry.
Kill switch: VLLM_SM70_FLASH_V100_SMALL_BATCH_PARTITION=0 (read at capture)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

from vllm_sm70.ops import fv100

if TYPE_CHECKING:
    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100Metadata

logger = logging.getLogger("vllm.sm70")

FULL_WINDOW = (-1, -1)

VERIFY_SWITCH = "VLLM_SM70_FLASH_V100_VERIFY_KERNEL"
_verify_logged = False


def verify_kernel_enabled() -> bool:
    """Verify-kernel kill switch: only "0" acts (the XQA expansion runs instead)."""
    return os.environ.get(VERIFY_SWITCH) != "0"


def use_verify_kernel(
    query: torch.Tensor, key_cache: torch.Tensor, metadata: FlashAttnV100Metadata
) -> bool:
    """The verify kernel takes a verify batch whose requests all have 8 tokens, up to
    the rows whose workspace fits fv100.VERIFY_WORKSPACE_BUDGET at this
    max_model_len. The switch is read here on every call, so a CUDA graph
    keeps the route it was captured with."""
    if not (
        bool(getattr(metadata, "spec_verify", False))
        and getattr(metadata, "verify_query_len", 0) == fv100.VERIFY_QUERY_LEN
        and verify_kernel_enabled()
        and fv100.verify_supported(query, key_cache)
    ):
        return False
    envelope = max(int(metadata.max_model_len), int(metadata.max_seq_len), 1)
    return query.shape[0] <= fv100.verify_max_rows(
        query.shape[1], query.shape[2], envelope
    )


def _log_verify_once() -> None:
    global _verify_logged
    if not _verify_logged:
        _verify_logged = True
        logger.info(
            "vllm_sm70: Flash-V100 DFlash2 verify kernel active (8 tokens per "
            "request, KV read once; %s=0 keeps the K07 XQA expansion).",
            VERIFY_SWITCH,
        )


SMALL_BATCH_SWITCH = "VLLM_SM70_FLASH_V100_SMALL_BATCH_PARTITION"
SMALL_BATCH_ROWS = 16  # two DFlash2 verify requests (8 rows each)
SMALL_BATCH_PARTITION = 256
_small_batch_logged = False


def small_batch_partition(rows: int) -> int | None:
    """Partition size for a captured XQA decode of `rows` rows, or None."""
    if os.environ.get(SMALL_BATCH_SWITCH) == "0":
        return None
    if not 1 <= rows <= SMALL_BATCH_ROWS:
        return None
    return SMALL_BATCH_PARTITION


def _log_small_batch_once() -> None:
    global _small_batch_logged
    if not _small_batch_logged:
        _small_batch_logged = True
        logger.info(
            "vllm_sm70: Flash-V100 captured XQA decode of <= %d rows uses "
            "%d-token partitions (%s=0 keeps the envelope's default).",
            SMALL_BATCH_ROWS,
            SMALL_BATCH_PARTITION,
            SMALL_BATCH_SWITCH,
        )


def split_kv_cache(
    kv_cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V views of upstream's logical (blocks, H_kv, block_size, 2*D) cache."""
    return fv100.split_lbnhc_kv_cache(kv_cache, head_size)


def use_xqa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    window: tuple[int, int],
    sinks: torch.Tensor | None,
) -> bool:
    """Whether the layer can take the XQA decode (XQA has no sink term)."""
    return (
        sinks is None
        and window == FULL_WINDOW
        and fv100.xqa_supported(query, key_cache)
    )


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # no CUDA (CPU tests)
        return False


def run_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata: FlashAttnV100Metadata,
    *,
    scale: float,
    window: tuple[int, int],
    sinks: torch.Tensor | None,
) -> None:
    """Decode rows [0, decode_rows), one query token each (row i <-> token i).

    Without spec decode a row is a request (padding rows of a captured batch
    have zero tokens and sit at the end). With the verify expansion a
    row is a verify token, with its own visible length and block-table row
    (metadata.decode_seq_lens / decode_block_table, persistent buffers).
    """
    num = metadata.decode_rows
    if num == 0:
        return
    q = query[:num]
    out = output[:num]
    seq_lens = metadata.decode_seq_lens
    block_table = metadata.decode_block_table
    captured = metadata.cudagraph_capture or _capturing()
    if captured:
        envelope = max(metadata.max_model_len, metadata.max_seq_len, 1)
        hints = dict(max_seq_len_hint=envelope, workspace_seq_capacity_hint=envelope)
    else:
        hints = dict(max_seq_len_hint=max(metadata.max_seq_len, 1))
    if use_xqa(q, key_cache, window, sinks) and use_verify_kernel(
        q, key_cache, metadata
    ):
        _log_verify_once()
        # The workspace always covers the envelope (one cache entry per
        # engine); only the launch grid follows the live length eagerly.
        fv100.flash_attn_decode_verify(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            out=out,
            max_seq_len_hint=hints["max_seq_len_hint"],
            workspace_seq_capacity_hint=max(
                metadata.max_model_len, metadata.max_seq_len, 1
            ),
        )
        return
    if use_xqa(q, key_cache, window, sinks):
        partition = small_batch_partition(num) if captured else None
        if partition is not None:
            hints["partition_size_hint"] = partition
            _log_small_batch_once()
        fv100.flash_attn_decode_paged_xqa(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            out=out,
            **hints,
        )
    else:
        fv100.flash_attn_decode_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            out=out,
            window_size=window,
            sinks=sinks,
            **hints,
        )


def run_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    metadata: FlashAttnV100Metadata,
    *,
    scale: float,
    causal: bool,
    window: tuple[int, int],
    sinks: torch.Tensor | None,
) -> None:
    """Varlen prefill: one uniform-M launch per query-length group."""
    num_heads, head_size = query.shape[1], query.shape[2]
    for group in metadata.prefill_groups:
        q = query.index_select(0, group.token_index).view(
            group.num_rows, group.query_len, num_heads, head_size
        )
        out = fv100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            metadata.block_table.index_select(0, group.rows),
            metadata.fv_seq_lens.index_select(0, group.rows),
            softmax_scale=scale,
            causal=causal,
            window_size=window,
            sinks=sinks,
        )
        output.index_copy_(0, group.token_index, out.view(-1, num_heads, head_size))


# ---------------------------------------------------------------------------
# One-launch sink decode.
# ---------------------------------------------------------------------------

# A sink-layer forward ran under capture since the last eager one.
_SINK_CAPTURE_PENDING = [False]
_SINK_CAPTURE_REPORTED: list[tuple[int, int] | None] = [None]


def _log_sink_capture_summary() -> None:
    totals = fv100.flash_attn_decode_paged_sinks_capture_counts()
    previous = _SINK_CAPTURE_REPORTED[0]
    if not any(totals) or totals == previous:
        return
    _SINK_CAPTURE_REPORTED[0] = totals
    # The totals grow with every captured size, so INFO only for the first
    # report or a new two-launch fallback; later growth goes to DEBUG. Each
    # fallback already warned with its reason when it was captured.
    first_or_fallback = previous is None or totals[1] != previous[1]
    (fv100.logger.info if first_or_fallback else fv100.logger.debug)(
        "Flash-V100 grouped sink decode in captured CUDA graphs so far: %d "
        "one-launch, %d two-launch (partition kernel plus reducer).",
        *totals,
    )


def init_fused_decode(impl) -> None:
    """At layer construction: mark sink layers the one-launch kernel takes and
    allocate the device's counters right away (eager, before any capture).

    vLLM v0.33.0's profile run has no attention metadata and the decode
    graphs are captured largest batch first, so the layer's first eager
    forward is only the capture warmup; allocating here does not depend on
    it. `fused_decode_hook` repeats the (idempotent) prepare on the first
    eager forward for devices that were not ready at construction.
    """
    impl._fv100_fused_sinks = bool(
        getattr(impl, "_fv100_reason", None) is None
        and impl.sinks is not None
        and impl.head_size == fv100.FUSED_DECODE_HEAD_DIM
        and impl.num_heads == fv100.FUSED_DECODE_GQA * impl.num_kv_heads
    )
    impl._fv100_fused_ready = False
    if not impl._fv100_fused_sinks:
        return
    device = impl.sinks.device
    if device.type != "cuda":
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        impl._fv100_fused_ready = fv100.flash_attn_decode_paged_sinks_prepare(device)
    except Exception:  # never fail model construction over an optimisation
        fv100.logger.debug("Flash-V100 fused decode prepare failed", exc_info=True)


def fused_decode_hook(impl, metadata: FlashAttnV100Metadata) -> None:
    """Every Flash-V100 forward: keep the one-launch sink decode ready.

    Sink layers only. Under capture nothing is allocated (a graph without
    counters decodes in two launches) and the capture is noted; the first
    eager forward after a capture reports the capture totals; the layer's
    first eager forward asks for the device's counters once (a no-op when
    VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0 or the op is not built).
    """
    if not getattr(impl, "_fv100_fused_sinks", False):
        return
    if _capturing():
        _SINK_CAPTURE_PENDING[0] = True
        return
    if _SINK_CAPTURE_PENDING[0]:
        _SINK_CAPTURE_PENDING[0] = False
        _log_sink_capture_summary()
    if impl._fv100_fused_ready:
        return
    impl._fv100_fused_ready = True
    device = metadata.seq_lens.device
    if device.type == "cuda":
        fv100.flash_attn_decode_paged_sinks_prepare(device)
