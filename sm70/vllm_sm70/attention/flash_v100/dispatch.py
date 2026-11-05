# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode/prefill dispatch onto the K05 Flash-V100 ops (K06).

Takes the place of the fork's decode/prefill executors
(flash_v100/{decode,prefill,dense_prefill,routing}.py, ~5.9K lines) for the
lean subset:

  decode   D256, no sinks, no window, GQA 4/6/8 -> tensor-core XQA
           otherwise (Flint D128 GQA8 sinks/window, other GQA) -> paged
           decode, with sinks when the layer has them
  prefill  rows grouped by query length -> paged prefill (generic, low-SMEM
           or BM32 phase chosen in C++), with sinks when present

CUDA graphs: a captured decode replays for any sequence length, so under
capture the launch grid covers max_model_len (the "workspace envelope");
eager decode launches only what the batch's max_seq_len needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_sm70.ops import fv100

if TYPE_CHECKING:
    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100Metadata

FULL_WINDOW = (-1, -1)


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
    """fork routing.py:133-135 plus decode.py:776-790 (XQA has no sink term)."""
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
    have zero tokens and sit at the end). With the K07 verify expansion a
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
    if metadata.cudagraph_capture or _capturing():
        envelope = max(metadata.max_model_len, metadata.max_seq_len, 1)
        hints = dict(max_seq_len_hint=envelope, workspace_seq_capacity_hint=envelope)
    else:
        hints = dict(max_seq_len_hint=max(metadata.max_seq_len, 1))
    if use_xqa(q, key_cache, window, sinks):
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


def fused_decode_hook(layer: torch.nn.Module, metadata: FlashAttnV100Metadata) -> None:
    """K14 (C4) hook point: prepare the one-launch sink decode. No-op in K06."""
    return None
