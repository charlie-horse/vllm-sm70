# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN8 support search plus exact FP32 rerank for the DFlash2 draft top-k.

The DFlash2 draft picks selector_top_k (16) candidates per drafted position
from the full-vocabulary LM head. Instead of upstream's dense FP16 GEMM over
the local vocabulary shard this route runs:
  1. the channel-FP8 QPN8 GEMM over an E4M3 copy of the head -> approximate
     FP16 logits;
  2. an unsorted top-64 per vocabulary chunk of at most VOCAB_CHUNK rows;
  3. exact FP32 logits of those candidates from the FP16 head (one Triton
     dot per candidate), whose top-k is the shard result;
then the same all-gather, merge, scale and soft cap as upstream. A true
top-16 token is missed only if its E4M3 logit is not in its chunk's top 64.
The E4M3 copy is made on the first eager call (the profile run). Scope: the
qualified Qwen3.8 DFlash2 contract, one drafting request (M <= 8).
Kill switches: VLLM_SM70_DFLASH2_QPN8_RERANK=0, VLLM_SM70_DFLASH2_VERIFIER=0."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from vllm_sm70.ops import dflash2_fp8, qpn8

logger = logging.getLogger("vllm.sm70.dflash2")

KILL_SWITCH = "VLLM_SM70_DFLASH2_QPN8_RERANK"
STATE_ATTR = "_sm70_dflash2_rerank"
DECLINED_ATTR = "_sm70_dflash2_rerank_declined"

VOCAB_CHUNK = 62080
CANDIDATES = 64
MAX_ROWS = 8
SPLIT_K = 8
ACCUMULATOR_CHAINS = 1
# Rows quantized per step, so prepare never holds an FP32 copy of the head.
QUANT_CHUNK_ROWS = 4096

OPS = ("fp8_qpn8_prepare_sm70", "fp8_qpn8_gemm_sm70_out")


def enabled() -> bool:
    return os.environ.get(KILL_SWITCH) != "0"


def rerank_wanted(vllm_config) -> bool:
    """On for the qualified Qwen3.8 DFlash2 contract with the verifier on."""
    return enabled() and dflash2_fp8.draft_fp8_wanted(vllm_config)


def ops_available() -> bool:
    return all(qpn8.has_op(name) for name in OPS)


def vocab_groups(rows: int) -> tuple[tuple[int, int], ...]:
    """[begin, end) chunks of the local vocabulary."""
    return tuple(
        (begin, min(begin + VOCAB_CHUNK, rows)) for begin in range(0, rows, VOCAB_CHUNK)
    )


def num_candidates(rows: int) -> int:
    return sum(min(CANDIDATES, end - begin) for begin, end in vocab_groups(rows))


@dataclass
class RerankState:
    codes: torch.Tensor  # QPN8 [K, N] u8
    scales: torch.Tensor  # [1, N] fp16 channel scales
    groups: tuple[tuple[int, int], ...]
    approx: torch.Tensor  # [MAX_ROWS, N] fp16, FP8 GEMM output
    cand_values: torch.Tensor  # [MAX_ROWS, C] fp16 (top-k scratch)
    cand_ids: torch.Tensor  # [MAX_ROWS, C] int64, shard-local
    exact: torch.Tensor  # [MAX_ROWS, C] fp32


def admission(lm_head, logits_processor, top_k: int) -> str | None:
    """Why this head cannot take the route, or None if it can."""
    if not ops_available():
        return "K09 QPN8 ops not built"
    if dflash2_fp8._batch_invariant():
        return "VLLM_BATCH_INVARIANT"
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.ndim != 2:
        return "no 2-D weight"
    if weight.dtype != torch.float16 or not weight.is_contiguous():
        return "weight is not contiguous fp16"
    if not dflash2_fp8._is_sm70(weight.device):
        return "not an SM70 device"
    if getattr(lm_head, "bias", None) is not None:
        return "has bias"
    rows, hidden = weight.shape
    if rows < CANDIDATES or rows % 32 or hidden % 128 or (hidden // 16) % SPLIT_K:
        return f"shape {tuple(weight.shape)} (needs N>=64, N%32=0, K%128=0)"
    # Every check here must decide the same on every TP rank: the route ends
    # in an all-gather that all ranks must enter with the same dtype. Vocab
    # padding sits on the last rank only, so test the global vocabulary.
    shard = getattr(lm_head, "shard_indices", None)
    org = getattr(lm_head, "org_vocab_size", None)
    if (
        shard is None
        or org is None
        or getattr(lm_head, "org_vocab_size_padded", None) != org
        or getattr(lm_head, "num_added_embeddings", 0)
        or shard.num_org_vocab_padding
        or shard.num_added_elements
    ):
        return "padded or extended vocabulary"
    if getattr(logits_processor, "logits_as_input", False):
        return "logits_as_input"
    head_dtype = getattr(logits_processor, "head_dtype", None)
    if head_dtype is not None and head_dtype != torch.float16:
        return f"head_dtype {head_dtype}"
    if not float(getattr(logits_processor, "scale", 1.0)) > 0.0:
        return "non-positive logit scale"
    if not 1 <= int(top_k) <= CANDIDATES:
        return f"top_k={top_k}"
    return None


@torch.no_grad()
def channel_fp8_quantize_chunked(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dflash2_fp8.channel_fp8_quantize in row chunks (bounded FP32 scratch)."""
    rows = weight.shape[0]
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
    for begin in range(0, rows, QUANT_CHUNK_ROWS):
        end = min(begin + QUANT_CHUNK_ROWS, rows)
        q, s = dflash2_fp8.channel_fp8_quantize(weight[begin:end])
        quantized[begin:end].copy_(q)
        scales[begin:end].copy_(s)
    return quantized, scales


@torch.no_grad()
def prepare(lm_head) -> RerankState:
    """Quantize, pack and allocate the route's buffers (eager, never captured)."""
    weight = lm_head.weight
    rows, _hidden = weight.shape
    quantized, scales = channel_fp8_quantize_chunked(weight.detach())
    codes, packed_scales = qpn8.fp8_qpn8_prepare_sm70(quantized, scales)
    del quantized, scales
    c = num_candidates(rows)
    device = weight.device
    state = RerankState(
        codes=codes,
        scales=packed_scales,
        groups=vocab_groups(rows),
        approx=torch.empty((MAX_ROWS, rows), dtype=torch.float16, device=device),
        cand_values=torch.empty((MAX_ROWS, c), dtype=torch.float16, device=device),
        cand_ids=torch.empty((MAX_ROWS, c), dtype=torch.int64, device=device),
        exact=torch.empty((MAX_ROWS, c), dtype=torch.float32, device=device),
    )
    setattr(lm_head, STATE_ATTR, state)
    logger.info(
        "vllm_sm70: DFlash2 QPN8 rerank prepared: head N%d/K%d, %d vocabulary "
        "chunks, %d candidates, E4M3 copy %.2f GiB (VLLM_SM70_DFLASH2_QPN8_RERANK=0 "
        "disables it).",
        rows,
        _hidden,
        len(state.groups),
        c,
        codes.numel() * codes.element_size() / 2**30,
    )
    return state


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # no CUDA (CPU tests)
        return False


def _state(lm_head, logits_processor, top_k: int) -> RerankState | None:
    state = getattr(lm_head, STATE_ATTR, None)
    if state is not None or getattr(lm_head, DECLINED_ATTR, False):
        return state
    if torch.compiler.is_compiling() or _capturing():
        # The profile run is eager and comes first; never prepare in a graph.
        return None
    reason = admission(lm_head, logits_processor, top_k)
    if reason is not None:
        setattr(lm_head, DECLINED_ATTR, True)
        logger.info("vllm_sm70: DFlash2 QPN8 rerank not admitted: %s.", reason)
        return None
    return prepare(lm_head)


# ------------------------------------------------------------ exact dots
def exact_logits_reference(
    x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor
) -> None:
    """out[m, c] = <x[m], weight[ids[m, c]]> in FP32 (CPU and fallback)."""
    out.copy_(torch.einsum("mk,mck->mc", x.float(), weight[ids].float()))


def exact_logits(
    x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor
) -> None:
    if x.is_cuda:
        from vllm_sm70.ops.dflash2_rerank_kernel import exact_logits as triton_dots

        if triton_dots(x, weight, ids, out):
            return
    exact_logits_reference(x, weight, ids, out)


def _merge_topk(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The cross-rank merge upstream uses."""
    try:
        from vllm.model_executor.layers.logits_processor import _topk
    except ImportError:  # pragma: no cover - vllm without the helper
        return torch.topk(values, k, dim=-1)
    return _topk(values, k)


# ------------------------------------------------------------- the route
def shard_topk(
    state: RerankState,
    x: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(values fp32 [M, k], shard-local ids int64 [M, k]), sorted descending."""
    m = x.shape[0]
    approx = state.approx[:m]
    # split 8, one chain, fast decoder off
    qpn8.fp8_qpn8_gemm_sm70_out(
        approx, x, state.codes, state.scales, SPLIT_K, ACCUMULATOR_CHAINS, False, False
    )
    # unsorted top-64 per chunk (the rerank re-orders)
    offset = 0
    for begin, end in state.groups:
        count = min(CANDIDATES, end - begin)
        values = state.cand_values[:m, offset : offset + count]
        ids = state.cand_ids[:m, offset : offset + count]
        torch.topk(approx[:, begin:end], count, dim=-1, sorted=False, out=(values, ids))
        if begin:
            ids.add_(begin)
        offset += count
    cand_ids = state.cand_ids[:m]
    exact = state.exact[:m]
    exact_logits(x, weight, cand_ids, exact)
    values, selected = torch.topk(exact, top_k, dim=-1, sorted=True)
    return values, cand_ids.gather(-1, selected)


def maybe_candidates(
    lm_head,
    logits_processor,
    hidden_states: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(global ids int64, values fp32) like upstream get_top_k_tokens, or None."""
    x = hidden_states
    if x.ndim != 2 or x.dtype != torch.float16:
        return None
    # Prepare on the first eager call whatever its M: the profile run drafts
    # for max_num_seqs requests, and the copy must exist before KV sizing.
    state = _state(lm_head, logits_processor, top_k)
    if state is None or not 1 <= x.shape[0] <= MAX_ROWS:
        return None
    if x.shape[1] != lm_head.weight.shape[1] or x.device != lm_head.weight.device:
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    values, ids = shard_topk(state, x, lm_head.weight, top_k)
    ids = ids + lm_head.shard_indices.org_vocab_start_index
    # as in upstream get_top_k_tokens
    if lm_head.tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_gather

        values = tensor_model_parallel_all_gather(values, dim=-1)
        ids = tensor_model_parallel_all_gather(ids, dim=-1)
        values, selected = _merge_topk(values, top_k)
        ids = ids.gather(-1, selected)
    scale = float(logits_processor.scale)
    if scale != 1.0:
        values = values * scale
    soft_cap = logits_processor.soft_cap
    if soft_cap is not None:
        values = torch.tanh(values / soft_cap) * soft_cap
    return ids, values
