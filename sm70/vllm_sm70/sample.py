# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch top-k/top-p sampling from vocab-parallel logit candidates (K16).

Each TP rank selects the 32 largest logits of its FP16 shard; the ranks
exchange only those (value, token) rows, and one kernel applies temperature,
top-k (keeping ties at the k-th value), top-p and the exponential race of
``random_sample`` to the gathered candidates. With k <= 20 this is the
reference sampler (``apply_top_k_top_p_pytorch`` + ``random_sample`` with
FP32 exponential noise) restricted to the only tokens it can return. The
kernel draws the exact exponentials ``q.exponential_()`` would draw for those
tokens and advances the generator by the same offset, so the same RNG state
samples the same token.

The route replaces the full-vocabulary logits all-gather, two
full-vocabulary softmaxes and the radix sort of the eager sampler.

Known limit: a rank sends at most 32 candidates. When a rank's 32nd
candidate still reaches the k-th value, logits tied with it beyond those 32
cannot be sampled, while the reference would keep them (top-k keeps all
ties). With k <= 20 that needs more than 12 extra exact FP16 ties on one
rank. The kernel counts such rows on the device; ``sample_topk_candidates``
reads the count back asynchronously every ``_OVERFLOW_CHECK_INTERVAL``
calls and logs a warning when it grows.

Ported from fork cf9492a vllm/v1/sample/ops/sm70_topk_candidates.py (see
csrc/sampler/PROVENANCE.md for the function map). This module knows nothing
about the model runner: callers pass the sampling metadata, the host copy of
the per-row top-k, and the LM head. The V1 runner hook (patch U4) that calls
it is vllm_sm70/sample_v1.py.

R4 (capture-time state): the overflow counter is created by ``prepare()``,
once per device, before any CUDA graph capture. ``sample_topk_candidates``
never creates it; without it the kernel runs uncounted.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from vllm_sm70.ops import sampler as ops

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.sample.metadata import SamplingMetadata

logger = logging.getLogger("vllm.sm70")

# Candidates per TP rank; the kernel's wire row holds 32 values + 32 ids.
LOCAL_K = ops.LOCAL_K
WIRE_WIDTH = ops.WIRE_WIDTH
# Largest admitted top-k: leaves 12 spare candidates per rank for logits
# tied with the k-th value, which the reference keeps.
MAX_TOP_K = 20
# Request rows admitted (decode at batch 1-2).
MAX_ROWS = 2
_MAX_KERNEL_ROWS = ops.MAX_KERNEL_ROWS
_MAX_LOCAL_VOCAB = ops.MAX_LOCAL_VOCAB
_MAX_TP = ops.MAX_TP
_OVERFLOW_CHECK_INTERVAL = 64


def ops_available() -> bool:
    return ops.available()


@cache
def _is_sm70(device_index: int | None) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 0)


class _OverflowMonitor:
    """Device counter of rows whose ties may exceed a rank's candidates."""

    def __init__(self, device: torch.device):
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.mirror = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.calls = 0
        self.reported = 0

    def after_call(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            return  # host bookkeeping is not replayed; skip it while capturing
        self.calls += 1
        if self.calls % _OVERFLOW_CHECK_INTERVAL:
            return
        # The mirror holds the count copied asynchronously one interval ago.
        seen = int(self.mirror[0])
        if seen > self.reported:
            logger.warning(
                "SM70 top-k candidate sampler: %d sampled rows had more logits "
                "tied with the k-th value than one rank's 32 candidates hold; "
                "those ties were not sampleable.",
                seen,
            )
            self.reported = seen
        self.mirror.copy_(self.counter, non_blocking=True)


_OVERFLOW_MONITORS: dict[torch.device, _OverflowMonitor] = {}


def _device_key(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def prepare(device: torch.device) -> None:
    """Create the device's overflow counter (idempotent).

    Call once per device before CUDA graph capture (the runner hook does it
    from the platform's kernel_warmup). Raises while a capture is running.
    """
    device = _device_key(device)
    if device in _OVERFLOW_MONITORS:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "vllm_sm70.sample.prepare() must run before CUDA graph capture (R4)"
        )
    _OVERFLOW_MONITORS[device] = _OverflowMonitor(device)


def _overflow_monitor(device: torch.device) -> _OverflowMonitor | None:
    return _OVERFLOW_MONITORS.get(_device_key(device))


def overflow_count(device: torch.device) -> int:
    """Synchronously read the device's tie-overflow count (tests, debugging)."""
    monitor = _overflow_monitor(device)
    if monitor is None:
        raise RuntimeError(f"no overflow counter on {device}; call prepare() first")
    return int(monitor.counter.item())


def metadata_reject_reason(
    sampling_metadata: SamplingMetadata,
    top_k_cpu: Sequence[int] | None,
    num_reqs: int,
) -> str | None:
    """CPU-only admission of the request batch; None means admitted.

    ``top_k_cpu`` is the host copy of the per-row top-k (upstream's
    SamplingMetadata has only the device tensor; reading that would sync).
    Penalties, logprobs, masks and greedy rows need the full-vocabulary path.
    Penalties are also the case that needs ``output_token_ids``, which the V1
    runner refreshes under async scheduling only after this runs. The caller
    checks its own state (spec decode, structured output, logits processors)
    separately.
    """
    if not 1 <= num_reqs <= MAX_ROWS:
        return "batch_size"
    if not sampling_metadata.all_random or sampling_metadata.all_greedy:
        return "sampling_mode"
    if sampling_metadata.max_num_logprobs is not None:
        return "logprobs"
    if sampling_metadata.logprob_token_ids:
        return "logprob_token_ids"
    if not sampling_metadata.no_penalties:
        return "penalties"
    if sampling_metadata.allowed_token_ids_mask is not None:
        return "allowed_token_ids"
    if sampling_metadata.bad_words_token_ids:
        return "bad_words"
    if (
        sampling_metadata.top_k is None
        or top_k_cpu is None
        or len(top_k_cpu) != num_reqs
        or not all(1 <= int(k) <= MAX_TOP_K for k in top_k_cpu)
    ):
        return "top_k"
    if sampling_metadata.temperature is None:
        return "temperature"
    generators = sampling_metadata.generators
    if generators and (num_reqs != 1 or len(generators) != 1):
        # Mixed seeded/unseeded rows draw from several generators.
        return "generators"
    return None


def argmax_invariant_reject_reason(
    sampling_metadata: SamplingMetadata,
) -> str | None:
    """Only an inactive min-p processor may be among the argmax-invariant ones."""
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        if isinstance(processor, MinPLogitsProcessor) and not processor.min_p_count:
            continue
        return "argmax_invariant_logits_processor"
    return None


def admission_reject_reason(
    sampling_metadata: SamplingMetadata,
    top_k_cpu: Sequence[int] | None,
    num_reqs: int,
    device: torch.device,
) -> str | None:
    """Device, batch and kernel admission of the sampler; None means admitted.

    The caller checks its own state (spec decode, structured output, thinking
    budget, non-argmax-invariant logits processors) before this.
    """
    if device.type != "cuda":
        return "non_cuda"
    if torch.cuda.get_device_capability(device) != (7, 0):
        return "non_sm70"
    reason = metadata_reject_reason(sampling_metadata, top_k_cpu, num_reqs)
    if reason is not None:
        return reason
    reason = argmax_invariant_reject_reason(sampling_metadata)
    if reason is not None:
        return reason
    if not ops_available():
        return "missing_ops"
    return None


def lm_head_reject_reason(
    logits_processor: LogitsProcessor, lm_head: VocabParallelEmbedding, tp_size: int
) -> str | None:
    """Rank-uniform LM-head admission; None means admitted.

    Decided from configuration only (the same on every TP rank), so either
    all ranks enter the collective in ``lm_head_topk_candidates`` or none.
    Checks every rank's shard, not just this one: the last rank holds the
    fewest valid logits.
    """
    if (
        getattr(logits_processor, "logits_as_input", False)
        or logits_processor.soft_cap is not None
        or logits_processor.scale != 1.0
    ):
        return "logits_processor_transform"
    head_dtype = getattr(logits_processor, "head_dtype", None)
    if head_dtype not in (None, torch.float16):
        return "head_dtype"
    if lm_head.weight.dtype != torch.float16:
        return "lm_head_dtype"
    if getattr(lm_head, "bias", None) is not None:
        return "lm_head_bias"
    if getattr(lm_head, "num_added_embeddings", 0) != 0:
        return "added_vocab"
    if getattr(lm_head, "tp_size", tp_size) != tp_size:
        return "lm_head_group"
    if not 1 <= tp_size <= _MAX_TP:
        return "tp_size"
    org_vocab = lm_head.org_vocab_size
    if getattr(logits_processor, "org_vocab_size", org_vocab) != org_vocab:
        return "vocab_size"
    per_rank = lm_head.org_vocab_size_padded // tp_size
    last_rank_valid = org_vocab - (tp_size - 1) * per_rank
    if (
        lm_head.org_vocab_size_padded % tp_size
        or not LOCAL_K <= last_rank_valid
        or lm_head.num_embeddings_per_partition > _MAX_LOCAL_VOCAB
        or lm_head.org_vocab_size_padded > ops.MAX_TOKEN_ID
    ):
        return "shard_shape"
    return None


def lm_head_topk_candidates(
    logits_processor: LogitsProcessor,
    lm_head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Rank-local top-32 logits of every TP rank for top-k sampling.

    Returns rank-major FP32 rows ``[tp_size * rows, 64]`` (32 values, then
    32 global token ids) for ``sample_topk_candidates``, or None before any
    collective when the shard does not qualify. The decision depends only on
    shapes and configuration, so every rank takes the same branch.
    """
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if lm_head_reject_reason(logits_processor, lm_head, tp_size) is not None:
        return None
    rows = hidden_states.shape[0]
    if not 1 <= rows <= _MAX_KERNEL_ROWS or hidden_states.dtype != torch.float16:
        return None
    shard = lm_head.shard_indices
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=None)
    wire = local_topk_candidates(
        logits,
        vocab_start=shard.org_vocab_start_index,
        num_valid=shard.num_org_elements,
    )
    if wire is None:
        # Only reachable if apply() broke the rank-uniform contract above
        # (dtype or layout); every rank sees the same dtype, so all return.
        return None
    if tp_size > 1:
        wire = tensor_model_parallel_all_gather(wire, dim=0)
    return wire


def sample_lm_head_candidates(
    logits_processor: LogitsProcessor,
    lm_head: VocabParallelEmbedding,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    vocab_size: int,
) -> torch.Tensor | None:
    """Sample admitted rows from the LM head's vocab-parallel candidates.

    Returns int32 token ids [rows], or None when the shard or tensors do not
    qualify (the caller then runs the full-vocabulary sampler).
    """
    from vllm.distributed import get_tensor_model_parallel_world_size

    wire = lm_head_topk_candidates(logits_processor, lm_head, hidden_states)
    if wire is None:
        return None
    sampled = sample_topk_candidates(
        wire,
        sampling_metadata,
        tp_size=get_tensor_model_parallel_world_size(),
        vocab_size=vocab_size,
    )
    if sampled is not None:
        _log_once("SM70 top-k candidate sampler enabled.")
    return sampled


def local_topk_candidates(
    logits: torch.Tensor, *, vocab_start: int, num_valid: int
) -> torch.Tensor | None:
    """Return this rank's [rows, 64] FP32 wire rows, or None if unsupported.

    ``logits`` are the rank's raw FP16 logits (row-major, any row stride);
    columns >= ``num_valid`` are vocabulary padding and never selected.
    """
    if not shard_supported(logits, vocab_start=vocab_start, num_valid=num_valid):
        return None
    return ops.local_top32_half(logits, vocab_start, num_valid)


def shard_supported(logits: torch.Tensor, *, vocab_start: int, num_valid: int) -> bool:
    rows = logits.shape[0] if logits.ndim == 2 else 0
    return (
        logits.is_cuda
        and logits.dtype == torch.float16
        and logits.ndim == 2
        and logits.stride(1) == 1
        and 1 <= rows <= _MAX_KERNEL_ROWS
        and LOCAL_K <= num_valid <= logits.shape[1] <= _MAX_LOCAL_VOCAB
        and vocab_start >= 0
        and vocab_start + logits.shape[1] <= ops.MAX_TOKEN_ID
        # The kernel's row-stride check (topk_candidates_sampler.cu:341-343).
        and (rows == 1 or logits.shape[1] <= logits.stride(0) <= (1 << 31) // rows)
        and _is_sm70(logits.device.index)
        and ops_available()
    )


def sample_topk_candidates(
    wire: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    tp_size: int,
    vocab_size: int,
) -> torch.Tensor | None:
    """Sample one token per row from rank-major gathered wire rows.

    The caller admits the batch with ``metadata_reject_reason`` first.
    Returns int32 token ids [rows], or None when the tensors do not qualify
    (the caller then runs the full-vocabulary sampler).
    """
    temperature = sampling_metadata.temperature
    top_k = sampling_metadata.top_k
    top_p = sampling_metadata.top_p
    if (
        not 1 <= tp_size <= _MAX_TP
        or wire.dtype != torch.float32
        or wire.ndim != 2
        or wire.shape[1] != WIRE_WIDTH
        or wire.shape[0] % tp_size != 0
        or not wire.is_contiguous()
        or not LOCAL_K <= vocab_size <= ops.MAX_TOKEN_ID
        or temperature is None
        or top_k is None
    ):
        return None
    rows = wire.shape[0] // tp_size
    generators = sampling_metadata.generators
    if (
        not 1 <= rows <= _MAX_KERNEL_ROWS
        or temperature.dtype != torch.float32
        or temperature.shape != (rows,)
        or not temperature.is_contiguous()
        or top_k.dtype != torch.int32
        or top_k.shape != (rows,)
        or not top_k.is_contiguous()
        or (
            top_p is not None
            and (
                top_p.dtype != torch.float32
                or top_p.shape != (rows,)
                or not top_p.is_contiguous()
            )
        )
        or (generators and (rows != 1 or len(generators) != 1))
    ):
        return None
    generator = next(iter(generators.values()), None) if generators else None
    monitor = _overflow_monitor(wire.device)
    sampled = ops.sample_topk_candidates(
        wire,
        temperature,
        top_k,
        top_p,
        generator,
        vocab_size,
        tp_size,
        None if monitor is None else monitor.counter,
    )
    if monitor is not None:
        monitor.after_call()
    return sampled


_logged: set[str] = set()


def _log_once(message: str, *args: Any) -> None:
    key = message % args if args else message
    if key not in _logged:
        _logged.add(key)
        logger.info(message, *args)
