# SPDX-License-Identifier: Apache-2.0
"""Greedy vocab-parallel top-1 for all-greedy batches (F12).

At temperature 0 the upstream V1 path all-gathers the full-vocabulary
logits (Flint TP2: 150144 FP16 values per row), converts them to FP32 and
runs the Sampler only to take ``argmax``. Here each TP rank takes the
argmax of its own LM-head shard, and the ranks all-gather one
(value, token id) pair per row and rank. Upstream already carries that
reduction as ``LogitsProcessor.get_top_tokens`` (UP model_executor/layers/
logits_processor.py:201-247: same LM-head GEMM through ``_apply_head``,
same soft cap and scale, padding masked to -inf, FP32 pairs), and this
module calls it.

Why the token equals upstream's ``Sampler.greedy_sample``: FP16 to FP32 is
exact, so the per-element values are those the Sampler compares. Each
rank's ``max`` returns its first maximal index, ranks own increasing vocab
ranges (no added vocab), and the cross-rank ``argmax`` picks the lowest
rank on ties, so the result is the first maximal index over the whole
vocabulary, as ``argmax`` over the gathered logits. Token ids travel as
FP32 and are exact below 2^24.

Port of fork cf9492a: switch VLLM_SM70_GREEDY_TOKEN_FASTPATH=1 (FORK
envs.py:7033, kernel_config.layer_execution.greedy_token_fastpath, runner
:1314), gate ``_can_use_greedy_token_fastpath`` (runner :7646-7695), call
``_sample_greedy_token_fastpath`` (runner :7845-7862), and the Flint
``get_top_tokens`` with ``fused_top1=False`` (FORK models/flint/model.py:
571-578, :597-603), i.e. the plain LM-head GEMM, which is what upstream's
``get_top_tokens`` does. The fork's DDTree greedy branch (runner :6940-6962,
:7150-7176) is spec decode, which the U4 hook never sees. The runner side
is ``Sm70HiddenStateSampler`` in vllm_sm70/sample_v1.py (patch U4).
"""

from __future__ import annotations

from typing import Any

import torch

# FP32 carries token ids exactly below 2^24.
MAX_VOCAB = 1 << 24


def metadata_reject_reason(sampling_metadata: Any) -> str | None:
    """Batch admission: the Sampler would do nothing but argmax.

    FORK runner :7673-7694. The caller checks spec decode, structured output,
    thinking budget and non-argmax-invariant logits processors (shared with
    K16). Argmax-invariant processors (min-p, top-k/top-p) cannot change a
    greedy token, so they are not checked.
    """
    if not sampling_metadata.all_greedy:
        return "non_greedy"
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
    return None


def device_reject_reason(device: torch.device | None) -> str | None:
    """FORK runner :7663-7668: CUDA and SM70 only."""
    if device is None or device.type != "cuda":
        return "non_cuda"
    if torch.cuda.get_device_capability(device) != (7, 0):
        return "non_sm70"
    return None


def lm_head_reject_reason(logits_processor: Any, lm_head: Any) -> str | None:
    """LM-head admission, from configuration only (the same on every rank,
    so either all TP ranks enter the pair all-gather or none does)."""
    if getattr(logits_processor, "logits_as_input", False):
        return "logits_as_input"
    if not callable(getattr(logits_processor, "get_top_tokens", None)):
        return "logits_processor"
    if logits_processor.scale <= 0.0:
        # get_top_tokens raises for it; a negative scale flips the argmax.
        return "scale"
    if getattr(lm_head, "num_added_embeddings", 0) != 0:
        # Added (LoRA) vocab sits after the padding in each shard, so shard
        # order would no longer be vocab order.
        return "added_vocab"
    org_vocab = getattr(lm_head, "org_vocab_size", None)
    if org_vocab is None or logits_processor.org_vocab_size != org_vocab:
        # Upstream trims the gathered logits to the processor's size.
        return "vocab_size"
    if getattr(lm_head, "org_vocab_size_padded", org_vocab) > MAX_VOCAB:
        return "vocab_size"
    if getattr(lm_head, "shard_indices", None) is None:
        return "lm_head"
    return None


def greedy_tokens(
    logits_processor: Any, lm_head: Any, hidden_states: torch.Tensor
) -> torch.Tensor:
    """int32 token ids [rows]: the argmax of the full-vocabulary logits."""
    top = logits_processor.get_top_tokens(lm_head, hidden_states)
    return top.to(torch.int32)
