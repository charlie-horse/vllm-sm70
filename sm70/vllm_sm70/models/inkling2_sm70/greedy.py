# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: greedy tokens without the full-vocabulary logits.

The model opts in to the V1 hidden-state sampler (vllm_sm70/sample_v1.py,
``supports_sm70_topk_sampler``), so an all-greedy batch the F12 fast path
admits (vllm_sm70/sample_greedy.py) takes ``get_top_tokens`` in place of
compute_logits, the fp16 logit all-gather and the Sampler. Batches it does
not admit (sampling, logprobs, penalties, ...) keep the upstream path; the
top-k candidate route refuses the soft cap before any work.

``get_top_tokens`` of the logits processor (model.py): the head GEMM, then
one kernel per shard that soft-caps each logit and keeps the first maximum
of every run of ``ops/top1.SPLIT_COLS`` columns, an all-gather of those
(value, id) pairs at TP > 1, and one kernel that picks the first maximal
pair. Tensors off the GPU, a scale other than 1 and vocabulary padding take
upstream's ``get_top_tokens``. Same token as upstream's argmax over the
gathered, soft-capped fp32 logits (lowest id on ties).

VLLM_SM70_INKLING_GREEDY_TOP1=0 withdraws the opt-in: every batch takes
compute_logits and the upstream Sampler. Read on every step.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_sm70 import envs

from .ops import _compat

__all__ = ["SWITCH", "enabled", "on_kernels", "top_tokens"]

SWITCH = "VLLM_SM70_INKLING_GREEDY_TOP1"
_enabled = envs.environment_variables()[SWITCH]


def enabled() -> bool:
    return _enabled()


def on_kernels(logits_processor: Any, lm_head: Any, x: torch.Tensor) -> bool:
    """Whether ``top_tokens`` runs the kernels for hidden states `x`; the
    same answer on every TP rank (configuration and shapes only)."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return bool(
        logits_processor.scale == 1.0
        and lm_head.shard_indices.num_org_vocab_padding == 0
        and lm_head.weight.dtype == torch.float16
    )


def top_tokens(
    logits: torch.Tensor, lm_head: Any, soft_cap: float | None
) -> torch.Tensor:
    """int64 [rows]: the greedy token of this rank's fp16 `logits` shard
    [rows, vocab / tp] over every shard of the LM head."""
    from .ops import top1

    pairs = top1.softcap_top1(
        logits, lm_head.shard_indices.org_vocab_start_index, soft_cap
    )
    rows, splits, _ = pairs.shape
    if lm_head.tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_gather

        pairs = tensor_model_parallel_all_gather(pairs.view(rows, splits * 2), dim=-1)
        pairs = pairs.view(rows, lm_head.tp_size * splits, 2)
    return top1.pick(pairs)
