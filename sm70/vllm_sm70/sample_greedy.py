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

Pair exchange over the custom all-reduce buffers (fork
``VLLM_SM70_TOP1_CUSTOM_AR``, default on): instead of the NCCL all-gather,
each rank writes its pairs into upstream's ``CustomAllreduce`` eager buffer
and one IPC kernel (``_sm70_C::custom_ar_top1_argmax``, csrc/custom_ar/
top1_custom_ar.cu) reduces them with the same first-index tie rule. Port of
FORK ``LogitsProcessor._maybe_custom_top1_argmax`` (logits_processor.py:
817-847), ``CustomAllreduce.custom_top1_argmax`` / ``top1_argmax``
(custom_all_reduce.py:771-790, :977-998) and ``cross_device_top1_argmax``
(csrc/custom_all_reduce.cuh:1720-1750). Rank-uniform admission
(``custom_ar_reject_reason``): the switch, TP > 1, CUDA, 1-128 rows, an
enabled single-node custom all-reduce of 2/4/6/8 ranks on the TP group
and a ``_sm70_C`` built with the kernel. Anything else, including
``VLLM_SM70_TOP1_CUSTOM_AR=0``, TP1, CPU, or a custom all-reduce that was
never initialised or is disabled, takes ``get_top_tokens`` and its
all-gather as before.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch

logger = logging.getLogger("vllm.sm70")

# Custom-AR pair exchange (csrc/custom_ar/top1_custom_ar.cu).
TOP1_CUSTOM_AR_SWITCH = "VLLM_SM70_TOP1_CUSTOM_AR"
KERNEL_NAMESPACE = "_sm70_C"
KERNEL_NAME = "custom_ar_top1_argmax"
# One CTA of 128 threads, one row each (fork: numel <= 256 FP32 values).
MAX_CUSTOM_AR_ROWS = 128
CUSTOM_AR_WORLD_SIZES = (2, 4, 6, 8)
# Attributes of upstream's CustomAllreduce the exchange reads.
_CA_ATTRS = ("_ptr", "disabled", "rank", "world_size", "buffer_ptrs", "max_size")

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


# ---------------------------------------------------------------------------
# Custom-AR pair exchange
# ---------------------------------------------------------------------------


@functools.cache
def _log_once(level: int, msg: str) -> None:
    logger.log(level, msg)


def custom_ar_switch_on() -> bool:
    from vllm_sm70 import envs

    return bool(envs.VLLM_SM70_TOP1_CUSTOM_AR)


def custom_ar_kernel_available() -> bool:
    """True if _sm70_C was built with csrc/custom_ar (VLLM_SM70_VLLM_CSRC_DIR)."""
    from vllm_sm70 import _ext

    if _ext.load() is None:
        return False
    try:
        return hasattr(getattr(torch.ops, KERNEL_NAMESPACE), KERNEL_NAME)
    except (AttributeError, RuntimeError):
        return False


def tp_custom_allreduce() -> Any:
    """Upstream's CustomAllreduce of the TP group, or None (no TP group, or
    custom all-reduce not initialised: disabled, unsupported, TP1)."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        communicator = get_tp_group().device_communicator
    except Exception:  # noqa: BLE001 (no TP group: TP1, CPU, tests)
        return None
    return getattr(communicator, "ca_comm", None)


def custom_ar_reject_reason(
    ca: Any, rows: int, tp_size: int, device: torch.device | None
) -> str | None:
    """Why the pair exchange must take the all-gather, or None.

    Every input is the same on all TP ranks (environment, configuration,
    batch rows, upstream's collective init), so either every rank enters the
    custom-AR kernel or every rank enters the all-gather. FORK
    ``_maybe_custom_top1_argmax`` + ``custom_top1_argmax`` checks, minus the
    capture branch (the U4 hook rejects capture before this).
    """
    if not custom_ar_switch_on():
        return "switch_off"
    if tp_size <= 1:
        return "tp1"
    if device is None or device.type != "cuda":
        return "non_cuda"
    if not 1 <= rows <= MAX_CUSTOM_AR_ROWS:
        return "rows"
    if ca is None or any(not hasattr(ca, a) for a in _CA_ATTRS):
        return "no_custom_ar"
    if ca.disabled or not ca._ptr:
        return "custom_ar_disabled"
    if ca.world_size != tp_size or ca.world_size not in CUSTOM_AR_WORLD_SIZES:
        return "world_size"
    # Real IPC peers, not the multi-node fallback that repeats the local
    # buffer (UP custom_all_reduce.py:309-311).
    ptrs = list(ca.buffer_ptrs)
    if len(ptrs) != ca.world_size or len(set(ptrs)) != len(ptrs):
        return "multi_node"
    if 2 * rows * 4 > ca.max_size:
        return "rows"
    if not custom_ar_kernel_available():
        return "kernel"
    return None


def local_pair(
    logits_processor: Any, lm_head: Any, hidden_states: torch.Tensor
) -> torch.Tensor:
    """This rank's [rows, 2] FP32 (max logit, global token id).

    The first half of upstream ``get_top_tokens`` (UP logits_processor.py:
    213-241), line for line: the same LM-head GEMM, soft cap, scale and
    padding mask, so the pair is the one get_top_tokens all-gathers.
    """
    logits = logits_processor._apply_head(lm_head, hidden_states, None)
    if logits_processor.soft_cap is not None:
        logits = (
            torch.tanh(logits / logits_processor.soft_cap) * logits_processor.soft_cap
        )
    if logits_processor.scale != 1.0:
        logits = logits * logits_processor.scale
    num_pad = lm_head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        logits[..., -num_pad:] = -float("inf")
    local_max_vals, local_max_indices = logits.max(dim=-1)
    global_indices = local_max_indices + lm_head.shard_indices.org_vocab_start_index
    return torch.stack([local_max_vals.float(), global_indices.float()], dim=-1)


def _kernel():
    return getattr(getattr(torch.ops, KERNEL_NAMESPACE), KERNEL_NAME)


def exchange_custom_ar(ca: Any, pair: torch.Tensor) -> torch.Tensor:
    """int64 [rows]: the cross-rank argmax of the pairs, same on every rank."""
    pair = pair.contiguous()
    out = torch.empty((pair.shape[0],), dtype=torch.int64, device=pair.device)
    _kernel()(ca._ptr, pair, out, ca.buffer_ptrs[ca.rank], ca.max_size)
    return out


def greedy_tokens(
    logits_processor: Any, lm_head: Any, hidden_states: torch.Tensor
) -> torch.Tensor:
    """int32 token ids [rows]: the argmax of the full-vocabulary logits."""
    tp_size = getattr(lm_head, "tp_size", 1)
    rows = hidden_states.shape[0]
    reason = custom_ar_reject_reason(None, rows, tp_size, hidden_states.device)
    if reason == "no_custom_ar":
        ca = tp_custom_allreduce()
        reason = custom_ar_reject_reason(ca, rows, tp_size, hidden_states.device)
        if reason is None:
            pair = local_pair(logits_processor, lm_head, hidden_states)
            try:
                top = exchange_custom_ar(ca, pair)
            except Exception as exc:  # noqa: BLE001 (fork: warn, then all-gather)
                _log_once(
                    logging.WARNING,
                    f"SM70 custom top1 argmax failed; falling back to all_gather: {exc}",
                )
            else:
                _log_once(logging.INFO, "SM70 custom top1 argmax path enabled.")
                return top.to(torch.int32)
    if reason not in (None, "switch_off", "tp1", "non_cuda", "rows"):
        _log_once(
            logging.INFO,
            f"SM70 custom top1 argmax unavailable ({reason}); "
            "greedy top-1 uses the all-gather.",
        )
    top = logits_processor.get_top_tokens(lm_head, hidden_states)
    return top.to(torch.int32)
