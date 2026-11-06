# SPDX-License-Identifier: Apache-2.0
"""V1 runner hook for the top-k candidate sampler (K16 on patch U4).

Patch U4 (patches/0005) adds ``Platform.get_hidden_state_sampler()``. The V1
``GPUModelRunner`` reads it once in ``__init__`` and, before
``compute_logits``, calls

    try_sample(model, sample_hidden_states, sampling_metadata,
               scheduler_output, spec_decode_metadata)

A returned ``SamplerOutput`` replaces ``compute_logits`` and ``_sample``;
None keeps both. ``Sm70CudaPlatform.get_hidden_state_sampler`` returns a new
``Sm70HiddenStateSampler`` unless ``VLLM_SM70_FLINT_SAMPLER=0``.

This is the only runner-facing code of K16; the sampling itself
(vllm_sm70/sample.py) takes plain arguments, so a V2 hook (D2) can reuse it.

Port of the fork's runner gate (FORK v1/worker/gpu_model_runner.py:
7104-7125, :7698-7800 ``_can_use_topk_candidate_sampler``, ``:7828-7843``
``_non_argmax_logits_processors_inactive``) and the Flint model's opt-in
(FORK models/flint/model.py:570-577, :605-614). Differences:

* The runner calls us; there is no runner-side ``logits = None`` gate.
* The per-row top-k host copy comes from the runner's ``InputBatch``
  (``top_k_cpu``), since upstream's ``SamplingMetadata`` has no
  ``top_k_cpu``. The runner is bound in ``prepare()``.
* ``prepare()`` runs from ``Sm70CudaPlatform.kernel_warmup`` (patch U1),
  i.e. after the model is loaded and before CUDA graph capture. It creates
  the overflow counter (R4) and reads ``use_fp64_gumbel``. Until it has run,
  ``try_sample`` returns None.
* Models opt in with a true ``supports_sm70_topk_sampler`` attribute (the
  fork attached ``sample_topk_candidates`` to the Flint model; M10 sets the
  attribute on the Flint model).

Returns None (upstream path) for: no prepare(); a model that has not opted
in or has no ``lm_head``/``logits_processor``; spec decode; structured output
(the grammar bitmask needs logits); FP64 Gumbel noise (upstream
``use_fp64_gumbel``: the kernel reproduces FP32 noise only); a running CUDA
graph capture; thinking-budget requests; active non-argmax-invariant logits
processors (min-tokens, logit bias, custom); penalties (they need
``output_token_ids``, which async scheduling refreshes only inside
``_sample``, after this hook); and every batch ``sample.py`` does not admit.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

import torch

from vllm_sm70 import sample

logger = logging.getLogger("vllm.sm70")

# The attribute a model sets to opt in (M10 sets it on the Flint model).
MODEL_OPT_IN = "supports_sm70_topk_sampler"


def non_argmax_logits_processors_inactive(sampling_metadata: Any) -> bool:
    """FORK gpu_model_runner.py:7828-7843: only empty min-tokens / logit-bias
    processors may be among the non-argmax-invariant ones."""
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        min_toks = getattr(processor, "min_toks", None)
        if min_toks is not None:
            if min_toks:
                return False
            continue
        biases = getattr(processor, "biases", None)
        if biases is not None:
            if biases:
                return False
            continue
        return False
    return True


class Sm70HiddenStateSampler:
    """U4 hidden-state sampler: TP-local exact top-k (k <= 20) sampling."""

    def __init__(self) -> None:
        self._runner: weakref.ReferenceType[Any] | None = None
        self._device: torch.device | None = None
        self._fp64_gumbel = False
        self._rejected: set[str] = set()

    @property
    def prepared(self) -> bool:
        return self._runner is not None

    def prepare(self, runner: Any, vllm_config: Any) -> None:
        """Bind the V1 runner and create device state, before capture (R4)."""
        model_config = getattr(vllm_config, "model_config", None)
        self._fp64_gumbel = bool(getattr(model_config, "use_fp64_gumbel", False))
        device = torch.device(runner.device)
        if (
            device.type == "cuda"
            and sample._is_sm70(device.index)
            and sample.ops_available()
        ):
            sample.prepare(device)
        self._device = device
        self._runner = weakref.ref(runner)

    def _reject(self, reason: str) -> None:
        if reason not in self._rejected:
            self._rejected.add(reason)
            logger.info("SM70 top-k candidate sampler rejected: %s", reason)

    def reject_reason(
        self,
        model: Any,
        sample_hidden_states: torch.Tensor,
        sampling_metadata: Any,
        scheduler_output: Any,
        spec_decode_metadata: Any,
    ) -> str | None:
        """Why this step must take the upstream path, or None to sample here."""
        runner = self._runner() if self._runner is not None else None
        if runner is None:
            return "not_prepared"
        if not getattr(model, MODEL_OPT_IN, False):
            return "model"
        if getattr(model, "lm_head", None) is None or not hasattr(
            model, "logits_processor"
        ):
            return "model"
        if spec_decode_metadata is not None:
            return "spec_decode"
        if getattr(scheduler_output, "has_structured_output_requests", False):
            return "structured_output"
        if self._fp64_gumbel:
            return "fp64_gumbel"
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return "capturing"
        holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
        if holder is not None and holder.has_tracked_requests():
            return "thinking_budget"
        if not non_argmax_logits_processors_inactive(sampling_metadata):
            return "logits_processor"
        input_batch = runner.input_batch
        num_reqs = input_batch.num_reqs
        if sample_hidden_states.shape[0] != num_reqs:
            return "batch_size"
        top_k_cpu = input_batch.top_k_cpu[:num_reqs].tolist()
        return sample.admission_reject_reason(
            sampling_metadata, top_k_cpu, num_reqs, self._device
        )

    def try_sample(
        self,
        model: Any,
        sample_hidden_states: torch.Tensor,
        sampling_metadata: Any,
        scheduler_output: Any,
        spec_decode_metadata: Any,
    ) -> Any:
        reason = self.reject_reason(
            model,
            sample_hidden_states,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
        )
        if reason is not None:
            self._reject(reason)
            return None
        logits_processor = model.logits_processor
        sampled = sample.sample_lm_head_candidates(
            logits_processor,
            model.lm_head,
            sample_hidden_states,
            sampling_metadata,
            vocab_size=logits_processor.org_vocab_size,
        )
        if sampled is None:
            self._reject("shape")
            return None
        from vllm.v1.outputs import SamplerOutput

        return SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None
        )


def prepare_worker(worker: Any) -> None:
    """Called from Sm70CudaPlatform.kernel_warmup (U1): bind the runner's
    hidden-state sampler, if it is ours, before CUDA graph capture."""
    runner = getattr(worker, "model_runner", None)
    hook = getattr(runner, "_hidden_state_sampler", None)
    if isinstance(hook, Sm70HiddenStateSampler):
        hook.prepare(runner, getattr(worker, "vllm_config", None))
