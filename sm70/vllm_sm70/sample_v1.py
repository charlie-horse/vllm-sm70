# SPDX-License-Identifier: Apache-2.0
"""V1 runner hook for the top-k candidate sampler and the greedy top-1 path.

patches/0005 adds ``Platform.get_hidden_state_sampler()``. The V1
``GPUModelRunner`` reads it once in ``__init__`` and, before
``compute_logits``, calls ``try_sample(model, sample_hidden_states,
sampling_metadata, scheduler_output, spec_decode_metadata)``. A returned
``SamplerOutput`` replaces ``compute_logits`` and ``_sample``; None keeps
both. Per step: an all-greedy batch takes vocab-parallel top-1
(sample_greedy.py) when the greedy fast path is on; otherwise the top-k
candidate sampler (sample.py) when VLLM_SM70_FLINT_SAMPLER is on.

``prepare()`` runs from ``Sm70CudaPlatform.kernel_warmup``, before CUDA
graph capture; until then ``try_sample`` returns None. Models opt in with a
true ``supports_sm70_topk_sampler`` attribute. ``reject_reason`` lists the
cases that take the upstream path."""

from __future__ import annotations

import logging
import weakref
from typing import Any

import torch

from vllm_sm70 import sample, sample_greedy

logger = logging.getLogger("vllm.sm70")

# The attribute a model sets to opt in (the Flint model sets it).
MODEL_OPT_IN = "supports_sm70_topk_sampler"


def non_argmax_logits_processors_inactive(sampling_metadata: Any) -> bool:
    """Only empty min-tokens / logit-bias processors may be among the
    non-argmax-invariant ones."""
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


TOPK = "top-k candidate sampler"
GREEDY = "greedy top-1 fast path"


class Sm70HiddenStateSampler:
    """Hidden-state sampler: TP-local exact top-k (k <= 20) sampling, and
    vocab-parallel top-1 for all-greedy batches."""

    def __init__(self, *, topk: bool = True, greedy: bool = True) -> None:
        self._runner: weakref.ReferenceType[Any] | None = None
        self._device: torch.device | None = None
        self._fp64_gumbel = False
        self._topk = topk
        # Refined from Sm70Config.greedy_token_fastpath in prepare().
        self._greedy = greedy
        self._rejected: set[tuple[str, str]] = set()
        self._greedy_logged = False

    @property
    def prepared(self) -> bool:
        return self._runner is not None

    def prepare(self, runner: Any, vllm_config: Any) -> None:
        """Bind the V1 runner and create device state, before capture."""
        from vllm_sm70.config import sm70_config

        model_config = getattr(vllm_config, "model_config", None)
        self._fp64_gumbel = bool(getattr(model_config, "use_fp64_gumbel", False))
        self._greedy = (
            self._greedy and sm70_config(vllm_config).greedy_token_fastpath
        )
        device = torch.device(runner.device)
        if (
            self._topk
            and device.type == "cuda"
            and sample._is_sm70(device.index)
            and sample.ops_available()
        ):
            sample.prepare(device)
        self._device = device
        self._runner = weakref.ref(runner)

    @property
    def greedy_enabled(self) -> bool:
        return self._greedy

    def route(self, sampling_metadata: Any) -> str | None:
        """GREEDY for an all-greedy batch with the fast path on, else TOPK
        with the top-k sampler on, else None (nothing to do: upstream)."""
        if self._greedy and getattr(sampling_metadata, "all_greedy", False):
            return GREEDY
        return TOPK if self._topk else None

    def _reject(self, route: str, reason: str) -> None:
        if (route, reason) not in self._rejected:
            self._rejected.add((route, reason))
            logger.info("SM70 %s rejected: %s", route, reason)

    def reject_reason(
        self,
        model: Any,
        sample_hidden_states: torch.Tensor,
        sampling_metadata: Any,
        scheduler_output: Any,
        spec_decode_metadata: Any,
    ) -> str | None:
        """Why this step must take the upstream path, or None to sample here
        (on the route ``route(sampling_metadata)`` picks)."""
        route = self.route(sampling_metadata)
        if route is None:
            return "disabled"
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
        if route == TOPK and self._fp64_gumbel:
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
        if route == GREEDY:
            return (
                sample_greedy.metadata_reject_reason(sampling_metadata)
                or sample_greedy.device_reject_reason(self._device)
                or sample_greedy.lm_head_reject_reason(
                    model.logits_processor, model.lm_head
                )
            )
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
        route = self.route(sampling_metadata)
        if reason is not None:
            if route is not None:
                self._reject(route, reason)
            return None
        from vllm.v1.outputs import SamplerOutput

        logits_processor = model.logits_processor
        if route == GREEDY:
            tokens = sample_greedy.greedy_tokens(
                logits_processor, model.lm_head, sample_hidden_states
            )
            if not self._greedy_logged:
                self._greedy_logged = True
                logger.info("SM70 greedy top-1 fast path enabled.")
            return SamplerOutput(
                sampled_token_ids=tokens.unsqueeze(-1), logprobs_tensors=None
            )
        sampled = sample.sample_lm_head_candidates(
            logits_processor,
            model.lm_head,
            sample_hidden_states,
            sampling_metadata,
            vocab_size=logits_processor.org_vocab_size,
        )
        if sampled is None:
            self._reject(TOPK, "shape")
            return None
        return SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None
        )


def prepare_worker(worker: Any) -> None:
    """Called from Sm70CudaPlatform.kernel_warmup: bind the runner's
    hidden-state sampler, if it is ours, before CUDA graph capture."""
    runner = getattr(worker, "model_runner", None)
    hook = getattr(runner, "_hidden_state_sampler", None)
    if isinstance(hook, Sm70HiddenStateSampler):
        hook.prepare(runner, getattr(worker, "vllm_config", None))
