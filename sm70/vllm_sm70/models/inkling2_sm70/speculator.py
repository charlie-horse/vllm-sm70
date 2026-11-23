# SPDX-License-Identifier: Apache-2.0
"""DSpark speculator for the Inkling 2 draft on SM70 (dspark.py).

The port of upstream's ``Inkling2DSparkSpeculator`` (vLLM c32251b
models/inkling2/speculator.py) onto the v0.33.0 DSpark speculator. It stores
the raw target features (each draft layer mixes them itself) and runs the
recurrent Markov head step by step. Draft logits are fp32 for greedy and
probabilistic drafts alike; Sm70CudaPlatform.get_speculator_cls (U6)
returns this class (draft.py). A step that verifies no drafts runs the
inherited propose with num_speculative_tokens 0, which stores the context
K/V only; f_rnn is recomputed at the next drafting step.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

__all__ = ["Inkling2DSparkSm70Speculator"]


class Inkling2DSparkSm70Speculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        if self.sample_from_anchor or self._draft_topk is not None:
            raise ValueError(
                "vllm_sm70: the Inkling 2 draft needs sample_from_anchor=False "
                "and no dspark_draft_topk."
            )
        if self.enable_adaptive_verification:
            raise ValueError(
                "vllm_sm70: the Inkling 2 draft has no confidence head for "
                "adaptive verification."
            )
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: features are stored eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor, also inside a FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )

    def init_cudagraph_manager(self, cudagraph_mode: Any) -> None:
        """Every draft step runs the whole block, also under a dynamic
        speculative-decoding schedule (spec_policy.py), whose tiers v0.33's
        graph manager would capture as draft query lengths
        (UP v1/worker/gpu/cudagraph_utils.py:276-296): it is built without
        the schedule."""
        spec = self.vllm_config.speculative_config
        schedule = spec.num_speculative_tokens_per_batch_size
        spec.num_speculative_tokens_per_batch_size = None
        try:
            super().init_cudagraph_manager(cudagraph_mode)
        finally:
            spec.num_speculative_tokens_per_batch_size = schedule

    def draft_logits_spec(self, vllm_config: Any) -> tuple[torch.dtype, float]:
        return torch.float32, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        """f_rnn from each request's last accepted token; no host sync."""
        assert num_rejected is not None
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        head = self.model.model.markov_head
        context = head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n_spec):
            state, bias = self.model.markov_step(state, prev, context[i])
            logits = self.model.markov_logits(base[:, i], bias)
            prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
