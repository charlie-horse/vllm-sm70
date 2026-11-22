# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (looped) on SM70: the registered model class, text path.

    Inkling2LoopSm70ForConditionalGeneration
      model            Inkling2Sm70LoopModel (loop.py): embedding, layer
                       visits, final norm; owns the caches of every visit
      lm_head          Inkling2Sm70LMHead: the unpadded vocabulary rows
      logits_processor Inkling2Sm70LogitsProcessor: head GEMM in the model
                       dtype, cast to fp32, then the soft cap

Weights load through weights.py (parameters) and expert_weights.py (experts,
into each layer's ``mlp``). Each TP rank builds this class and loads its own
shards: a slice of the attention heads and of the expert intermediate units;
embedding, norms, convs, routers and the LM head are whole on every rank.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .expert_weights import Inkling2ExpertLoader
from .loop import Inkling2Sm70LoopModel, PartialAudit, VisitDump

__all__ = [
    "Inkling2LoopSm70ForConditionalGeneration",
    "Inkling2Sm70LMHead",
    "Inkling2Sm70LogitsProcessor",
]


class Inkling2Sm70LMHead(nn.Module):
    """``weight`` [unpadded vocab, H] in the model dtype, whole on every
    rank. The vocabulary attributes are those of an unsharded
    ``VocabParallelEmbedding``, which the greedy sampler reads."""

    def __init__(self, vocab_size: int, hidden_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype), requires_grad=False
        )
        self.tp_size = 1
        self.org_vocab_size = vocab_size
        self.org_vocab_size_padded = vocab_size
        self.num_embeddings_per_partition = vocab_size
        self.num_added_embeddings = 0
        self.shard_indices = SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=0
        )


class Inkling2Sm70LogitsProcessor(LogitsProcessor):
    """The head GEMM runs in the model dtype and its result is cast to fp32,
    so the soft cap of the base class applies to fp32 logits."""

    def _apply_head(
        self,
        lm_head: Any,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = F.linear(hidden_states.to(lm_head.weight.dtype), lm_head.weight)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits.float()


class Inkling2LoopSm70ForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        from vllm_sm70.config import sm70_config

        from .quant_config import load_expert_formats

        model_config = vllm_config.model_config
        config = model_config.hf_config
        text_config = config.get_text_config()
        dtype = model_config.dtype
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1:
            raise ValueError("vllm_sm70: Inkling 2 does not support pipeline parallel.")
        if model_config.head_dtype != dtype:
            raise ValueError(
                f"vllm_sm70: Inkling 2 needs head_dtype equal to the model dtype "
                f"{dtype}, got {model_config.head_dtype}."
            )
        self.config = config
        self.text_config = text_config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.expert_formats = load_expert_formats(
            model_config.model, getattr(model_config, "revision", None)
        )
        self.model = Inkling2Sm70LoopModel(
            text_config,
            vllm_config=vllm_config,
            prefix=f"{prefix}model",
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype=dtype,
        )
        vocab_size = int(text_config.unpadded_vocab_size)
        self.lm_head = Inkling2Sm70LMHead(vocab_size, text_config.hidden_size, dtype)
        self.logits_processor = Inkling2Sm70LogitsProcessor(
            vocab_size,
            org_vocab_size=vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )
        dump_dir = sm70_config(vllm_config).inkling2.visit_dump_dir
        if dump_dir:
            self.model.visit_hook = VisitDump(dump_dir, tp_rank)
            self.model.partial_audit = PartialAudit(
                len(self.model.visits), dump_dir, tp_rank
            )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """fp32 logits [T, unpadded vocab], soft-capped."""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        layers = self.model.layers
        experts = Inkling2ExpertLoader(
            self.text_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            formats=self.expert_formats,
            sink=lambda layer: layers[layer.layer].mlp.load_expert_layer(layer),
        )
        loaded = self.model.weight_map.load_into(self, weights, expert_loader=experts)
        experts.finish()
        return loaded
