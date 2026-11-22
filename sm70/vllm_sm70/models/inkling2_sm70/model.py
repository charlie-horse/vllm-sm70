# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (looped) on SM70: the registered model class.

    Inkling2LoopSm70ForConditionalGeneration
      model            Inkling2Sm70LoopModel (loop.py): embedding, layer
                       visits, final norm; owns the caches of every visit
      lm_head          Inkling2Sm70LMHead: the unpadded vocabulary rows
      logits_processor Inkling2Sm70LogitsProcessor: head GEMM in the model
                       dtype, cast to fp32, then the soft cap
      visual           Inkling2VisionEncoder (vision.py), or None when the
                       image limit per prompt is 0

Weights load through weights.py (parameters), expert_weights.py (experts,
into each layer's ``mlp``) and the encoder's own loader. Each TP rank builds
this class and loads its own shards: a slice of the attention heads and of
the expert intermediate units; embedding, norms, convs, routers, the LM head
and the vision encoder are whole on every rank.

Images: ``embed_multimodal`` returns one ``[rows * cols, H]`` tensor per
image; ``embed_input_ids`` writes them over the image-token rows of the
normed text embeddings, so image rows skip ``embed_norm``. An image over
``sm70.inkling2.max_image_encoder_tokens`` is refused (the processor refuses
it first), and the encoder's attention takes its queries in chunks sized so
that the math SDPA kernel stays inside ``VISION_ATTENTION_BUDGET`` too.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY

from .expert_weights import Inkling2ExpertLoader
from .fp16_audit import walk_process_weights_after_loading
from .loop import Inkling2Sm70LoopModel, PartialAudit, VisitDump
from .processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from .vision import Inkling2VisionEncoder

logger = init_logger("vllm.sm70.inkling2")

__all__ = [
    "Inkling2LoopSm70ForConditionalGeneration",
    "Inkling2Sm70LMHead",
    "Inkling2Sm70LogitsProcessor",
    "vision_query_chunk",
]

# Bytes the math SDPA kernel takes per (query, key, head) score in fp16:
# 5.59 GiB for 6401 tokens and 16 heads on a V100. The memory-efficient
# kernel torch picks there takes 0.012 GiB.
MATH_SDPA_BYTES_PER_SCORE = 10
# Bound on one vision attention call if torch falls to the math kernel.
VISION_ATTENTION_BUDGET = 1 << 30
MAX_VISION_QUERY_CHUNK = 1024


def vision_query_chunk(max_encoder_tokens: int, num_heads: int) -> int:
    """Queries per vision attention call that keep the math SDPA kernel
    inside VISION_ATTENTION_BUDGET for an image of ``max_encoder_tokens``."""
    per_query = max_encoder_tokens * num_heads * MATH_SDPA_BYTES_PER_SCORE
    return max(1, min(MAX_VISION_QUERY_CHUNK, VISION_ATTENTION_BUDGET // per_query))


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


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2LoopSm70ForConditionalGeneration(nn.Module, SupportsMultiModal):
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
        inkling2 = sm70_config(vllm_config).inkling2
        if inkling2.visit_dump_dir:
            self.model.visit_hook = VisitDump(inkling2.visit_dump_dir, tp_rank)
            self.model.partial_audit = PartialAudit(
                len(self.model.visits), inkling2.visit_dump_dir, tp_rank
            )

        vision_config = getattr(config, "vision_config", None)
        mm_config = getattr(model_config, "multimodal_config", None)
        self.visual: Inkling2VisionEncoder | None = None
        self.max_image_tokens = 0
        if (
            vision_config is not None
            and mm_config is not None
            and mm_config.get_limit_per_prompt("image") > 0
        ):
            cap = inkling2.max_image_encoder_tokens
            self.visual = Inkling2VisionEncoder(
                vision_config,
                text_config.hidden_size,
                dtype=dtype,
                attn_query_chunk=vision_query_chunk(
                    cap, vision_config["num_attention_heads"]
                ),
            )
            self.max_image_tokens = (cap - 1) // self.visual.merge_size**2
        self.model.weight_map.with_visual = self.visual is not None

    def get_language_model(self) -> nn.Module:
        return self

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """One [rows * cols, H] tensor per image, in prompt order."""
        pixels = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if pixels is None or grid_thw is None:
            return ()
        assert isinstance(pixels, torch.Tensor)
        assert isinstance(grid_thw, torch.Tensor)
        if self.visual is None:
            raise ValueError(
                "vllm_sm70: Inkling 2 was loaded without its vision encoder "
                "(image limit per prompt 0) and cannot embed images."
            )
        sizes = grid_thw.prod(-1).tolist()
        if sizes and max(sizes) > self.max_image_tokens:
            raise ValueError(
                f"vllm_sm70: an Inkling 2 image has {max(sizes)} image tokens, "
                f"above the limit of {self.max_image_tokens} "
                "(sm70.inkling2.max_image_encoder_tokens)."
            )
        return self.visual(pixels, grid_thw).split(sizes)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normed token embeddings [T, H]; image features replace the rows
        of ``is_multimodal`` as they are, without ``embed_norm``."""
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError(
                "vllm_sm70: Inkling 2 needs is_multimodal with image embeddings."
            )
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

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
        audit = self.model.weight_map.audit
        experts = Inkling2ExpertLoader(
            self.text_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            formats=self.expert_formats,
            sink=lambda layer: layers[layer.layer].mlp.load_expert_layer(
                layer, range_check=audit.check
            ),
        )
        visual: list[tuple[str, torch.Tensor]] = []

        visual_dtypes = (
            {}
            if self.visual is None
            else {n: p.dtype for n, p in self.visual.named_parameters()}
        )

        def keep_visual(name: str, tensor: torch.Tensor) -> None:
            if visual_dtypes.get(name) == torch.float16:
                audit.check(f"model.visual.{name}", tensor)
            visual.append((name, tensor))

        loaded = self.model.weight_map.load_into(
            self,
            weights,
            expert_loader=experts,
            visual_loader=None if self.visual is None else keep_visual,
        )
        experts.finish()
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual)}
        logger.info("Inkling 2 rank %d %s", self.tp_rank, audit.summary())
        return loaded

    def process_weights_after_loading(self) -> None:
        """vLLM calls this on the top-level model only; it reaches the
        submodules' hooks."""
        walk_process_weights_after_loading(self)
