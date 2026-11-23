# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (looped) on SM70: the registered model class.

    Inkling2LoopSm70ForConditionalGeneration
      model            Inkling2Sm70LoopModel (loop.py): embedding, layer
                       visits, final norm; owns the caches of every visit
      lm_head          Inkling2Sm70LMHead: the unpadded vocabulary rows, or
                       this rank's part of them (weights.py ``head_sharded``)
      logits_processor Inkling2Sm70LogitsProcessor: head GEMM (a GEMV of
                       2 to 8 rows, gemv.py) in the model dtype,
                       all-gather of a sharded head, cast to fp32, then
                       the soft cap; greedy tokens of a batch the
                       V1 sampler hook admits without the logits (greedy.py)
      visual           Inkling2VisionEncoder (vision.py), or None when the
                       image limit per prompt is 0

Weights load through weights.py (parameters), expert_weights.py (experts,
into each layer's ``mlp``) and the encoder's own loader. Each TP rank builds
this class and loads its own shards: a slice of the attention heads and of
the expert intermediate units and, when sharded, of the LM head and the
token embedding rows; norms, convs, routers and the vision encoder are
whole on every rank.

DSpark draft features (``SupportsEagle3``): with aux layers set, ``forward``
returns ``(final, [input of each listed visit..., final])``. A visit input
is the AttnRes partial block, upstream's ``stack[num_blocks]``.

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
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY

from . import act_quant, gemv, greedy
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
from .weights import pack_parameters

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
    """``weight`` [rows, H] in the model dtype: the unpadded vocabulary, or
    at ``tp_size`` > 1 rank ``tp_rank``'s equal share of it. The vocabulary
    attributes are those of a ``VocabParallelEmbedding`` without padding or
    added vocabulary, which the logits processor and the greedy sampler
    read."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size % tp_size:
            raise ValueError(
                f"vllm_sm70: an LM head of {vocab_size} rows does not split "
                f"over {tp_size} ranks."
            )
        rows = vocab_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(rows, hidden_size, dtype=dtype), requires_grad=False
        )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.org_vocab_size = vocab_size
        self.org_vocab_size_padded = vocab_size
        self.num_embeddings_per_partition = rows
        self.num_added_embeddings = 0
        self.shard_indices = SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=tp_rank * rows
        )


def _head_logits(
    lm_head: Any, hidden_states: torch.Tensor, embedding_bias: torch.Tensor | None
) -> torch.Tensor:
    """This rank's logits in the model dtype (gemv.py from 2 rows)."""
    logits = gemv.linear(
        hidden_states.to(lm_head.weight.dtype),
        lm_head.weight,
        min_rows=gemv.HEAD_MIN_ROWS,
    )
    if embedding_bias is not None:
        logits = logits + embedding_bias
    return logits


class Inkling2Sm70LogitsProcessor(LogitsProcessor):
    """The head GEMM runs in the model dtype and its result is cast to fp32,
    so the soft cap of the base class applies to fp32 logits. A sharded
    head's logits are all-gathered in the model dtype, before the cast."""

    def _apply_head(
        self,
        lm_head: Any,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _head_logits(lm_head, hidden_states, embedding_bias).float()

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: Any,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        logits = _head_logits(lm_head, hidden_states, embedding_bias)
        if lm_head.tp_size > 1 and not skip_gather:
            logits = self._gather_logits(logits)
            if logits is None:
                return None
        return logits.float()

    def get_top_tokens(
        self,
        lm_head: Any,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """int64 [rows]: the greedy tokens (greedy.py)."""
        if not greedy.on_kernels(self, lm_head, hidden_states):
            return super().get_top_tokens(lm_head, hidden_states, embedding_bias)
        logits = _head_logits(lm_head, hidden_states, embedding_bias)
        return greedy.top_tokens(logits, lm_head, self.soft_cap)


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2LoopSm70ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsEagle3
):
    @property
    def supports_sm70_topk_sampler(self) -> bool:
        """The V1 sampler hook's opt-in (greedy.py)."""
        return greedy.enabled()

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
        head_tp = tp_size if self.model.weight_map.head_sharded else 1
        self.lm_head = Inkling2Sm70LMHead(
            vocab_size,
            text_config.hidden_size,
            dtype,
            tp_size=head_tp,
            tp_rank=tp_rank if head_tp > 1 else 0,
        )
        self.logits_processor = Inkling2Sm70LogitsProcessor(
            vocab_size,
            org_vocab_size=vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )
        inkling2 = sm70_config(vllm_config).inkling2
        act_quant.install(
            self.model.layers, self.expert_formats, inkling2.expert_act_fake_quant
        )
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
        if self.model.merged:
            pack_parameters(self, self.model.weight_map.packs, copy=False)

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
        """Layer visits (loop passes counted) whose input is returned; an id
        past the last visit adds nothing, the final state is always last."""
        self.model.aux_hidden_state_layers = tuple(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """The checkpoint's ``mtp_config.target_layer_ids``."""
        mtp_config = getattr(self.config, "mtp_config", None) or {}
        layers = tuple(mtp_config.get("target_layer_ids", ()))
        if not layers:
            raise ValueError(
                "vllm_sm70: this Inkling 2 checkpoint has no "
                "mtp_config.target_layer_ids to capture."
            )
        return layers

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
        submodules' hooks. Packs a move to another device split apart again
        (weights.py ``pack_parameters``)."""
        if self.model.merged:
            moved = pack_parameters(self, self.model.weight_map.packs)
            if moved:
                logger.info(
                    "Inkling 2 rank %d: repacked %d packs", self.tp_rank, len(moved)
                )
        walk_process_weights_after_loading(self)
