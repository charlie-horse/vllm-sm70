# SPDX-License-Identifier: Apache-2.0
"""SM70 config classes for Inkling 2: upstream's (configs.py), LM head dtype
set to the model dtype.

Upstream defaults ``head_dtype`` to float32 (configs.py:220), which sends
logits through ``torch.mm(out_dtype=float32)``; vllm-sm70's samplers accept
only a head in the model dtype (vllm_sm70/sample.py). ``"model"`` is vLLM's
spelling of "the model dtype" (UP config/model.py ``_get_head_dtype``), so
the head follows ``--dtype``. An explicit ``head_dtype`` (config.json or
``--hf-overrides``) still wins.
"""

from __future__ import annotations

from typing import Any

from .configs import Inkling2LoopMMConfig, Inkling2MMConfig, Inkling2ModelConfig

HEAD_DTYPE = "model"
# Attribute the engine-config hook (config.py) sets on the model's hf_config:
# Sm70Config.inkling2.max_image_encoder_tokens, for the image processor.
IMAGE_ENCODER_TOKEN_CAP_ATTR = "sm70_max_image_encoder_tokens"

__all__ = [
    "CONFIG_CLASSES",
    "HEAD_DTYPE",
    "IMAGE_ENCODER_TOKEN_CAP_ATTR",
    "Inkling2LoopSm70MMConfig",
    "Inkling2Sm70MMConfig",
]


# The explicit __init__ of both classes is required: transformers generates
# one otherwise (see upstream's note at configs.py:233).
class Inkling2Sm70MMConfig(Inkling2MMConfig):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("head_dtype", HEAD_DTYPE)
        super().__init__(**kwargs)


class Inkling2LoopSm70MMConfig(Inkling2LoopMMConfig):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("head_dtype", HEAD_DTYPE)
        super().__init__(**kwargs)


# model_type -> config class, as upstream registers them in vLLM's
# _CONFIG_REGISTRY (UP c32251b transformers_utils/config.py:180-182).
CONFIG_CLASSES: dict[str, type] = {
    "inkling2": Inkling2Sm70MMConfig,
    "inkling2_loop": Inkling2LoopSm70MMConfig,
    "inkling2_model": Inkling2ModelConfig,
}
