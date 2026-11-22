# SPDX-License-Identifier: Apache-2.0
"""Engine-config hook for Inkling2LoopSm70ForConditionalGeneration.

``vllm_sm70.models.inkling2_sm70.register()`` installs it as
``MODELS_CONFIG_MAP[ARCH]`` (a dict insert). UP calls it from
``VllmConfig.try_verify_and_update_config`` (UP config/vllm.py:1539).

* Refuses anything other than fp16 on cc 7.0, a speculative config and
  prompt embeddings (upstream models/inkling2/nvidia/model.py:556-559).
* Attention: the model owns its attention layers and KV caches, so no layer
  goes through the platform's backend selection (Flash-V100 or TRITON_ATTN).
  ``attention_config.backend`` stays unset and an explicit
  ``--attention-backend`` is refused.
* Logs the routes once per process.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger("vllm.sm70.inkling2")

ARCH = "Inkling2LoopSm70ForConditionalGeneration"

_logged = False


def _log_routes_once(visit_dump_dir: str | None) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    logger.info(
        "Inkling 2 SM70: fp16 on cc 7.0; model-owned attention and KV caches "
        "(neither FLASH_ATTN_V100 nor TRITON_ATTN); no speculative decoding, "
        "no prompt embeddings; per-visit state dump %s.",
        f"to {visit_dump_dir}" if visit_dump_dir else "off",
    )


class Inkling2Sm70ModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Raises ValueError for an engine this model can't serve."""
        import torch

        from vllm.platforms import current_platform
        from vllm_sm70.config import sm70_config

        capability = (
            current_platform.get_device_capability()
            if current_platform.is_cuda()
            else None
        )
        if capability is None or (capability.major, capability.minor) != (7, 0):
            raise ValueError(
                f"{ARCH} needs a compute capability 7.0 GPU (V100), got "
                f"{capability}. Drop --hf-overrides architectures to serve "
                "upstream's Inkling 2 model on other GPUs."
            )
        model_config = vllm_config.model_config
        if model_config.dtype != torch.float16:
            raise ValueError(
                f"{ARCH} needs --dtype half (V100 has no bf16), got "
                f"{model_config.dtype}."
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                f"{ARCH} does not support speculative decoding; drop "
                "--speculative-config."
            )
        if getattr(model_config, "enable_prompt_embeds", False):
            raise ValueError(
                f"{ARCH} does not support prompt embeddings; drop "
                "--enable-prompt-embeds."
            )
        backend = vllm_config.attention_config.backend
        if backend is not None:
            raise ValueError(
                f"{ARCH} owns its attention layers and KV caches and uses no "
                f"vLLM attention backend, got --attention-backend "
                f"{getattr(backend, 'name', backend)}; drop it."
            )
        _log_routes_once(sm70_config(vllm_config).inkling2.visit_dump_dir)
