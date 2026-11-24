# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-config hook for FlintSm70ForCausalLM.

``vllm_sm70.models.flint_sm70.register()`` installs it as
``MODELS_CONFIG_MAP["FlintSm70ForCausalLM"]``. Upstream calls it from
``VllmConfig.try_verify_and_update_config``, before
``set_splitting_ops_for_v1``.

* Refuses anything other than fp16 on cc 7.0.
* Attention backend: unless the user picked one, Flash-V100
  (``AttentionBackendEnum.CUSTOM``) when its sink route admits the engine
  (see ``flash_v100_sink_route_rejection``), else TRITON_ATTN, which
  supports sinks but needs an FP16 KV cache on SM70.
  Kill switch: VLLM_SM70_FLASH_V100=0 -> TRITON_ATTN.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger("vllm.sm70.flint")

ARCH = "FlintSm70ForCausalLM"
_FP16_KV = ("auto", "float16", "half")

# Head sizes whose generic Flash-V100 paged prefill and decode apply the learned sinks;
# the D256-specialized, XQA and FP8 kernels have no sink term.
FLASH_V100_SINK_HEAD_SIZES = (128,)
# The Flash-V100 paged kernels need a KV block size that is a multiple of 16.
FLASH_V100_BLOCK_MULTIPLE = 16


def flash_v100_sink_route_rejection(vllm_config: "VllmConfig") -> str | None:
    """Why the Flash-V100 sink route cannot serve this Flint engine, or None.

    An engine is never routed to a backend whose layers would then fail
    closed. Checks the kill switch, the sink ops
    (``vllm_sm70.ops.fv100.supports_sinks()``), that CUSTOM resolves to our
    Flash-V100 backend, fp16, the head size, an FP16 KV cache, a KV block
    size that is a multiple of 16, and no speculative decoding."""
    import torch

    from vllm_sm70 import attention
    from vllm_sm70.ops import fv100

    if not attention.flash_v100_enabled():
        return f"{attention.KILL_SWITCH}=0"
    if not attention._ops_available():
        return "the Flash-V100 ops are missing from _sm70_C"
    if not fv100.supports_sinks():
        return "the Flash-V100 sink ops are missing from _sm70_C"
    from vllm_sm70.attention.flash_v100 import BACKEND_PATH

    if attention.custom_backend_path() != BACKEND_PATH:
        return (
            "AttentionBackendEnum.CUSTOM is overridden to "
            f"{attention.custom_backend_path()}"
        )
    model_config = vllm_config.model_config
    if model_config.dtype != torch.float16:
        return f"Flash-V100 computes in float16, not {model_config.dtype}"
    head_dim = getattr(model_config.hf_text_config, "head_dim", None)
    if head_dim not in FLASH_V100_SINK_HEAD_SIZES:
        return f"Flash-V100 has no sink route for head_dim={head_dim}"
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype not in _FP16_KV:
        return (
            "the Flash-V100 sink route needs an FP16 KV cache, not "
            f"kv_cache_dtype={cache_dtype!r}"
        )
    block_size = getattr(vllm_config.cache_config, "block_size", None)
    if block_size is not None and block_size % FLASH_V100_BLOCK_MULTIPLE:
        return (
            f"the Flash-V100 paged kernels need a KV block size that is a "
            f"multiple of {FLASH_V100_BLOCK_MULTIPLE}, not {block_size}"
        )
    if vllm_config.speculative_config is not None:
        return "the Flash-V100 speculative verification routes do not apply sinks"
    return None


class FlintSm70ModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Raises ValueError unless the device is cc 7.0 and dtype is fp16."""
        import torch

        from vllm.platforms import current_platform

        capability = (
            current_platform.get_device_capability()
            if current_platform.is_cuda()
            else None
        )
        if capability is None or (capability.major, capability.minor) != (7, 0):
            raise ValueError(
                f"{ARCH} needs a compute capability 7.0 GPU (V100), got "
                f"{capability}. Drop --hf-overrides architectures to serve "
                "upstream FlintForCausalLM on other GPUs."
            )
        dtype = vllm_config.model_config.dtype
        if dtype != torch.float16:
            raise ValueError(
                f"{ARCH} needs --dtype half (V100 has no bf16), got {dtype}."
            )
        FlintSm70ModelConfig._select_attention_backend(vllm_config)

    @staticmethod
    def _select_attention_backend(vllm_config: "VllmConfig") -> None:
        """Pick an SM70 backend that applies Flint's learned sinks: Flash-V100
        when its sink route admits the engine, else TRITON_ATTN. An explicitly
        chosen backend is left unchanged."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attention_config = vllm_config.attention_config
        if attention_config.backend is None:
            reason = flash_v100_sink_route_rejection(vllm_config)
            if reason is None:
                attention_config.backend = AttentionBackendEnum.CUSTOM
                logger.info(
                    "Flint SM70: using FLASH_ATTN_V100 (CUSTOM), whose generic "
                    "D128 paged prefill and decode apply the learned attention "
                    "sinks (VLLM_SM70_FLASH_V100=0 selects TRITON_ATTN)."
                )
                return
            attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "Flint SM70: using TRITON_ATTN for the learned attention sinks "
                "because %s.",
                reason,
            )
        cache_dtype = vllm_config.cache_config.cache_dtype
        if (
            attention_config.backend == AttentionBackendEnum.TRITON_ATTN
            and cache_dtype not in _FP16_KV
        ):
            raise ValueError(
                f"Flint SM70 on TRITON_ATTN needs an FP16 KV cache, got "
                f"kv_cache_dtype={cache_dtype!r}; Triton can't compile FP8 E4M3 "
                "on SM70. Use --kv-cache-dtype auto."
            )
