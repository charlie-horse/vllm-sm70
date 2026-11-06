# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-config hook for FlintSm70ForCausalLM.

Port of charlie-horse/vLLM cf9492a ``vllm/model_executor/models/config.py``
:341-417 (``FlintModelConfig``). ``vllm_sm70.models.flint_sm70.register()``
installs it as ``MODELS_CONFIG_MAP["FlintSm70ForCausalLM"]``. That is a dict
insert, with no edit to UP 4ba787e models/config.py:1211. UP calls it from
``VllmConfig.try_verify_and_update_config`` (UP config/vllm.py:1539,
2735-2764), which runs before ``set_splitting_ops_for_v1`` (UP :2161).

v1 scope:

* Refuse anything other than fp16 on cc 7.0. UP's own Flint asserts bf16 and
  FlashInfer (UP models/flint/model.py:318, 321); this class is the SM70
  variant.
* Attention backend (fork :375-417, :276-305): unless the user picked one,
  Flash-V100 (K06, ``AttentionBackendEnum.CUSTOM``) when its sink route
  admits the engine (see ``flash_v100_sink_route_rejection``), else
  TRITON_ATTN, which supports sinks (UP triton_attn.py:387) on every compute
  capability (:405) but needs an FP16 KV cache on SM70 (fork :406-415).
  Kill switch: VLLM_SM70_FLASH_V100=0 -> TRITON_ATTN.
* The fork's TP2 all-reduce + RMSNorm fusion (fork :308-372) is deferred. It
  needs the fork-only ``pass_config.fuse_allreduce_add_rms_tp2``.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger("vllm.sm70.flint")

ARCH = "FlintSm70ForCausalLM"
_FP16_KV = ("auto", "float16", "half")

# ported from FORK vllm/model_executor/models/config.py:209-216. Head sizes
# whose generic Flash-V100 paged prefill and decode apply the learned sinks;
# the D256-specialized, XQA and FP8 kernels have no sink term.
FLASH_V100_SINK_HEAD_SIZES = (128,)
# The Flash-V100 paged kernels need a KV block size that is a multiple of 16.
FLASH_V100_BLOCK_MULTIPLE = 16


# ported from FORK vllm/model_executor/models/config.py:276-305
def flash_v100_sink_route_rejection(vllm_config: "VllmConfig") -> str | None:
    """Why the Flash-V100 sink route cannot serve this Flint engine, or None.

    Mirrors the fork's admission so an engine is never routed to a backend
    whose layers would then fail closed. The fork's
    ``attention_config.flash_v100.enabled`` maps to our kill switch, and its
    ``flash_attn_v100.SUPPORTS_SINKS`` to ``vllm_sm70.ops.fv100.supports_sinks()``
    (#80).
    Two checks beyond the fork (Kai/Lena's audit): the KV block size must be a
    multiple of 16, and CUSTOM must resolve to our Flash-V100 backend."""
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


# ported from FORK vllm/model_executor/models/config.py:341-417
class FlintSm70ModelConfig(VerifyAndUpdateConfig):
    # ported from FORK vllm/model_executor/models/config.py:342-347
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

    # ported from FORK vllm/model_executor/models/config.py:375-417
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
