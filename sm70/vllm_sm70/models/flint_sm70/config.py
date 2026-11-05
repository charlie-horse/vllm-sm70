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
* Attention backend: TRITON_ATTN unless the user picked one. It supports sinks
  (UP triton_attn.py:387) and every compute capability (:405), and needs an
  FP16 KV cache on SM70 (fork :406-415). Kai's Flash-V100 sink route
  (fork :375-399, :276-305) arrives with his backend.
* The fork's TP2 all-reduce + RMSNorm fusion (fork :308-372) is deferred. It
  needs the fork-only ``pass_config.fuse_allreduce_add_rms_tp2``.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

ARCH = "FlintSm70ForCausalLM"
_FP16_KV = ("auto", "float16", "half")


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
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attention_config = vllm_config.attention_config
        if attention_config.backend is None:
            attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info("Flint SM70: TRITON_ATTN applies the learned attention sinks.")
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
