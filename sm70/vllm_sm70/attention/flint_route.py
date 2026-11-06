# SPDX-License-Identifier: Apache-2.0
"""Flint on SM70: engine-wide attention backend choice for learned sinks.

Port of fork cf9492a vllm/model_executor/models/config.py:
  FLASH_V100_SINK_HEAD_SIZES        :209-215
  flash_v100_supports_sinks()       :218-229
  _flash_v100_sink_route_rejection  :276-305 -> flint_sink_route_rejection
  _select_sm70_attention_backend    :374-415 -> select_flint_attention_backend

Every Flint layer has learned sinks (fp32 [H] per layer). Flash-V100 applies
them on the D128 generic paged prefill and the D128 paged decode (grouped
two-launch, or the K14 one-launch fused kernel). The rejection below mirrors
what an engine needs so that no layer routed to Flash-V100 then falls back
or fails (FlashAttnV100Impl._unsupported_reason, FlashAttnV100Backend.
validate_configuration, which upstream applies to an explicit backend and
raises on). Anything else keeps TRITON_ATTN, which applies sinks too.

The caller (M5's FlintSm70ModelConfig) has already checked cc 7.0 and fp16.

Adaptations from the fork:
* The `flash_attn_v100.SUPPORTS_SINKS` flag is replaced by the presence of
  the K05 sink ops in _sm70_C (the fork computed the flag from the same two
  entry points, flash_attn_v100/__init__.py).
* `attention_config.flash_v100.enabled` is replaced by the kill switches
  VLLM_SM70_FLASH_V100 (K06, all layers) and VLLM_SM70_FLINT_FLASH_V100
  (this route only).
* FLASH_ATTN_V100 is AttentionBackendEnum.CUSTOM here (K06 registers our
  backend there); the CUSTOM registration is checked.
* New: KV block size must be a multiple of 16 (FlashAttnV100Backend.
  supports_block_size; upstream raises for an explicit backend otherwise).
* The fork only warned for an FP8 E4M3 KV cache on TRITON_ATTN; M5 raises
  (kept: TRITON_ATTN cannot compile FP8 E4M3 on SM70).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# A child of "vllm", so INFO lines reach the engine log (see #76).
logger = logging.getLogger("vllm.sm70.attention")

FLINT_KILL_SWITCH = "VLLM_SM70_FLINT_FLASH_V100"

FLASH_V100_SINK_HEAD_SIZES = (128,)
"""Head sizes whose Flash-V100 paged prefill and decode apply sinks (fork
:209). The D256 kernels (XQA decode, low-SMEM / BM32 prefill) have no sink
term; widen only with a paired kernel and quality check."""

# The sink ops checked by vllm_sm70.ops.fv100.supports_sinks() (for messages).
SINK_OPS = ("fv100_prefill_paged_sinks", "fv100_decode_paged_sinks")
FP16_KV_CACHE_DTYPES = ("auto", "float16")
_TRITON_FP16_KV = ("auto", "float16", "half")


def flash_v100_supports_sinks() -> bool:
    """fv100.supports_sinks(): both K05 sink ops in _sm70_C (fork :218-229)."""
    try:
        from vllm_sm70.ops import fv100

        return fv100.supports_sinks()
    except Exception:  # extension missing or unloadable: no sink route
        return False


def flint_flash_v100_enabled() -> bool:
    """Kill switch: VLLM_SM70_FLINT_FLASH_V100=0 keeps Flint on TRITON_ATTN."""
    return os.environ.get(FLINT_KILL_SWITCH) != "0"


def _head_dim(model_config: Any) -> int | None:
    head_dim = getattr(getattr(model_config, "hf_text_config", None), "head_dim", None)
    if head_dim is None and hasattr(model_config, "get_head_size"):
        head_dim = model_config.get_head_size()
    return head_dim


def _custom_is_ours() -> bool:
    from vllm_sm70.attention import custom_backend_path
    from vllm_sm70.attention.flash_v100 import BACKEND_PATH

    return custom_backend_path() == BACKEND_PATH


def flint_sink_route_rejection(vllm_config: VllmConfig) -> str | None:
    """Why the Flash-V100 sink route cannot serve this engine, or None.

    fork :276-305. Mirrors the per-layer admission so that an engine is
    never routed to a backend whose layers would then fail closed.
    """
    import torch

    from vllm_sm70.attention import KILL_SWITCH, flash_v100_enabled

    if not flash_v100_enabled():
        return f"{KILL_SWITCH}=0"
    if not flint_flash_v100_enabled():
        return f"{FLINT_KILL_SWITCH}=0"
    if not flash_v100_supports_sinks():
        return f"_sm70_C lacks the Flash-V100 sink ops {SINK_OPS}"
    if not _custom_is_ours():
        return "AttentionBackendEnum.CUSTOM is not the vllm_sm70 Flash-V100 backend"
    model_config = vllm_config.model_config
    if model_config.dtype != torch.float16:
        return f"Flash-V100 computes in float16, not {model_config.dtype}"
    head_dim = _head_dim(model_config)
    if head_dim not in FLASH_V100_SINK_HEAD_SIZES:
        return f"Flash-V100 has no sink route for head_dim={head_dim}"
    cache_config = vllm_config.cache_config
    cache_dtype = cache_config.cache_dtype
    if cache_dtype not in FP16_KV_CACHE_DTYPES:
        return (
            "the Flash-V100 sink route needs an FP16 KV cache, not "
            f"kv_cache_dtype={cache_dtype!r}"
        )
    block_size = getattr(cache_config, "block_size", None)
    if block_size is not None and block_size % 16 != 0:
        return f"Flash-V100 needs a KV block size that is a multiple of 16, not {block_size}"
    if getattr(vllm_config, "speculative_config", None) is not None:
        # fork rule kept: K07's verify rows do pass the sinks to the paged
        # decode/prefill (tests/attention/test_fv100_spec_cpu.py), but no GPU
        # check covers sinks under verification and Flint has no drafter.
        return "the Flash-V100 speculative verification routes are not validated with sinks"
    return None


def _fused_decode_state() -> str:
    try:
        from vllm_sm70.ops import fv100
    except Exception:
        return "K14 fused decode off: ops not loaded"
    if not fv100.has_op(fv100.FUSED_DECODE_OP):
        return "K14 fused decode off: not built"
    if not fv100.fused_decode_enabled():
        return f"K14 fused decode off: {fv100.FUSED_DECODE_ENV}=0"
    return "K14 fused decode on"


def select_flint_attention_backend(vllm_config: VllmConfig) -> None:
    """Pick Flint's SM70 attention backend (fork :374-415).

    An explicit attention_config.backend is left unchanged. Otherwise CUSTOM
    (Flash-V100) when flint_sink_route_rejection() is None, else TRITON_ATTN
    with the reason logged. TRITON_ATTN (chosen or explicit) needs an FP16 KV
    cache on SM70: ValueError otherwise (M5's rule).
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    attention_config = vllm_config.attention_config
    if attention_config.backend is None:
        reason = flint_sink_route_rejection(vllm_config)
        if reason is None:
            attention_config.backend = AttentionBackendEnum.CUSTOM
            logger.info(
                "Flint on SM70: using Flash-V100 (sinks; %s). "
                "%s=0 selects TRITON_ATTN.",
                _fused_decode_state(),
                FLINT_KILL_SWITCH,
            )
            return
        attention_config.backend = AttentionBackendEnum.TRITON_ATTN
        logger.info(
            "Flint on SM70: using TRITON_ATTN for the learned attention sinks "
            "because %s.",
            reason,
        )
    cache_dtype = vllm_config.cache_config.cache_dtype
    if (
        attention_config.backend == AttentionBackendEnum.TRITON_ATTN
        and cache_dtype not in _TRITON_FP16_KV
    ):
        raise ValueError(
            "Flint SM70 on TRITON_ATTN needs an FP16 KV cache, got "
            f"kv_cache_dtype={cache_dtype!r}; Triton can't compile FP8 E4M3 "
            "on SM70. Use --kv-cache-dtype auto."
        )
