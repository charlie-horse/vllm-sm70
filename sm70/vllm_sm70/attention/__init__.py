# SPDX-License-Identifier: Apache-2.0
"""SM70 attention backends (Flash-V100, K05-K07).

register() points AttentionBackendEnum.CUSTOM at the Flash-V100 backend.
Selection happens in Sm70CudaPlatform.get_attn_backend_cls (select_backend
below); an explicit --attention-backend always wins. Flint sets the backend
engine-wide from its model config hook instead: see flint_route.py
(flint_sink_route_rejection, select_flint_attention_backend).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from vllm_sm70.attention.flash_v100 import BACKEND_PATH

logger = logging.getLogger("vllm.sm70")

KILL_SWITCH = "VLLM_SM70_FLASH_V100"
SUPPORTED_HEAD_SIZES = (128, 256)

_registered = False
_logged_choice: set[str] = set()


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent."""
    global _registered
    if _registered:
        return
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_PATH)
    _registered = True


def flash_v100_enabled() -> bool:
    """Kill switch: VLLM_SM70_FLASH_V100=0 -> upstream (TRITON_ATTN)."""
    return os.environ.get(KILL_SWITCH) != "0"


def _ops_available() -> bool:
    try:
        from vllm_sm70.ops import fv100

        return fv100.available()
    except Exception:
        return False


def flash_v100_rejection(
    selected_backend: Any, attn_selector_config: Any
) -> str | None:
    """Why Flash-V100 is not chosen for this layer, or None to choose it.

    (a) no backend selected explicitly, (b) kill switch not "0", (c) head
    size 128/256, (d) fp16, (e) the K05 ops are in the loaded _sm70_C; then
    the backend's own validate_configuration (KV dtype, MLA, sparse, ...).
    """
    if selected_backend is not None:
        return f"backend {getattr(selected_backend, 'name', selected_backend)} selected"
    if not flash_v100_enabled():
        return f"{KILL_SWITCH}=0"
    head_size = getattr(attn_selector_config, "head_size", None)
    if head_size not in SUPPORTED_HEAD_SIZES:
        return f"head size {head_size}"
    import torch

    dtype = getattr(attn_selector_config, "dtype", None)
    if dtype != torch.float16:
        return f"dtype {dtype}"
    if not _ops_available():
        return "Flash-V100 ops missing from _sm70_C"
    return None


def custom_backend_path() -> str:
    """CUSTOM's class path (respects overrides); ours if CUSTOM is unset."""
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        return AttentionBackendEnum.CUSTOM.get_path()
    except Exception:
        return BACKEND_PATH


def validate(attn_selector_config: Any, device_capability: Any) -> list[str]:
    from vllm_sm70.attention.flash_v100.backend import FlashAttnV100Backend

    return FlashAttnV100Backend.validate_configuration(
        device_capability=device_capability, **attn_selector_config._asdict()
    )


def log_choice_once(choice: str, detail: str) -> None:
    key = f"{choice}:{detail}"
    if key not in _logged_choice:
        _logged_choice.add(key)
        logger.info("vllm_sm70: attention backend %s (%s)", choice, detail)
