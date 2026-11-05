# SPDX-License-Identifier: Apache-2.0
"""vllm.platform_plugins entry point and SM70 readiness checks.

`activate()` runs inside `vllm.platforms.resolve_current_platform_cls_qualname`
(UP platforms/__init__.py:240-292), i.e. while `current_platform` is still
unresolved. So this module must not import anything that reads
`vllm.platforms.current_platform` at import time (that re-enters resolution).
It only uses NVML, like upstream's cuda_platform_plugin
(UP platforms/__init__.py:57-84), and never initialises CUDA.

`Sm70CudaPlatform` lives in `vllm_sm70._cuda_platform` and is resolved
lazily via module `__getattr__`, because importing `vllm.platforms.cuda`
needs the compiled `vllm._C_stable_libtorch` (UP platforms/cuda.py:23).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

logger = logging.getLogger("vllm.sm70")

PLATFORM_QUALNAME = "vllm_sm70.platform.Sm70CudaPlatform"
SM70_CAPABILITY = (7, 0)


def _import_pynvml() -> Any:
    try:
        # vLLM's vendored copy, the same one its CUDA detection uses.
        from vllm.utils.import_utils import import_pynvml

        return import_pynvml()
    except ImportError:
        import pynvml  # type: ignore[import-not-found]

        return pynvml


def _visible_entries() -> list[str] | None:
    """CUDA_VISIBLE_DEVICES entries, or None if unset (all devices)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [e.strip() for e in raw.split(",") if e.strip()]


def _visible_capabilities(pynvml: Any) -> list[tuple[int, int]]:
    count = pynvml.nvmlDeviceGetCount()
    entries = _visible_entries()
    handles = []
    if entries is None:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    else:
        for entry in entries:
            if entry.lstrip("-").isdigit():
                index = int(entry)
                if index < 0 or index >= count:
                    break  # CUDA ignores this entry and everything after it
                handles.append(pynvml.nvmlDeviceGetHandleByIndex(index))
            else:
                handles.append(pynvml.nvmlDeviceGetHandleByUUID(entry))
    return [tuple(pynvml.nvmlDeviceGetCudaComputeCapability(h)) for h in handles]


@functools.cache
def all_visible_gpus_are_sm70() -> bool:
    """True iff at least one GPU is visible and every visible GPU is cc 7.0."""
    try:
        pynvml = _import_pynvml()
        pynvml.nvmlInit()
    except Exception as e:  # no driver, no NVML, CPU-only host
        logger.debug("vllm_sm70: NVML unavailable (%s); not activating", e)
        return False
    try:
        caps = _visible_capabilities(pynvml)
    except Exception as e:
        logger.debug("vllm_sm70: NVML query failed (%s); not activating", e)
        return False
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return bool(caps) and all(c == SM70_CAPABILITY for c in caps)


def activate() -> str | None:
    """vllm.platform_plugins entry point."""
    if all_visible_gpus_are_sm70():
        return PLATFORM_QUALNAME
    return None


@functools.cache
def is_sm70_ready() -> bool:
    """All visible GPUs are cc 7.0 and `_sm70_C` is loaded. No CUDA init."""
    from vllm_sm70 import _ext

    return all_visible_gpus_are_sm70() and _ext.is_loaded()


def __getattr__(name: str) -> Any:
    if name == "Sm70CudaPlatform":
        from vllm_sm70._cuda_platform import Sm70CudaPlatform

        return Sm70CudaPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
