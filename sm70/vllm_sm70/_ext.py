# SPDX-License-Identifier: Apache-2.0
"""Loading and fingerprinting of the `vllm_sm70._sm70_C` extension."""

from __future__ import annotations

import functools
import hashlib
import importlib
import logging
from types import ModuleType

logger = logging.getLogger("vllm.sm70")


@functools.cache
def load() -> ModuleType | None:
    """Import `_sm70_C` once. Returns None (and logs once) if it is absent."""
    try:
        import torch  # noqa: F401  (loads libtorch before our .so)

        return importlib.import_module("vllm_sm70._sm70_C")
    except ImportError as e:
        logger.info(
            "vllm_sm70: _sm70_C is not available (%s); sm70 kernels are off "
            "and vLLM runs its upstream paths.",
            e,
        )
        return None


def is_loaded() -> bool:
    return load() is not None


@functools.cache
def abi_version() -> int | None:
    if load() is None:
        return None
    import torch

    return int(torch.ops._sm70_C.abi_version())


@functools.cache
def so_sha256() -> str | None:
    module = load()
    path = getattr(module, "__file__", None) if module is not None else None
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint() -> str:
    """Stable id of the loaded extension: "abi=<n>;so=<sha256>" or "none"."""
    if load() is None:
        return "none"
    return f"abi={abi_version()};so={so_sha256()}"
