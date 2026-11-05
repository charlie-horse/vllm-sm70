# SPDX-License-Identifier: Apache-2.0
"""Sm70CudaPlatform: upstream CudaPlatform plus the SM70 hooks.

Imported only via `vllm_sm70.platform.Sm70CudaPlatform`, i.e. once vLLM
resolves our platform plugin. Importing this module needs vLLM's compiled
`_C_stable_libtorch` (UP platforms/cuda.py:23).

The area imports (quant etc.) are not done at module import: this module is
imported *during* `current_platform` resolution (UP platforms/__init__.py:
300-317, before `_current_platform` is assigned), and the area modules import
vLLM modules that read `current_platform` at import time (e.g. UP
model_executor/layers/quantization/__init__.py:11). Doing it here would
re-enter resolution and create a second platform instance. Instead the areas
are imported from the first post-resolution hook: `import_kernels()` (run on
`import vllm._custom_ops`, UP _custom_ops.py:21) or `check_and_update_config()`.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.platforms.cuda import CudaPlatform

from vllm_sm70 import AREAS, _ext

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger("vllm.sm70")

_areas_imported = False


def _ensure_areas_imported() -> None:
    """Import (not register) every area once, so their classes are importable
    and import-time registrations (e.g. quant configs) have happened."""
    global _areas_imported
    if _areas_imported:
        return
    _areas_imported = True
    for area in AREAS:
        try:
            importlib.import_module(f"vllm_sm70.{area}")
        except Exception:
            logger.exception("vllm_sm70: importing area %r failed", area)


class Sm70CudaPlatform(CudaPlatform):
    @classmethod
    def import_kernels(cls) -> None:
        super().import_kernels()
        _ext.load()
        _ensure_areas_imported()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: Any,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> str:
        # K06 adds Flash-V100 here; until then, upstream selection
        # (TRITON_ATTN on SM70).
        return super().get_attn_backend_cls(
            selected_backend, attn_selector_config, num_heads
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().check_and_update_config(vllm_config)
        _ensure_areas_imported()
        from vllm_sm70.passes import install_composite_pass

        install_composite_pass(vllm_config.compilation_config)

    # U1 hooks (patches/0001): defaults keep upstream behaviour.
    @classmethod
    def get_unquantized_gemm(
        cls, linear_backend: str = "auto"
    ) -> Callable[..., Any] | None:
        return None

    @classmethod
    def kernel_warmup(cls, worker: Any) -> None:
        return None
