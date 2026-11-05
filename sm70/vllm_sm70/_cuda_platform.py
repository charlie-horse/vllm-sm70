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

import atexit
import importlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.platforms.cuda import CudaPlatform

import vllm_sm70
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
        except Exception as exc:
            vllm_sm70._area_status.setdefault(
                area, f"import {type(exc).__name__}: {exc}"
            )
            logger.exception("vllm_sm70: importing area %r failed", area)


def _enforce_strict() -> None:
    """Fail-fast on SM70: a failed area must stop startup.

    Called only from Sm70CudaPlatform hooks, i.e. only when the sm70 platform
    is active (every visible GPU is cc 7.0). vLLM swallows exceptions raised
    from a general plugin's register() (UP plugins/__init__.py:66-71), so
    the raise has to happen here. VLLM_SM70_PLUGIN_STRICT=0 downgrades it.
    """
    failed = vllm_sm70.failed_areas()
    if not failed:
        return
    detail = "; ".join(f"{a}: {e}" for a, e in failed.items())
    if os.environ.get("VLLM_SM70_PLUGIN_STRICT") == "0":
        logger.warning("vllm_sm70: areas failed (strict off): %s", detail)
        return
    raise RuntimeError(
        "vllm_sm70: plugin areas failed to register on SM70: "
        f"{detail}. Set VLLM_SM70_PLUGIN_STRICT=0 to start anyway."
    )


_shutdown_registered = False


def _tm_runtime_close() -> None:
    """Sync and release the process-wide TurboMind runtime (Quinn K02 N1)."""
    try:
        import torch

        op = getattr(getattr(torch.ops, "_sm70_C", None), "tm_runtime_close", None)
        if op is not None and torch.cuda.is_initialized():
            op()
    except Exception:  # never fail interpreter shutdown
        logger.debug("vllm_sm70: tm_runtime_close failed", exc_info=True)


def _register_shutdown() -> None:
    global _shutdown_registered
    if not _shutdown_registered:
        _shutdown_registered = True
        atexit.register(_tm_runtime_close)


class Sm70CudaPlatform(CudaPlatform):
    @classmethod
    def import_kernels(cls) -> None:
        super().import_kernels()
        _ext.load()
        _ensure_areas_imported()
        vllm_sm70.retry_failed_areas()
        _register_shutdown()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: Any,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> str:
        # K06: Flash-V100 (AttentionBackendEnum.CUSTOM) for fp16 decoder
        # layers with head size 128/256 when no backend was chosen and the
        # K05 ops are built; otherwise upstream (TRITON_ATTN on SM70).
        from vllm_sm70 import attention

        reason = attention.flash_v100_rejection(selected_backend, attn_selector_config)
        if reason is None:
            capability = cls.get_device_capability()
            invalid = attention.validate(attn_selector_config, capability)
            if not invalid:
                attention.log_choice_once(
                    "FLASH_ATTN_V100",
                    f"head_size={attn_selector_config.head_size}",
                )
                return attention.custom_backend_path()
            reason = "; ".join(invalid)
        path = super().get_attn_backend_cls(
            selected_backend, attn_selector_config, num_heads
        )
        attention.log_choice_once(path.rsplit(".", 1)[-1], reason)
        return path

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().check_and_update_config(vllm_config)
        _ensure_areas_imported()
        vllm_sm70.retry_failed_areas()
        _enforce_strict()
        logger.info(
            "vllm_sm70: platform %s.%s; areas: %s",
            cls.__module__,
            cls.__qualname__,
            vllm_sm70.registered_areas(),
        )
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

    # K17: upstream's own per-op IR priority hook (UP platforms/interface.py:
    # 1452, CudaPlatform's override platforms/cuda.py:712-734, merged into
    # KernelConfig.ir_op_priority at config/kernel.py:461-478; a user-set
    # priority stays first). Under Inductor upstream picks ["native"], whose
    # fused_add_rms_norm keeps the residual sum in FP32 before rounding
    # (UP ir/ops/layernorm.py:44-62). On SM70 pin the C kernel (FP16
    # residual, __hadd2), so the unfused graph, the kill-switched path and
    # the fused TP2 all-reduce + RMSNorm collective (K17) share one rounding.
    # "native" stays as the fallback for args vllm_c does not support (a
    # weight dtype other than the input's, variance_size).
    SM70_FUSED_ADD_RMS_NORM_PRIORITY = ("vllm_c", "native")

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config: VllmConfig) -> Any:
        import dataclasses

        priority = super().get_default_ir_op_priority(vllm_config)
        # Flint only (K17 admission): there the fused AR kernel keeps the
        # residual in fp16, so the unfused path (switch off) must too. Other
        # models keep upstream's priority (fork parity).
        if not _k17_admits(vllm_config):
            return priority
        return dataclasses.replace(
            priority, fused_add_rms_norm=list(cls.SM70_FUSED_ADD_RMS_NORM_PRIORITY)
        )


def _k17_admits(vllm_config: Any) -> bool:
    """True when K17's rejection() is None for this engine config."""
    if getattr(vllm_config, "model_config", None) is None:
        return False
    try:
        from vllm_sm70 import ar_rms_fusion

        tp = vllm_config.parallel_config.tensor_parallel_size
        return ar_rms_fusion.rejection(vllm_config, tp) is None
    except Exception:
        logger.debug("vllm_sm70: K17 admission check failed", exc_info=True)
        return False
