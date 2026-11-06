# SPDX-License-Identifier: Apache-2.0
"""M5 flint-sm70-model: Flint (RecursiveSI/flint, ModelOpt W4A16 NVFP4) on SM70.

What: ``FlintSm70ForCausalLM`` (model.py), a port of the fork's Flint model
  (FORK cf9492a vllm/models/flint/) on upstream v0.33.0 layers, with FP16
  rescaling (fp16_rescale.py) and the engine-config hook (config.py). The
  fused q/k norm + RoPE + KV write is K15's (vllm_sm70/flint/
  qk_norm_rope_kv.py, #54), which the model calls. It reuses upstream's
  ``vllm.models.flint.configs.FlintConfig``.
Why: upstream's FlintForCausalLM asserts bf16 + FlashInfer (UP 4ba787e
  models/flint/model.py:318, 321), and V100 has neither. A separate
  architecture name keeps upstream's class untouched; the c4130 serve args
  select ours with
  ``--hf-overrides '{"architectures": ["FlintSm70ForCausalLM"]}'``.
Kill-switch: VLLM_SM70_FLINT_QK_ROPE_KV=0 turns off the fused op only
  (unfused norms + RoPE + upstream Attention.forward). The model itself is
  opt-in through --hf-overrides.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import FlintSm70ModelConfig
    from .model import FlintSm70ForCausalLM

logger = logging.getLogger("vllm.sm70")

ARCH = "FlintSm70ForCausalLM"
LAZY_TARGET = "vllm_sm70.models.flint_sm70:FlintSm70ForCausalLM"
__all__ = ["ARCH", "FlintSm70ForCausalLM", "FlintSm70ModelConfig", "register"]

_registered = False


def register() -> bool:
    """Register ``FlintSm70ForCausalLM`` and its config hook. Idempotent.

    Skips (returns False) when the visible GPUs aren't all cc 7.0. Raises when
    they are but ``_sm70_C`` is missing (K01 fail-fast), and when the registry
    or MODELS_CONFIG_MAP didn't take the entry. It marks itself done only
    after every step succeeded, so a retry runs it again. The lazy
    "<module>:<class>" form keeps model and CUDA imports out of plugin load
    (UP models/registry.py:1176-1212).
    """
    global _registered
    if _registered:
        return True
    from vllm_sm70 import _ext
    from vllm_sm70 import platform as sm70_platform

    if not sm70_platform.is_sm70_ready():
        if sm70_platform.all_visible_gpus_are_sm70() and not _ext.is_loaded():
            raise RuntimeError(
                f"vllm_sm70: every visible GPU is cc 7.0 but _sm70_C is not "
                f"loaded; {ARCH} can't run. Fix the extension build."
            )
        logger.debug("No cc 7.0 GPUs visible: %s not registered.", ARCH)
        return False

    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import FlintSm70ModelConfig

    ModelRegistry.register_model(ARCH, LAZY_TARGET)
    entry = ModelRegistry.models.get(ARCH)
    module_name, _, class_name = LAZY_TARGET.partition(":")
    if (getattr(entry, "module_name", None), getattr(entry, "class_name", None)) != (
        module_name,
        class_name,
    ):
        raise RuntimeError(
            f"vllm_sm70: {ARCH!r} resolves to {entry!r}, not {LAZY_TARGET}; "
            "upstream registry semantics changed."
        )
    MODELS_CONFIG_MAP[ARCH] = FlintSm70ModelConfig
    if MODELS_CONFIG_MAP.get(ARCH) is not FlintSm70ModelConfig:
        raise RuntimeError(f"vllm_sm70: MODELS_CONFIG_MAP[{ARCH!r}] did not take.")
    _registered = True
    return True


def __getattr__(name: str):
    # Lazy, so plugin load and the config hook don't import layers or Triton.
    if name == "FlintSm70ForCausalLM":
        from .model import FlintSm70ForCausalLM

        return FlintSm70ForCausalLM
    if name == "FlintSm70ModelConfig":
        from .config import FlintSm70ModelConfig

        return FlintSm70ModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
