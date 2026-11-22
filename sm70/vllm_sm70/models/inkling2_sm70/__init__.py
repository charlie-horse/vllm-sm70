# SPDX-License-Identifier: Apache-2.0
"""M12 inkling2-sm70-model: Inkling 2 (looped, e.g. Inkling 2 Mini) on SM70.

What: the architecture name ``Inkling2LoopSm70ForConditionalGeneration``
  (model.py), upstream's Inkling 2 config classes vendored unchanged
  (configs.py, see PROVENANCE.md) and the SM70 subclasses that keep the LM
  head in the model dtype (sm70_configs.py).
Why: vLLM v0.33.0 has no ``inkling2`` config type or architecture, and
  upstream's model asserts bf16 and SM90+ kernels. A separate architecture
  name keeps upstream's class untouched; serve args select ours with
  ``--hf-overrides '{"architectures":
  ["Inkling2LoopSm70ForConditionalGeneration"]}'``.
Registration: the config classes go into vLLM's ``_CONFIG_REGISTRY`` in every
  process (a CPU process parses config.json too); the model class only when
  the visible GPUs are all cc 7.0.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Inkling2LoopSm70ForConditionalGeneration

logger = logging.getLogger("vllm.sm70")

ARCH = "Inkling2LoopSm70ForConditionalGeneration"
LAZY_TARGET = "vllm_sm70.models.inkling2_sm70:Inkling2LoopSm70ForConditionalGeneration"
__all__ = [
    "ARCH",
    "Inkling2LoopSm70ForConditionalGeneration",
    "register",
    "register_configs",
]

_registered = False


def register_configs() -> None:
    """Add the Inkling 2 config classes to vLLM's ``_CONFIG_REGISTRY``.

    Idempotent. A model type that already has another class keeps it (a vLLM
    that ships its own Inkling 2 configs wins) and is logged once per call.
    Imports transformers' config base only: no model, Triton or CUDA module.
    """
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .sm70_configs import CONFIG_CLASSES

    for model_type, config_class in CONFIG_CLASSES.items():
        # dict.get, not []: LazyConfigDict.__getitem__ imports the class a
        # string entry names (UP transformers_utils/config.py:113-119).
        current = dict.get(_CONFIG_REGISTRY, model_type)
        if current is None:
            _CONFIG_REGISTRY[model_type] = config_class
        elif current is not config_class:
            logger.warning(
                "vllm_sm70: config type %r is already registered as %r; "
                "keeping it, %s is not used.",
                model_type,
                current,
                config_class.__name__,
            )


def register() -> bool:
    """Register the config classes, then ``ARCH`` on SM70. Idempotent.

    The config classes register first, on any device. The model class is
    skipped (returns False) when the visible GPUs aren't all cc 7.0. Raises
    when they are but ``_sm70_C`` is missing (K01 fail-fast), and when the
    registry didn't take the entry. It marks itself done only after every
    step succeeded, so a retry runs it again. The lazy "<module>:<class>"
    form keeps model and CUDA imports out of plugin load
    (UP models/registry.py:1176-1212).
    """
    global _registered
    if _registered:
        return True
    register_configs()

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
    _registered = True
    return True


def __getattr__(name: str):
    # Lazy, so plugin load doesn't import torch.nn layers or Triton.
    if name == "Inkling2LoopSm70ForConditionalGeneration":
        from .model import Inkling2LoopSm70ForConditionalGeneration

        return Inkling2LoopSm70ForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
