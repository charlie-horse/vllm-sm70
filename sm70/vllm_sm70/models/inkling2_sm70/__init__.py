# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (looped, e.g. Inkling 2 Mini) on SM70.

The model ``Inkling2LoopSm70ForConditionalGeneration`` (model.py, loop.py,
layers.py: the text stack on one TP rank), upstream's Inkling 2 config
classes vendored unchanged (configs.py, see PROVENANCE.md) and their SM70
subclasses (sm70_configs.py), the renderer mode and the reasoning and tool
parsers ``inkling2_sm70`` (renderer.py, inkling2_encoding.py, parsers.py).
Serve args select the model with ``--hf-overrides '{"architectures":
["Inkling2LoopSm70ForConditionalGeneration"]}'``.
The config classes (``inkling2_dspark`` included), the renderer mode and the
parser names register in every process; the model class and the draft's
``Inkling2DSparkSm70Model`` (draft.py) only when the visible GPUs are all
cc 7.0.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Inkling2LoopSm70ForConditionalGeneration

logger = logging.getLogger("vllm.sm70")

ARCH = "Inkling2LoopSm70ForConditionalGeneration"
LAZY_TARGET = "vllm_sm70.models.inkling2_sm70:Inkling2LoopSm70ForConditionalGeneration"
RENDERER_MODE = "inkling2_sm70"
RENDERER_TARGET = ("vllm_sm70.models.inkling2_sm70.renderer", "Inkling2Sm70Renderer")
# The mode only selects the renderer; token operations use the HF tokenizer.
TOKENIZER_TARGET = ("vllm.tokenizers.hf", "CachedHfTokenizer")
PARSER_NAME = "inkling2_sm70"
PARSER_MODULE = "vllm_sm70.models.inkling2_sm70.parsers"
REASONING_PARSER_TARGET = (PARSER_MODULE, "Inkling2Sm70ParserReasoningAdapter")
TOOL_PARSER_TARGET = (PARSER_MODULE, "Inkling2Sm70EngineToolParser")
__all__ = [
    "ARCH",
    "PARSER_MODULE",
    "PARSER_NAME",
    "RENDERER_MODE",
    "Inkling2LoopSm70ForConditionalGeneration",
    "register",
    "register_configs",
    "register_parsers",
    "register_renderer",
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


def register_renderer() -> None:
    """Add ``RENDERER_MODE`` to vLLM's tokenizer and renderer registries.

    Idempotent. The renderer mode is the tokenizer mode
    (UP renderers/registry.py:84-90), so both registries get the name.
    Imports neither renderer.py nor a tokenizer: the entries are lazy.
    """
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    if TokenizerRegistry.tokenizers.get(RENDERER_MODE) != TOKENIZER_TARGET:
        TokenizerRegistry.register(RENDERER_MODE, *TOKENIZER_TARGET)
    if RENDERER_REGISTRY.renderers.get(RENDERER_MODE) != RENDERER_TARGET:
        RENDERER_REGISTRY.register(RENDERER_MODE, *RENDERER_TARGET)


def register_parsers() -> None:
    """Add ``PARSER_NAME`` to vLLM's reasoning and tool parser managers.

    Idempotent. The entries are lazy: parsers.py is imported when a name is
    first resolved, or by ``--reasoning-parser-plugin``/``--tool-parser-plugin``.
    """
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_lazy_module(PARSER_NAME, *REASONING_PARSER_TARGET)
    ToolParserManager.register_lazy_module(PARSER_NAME, *TOOL_PARSER_TARGET)


def register() -> bool:
    """Register configs, renderer mode and parsers, then ``ARCH`` and the
    draft's architecture on SM70.

    Idempotent. The config classes, the renderer mode and the parser names
    register first, on any device. The model class is skipped (returns
    False) when the visible GPUs aren't all cc 7.0. Raises
    when they are but ``_sm70_C`` is missing (fail fast), and when the
    registry didn't take the entry. It marks itself done only after every
    step succeeded, so a retry runs it again. The lazy "<module>:<class>"
    form keeps model and CUDA imports out of plugin load
    (UP models/registry.py:1176-1212).
    """
    global _registered
    if _registered:
        return True
    register_configs()
    register_renderer()
    register_parsers()

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

    from .draft import DRAFT_ARCH, DRAFT_LAZY_TARGET

    for arch, target in ((ARCH, LAZY_TARGET), (DRAFT_ARCH, DRAFT_LAZY_TARGET)):
        ModelRegistry.register_model(arch, target)
        entry = ModelRegistry.models.get(arch)
        module_name, _, class_name = target.partition(":")
        if (
            getattr(entry, "module_name", None),
            getattr(entry, "class_name", None),
        ) != (module_name, class_name):
            raise RuntimeError(
                f"vllm_sm70: {arch!r} resolves to {entry!r}, not {target}; "
                "upstream registry semantics changed."
            )
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import Inkling2Sm70ModelConfig

    MODELS_CONFIG_MAP[ARCH] = Inkling2Sm70ModelConfig
    if MODELS_CONFIG_MAP.get(ARCH) is not Inkling2Sm70ModelConfig:
        raise RuntimeError(f"vllm_sm70: MODELS_CONFIG_MAP[{ARCH!r}] did not take.")
    _registered = True
    return True


def __getattr__(name: str):
    # Lazy, so plugin load doesn't import torch.nn layers or Triton.
    if name == "Inkling2LoopSm70ForConditionalGeneration":
        from .model import Inkling2LoopSm70ForConditionalGeneration

        return Inkling2LoopSm70ForConditionalGeneration
    if name == "Inkling2Sm70ModelConfig":
        from .config import Inkling2Sm70ModelConfig

        return Inkling2Sm70ModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
