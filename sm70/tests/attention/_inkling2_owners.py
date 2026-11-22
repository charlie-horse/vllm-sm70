# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini cache owners for the M12 CPU tests: the vendored
config.json, a stand-in vLLM config, and the owners built from both."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

CONFIG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models"
    / "data"
    / "inkling2_mini"
    / "config.json"
)


def text_config():
    from vllm_sm70.models.inkling2_sm70.configs import Inkling2ModelConfig

    return Inkling2ModelConfig(**json.loads(CONFIG.read_text())["text_config"])


def owner_vllm_config(block_size: int, dtype):
    """The fields the owners read; ``static_forward_context`` starts empty."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=dtype),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def build(block_size: int, tp: int, dtype, config=None):
    """(owners, vllm_config) for ``config`` (default: Inkling 2 Mini)."""
    from vllm_sm70.models.inkling2_sm70.cache import build_cache_owners

    vllm_config = owner_vllm_config(block_size, dtype)
    owners = build_cache_owners(
        config or text_config(), vllm_config=vllm_config, tp_size=tp
    )
    return owners, vllm_config
