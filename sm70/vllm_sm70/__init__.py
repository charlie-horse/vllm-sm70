# SPDX-License-Identifier: Apache-2.0
"""vllm-sm70: Volta (SM70) support for vLLM as an out-of-tree plugin.

Entry points (see nix-entry_points.txt):
  vllm.general_plugins  sm70 = vllm_sm70:register
  vllm.platform_plugins sm70 = vllm_sm70.platform:activate

No heavy imports at module level: vLLM swallows plugin import errors
(UP plugins/__init__.py:66-71), and the platform plugin is imported while
`vllm.platforms.current_platform` is still being resolved.
"""

from __future__ import annotations

import importlib
import logging

__version__ = "0.1.0"

logger = logging.getLogger("vllm.sm70")

# register() order; each is vllm_sm70.<area>.register().
AREAS: tuple[str, ...] = (
    "ops",
    "attention",
    "gdn",
    "fused_moe",
    "quant",
    "models",
    "passes",
)

_registered = False


def register() -> None:
    """vllm.general_plugins entry point. Idempotent, safe without _sm70_C.

    vLLM calls this in every process (UP plugins/__init__.py:77-90), possibly
    more than once.
    """
    global _registered
    if _registered:
        return
    _registered = True

    from vllm_sm70 import _ext, envs

    try:
        envs.merge_into_vllm()
    except Exception:
        logger.exception("vllm_sm70: could not merge env vars into vllm.envs")

    _ext.load()

    for area in AREAS:
        _register_area(area)


def _register_area(area: str) -> None:
    try:
        module = importlib.import_module(f"vllm_sm70.{area}")
        module.register()
    except Exception:
        # One broken area must not take the others (or vLLM) down.
        logger.exception("vllm_sm70: register() failed for area %r", area)
