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

# Outcome of each area's register(): "ok" or the error text. Read it through
# registered_areas(); Sm70CudaPlatform turns failures into a startup error
# on SM70 (fail-fast), since vLLM swallows exceptions raised here.
_area_status: dict[str, str] = {}


def registered_areas() -> dict[str, str]:
    """Area name -> "ok" or the error text, for every area tried so far."""
    return dict(_area_status)


def failed_areas() -> dict[str, str]:
    return {a: s for a, s in _area_status.items() if s != "ok"}


def retry_failed_areas() -> None:
    """Re-run register() for areas that failed (e.g. on import order)."""
    for area in list(failed_areas()):
        _register_area(area)


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
    _log_summary()


def _register_area(area: str) -> None:
    try:
        module = importlib.import_module(f"vllm_sm70.{area}")
        module.register()
    except Exception as exc:
        # One broken area must not take the others down here; on SM70 the
        # platform raises later (Sm70CudaPlatform._enforce_strict).
        _area_status[area] = f"{type(exc).__name__}: {exc}"
        logger.exception("vllm_sm70: register() failed for area %r", area)
    else:
        _area_status[area] = "ok"


def _log_summary() -> None:
    failed = failed_areas()
    logger.info(
        "vllm_sm70: areas registered: %s%s",
        ", ".join(a for a, s in _area_status.items() if s == "ok") or "none",
        f"; FAILED: {', '.join(failed)}" if failed else "",
    )
