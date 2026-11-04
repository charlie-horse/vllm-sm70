# SPDX-License-Identifier: Apache-2.0
"""All `VLLM_SM70_*` environment variables, in one registry.

Rules:
- Kill switches: only the value "0" acts, and it turns a feature off. The
  environment can never turn a feature on; defaults are decided in code.
- Each area adds its variables to its own section in `_SECTIONS`.
- `merge_into_vllm()` adds every getter to `vllm.envs.environment_variables`
  (setdefault, never overriding upstream). This has two effects:
  `Platform.validate_environ` (UP platforms/interface.py:1458-1471) accepts
  the names, and `vllm.envs.compile_factors()` (UP envs.py:2353) hashes them
  into the torch.compile cache key. Lookups through `vllm.envs.<NAME>`
  still work after `enable_envs_cache()` (UP envs.py:2310), because the
  cached `__getattr__` (UP envs.py:2293) reads the same dict on a miss.

This module imports nothing from vllm at import time, so it is safe to
import during platform resolution and in tests without vllm.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def kill_switch(name: str) -> Callable[[], bool]:
    """Getter that is True (feature enabled) unless `name` == "0"."""

    def getter() -> bool:
        return os.environ.get(name) != "0"

    getter.__name__ = f"env_{name}"
    return getter


def _debug_flag(name: str) -> Callable[[], bool]:
    """Diagnostics only (extra logging); it never changes what runs."""

    def getter() -> bool:
        return os.environ.get(name, "") not in ("", "0")

    getter.__name__ = f"env_{name}"
    return getter


def _ext_fingerprint() -> str:
    from vllm_sm70._ext import fingerprint

    return fingerprint()


# Per-area sections, in register() order. Values are zero-argument getters.
_SECTIONS: dict[str, dict[str, Callable[[], Any]]] = {
    "core": {
        # Extra plugin logging (diagnostic, not a feature).
        "VLLM_SM70_PLUGIN_DEBUG": _debug_flag("VLLM_SM70_PLUGIN_DEBUG"),
        # Not meant to be set: a derived getter so that compile_factors()
        # changes whenever _sm70_C is rebuilt (abi_version + .so sha256).
        # vLLM hashes .py files, not the .so (UP compilation/backends.py:1064).
        "VLLM_SM70_EXT_FINGERPRINT": _ext_fingerprint,
    },
    "ops": {},
    "attention": {},
    "gdn": {},
    "fused_moe": {},
    "quant": {},
    "models": {},
    "passes": {},
}


def environment_variables() -> dict[str, Callable[[], Any]]:
    """All getters, flattened in section order."""
    flat: dict[str, Callable[[], Any]] = {}
    for section in _SECTIONS.values():
        for name, getter in section.items():
            assert name.startswith("VLLM_SM70_"), name
            assert name not in flat, f"duplicate env var {name}"
            flat[name] = getter
    return flat


def merge_into_vllm() -> list[str]:
    """Add our getters to vllm.envs. Idempotent; returns names newly added."""
    import vllm.envs as vllm_envs

    added = []
    target = vllm_envs.environment_variables
    for name, getter in environment_variables().items():
        if name not in target:
            target[name] = getter
            added.append(name)
    return added


def __getattr__(name: str) -> Any:
    getters = environment_variables()
    if name in getters:
        return getters[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables())
