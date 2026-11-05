# SPDX-License-Identifier: Apache-2.0
"""One composite post-grad Inductor pass carrying every SM70 fusion pattern.

UP's PostGradPassManager hard-codes its passes (UP compilation/passes/
pass_manager.py:139-222). The one open slot is
`compilation_config.inductor_compile_config["post_grad_custom_post_pass"]`,
which UP compilation/backends.py:983-1000 appends to the manager. The
manager's uuid() includes each pass's uuid() (pass_manager.py:229-250), and
Inductor uses that uuid in its FX-graph cache key (torch
_inductor/codecache.py, FxGraphHashDetails, post_grad_custom_post_pass).

vLLM's own compile-cache directory does *not* see this pass: the slot is
excluded from CompilationConfig.compute_hash (UP config/compilation.py:
832-834), and the cache dir is keyed by env, config and .py-source hashes
(UP backends.py:1058-1080). The .so fingerprint therefore also goes into
vllm.envs.compile_factors() through VLLM_SM70_EXT_FINGERPRINT (envs.py).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import fx
from vllm.compilation.passes.inductor_pass import InductorPass

logger = logging.getLogger("vllm.sm70")

PASS_KEY = "post_grad_custom_post_pass"

# Bump when the composite's own logic changes in a way the source hash
# would not show.
COMPOSITE_VERSION = 1


class _Pattern:
    __slots__ = ("name", "version", "fn")

    def __init__(self, name: str, version: int, fn: Callable[[fx.Graph], None]):
        self.name, self.version, self.fn = name, version, fn


# Filled by areas (e.g. K17 AR+RMSNorm) via register_pattern(). Empty in K01.
_PATTERNS: list[_Pattern] = []


def register_pattern(name: str, version: int, fn: Callable[[fx.Graph], None]) -> None:
    """Add a graph rewrite. Idempotent per name; order is registration order."""
    for p in _PATTERNS:
        if p.name == name:
            if p.version != version:
                raise ValueError(
                    f"pattern {name!r} registered twice with versions "
                    f"{p.version} and {version}"
                )
            return
    _PATTERNS.append(_Pattern(name, version, fn))


def patterns() -> list[tuple[str, int]]:
    return [(p.name, p.version) for p in _PATTERNS]


class Sm70CompositePass(InductorPass):
    """Runs `inner` passes (a pre-existing custom pass, if any), then ours."""

    def __init__(self, inner: list[InductorPass] | None = None) -> None:
        self.inner: list[InductorPass] = list(inner or [])

    def __call__(self, graph: fx.Graph) -> None:
        for p in self.inner:
            p(graph)
        for p in _PATTERNS:
            p.fn(graph)

    def uuid(self) -> str:
        from vllm_sm70 import _ext

        state: dict[str, Any] = {
            "pass": "vllm_sm70.Sm70CompositePass",
            "version": COMPOSITE_VERSION,
            "source": InductorPass.hash_source(type(self)),
            "patterns": [list(p) for p in patterns()],
            "ext": _ext.fingerprint(),
            "inner": [p.uuid() for p in self.inner],
        }
        return InductorPass.hash_dict(state)


def install_composite_pass(compilation_config: Any) -> Sm70CompositePass:
    """Put Sm70CompositePass in the custom-post-pass slot. Idempotent; an
    existing custom pass is kept and runs first, inside the composite."""
    icc = compilation_config.inductor_compile_config
    existing = icc.get(PASS_KEY)
    if isinstance(existing, Sm70CompositePass):
        return existing
    if existing is None:
        composite = Sm70CompositePass()
    elif isinstance(existing, InductorPass):
        composite = Sm70CompositePass(inner=[existing])
    else:
        raise TypeError(
            f"inductor_compile_config[{PASS_KEY!r}] is a {type(existing).__name__}, "
            "not an InductorPass; vLLM (UP backends.py:991-995) requires one"
        )
    icc[PASS_KEY] = composite
    return composite


def register() -> None:
    """Area hook. K01 has no patterns; later patches call register_pattern()."""
