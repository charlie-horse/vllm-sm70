# SPDX-License-Identifier: Apache-2.0
"""Ahead-of-time compilation of a Triton kernel for sm_70, without a GPU.

``compile_ptx(kernel, args)`` runs Triton's full pipeline for
``GPUTarget("cuda", 70, 32)`` through ptxas and returns the PTX. It enforces
the language rules Triton's CPU interpreter skips (constexpr typing, static
control flow). `args` names every kernel argument: a type string ("*fp16",
"i32", "fp32", ...) for a runtime argument, any other value for a constexpr.
``SM70_BANNED`` lists PTX that sm_70 lacks or runs slowly.

``skip_reason()`` is None when Triton's CUDA backend and its ptxas are there.
Under TRITON_INTERPRET=1 the kernels are not JIT functions: compile in a
process without it.
"""

from __future__ import annotations

import os
from typing import Any

SM70_BANNED = ("tanh.approx", "mma", ".bf16")


def skip_reason() -> str | None:
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "TRITON_INTERPRET=1: the kernels are interpreted functions"
    try:
        from triton.backends.compiler import GPUTarget  # noqa: F401
        from triton.backends.nvidia.compiler import get_ptxas
        from triton.compiler import ASTSource  # noqa: F401
    except ImportError as exc:
        return f"no Triton CUDA backend: {exc}"
    try:
        try:
            ptxas = get_ptxas(70)
        except TypeError:  # Triton before the per-arch lookup
            ptxas = get_ptxas()
    except Exception as exc:  # noqa: BLE001 - the backend raises when it finds none
        return f"no ptxas: {exc}"
    path = getattr(ptxas, "path", ptxas)
    if not path or not os.access(str(path), os.X_OK):
        return f"no ptxas at {path}"
    return None


def compile_ptx(
    kernel: Any,
    args: dict[str, Any],
    *,
    num_warps: int = 4,
    enable_fp_fusion: bool = True,
) -> str:
    """PTX of `kernel` compiled for sm_70 with `args` (see module doc)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    names = list(kernel.arg_names)
    assert set(args) == set(names), (
        f"missing {sorted(set(names) - set(args))}, "
        f"unknown {sorted(set(args) - set(names))}"
    )
    signature, constexprs = {}, {}
    for name in names:
        value = args[name]
        if isinstance(value, str):
            signature[name] = value
        else:
            signature[name] = "constexpr"
            constexprs[name] = value
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs),
        target=GPUTarget("cuda", 70, 32),
        options={"num_warps": num_warps, "enable_fp_fusion": enable_fp_fusion},
    )
    return compiled.asm["ptx"]


def banned_in(ptx: str) -> list[str]:
    return [token for token in SM70_BANNED if token in ptx]
