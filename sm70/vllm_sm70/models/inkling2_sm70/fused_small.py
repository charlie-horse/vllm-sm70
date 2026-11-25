# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the route of the small decode chains.

A decode-only batch (backend.is_decode_only) runs these torch chains as one
Triton kernel each:

    prologue.py         q and r clips, tau, r @ rel_proj   ops/small_ops.py query_rel
    attention_torch.py  the output gate                    ops/small_ops.py gate
    moe_tm.py           the soft-capped SwiGLU             ops/activation.py

The short convs (layers.py) then skip their padding-row mask: their inputs
are zero there. VLLM_SM70_INKLING_FUSED_SMALL_OPS=0 keeps the torch ops.
Tensors off the GPU take the torch ops, except in Triton's interpreter. The
switch is read on every eager call; a replayed CUDA graph keeps its route.

Other batches (``on_mixed``) run query_rel, the gate and the expert router
(ops/routing.py ``route_blocks``) as kernels too; their SwiGLU and convs keep
the torch route. VLLM_SM70_INKLING_MIXED_SMALL_OPS=0 keeps the torch ops.
"""

from __future__ import annotations

import torch

from vllm_sm70 import envs

from .ops import _compat

__all__ = ["MIXED_SWITCH", "SWITCH", "on_fused", "on_mixed", "on_small"]

SWITCH = "VLLM_SM70_INKLING_FUSED_SMALL_OPS"
_enabled = envs.environment_variables()[SWITCH]
MIXED_SWITCH = "VLLM_SM70_INKLING_MIXED_SMALL_OPS"
_mixed_enabled = envs.environment_variables()[MIXED_SWITCH]


def on_fused(x: torch.Tensor, decode_only: bool) -> bool:
    """Whether the small chains on `x`, of a decode-only batch or not, run
    the kernels (module docstring)."""
    if not decode_only:
        return False
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()


def on_mixed(x: torch.Tensor) -> bool:
    """Whether the small chains on `x` of a batch that is not decode-only
    run the kernels (module docstring)."""
    return on_fused(x, True) and _mixed_enabled()


def on_small(x: torch.Tensor, decode_only: bool) -> bool:
    """Whether query_rel and the gate on `x` run their kernels: ``on_fused``
    in a decode-only batch, else ``on_mixed``."""
    return on_fused(x, True) if decode_only else on_mixed(x)
