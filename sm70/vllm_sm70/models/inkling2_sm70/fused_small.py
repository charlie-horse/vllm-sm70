# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the route of the small decode chains.

A decode-only batch (backend.is_decode_only) runs these torch chains as one
Triton kernel each:

    prologue.py         q and r clips, tau, r @ rel_proj and the padding
                        rows                     ops/small_ops.py query_rel
    attention_torch.py  the output gate          ops/small_ops.py gate
    moe_tm.py           the soft-capped SwiGLU   ops/activation.py
    layers.py           the short convs skip their padding-row mask: their
                        inputs are zero there (the attention output, gated
                        to zero, and the expert sum of padding rows)

Every other batch keeps the torch ops. VLLM_SM70_INKLING_FUSED_SMALL_OPS=0
puts decode-only batches back on them. Tensors off the GPU take the torch
ops, except in Triton's interpreter. The switch is read on every eager call;
a replayed CUDA graph keeps the route it was captured with.
"""

from __future__ import annotations

import torch

from vllm_sm70 import envs

from .ops import _compat

__all__ = ["SWITCH", "on_fused"]

SWITCH = "VLLM_SM70_INKLING_FUSED_SMALL_OPS"
_enabled = envs.environment_variables()[SWITCH]


def on_fused(x: torch.Tensor, decode_only: bool) -> bool:
    """Whether the small chains on `x`, of a decode-only batch or not, run
    the kernels (module docstring)."""
    if not decode_only:
        return False
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()
