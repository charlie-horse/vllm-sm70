# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the route of the weight-streaming GEMVs of small
batches.

    linear      x [M, K] @ weight [N, K]^T: the GEMV of ops/small_m_gemv.py
                for 1 <= M <= 8 (``min_rows`` up), F.linear otherwise.
                Serves the qkvr projection of a local layer (prologue.py),
                the attention output projection (attention_torch.py) and,
                from 2 rows, the LM head (model.py)
    segments    one GEMV launch for the projections of a gated or global
                layer, whose weight row segments read different inputs
                (prologue.py ``project_qkvr``, ``project_packed``); None
                when the route is off
    on_nvfp4    whether the one-row expert slots of moe_tm.py take the
                NVFP4 GEMV of ops/nvfp4_gemv.py (W13 with the soft-capped
                SwiGLU in its epilogue, then W2) instead of the two
                TurboMind stages and the SwiGLU kernel

The dense route needs fp16 operands with unit stride along K.
VLLM_SM70_INKLING_SMALL_M_GEMV=0 sends the dense projections back to
cuBLAS; VLLM_SM70_INKLING_NVFP4_GEMV=0 sends the expert slots back to
TurboMind. Tensors off the GPU take cuBLAS and TurboMind, except in
Triton's interpreter. Both switches are read on every eager call; a
replayed CUDA graph keeps the route it was captured with.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_sm70 import envs

from .ops import _compat
from .ops.small_m_gemv import MAX_ROWS, segment_gemv

__all__ = [
    "HEAD_MIN_ROWS",
    "NVFP4_SWITCH",
    "SMALL_M_SWITCH",
    "linear",
    "on_nvfp4",
    "on_small_m",
    "segments",
]

SMALL_M_SWITCH = "VLLM_SM70_INKLING_SMALL_M_GEMV"
NVFP4_SWITCH = "VLLM_SM70_INKLING_NVFP4_GEMV"
_small_m_enabled = envs.environment_variables()[SMALL_M_SWITCH]
_nvfp4_enabled = envs.environment_variables()[NVFP4_SWITCH]
# cuBLAS streams the LM head at about 90% of HBM peak for one row.
HEAD_MIN_ROWS = 2


def _on_device(x: torch.Tensor) -> bool:
    return _compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)


def on_small_m(x: torch.Tensor, weight: torch.Tensor, min_rows: int = 1) -> bool:
    """Whether x [M, K] @ weight [N, K]^T takes the GEMV."""
    return (
        x.ndim == 2
        and weight.ndim == 2
        and min_rows <= x.shape[0] <= MAX_ROWS
        and x.dtype == weight.dtype == torch.float16
        and x.shape[1] == weight.shape[1]
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and _on_device(x)
        and _small_m_enabled()
    )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    min_rows: int = 1,
) -> torch.Tensor:
    """x @ weight^T in the dtype of x, written to `out` [M, N] when given."""
    if not on_small_m(x, weight, min_rows) or (out is not None and out.stride(1) != 1):
        if out is None:
            return F.linear(x, weight)
        return torch.mm(x, weight.t(), out=out)
    if out is None:
        out = x.new_empty(x.shape[0], weight.shape[0])
    return segment_gemv((x,), weight, (), out)


def segments(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    bounds: tuple[int, ...],
) -> torch.Tensor | None:
    """[M, N]: rows of `weight` [N, K] before bounds[0] times inputs[0],
    then inputs[1], inputs[2], inputs[1], inputs[2] (2 or 4 `bounds`), as
    one launch; None when the route does not take it."""
    x = inputs[0]
    if not on_small_m(x, weight) or any(
        t.shape != x.shape or t.dtype != x.dtype or t.stride(1) != 1 for t in inputs[1:]
    ):
        return None
    out = x.new_empty(x.shape[0], weight.shape[0])
    return segment_gemv(inputs, weight, bounds, out)


def on_nvfp4(x: torch.Tensor) -> bool:
    """Whether the one-row expert slots on rows `x` take the NVFP4 GEMV."""
    return _on_device(x) and _nvfp4_enabled()
