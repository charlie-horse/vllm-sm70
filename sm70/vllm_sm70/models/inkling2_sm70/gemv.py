# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the route of the weight-streaming GEMVs of small batches.

    linear      x [M, K] @ weight [N, K]^T: the GEMV of ops/small_m_gemv.py for
                ``min_rows`` <= M <= ``max_rows``, else F.linear. Serves a local
                layer's qkvr projection, the attention output projection and,
                up to 4 rows, the LM head
    segments    one GEMV launch for a gated or global layer's row segments
    on_nvfp4    whether one-row expert slots of moe_tm.py take ops/nvfp4_gemv.py
                instead of the TurboMind stages and the SwiGLU kernel

The dense route needs fp16 operands with unit stride along K. Switches
VLLM_SM70_INKLING_SMALL_M_GEMV=0 and VLLM_SM70_INKLING_NVFP4_GEMV=0 are read on
every eager call; a replayed graph keeps its route. Tensors off the GPU take
cuBLAS and TurboMind, except in Triton's interpreter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_sm70 import envs

from .ops import _compat
from .ops.small_m_gemv import MAX_ROWS, segment_gemv

__all__ = [
    "HEAD_MAX_ROWS",
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
# Rows of an LM head shard that take the GEMV; larger batches take cuBLAS.
HEAD_MIN_ROWS = 1
HEAD_MAX_ROWS = 4


def _on_device(x: torch.Tensor) -> bool:
    return _compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)


def on_small_m(
    x: torch.Tensor,
    weight: torch.Tensor,
    min_rows: int = 1,
    max_rows: int = MAX_ROWS,
) -> bool:
    """Whether x [M, K] @ weight [N, K]^T takes the GEMV."""
    return (
        x.ndim == 2
        and weight.ndim == 2
        and min_rows <= x.shape[0] <= min(max_rows, MAX_ROWS)
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
    max_rows: int = MAX_ROWS,
) -> torch.Tensor:
    """x @ weight^T in the dtype of x, written to `out` [M, N] when given."""
    if not on_small_m(x, weight, min_rows, max_rows) or (
        out is not None and out.stride(1) != 1
    ):
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
