# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 dense projections of at most ``MAX_ROWS`` tokens as one Triton
GEMV (sm_70: no tl.dot, fp32 FMA).

    segment_gemv   out [M, N] = x_s [M, K] @ weight[rows of s]^T for each
                   row segment s of `weight` [N, K], in one launch

A segment is a run of weight rows that reads one input: segments
alternate over the inputs as (x0) or (x0, x1, x2) or (x0, x1, x2, x1, x2),
the row layout of a gated or global layer's ``attn.qkvr.weight`` (query
rows, k, v) with the key and value tap rows packed after it. Each program
covers ``BLOCK_N`` rows of one segment over all of K and all M rows, so
every weight byte is read once. Products and sums run in fp32 and round
once to the dtype of `out`.

The grid is static for static M and shapes, nothing is read back to the
host and nothing is allocated, so a call captures in a CUDA graph.
"""

from __future__ import annotations

import itertools
from typing import Final

import torch

from ._compat import tl, triton

__all__ = ["MAX_ROWS", "config", "segment_gemv"]

MAX_ROWS: Final = 8
MAX_SEGMENTS: Final = 5
# BLOCK_M -> (BLOCK_N, BLOCK_K, num_warps); the fp32 partials of a program
# are BLOCK_M * BLOCK_N * BLOCK_K values.
CONFIGS: Final = {
    1: (8, 256, 2),
    2: (16, 256, 2),
    4: (8, 128, 2),
    8: (8, 128, 2),
}
# Segment input order (module doc).
SEGMENT_INPUTS: Final = (0, 1, 2, 1, 2)


@triton.jit
def _segment_gemv_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    w_ptr,
    out_ptr,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_w,
    stride_out,
    rows,
    k_in,
    n_out,
    n1,
    n2,
    n3,
    n4,
    b1,
    b2,
    b3,
    b4,
    SEGMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Segment s holds weight rows [n_s, n_{s+1}) (n_0 = 0, n_5 = n_out) and
    programs [b_s, b_{s+1}); unused boundaries equal n_out and the grid."""
    pid = tl.program_id(0)
    if SEGMENTS == 1:
        x_ptr = x0_ptr
        stride_x = stride_x0
        start = pid * BLOCK_N
        end = n_out
    else:
        seg = (
            (pid >= b1).to(tl.int32)
            + (pid >= b2).to(tl.int32)
            + (pid >= b3).to(tl.int32)
            + (pid >= b4).to(tl.int32)
        )
        first = tl.where(
            seg == 0,
            0,
            tl.where(seg == 1, n1, tl.where(seg == 2, n2, tl.where(seg == 3, n3, n4))),
        )
        end = tl.where(
            seg == 0,
            n1,
            tl.where(
                seg == 1, n2, tl.where(seg == 2, n3, tl.where(seg == 3, n4, n_out))
            ),
        )
        block = tl.where(
            seg == 0,
            0,
            tl.where(seg == 1, b1, tl.where(seg == 2, b2, tl.where(seg == 3, b3, b4))),
        )
        start = first + (pid - block) * BLOCK_N
        # Inputs x0, x1, x2, x1, x2 for segments 0..4.
        which = tl.where(seg == 0, 0, 2 - seg % 2)
        x_ptr = tl.where(which == 0, x0_ptr, tl.where(which == 1, x1_ptr, x2_ptr))
        stride_x = tl.where(
            which == 0, stride_x0, tl.where(which == 1, stride_x1, stride_x2)
        )
    n = start + tl.arange(0, BLOCK_N)
    n_mask = n < end
    m = tl.arange(0, BLOCK_M)
    m_mask = m < rows
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_w
    x_rows = x_ptr + m[:, None].to(tl.int64) * stride_x
    acc = tl.zeros([BLOCK_M, BLOCK_N, BLOCK_K], dtype=tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < k_in
        w = tl.load(
            w_rows + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)
        x = tl.load(
            x_rows + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    y = tl.sum(acc, axis=2)
    tl.store(
        out_ptr + m[:, None].to(tl.int64) * stride_out + n[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def config(rows: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps) for `rows` input rows."""
    block_m = 1 << (rows - 1).bit_length()
    return (block_m, *CONFIGS[block_m])


def segment_gemv(
    inputs: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bounds: tuple[int, ...],
    out: torch.Tensor,
) -> torch.Tensor:
    """Write `out` [M, N] for `weight` [N, K] and `inputs`, one [M, K] tensor
    or three; `bounds` are the inner segment boundaries (rows of `weight`):
    () for one input, 2 or 4 ascending ones for three (module doc)."""
    rows, k_in = inputs[0].shape
    n_out = weight.shape[0]
    assert 1 <= rows <= MAX_ROWS, rows
    assert (len(inputs), len(bounds)) in ((1, 0), (3, 2), (3, 4)), bounds
    assert weight.shape[1] == k_in and weight.stride(1) == 1, weight.shape
    assert out.shape == (rows, n_out) and out.stride(1) == 1, out.shape
    for x in inputs:
        assert x.shape == (rows, k_in) and x.stride(1) == 1, x.shape
    block_m, block_n, block_k, warps = config(rows)
    edges = (0, *bounds, *(n_out,) * (MAX_SEGMENTS - 1 - len(bounds)), n_out)
    assert all(a <= b for a, b in itertools.pairwise(edges)), edges
    starts = [0]
    for a, b in itertools.pairwise(edges):
        starts.append(starts[-1] + -(-(b - a) // block_n))
    grid = (starts[-1],)
    if not grid[0]:
        return out
    x0, x1, x2 = inputs if len(inputs) == 3 else inputs * 3
    _segment_gemv_kernel[grid](
        x0, x1, x2, weight, out,
        x0.stride(0), x1.stride(0), x2.stride(0), weight.stride(0), out.stride(0),
        rows, k_in, n_out, *edges[1:5], *starts[1:5],
        SEGMENTS=1 if len(inputs) == 1 else MAX_SEGMENTS,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=warps,
    )  # fmt: skip
    return out
