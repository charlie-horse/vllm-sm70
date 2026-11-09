# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2's Triton dot: exact FP32 logits of the reranked candidates.

FORK cf9492a vllm/model_executor/kernels/lm_head/fp32.py:12-43
(``indexed_fp32_logits``): one program per (row, candidate) computes
<x[row], weight[ids[row, c]]> in FP32 from the FP16 head, with no FP16
rounding of the result. Here the dot loops over K in blocks instead of one
next_pow2(K)-wide block. Kept apart from dflash2_rerank.py so that module
imports without vLLM (CPU tests).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

BLOCK = 1024


@triton.jit
def _exact_logits_kernel(
    x_ptr,
    w_ptr,
    ids_ptr,
    out_ptr,
    x_stride,
    ids_stride,
    out_stride,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cand = tl.program_id(1)
    token = tl.load(ids_ptr + row * ids_stride + cand).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + token * K + offs, mask=mask, other=0.0)
        acc += x.to(tl.float32) * w.to(tl.float32)
    tl.store(out_ptr + row * out_stride + cand, tl.sum(acc, axis=0))


def exact_logits(
    x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor
) -> bool:
    """out[m, c] = <x[m], weight[ids[m, c]]> (FP32). False if Triton is absent.

    x [M, K] fp16 (row stride free, unit column stride), weight [N, K]
    contiguous fp16, ids [M, C] int64, out [M, C] fp32.
    """
    if not HAS_TRITON:
        return False
    _exact_logits_kernel[(x.shape[0], ids.shape[1])](
        x,
        weight,
        ids,
        out,
        x.stride(0),
        ids.stride(0),
        out.stride(0),
        K=weight.shape[1],
        BLOCK=BLOCK,
        num_warps=4,
    )
    return True
