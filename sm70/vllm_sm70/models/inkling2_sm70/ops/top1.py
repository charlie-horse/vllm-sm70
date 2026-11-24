# SPDX-License-Identifier: Apache-2.0
"""Greedy top-1 of soft-capped LM-head logits, per vocabulary shard.

    softcap_top1   fp16 logits [rows, n] of one shard -> fp32 candidates
                   [rows, splits, 2]: (max of cap * tanh(x / cap), global
                   token id) of each run of SPLIT_COLS columns
    pick           candidates [rows, N, 2] in vocabulary order -> int64
                   token ids [rows]: the id of the first maximal value

The cap is applied per element in fp32, as the reference's division, tanh
and product. Ties go to the lowest token id: the first maximum inside a
block, the earlier block on equal maxima, the earlier candidate in
``pick``. Token ids travel as fp32 (exact below 2^24).
"""

from __future__ import annotations

import torch

from ._compat import tanh, tl, triton

__all__ = ["BLOCK", "SPLIT_COLS", "pick", "softcap_top1", "num_splits"]

BLOCK = 1024
SPLIT_COLS = 4 * BLOCK
MAX_TOKEN_ID = 1 << 24


@triton.jit
def _softcap_top1_kernel(
    logits_ptr,
    out_ptr,
    stride_row,
    n_cols,
    vocab_start,
    cap,
    SOFT_CAP: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    splits = tl.num_programs(1)
    start = split * SPLIT
    end = tl.minimum(start + SPLIT, n_cols)
    best = -float("inf")
    best_col = start
    for off in tl.static_range(0, SPLIT, BLOCK):
        cols = start + off + tl.arange(0, BLOCK)
        mask = cols < end
        x = tl.load(logits_ptr + row * stride_row + cols, mask, other=0.0)
        x = x.to(tl.float32)
        if SOFT_CAP:
            x = tanh(tl.div_rn(x, cap)) * cap
        x = tl.where(mask, x, -float("inf"))
        block_max = tl.max(x, 0)
        block_col = start + off + tl.argmax(x, 0, tie_break_left=True)
        better = block_max > best
        best_col = tl.where(better, block_col, best_col)
        best = tl.where(better, block_max, best)
    out = out_ptr + (row * splits + split) * 2
    tl.store(out, best)
    tl.store(out + 1, (best_col + vocab_start).to(tl.float32))


@triton.jit
def _pick_kernel(pairs_ptr, out_ptr, n, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    base = pairs_ptr + row * n * 2 + offs * 2
    values = tl.load(base, mask, other=-float("inf"))
    ids = tl.load(base + 1, mask, other=0.0)
    first = tl.argmax(values, 0, tie_break_left=True)
    token = tl.sum(tl.where(offs == first, ids, 0.0), 0)
    tl.store(out_ptr + row, token.to(tl.int64))


def num_splits(n_cols: int) -> int:
    return triton.cdiv(n_cols, SPLIT_COLS)


def softcap_top1(
    logits: torch.Tensor, vocab_start: int, soft_cap: float | None
) -> torch.Tensor:
    """fp32 [rows, num_splits(n), 2] candidates of fp16 `logits` [rows, n],
    whose column 0 is token `vocab_start`."""
    rows, n_cols = logits.shape
    if logits.stride(1) != 1:
        raise ValueError("vllm_sm70: top-1 logits need unit column stride.")
    if vocab_start + n_cols > MAX_TOKEN_ID:
        raise ValueError("vllm_sm70: top-1 token ids above 2^24 are not exact.")
    splits = num_splits(n_cols)
    out = torch.empty((rows, splits, 2), dtype=torch.float32, device=logits.device)
    if rows:
        _softcap_top1_kernel[(rows, splits)](
            logits,
            out,
            logits.stride(0),
            n_cols,
            vocab_start,
            float(soft_cap or 1.0),
            SOFT_CAP=soft_cap is not None,
            SPLIT=SPLIT_COLS,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def pick(pairs: torch.Tensor) -> torch.Tensor:
    """int64 [rows]: the token of the first maximal candidate of each row of
    fp32 `pairs` [rows, N, 2]."""
    pairs = pairs.contiguous()
    rows, n, _ = pairs.shape
    out = torch.empty((rows,), dtype=torch.int64, device=pairs.device)
    if rows:
        _pick_kernel[(rows,)](
            pairs, out, n, BLOCK_N=triton.next_power_of_2(n), num_warps=1
        )
    return out
