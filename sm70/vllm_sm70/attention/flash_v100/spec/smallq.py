# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch verify-row expansion (A1, small-query metadata).

Port of fork cf9492a
vllm/v1/attention/backends/flash_v100/spec/smallq_metadata.py:24-140
(`_sm70_prepare_smallq_decode_metadata_kernel`, the per-group kernel behind
FUSED_SMALLQ_METADATA), reduced to what VerifyExpansion.expand writes. It
replaces the torch expansion (repeat_interleave, index, index_select, adds
and copies: ~20 launches) with one Triton launch per full-attention cache
group. Bea's E2 trace of Qwen3.8 row 4 at TP2 shows 5 groups x ~56 us of
those launches per verify step.

Not ported: the fork's grouped variant (GROUPED_SMALLQ_METADATA, one launch
for every group). It needs a hook in the runner's prepare_attn (upstream
mamba_hybrid.py) and saves only the remaining ~4 launches per step.

Differences from the fork kernel, all to keep VerifyExpansion's contract:
* Padding tokens [num_mapped_tokens, num_tokens) get seq_len 0 and their
  block-table rows are left alone (the fork writes zeros); a zero seq_len
  means the decode kernels never read them.
* Visible length is seq_len - query_len + 1 + j with no max(seq_len,
  query_len) clamp, as in the torch path.
* The fork's query_start_loc refresh is dropped: the Flash-V100 decode
  route does not read query_start_loc.

Kill switch: VLLM_SM70_FLASH_V100_SMALLQ_METADATA=0 restores the torch
expansion (only "0" acts).
"""

from __future__ import annotations

import os

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

KILL_SWITCH = "VLLM_SM70_FLASH_V100_SMALLQ_METADATA"

# Block-table columns copied per loop iteration.
COL_BLOCK = 128


def smallq_metadata_enabled() -> bool:
    """Only "0" acts (the torch expansion runs instead)."""
    return os.environ.get(KILL_SWITCH) != "0"


def can_expand(device: torch.device) -> bool:
    return HAS_TRITON and device.type == "cuda" and smallq_metadata_enabled()


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_mapped_tokens",
        "width",
        "table_stride",
        "out_table_stride",
    ]
)
def _expand_verify_rows_kernel(
    out_seq_ptr,
    out_table_ptr,
    qsl_ptr,
    seq_lens_ptr,
    table_ptr,
    table_stride,
    out_table_stride,
    num_rows,
    num_mapped_tokens,
    width,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    # One program per output token (padding included).
    tok = tl.program_id(0)
    is_real = tok < num_mapped_tokens

    # The row owning `tok` is the number of row ends at or before it. Rows
    # with no tokens have equal boundaries and are skipped. Padding tokens
    # clamp to the last row so every load stays in bounds.
    rows = tl.arange(0, ROW_BLOCK)
    ends = tl.load(qsl_ptr + rows + 1, mask=rows < num_rows, other=0x7FFFFFFF)
    row = tl.sum((tok >= ends).to(tl.int32), axis=0)
    row = tl.minimum(row, num_rows - 1)

    start = tl.load(qsl_ptr + row)
    end = tl.load(qsl_ptr + row + 1)
    seq_len = tl.load(seq_lens_ptr + row)
    visible = seq_len - (end - start) + 1 + (tok - start)
    tl.store(out_seq_ptr + tok, tl.where(is_real, visible, 0))

    cols = tl.arange(0, COL_BLOCK)
    src = table_ptr + row.to(tl.int64) * table_stride
    dst = out_table_ptr + tok.to(tl.int64) * out_table_stride
    for c0 in range(0, width, COL_BLOCK):
        mask = ((c0 + cols) < width) & is_real
        ids = tl.load(src + c0 + cols, mask=mask, other=0)
        tl.store(dst + c0 + cols, ids, mask=mask)


def expand_verify_rows(
    out_seq_lens: torch.Tensor,
    out_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_rows: int,
    num_mapped_tokens: int,
    num_tokens: int,
    row_block: int,
) -> None:
    """Write out_seq_lens[:num_tokens] and out_block_table[:num_mapped_tokens].

    Same contract as VerifyExpansion.expand (the caller checks capacity).
    query_start_loc, seq_lens and block_table are int32 on the device;
    row_block is a power of two above num_rows, fixed per builder so the
    kernel compiles once (before CUDA graph capture).
    """
    assert num_rows >= 1 and num_mapped_tokens >= 1 and num_tokens >= 1
    assert num_rows < row_block
    for name, t in (
        ("query_start_loc", query_start_loc),
        ("seq_lens", seq_lens),
        ("block_table", block_table),
    ):
        if t.dtype != torch.int32:
            raise TypeError(f"small-query expansion needs int32 {name}, got {t.dtype}")
    if block_table.stride(1) != 1 or out_block_table.stride(1) != 1:
        raise ValueError("small-query expansion needs row-major block tables")
    _expand_verify_rows_kernel[(num_tokens,)](
        out_seq_lens,
        out_block_table,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        out_block_table.stride(0),
        num_rows,
        num_mapped_tokens,
        block_table.shape[1],
        ROW_BLOCK=row_block,
        COL_BLOCK=COL_BLOCK,
        num_warps=1,
    )
