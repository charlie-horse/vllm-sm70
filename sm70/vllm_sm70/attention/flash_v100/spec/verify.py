# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-row expansion into persistent decode buffers (K07).

Port of the smallq -> XQA route of fork cf9492a
vllm/v1/attention/backends/flash_v100/spec/{verifier.py:292-400,
smallq_metadata.py, verify_metadata.py, builder.py}, reduced to what v0.33.0
needs:

A causal query row r with L tokens over a context of S keys (its queries are
positions S-L .. S-1) is exactly L independent decode rows: token j sees the
first S - L + 1 + j keys through the same block table. The fork test
test_sm70_flash_v100_mtp_smallq_exactness.py pins that equivalence.

Capture safety: v0.33 builds attention metadata eagerly every step and
replays the captured forward, which therefore must read only persistent
tensors that build() refreshes. The expanded seq_lens and block-table rows
live in buffers allocated once per builder (outside capture) and are
rewritten in place with device ops (no host sync); forward reads fixed-size
prefixes of them. The decode route itself (XQA / scalar) was already
capture-safe (K06), so uniform verify batches can be captured
(UNIFORM_BATCH) without the K06 downgrade.

A1: on CUDA the expansion is one Triton launch (spec/smallq.py) instead of
~20 torch launches; VLLM_SM70_FLASH_V100_SMALLQ_METADATA=0 (read when the
builder is built) keeps the torch path, which CPU tensors always take.
"""

from __future__ import annotations

import torch


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


class VerifyExpansion:
    """Persistent buffers for token-wise verify rows of one metadata builder."""

    def __init__(
        self, max_rows: int, device: torch.device, max_reqs: int | None = None
    ) -> None:
        self.max_rows = max(int(max_rows), 1)
        self.device = device
        self.seq_lens = torch.zeros(self.max_rows, dtype=torch.int32, device=device)
        self.block_table: torch.Tensor | None = None  # [max_rows, width], lazily
        self._arange = torch.arange(self.max_rows, dtype=torch.int64, device=device)
        # One kernel shape per builder (compiled before capture): rows are
        # requests, at most max_reqs (else max_rows, one token each).
        self.row_block = _next_pow2(int(max_reqs or self.max_rows) + 1)
        self.uses_kernel = False
        if torch.device(device).type == "cuda":
            from vllm_sm70.attention.flash_v100.spec import smallq

            self.uses_kernel = smallq.can_expand(torch.device(device))

    def _ensure_block_table(self, width: int) -> torch.Tensor:
        if self.block_table is None or self.block_table.shape[1] < width:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Flash-V100 verify buffers must be sized before CUDA graph "
                    "capture (run an eager step first)"
                )
            self.block_table = torch.zeros(
                (self.max_rows, width), dtype=torch.int32, device=self.device
            )
        return self.block_table

    def expand(
        self,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_rows: int,
        num_mapped_tokens: int,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill buffers for request rows [0, num_rows) and return views.

        num_mapped_tokens: tokens the rows own (query_start_loc[num_rows],
        known on the host); num_tokens: decode-region token count the forward
        runs (>= num_mapped_tokens; the rest are padding, seq_len 0).
        Returns (seq_lens[num_tokens], block_table[num_tokens, width]).
        """
        if num_tokens > self.max_rows:
            raise ValueError(
                f"{num_tokens} verify tokens exceed the {self.max_rows}-row "
                "buffer (max_num_seqs * (1 + num_speculative_tokens))"
            )
        width = block_table.shape[1]
        table = self._ensure_block_table(width)
        out_seq = self.seq_lens[:num_tokens]
        out_table = table[:num_tokens, :width]
        if (
            num_rows
            and num_mapped_tokens
            and self.uses_kernel
            and num_rows < self.row_block
            and query_start_loc.dtype == seq_lens.dtype == block_table.dtype
            and block_table.dtype == torch.int32
            and block_table.stride(1) == 1
        ):
            from vllm_sm70.attention.flash_v100.spec.smallq import (
                expand_verify_rows,
            )

            expand_verify_rows(
                out_seq,
                out_table,
                query_start_loc,
                seq_lens,
                block_table,
                num_rows=num_rows,
                num_mapped_tokens=num_mapped_tokens,
                num_tokens=num_tokens,
                row_block=self.row_block,
            )
            return out_seq, out_table
        out_seq.zero_()
        if num_rows and num_mapped_tokens:
            qsl = query_start_loc[: num_rows + 1].to(torch.int64)
            qlens = qsl[1:] - qsl[:-1]
            req = torch.repeat_interleave(
                self._arange[:num_rows], qlens, output_size=num_mapped_tokens
            )
            pos = self._arange[:num_mapped_tokens] - qsl[req]
            expanded = seq_lens[:num_rows].to(torch.int64)[req] - qlens[req] + 1 + pos
            out_seq[:num_mapped_tokens].copy_(expanded)
            out_table[:num_mapped_tokens].copy_(block_table.index_select(0, req))
        return out_seq, out_table


def expand_reference(
    query_lens: list[int], seq_lens: list[int]
) -> tuple[list[int], list[int]]:
    """Host reference: (row of each token, visible length of each token)."""
    rows, lens = [], []
    for r, (q, s) in enumerate(zip(query_lens, seq_lens)):
        for j in range(q):
            rows.append(r)
            lens.append(s - q + 1 + j)
    return rows, lens
