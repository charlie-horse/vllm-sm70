# SPDX-License-Identifier: Apache-2.0
"""References and paged-KV builders for the K05 Flash-V100 GPU tests."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FULL = (-1, -1)
# Flint sliding_window=513: the query token plus the 512 keys before it.
SWA513 = (512, 0)


def paged_kv(
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device,
    seed: int,
    *,
    layout: str = "separate",
):
    """fp16 K/V in shuffled physical blocks plus a per-row block table.

    layout="separate": two contiguous [blocks, block_size, H_kv, D] tensors
    (the fork's layout). layout="lbnhc": one upstream v0.33.0 cache, logical
    (blocks, H_kv, block_size, 2D) stored as [blocks, block_size, H_kv, 2D],
    split with fv100.split_lbnhc_kv_cache (K and V interleaved per head).
    Unused blocks hold finite garbage that must never reach the output.
    """
    gen = torch.Generator().manual_seed(seed)
    blocks_per_seq = [max(1, math.ceil(n / block_size)) for n in seq_lens]
    num_blocks = sum(blocks_per_seq) + 3
    physical = torch.randperm(num_blocks, generator=gen)
    block_table = torch.zeros((len(seq_lens), max(blocks_per_seq)), dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(blocks_per_seq):
        block_table[row, :count] = physical[cursor : cursor + count]
        cursor += count
    if layout == "separate":
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        k = torch.randn(shape, generator=gen).to(device, torch.float16)
        v = torch.randn(shape, generator=gen).to(device, torch.float16)
        return k, v, block_table.to(device), None
    assert layout == "lbnhc", layout
    from vllm_sm70.ops import fv100

    storage = torch.randn(
        (num_blocks, block_size, num_kv_heads, 2 * head_dim), generator=gen
    ).to(device, torch.float16)
    logical = storage.permute(0, 2, 1, 3)  # (blocks, H_kv, block_size, 2D)
    k, v = fv100.split_lbnhc_kv_cache(logical, head_dim)
    return k, v, block_table.to(device), logical


def gather_kv(k_cache, v_cache, block_table_row, length):
    """Linear [length, H_kv, D] K/V of one row."""
    block_size = k_cache.shape[1]
    pos = torch.arange(length, device=k_cache.device)
    blocks = block_table_row[pos // block_size].long()
    offs = pos % block_size
    return k_cache[blocks, offs], v_cache[blocks, offs]


def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool = True,
    window: tuple[int, int] = FULL,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """fp32 attention of q [M, H, D] over k, v [N, H_kv, D] via SDPA.

    The M queries are the last M positions. Masks are an additive float
    bias; a learned sink is one extra key with a zero key and value whose
    bias is the sink logit, so the softmax gains exp(sink) and no value.
    """
    m, num_heads, head_dim = q.shape
    n, num_kv_heads, _ = k.shape
    group = num_heads // num_kv_heads
    kk = k.float().repeat_interleave(group, dim=1).transpose(0, 1)  # [H, N, D]
    vv = v.float().repeat_interleave(group, dim=1).transpose(0, 1)
    qq = q.float().transpose(0, 1)  # [H, M, D]
    q_pos = torch.arange(m, device=q.device)[:, None] + max(n - m, 0)
    kv_pos = torch.arange(n, device=q.device)[None, :]
    keep = torch.ones(m, n, dtype=torch.bool, device=q.device)
    if causal:
        keep &= kv_pos <= q_pos
    left, right = window
    if left >= 0:
        keep &= kv_pos >= q_pos - left
    if right >= 0:
        keep &= kv_pos <= q_pos + right
    bias = torch.zeros(m, n, device=q.device).masked_fill(~keep, float("-inf"))
    bias = bias.expand(num_heads, m, n)
    if sinks is not None:
        zeros = torch.zeros(num_heads, 1, head_dim, device=q.device)
        kk = torch.cat([kk, zeros], dim=1)
        vv = torch.cat([vv, zeros], dim=1)
        sink_bias = sinks.float().to(q.device).view(num_heads, 1, 1)
        bias = torch.cat([bias, sink_bias.expand(num_heads, m, 1)], dim=-1)
    out = F.scaled_dot_product_attention(
        qq[None], kk[None], vv[None], attn_mask=bias[None], scale=scale
    )[0]
    return out.transpose(0, 1)  # [M, H, D]


def decode_reference(
    q, k_cache, v_cache, block_table, seq_lens, *, scale, window, sinks
):
    """fp32 decode: row b's single query is position seq_lens[b] - 1."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        n = int(seq_lens[b])
        k, v = gather_kv(k_cache, v_cache, block_table[b], n)
        out[b] = sdpa_reference(
            q[b : b + 1], k, v, scale=scale, causal=True, window=window, sinks=sinks
        )[0]
    return out


def prefill_reference(
    q, k_cache, v_cache, block_table, seq_lens, *, scale, causal, window, sinks
):
    """fp32 prefill of q [B, M, H, D]; row b's queries end at seq_lens[b]."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        n = int(seq_lens[b])
        k, v = gather_kv(k_cache, v_cache, block_table[b], n)
        out[b] = sdpa_reference(
            q[b], k, v, scale=scale, causal=causal, window=window, sinks=sinks
        )
    return out


def sinks_for(num_heads: int, device, seed: int) -> torch.Tensor:
    """Per-head sinks from negligible to dominant, shuffled across heads."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.linspace(-2.0, 10.0, num_heads, dtype=torch.float32)
    return values[torch.randperm(num_heads, generator=gen)].to(device)
