# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: short convs over the paged conv-state cache, in torch.

The conv-state cache of a layer visit ([blocks, heads, tokens, head_size],
``Inkling2ConvState``) holds the conv input of each stream at every absolute
position. One call convolves the tokens of a batch and stores their inputs:

    tap `lag` tokens back of batch row t, at position p of its request:
      zero           when p - lag < 0 or the row has no slot (padding)
      x[t - lag]     when row t - lag belongs to the same request
      cache[p - lag] otherwise, through the request's block table

    static   y = (sum_k w[c, k] * tap[K - 1 - k] + x) / sqrt(K + 1)
    dynamic  y = (sum_l (w[l, c] + dyn[t, head(c), l]) * tap[l] + x)
                 / sqrt(2 K + 1)

The sums run in fp32; `y` is rounded to the dtype of `x`. Taps are read
before the batch is stored. Tap rule as upstream's Triton kernels
(UP models/inkling2/nvidia/ops/sconv.py ``_tap_sum``).

Shapes: T batch tokens (padded), C = heads * stream width channels, K taps.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .backend import cache_metadata

__all__ = [
    "conv_taps",
    "dynamic_conv",
    "short_conv",
    "static_conv",
    "store_taps",
]


def conv_taps(
    x: torch.Tensor,
    kernel_size: int,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
) -> list[torch.Tensor]:
    """fp32 taps of `x` [T, C]: entry `lag` is the conv input `lag` tokens
    back, [T, C]. `metadata` is the batch's ``Inkling2SconvMetadata``,
    `stream` the (offset, width) of the stream inside a cache head."""
    num_tokens, channels = x.shape
    offset, width = stream
    block_tokens = cache.shape[2]
    if channels != cache.shape[1] * width:
        raise ValueError(
            f"vllm_sm70: {channels} conv channels do not fill "
            f"{cache.shape[1]} heads of width {width}."
        )
    positions = metadata.positions[:num_tokens]
    seq_idx = metadata.seq_idx[:num_tokens]
    query_start = metadata.query_start[:num_tokens]
    valid = metadata.slot_mapping[:num_tokens] >= 0
    block_table = metadata.block_table
    row = torch.arange(num_tokens, device=x.device)
    current = x.to(cache.dtype).float()
    taps = [current * valid[:, None]]
    for lag in range(1, kernel_size):
        source = positions - lag
        inside = valid & (source >= 0)
        same_batch = inside & (row - lag >= query_start)
        tap = current[(row - lag).clamp(min=0)] * same_batch[:, None]
        cached = (inside & ~same_batch).nonzero().squeeze(1)
        if cached.numel():
            source = source[cached]
            block = block_table[seq_idx[cached].long(), source // block_tokens]
            values = cache[
                block.long(), :, source % block_tokens, offset : offset + width
            ]
            tap[cached] = values.reshape(-1, channels).float()
        taps.append(tap)
    return taps


def store_taps(
    x: torch.Tensor, cache: torch.Tensor, metadata: Any, stream: tuple[int, int]
) -> None:
    """Store the conv inputs `x` [T, C] at the tokens' cache slots; tokens
    with a negative slot are skipped."""
    offset, width = stream
    block_tokens = cache.shape[2]
    slots = metadata.slot_mapping[: x.shape[0]]
    rows = (slots >= 0).nonzero().squeeze(1)
    slots = slots[rows]
    cache[slots // block_tokens, :, slots % block_tokens, offset : offset + width] = (
        x[rows].to(cache.dtype).unflatten(-1, (cache.shape[1], width))
    )


def static_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Static conv of `x` [T, C] over one stream of the conv-state cache;
    stores `x`. weight [C, K] or [C, 1, K], oldest tap first. The result is
    written to `out` when given."""
    weight = weight.float().reshape(weight.shape[0], -1)
    kernel_size = weight.shape[-1]
    taps = conv_taps(x, kernel_size, cache, metadata, stream)
    acc = taps[kernel_size - 1] * weight[:, 0]
    for k in range(1, kernel_size):
        acc = acc + taps[kernel_size - 1 - k] * weight[:, k]
    store_taps(x, cache, metadata, stream)
    y = ((acc + x.to(cache.dtype).float()) / math.sqrt(kernel_size + 1)).to(x.dtype)
    return y if out is None else out.copy_(y)


def dynamic_conv(
    x: torch.Tensor,
    static_weight: torch.Tensor,
    dynamic_taps: torch.Tensor,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dynamic conv of `x` [T, C] over one stream of the conv-state cache;
    stores `x`. static_weight [K, C] and dynamic_taps [T, heads, K] or
    [T, heads * K], newest tap first; a head's taps apply to its C / heads
    channels. The result is written to `out` when given."""
    kernel_size, channels = static_weight.shape
    heads = cache.shape[1]
    static_weight = static_weight.float()
    dynamic_taps = dynamic_taps.float().reshape(x.shape[0], heads, kernel_size)
    taps = conv_taps(x, kernel_size, cache, metadata, stream)
    acc = None
    for lag in range(kernel_size):
        weight = static_weight[lag] + dynamic_taps[:, :, lag].repeat_interleave(
            channels // heads, dim=1
        )
        acc = taps[lag] * weight if acc is None else acc + taps[lag] * weight
    store_taps(x, cache, metadata, stream)
    y = (acc + x.to(cache.dtype).float()) / math.sqrt(2 * kernel_size + 1)
    y = y.to(x.dtype)
    return y if out is None else out.copy_(y)


@eager_break_during_capture
def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    owner: Any,
    stream: int,
    out: torch.Tensor,
    dynamic_taps: torch.Tensor | None = None,
) -> None:
    """Conv of `x` [T, C] over stream `stream` of the conv state `owner`
    (``Inkling2ConvState``) with the current forward's metadata, into `out`.
    Dynamic when `dynamic_taps` is given. A forward without conv-state
    metadata copies `x`."""
    metadata = cache_metadata(owner)
    if metadata is None:
        out.copy_(x)
        return
    stream_range = owner.stream_ranges[stream]
    if dynamic_taps is None:
        static_conv(x, weight, owner.kv_cache, metadata, stream_range, out)
    else:
        dynamic_conv(
            x, weight, dynamic_taps, owner.kv_cache, metadata, stream_range, out
        )
