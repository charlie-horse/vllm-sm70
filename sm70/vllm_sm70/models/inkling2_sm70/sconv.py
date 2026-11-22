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
(UP models/inkling2/nvidia/ops/sconv.py ``_tap_sum``). A tap that is zero by
the rule is selected, not multiplied, so a non-finite value of another
request stays out of the sum.

Shapes: T batch tokens (padded), C = heads * stream width channels, K taps.
The metadata covers the first n = ``num_actual_tokens`` tokens (backend.py);
the ops read those rows of `x` and return zero rows after them. In a
decode-only batch n is the length of the slot mapping instead, a static
shape, and the ops read and write the cache without a host sync: the taps
are gathered for every row and selected, and a row with a negative slot
writes the null-block slot 0 back unchanged. ``short_conv`` zeroes the
output rows of padding there.

Switch: ``short_conv`` and the prologue's K/V convs run one Triton kernel per
call (ops/paged_conv.py) unless VLLM_SM70_INKLING_TRITON_SCONV=0. Tensors off
the GPU take the torch ops below, except in Triton's interpreter.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from vllm_sm70 import envs

from .backend import cache_metadata, is_decode_only
from .ops import _compat
from .ops import paged_conv as kernels

__all__ = [
    "SWITCH",
    "conv_taps",
    "dynamic_conv",
    "on_triton",
    "short_conv",
    "static_conv",
    "store_taps",
]

SWITCH = "VLLM_SM70_INKLING_TRITON_SCONV"
_enabled = envs.environment_variables()[SWITCH]


def on_triton(x: torch.Tensor) -> bool:
    """Whether the convs on `x` run the Triton kernels."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()


def _block_tokens(cache: torch.Tensor, metadata: Any) -> int:
    """Tokens per cache block; the metadata's block table and slots must
    count in the same blocks."""
    if cache.shape[2] != metadata.block_size:
        raise ValueError(
            f"vllm_sm70: conv-state blocks hold {cache.shape[2]} tokens, the "
            f"metadata has block size {metadata.block_size}."
        )
    return cache.shape[2]


def _check_channels(channels: int, cache: torch.Tensor, width: int) -> None:
    if channels != cache.shape[1] * width:
        raise ValueError(
            f"vllm_sm70: {channels} conv channels do not fill "
            f"{cache.shape[1]} heads of width {width}."
        )


def _real_rows(x: torch.Tensor, metadata: Any) -> int:
    """n: the batch tokens the metadata covers (module docstring)."""
    if is_decode_only(metadata):
        return min(x.shape[0], metadata.slot_mapping.shape[0])
    num_tokens = metadata.num_actual_tokens
    if not 0 <= num_tokens <= min(x.shape[0], metadata.slot_mapping.shape[0]):
        raise ValueError(
            f"vllm_sm70: metadata of {num_tokens} tokens "
            f"({metadata.slot_mapping.shape[0]} slots) for {x.shape[0]} rows."
        )
    return num_tokens


def _padded(y: torch.Tensor, x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """`y` [n, C] in rows [0, n) of `out` (new when None), zero after."""
    num_tokens = y.shape[0]
    if out is None:
        if num_tokens == x.shape[0]:
            return y
        out = x.new_empty(x.shape)
    out[:num_tokens] = y
    out[num_tokens:] = 0
    return out


def conv_taps(
    x: torch.Tensor,
    kernel_size: int,
    cache: torch.Tensor,
    metadata: Any,
    stream: tuple[int, int],
) -> list[torch.Tensor]:
    """fp32 taps of the real tokens `x` [n, C]: entry `lag` is the conv input
    `lag` tokens back, [n, C]. `metadata` is the batch's
    ``Inkling2SconvMetadata``, `stream` the (offset, width) of the stream
    inside a cache head."""
    num_tokens, channels = x.shape
    offset, width = stream
    block_tokens = _block_tokens(cache, metadata)
    _check_channels(channels, cache, width)
    positions = metadata.positions[:num_tokens]
    seq_idx = metadata.seq_idx[:num_tokens]
    query_start = metadata.query_start[:num_tokens]
    valid = metadata.slot_mapping[:num_tokens] >= 0
    block_table = metadata.block_table
    row = torch.arange(num_tokens, device=x.device)
    current = x.to(cache.dtype).float()
    zero = current.new_zeros(())
    taps = [torch.where(valid[:, None], current, zero)]
    if is_decode_only(metadata):
        seq_idx = seq_idx.long()
        for lag in range(1, kernel_size):
            source = positions - lag
            inside = valid & (source >= 0)
            same_batch = inside & (row - lag >= query_start)
            source = source.clamp(min=0)
            block = block_table[seq_idx, source // block_tokens].long()
            values = cache[block, :, source % block_tokens, offset : offset + width]
            cached = torch.where(inside[:, None], values.flatten(1).float(), zero)
            previous = current[(row - lag).clamp(min=0)]
            taps.append(torch.where(same_batch[:, None], previous, cached))
        return taps
    for lag in range(1, kernel_size):
        source = positions - lag
        inside = valid & (source >= 0)
        same_batch = inside & (row - lag >= query_start)
        tap = torch.where(same_batch[:, None], current[(row - lag).clamp(min=0)], zero)
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
    """Store the conv inputs of the real tokens `x` [n, C] at their cache
    slots; tokens with a negative slot are skipped."""
    offset, width = stream
    block_tokens = _block_tokens(cache, metadata)
    slots = metadata.slot_mapping[: x.shape[0]]
    if is_decode_only(metadata):
        new = x.to(cache.dtype).unflatten(-1, (cache.shape[1], width))
        _store_rows(cache[..., offset : offset + width], slots, new)
        return
    rows = (slots >= 0).nonzero().squeeze(1)
    slots = slots[rows]
    cache[slots // block_tokens, :, slots % block_tokens, offset : offset + width] = (
        x[rows].to(cache.dtype).unflatten(-1, (cache.shape[1], width))
    )


def _store_rows(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """cache[slot] = row for every row of `rows` [n, heads, width] with a
    slot, without a host sync: a row with a negative slot writes slot 0 of
    the null block back unchanged. `cache` [blocks, heads, tokens, width]."""
    block_tokens = cache.shape[2]
    keep = (slots < 0)[:, None, None]
    slots = slots.clamp(min=0)
    block, token = slots // block_tokens, slots % block_tokens
    cache[block, :, token] = torch.where(keep, cache[block, :, token], rows)


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
    full, x = x, x[: _real_rows(x, metadata)]
    weight = weight.float().reshape(weight.shape[0], -1)
    kernel_size = weight.shape[-1]
    taps = conv_taps(x, kernel_size, cache, metadata, stream)
    acc = taps[kernel_size - 1] * weight[:, 0]
    for k in range(1, kernel_size):
        acc = acc + taps[kernel_size - 1 - k] * weight[:, k]
    store_taps(x, cache, metadata, stream)
    y = ((acc + x.to(cache.dtype).float()) / math.sqrt(kernel_size + 1)).to(x.dtype)
    return _padded(y, full, out)


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
    full, x = x, x[: _real_rows(x, metadata)]
    kernel_size, channels = static_weight.shape
    heads = cache.shape[1]
    static_weight = static_weight.float()
    dynamic_taps = dynamic_taps[: x.shape[0]].float()
    dynamic_taps = dynamic_taps.reshape(x.shape[0], heads, kernel_size)
    taps = conv_taps(x, kernel_size, cache, metadata, stream)
    acc = None
    for lag in range(kernel_size):
        weight = static_weight[lag] + dynamic_taps[:, :, lag].repeat_interleave(
            channels // heads, dim=1
        )
        acc = taps[lag] * weight if acc is None else acc + taps[lag] * weight
    store_taps(x, cache, metadata, stream)
    y = (acc + x.to(cache.dtype).float()) / math.sqrt(2 * kernel_size + 1)
    return _padded(y.to(x.dtype), full, out)


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
    if on_triton(x):
        _block_tokens(owner.kv_cache, metadata)
        _check_channels(x.shape[1], owner.kv_cache, stream_range[1])
        kernels.short_conv(
            x,
            weight,
            dynamic_taps,
            owner.kv_cache,
            metadata,
            stream_range,
            _real_rows(x, metadata),
            out,
        )
    elif dynamic_taps is None:
        static_conv(x, weight, owner.kv_cache, metadata, stream_range, out)
    else:
        dynamic_conv(
            x, weight, dynamic_taps, owner.kv_cache, metadata, stream_range, out
        )
    if is_decode_only(metadata):
        # Padding rows of a decode-only batch are zero, as after n.
        rows = _real_rows(x, metadata)
        out[:rows].masked_fill_((metadata.slot_mapping[:rows] < 0)[:, None], 0.0)
