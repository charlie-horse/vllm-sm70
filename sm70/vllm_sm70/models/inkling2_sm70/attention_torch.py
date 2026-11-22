# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: torch attention over the paged K/V cache.

`paged_attention` is the runtime torch path of one attention layer visit: an
eager break that reads the owner's metadata from the forward context.
`attention_with_metadata` is the same op on an explicit cache and metadata,
the oracle of the attention kernels. Both read what the prologue produced:
clipped queries, relative-position logits, sink logits, the output gate, and
K/V already written to the cache for every token of the batch.

    logit[q, h, k] = q . k * D^-0.5 + rel_logits[q, h, d],  d = q_pos - k_pos
    (the bias is zero outside 0 <= d < extent)
    keys: 0 <= d, and d < sliding_window in a sliding layer
    one sink column per query and head, fp32 softmax, sink column dropped
    out = (weights @ v) * sigmoid(gate)

Keys are gathered through the block table in steps of at most `key_chunk`
key tokens and folded into a running fp32 softmax, so memory does not grow
with the context. A request with more than one query shares one key range
per `query_chunk` queries (a sliding layer reads the window only); requests
with one query are batched with a key range each, `OWN_KEY_STEPS *
key_chunk` gathered keys per step.

Transient memory of one step, beside the op's inputs and output:

    scores  at most 4 live tensors of query_chunk * key_chunk * heads fp32
            values: 4 * 4 B * 256 * 1024 * 24 = 101 MB at the defaults with
            the 24 heads of an Inkling 2 Mini global layer
    K | V   the gathered keys in the cache dtype and the values in fp32,
            kv_heads * D * 8 B per key: 8 MB for key_chunk keys of a request
            with several queries, 67 MB for the single-query requests

`hf_rounding` is a debug switch: the softmax of all keys runs in one step and
its weights are rounded to the value dtype before they multiply V, as the
reference does (transformers modeling_inkling2_loop.py). Memory then grows
with the context.

Shapes: P padded batch tokens, T real tokens, R requests, D head dim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

if TYPE_CHECKING:
    from .backend import Inkling2AttentionMetadata
    from .cache import Inkling2AttentionCache

__all__ = [
    "DEFAULT_KEY_CHUNK",
    "DEFAULT_QUERY_CHUNK",
    "OWN_KEY_STEPS",
    "attention_with_metadata",
    "gate_and_project",
    "gather_attention",
    "kv_cache_views",
    "paged_attention",
]

DEFAULT_QUERY_CHUNK = 256
DEFAULT_KEY_CHUNK = 1024
# Single-query requests gather this many key chunks per step between them.
OWN_KEY_STEPS = 8
# Key chunk of `hf_rounding`: every key in one step.
_ALL_KEYS = 1 << 62


def kv_cache_views(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V views [num_blocks, block_size, kv_heads, D] of a bound cache
    [num_blocks, kv_heads, block_size, 2 * D] (K | V per head), no copy."""
    if kv_cache.dim() != 4 or kv_cache.shape[-1] != 2 * head_dim:
        raise ValueError(
            f"vllm_sm70: expected a K/V cache (blocks, kv_heads, block_size, "
            f"{2 * head_dim}), got {tuple(kv_cache.shape)}."
        )
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    return key_cache, value_cache


class _Softmax:
    """Running softmax of a set of queries over key steps, in fp32. It starts
    at the sink column, whose value is zero. With `hf_rounding` it takes one
    step and rounds the weights to the value dtype."""

    def __init__(
        self,
        query: torch.Tensor,
        sinks: torch.Tensor,
        groups: int,
        hf_rounding: bool = False,
    ) -> None:
        # query [B, heads, Q, D] -> [B, kv_heads, groups * Q, D]
        self.groups = groups
        self.num_queries = query.shape[2]
        self.query = query.unflatten(1, (-1, groups)).flatten(2, 3)
        self.scaling = query.shape[-1] ** -0.5
        self.hf_rounding = hf_rounding
        self.exact: torch.Tensor | None = None
        self.peak = sinks.float()  # [B, heads, Q]
        self.total = torch.ones_like(self.peak)
        self.acc = self.peak.new_zeros((*self.peak.shape, query.shape[-1]))

    def add(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        allowed: torch.Tensor,
    ) -> None:
        """Fold in N keys. key, value [B, kv_heads, N, D]; bias
        [B, heads, Q, N]; allowed bool, broadcastable to the bias."""
        scores = torch.matmul(self.query, key.transpose(2, 3)) * self.scaling
        scores = scores.unflatten(2, (self.groups, self.num_queries)).flatten(1, 2)
        scores = (scores + bias).float().masked_fill_(~allowed, float("-inf"))
        if self.hf_rounding:
            if self.exact is not None:
                raise ValueError("vllm_sm70: hf_rounding takes one key step.")
            full = torch.cat([scores, self.peak.unsqueeze(-1)], dim=-1)
            weights = torch.softmax(full, dim=-1)[..., :-1].to(value.dtype)
            exact = torch.matmul(
                weights.unflatten(1, (-1, self.groups)).flatten(2, 3), value
            )
            exact = exact.unflatten(2, (self.groups, self.num_queries))
            self.exact = exact.flatten(1, 2)
            return
        peak = torch.maximum(self.peak, scores.amax(-1))
        probs = scores.sub_(peak.unsqueeze(-1)).exp_()
        carry = torch.exp(self.peak - peak)
        weighted = torch.matmul(
            probs.unflatten(1, (-1, self.groups)).flatten(2, 3), value.float()
        )
        weighted = weighted.unflatten(2, (self.groups, self.num_queries))
        self.acc = self.acc * carry.unsqueeze(-1) + weighted.flatten(1, 2)
        self.total = self.total * carry + probs.sum(-1)
        self.peak = peak

    def result(self) -> torch.Tensor:
        """[B, heads, Q, D], fp32."""
        if self.exact is not None:
            return self.exact.float()
        return self.acc / self.total.unsqueeze(-1)


def _shared_keys(
    query: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    kv_cache: torch.Tensor,
    md: Inkling2AttentionMetadata,
    out: torch.Tensor,
    *,
    request: int,
    start: int,
    stop: int,
    first_position: int,
    query_chunk: int,
    key_chunk: int,
    hf_rounding: bool = False,
) -> None:
    """Batch tokens [start, stop) of one request: each query chunk reads one
    key range, from the first key any of its queries may see to its last
    query's position."""
    heads, head_dim = query.shape[1], query.shape[2]
    groups = heads // kv_cache.shape[1]
    extent = rel_logits.shape[-1]
    window = md.sliding_window
    block_size = md.block_size
    device = query.device
    table = md.block_table[request]
    for lo in range(start, stop, query_chunk):
        hi = min(lo + query_chunk, stop)
        positions = md.positions[lo:hi]
        state = _Softmax(
            query[lo:hi].transpose(0, 1).unsqueeze(0),
            sinks[lo:hi].t().unsqueeze(0),
            groups,
            hf_rounding,
        )
        bias_rows = rel_logits[lo:hi]
        key_stop = first_position + hi - start
        key_start = 0
        if window is not None:
            key_start = max(0, first_position + lo - start - window + 1)
        for k0 in range(key_start, key_stop, key_chunk):
            key_pos = torch.arange(k0, min(k0 + key_chunk, key_stop), device=device)
            blocks = table[torch.div(key_pos, block_size, rounding_mode="floor")]
            states = kv_cache[blocks.long(), :, key_pos % block_size]
            states = states.transpose(0, 1).unsqueeze(0)
            distance = positions.unsqueeze(1) - key_pos.unsqueeze(0)
            allowed = distance >= 0
            if window is not None:
                allowed &= distance < window
            index = distance.clamp(0, extent - 1).unsqueeze(1)
            bias = bias_rows.gather(2, index.expand(-1, heads, -1))
            outside = (distance < 0) | (distance >= extent)
            bias = bias.masked_fill(outside.unsqueeze(1), 0.0)
            state.add(
                states[..., :head_dim],
                states[..., head_dim:],
                bias.transpose(0, 1).unsqueeze(0),
                allowed,
            )
        out[lo:hi] = state.result()[0].transpose(0, 1).to(out.dtype)


def _own_keys(
    query: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    kv_cache: torch.Tensor,
    md: Inkling2AttentionMetadata,
    out: torch.Tensor,
    *,
    tokens: torch.Tensor,
    max_seq_len: int,
    key_chunk: int,
    hf_rounding: bool = False,
) -> None:
    """Batch tokens `tokens`, one per request: each query reads its own keys,
    newest first, so key column j is at distance j."""
    heads, head_dim = query.shape[1], query.shape[2]
    groups = heads // kv_cache.shape[1]
    extent = rel_logits.shape[-1]
    block_size = md.block_size
    num_keys = max_seq_len
    if md.sliding_window is not None:
        num_keys = min(num_keys, md.sliding_window)
    step = min(num_keys, key_chunk)
    rows_per_step = max(1, OWN_KEY_STEPS * key_chunk // max(step, 1))
    for row in range(0, tokens.shape[0], rows_per_step):
        chunk = tokens[row : row + rows_per_step]
        positions = md.positions[chunk].unsqueeze(1)
        requests = md.seq_idx[chunk].long().unsqueeze(1)
        state = _Softmax(
            query[chunk].unsqueeze(2), sinks[chunk].unsqueeze(2), groups, hf_rounding
        )
        bias_rows = rel_logits[chunk]
        for j0 in range(0, num_keys, step):
            j1 = min(j0 + step, num_keys)
            offsets = torch.arange(j0, j1, device=query.device)
            key_pos = positions - offsets
            allowed = key_pos >= 0
            key_pos = key_pos.clamp(min=0)
            blocks = md.block_table[
                requests, torch.div(key_pos, block_size, rounding_mode="floor")
            ]
            states = kv_cache[blocks.long(), :, key_pos % block_size].transpose(1, 2)
            bias = bias_rows.new_zeros((chunk.shape[0], heads, j1 - j0))
            if j0 < extent:
                covered = min(j1, extent) - j0
                bias[..., :covered] = bias_rows[..., j0 : j0 + covered]
            state.add(
                states[..., :head_dim],
                states[..., head_dim:],
                bias.unsqueeze(2),
                allowed[:, None, None, :],
            )
        out[chunk] = state.result()[:, :, 0].to(out.dtype)


def gather_attention(
    query: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    kv_cache: torch.Tensor,
    md: Inkling2AttentionMetadata,
    out: torch.Tensor,
    *,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    key_chunk: int = DEFAULT_KEY_CHUNK,
    hf_rounding: bool = False,
) -> torch.Tensor:
    """Attention output before the gate, written to `out` [P, heads, D].

    query [P, heads, D] clipped; rel_logits [P, heads, extent], indexed by
    distance; sinks [P, heads]; kv_cache [num_blocks, kv_heads, block_size,
    2 * D] holding K | V of every key, the batch's own tokens included.
    Rows from `md.num_actual_tokens` on are zeroed.
    """
    if query_chunk < 1 or key_chunk < 1:
        raise ValueError("vllm_sm70: attention chunk sizes must be positive.")
    heads, head_dim = query.shape[1], query.shape[2]
    kv_heads = kv_cache.shape[1]
    if kv_cache.dim() != 4 or kv_cache.shape[-1] != 2 * head_dim or heads % kv_heads:
        raise ValueError(
            f"vllm_sm70: K/V cache {tuple(kv_cache.shape)} does not fit "
            f"{heads} query heads of size {head_dim}."
        )
    if kv_cache.shape[2] != md.block_size:
        raise ValueError(
            f"vllm_sm70: K/V cache blocks hold {kv_cache.shape[2]} tokens, "
            f"the metadata has block size {md.block_size}."
        )
    if hf_rounding:
        key_chunk = _ALL_KEYS
    num_tokens = md.num_actual_tokens
    out[num_tokens:] = 0
    query_lens = np.asarray(md.query_lens_cpu[: md.num_reqs])
    seq_lens = np.asarray(md.seq_lens_cpu[: md.num_reqs])
    starts = np.concatenate(([0], np.cumsum(query_lens)))
    single = np.flatnonzero(query_lens == 1)
    if single.size:
        _own_keys(
            query,
            rel_logits,
            sinks,
            kv_cache,
            md,
            out,
            tokens=torch.as_tensor(starts[single], device=query.device),
            max_seq_len=int(seq_lens[single].max()),
            key_chunk=key_chunk,
            hf_rounding=hf_rounding,
        )
    for request in np.flatnonzero(query_lens > 1).tolist():
        start, stop = int(starts[request]), int(starts[request + 1])
        _shared_keys(
            query,
            rel_logits,
            sinks,
            kv_cache,
            md,
            out,
            request=request,
            start=start,
            stop=stop,
            first_position=int(seq_lens[request]) - (stop - start),
            query_chunk=query_chunk,
            key_chunk=key_chunk,
            hf_rounding=hf_rounding,
        )
    return out


def gate_and_project(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    out_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """`attn_output` [P, heads, D] times sigmoid(`gate` [P, heads * D]) in
    fp32, rounded to the dtype of `attn_output`, then the output projection
    `out_weight` [H, heads * D] when given. Written to `output`."""
    gated = (attn_output.flatten(1).float() * torch.sigmoid(gate.float())).to(
        attn_output.dtype
    )
    if out_weight is None:
        output.copy_(gated)
    elif out_weight.dtype == output.dtype == gated.dtype:
        torch.mm(gated, out_weight.t(), out=output)
    else:
        output.copy_(torch.mm(gated.to(out_weight.dtype), out_weight.t()))
    return output


def attention_with_metadata(
    query: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    gate: torch.Tensor,
    kv_cache: torch.Tensor,
    md: Inkling2AttentionMetadata | None,
    output: torch.Tensor,
    out_weight: torch.Tensor | None = None,
    *,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    key_chunk: int = DEFAULT_KEY_CHUNK,
    hf_rounding: bool = False,
) -> torch.Tensor:
    """One attention layer visit over `kv_cache` and its metadata, written to
    `output`. Arguments as `paged_attention`. Without metadata or a bound
    cache the output is zero. `hf_rounding` (debug): one key step with the
    softmax weights rounded to the value dtype."""
    if md is None or kv_cache.numel() == 0:
        return output.zero_()
    attn_output = gather_attention(
        query,
        rel_logits,
        sinks,
        kv_cache,
        md,
        query.new_empty(query.shape),
        query_chunk=query_chunk,
        key_chunk=key_chunk,
        hf_rounding=hf_rounding,
    )
    return gate_and_project(attn_output, gate, output, out_weight)


@eager_break_during_capture
def paged_attention(
    query: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    gate: torch.Tensor,
    owner: Inkling2AttentionCache,
    output: torch.Tensor,
    out_weight: torch.Tensor | None = None,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    key_chunk: int = DEFAULT_KEY_CHUNK,
    hf_rounding: bool = False,
) -> None:
    """One attention layer visit of the cache owner `owner`, written to
    `output`. The metadata is the current forward's, read on every call.

    query [P, heads, D], rel_logits [P, heads, extent], sinks [P, heads] and
    gate [P, heads * D] (before the sigmoid) come from the prologue, which
    has also written the batch's K/V to the owner's cache. `output` is
    [P, H] with `out_weight` [H, heads * D], else the gated [P, heads * D].
    Padding rows are zero.
    """
    from .backend import cache_metadata

    attention_with_metadata(
        query,
        rel_logits,
        sinks,
        gate,
        owner.kv_cache,
        cache_metadata(owner),
        output,
        out_weight,
        query_chunk=query_chunk,
        key_chunk=key_chunk,
        hf_rounding=hf_rounding,
    )
