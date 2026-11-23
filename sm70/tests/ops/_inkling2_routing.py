# SPDX-License-Identifier: Apache-2.0
"""Cases shared by the CPU (interpreted) and GPU tests of the Inkling 2
routing kernels (inkling2_sm70/ops/routing.py).

The torch routing is ``torch_ref.router`` with ``moe.route``'s padding
masks; ``torch_route`` feeds it given logits through an identity gate,
which reproduces them exactly. ``lowest_id_route`` is the same routing with
top-k ties kept at the lowest expert id (torch.topk's rule on CUDA; CPU
torch.topk breaks ties another way). Each check returns the largest weight
difference after asserting the ids are identical.
"""

from __future__ import annotations

import torch

from vllm_sm70.models.inkling2_sm70.ops import routing, torch_ref

WEIGHT_TOL = 1e-6


def by_id(ids: torch.Tensor, weights: torch.Tensor):
    """ids and weights with each row's ids ascending."""
    ids, order = ids.sort(dim=1)
    return ids, weights.gather(1, order)


def torch_route(logits, bias, top_k, scale, is_padding=None, pad_id=None):
    """(int64 ids, fp32 weights) [M, k], ascending ids per token."""
    experts = logits.shape[1]
    eye = torch.eye(experts, dtype=torch.float32, device=logits.device)
    _, weights, ids = torch_ref.router(logits, eye, bias, top_k, scale)
    if is_padding is not None:
        pad = is_padding.reshape(-1, 1).bool()
        ids = ids.masked_fill(pad, experts)
        weights = weights.masked_fill(pad, 0.0)
    ids, weights = by_id(ids, weights)
    if pad_id is not None:
        ids = ids.masked_fill(ids >= experts, pad_id)
    return ids, weights


def lowest_id_route(logits, bias, top_k, scale, is_padding=None, pad_id=None):
    """``torch_route`` with ties kept at the lowest expert id."""
    experts = logits.shape[1]
    scores = torch.sigmoid(logits)
    ids = torch.sort(-(scores + bias), dim=1, stable=True).indices[:, :top_k]
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * scale
    if is_padding is not None:
        pad = is_padding.reshape(-1, 1).bool()
        ids = ids.masked_fill(pad, experts)
        weights = weights.masked_fill(pad, 0.0)
    ids, weights = by_id(ids, weights)
    if pad_id is not None:
        ids = ids.masked_fill(ids >= experts, pad_id)
    return ids, weights


def check_route(routed, want_ids, want_weights) -> float:
    assert routed.ids.dtype == torch.int32
    assert torch.equal(routed.ids.long().cpu(), want_ids.long().cpu())
    return float((routed.weights - want_weights).abs().max())


def tie_logits(tokens, experts, seed, device="cpu"):
    """Logits from five values: most rows have ties across the top-k edge."""
    g = torch.Generator().manual_seed(seed)
    values = torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0])
    return values[torch.randint(0, 5, (tokens, experts), generator=g)].to(device)


def check_dispatch(routed, x, rows, sorted_ids, offsets, experts):
    """Against the slot sort of moe_tm.py (torch.sort of the flat ids,
    stable; padding = E last, clamped to E - 1)."""
    tokens, top_k = routed.ids.shape
    slots = tokens * top_k
    flat = routed.ids.reshape(-1).long()
    flat = flat.masked_fill((flat < 0) | (flat > experts), experts)
    want_sorted, order = torch.sort(flat, stable=True)
    tokens_of = order // top_k
    clamped = want_sorted.clamp(max=experts - 1)
    assert torch.equal(sorted_ids[:slots].long(), clamped)
    bounds = torch.arange(experts, device=flat.device)
    want_offsets = torch.cat(
        [flat.new_zeros(1), torch.searchsorted(clamped, bounds, right=True)]
    )
    assert torch.equal(offsets[: experts + 1].long(), want_offsets)
    assert torch.equal(rows[:slots], x[tokens_of])
    pos = routed.pos.reshape(-1).long()
    assert torch.equal(pos.sort().values, torch.arange(slots, device=pos.device))
    assert torch.equal(order[pos], torch.arange(slots, device=pos.device))


def combine_reference(y, routed, experts, dtype):
    """The weighted sum as moe_tm.py builds it: index_add_ of the sorted
    rows in fp32, padding slots zeroed."""
    tokens, top_k = routed.ids.shape
    total = torch.zeros(tokens, y.shape[1], dtype=torch.float32, device=y.device)
    ids = routed.ids.long()
    rows = (
        routed.pos.long()
        if routed.pos is not None
        else torch.arange(tokens * top_k, device=y.device).view(tokens, top_k)
    )
    for j in range(top_k):
        part = y[rows[:, j]].float() * routed.weights[:, j : j + 1]
        valid = ((ids[:, j] >= 0) & (ids[:, j] < experts)).unsqueeze(1)
        total += part.masked_fill(~valid, 0.0)
    return total.to(dtype)


def run_dispatch(logits, bias, top_k, scale, is_padding, x, pad_id):
    tokens, experts = logits.shape
    slots = tokens * top_k
    device = logits.device
    rows = torch.full(
        (slots + 3, x.shape[1]), float("nan"), dtype=x.dtype, device=device
    )
    sorted_ids = torch.full((slots + 3,), -7, dtype=torch.int32, device=device)
    offsets = torch.full((experts + 1,), -7, dtype=torch.int32, device=device)
    routed = routing.route(
        logits, bias, top_k, scale, is_padding,
        pad_id=pad_id, dispatch=(x, rows, sorted_ids, offsets),
    )  # fmt: skip
    return routed, rows, sorted_ids, offsets
