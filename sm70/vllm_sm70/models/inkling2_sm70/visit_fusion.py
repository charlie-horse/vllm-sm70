# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: fewer, larger kernels per layer visit of a decode-only
batch (ops/visit.py, ops/routing.py).

    norms.py      a short conv, the residual it feeds and the   visit.conv_mix,
                  next AttnRes mix (or a lone conv and add)     conv_residual
    norms.py      the three message-gate launches               visit.gate_blend
    prologue.py   query_rel and key_value_conv                  visit.prologue
    moe.py        router_logits and route              routing.route(RouterInput)

The convs fuse on a decode-only batch on the kernel routes of norms.py,
sconv.py and fused_small.py (``on_visit``). VLLM_SM70_INKLING_VISIT_FUSION=0
turns the route off; it is read on every eager call. A layer visit on this
route returns a ``Pending`` output (``defer``): its MLP conv and residual add
run inside the next mix, or in ``value()``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from vllm_sm70 import envs

from . import sconv
from .backend import cache_metadata
from .fused_small import on_fused
from .ops import _compat
from .ops import visit as kernels

__all__ = ["SWITCH", "Conv", "Pending", "enabled", "on_decode", "on_visit", "value"]

SWITCH = "VLLM_SM70_INKLING_VISIT_FUSION"
_enabled = envs.environment_variables()[SWITCH]


def enabled() -> bool:
    return _enabled()


def on_decode(x: torch.Tensor, decode_only: bool) -> bool:
    """Whether the ops on `x` of a decode-only batch (or not) may fuse."""
    if not decode_only:
        return False
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()


def on_visit(x: torch.Tensor, decode_only: bool, triton_norms: bool) -> bool:
    """Whether the convs of a visit on `x` fuse with the ops around them:
    ``on_decode`` and the kernel routes of norms.py (`triton_norms`, the
    bank's), sconv.py and fused_small.py."""
    return (
        triton_norms
        and on_decode(x, decode_only)
        and sconv.on_triton(x)
        and on_fused(x, decode_only)
    )


class Conv(NamedTuple):
    """A static short conv not run yet: x [T, C] over stream `stream` of the
    conv state `owner`, weight [C, K] or [C, 1, K]."""

    x: torch.Tensor
    weight: torch.Tensor
    owner: Any
    stream: int


class Pending:
    """A sublayer state that a short conv still has to produce.

    With `state_out` (a layer output): ((base + update) + conv), rounded
    into `state_out`. Otherwise (the residual before the MLP): the conv
    output goes to `conv_out` and the state is base + conv. `base` may be
    None. `mix` runs the conv inside the next AttnRes mix; `resolve` runs it
    on its own."""

    def __init__(
        self,
        conv: Conv,
        base: torch.Tensor | None,
        *,
        update: torch.Tensor | None = None,
        conv_out: torch.Tensor | None = None,
        state_out: torch.Tensor | None = None,
    ) -> None:
        assert (state_out is None) == (update is None) != (conv_out is None)
        self.conv, self.base, self.update = conv, base, update
        self.conv_out, self.state_out = conv_out, state_out
        self.done = False

    @property
    def dtype(self) -> torch.dtype:
        return (self.state_out if self.conv_out is None else self.conv_out).dtype

    @property
    def shape(self) -> torch.Size:
        return self.conv.x.shape

    def fits(self, num_queries: int) -> bool:
        return 0 < num_queries * self.shape[0] <= kernels.MIX_MAX_PROGRAMS

    def mix(
        self,
        stack: torch.Tensor,
        query: torch.Tensor,
        norm_weight: torch.Tensor | None,
        norm_eps: float,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        """norms._mix of the normed states `stack` [T, N, H] and this state:
        [Q, T, H], normed with `norm_weight` in `dtype`, or fp32 without."""
        assert not self.done and self.fits(query.shape[0])
        out_dtype = dtype if norm_weight is not None else torch.float32
        out = self.conv.x.new_empty((query.shape[0], *self.shape), dtype=out_dtype)
        _conv_mix(
            *self.conv, self.base, self.update, self.conv_out, self.state_out,
            stack.transpose(0, 1), query, norm_weight, norm_eps, out,
        )  # fmt: skip
        self.done = True
        return out

    def resolve(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the conv (and add) alone; (last, addend) of the state as
        ``Residual.parts``."""
        if not self.done:
            if self.state_out is not None:
                _conv_residual(*self.conv, self.base, self.update, self.state_out)
            else:
                sconv.short_conv(
                    self.conv.x, self.conv.weight, self.conv.owner,
                    self.conv.stream, self.conv_out, zero_padding=False,
                )  # fmt: skip
            self.done = True
        if self.state_out is not None:
            return self.state_out, None
        if self.base is None:
            return self.conv_out, None
        return self.base, self.conv_out


def value(state: torch.Tensor | Pending) -> torch.Tensor:
    """The tensor of a layer output, run when still pending."""
    if isinstance(state, Pending):
        return state.resolve()[0]
    return state


def _real_rows(x: torch.Tensor, owner: Any) -> tuple[Any, torch.Tensor, int]:
    metadata = cache_metadata(owner)
    sconv._block_tokens(owner.kv_cache, metadata)
    return metadata, owner.kv_cache, sconv._real_rows(x, metadata)


@eager_break_during_capture
def _conv_mix(
    x: torch.Tensor,
    weight: torch.Tensor,
    owner: Any,
    stream: int,
    base: torch.Tensor | None,
    update: torch.Tensor | None,
    conv_out: torch.Tensor | None,
    state_out: torch.Tensor | None,
    stack: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    out: torch.Tensor,
) -> None:
    metadata, cache, rows = _real_rows(x, owner)
    kernels.conv_mix(
        x, weight, cache, metadata, owner.stream_ranges[stream], rows,
        base, update, conv_out, state_out, stack, query, norm_weight, norm_eps, out,
    )  # fmt: skip


@eager_break_during_capture
def _conv_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    owner: Any,
    stream: int,
    base: torch.Tensor | None,
    update: torch.Tensor,
    out: torch.Tensor,
) -> None:
    metadata, cache, rows = _real_rows(x, owner)
    kernels.conv_residual(
        x, weight, cache, metadata, owner.stream_ranges[stream], rows, base, update,
        out,
    )  # fmt: skip
