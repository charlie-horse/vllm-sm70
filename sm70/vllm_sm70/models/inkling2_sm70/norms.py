# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the row-wise ops between the sublayers, on the vendored
Triton kernels (ops/norm.py, attn_res.py, loop.py) or on torch
(ops/torch_ref.py).

    rms_norm          weighted RMS norm of one stream
    normed_state      one key of the AttnRes mixes
    Bank              the keys of one forward, in one buffer
    message_sources   what the message mixes of a loop pass read
    message_mix       the messages of one core layer
    sublayer_inputs   AttnRes mix, message gates and RMS norm of the input
                      streams of a sublayer

Switch: the kernels run only with VLLM_SM70_INKLING_TRITON_NORMS=1. Without
it every op takes the torch path, as tensors off the GPU do. One kernel launch
replaces the torch ops of a mix with its norm; the two GEMMs of the message
gates stay on torch.bmm. The switch and ``on_triton`` are read on every
eager call; a replayed CUDA graph keeps the route it was captured with.

The outputs of `message_sources` and `message_mix` are laid out per path and
are read only by `message_mix` and `sublayer_inputs`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_sm70 import envs

from .ops import _compat, torch_ref
from .ops import attn_res as attn_res_ops
from .ops import loop as loop_ops
from .ops import norm as norm_ops

__all__ = [
    "SWITCH",
    "Bank",
    "message_mix",
    "message_sources",
    "normed_state",
    "on_triton",
    "rms_norm",
    "sublayer_inputs",
]

SWITCH = "VLLM_SM70_INKLING_TRITON_NORMS"
_enabled = envs.environment_variables()[SWITCH]

# Unit norm weights [H] fp32 per (device, H), for the weightless norm.
_ONES: dict[tuple[torch.device, int], torch.Tensor] = {}


def on_triton(x: torch.Tensor) -> bool:
    """Whether the ops on `x` run the Triton kernels."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()


def _ones(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.shape[-1])
    if key not in _ONES:
        _ONES[key] = torch.ones(key[1], dtype=torch.float32, device=x.device)
    return _ONES[key]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm of x [T, H] with weight [H], in the dtype of x."""
    if on_triton(x):
        return norm_ops.rms_norm(x, weight, eps)
    return torch_ref.rms_norm(x, weight, eps)


def normed_state(x: torch.Tensor) -> torch.Tensor:
    """[T, 1, H] fp32: the unweighted RMS norm of x [T, H], one AttnRes key."""
    if on_triton(x):
        return norm_ops.rms_norm(x, _ones(x), 1e-6, torch.float32).unsqueeze(-2)
    return torch_ref.rms_norm(x.float()).unsqueeze(-2)


class Bank:
    """The normed block states of one forward, fp32, in one buffer
    [T, capacity, H]: `count` states are kept and the slot after them is
    scratch for the current state of a mix, so no visit copies the bank."""

    def __init__(self, like: torch.Tensor, capacity: int) -> None:
        num_tokens, hidden_size = like.shape
        self.buffer = like.new_empty(
            (num_tokens, capacity + 1, hidden_size), dtype=torch.float32
        )
        self.count = 0

    def states(self, start: int = 0) -> torch.Tensor:
        """[T, count - start, H]: the kept states from `start` on, a view."""
        return self.buffer[:, start : self.count]

    def with_state(self, x: torch.Tensor) -> torch.Tensor:
        """[T, count + 1, H]: the kept states and the normed `x` [T, H], a
        view that is valid until the next call or append."""
        self.buffer[:, self.count] = normed_state(x)[:, 0]
        return self.buffer[:, : self.count + 1]

    def append(self, x: torch.Tensor) -> None:
        """Keep the normed `x` [T, H] as the next block state."""
        self.with_state(x)
        self.count += 1


def message_sources(bank: torch.Tensor, state: torch.Tensor) -> Any:
    """The sources of a pass's message mixes: the normed block states `bank`
    [T, N, H] fp32 of the previous pass and its output `state` [T, H].
    `bank` is copied: the next pass overwrites a `Bank` view."""
    if on_triton(state):
        return bank.clone(), state.contiguous()
    return torch.cat([bank, normed_state(state)], dim=-2)


def message_mix(sources: Any, query: torch.Tensor) -> torch.Tensor:
    """The fp32 messages of the streams `query` [S, H] fp32 has a row for."""
    if isinstance(sources, tuple):
        bank, state = sources
        return _mix(bank, state, query, None, 0.0, None)
    return torch_ref.attn_res_mix(sources, query)


def _mix(
    bank: torch.Tensor,
    state: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """[Q, T, H]: the AttnRes mixes in fp32, or normed with `norm_weight`
    [Q, H] in `dtype`."""
    return attn_res_ops.attn_res(
        bank.transpose(0, 1),
        state,
        None,
        query,
        norm_weight,
        eps,
        True,
        stack_normed=True,
        q_scale=state.shape[-1] ** -0.5,
        out_dtype=dtype,
    )


def _blend(
    gates: Sequence[Any],
    dests: torch.Tensor,
    messages: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[S, T, H] `dtype`: the message gates `gates` on fp32 [S, T, H] `dests`
    and `messages`, each output normed with its row of `norm_weight`."""
    if len(gates) == 1:
        down = gates[0].down_weight.unsqueeze(0)
    else:
        down = torch.stack([gate.down_weight for gate in gates])
    gain_forget = torch.cat(
        [w for gate in gates for w in (gate.gain_weight, gate.forget_weight)]
    ).view(len(gates), -1, down.shape[1])
    pair = loop_ops.pair_rms_norm(dests, messages, gates[0].eps, down.dtype)
    low = torch.bmm(pair, down.transpose(1, 2)).float()
    gains = torch.bmm(low, gain_forget.float().transpose(1, 2))
    return loop_ops.blend_rms_norm(dests, messages, gains, norm_weight, eps, dtype)


def _triton_inputs(
    bank: torch.Tensor,
    state: torch.Tensor,
    query: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    gates: Sequence[Any],
    messages: torch.Tensor | None,
    first: int,
) -> list[torch.Tensor]:
    rows = norm_weight.shape[0]
    state = state.contiguous()
    if not gates:
        if query is None:
            return [
                norm_ops.rms_norm(state, norm_weight[i], eps, dtype)
                for i in range(rows)
            ]
        if query.shape == norm_weight.shape:
            return list(_mix(bank, state, query, norm_weight, eps, dtype).unbind(0))
    if query is None:
        dests = state.float().unsqueeze(0)
    else:
        dests = _mix(bank, state, query, None, 0.0, None)
    if not gates:
        return [
            norm_ops.rms_norm(dests[i % len(dests)], norm_weight[i % rows], eps, dtype)
            for i in range(max(len(dests), rows))
        ]
    assert messages is not None
    num = len(gates)
    blended = _blend(
        gates,
        dests.expand(num, -1, -1),
        messages[first : first + num],
        norm_weight.expand(num, -1),
        eps,
        dtype,
    )
    return list(blended.unbind(0))


def sublayer_inputs(
    bank: Bank,
    state: torch.Tensor,
    query: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    *,
    gates: Sequence[Any] = (),
    messages: torch.Tensor | None = None,
    first: int = 0,
) -> list[torch.Tensor]:
    """The normed input streams [T, H] of a sublayer, in `dtype`.

    bank: the normed block states. state [T, H]: the current residual
    state. query [Q, H] fp32: the AttnRes mix of each stream; None:
    the state itself. norm_weight [W, H]: the norm of each stream. gates: the
    message gates of the streams, which read the rows `first`... of
    `messages` (`message_mix`); empty in a first pass. One mix or norm row
    serves every stream: len(gates) streams, without gates max(Q, W).
    """
    if on_triton(state):
        return _triton_inputs(
            bank.states(), state, query, norm_weight, eps, dtype, gates, messages, first
        )
    if query is None:
        streams = [state]
    else:
        states = bank.with_state(state)
        streams = list(torch_ref.attn_res_mix(states, query).unbind(-2))
    num = len(gates) if gates else max(len(streams), len(norm_weight))
    streams = streams * (num // len(streams))
    if gates:
        assert messages is not None
        streams = [
            gate(x, messages[:, first + i])
            for i, (gate, x) in enumerate(zip(gates, streams))
        ]
    return [
        torch_ref.rms_norm(x, norm_weight[i % len(norm_weight)], eps).to(dtype)
        for i, x in enumerate(streams)
    ]
