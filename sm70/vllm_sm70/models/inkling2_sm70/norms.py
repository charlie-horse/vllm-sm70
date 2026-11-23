# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the row-wise ops between the sublayers, on the vendored
Triton kernels (ops/norm.py, attn_res.py, loop.py) and the gate and residual
kernels of ops/fused.py, or on torch (ops/torch_ref.py).

    rms_norm          weighted RMS norm of one stream
    normed_state      one key of the AttnRes mixes
    Bank              the keys of one forward, in one buffer, and the route
                      of the forward's ops
    Residual          the fp32 residual stream inside a layer
    message_sources   what the message mixes of a loop pass read
    message_mix       the messages of one core layer
    sublayer_inputs   AttnRes mix, message gates and RMS norm of the input
                      streams of a sublayer

Route (``on_triton``): the kernels run on a decode-only batch
(backend.is_decode_only), the torch ops on every other batch.
VLLM_SM70_INKLING_TRITON_NORMS=0 puts every batch on the torch ops, =1 every
batch on the kernels. Tensors off the GPU take the torch ops, except in
Triton's interpreter. A forward's route is fixed when its ``Bank`` is made;
a replayed CUDA graph keeps the route it was captured with.

On the kernels one launch replaces the torch ops of a mix with its norm, and
a layer's residual add is summed inside the kernels that read it. The
message gates of a batch of at most ``fused.MAX_TOKENS`` tokens are three
launches (``fused.gate_down``, ``fused.gate_up``, ``loop.blend_rms_norm``);
larger batches run the two GEMMs on torch.bmm between ``loop.pair_rms_norm``
and the blend.
On the route of visit_fusion.py a mix over a ``Pending`` state runs the
short conv that produces it in the same launch (ops/visit.py ``conv_mix``)
and the three gate launches are one (``gate_blend``).

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
from .ops import fused as fused_ops
from .ops import loop as loop_ops
from .ops import norm as norm_ops
from .ops import visit as visit_ops
from .visit_fusion import Pending, on_decode

__all__ = [
    "SWITCH",
    "Bank",
    "Residual",
    "message_mix",
    "message_sources",
    "normed_state",
    "on_triton",
    "rms_norm",
    "sublayer_inputs",
]

SWITCH = "VLLM_SM70_INKLING_TRITON_NORMS"
_mode = envs.environment_variables()[SWITCH]

# Unit norm weights [H] fp32 per (device, H), for the weightless norm.
_ONES: dict[tuple[torch.device, int], torch.Tensor] = {}


def on_triton(x: torch.Tensor, decode_only: bool = False) -> bool:
    """Whether the ops on `x`, of a decode-only batch or not, run the
    kernels (module docstring)."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    mode = _mode()
    return decode_only if mode is None else mode


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


def normed_state(x: torch.Tensor, triton: bool | None = None) -> torch.Tensor:
    """[T, 1, H] fp32: the unweighted RMS norm of x [T, H], one AttnRes key;
    on the kernels when `triton` (None: ``on_triton(x)``)."""
    if on_triton(x) if triton is None else triton:
        return norm_ops.rms_norm(x, _ones(x), 1e-6, torch.float32).unsqueeze(-2)
    return torch_ref.rms_norm(x.float()).unsqueeze(-2)


class Bank:
    """The normed block states of one forward, fp32, in one buffer
    [T, capacity, H]: `count` states are kept and the slot after them is
    scratch for the current state of a mix, so no visit copies the bank.
    `triton`, the route of the forward: ``on_triton(like, decode_only)``."""

    def __init__(
        self, like: torch.Tensor, capacity: int, *, decode_only: bool = False
    ) -> None:
        num_tokens, hidden_size = like.shape
        self.buffer = like.new_empty(
            (num_tokens, capacity + 1, hidden_size), dtype=torch.float32
        )
        self.count = 0
        self.triton = on_triton(like, decode_only)
        # The message gates run as one launch (visit_fusion.py).
        self.fused_gates = self.triton and on_decode(like, decode_only)

    def states(self, start: int = 0) -> torch.Tensor:
        """[T, count - start, H]: the kept states from `start` on, a view."""
        return self.buffer[:, start : self.count]

    def with_state(self, x: torch.Tensor) -> torch.Tensor:
        """[T, count + 1, H]: the kept states and the normed `x` [T, H], a
        view that is valid until the next call or append."""
        self.buffer[:, self.count] = normed_state(x, self.triton)[:, 0]
        return self.buffer[:, : self.count + 1]

    def append(self, x: torch.Tensor) -> None:
        """Keep the normed `x` [T, H] as the next block state."""
        self.with_state(x)
        self.count += 1


class Residual:
    """The fp32 residual stream inside a layer: `base` [T, H] (the layer
    input; None at a block start) plus `update` [T, H] (the attention
    output), on the route `triton`. The torch route sums it here (`value`);
    on the kernels `sublayer_inputs` and `add` read the two parts."""

    def __init__(
        self, base: torch.Tensor | None, update: torch.Tensor, triton: bool
    ) -> None:
        self.base, self.update = base, update
        self.value: torch.Tensor | None = None
        if not triton:
            self.value = update.float()
            if base is not None:
                self.value = base.float() + self.value

    def parts(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(last, addend): the residual is last + addend, or last alone."""
        if self.base is None:
            return self.update.contiguous(), None
        return self.base.contiguous(), self.update.contiguous()

    def add(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """The layer output: residual + x [T, H], rounded to `dtype`."""
        if self.value is not None:
            return (self.value + x).to(dtype)
        last, addend = self.parts()
        base, update = (None, last) if addend is None else (last, addend)
        return fused_ops.residual_add(base, update, x.contiguous(), dtype)


def message_sources(
    bank: torch.Tensor, state: torch.Tensor, triton: bool | None = None
) -> Any:
    """The sources of a pass's message mixes: the normed block states `bank`
    [T, N, H] fp32 of the previous pass and its output `state` [T, H]; on
    the kernels when `triton` (None: ``on_triton(state)``). `bank` is
    copied: the next pass overwrites a `Bank` view."""
    if on_triton(state) if triton is None else triton:
        return bank.clone(), state.contiguous()
    return torch.cat([bank, normed_state(state, False)], dim=-2)


def message_mix(sources: Any, query: torch.Tensor) -> torch.Tensor:
    """The fp32 messages of the streams `query` [S, H] fp32 has a row for."""
    if isinstance(sources, tuple):
        bank, state = sources
        return _mix(bank, state, None, query, None, 0.0, None)
    return torch_ref.attn_res_mix(sources, query)


def _mix(
    bank: torch.Tensor,
    state: torch.Tensor,
    addend: torch.Tensor | None,
    query: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """[Q, T, H]: the AttnRes mixes over the current state `state` (plus
    `addend`) in fp32, or normed with `norm_weight` [Q, H] in `dtype`."""
    return attn_res_ops.attn_res(
        bank.transpose(0, 1),
        state,
        addend,
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
    fused: bool = False,
) -> torch.Tensor:
    """[S, T, H] `dtype`: the message gates `gates` on fp32 [S, T, H] `dests`
    and `messages`, each output normed with its row of `norm_weight`; one
    launch when `fused` (visit_fusion.py) and the batch is small."""
    rows = dests.shape[0] * dests.shape[1]
    small = dests.shape[1] <= fused_ops.MAX_TOKENS
    if fused and small and rows <= visit_ops.GATE_MAX_ROWS:
        return visit_ops.gate_blend(
            dests, messages, [gate.down_weight for gate in gates],
            [gate.gain_weight for gate in gates],
            [gate.forget_weight for gate in gates],
            gates[0].eps, norm_weight, eps, dtype,
        )  # fmt: skip
    if dests.shape[1] <= fused_ops.MAX_TOKENS:
        low = fused_ops.gate_down(
            dests, messages, [gate.down_weight for gate in gates], gates[0].eps
        )
        gains = fused_ops.gate_up(
            low,
            [gate.gain_weight for gate in gates],
            [gate.forget_weight for gate in gates],
        )
        return loop_ops.blend_rms_norm(dests, messages, gains, norm_weight, eps, dtype)
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
    state: torch.Tensor | None,
    addend: torch.Tensor | None,
    query: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    gates: Sequence[Any],
    messages: torch.Tensor | None,
    first: int,
    pending: Pending | None = None,
    fused_gates: bool = False,
) -> list[torch.Tensor]:
    rows = norm_weight.shape[0]
    assert query is not None or addend is None
    assert (pending is None) != (state is None)
    if pending is None:
        state = state.contiguous()

    def mix(
        weight: torch.Tensor | None, mix_eps: float, out_dtype: torch.dtype | None
    ) -> torch.Tensor:
        if pending is not None:
            return pending.mix(bank, query, weight, mix_eps, out_dtype)
        return _mix(bank, state, addend, query, weight, mix_eps, out_dtype)

    if not gates:
        if query is None:
            return [
                norm_ops.rms_norm(state, norm_weight[i], eps, dtype)
                for i in range(rows)
            ]
        if query.shape == norm_weight.shape:
            return list(mix(norm_weight, eps, dtype).unbind(0))
    if query is None:
        dests = state.float().unsqueeze(0)
    else:
        dests = mix(None, 0.0, None)
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
        fused_gates,
    )
    return list(blended.unbind(0))


def sublayer_inputs(
    bank: Bank,
    state: torch.Tensor | Residual | Pending,
    query: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
    *,
    gates: Sequence[Any] = (),
    messages: torch.Tensor | None = None,
    first: int = 0,
) -> list[torch.Tensor]:
    """The normed input streams [T, H] of a sublayer, in `dtype`, on the
    route of `bank`.

    bank: the normed block states. state [T, H], a `Residual` of the route
    or a visit_fusion ``Pending`` of the kernel route: the current residual
    state; a mix of a pending state runs its conv. query [Q, H] fp32: the AttnRes mix
    of each stream; None: the state itself. norm_weight [W, H]: the norm of
    each stream. gates: the message gates of the streams, which read the
    rows `first`... of `messages` (`message_mix`); empty in a first pass.
    One mix or norm row serves every stream: len(gates) streams, without
    gates max(Q, W).
    """
    addend = None
    pending = state if isinstance(state, Pending) else None
    if pending is not None:
        if bank.triton and query is not None and pending.fits(query.shape[0]):
            state = None
        else:
            (state, addend), pending = pending.resolve(), None
    elif isinstance(state, Residual):
        if state.value is None:
            state, addend = state.parts()
        else:
            state = state.value
    if bank.triton:
        return _triton_inputs(
            bank.states(),
            state,
            addend,
            query,
            norm_weight,
            eps,
            dtype,
            gates,
            messages,
            first,
            pending,
            bank.fused_gates,
        )
    assert addend is None, "a Residual of the kernel route"
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
