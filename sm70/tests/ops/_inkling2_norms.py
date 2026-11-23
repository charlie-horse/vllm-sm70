# SPDX-License-Identifier: Apache-2.0
"""Cases shared by the Inkling 2 Triton norm tests (CPU interpreter and GPU).

Every ``check_*`` runs one kernel of ops/norm.py, attn_res.py, loop.py or
fused.py, or the ``norms`` route built on them, on random inputs of the
model dtype `dtype` and returns the largest error against the torch
reference ops (ops/torch_ref.py), relative to the largest reference value.
``*_outputs`` return what the route gives, to compare across routes.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm_sm70.models.inkling2_sm70 import norms
from vllm_sm70.models.inkling2_sm70.ops import attn_res as attn_res_ops
from vllm_sm70.models.inkling2_sm70.ops import fused as fused_ops
from vllm_sm70.models.inkling2_sm70.ops import loop as loop_ops
from vllm_sm70.models.inkling2_sm70.ops import norm as norm_ops
from vllm_sm70.models.inkling2_sm70.ops import torch_ref

EPS = 1e-6
RANK = 4
# (T, N, Q): row kernel (T * Q <= 256) and blocked kernel, no to three blocks.
ATTN_RES_CASES = [(1, 0, 1), (1, 3, 3), (5, 1, 3), (7, 2, 1), (86, 3, 3), (257, 1, 1)]


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.isfinite(got).all()
    got, want = got.float().cpu(), want.float().cpu()
    return float((got - want).abs().max() / want.abs().max())


class Gate(SimpleNamespace):
    """A message gate with the parameters and the forward of
    Inkling2Sm70MessageGate."""

    def __call__(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return torch_ref.message_gate(
            state,
            message,
            self.down_weight,
            self.forget_weight,
            self.gain_weight,
            self.eps,
        )


class Inputs:
    """Seeded random tensors on one device; `dtype` is the model dtype."""

    def __init__(
        self, device, dtype: torch.dtype, seed: int = 0, rank: int = RANK
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator().manual_seed(seed)
        self.rank = rank

    def randn(self, *shape: int, dtype: torch.dtype | None = None, scale=1.0):
        x = torch.randn(shape, generator=self.generator) * scale
        return x.to(self.device, dtype or self.dtype)

    def weight(self, *shape: int, dtype: torch.dtype | None = None):
        x = 1.0 + 0.2 * torch.randn(shape, generator=self.generator)
        return x.to(self.device, dtype or self.dtype)

    def gates(self, count: int, hidden: int) -> list[Gate]:
        return [
            Gate(
                eps=EPS,
                down_weight=self.randn(self.rank, 2 * hidden, scale=hidden**-0.5),
                gain_weight=self.randn(hidden, self.rank, dtype=torch.float32),
                forget_weight=self.randn(hidden, self.rank, dtype=torch.float32),
            )
            for _ in range(count)
        ]


def check_rms_norm(inputs: Inputs, tokens: int, hidden: int) -> float:
    """`rms_norm` on a row-strided input, in the model dtype and in fp32."""
    wide = inputs.randn(tokens, hidden + 3, scale=3.0)
    x, weight = wide[:, :hidden], inputs.weight(hidden)
    assert not x.is_contiguous() or tokens == 1
    out = norm_ops.rms_norm(x, weight, EPS)
    assert out.dtype == inputs.dtype
    err = rel_err(out, torch_ref.rms_norm(x, weight, EPS))
    out32 = norm_ops.rms_norm(x, weight, EPS, torch.float32)
    assert out32.dtype == torch.float32
    want32 = torch_ref.rms_norm(x.float(), weight.float(), EPS)
    return max(err, rel_err(out32, want32))


def check_attn_res(
    inputs: Inputs,
    tokens: int,
    hidden: int,
    blocks: int,
    queries: int,
    *,
    norm: bool,
    addend: bool,
    stack_normed: bool,
) -> float:
    """`attn_res` on raw block states [N, T, H] with pre-scaled queries (the
    upstream call), or on normed fp32 ones laid out [T, N, H] with `q_scale`
    (the call of norms.py)."""
    dtype = inputs.dtype
    stack = inputs.randn(blocks, tokens, hidden, scale=2.0)
    last = inputs.randn(tokens, hidden)
    extra = inputs.randn(tokens, hidden) if addend else None
    query = inputs.randn(queries, hidden, dtype=torch.float32)
    weight = inputs.weight(queries, hidden) if norm else None

    partial = last.float() if extra is None else last.float() + extra.float()
    states = stack.float().permute(1, 0, 2)
    want = torch_ref.attn_res(states[None], partial[None], query)[0].transpose(0, 1)
    if weight is not None:
        want = torch_ref.rms_norm(want, weight.float()[:, None], EPS).to(dtype)

    scale = hidden**-0.5
    if stack_normed:
        bank = torch_ref.rms_norm(states).contiguous()
        got = attn_res_ops.attn_res(
            bank.transpose(0, 1),
            last,
            extra,
            query,
            weight,
            EPS,
            True,
            stack_normed=True,
            q_scale=scale,
            out_dtype=dtype,
        )
    else:
        got = attn_res_ops.attn_res(
            stack, last, extra, query * scale, weight, EPS, True
        )
    assert got.dtype == (dtype if norm else torch.float32)
    return rel_err(got, want)


def _gate_reference(gate, dest, message, weight, dtype) -> torch.Tensor:
    return torch_ref.rms_norm(gate(dest, message), weight.float(), EPS).to(dtype)


def check_pair_rms_norm(inputs: Inputs, streams: int, tokens: int, hidden: int):
    """`pair_rms_norm`, with one destination broadcast over the streams."""
    dest = inputs.randn(1, tokens, hidden, dtype=torch.float32, scale=2.0)
    dest = dest.expand(streams, -1, -1)
    message = inputs.randn(streams, tokens, hidden, dtype=torch.float32)
    got = loop_ops.pair_rms_norm(dest, message, EPS, inputs.dtype)
    assert got.dtype == inputs.dtype
    want = torch.cat([torch_ref.rms_norm(dest), torch_ref.rms_norm(message)], dim=-1)
    return rel_err(got, want.to(inputs.dtype))


def check_gate_kernels(inputs: Inputs, streams: int, tokens: int, hidden: int):
    """`fused.gate_down` against the pair norm and the down GEMM of the
    message gates, with one destination broadcast over the streams, and
    `fused.gate_up` against the gain and forget GEMMs; all weights in the
    model dtype."""
    gates = inputs.gates(streams, hidden)
    for gate in gates:
        gate.gain_weight = gate.gain_weight.to(inputs.dtype)
        gate.forget_weight = gate.forget_weight.to(inputs.dtype)
    dest = inputs.randn(1, tokens, hidden, dtype=torch.float32, scale=2.0)
    dest = dest.expand(streams, -1, -1)
    message = inputs.randn(streams, tokens, hidden, dtype=torch.float32)
    pair = torch.cat([torch_ref.rms_norm(dest), torch_ref.rms_norm(message)], -1)
    want_low = torch.stack(
        [
            F.linear(pair[i].to(gate.down_weight.dtype), gate.down_weight).float()
            for i, gate in enumerate(gates)
        ]
    )
    low = fused_ops.gate_down(dest, message, [g.down_weight for g in gates], EPS)
    gains = fused_ops.gate_up(
        want_low, [g.gain_weight for g in gates], [g.forget_weight for g in gates]
    )
    want = torch.stack(
        [
            torch.cat(
                [
                    F.linear(want_low[i], gate.gain_weight.float()),
                    F.linear(want_low[i], gate.forget_weight.float()),
                ],
                dim=-1,
            )
            for i, gate in enumerate(gates)
        ]
    )
    assert low.dtype == gains.dtype == torch.float32
    return max(rel_err(low, want_low), rel_err(gains, want))


def check_residual_add(inputs: Inputs, tokens: int, hidden: int) -> float:
    """`fused.residual_add` with and without a base, against the fp32 adds
    of the layer; 0.0 when bit-identical."""
    base, update, addend = (inputs.randn(tokens, hidden) for _ in range(3))
    err = 0.0
    for b in (base, None):
        got = fused_ops.residual_add(b, update, addend, inputs.dtype)
        residual = update.float() if b is None else b.float() + update.float()
        want = (residual + addend).to(inputs.dtype)
        err = max(err, 0.0 if torch.equal(got, want) else rel_err(got, want))
    return err


def check_blend(inputs: Inputs, streams: int, tokens: int, hidden: int) -> float:
    """The message gates and the norm of each stream as norms.py runs them
    (the gate kernels up to fused.MAX_TOKENS tokens, else `pair_rms_norm`
    and torch.bmm; then `blend_rms_norm`), against the message gate and the
    norm of each stream."""
    dtype = inputs.dtype
    gates = inputs.gates(streams, hidden)
    dests = inputs.randn(streams, tokens, hidden, dtype=torch.float32, scale=2.0)
    messages = inputs.randn(streams, tokens, hidden, dtype=torch.float32)
    weight = inputs.weight(streams, hidden)
    got = norms._blend(gates, dests, messages, weight, EPS, dtype)
    assert got.dtype == dtype
    want = torch.stack(
        [
            _gate_reference(gate, dests[i], messages[i], weight[i], dtype)
            for i, gate in enumerate(gates)
        ]
    )
    return rel_err(got, want)


# (name, query rows, norm rows, gates, first message row)
ROUTE_CASES = [
    ("state_one_norm", 0, 1, 0, 0),
    ("state_three_norms", 0, 3, 0, 0),
    ("local_attention", 1, 1, 0, 0),
    ("global_attention", 3, 3, 0, 0),
    ("one_query_three_norms", 1, 3, 0, 0),
    ("local_attention_messages", 1, 1, 3, 0),
    ("global_attention_messages", 3, 3, 3, 0),
    ("mlp_messages", 1, 1, 1, 3),
]


def route_outputs(
    inputs: Inputs, case: tuple, tokens: int, hidden: int, blocks: int
) -> list[torch.Tensor]:
    """What norms.py returns for one sublayer of `case` on the current route:
    the embedding norm, the input streams and, with blocks, the bank."""
    _, num_queries, num_norms, num_gates, first = case
    dtype = inputs.dtype
    state = inputs.randn(tokens, hidden, scale=2.0)
    bank = norms.Bank(state, blocks)
    for _ in range(blocks):
        bank.append(inputs.randn(tokens, hidden))
    query = None
    if num_queries:
        query = inputs.randn(num_queries, hidden, dtype=torch.float32)
    weight = inputs.weight(num_norms, hidden)
    gates = inputs.gates(num_gates, hidden)
    messages = None
    if gates:
        sources = norms.message_sources(bank.states(1), inputs.randn(tokens, hidden))
        message_query = inputs.randn(4, hidden, dtype=torch.float32)
        messages = norms.message_mix(sources, message_query)
    streams = norms.sublayer_inputs(
        bank,
        state,
        query,
        weight,
        EPS,
        dtype,
        gates=gates,
        messages=messages,
        first=first,
    )
    assert all(x.dtype == dtype and x.shape == state.shape for x in streams)
    embedded = norms.rms_norm(state, weight[0], EPS)
    assert embedded.dtype == dtype
    return [embedded, *streams, *([bank.states().clone()] if blocks else [])]


def residual_outputs(
    inputs: Inputs,
    tokens: int,
    hidden: int,
    blocks: int,
    *,
    start: bool,
    gated: bool,
) -> list[torch.Tensor]:
    """What norms.py gives for the MLP sublayer of a layer on the current
    route: the input stream from the `Residual` of the layer input (none at
    a block `start`) and the attention output, with one message gate when
    `gated`, and the layer output after an MLP output."""
    dtype = inputs.dtype
    layer_input = inputs.randn(tokens, hidden, scale=2.0)
    attn_out, mlp_out = inputs.randn(tokens, hidden), inputs.randn(tokens, hidden)
    bank = norms.Bank(layer_input, blocks + 1)
    for _ in range(blocks):
        bank.append(inputs.randn(tokens, hidden))
    query = inputs.randn(1, hidden, dtype=torch.float32)
    weight = inputs.weight(1, hidden)
    gates, messages = inputs.gates(int(gated), hidden), None
    if gated:
        sources = norms.message_sources(bank.states(), inputs.randn(tokens, hidden))
        messages = norms.message_mix(
            sources, inputs.randn(4, hidden, dtype=torch.float32)
        )
    residual = norms.Residual(None if start else layer_input, attn_out, bank.triton)
    (x,) = norms.sublayer_inputs(
        bank,
        residual,
        query,
        weight,
        EPS,
        dtype,
        gates=gates,
        messages=messages,
        first=3,
    )
    assert x.dtype == dtype and x.shape == layer_input.shape
    return [x, residual.add(mlp_out, dtype)]
