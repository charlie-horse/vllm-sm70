# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts in torch, with every expert held in the model dtype.

One module per layer, mounted as ``mlp``: the fp32 router (``gate.weight``,
``gate.bias``) and the dequantized expert weights of one TP rank
(``expert_weights.ExpertLayer.dequantize``). ``forward`` is one eager-break op
that writes this rank's partial expert sum into a caller buffer. At TP > 1
each rank holds a contiguous slice of the intermediate units, so the caller
sums the partials of all ranks with one all-reduce.

Rounding follows the reference: routing in fp32, both GEMMs in the model
dtype, the SwiGLU soft caps in fp32, the routing weight applied after W2 and
the sum over a token's experts in fp32, rounded once to the model dtype.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .expert_weights import ExpertLayer, RangeCheck, check_cast_range
from .ops import torch_ref


def _text_config(config: Any) -> Any:
    get = getattr(config, "get_text_config", None)
    return get() if callable(get) else config


class Inkling2Sm70Gate(nn.Module):
    """The top-k router: fp32 ``weight`` [E, H] and selection ``bias`` [E]."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        text = _text_config(config)
        experts, hidden = int(text.n_routed_experts), int(text.hidden_size)
        self.top_k = int(text.num_experts_per_tok)
        self.route_scale = float(text.route_scale)
        self.weight = nn.Parameter(
            torch.empty(experts, hidden, dtype=torch.float32), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(experts, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [M, H] -> fp32 (logits [M, E], weights [M, k], indices [M, k])."""
        return torch_ref.router(x, self.weight, self.bias, self.top_k, self.route_scale)


class Inkling2Sm70Experts(nn.Module):
    """Router and experts of one layer on one TP rank.

    ``w13`` [E, 2 * I / tp, H] holds the gate rows then the up rows and ``w2``
    [E, H, I / tp]; both are filled by ``load_expert_layer``.
    """

    def __init__(
        self,
        config: Any,
        *,
        tp_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        text = _text_config(config)
        self.num_experts = int(text.n_routed_experts)
        self.hidden_size = int(text.hidden_size)
        intermediate = int(text.intermediate_size)
        if intermediate % tp_size:
            raise ValueError(
                f"Inkling 2: expert intermediate size {intermediate} "
                f"doesn't split over {tp_size} ranks"
            )
        self.local_intermediate = intermediate // tp_size
        self.gate_cap = float(text.mlp_gate_softcap)
        self.up_cap = float(text.mlp_up_softcap)
        if self.gate_cap <= 0 or self.up_cap <= 0:
            raise ValueError(
                "Inkling 2: mlp_gate_softcap and mlp_up_softcap must be positive, "
                f"got {self.gate_cap} and {self.up_cap}"
            )
        self.gate = Inkling2Sm70Gate(text)
        e, h, i = self.num_experts, self.hidden_size, self.local_intermediate
        self.register_buffer(
            "w13", torch.empty(e, 2 * i, h, dtype=dtype), persistent=False
        )
        self.register_buffer("w2", torch.empty(e, h, i, dtype=dtype), persistent=False)

    def load_expert_layer(
        self, layer: ExpertLayer, range_check: RangeCheck | None = check_cast_range
    ) -> None:
        """Dequantize `layer` (already sliced to this rank) into w13 and w2;
        `range_check` sees every cast block (expert_weights.py)."""
        w13, w2 = layer.dequantize(self.w13.dtype, range_check=range_check)
        for name, got, buffer in (("w13", w13, self.w13), ("w2", w2, self.w2)):
            if got.shape != buffer.shape:
                raise ValueError(
                    f"Inkling 2: layer {layer.layer} {name} has shape "
                    f"{tuple(got.shape)}, expected {tuple(buffer.shape)}"
                )
            buffer.copy_(got)

    def route(
        self, x: torch.Tensor, is_padding: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 (logits [M, E], weights [M, k], indices [M, k]) of x [M, H].

        Rows where the bool `is_padding` [M] is set get the expert id E (no
        expert) and weight 0.
        """
        logits, weights, indices = self.gate(x)
        if is_padding is not None:
            pad = is_padding.reshape(-1, 1).bool()
            indices = indices.masked_fill(pad, self.num_experts)
            weights = weights.masked_fill(pad, 0.0)
        return logits, weights, indices

    def expert_sum(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write this rank's weighted expert sum of x [M, H] into `output`.

        top_k_index, top_k_weights [M, k] as `route` returns them; ids outside
        0..E-1 select no expert.
        """
        k = top_k_index.shape[-1]
        flat = top_k_index.reshape(-1)
        none = self.num_experts
        flat = flat.masked_fill((flat < 0) | (flat > none), none)
        order = torch.argsort(flat, stable=True)
        tokens = order // k
        weights = top_k_weights.reshape(-1)[order].float()
        counts = torch.bincount(flat, minlength=self.num_experts + 1)
        total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        start = 0
        for expert, count in enumerate(counts[: self.num_experts].tolist()):
            if not count:
                continue
            rows = tokens[start : start + count]
            gate, up = F.linear(x[rows], self.w13[expert]).chunk(2, dim=-1)
            act = torch_ref.softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)
            out = F.linear(act, self.w2[expert]).float()
            total.index_add_(0, rows, out * weights[start : start + count, None])
            start += count
        output.copy_(total)

    @eager_break_during_capture
    def forward(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        is_padding: torch.Tensor | None = None,
    ) -> None:
        """x [M, H] in the model dtype -> `output` [M, H], this rank's partial.

        `output` must not alias `x`. Padding rows are written as zeros.
        """
        _, weights, indices = self.route(x, is_padding)
        self.expert_sum(x, indices, weights, output)
