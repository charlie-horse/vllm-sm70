# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the modules of one decoder layer, on the torch paths.

    Inkling2Sm70Attention      projections, prologue, paged attention, o_proj
    Inkling2Sm70MessageGate    cross-pass gate of one input stream
    Inkling2Sm70DecoderLayer   one layer visit: AttnRes mixes, message gates,
                               norms, attention, experts and both short convs
    Inkling2Sm70AttnRes        the learned queries of the AttnRes mixes

Parameters carry the names, shapes and dtypes of the weight map
(weights.py); every module creates them from the map's specs. Tensors are
[T, ...] over the padded batch tokens, no batch dim.

Rounding as the reference (transformers modeling_inkling2_loop.py): the
residual stream and the block states are in the model dtype between visits;
AttnRes mixes, message gates, norms and residual adds run in fp32; each norm
rounds to the model dtype, and so do the conv outputs and the expert sum
before the fp32 add.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .attention_torch import paged_attention
from .backend import cache_metadata
from .cache import STREAM_ATTN, STREAM_MLP
from .moe import Inkling2Sm70Experts
from .ops.torch_ref import attn_res_mix, message_gate, rms_norm
from .prologue import attention_prologue, project_qkvr
from .sconv import short_conv
from .weights import MESSAGE_STREAMS, ParamSpec, qkvr_sizes

__all__ = [
    "Inkling2Sm70Attention",
    "Inkling2Sm70AttnRes",
    "Inkling2Sm70DecoderLayer",
    "Inkling2Sm70MessageGate",
    "Weights",
    "all_reduce",
    "mix",
    "normed_state",
    "padding_mask",
]

Specs = Mapping[str, ParamSpec]


def _parameter(specs: Specs, name: str) -> nn.Parameter:
    spec = specs[name]
    return nn.Parameter(torch.empty(spec.shape, dtype=spec.dtype), requires_grad=False)


class Weights(nn.Module):
    """The parameters `<prefix>.<leaf>` of the weight map, as attributes."""

    def __init__(self, specs: Specs, prefix: str, leaves: tuple[str, ...]) -> None:
        super().__init__()
        for leaf in leaves:
            setattr(self, leaf, _parameter(specs, f"{prefix}.{leaf}"))


def all_reduce(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    """The sum of the ranks' partials `x`; `x` itself on one rank."""
    if tp_size == 1:
        return x
    from vllm.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x)


def normed_state(x: torch.Tensor) -> torch.Tensor:
    """[T, 1, H] fp32: the unweighted RMS norm of x [T, H], one AttnRes key."""
    return rms_norm(x.float()).unsqueeze(-2)


def mix(bank: torch.Tensor, partial: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """AttnRes mix [T, Q, H] fp32 of the normed block states `bank`
    [T, N, H] and the current state `partial` [T, H] by `query` [Q, H]."""
    return attn_res_mix(torch.cat([bank, normed_state(partial)], dim=-2), query)


@eager_break_during_capture
def padding_mask(owner: Any, out: torch.Tensor) -> None:
    """Write into the bool `out` [T] which batch tokens are padding (no slot
    in the cache of `owner`). All false in a forward without metadata."""
    metadata = cache_metadata(owner)
    if metadata is None:
        out.zero_()
    else:
        out.copy_(metadata.slot_mapping[: out.shape[0]] < 0)


class Inkling2Sm70Attention(nn.Module):
    """Attention of one layer on one TP rank: its q heads and KV heads."""

    def __init__(
        self,
        config: Any,
        specs: Specs,
        prefix: str,
        *,
        is_local: bool,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.is_local = is_local
        self.sizes = qkvr_sizes(config, is_local, tp_size)
        self.num_heads = self.sizes[3]
        self.head_dim = self.sizes[0] // self.num_heads
        self.hidden_size = int(config.hidden_size)
        self.qk_clip = float(config.qk_l2_clip_threshold)
        self.rel_clip = float(config.rel_l2_clip_threshold)
        self.eps = float(config.rms_norm_eps)
        self.log_scaling: tuple[float, float] | None = None
        if not is_local and config.log_scaling_n_floor is not None:
            self.log_scaling = (
                float(config.log_scaling_alpha),
                float(config.log_scaling_n_floor),
            )
        self.qkvr = Weights(specs, f"{prefix}.qkvr", ("weight",))
        self.wo_ud = Weights(specs, f"{prefix}.wo_ud", ("weight",))
        for leaf in ("q_scale", "r_scale", "rel_logits_proj"):
            setattr(self, leaf, _parameter(specs, f"{prefix}.{leaf}"))
        conv = ("weight",) if is_local else ("static_w", "weight_proj")
        self.k_sconv = Weights(specs, f"{prefix}.k_sconv", conv)
        self.v_sconv = Weights(specs, f"{prefix}.v_sconv", conv)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None,
        x_v: torch.Tensor | None,
        attn_owner: Any,
        conv_owner: Any,
    ) -> torch.Tensor:
        """Normed inputs [T, H] -> this rank's partial output [T, H]. `x_k`
        and `x_v` None: the keys and values read `x_q`. Stores the batch's
        K/V and conv inputs in the visit's caches."""
        q, r, gate, sink, k, v = project_qkvr(
            x_q, x_k, x_v, self.qkvr.weight, self.sizes
        )
        if self.is_local:
            k_conv, v_conv = self.k_sconv.weight, self.v_sconv.weight
            k_taps = v_taps = None
        else:
            k_conv, v_conv = self.k_sconv.static_w, self.v_sconv.static_w
            k_taps = F.linear(x_q if x_k is None else x_k, self.k_sconv.weight_proj)
            v_taps = F.linear(x_q if x_v is None else x_v, self.v_sconv.weight_proj)
        num_tokens = x_q.shape[0]
        q_out = x_q.new_empty((num_tokens, self.num_heads, self.head_dim))
        rel_out = x_q.new_empty(
            (num_tokens, self.num_heads, self.rel_logits_proj.shape[-1])
        )
        attention_prologue(
            q,
            r,
            k,
            v,
            self.q_scale,
            self.r_scale,
            self.rel_logits_proj,
            k_conv,
            v_conv,
            k_taps,
            v_taps,
            attn_owner,
            conv_owner,
            q_out,
            rel_out,
            qk_clip=self.qk_clip,
            rel_clip=self.rel_clip,
            eps=self.eps,
            log_scaling=self.log_scaling,
        )
        output = x_q.new_empty((num_tokens, self.hidden_size))
        paged_attention(
            q_out, rel_out, sink, gate, attn_owner, output, self.wo_ud.weight
        )
        return output


class Inkling2Sm70MessageGate(nn.Module):
    """(1 - tanh(forget)) * state + tanh(gain) * message of one stream."""

    def __init__(self, config: Any, specs: Specs, prefix: str) -> None:
        super().__init__()
        self.eps = float(config.rms_norm_eps)
        for leaf in ("down_weight", "gain_weight", "forget_weight"):
            setattr(self, leaf, _parameter(specs, f"{prefix}.{leaf}"))

    def forward(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """state, message [T, H] -> [T, H] fp32."""
        return message_gate(
            state,
            message,
            self.down_weight,
            self.forget_weight,
            self.gain_weight,
            self.eps,
        )


class Inkling2Sm70DecoderLayer(nn.Module):
    """One decoder layer. A loop core layer has `controllers`, the message
    gates of its passes after the first."""

    def __init__(
        self,
        config: Any,
        specs: Specs,
        prefix: str,
        layer_idx: int,
        *,
        tp_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.is_local = layer_idx in set(config.local_layer_ids)
        self.starts_block = layer_idx % int(config.n_layers_per_attn_res_block) == 0
        self.eps = float(config.rms_norm_eps)
        self.attn_norm = Weights(specs, f"{prefix}.attn_norm", ("weight",))
        self.attn = Inkling2Sm70Attention(
            config, specs, f"{prefix}.attn", is_local=self.is_local, tp_size=tp_size
        )
        self.attn_sconv = Weights(specs, f"{prefix}.attn_sconv", ("weight",))
        self.mlp_norm = Weights(specs, f"{prefix}.mlp_norm", ("weight",))
        self.mlp = Inkling2Sm70Experts(config, tp_size=tp_size, dtype=dtype)
        self.mlp_sconv = Weights(specs, f"{prefix}.mlp_sconv", ("weight",))
        if layer_idx in config.loop_core_layers:
            self.controllers = nn.ModuleDict(
                {
                    stream: Inkling2Sm70MessageGate(
                        config, specs, f"{prefix}.controllers.{stream}"
                    )
                    for stream, _ in MESSAGE_STREAMS
                }
            )

    def _attention_inputs(
        self,
        hidden: torch.Tensor,
        bank: torch.Tensor,
        attn_query: torch.Tensor | None,
        messages: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Normed (x_q, x_k, x_v) in the model dtype; x_k and x_v are None
        when the three inputs are one."""
        if attn_query is None:
            streams = [hidden]
        else:
            streams = list(mix(bank, hidden, attn_query).unbind(-2))
        norms = self.attn_norm.weight
        if messages is not None or len(norms) > 1:
            streams = streams * (3 // len(streams))
        if messages is not None:
            streams = [
                self.controllers[stream](x, messages[:, i])
                for i, ((stream, _), x) in enumerate(zip(MESSAGE_STREAMS, streams))
            ]
        normed = [
            rms_norm(x, norms[i % len(norms)], self.eps).to(hidden.dtype)
            for i, x in enumerate(streams)
        ]
        if len(normed) == 1:
            return normed[0], None, None
        return normed[0], normed[1], normed[2]

    def forward(
        self,
        hidden: torch.Tensor,
        bank: torch.Tensor,
        attn_query: torch.Tensor | None,
        mlp_query: torch.Tensor,
        messages: torch.Tensor | None,
        attn_owner: Any,
        conv_owner: Any,
        is_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One visit of the layer over the caches of the visit's owners.

        hidden [T, H]: the residual stream, model dtype. bank [T, N, H] fp32:
        the normed block states. attn_query [1 or 3, H], None in the first
        layer; mlp_query [1, H]. messages [T, 4, H] fp32 in the order of
        ``MESSAGE_STREAMS``, None in a first pass. is_padding [T] bool.
        Returns the new residual stream and the bank.
        """
        dtype = hidden.dtype
        x_q, x_k, x_v = self._attention_inputs(hidden, bank, attn_query, messages)
        attn = all_reduce(
            self.attn(x_q, x_k, x_v, attn_owner, conv_owner), self.tp_size
        )
        attn_out = attn.new_empty(attn.shape)
        short_conv(attn, self.attn_sconv.weight, conv_owner, STREAM_ATTN, attn_out)
        # A block boundary restarts the residual from the attention output.
        if self.starts_block:
            bank = torch.cat([bank, normed_state(hidden)], dim=-2)
            residual = attn_out.float()
        else:
            residual = hidden.float() + attn_out.float()

        x = mix(bank, residual, mlp_query)[:, 0]
        if messages is not None:
            x = self.controllers[MESSAGE_STREAMS[3][0]](x, messages[:, 3])
        x = rms_norm(x, self.mlp_norm.weight[0], self.eps).to(dtype)
        partial = x.new_empty(x.shape)
        self.mlp(x, partial, is_padding)
        expert_sum = all_reduce(partial, self.tp_size)
        mlp_out = expert_sum.new_empty(expert_sum.shape)
        short_conv(expert_sum, self.mlp_sconv.weight, conv_owner, STREAM_MLP, mlp_out)
        return (residual + mlp_out).to(dtype), bank


class Inkling2Sm70AttnRes(nn.Module):
    """Queries of the AttnRes mixes: `attn_queries[i]` is the attention input
    of layer i + 1 ([1, H], [3, H] for q, k, v in a global layer),
    `mlp_queries[i]` the MLP input of layer i, `output_query` the final one."""

    def __init__(self, config: Any, specs: Specs, prefix: str) -> None:
        super().__init__()
        num_layers = int(config.num_hidden_layers)
        self.attn_queries = nn.ParameterList(
            _parameter(specs, f"{prefix}.attn_queries.{i}")
            for i in range(num_layers - 1)
        )
        self.mlp_queries = nn.ParameterList(
            _parameter(specs, f"{prefix}.mlp_queries.{i}") for i in range(num_layers)
        )
        self.output_query = _parameter(specs, f"{prefix}.output_query")
