# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the looped decoder stack.

`Inkling2Sm70LoopModel` runs the layer visits of one forward in the order of
the reference (transformers modeling_inkling2_loop.py
``Inkling2LoopTextModel.forward``): the prelude, ``num_loop_passes`` passes
of the loop core, then the coda.

    visit        (layer, pass); a core layer has one visit per pass, each
                 with K/V and conv-state caches of its own (cache.py)
    hidden       the residual stream [T, H], model dtype
    bank         the normed block states [T, N, H] fp32: one per block
                 boundary passed, the keys and values of the AttnRes mixes
    pass restart every pass after the first starts again from the hidden
                 and the bank at the loop entry
    messages     AttnRes mix of the previous pass's own block states and its
                 output, one query row per input stream of each core layer

T is the padded batch; the caches hold everything before it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .backend import cache_metadata, is_decode_only
from .cache import build_cache_owners
from .layers import (
    PARTIAL_SITES,
    Bank,
    Inkling2Sm70AttnRes,
    Inkling2Sm70DecoderLayer,
    Weights,
    padding_mask,
)
from .norms import message_mix, message_sources, rms_norm, sublayer_inputs
from .weights import build_weight_map, merged_enabled

__all__ = ["Inkling2Sm70LoopModel", "PartialAudit", "VisitDump", "layer_visits"]

# (visit index, layer, pass, residual stream after the visit) -> None
VisitHook = Callable[[int, int, int, torch.Tensor], None]


def layer_visits(config: Any) -> list[tuple[int, int]]:
    """(layer, pass) of every visit in execution order."""
    core = config.loop_core_layers
    visits = [(layer, 0) for layer in range(core.start)]
    visits += [(layer, p) for p in range(config.num_loop_passes) for layer in core]
    visits += [(layer, 0) for layer in range(core.stop, config.num_hidden_layers)]
    return visits


class VisitDump:
    """A visit hook that saves every visit's residual stream under
    `directory`: ``forward<n>/visit<v>_layer<l>_pass<p>.pt`` holds the
    [T, H] tensor of forward `n` of this rank's process."""

    def __init__(self, directory: str, rank: int = 0) -> None:
        self.directory = Path(directory) / f"rank{rank}"
        self.forward = -1

    def __call__(self, visit: int, layer: int, pass_idx: int, state: torch.Tensor):
        _save_visit(self, visit, layer, pass_idx, state)


@eager_break_during_capture
def _save_visit(
    dump: VisitDump, visit: int, layer: int, pass_idx: int, state: torch.Tensor
) -> None:
    if visit == 0:
        dump.forward += 1
    folder = dump.directory / f"forward{dump.forward:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"visit{visit:02d}_layer{layer:02d}_pass{pass_idx}.pt"
    torch.save(state.detach().cpu(), folder / name)


class PartialAudit(nn.Module):
    """Running maxima of this rank's partials before the all-reduces.

    ``maxima`` [visits, 2] fp32: max |x| over every forward so far, per visit
    and site (``layers.PARTIAL_SITES``: the o_proj partial, the expert
    partial). A non-finite partial makes its entry inf or NaN. With a
    `directory`, ``save`` writes it to ``rank<r>/partial_maxima.pt``.
    """

    def __init__(
        self, num_visits: int, directory: str | None = None, rank: int = 0
    ) -> None:
        super().__init__()
        self.path = Path(directory) / f"rank{rank}" if directory else None
        self.register_buffer(
            "maxima",
            torch.zeros(num_visits, len(PARTIAL_SITES), dtype=torch.float32),
            persistent=False,
        )

    def record(self, visit: int, site: str, partial: torch.Tensor) -> None:
        entry = self.maxima[visit, PARTIAL_SITES.index(site)]
        entry.copy_(torch.maximum(entry, partial.detach().abs().amax().float()))

    def report(self) -> dict[str, float]:
        """Site -> the largest entry over the visits."""
        return dict(zip(PARTIAL_SITES, self.maxima.amax(0).tolist()))

    def save(self) -> None:
        if self.path is not None:
            _save_partials(self)


@eager_break_during_capture
def _save_partials(audit: PartialAudit) -> None:
    audit.path.mkdir(parents=True, exist_ok=True)
    torch.save(audit.maxima.detach().cpu(), audit.path / "partial_maxima.pt")


class Inkling2Sm70LoopModel(nn.Module):
    """Embedding, decoder layers, AttnRes queries, message queries and the
    final norm of one TP rank, with the cache owners of every visit. The
    ranks' residual streams are the same; a layer sums the partials.

    `aux_hidden_state_layers`: visit indices whose input is returned beside
    the output. `visit_hook`: called after every visit. `partial_audit`: a
    ``PartialAudit`` that records the partials of every visit, or None.
    """

    def __init__(
        self,
        config: Any,
        *,
        vllm_config: Any,
        prefix: str = "model",
        tp_size: int = 1,
        tp_rank: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if config.loop_config is None:
            raise ValueError(
                "vllm_sm70: Inkling 2 without a loop_config is not ported."
            )
        self.config = config
        self.tp_size = tp_size
        self.dtype = dtype
        self.eps = float(config.rms_norm_eps)
        self.weight_map = build_weight_map(
            config, tp_size=tp_size, tp_rank=tp_rank, dtype=dtype
        )
        self.merged = merged_enabled()
        specs = self.weight_map.params
        self.embed_tokens = Weights(specs, "model.embed_tokens", ("weight",))
        self.embed_norm = Weights(specs, "model.embed_norm", ("weight",))
        self.layers = nn.ModuleList(
            Inkling2Sm70DecoderLayer(
                config,
                specs,
                f"model.layers.{layer}",
                layer,
                tp_size=tp_size,
                dtype=dtype,
                merged=self.merged,
            )
            for layer in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2Sm70AttnRes(config, specs, "model.attn_res")
        self.message_queries = nn.Parameter(
            torch.empty(
                specs["model.message_queries"].shape,
                dtype=specs["model.message_queries"].dtype,
            ),
            requires_grad=False,
        )
        self.norm = Weights(specs, "model.norm", ("weight",))
        # Not a submodule: the owners are found through the forward context.
        self.caches = build_cache_owners(
            config, prefix=f"{prefix}.layers", vllm_config=vllm_config, tp_size=tp_size
        )
        self.visits = layer_visits(config)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.visit_hook: VisitHook | None = None
        self.partial_audit: PartialAudit | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normed token embeddings [T, H]. Image features replace rows of it."""
        embedded = F.embedding(input_ids, self.embed_tokens.weight)
        return rms_norm(embedded, self.embed_norm.weight[0], self.eps)

    def _capture(self, state: torch.Tensor) -> torch.Tensor:
        return state.clone()

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """inputs_embeds [T, H] (``embed_input_ids``, image rows merged) ->
        the final normed state [T, H], the input of the LM head. With
        `aux_hidden_state_layers` set: (that, [input of each listed visit...,
        that]). Token positions come from the caches' metadata."""
        core = self.config.loop_core_layers
        hidden = inputs_embeds
        num_tokens = hidden.shape[0]
        bank = Bank(
            hidden,
            sum(layer.starts_block for layer in self.layers),
            decode_only=is_decode_only(cache_metadata(self.caches.conv[0, 0])),
        )
        is_padding = torch.empty(num_tokens, dtype=torch.bool, device=hidden.device)
        padding_mask(self.caches.conv[0, 0], is_padding)
        aux: list[torch.Tensor] = []
        audit = self.partial_audit
        entry_hidden = message_bank = None
        entry_count = 0
        for visit, (layer_idx, pass_idx) in enumerate(self.visits):
            if layer_idx == core.start:
                if pass_idx == 0:
                    entry_hidden, entry_count = hidden, bank.count
                else:
                    # The previous pass's own block states after the entry
                    # one, and its output.
                    message_bank = message_sources(
                        bank.states(entry_count + 1), hidden, bank.triton
                    )
                    hidden, bank.count = entry_hidden, entry_count
            messages = None
            if pass_idx:
                messages = message_mix(
                    message_bank, self.message_queries[layer_idx - core.start]
                )
            if visit in self.aux_hidden_state_layers:
                aux.append(self._capture(hidden))
            hidden = self.layers[layer_idx](
                hidden,
                bank,
                self.attn_res.attn_queries[layer_idx - 1] if layer_idx else None,
                self.attn_res.mlp_queries[layer_idx],
                messages,
                self.caches.attn[layer_idx, pass_idx],
                self.caches.conv[layer_idx, pass_idx],
                is_padding,
                None if audit is None else functools.partial(audit.record, visit),
            )
            if self.visit_hook is not None:
                self.visit_hook(visit, layer_idx, pass_idx, hidden)
        if audit is not None:
            audit.save()
        (final,) = sublayer_inputs(
            bank,
            hidden,
            self.attn_res.output_query,
            self.norm.weight,
            self.eps,
            hidden.dtype,
        )
        if self.aux_hidden_state_layers:
            return final, [*aux, final]
        return final
