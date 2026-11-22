# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: checkpoint tensors -> model parameters (non-expert).

One table, built from the text config, maps every checkpoint tensor of the
language model to one parameter (``Route``) or to an owner outside this module
(``Skip``: expert weights, vision tower). Parameter names are the checkpoint
names after ``RENAMES``, with the per-layer projections fused as in
``QKVR_SHARDS`` and ``NORM_SHARDS`` (upstream vLLM's layout,
vllm/models/inkling2/nvidia/model.py ``_renames`` / ``_shards``). Each rank
keeps its slice of the head-sharded tensors; everything else is replicated.

Dtypes: a parameter whose Hugging Face name contains an entry of
``HF_FP32_MODULES`` is float32; every other parameter is in the model dtype.
Checkpoint values are cast (bf16 -> fp16 for the released checkpoint) and are
otherwise unchanged: queries are not pre-scaled.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = [
    "HF_FP32_MODULES",
    "MESSAGE_STREAMS",
    "NORM_SHARDS",
    "QKVR_SHARDS",
    "RENAMES",
    "Inkling2WeightMap",
    "Inkling2Weights",
    "ParamSpec",
    "Route",
    "Skip",
    "build_weight_map",
    "load_weights",
    "qkvr_sizes",
]

# transformers modeling_inkling2_loop.py ``_keep_in_fp32_modules_strict``,
# verbatim; matched as substrings of the Hugging Face parameter name.
HF_FP32_MODULES: tuple[str, ...] = (
    "e_score_correction_bias",
    "mlp.gate.weight",
    "post_attention_convolution",
    "post_mlp_convolution",
    "sconv.conv1d",
    "attention_input",
    "mlp_input",
    "final_input",
    "message_input",
    "message_gate_proj",
    "forget_gate_proj",
)

# Checkpoint substring -> parameter substring, applied in order.
RENAMES: tuple[tuple[str, str], ...] = (
    ("model.llm.unembed.", "lm_head."),
    ("model.llm.embed.", "model.embed_tokens."),
    ("model.llm.", "model."),
    (".q_l2_clip.scale", ".q_scale"),
    (".rel_l2_clip.scale", ".r_scale"),
    (".rel_logits_proj.proj", ".rel_logits_proj"),
)

# Rows of ``attn.qkvr.weight`` in order: (checkpoint module, Hugging Face
# module under ``self_attn``). The first four project the query input.
QKVR_SHARDS: tuple[tuple[str, str], ...] = (
    ("wq_du", "q_proj.linear"),
    ("wr_du", "relative_position_bias.linear"),
    ("gate_attn", "o_proj.gate_proj"),
    ("attn_sink_proj", "sink_proj"),
    ("wk_dv", "k_proj.linear"),
    ("wv_dv", "v_proj.linear"),
)

# Rows of a full-attention layer's ``attn_norm.weight`` in order.
NORM_SHARDS: tuple[str, ...] = ("q", "k", "v")

# Checkpoint controller name -> Hugging Face gate name, in the order of the
# second axis of ``model.message_queries``.
MESSAGE_STREAMS: tuple[tuple[str, str], ...] = (
    ("attn", "q"),
    ("k", "k"),
    ("v", "v"),
    ("mlp", "mlp"),
)

_CONTROLLER_WEIGHTS = (
    ("down_weight", "down_proj"),
    ("gain_weight", "message_gate_proj"),
    ("forget_weight", "forget_gate_proj"),
)
_SOURCE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_LLM = "model.llm."
_HF = "model.language_model."
_VISUAL = "model.visual."
_EXPERT_RE = re.compile(
    r"model\.llm\.layers\.(\d+)\.mlp\.experts\."
    r"w(?:13|2)_weight(?:_scale|_amax|_outer_scale_denominator)?"
)


@dataclass(frozen=True)
class ParamSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class Route:
    """One checkpoint tensor's place in one parameter.

    The tensor is cut to ``rows`` leading rows, narrowed along ``tp_dim`` to
    this rank's part, grouped in ``pairs`` along the first axis, cast, and
    written to ``param[index]`` (the whole parameter when ``index`` is None).
    """

    source: str
    param: str
    hf_name: str
    source_shape: tuple[int, ...]
    index: tuple[Any, ...] | None = None
    tp_dim: int | None = None
    rows: int | None = None
    pairs: bool = False


@dataclass(frozen=True)
class Skip:
    """A checkpoint tensor another loader owns: ``experts`` or ``visual``."""

    source: str
    owner: str


@dataclass
class Inkling2Weights:
    params: dict[str, torch.Tensor]
    experts: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    visual: list[tuple[str, torch.Tensor]] = field(default_factory=list)


def _text_config(config: Any) -> Any:
    get = getattr(config, "get_text_config", None)
    return config if get is None else get()


def _rename(name: str) -> str:
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def qkvr_sizes(config: Any, is_local: bool, tp_size: int = 1) -> tuple[int, ...]:
    """Row counts of one rank's ``attn.qkvr.weight``, in ``QKVR_SHARDS`` order."""
    text = _text_config(config)
    heads = text.swa_num_attention_heads if is_local else text.num_attention_heads
    kv_heads = text.swa_num_key_value_heads if is_local else text.num_key_value_heads
    head_dim = text.swa_head_dim if is_local else text.head_dim
    if heads % tp_size or kv_heads % tp_size:
        raise ValueError(
            f"Inkling 2: {heads} query heads and {kv_heads} KV heads don't "
            f"split over {tp_size} ranks"
        )
    heads, kv_heads = heads // tp_size, kv_heads // tp_size
    q_size, kv_size = heads * head_dim, kv_heads * head_dim
    return (q_size, heads * text.d_rel, q_size, heads, kv_size, kv_size)


class Inkling2WeightMap:
    """The checkpoint -> parameter table of one TP rank.

    ``params``: parameter name -> ``ParamSpec`` of this rank.
    ``routes``: checkpoint name -> ``Route``; ``classify`` also answers for
    the tensors this module doesn't own.
    """

    def __init__(
        self,
        config: Any,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        dtype: torch.dtype = torch.float16,
        with_visual: bool = False,
    ) -> None:
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"Inkling 2: TP rank {tp_rank} outside 0..{tp_size - 1}")
        if not dtype.is_floating_point:
            raise ValueError(f"Inkling 2: model dtype {dtype} is not a float type")
        self.config = _text_config(config)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.with_visual = with_visual
        self.params: dict[str, ParamSpec] = {}
        self.routes: dict[str, Route] = {}
        self._build()

    # Table construction.

    def _local_shape(self, route: Route) -> tuple[int, ...]:
        shape = list(route.source_shape)
        if route.rows is not None:
            shape[0] = route.rows
        if route.tp_dim is not None:
            if shape[route.tp_dim] % self.tp_size:
                raise ValueError(
                    f"{route.source}: dim {route.tp_dim} of {route.source_shape} "
                    f"doesn't split over {self.tp_size} ranks"
                )
            shape[route.tp_dim] //= self.tp_size
        if route.pairs:
            shape[:1] = [shape[0] // 2, 2]
        return tuple(shape)

    def _add(
        self,
        source: str,
        hf_name: str,
        source_shape: tuple[int, ...],
        *,
        param: str | None = None,
        param_shape: tuple[int, ...] | None = None,
        **route_fields: Any,
    ) -> None:
        source = _LLM + source
        route = Route(
            source=source,
            param=_rename(source) if param is None else param,
            hf_name=hf_name if hf_name.startswith("lm_head.") else _HF + hf_name,
            source_shape=source_shape,
            **route_fields,
        )
        fp32 = any(module in route.hf_name for module in HF_FP32_MODULES)
        spec = ParamSpec(
            self._local_shape(route) if param_shape is None else param_shape,
            torch.float32 if fp32 else self.dtype,
        )
        if self.params.setdefault(route.param, spec) != spec:
            raise AssertionError(f"{route.param}: conflicting specs")
        assert source not in self.routes, source
        self.routes[source] = route

    def _build(self) -> None:
        text = self.config
        hidden = text.hidden_size
        num_layers = text.num_hidden_layers
        local_ids = set(text.local_layer_ids)
        row = (1, hidden)

        self._add(
            "embed.weight", "embed_tokens.weight", (text.padded_vocab_size, hidden)
        )
        self._add(
            "unembed.weight",
            "lm_head.weight",
            (text.padded_vocab_size, hidden),
            rows=text.unpadded_vocab_size,
        )
        self._add(
            "embed_norm.weight",
            "embed_tokens.embed_norm.weight",
            (hidden,),
            param_shape=row,
        )
        self._add("norm.weight", "norm.weight", (hidden,), param_shape=row)
        self._add(
            "attn_res.output_query", "final_input.query", (hidden,), param_shape=row
        )
        for layer in range(num_layers):
            self._add(
                f"attn_res.mlp_queries.{layer}",
                f"layers.{layer}.mlp_input.query",
                (hidden,),
                param_shape=row,
            )
        # attn_queries.{i} is the attention-input query of layer i + 1.
        for layer in range(1, num_layers):
            shape = (hidden,) if layer in local_ids else (len(NORM_SHARDS), hidden)
            self._add(
                f"attn_res.attn_queries.{layer - 1}",
                f"layers.{layer}.attention_input.query",
                shape,
                param_shape=(1, hidden) if layer in local_ids else shape,
            )
        core = text.loop_core_layers
        if core:
            # [query, mlp] and [key, value] per core layer -> q, k, v, mlp.
            shape = (len(core), len(MESSAGE_STREAMS), hidden)
            for name, columns in (
                ("middle_cycle_cross_pass_queries", [0, 3]),
                ("middle_cycle_cross_pass_kv_queries", [1, 2]),
            ):
                self._add(
                    name,
                    "message_input.query",
                    (2 * len(core), hidden),
                    param="model.message_queries",
                    param_shape=shape,
                    index=(slice(None), columns),
                    pairs=True,
                )
        for layer in range(num_layers):
            self._build_layer(layer, layer in local_ids, layer in core)

    def _build_layer(self, layer: int, is_local: bool, in_core: bool) -> None:
        text = self.config
        hidden, kernel = text.hidden_size, text.sconv_kernel_size
        src = hf = f"layers.{layer}."
        param = f"model.layers.{layer}."
        full = qkvr_sizes(text, is_local)
        local = qkvr_sizes(text, is_local, self.tp_size)
        q_size, _, _, heads, kv_size, _ = full
        kv_heads = (
            text.swa_num_key_value_heads if is_local else text.num_key_value_heads
        )
        head_dim = q_size // heads
        attn, hf_attn = src + "attn.", hf + "self_attn."

        offset = 0
        for (module, hf_module), rows, local_rows in zip(QKVR_SHARDS, full, local):
            self._add(
                f"{attn}{module}.weight",
                f"{hf_attn}{hf_module}.weight",
                (rows, hidden),
                param=param + "attn.qkvr.weight",
                param_shape=(sum(local), hidden),
                index=(slice(offset, offset + local_rows),),
                tp_dim=0,
            )
            offset += local_rows
        self._add(
            attn + "wo_ud.weight",
            hf_attn + "o_proj.linear.weight",
            (hidden, q_size),
            tp_dim=1,
        )
        self._add(
            attn + "q_l2_clip.scale",
            hf_attn + "q_proj.norm.scale",
            (heads, head_dim),
            tp_dim=0,
        )
        self._add(
            attn + "rel_l2_clip.scale",
            hf_attn + "relative_position_bias.norm.scale",
            (heads, text.d_rel),
            tp_dim=0,
        )
        extent = text.sliding_window_size if is_local else text.rel_extent
        self._add(
            attn + "rel_logits_proj.proj",
            hf_attn + "relative_position_bias.proj",
            (text.d_rel, extent),
        )
        for stream in ("k", "v"):
            conv, hf_conv = f"{attn}{stream}_sconv.", f"{hf_attn}{stream}_proj.sconv."
            if is_local:
                self._add(
                    conv + "weight",
                    hf_conv + "conv1d.weight",
                    (kv_size, 1, kernel),
                    tp_dim=0,
                )
            else:
                self._add(
                    conv + "static_w",
                    hf_conv + "static_weight",
                    (kernel, kv_size),
                    tp_dim=1,
                )
                self._add(
                    conv + "weight_proj",
                    hf_conv + "weight_proj.weight",
                    (kv_heads * kernel, hidden),
                    tp_dim=0,
                )

        norm = param + "attn_norm.weight"
        if is_local:
            self._add(
                src + "attn_norm.weight",
                hf + "input_layernorm.weight",
                (hidden,),
                param_shape=(1, hidden),
            )
        else:
            for i, stream in enumerate(NORM_SHARDS):
                self._add(
                    f"{src}{stream}_input_norm.weight",
                    f"{hf}{stream}_input_layernorm.weight",
                    (hidden,),
                    param=norm,
                    param_shape=(len(NORM_SHARDS), hidden),
                    index=(i,),
                )
        self._add(
            src + "mlp_norm.weight",
            hf + "post_attention_layernorm.weight",
            (hidden,),
            param_shape=(1, hidden),
        )
        for conv, hf_conv in (
            ("attn_sconv", "post_attention_convolution"),
            ("mlp_sconv", "post_mlp_convolution"),
        ):
            self._add(
                f"{src}{conv}.weight",
                f"{hf}{hf_conv}.conv1d.weight",
                (hidden, 1, kernel),
            )
        experts = text.n_routed_experts
        self._add(src + "mlp.gate.weight", hf + "mlp.gate.weight", (experts, hidden))
        self._add(
            src + "mlp.gate.bias", hf + "mlp.gate.e_score_correction_bias", (experts,)
        )
        if in_core:
            rank = text.loop_config["gate_rank"]
            shapes = ((rank, 2 * hidden), (hidden, rank), (hidden, rank))
            for stream, hf_stream in MESSAGE_STREAMS:
                for (weight, hf_weight), shape in zip(_CONTROLLER_WEIGHTS, shapes):
                    self._add(
                        f"{src}controllers.{stream}.{weight}",
                        f"{hf}{hf_stream}_message_gate.{hf_weight}.weight",
                        shape,
                    )

    # Lookup and conversion.

    def classify(self, name: str) -> Route | Skip:
        """The destination of checkpoint tensor ``name``; raises on a name the
        model doesn't have."""
        route = self.routes.get(name)
        if route is not None:
            return route
        match = _EXPERT_RE.fullmatch(name)
        if match and int(match.group(1)) < self.config.num_hidden_layers:
            return Skip(name, "experts")
        if name.startswith(_VISUAL):
            return Skip(name, "visual")
        raise ValueError(f"Inkling 2: unexpected checkpoint tensor {name!r}")

    def localize(self, route: Route, tensor: torch.Tensor) -> torch.Tensor:
        """This rank's part of checkpoint tensor ``tensor``, in the parameter
        dtype and the shape of ``param[route.index]``."""
        if tuple(tensor.shape) != route.source_shape:
            raise ValueError(
                f"{route.source}: checkpoint shape {tuple(tensor.shape)} != "
                f"{route.source_shape}"
            )
        if tensor.dtype not in _SOURCE_DTYPES:
            raise ValueError(
                f"{route.source}: checkpoint dtype {tensor.dtype} is not a "
                "float weight dtype"
            )
        if route.rows is not None:
            tensor = tensor[: route.rows]
        if route.tp_dim is not None:
            size = tensor.shape[route.tp_dim] // self.tp_size
            tensor = tensor.narrow(route.tp_dim, self.tp_rank * size, size)
        if route.pairs:
            tensor = tensor.unflatten(0, (-1, 2))
        return tensor.to(self.params[route.param].dtype)

    def _write(self, route: Route, target: torch.Tensor, tensor: torch.Tensor) -> None:
        value = self.localize(route, tensor)
        if route.index is None:
            target.copy_(value.reshape(target.shape))
        else:
            target[route.index] = value.reshape(target[route.index].shape)

    def _run(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        target: Callable[[str], torch.Tensor],
        experts: Callable[[str, torch.Tensor], None],
        visual: Callable[[str, torch.Tensor], None],
    ) -> None:
        seen: set[str] = set()
        with torch.no_grad():
            for name, tensor in weights:
                if name in seen:
                    raise ValueError(f"Inkling 2: duplicate checkpoint tensor {name!r}")
                seen.add(name)
                route = self.classify(name)
                if isinstance(route, Route):
                    self._write(route, target(route.param), tensor)
                elif route.owner == "experts":
                    experts(name, tensor)
                elif self.with_visual:
                    visual(name.removeprefix(_VISUAL), tensor)
        if missing := self.routes.keys() - seen:
            raise ValueError(
                f"Inkling 2: missing checkpoint tensors: {sorted(missing)}"
            )

    def convert(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Inkling2Weights:
        """This rank's parameters as new tensors, plus the untouched expert
        tensors and (with ``with_visual``) the vision tensors without their
        ``model.visual.`` prefix."""
        out = Inkling2Weights(
            {
                name: torch.empty(spec.shape, dtype=spec.dtype)
                for name, spec in self.params.items()
            }
        )
        self._run(
            weights,
            out.params.__getitem__,
            lambda name, tensor: out.experts.append((name, tensor)),
            lambda name, tensor: out.visual.append((name, tensor)),
        )
        return out

    def load_into(
        self,
        model: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        expert_loader: Callable[[str, torch.Tensor], None] | None = None,
        visual_loader: Callable[[str, torch.Tensor], None] | None = None,
    ) -> set[str]:
        """Copy the checkpoint into ``model``'s parameters; returns the names
        loaded. ``model`` must hold exactly ``self.params`` (name, shape and
        dtype) outside ``.mlp.experts.`` and ``visual.``, which the two
        loaders own: each receives its tensors by checkpoint name (vision
        names without ``model.visual.``) in checkpoint order, and the caller
        adds the names they loaded. Without an expert loader the expert
        tensors are dropped."""
        if (visual_loader is not None) != self.with_visual:
            raise ValueError("Inkling 2: visual_loader must match with_visual")
        params = {
            name: param
            for name, param in model.named_parameters()
            if ".mlp.experts." not in name and not name.startswith("visual.")
        }
        problems = [
            f"missing parameter {n}" for n in self.params.keys() - params.keys()
        ]
        problems += [
            f"unmapped parameter {n}" for n in params.keys() - self.params.keys()
        ]
        for name in self.params.keys() & params.keys():
            spec, param = self.params[name], params[name]
            if (tuple(param.shape), param.dtype) != (spec.shape, spec.dtype):
                problems.append(
                    f"{name}: {tuple(param.shape)} {param.dtype}, the loader "
                    f"has {spec.shape} {spec.dtype}"
                )
        if problems:
            raise ValueError("Inkling 2 weight map: " + "; ".join(sorted(problems)))
        self._run(
            weights,
            lambda name: params[name].data,
            expert_loader or (lambda name, tensor: None),
            visual_loader or (lambda name, tensor: None),
        )
        return set(self.params)


def build_weight_map(
    config: Any,
    *,
    tp_size: int | None = None,
    tp_rank: int | None = None,
    dtype: torch.dtype = torch.float16,
    with_visual: bool = False,
) -> Inkling2WeightMap:
    """The table for ``config`` (the model config or its text config). TP size
    and rank default to this process's."""
    if tp_size is None or tp_rank is None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = get_tensor_model_parallel_world_size() if tp_size is None else tp_size
        tp_rank = get_tensor_model_parallel_rank() if tp_rank is None else tp_rank
    return Inkling2WeightMap(
        config, tp_size=tp_size, tp_rank=tp_rank, dtype=dtype, with_visual=with_visual
    )


def load_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    config: Any,
    *,
    tp_size: int | None = None,
    tp_rank: int | None = None,
    dtype: torch.dtype = torch.float16,
    expert_loader: Callable[[str, torch.Tensor], None] | None = None,
    visual_loader: Callable[[str, torch.Tensor], None] | None = None,
) -> set[str]:
    """``build_weight_map(...).load_into(model, weights, ...)``; the vision
    tower is expected exactly when ``visual_loader`` is given."""
    weight_map = build_weight_map(
        config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        with_visual=visual_loader is not None,
    )
    return weight_map.load_into(
        model, weights, expert_loader=expert_loader, visual_loader=visual_loader
    )
