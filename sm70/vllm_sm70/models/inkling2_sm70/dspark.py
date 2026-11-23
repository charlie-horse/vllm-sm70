# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 DSpark draft (``mtp.safetensors``) on SM70, in fp16.

    Inkling2DSparkSm70Model   registered as ``Inkling2DSparkSm70Model``
      model.layers[i]         3 dense layers: block attention over context
                              K/V built from target features, SwiGLU MLP
      model.markov_head       recurrent (rnn_hidden) correction of the logits
      lm_head                 the target's, shared by load_dspark_model

The port of upstream's ``Inkling2DSparkModel`` (vLLM c32251b
models/inkling2/nvidia/dspark.py) without flashinfer, FlashAttention or
the MNNVL all-reduce. Differences:

* fp16 weights and GEMMs; the residual stream is fp32 (upstream: bf16).
  Weights outside the fp16 range refuse the load.
* TP splits attention heads and the MLP like upstream (6 all-reduces per
  draft step), so a rank's draft K/V page equals the target's.
* Each layer owns an ``Inkling2AttentionCache``; a local layer's window is
  ``sliding_window_size + 1`` so the KV manager keeps the
  ``sliding_window_size`` positions before the block that it reads.
* The Markov head's rnn_proj runs in fp32; the fold of hidden_out into the
  LM head is fp64, stored in the model dtype. Logits are fp32.
* A TP-sharded target LM head (weights.py ``head_sharded``): each rank folds
  its vocabulary rows and the ranks all-gather the fold once at load; the
  base logits are all-gathered in the model dtype each draft step. Every
  rank then holds the logits of the whole vocabulary.

Kernels (ops/dspark.py) run on CUDA tensors and in Triton's interpreter;
other tensors, or ``VLLM_SM70_INKLING_DSPARK_KERNELS=0``, take their torch
forms.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from vllm_sm70 import envs

from .attention_torch import kv_cache_views
from .backend import cache_metadata
from .cache import Inkling2AttentionCache
from .layers import all_reduce
from .ops import _compat
from .ops import dspark as ops
from .sconv import _store_rows

__all__ = ["SWITCH", "Inkling2DSparkSm70Model", "kernels_on"]

SWITCH = "VLLM_SM70_INKLING_DSPARK_KERNELS"
_enabled = envs.environment_variables()[SWITCH]
FP16_MAX = 65504.0
FOLD_ROWS = 4096


def kernels_on(x: torch.Tensor) -> bool:
    """Whether the ops on `x` run the Triton kernels."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()


def _param(*shape: int, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, dtype) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(dtype)


class Inkling2DraftAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        layer: int,
        *,
        prefix: str,
        tp_size: int,
        dtype: torch.dtype,
        vllm_config: Any,
    ) -> None:
        super().__init__()
        if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
            raise ValueError(
                f"vllm_sm70: the Inkling 2 draft's heads do not split over TP {tp_size}."
            )
        d, hidden = config.head_dim, config.hidden_size
        self.head_dim = d
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size, self.kv_size = self.num_heads * d, self.num_kv_heads * d
        self.eps = float(config.rms_norm_eps)
        self.block = int(config.draft_block_size)
        self.is_local = layer in config.local_layer_ids
        window = int(config.sliding_window_size)
        # Keys a block reads: itself and, if local, the window before it.
        self.window = window + self.block if self.is_local else 0
        self.qkv_proj = _param(self.q_size + 2 * self.kv_size, hidden, dtype=dtype)
        self.kv_proj_ctx = _param(2 * self.kv_size, hidden, dtype=dtype)
        self.o_proj = _param(hidden, self.q_size, dtype=dtype)
        self.q_norm = _param(d, dtype=dtype)
        self.k_norm = _param(d, dtype=dtype)
        self.k_norm_ctx = _param(d, dtype=dtype)
        self.cache = Inkling2AttentionCache(
            prefix=f"{prefix}.attn",
            num_kv_heads=config.num_key_value_heads,
            head_dim=d,
            sliding_window=window + 1 if self.is_local else None,
            layer_idx=layer,
            vllm_config=vllm_config,
            tp_size=tp_size,
        )

    def _store(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
        cache = self.cache.kv_cache
        if cache.numel() == 0:
            return
        rows = torch.cat([k, v], dim=-1).to(cache.dtype)
        _store_rows(cache, slots[: rows.shape[0]], rows)

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, cos_sin: torch.Tensor
    ) -> torch.Tensor:
        """This rank's partial attention output [T, H] (before all-reduce)."""
        kernel = kernels_on(x)
        tokens = x.shape[0]
        qkv = F.linear(x, self.qkv_proj)
        ops.qk_norm_rope_(
            qkv,
            positions[:tokens],
            self.q_norm,
            self.k_norm,
            cos_sin,
            self.num_heads,
            self.num_kv_heads,
            self.eps,
            kernel=kernel,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        out = q.new_zeros((tokens, self.q_size))
        md = cache_metadata(self.cache)
        if md is not None:
            shape = (-1, self.num_kv_heads, self.head_dim)
            self._store(k.reshape(shape), v.reshape(shape), md.slot_mapping)
            key_cache, value_cache = kv_cache_views(self.cache.kv_cache, self.head_dim)
            num_reqs = min(tokens // self.block, md.seq_lens.shape[0])
            n = num_reqs * self.block
            ops.block_attention(
                q[:n].reshape(n, self.num_heads, self.head_dim),
                key_cache,
                value_cache,
                out[:n],
                block_table=md.block_table[:num_reqs],
                seq_lens=md.seq_lens[:num_reqs],
                block=self.block,
                window=self.window,
                kernel=kernel,
            )
        return F.linear(out, self.o_proj)

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        kernel = kernels_on(context)
        kv = F.linear(context, self.kv_proj_ctx)
        k, v = kv.split([self.kv_size, self.kv_size], -1)
        ops.qk_norm_rope_(
            k,
            positions,
            self.k_norm_ctx,
            self.k_norm_ctx,
            cos_sin,
            0,
            self.num_kv_heads,
            self.eps,
            kernel=kernel,
        )
        if slot_mapping is not None:
            shape = (-1, self.num_kv_heads, self.head_dim)
            self._store(k.reshape(shape), v.reshape(shape), slot_mapping)


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer: int,
        *,
        prefix: str,
        tp_size: int,
        dtype: torch.dtype,
        vllm_config: Any,
    ) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        if inter % tp_size:
            raise ValueError(
                f"vllm_sm70: the Inkling 2 draft MLP does not split over TP {tp_size}."
            )
        self.eps = float(config.rms_norm_eps)
        self.tp_size = tp_size
        self.dtype = dtype
        self.input_layernorm = _param(hidden, dtype=dtype)
        self.post_attention_layernorm = _param(hidden, dtype=dtype)
        self.dflare_norm = _param(hidden, dtype=dtype)
        self.self_attn = Inkling2DraftAttention(
            config,
            layer,
            prefix=f"{prefix}.self_attn",
            tp_size=tp_size,
            dtype=dtype,
            vllm_config=vllm_config,
        )
        local = inter // tp_size
        self.gate_up_proj = _param(2 * local, hidden, dtype=dtype)
        self.down_proj = _param(hidden, local, dtype=dtype)

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's partial MLP output and the fp32 residual."""
        if residual is None:
            residual = x.float()
        else:
            residual = residual + all_reduce(x, self.tp_size).float()
        h = _rms_norm(residual, self.input_layernorm, self.eps, self.dtype)
        attn = self.self_attn(positions, h, cos_sin)
        residual = residual + all_reduce(attn, self.tp_size).float()
        h = _rms_norm(residual, self.post_attention_layernorm, self.eps, self.dtype)
        act = ops.silu_and_mul(F.linear(h, self.gate_up_proj), kernel=kernels_on(h))
        return F.linear(act, self.down_proj), residual


class Inkling2DraftMarkovHead(nn.Module):
    """Recurrent (rnn_hidden) correction of the base logits. The rank-r
    latent goes through hidden_out and the shared LM head, folded once into
    a (vocab, r) projection."""

    def __init__(self, config: Any, dtype: torch.dtype) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.rank = rank
        self.dtype = dtype
        self.markov_w1 = _param(config.padded_vocab_size, rank, dtype=dtype)
        self.rnn_proj_weight = _param(
            3 * rank, 2 * rank + 2 * hidden, dtype=torch.float32
        )
        self.rnn_proj_bias = _param(3 * rank, dtype=torch.float32)
        self.hidden_out_weight = _param(hidden, rank, dtype=torch.float32)
        self.hidden_out_bias = _param(hidden, dtype=torch.float32)
        self.dflare_norm = _param(hidden, dtype=dtype)
        self.register_buffer(
            "fold_weight", torch.empty(0, rank, dtype=dtype), persistent=False
        )
        self.register_buffer("fold_bias", torch.empty(0, dtype=dtype), persistent=False)

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor, head_tp: int = 1) -> None:
        """fold_weight = head @ hidden_out, fold_bias = head @ bias, in fp64.
        With ``head_tp`` > 1, `lm_head_weight` is this rank's share of the
        vocabulary rows and the ranks all-gather the folded rows (a
        collective). A refold of the same shape refills the same tensors
        (CUDA graphs)."""
        rows = lm_head_weight.shape[0]
        total = rows * head_tp
        if self.fold_weight.shape[0] != total:
            self.fold_weight = lm_head_weight.new_empty(
                (total, self.rank), dtype=self.dtype
            )
            self.fold_bias = lm_head_weight.new_empty((total,), dtype=self.dtype)
        weight, bias_out = self.fold_weight, self.fold_bias
        if head_tp > 1:
            weight = lm_head_weight.new_empty((rows, self.rank), dtype=self.dtype)
            bias_out = lm_head_weight.new_empty((rows,), dtype=self.dtype)
        projection = self.hidden_out_weight.double()
        bias = self.hidden_out_bias.double()
        for start in range(0, rows, FOLD_ROWS):
            part = slice(start, start + FOLD_ROWS)
            head = lm_head_weight[part].double()
            weight[part].copy_(head @ projection)
            bias_out[part].copy_(head @ bias)
        if head_tp > 1:
            from vllm.distributed import tensor_model_parallel_all_gather

            self.fold_weight.copy_(tensor_model_parallel_all_gather(weight, dim=0))
            self.fold_bias.copy_(tensor_model_parallel_all_gather(bias_out, dim=0))

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 rnn_proj bias plus its [hidden, f_rnn] term for hidden
        [R * steps, H] ordered (req, step); returns (step, req, 3 r)."""
        num_reqs, width = f_rnn.shape
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1).float()
        weight = self.rnn_proj_weight[:, 2 * self.rank :]
        return F.linear(inputs, weight, self.rnn_proj_bias)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step: the new state [R, r] and the bias [R, V]."""
        inputs = torch.cat([state, self.markov_w1[prev_tokens]], dim=-1).float()
        raw = context + F.linear(inputs, self.rnn_proj_weight[:, : 2 * self.rank])
        state, latent = ops.markov_gate(
            raw.contiguous(), state, kernel=kernels_on(state)
        )
        return state, F.linear(latent, self.fold_weight, self.fold_bias)


class Inkling2DSparkSm70Decoder(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        tp_size: int,
        dtype: torch.dtype,
        max_model_len: int,
        vllm_config: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.dtype = dtype
        self.eps = float(config.rms_norm_eps)
        self.hidden = config.hidden_size
        self.block = int(config.draft_block_size)
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(
                config,
                i,
                prefix=f"{prefix}.layers.{i}",
                tp_size=tp_size,
                dtype=dtype,
                vllm_config=vllm_config,
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = _param(self.hidden, dtype=dtype)
        self.markov_head = Inkling2DraftMarkovHead(config, dtype)
        self.fuse_weight = _param(
            config.num_hidden_layers + 1, len(config.target_layer_ids), dtype=dtype
        )
        self.mask_embed = _param(self.hidden, dtype=dtype)
        self.register_buffer(
            "fuse_alpha", torch.empty(self.fuse_weight.shape), persistent=False
        )
        self.register_buffer(
            "cos_sin",
            ops.rope_cache(config.head_dim, max_model_len, float(config.rope_theta)),
            persistent=False,
        )
        # The target's normed token embedding (attach_target).
        self.embed: Callable[[torch.Tensor], torch.Tensor] | None = None

    def mix(self, features: torch.Tensor, row: int, norm: torch.Tensor) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.hidden)
        return ops.mix_rmsnorm(
            targets, self.fuse_alpha[row], norm, self.eps, kernel=kernels_on(features)
        )

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The target's normed embedding of each block's anchor, then masks."""
        assert self.embed is not None, "attach_target() was not called"
        ids = input_ids.view(-1, self.block)
        x = self.mask_embed.expand(ids.shape[0], self.block, -1).clone()
        x[:, 0] = self.embed(ids[:, 0].contiguous()).to(x.dtype)
        return x.view(-1, self.hidden)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin)
        assert residual is not None
        residual = residual + all_reduce(x, self.tp_size).float()
        return _rms_norm(residual, self.norm, self.eps, self.dtype)


class Inkling2DSparkSm70Model(nn.Module):
    """The Inkling 2 DSpark draft; plugged in by
    ``Inkling2DSparkSm70Speculator`` (speculator.py)."""

    allow_patterns_overrides: ClassVar[list[str]] = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj", "q"),
        (".k_proj.", ".qkv_proj", "k"),
        (".v_proj.", ".qkv_proj", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx", "k_ctx"),
        (".v_proj_ctx.", ".kv_proj_ctx", "v_ctx"),
        (".gate_proj.", ".gate_up_proj", "gate"),
        (".up_proj.", ".gate_up_proj", "up"),
        (".o_proj.", ".o_proj", "o"),
        (".down_proj.", ".down_proj", "down"),
    )

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        k = spec.num_speculative_tokens
        if k != config.draft_block_size - 1:
            raise ValueError(
                f"vllm_sm70: the Inkling 2 draft needs num_speculative_tokens "
                f"{config.draft_block_size - 1}, got {k}."
            )
        max_model_len = vllm_config.model_config.max_model_len
        if max_model_len > config.max_position_embeddings:
            raise ValueError(
                f"vllm_sm70: the Inkling 2 draft reaches position "
                f"{config.max_position_embeddings}; lower --max-model-len."
            )
        self.config = config
        self.dtype = vllm_config.model_config.dtype
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.model = Inkling2DSparkSm70Decoder(
            config,
            prefix=f"{prefix}model.layers.dspark",
            tp_size=self.tp_size,
            dtype=self.dtype,
            max_model_len=max_model_len,
            vllm_config=vllm_config,
        )
        self.lm_head: nn.Module | None = None

    # --- target sharing ---------------------------------------------------------

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Use the target's normed embedding; fold the shared LM head."""
        assert self.lm_head is not None, "load_dspark_model did not share lm_head"
        body = target_model.model
        self.model.embed = body.embed_input_ids
        self.refresh_derived_weights()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        assert self.lm_head is not None
        self.model.markov_head.fold(self.lm_head.weight, self._head_tp())
        self.model.fuse_alpha.copy_(torch.softmax(self.model.fuse_weight.float(), -1))

    def _head_tp(self) -> int:
        """The number of ranks the shared LM head's vocabulary is split over."""
        assert self.lm_head is not None
        head_tp = int(getattr(self.lm_head, "tp_size", 1))
        if head_tp not in (1, self.tp_size):
            raise ValueError(
                f"vllm_sm70: the Inkling 2 LM head is split over {head_tp} "
                f"ranks, the draft runs on {self.tp_size}."
            )
        return head_tp

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.cache.prefix for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.model.layers)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    # --- the draft step ----------------------------------------------------------

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each draft layer mixes the raw target features itself.
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | torch.Tensor | None = None,
    ) -> None:
        for i, layer in enumerate(self.model.layers):
            slots = context_slot_mappings
            if isinstance(slots, list):
                slots = slots[i]
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context, context_positions, self.model.cos_sin, slots
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """Markov-head context from the last committed token's features."""
        head = self.model.markov_head
        return self.model.mix(features, -1, head.dflare_norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits [N, V] of the whole vocabulary in the model dtype; a
        sharded LM head's are all-gathered."""
        assert self.lm_head is not None
        logits = F.linear(hidden_states, self.lm_head.weight)
        if self._head_tp() > 1:
            from vllm.distributed import tensor_model_parallel_all_gather

            logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits

    def markov_step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.markov_head.step(state, prev_tokens, context)

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """fp32 soft-capped logits [R, vocab]."""
        logits = ops.add_softcap(
            base_logits,
            bias,
            self.config.final_logit_softcapping,
            kernel=kernels_on(base_logits),
        )
        return logits[:, : self.config.vocab_size]

    # --- weights -------------------------------------------------------------------

    def _shard(self, kind: str, weight: torch.Tensor) -> torch.Tensor:
        rank, size = self.tp_rank, self.tp_size
        if kind in ("o", "down"):
            return weight.chunk(size, dim=1)[rank]
        return weight.chunk(size, dim=0)[rank]

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        pending: dict[str, dict[str, torch.Tensor]] = {}
        loaded: set[str] = set()
        for name, weight in weights:
            name = name if name.startswith("model.") else f"model.{name}"
            name = name.replace(".mlp.", ".")
            name = name.replace(".rnn_proj.weight", ".rnn_proj_weight")
            name = name.replace(".rnn_proj.bias", ".rnn_proj_bias")
            name = name.replace(".hidden_out.weight", ".hidden_out_weight")
            name = name.replace(".hidden_out.bias", ".hidden_out_bias")
            name = name.replace(".markov_w1.weight", ".markov_w1")
            name = name.replace("_layernorm.weight", "_layernorm")
            name = name.replace(".dflare_norm.weight", ".dflare_norm")
            for norm in ("q_norm", "k_norm", "k_norm_ctx"):
                name = name.replace(f".{norm}.weight", f".{norm}")
            if name == "model.norm.weight":
                name = "model.norm"
            for old, new, kind in self._stacked:
                if old in name:
                    target = name.replace(old, new + ".").removesuffix(".weight")
                    target = target.removesuffix(".")
                    pending.setdefault(target, {})[kind] = self._shard(kind, weight)
                    break
            else:
                if name not in params:
                    raise ValueError(f"vllm_sm70: unexpected draft tensor {name}")
                self._copy(params[name], weight, name)
                loaded.add(name)
        for target, parts in pending.items():
            if "q" in parts:
                order = ("q", "k", "v")
            elif "k_ctx" in parts:
                order = ("k_ctx", "v_ctx")
            elif "gate" in parts:
                order = ("gate", "up")
            else:
                order = tuple(parts)
            if set(order) != set(parts):
                raise ValueError(f"vllm_sm70: draft tensor {target} is incomplete")
            self._copy(params[target], torch.cat([parts[k] for k in order]), target)
            loaded.add(target)
        return loaded

    @staticmethod
    def _copy(param: torch.Tensor, weight: torch.Tensor, name: str) -> None:
        if tuple(param.shape) != tuple(weight.shape):
            raise ValueError(
                f"vllm_sm70: draft tensor {name} has shape {tuple(weight.shape)}, "
                f"expected {tuple(param.shape)}"
            )
        if param.dtype == torch.float16 and weight.is_floating_point():
            peak = weight.detach().float().abs().max().item() if weight.numel() else 0.0
            if not peak <= FP16_MAX:
                raise ValueError(
                    f"vllm_sm70: draft tensor {name} exceeds the fp16 range ({peak})."
                )
        param.copy_(weight)
