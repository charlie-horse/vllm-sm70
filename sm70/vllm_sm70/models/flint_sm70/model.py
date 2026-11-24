# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint on SM70 (fp16), on upstream vLLM v0.33.0 layers.

Reuses upstream's ``vllm.models.flint.configs.FlintConfig``. Routed experts
are built by ``FusedMoEFactory``, which makes its own expert mapping. The
fused q/k norm + RoPE + KV write is the op of vllm_sm70/flint/, followed by
``attention_after_fused_write``. The model opts in to the SM70 top-k and
greedy samplers with ``supports_sm70_topk_sampler``. FP16 range handling is
in fp16_rescale.py.

Quantization isn't special-cased here: dense ModelOpt W4A16 linears go to
vllm_sm70/quant/nvfp4_linear.py and routed experts to
vllm_sm70/quant/nvfp4_moe.py through the quant config. The fp16 lm_head and
router GEMMs go through the platform GEMM hook.
"""

from collections.abc import Iterable, Iterator
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.flint.configs import FlintConfig  # upstream, reused as is
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name

from vllm_sm70.config import sm70_config

# Importing qk_norm_rope_kv registers the op in torch.ops.vllm.
from vllm_sm70.flint import QK_ROPE_KV_OP_NAME
from vllm_sm70.flint.qk_norm_rope_kv import (
    attention_after_fused_write,
    fused_route_rejection,
)

from . import fp16_rescale

logger = init_logger("vllm.sm70.flint")

# The fused q/k norm + RoPE + KV-write op, torch.ops.vllm.<QK_ROPE_KV_OP_NAME>.
_QK_ROPE_KV_OP = getattr(torch.ops.vllm, QK_ROPE_KV_OP_NAME)

# Shards that one fused linear loads. A fused NVFP4 linear keeps a single
# global scale, so the shards' weight_scale_2 must agree.
_MERGED_SHARDS = {
    ".q_proj.": ".qkv_proj.",
    ".k_proj.": ".qkv_proj.",
    ".v_proj.": ".qkv_proj.",
    ".gate_proj.": ".gate_up_proj.",
    ".up_proj.": ".gate_up_proj.",
}


def _sink_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Load this rank's heads of the [total_num_heads] sink tensor in place,
    keeping the parameter's fp32 dtype."""
    heads_per_rank = param.shape[0]
    start = get_tensor_model_parallel_rank() * heads_per_rank
    param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))


def _layer_rope_parameters(config: FlintConfig, layer_idx: int) -> dict:
    """RoPE parameters of one layer, per layer type when the config has them."""
    rope_parameters = dict(config.rope_parameters or {})
    if rope_parameters and all(isinstance(v, dict) for v in rope_parameters.values()):
        layer_type = config.layer_types[layer_idx]
        rope_parameters = dict(rope_parameters.get(layer_type) or {})
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    return rope_parameters


class FlintSm70MLP(nn.Module):
    """Gated SiLU MLP. Dense linears go to the SM70 NVFP4 kernel via the
    quant config; nothing model-side."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class FlintSm70SparseMoeBlock(nn.Module):
    """Sigmoid top-k routed experts plus an ungated shared expert, combined
    and all-reduced inside the MoE runner.

    The routed-expert method (ModelOptNvFp4Sm70MoEMethod) comes from the
    quant config, not from here.
    """

    def __init__(
        self,
        config: FlintConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.router_score_function != "sigmoid":
            raise ValueError(
                "Flint routes with sigmoid scores, got "
                f"router_score_function={config.router_score_function!r}"
            )
        if get_tensor_model_parallel_world_size() > config.num_experts:
            raise ValueError("Tensor parallel size exceeds the number of experts.")
        # The fp16 GEMM backend is the platform GEMM hook.
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert = FlintSm70MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_expert",
        )
        # The expert mapping comes from the default
        # ckpt_names=("gate_proj", "down_proj", "up_proj"). The fused router
        # binds through the platform getter, not here.
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            scoring_func="sigmoid",
            use_grouped_topk=False,
            e_score_correction_bias=None,
            router_logits_dtype=torch.float32,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The gate runs inside the runner.
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


class FlintSm70Attention(nn.Module):
    """Attention with q/k RMSNorm, RoPE and learned sinks; fused or unfused
    q/k norm + RoPE + KV write, decided once per layer."""

    def __init__(
        self,
        config: FlintConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
        fused_enabled: bool = True,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = None
        if layer_idx not in config.no_rope_layer_indices:
            # fp16 cos/sin (default dtype); the fused op also admits fp32.
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters=_layer_rope_parameters(config, layer_idx),
            )
            if self.rotary_emb.rotary_dim != self.head_dim:
                raise ValueError("Flint applies RoPE to the full head dimension")

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if is_sliding else None
        self.softmax_offset = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(self.softmax_offset, {"weight_loader": _sink_weight_loader})
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
            sinks=self.softmax_offset,
        )
        # The decision is made once and is static per layer.
        is_sm70 = current_platform.is_cuda() and current_platform.is_device_capability(
            70
        )
        reason = fused_route_rejection(
            enabled=fused_enabled,
            is_sm70=is_sm70,
            dtype=torch.get_default_dtype(),
            head_dim=self.head_dim,
            q_eps=self.q_norm.variance_epsilon,
            k_eps=self.k_norm.variance_epsilon,
            rotary_emb=self.rotary_emb,
            kv_cache_dtype=self.attn.kv_cache_dtype,
            attn_type=self.attn.attn_type,
            # vLLM v0.33.0 Attention has no such attribute; the getattr
            # keeps the gate correct if it exists.
            calculate_kv_scales=bool(
                getattr(self.attn, "calculate_kv_scales", False)
            ),
            kv_sharing=self.attn.kv_sharing_target_layer_name is not None,
            backend_writes_kv_in_forward=(
                self.attn.attn_backend.forward_includes_kv_cache_update
            ),
        )
        self.fused_qk_rope_kv = reason is None
        if self.fused_qk_rope_kv:
            logger.info_once("Flint SM70: fused q/k norm, RoPE and KV-cache write.")
        else:
            logger.info_once("Flint SM70: unfused q/k norm/RoPE because %s.", reason)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Unfused route (stock Attention.forward)."""
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fused_qk_rope_kv:
            return self._forward_fused(positions, qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_fused(
        self, positions: torch.Tensor, qkv: torch.Tensor
    ) -> torch.Tensor:
        """Fused route on vllm_sm70/flint/qk_norm_rope_kv.py.

        The registered op normalizes q/k, applies RoPE and writes the KV
        cache (or hands the write to the backend's writer). We are compiled
        (@support_torch_compile), so we call the registered op; the platform
        hook makes it a splitting op next to ``unified_kv_cache_update``
        (vllm_sm70/flint/__init__.py ``install_splitting_op``). Then
        ``attention_after_fused_write`` runs ``unified_attention_with_output``
        without a second KV write; it consumes the op's q/k outputs, which
        orders it after the op.
        """
        num_tokens = qkv.shape[0]
        q = torch.empty((num_tokens, self.q_size), dtype=qkv.dtype, device=qkv.device)
        k = torch.empty((num_tokens, self.kv_size), dtype=qkv.dtype, device=qkv.device)
        v = qkv[:, self.q_size + self.kv_size :]
        _QK_ROPE_KV_OP(
            qkv,
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            positions,
            None if self.rotary_emb is None else self.rotary_emb.cos_sin_cache,
            self.q_norm.variance_epsilon,
            _encode_layer_name(self.attn.layer_name),
        )
        output = attention_after_fused_write(self.attn, q, k, v)
        out, _ = self.o_proj(output)
        return out


class FlintSm70DecoderLayer(nn.Module):
    """Decoder layer, with the FP16 stream scales from fp16_rescale."""

    def __init__(
        self,
        config: FlintConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
        fp16_stream_shift: bool = False,
        fused_enabled: bool = True,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.residual_scale, self.attn_out_scale = fp16_rescale.stream_scales(
            layer_idx, fp16_stream_shift
        )
        self.self_attn = FlintSm70Attention(
            config,
            layer_idx,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
            fused_enabled=fused_enabled,
        )
        self.mlp: FlintSm70MLP | FlintSm70SparseMoeBlock
        if layer_idx in config.mlp_only_layers:
            self.mlp = FlintSm70MLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = FlintSm70SparseMoeBlock(
                config, quant_config, prefix=f"{prefix}.mlp"
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        if self.attn_out_scale is not None:
            hidden_states = hidden_states * self.attn_out_scale
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.residual_scale is not None:
            residual = residual * self.residual_scale
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class FlintSm70Model(nn.Module):
    """Embedding, decoder layers and final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        # Per-layer sliding windows come from layer_types.
        if cache_config is not None:
            cache_config.sliding_window = None

        # Kill: VLLM_SM70_FLINT_QK_ROPE_KV=0 -> unfused q/k norm + RoPE and
        # upstream's Attention.forward.
        fused_enabled = bool(sm70_config(vllm_config).flint_qk_rope_kv)

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()
        fp16_stream_shift = vllm_config.model_config.dtype == torch.float16
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlintSm70DecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix=prefix,
                fp16_stream_shift=fp16_stream_shift,
                fused_enabled=fused_enabled,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class FlintSm70ForCausalLM(nn.Module, SupportsPP):
    """Flint causal LM on SM70."""

    # Opt in to the top-k candidate sampler (vllm_sm70/sample_v1.py
    # MODEL_OPT_IN). It samples from self.lm_head and self.logits_processor.
    # VLLM_SM70_FLINT_SAMPLER=0 is read in
    # Sm70CudaPlatform.get_hidden_state_sampler. The same opt-in admits the
    # greedy top-1 (vllm_sm70/sample_greedy.py); its kill switch is
    # VLLM_SM70_GREEDY_TOKEN_FASTPATH=0.
    supports_sm70_topk_sampler = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.fp16 = vllm_config.model_config.dtype == torch.float16
        self.model = FlintSm70Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            # Unquantized lm_head, plain embedding method. Its fp16 GEMM goes
            # through the platform GEMM hook.
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights through the lm_head scale, global-scale check and,
        in fp16, the FP16 rescale pipeline."""
        lm_head_scale = fp16_rescale.LmHeadScale()
        weights = lm_head_scale.intercept(weights)
        weights = _check_merged_global_scales(weights)
        if self.fp16:
            weights = fp16_rescale.fp16_weight_pipeline(weights)
        if self.config.tie_word_embeddings:
            # Tied: skip lm_head.* (AutoWeightsLoader has no skip_prefixes).
            weights = ((n, t) for n, t in weights if not n.startswith("lm_head."))
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        lm_head_scale.apply(
            None if isinstance(self.lm_head, PPMissingLayer) else self.lm_head,
            self.logits_processor,
        )
        return loaded


def _check_merged_global_scales(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Raise if the NVFP4 global scales of shards
    fused into one linear differ."""
    seen: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        if name.endswith(".weight_scale_2") and ".experts." not in name:
            for shard, merged in _MERGED_SHARDS.items():
                if shard in name:
                    key = name.replace(shard, merged)
                    value = tensor.detach().reshape(()).float().cpu()
                    ref = seen.setdefault(key, value)
                    if not torch.equal(ref, value):
                        raise ValueError(
                            f"{name}: global scale {value.item()} differs from "
                            f"the other shards of {key} ({ref.item()}); the "
                            "checkpoint must share one weight_scale_2 per "
                            "fused linear"
                        )
                    break
        yield name, tensor
