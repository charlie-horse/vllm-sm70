# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint on SM70 (2x V100, fp16, TP2), on UP vLLM v0.33.0 (4ba787e) layers.

Port of charlie-horse/vLLM cf9492a ``vllm/models/flint/model.py`` (753 L). It
reuses UP ``vllm.models.flint.configs.FlintConfig`` (identical to the fork's
except one comment line, UP vllm/models/flint/configs.py:88). Each fork-only
core hook is replaced as follows (UP paths under vllm/model_executor/layers/
unless given in full):

=====================================  =======================================
fork (cf9492a)                         here (UP 4ba787e)
=====================================  =======================================
FusedMoE(expert_mapping=...) :211-226  FusedMoEFactory (UP fused_moe/layer.py:
                                       92). RoutedExperts.load_weights builds
                                       its own mapping (UP fused_moe/
                                       routed_experts.py:1004-1008)
runner.router_provider :229            dropped in v1 (fused router deferred)
Attention.forward(kv_cache_updated=)   K15's op (torch.ops.vllm.<QK_ROPE_KV_
  :386                                 OP_NAME>), then K15's attention_after_
                                       fused_write (vllm_sm70/flint/)
kernel_config.sm70_flint_qk_rope_kv    sm70_config(vllm_config).flint_qk_rope_kv
  :314-318
layer_policy(...).value("flint_        dropped in v1 (sampler deferred)
  sampler") :570-577
AutoWeightsLoader(skip_prefixes=)      filter lm_head.* when tied (UP
  :629-632                             AutoWeightsLoader has no skip_prefixes,
                                       vllm/model_executor/models/
                                       utils.py:242-248)
FP16 rescale :87-107, :402-447, :623-  fp16_rescale.py
  628, :647-668, :696-753
=====================================  =======================================

Not ported (fork cf9492a model.py): ``_expert_params_mapping`` :117-126 (UP
RoutedExperts builds the mapping), ``get_expert_mapping`` :616-617, the
sampler hooks ``_get_top_tokens`` :597-603 and ``_sample_topk_candidates``
:605-614 (deferred to M10), and the fused-router hook :227-229 (M10).

Quantization isn't special-cased here: dense ModelOpt W4A16 linears go to
M3 (vllm_sm70/quant/nvfp4_linear.py) and routed experts to M4
(vllm_sm70/quant/nvfp4_moe.py) through the quant config. The fp16 lm_head
and router GEMMs go through Kai's U1 platform GEMM hook.
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
from vllm.models.flint.configs import FlintConfig  # UP, reused as is
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name

from vllm_sm70.config import sm70_config

# K15 (#54). Importing qk_norm_rope_kv registers the op in torch.ops.vllm.
from vllm_sm70.flint import QK_ROPE_KV_OP_NAME
from vllm_sm70.flint.qk_norm_rope_kv import (
    attention_after_fused_write,
    fused_route_rejection,
)

from . import fp16_rescale

logger = init_logger("vllm.sm70.flint")

# K15's fused q/k norm + RoPE + KV-write op, torch.ops.vllm.<QK_ROPE_KV_OP_NAME>.
_QK_ROPE_KV_OP = getattr(torch.ops.vllm, QK_ROPE_KV_OP_NAME)

# Shards that one fused linear loads. A fused NVFP4 linear keeps a single
# global scale, so the shards' weight_scale_2 must agree (fork :76-84).
# ported from FORK vllm/models/flint/model.py:76-84
_MERGED_SHARDS = {
    ".q_proj.": ".qkv_proj.",
    ".k_proj.": ".qkv_proj.",
    ".v_proj.": ".qkv_proj.",
    ".gate_proj.": ".gate_up_proj.",
    ".up_proj.": ".gate_up_proj.",
}


# ported from FORK vllm/models/flint/model.py:110-114
def _sink_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Load this rank's heads of the [total_num_heads] sink tensor (fork
    :110-114). UP's ``make_attention_sink`` (UP model_loader/weight_utils.py:
    1584) would also do; this keeps the fork's fp32 in-place semantics."""
    heads_per_rank = param.shape[0]
    start = get_tensor_model_parallel_rank() * heads_per_rank
    param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))


# ported from FORK vllm/models/flint/model.py:129-137
def _layer_rope_parameters(config: FlintConfig, layer_idx: int) -> dict:
    """fork :129-137, verbatim."""
    rope_parameters = dict(config.rope_parameters or {})
    if rope_parameters and all(isinstance(v, dict) for v in rope_parameters.values()):
        layer_type = config.layer_types[layer_idx]
        rope_parameters = dict(rope_parameters.get(layer_type) or {})
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    return rope_parameters


# ported from FORK vllm/models/flint/model.py:140-170
class FlintSm70MLP(nn.Module):
    """fork :140-170 verbatim. Dense linears go to the SM70 NVFP4 kernel via
    the quant config (Kai/Otto); nothing model-side."""

    # ported from FORK vllm/models/flint/model.py:141-165
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

    # ported from FORK vllm/models/flint/model.py:167-170
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


# ported from FORK vllm/models/flint/model.py:173-233
class FlintSm70SparseMoeBlock(nn.Module):
    """fork :173-233. Sigmoid top-k routed experts plus an ungated shared
    expert, combined and all-reduced inside the MoE runner.

    The routed-expert method (ModelOptNvFp4Sm70MoEMethod, Otto's
    fused_moe/method.py) comes from the quant config, not from here.
    """

    # ported from FORK vllm/models/flint/model.py:177-229
    def __init__(
        self,
        config: FlintConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.router_score_function != "sigmoid":  # fork :184-188
            raise ValueError(
                "Flint routes with sigmoid scores, got "
                f"router_score_function={config.router_score_function!r}"
            )
        if get_tensor_model_parallel_world_size() > config.num_experts:
            raise ValueError("Tensor parallel size exceeds the number of experts.")
        # fork :196-203. UP GateLinear has out_dtype/force_fp32_compute
        # (UP fused_moe/router/gate_linear.py:62-74). The fp16 GEMM backend is
        # Kai's platform hook U1.
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
        # FORK vllm/models/flint/model.py:211-226 -> UP FusedMoEFactory
        # (UP vllm/model_executor/layers/fused_moe/layer.py:92-147). Dropped:
        # expert_mapping=, since UP builds the mapping
        # (vllm/model_executor/layers/fused_moe/routed_experts.py:1007) from the
        # default ckpt_names=("gate_proj","down_proj","up_proj")
        # (vllm/model_executor/layers/fused_moe/layer.py:125); and the fork's
        # runner.router_provider (:229, fused router deferred).
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

    # ported from FORK vllm/models/flint/model.py:231-233
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The gate runs inside the runner, as in UP models/flint/model.py:240.
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


# ported from FORK vllm/models/flint/model.py:236-388
class FlintSm70Attention(nn.Module):
    """fork :236-388."""

    # ported from FORK vllm/models/flint/model.py:237-347
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
            # fp16 cos/sin (default dtype), as the fork built it. K15 also
            # admits fp32, which UP Flint uses (UP :287-292).
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
        # fork :312-347. The decision is made once and is static per layer.
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
            # UP v0.33.0 Attention computes no KV scales at run time (no
            # such attribute); the getattr keeps K15's gate if it returns.
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

    # ported from FORK vllm/models/flint/model.py:349-366
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Unfused route, fork :349-366 verbatim (stock Attention.forward)."""
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

    # ported from FORK vllm/models/flint/model.py:368-388
    def _forward_fused(
        self, positions: torch.Tensor, qkv: torch.Tensor
    ) -> torch.Tensor:
        """fork :368-388, on K15 (vllm_sm70/flint/qk_norm_rope_kv.py).

        K15's registered op normalizes q/k, applies RoPE and writes the KV
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


# ported from FORK vllm/models/flint/model.py:391-449
class FlintSm70DecoderLayer(nn.Module):
    """fork :391-449, with the FP16 stream scales from fp16_rescale."""

    # ported from FORK vllm/models/flint/model.py:392-429
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

    # ported from FORK vllm/models/flint/model.py:431-449
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


# ported from FORK vllm/models/flint/model.py:452-524 (FlintModel)
@support_torch_compile
class FlintSm70Model(nn.Module):
    """fork :452-524."""

    # ported from FORK vllm/models/flint/model.py:454-495
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        # Per-layer sliding windows come from layer_types (fork :461-465).
        if cache_config is not None:
            cache_config.sliding_window = None

        # Kill: VLLM_SM70_FLINT_QK_ROPE_KV=0 -> unfused q/k norm + RoPE and
        # upstream's Attention.forward (fork read kernel_config, :312-318).
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

    # ported from FORK vllm/models/flint/model.py:497-498
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # ported from FORK vllm/models/flint/model.py:500-524
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


# ported from FORK vllm/models/flint/model.py:527-668
class FlintSm70ForCausalLM(nn.Module, SupportsPP):
    """fork :527-668, minus the sampler hooks (:570-577, :597-614, deferred)."""

    # fork :528-542 == UP models/flint/model.py:600-614
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

    # ported from FORK vllm/models/flint/model.py:544-580
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
            # Unquantized lm_head, plain embedding method (fork :555-564). Its
            # fp16 GEMM goes through Kai's platform hook U1.
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

    # ported from FORK vllm/models/flint/model.py:582-583
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    # ported from FORK vllm/models/flint/model.py:585-592
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    # ported from FORK vllm/models/flint/model.py:594-595
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ported from FORK vllm/models/flint/model.py:619-636
    # (+ :638-668 via fp16_rescale.LmHeadScale)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """fork :619-636."""
        lm_head_scale = fp16_rescale.LmHeadScale()
        weights = lm_head_scale.intercept(weights)
        weights = _check_merged_global_scales(weights)
        if self.fp16:
            weights = fp16_rescale.fp16_weight_pipeline(weights)
        if self.config.tie_word_embeddings:
            # Replaces the fork's AutoWeightsLoader(skip_prefixes=["lm_head."]).
            weights = ((n, t) for n, t in weights if not n.startswith("lm_head."))
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        lm_head_scale.apply(
            None if isinstance(self.lm_head, PPMissingLayer) else self.lm_head,
            self.logits_processor,
        )
        return loaded


# ported from FORK vllm/models/flint/model.py:671-693
def _check_merged_global_scales(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """fork :671-693 verbatim. Raise if the NVFP4 global scales of shards
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
