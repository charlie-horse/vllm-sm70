# SPDX-License-Identifier: Apache-2.0
"""Dense stateless oracle of the Inkling 2 text stack, for tests only.

`Inkling2Oracle` runs whole sequences through the ops of
`vllm_sm70.models.inkling2_sm70.ops.torch_ref` in the order of the Hugging
Face reference (`Inkling2LoopTextModel.forward` and `Inkling2LoopForCausalLM`):
no cache, no paging, every visit recomputed from position 0. `Inkling2Weights`
reads a checkpoint in the released names and dequantises its packed experts.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from vllm_sm70.models.inkling2_sm70 import weights as loader
from vllm_sm70.models.inkling2_sm70.configs import Inkling2ModelConfig
from vllm_sm70.models.inkling2_sm70.ops import torch_ref as ref

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class Inkling2Weights:
    """A checkpoint in the released names, as the reference holds it.

    `weights(name)` is the tensor `model.llm.<name>` in fp32 where the
    reference keeps fp32 (`weights.HF_FP32_MODULES`, by the loader's route
    table), else in `dtype`. A dtype other than fp32 needs `text_config`.
    """

    def __init__(
        self,
        native: dict[str, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        prefix: str = "model.llm.",
        text_config: dict | None = None,
    ):
        self.native = native
        self.dtype = dtype
        self.prefix = prefix
        self.text_config = text_config
        self._fp32: set[str] | None = None
        self._experts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def load(
        cls, directory: pathlib.Path | str, dtype: torch.dtype = torch.float32
    ) -> Inkling2Weights:
        directory = pathlib.Path(directory)
        config = json.loads((directory / "config.json").read_text())["text_config"]
        native = load_file(directory / "model.safetensors")
        return cls(native, dtype, text_config=config)

    def _keeps_fp32(self, key: str) -> bool:
        if self._fp32 is None:
            if self.text_config is None:
                raise ValueError("a dtype other than fp32 needs text_config")
            routes = loader.build_weight_map(
                Inkling2ModelConfig(**self.text_config),
                tp_size=1,
                tp_rank=0,
                dtype=self.dtype,
            ).routes
            self._fp32 = {
                source
                for source, route in routes.items()
                if any(m in route.hf_name for m in loader.HF_FP32_MODULES)
            }
        return key in self._fp32

    def __call__(self, name: str) -> torch.Tensor:
        key = self.prefix + name
        tensor = self.native[key]
        if self.dtype == torch.float32 or self._keeps_fp32(key):
            return tensor.float()
        return tensor.to(self.dtype)

    def _dequantize(self, key: str) -> torch.Tensor:
        """fp32 values of a packed expert tensor [E, out, in]."""
        native = self.native
        weight, scale = native[key], native[key + "_scale"].float()
        if key + "_amax" not in native:
            # MXFP8: e4m3 values times 2**(scale - 127) per 32 values.
            blocks = weight.float().unflatten(-1, (-1, 32))
            return (blocks * torch.exp2(scale - 127)[..., None]).flatten(-2)
        # NVFP4: low nibble first, one e4m3 scale per 16 values, one factor
        # per expert.
        lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1))
        codes = weight.long()
        values = torch.stack((lut[codes & 0xF], lut[codes >> 4]), dim=-1).flatten(-2)
        values = values.unflatten(-1, (-1, 16)) * scale[..., None]
        factor = native[key + "_amax"] / native[key + "_outer_scale_denominator"]
        return (values * factor[:, None, None, None]).flatten(-2)

    def experts(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up [E, 2 * I, H] with the gate rows first, down [E, H, I])."""
        if layer not in self._experts:
            p = f"{self.prefix}layers.{layer}.mlp.experts."
            w13 = self._dequantize(p + "w13_weight")
            gate_up = torch.cat((w13[:, 0::2], w13[:, 1::2]), dim=1)
            down = self._dequantize(p + "w2_weight")
            self._experts[layer] = (gate_up.to(self.dtype), down.to(self.dtype))
        return self._experts[layer]

    def attention(self, layer: int) -> ref.AttentionWeights:
        a = f"layers.{layer}.attn."
        dynamic = self.prefix + a + "k_sconv.static_w" in self.native
        conv = "static_w" if dynamic else "weight"
        return ref.AttentionWeights(
            q=self(a + "wq_du.weight"),
            k=self(a + "wk_dv.weight"),
            v=self(a + "wv_dv.weight"),
            rel=self(a + "wr_du.weight"),
            out=self(a + "wo_ud.weight"),
            out_gate=self(a + "gate_attn.weight"),
            sink=self(a + "attn_sink_proj.weight"),
            q_scale=self(a + "q_l2_clip.scale"),
            rel_scale=self(a + "rel_l2_clip.scale"),
            rel_proj=self(a + "rel_logits_proj.proj"),
            k_conv=self(a + f"k_sconv.{conv}"),
            v_conv=self(a + f"v_sconv.{conv}"),
            k_conv_proj=self(a + "k_sconv.weight_proj") if dynamic else None,
            v_conv_proj=self(a + "v_sconv.weight_proj") if dynamic else None,
        )

    def message_queries(self) -> torch.Tensor:
        """[loop layers, 4, H] in the order query, key, value, mlp."""
        q_mlp = self("middle_cycle_cross_pass_queries").unflatten(0, (-1, 2))
        k_v = self("middle_cycle_cross_pass_kv_queries").unflatten(0, (-1, 2))
        return torch.cat((q_mlp[:, :1], k_v, q_mlp[:, 1:]), dim=1)


@dataclasses.dataclass(frozen=True)
class OracleOutput:
    """One forward. V visits in execution order (`Inkling2Oracle.visit_layers`)."""

    inputs_embeds: torch.Tensor  # [B, T, H], the input of visit 0
    visit_states: torch.Tensor  # [V, B, T, H], the output of each visit
    router_logits: torch.Tensor  # [V, B, T, E], fp32
    final_hidden_state: torch.Tensor  # [B, T, H], the input of the LM head
    logits: torch.Tensor  # [B, T, unpadded vocab], fp32, capped


class Inkling2Oracle:
    """The text stack of one checkpoint over whole sequences."""

    def __init__(self, text_config: dict, weights: Inkling2Weights):
        self.config = text_config
        self.weights = weights
        loop = text_config["loop_config"]
        self.loop_start, self.loop_end = loop["core_start"], loop["core_end"]
        self.num_passes = loop["num_passes"]
        self.block = text_config["n_layers_per_attn_res_block"]
        self.local_layers = frozenset(text_config["local_layer_ids"])
        layers = range(text_config["num_hidden_layers"])
        loop_layers = list(layers[self.loop_start : self.loop_end])
        self.visit_layers: list[int] = (
            list(layers[: self.loop_start])
            + loop_layers * self.num_passes
            + list(layers[self.loop_end :])
        )

    @classmethod
    def from_checkpoint(
        cls, directory: pathlib.Path | str, dtype: torch.dtype = torch.float32
    ) -> Inkling2Oracle:
        """The oracle of `directory`'s config.json and model.safetensors."""
        directory = pathlib.Path(directory)
        config = json.loads((directory / "config.json").read_text())["text_config"]
        return cls(config, Inkling2Weights.load(directory, dtype))

    def is_full(self, layer: int) -> bool:
        return layer not in self.local_layers

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normed token embeddings [..., H]. Image features replace rows of it."""
        w = self.weights
        embedded = F.embedding(input_ids, w("embed.weight"))
        return ref.rms_norm(
            embedded, w("embed_norm.weight"), self.config["rms_norm_eps"]
        )

    def _gate(
        self, layer: int, name: str, hidden: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        c = f"layers.{layer}.controllers.{name}."
        w = self.weights
        return ref.message_gate(
            hidden,
            message,
            w(c + "down_weight"),
            w(c + "forget_weight"),
            w(c + "gain_weight"),
            self.config["rms_norm_eps"],
        )

    def _layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        blocks: torch.Tensor,
        messages: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One visit: (output [B, T, H], block states, router logits [B * T, E])."""
        cfg, w = self.config, self.weights
        p = f"layers.{layer}."
        eps = cfg["rms_norm_eps"]
        dtype = hidden.dtype
        full = self.is_full(layer)
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        if not full:
            allowed = allowed & (distance < cfg["sliding_window_size"])

        # Query, key and value inputs: the block residual, the previous
        # pass's messages, then the norm.
        if layer == 0:
            inputs = [hidden] * 3
        else:
            query = w(f"attn_res.attn_queries.{layer - 1}")
            inputs = list(ref.attn_res(blocks, hidden, query).unbind(-2))
            inputs = inputs * (3 // len(inputs))
        if messages is not None:
            inputs = [
                self._gate(layer, name, x, messages[..., i, :])
                for i, (name, x) in enumerate(zip(("attn", "k", "v"), inputs))
            ]
        norms = (
            ("q_input_norm", "k_input_norm", "v_input_norm")
            if full
            else ("attn_norm",) * 3
        )
        inputs = [
            ref.rms_norm(x, w(f"{p}{norm}.weight"), eps).to(dtype)
            for norm, x in zip(norms, inputs)
        ]
        out, _ = ref.attention(
            *inputs,
            w.attention(layer),
            num_kv_heads=cfg[
                "num_key_value_heads" if full else "swa_num_key_value_heads"
            ],
            allowed=allowed,
            relative_distance=distance,
            position_ids=positions[None],
            qk_clip=cfg["qk_l2_clip_threshold"],
            rel_clip=cfg["rel_l2_clip_threshold"],
            eps=eps,
            log_scaling=(cfg["log_scaling_alpha"], cfg["log_scaling_n_floor"])
            if full
            else None,
        )
        out = ref.static_conv(out, w(p + "attn_sconv.weight")).float()

        # A block boundary restarts the residual from the attention output.
        if layer % self.block == 0:
            blocks = torch.cat([blocks, hidden.unsqueeze(-2)], dim=-2)
            residual = out
        else:
            residual = hidden.float() + out

        x = ref.attn_res(blocks, residual, w(f"attn_res.mlp_queries.{layer}"))[
            ..., 0, :
        ]
        if messages is not None:
            x = self._gate(layer, "mlp", x, messages[..., 3, :])
        x = ref.rms_norm(x, w(p + "mlp_norm.weight"), eps).to(dtype)
        router_logits, top_k_weights, top_k_index = ref.router(
            x,
            w(p + "mlp.gate.weight"),
            w(p + "mlp.gate.bias"),
            cfg["num_experts_per_tok"],
            cfg["route_scale"],
        )
        y = ref.experts(
            x.reshape(-1, x.shape[-1]),
            top_k_index,
            top_k_weights,
            *w.experts(layer),
            cfg["mlp_gate_softcap"],
            cfg["mlp_up_softcap"],
        ).reshape(x.shape)
        y = ref.static_conv(y, w(p + "mlp_sconv.weight"))
        return (residual + y).to(dtype), blocks, router_logits

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> OracleOutput:
        """One prefill of input_ids [B, T] or of inputs_embeds [B, T, H].

        `inputs_embeds` is `embed(input_ids)` with any image features already
        written over the image-token rows.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids and inputs_embeds")
        w = self.weights
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        inputs_embeds = inputs_embeds.to(w.dtype)
        states: list[torch.Tensor] = []
        routers: list[torch.Tensor] = []

        def run(layers, hidden, blocks, previous=None):
            for layer in layers:
                messages = None
                if previous is not None:
                    query = message_queries[layer - self.loop_start]
                    messages = ref.attn_res(*previous, query)
                hidden, blocks, logits = self._layer(layer, hidden, blocks, messages)
                states.append(hidden)
                routers.append(logits.unflatten(0, hidden.shape[:2]))
            return hidden, blocks

        num_layers = self.config["num_hidden_layers"]
        loop = range(self.loop_start, self.loop_end)
        blocks = inputs_embeds.new_zeros(
            *inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1]
        )
        pre_hidden, pre_blocks = run(range(self.loop_start), inputs_embeds, blocks)
        hidden, blocks = run(loop, pre_hidden, pre_blocks)
        message_queries = w.message_queries()
        for _ in range(1, self.num_passes):
            # Messages read the previous pass's own block states and its
            # output; the pass restarts from the state before the loop.
            previous = (blocks[..., pre_blocks.shape[-2] + 1 :, :], hidden)
            hidden, blocks = run(loop, pre_hidden, pre_blocks, previous)
        hidden, blocks = run(range(self.loop_end, num_layers), hidden, blocks)

        final = ref.attn_res(blocks, hidden, w("attn_res.output_query"))[..., 0, :]
        final = ref.rms_norm(final, w("norm.weight"), self.config["rms_norm_eps"])
        final = final.to(inputs_embeds.dtype)
        head = w("unembed.weight")[: self.config["unpadded_vocab_size"]]
        logits = ref.logit_cap(
            F.linear(final, head), self.config["final_logit_softcapping"]
        )
        return OracleOutput(
            inputs_embeds=inputs_embeds,
            visit_states=torch.stack(states),
            router_logits=torch.stack(routers),
            final_hidden_state=final,
            logits=logits,
        )

    def greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy continuation [B, max_new_tokens] of prompt_ids [B, T].

        Every step is a full prefill of the sequence so far.
        """
        ids = prompt_ids
        for _ in range(max_new_tokens):
            token = self.forward(ids).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, token], dim=1)
        return ids[:, prompt_ids.shape[1] :]
