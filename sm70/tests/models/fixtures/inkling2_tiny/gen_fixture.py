# SPDX-License-Identifier: Apache-2.0
"""Generator of the Inkling 2 tiny fixtures (this directory).

Runs the Hugging Face reference classes (transformers at TRANSFORMERS_COMMIT,
CPU fp32) on a tiny random checkpoint written in the released checkpoint's
names and layouts, and saves what the port is compared with. It needs that
transformers tree first on PYTHONPATH; the tests that read the fixtures do
not. Usage and file list: PROVENANCE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pathlib
import re
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

HERE = pathlib.Path(__file__).resolve().parent
TRANSFORMERS_COMMIT = "2f1bc5e321b9a736200a590d6e7743ccdeb60772"
# The data seed is one whose routing has no near tie (min_router_gap in
# manifest.json), so fp32 rounding cannot change an expert choice.
WEIGHT_SEED, DATA_SEED = 7002, 7014
FILES = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors",
    "ops.safetensors",
    "cases.safetensors",
)
# Exit codes of --check.
SAME, DIFFERENT, NOT_REFERENCE, NOT_REPRODUCIBLE = 0, 1, 2, 3
# Bound on float differences between torch builds and CPU instruction sets.
FLOAT_TOLERANCE = 1e-5

H, EXPERTS, INTER, RANK, KERNEL = 64, 8, 32, 2, 4
VOCAB, UNPADDED, IMAGE_TOKEN = 80, 72, 71
FULL = (6, 2, 4)  # attention heads, key/value heads, head dim
SWA = (4, 2, 4)
D_REL = 4
VIS = {
    "hidden": 16,
    "layers": 2,
    "heads": 2,
    "patch": 4,
    "merge": 2,
    "grid": 4,
    "proj": 24,
    "mlp": 32,
}
NVFP4_LAYERS = range(20)
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# Real field names and structure (data/inkling2_mini/config.json), tiny values.
CONFIG = {
    "dtype": "bfloat16",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 1,
    "image_token_id": IMAGE_TOKEN,
    "text_config": {
        "model_type": "inkling2_loop_text",
        "eos_token_id": 1,
        "model_max_length": 4096,
        "torch_dtype": "bfloat16",
        "hidden_size": H,
        "num_hidden_layers": 24,
        "vocab_size": VOCAB,
        "num_attention_heads": FULL[0],
        "num_key_value_heads": FULL[1],
        "head_dim": FULL[2],
        "d_rel": D_REL,
        "rel_extent": 8,
        "q_bias": False,
        "o_bias": False,
        "log_scaling_n_floor": 16,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True,
        "qk_l2_clip_threshold": 2.0,
        "rel_l2_clip_threshold": 1.0,
        "local_layer_ids": [i for i in range(24) if i % 6 != 5],
        "sconv_kernel_size": KERNEL,
        "unpadded_vocab_size": UNPADDED,
        "final_logit_softcapping": 30.0,
        "swa_head_dim": SWA[2],
        "swa_num_attention_heads": SWA[0],
        "swa_num_key_value_heads": SWA[1],
        "sliding_window_size": 8,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": True,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "n_shared_experts": 0,
        "intermediate_size": INTER,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": True,
        "loop_config": {
            "core_start": 6,
            "core_end": 18,
            "num_passes": 3,
            "gate_rank": RANK,
            "is_varloop": False,
        },
    },
    "vision_config": {
        "model_type": "inkling2_vision",
        "text_hidden_size": H,
        "temporal_patch_size": 2,
        "native_patch_size": VIS["patch"],
        "adapter_merge_size": VIS["merge"],
        "projection_hidden_dim": VIS["proj"],
        "projector_bias": False,
        "hidden_size": VIS["hidden"],
        "num_hidden_layers": VIS["layers"],
        "num_attention_heads": VIS["heads"],
        "mlp_ratio": VIS["mlp"] / VIS["hidden"],
        "posemb_grid_size": VIS["grid"],
        "layer_norm_eps": 1e-05,
    },
    "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
}

# Modules whose inputs and outputs go to ops.safetensors: op name -> (module
# path, calls to keep). Calls of a module are counted from 0 within the one
# forward; a loop layer's module is called once per pass, message_input once
# per loop layer in passes 2 and 3.
LM = "model.language_model."
OPS = {
    "embed": (LM + "embed_tokens", (0,)),
    "rmsnorm": (LM + "layers.1.input_layernorm", (0,)),
    "rmsnorm_unweighted": (LM + "layers.1.self_attn.v_proj.norm", (0,)),
    "l2_clip": (LM + "layers.1.self_attn.k_proj.norm", (0,)),
    "scaled_l2_clip": (LM + "layers.1.self_attn.q_proj.norm", (0,)),
    "log_scaled_l2_clip": (LM + "layers.5.self_attn.q_proj.norm", (0,)),
    "log_scale": (LM + "layers.5.self_attn.q_proj.norm.log_scale", (0,)),
    "rel_l2_clip_sliding": (
        LM + "layers.1.self_attn.relative_position_bias.norm",
        (0,),
    ),
    "rel_l2_clip_full": (LM + "layers.5.self_attn.relative_position_bias.norm", (0,)),
    "rel_bias_sliding": (LM + "layers.1.self_attn.relative_position_bias", (0,)),
    "rel_bias_full": (LM + "layers.5.self_attn.relative_position_bias", (0,)),
    "sconv_static_key": (LM + "layers.1.self_attn.k_proj.sconv", (0,)),
    "sconv_static_hidden": (LM + "layers.1.post_attention_convolution", (0,)),
    "sconv_dynamic_key": (LM + "layers.5.self_attn.k_proj.sconv", (0,)),
    "sconv_dynamic_value": (LM + "layers.5.self_attn.v_proj.sconv", (0,)),
    "query_proj_full": (LM + "layers.5.self_attn.q_proj", (0,)),
    "key_proj_sliding": (LM + "layers.1.self_attn.k_proj", (0,)),
    "value_proj_full": (LM + "layers.5.self_attn.v_proj", (0,)),
    "gated_output_proj": (LM + "layers.1.self_attn.o_proj", (0,)),
    "attn_sliding": (LM + "layers.1.self_attn", (0,)),
    "attn_full": (LM + "layers.5.self_attn", (0,)),
    "attn_sliding_loop": (LM + "layers.6.self_attn", (1,)),
    "attn_full_loop": (LM + "layers.11.self_attn", (2,)),
    "attn_res_one_query": (LM + "layers.1.attention_input", (0,)),
    "attn_res_three_queries": (LM + "layers.5.attention_input", (0,)),
    "attn_res_mlp": (LM + "layers.12.mlp_input", (1,)),
    "attn_res_final": (LM + "final_input", (0,)),
    "attn_res_message": (LM + "message_input", (17,)),
    "message_gate_query": (LM + "layers.6.q_message_gate", (0,)),
    "message_gate_key": (LM + "layers.11.k_message_gate", (1,)),
    "router": (LM + "layers.1.mlp.gate", (0,)),
    "swiglu": (LM + "layers.1.mlp.experts.act_fn", (0,)),
    "experts_nvfp4": (LM + "layers.1.mlp.experts", (0,)),
    "experts_mxfp8": (LM + "layers.23.mlp.experts", (0,)),
    "logit_softcap": ("logit_softcap", (0,)),
}
# Attention modules also return the attention weights; only these keep them.
KEEP_WEIGHTS = ("attn_sliding.0", "attn_full.0")


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


# Standard deviations of the random weights. They keep the softmax logits,
# the message gates and the expert share of the residual stream small enough
# that fp32 rounding is amplified about 20 times over the 48 visits.
STD = {
    "query": 0.5,  # attn_res and message queries
    "q": 0.15,
    "k": 0.15,
    "v": 0.2,
    "rel": 0.05,
    "rel_proj": 0.5,
    "out": 0.4,
    "out_gate": 0.2,
    "sink": 0.1,
    "conv": 0.5,
    "dynamic_conv": 0.1,
    "router": 0.3,
    "router_bias": 0.5,
    "gate_down": 0.05,
    "gate_out": 0.15,
    "w13": 0.12,
    "w2": 0.005,
    "unembed": 0.1,
}


def native_weights(g: torch.Generator) -> dict[str, torch.Tensor]:
    """A random checkpoint in the released names, shapes and dtypes."""

    def n(*shape, std=1.0, mean=0.0):
        return torch.randn(*shape, generator=g) * std + mean

    def norm_weight():
        return _bf16(n(H, std=0.2, mean=1.0))

    w: dict[str, torch.Tensor] = {}
    llm = "model.llm."
    w[llm + "embed.weight"] = _bf16(n(VOCAB, H))
    w[llm + "embed_norm.weight"] = norm_weight()
    w[llm + "norm.weight"] = norm_weight()
    # Padded rows are not zero, so a load that keeps them changes the logits.
    w[llm + "unembed.weight"] = _bf16(n(VOCAB, H, std=STD["unembed"]))
    w[llm + "attn_res.output_query"] = n(H, std=STD["query"])
    w[llm + "middle_cycle_cross_pass_queries"] = n(2 * 12, H, std=STD["query"])
    w[llm + "middle_cycle_cross_pass_kv_queries"] = n(2 * 12, H, std=STD["query"])
    for i in range(24):
        p = f"{llm}layers.{i}."
        full = i % 6 == 5
        heads, kv_heads, head_dim = FULL if full else SWA
        q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
        if i > 0:
            shape = (3, H) if full else (H,)
            w[f"{llm}attn_res.attn_queries.{i - 1}"] = n(*shape, std=STD["query"])
        w[f"{llm}attn_res.mlp_queries.{i}"] = n(H, std=STD["query"])
        w[p + "attn.wq_du.weight"] = _bf16(
            n(q_dim, H, std=STD["q"] * (4 / head_dim) ** 0.5)
        )
        w[p + "attn.wk_dv.weight"] = _bf16(
            n(kv_dim, H, std=STD["k"] * (4 / head_dim) ** 0.5)
        )
        w[p + "attn.wv_dv.weight"] = _bf16(n(kv_dim, H, std=STD["v"]))
        w[p + "attn.wr_du.weight"] = _bf16(n(heads * D_REL, H, std=STD["rel"]))
        w[p + "attn.wo_ud.weight"] = _bf16(
            n(H, q_dim, std=STD["out"] * (16 / q_dim) ** 0.5)
        )
        w[p + "attn.gate_attn.weight"] = _bf16(n(q_dim, H, std=STD["out_gate"]))
        w[p + "attn.attn_sink_proj.weight"] = _bf16(n(heads, H, std=STD["sink"]))
        w[p + "attn.q_l2_clip.scale"] = _bf16(n(heads, head_dim, std=0.2, mean=1.0))
        w[p + "attn.rel_l2_clip.scale"] = _bf16(n(heads, D_REL, std=0.2, mean=1.0))
        # Sliding layers take their rel extent from the window (both 8 here).
        w[p + "attn.rel_logits_proj.proj"] = _bf16(n(D_REL, 8, std=STD["rel_proj"]))
        for name in ("k", "v"):
            conv = p + f"attn.{name}_sconv."
            if full:
                w[conv + "static_w"] = _bf16(n(KERNEL, kv_dim, std=STD["conv"]))
                w[conv + "weight_proj"] = _bf16(
                    n(kv_heads * KERNEL, H, std=STD["dynamic_conv"])
                )
            else:
                w[conv + "weight"] = _bf16(n(kv_dim, 1, KERNEL, std=STD["conv"]))
        for name in (
            ("q_input_norm", "k_input_norm", "v_input_norm") if full else ("attn_norm",)
        ):
            w[p + f"{name}.weight"] = norm_weight()
        w[p + "mlp_norm.weight"] = norm_weight()
        w[p + "attn_sconv.weight"] = _bf16(n(H, 1, KERNEL, std=STD["conv"]))
        w[p + "mlp_sconv.weight"] = _bf16(n(H, 1, KERNEL, std=STD["conv"]))
        w[p + "mlp.gate.weight"] = _bf16(n(EXPERTS, H, std=STD["router"]))
        w[p + "mlp.gate.bias"] = n(EXPERTS, std=STD["router_bias"])
        if 6 <= i < 18:
            for name in ("attn", "k", "v", "mlp"):
                c = p + f"controllers.{name}."
                w[c + "down_weight"] = _bf16(n(RANK, 2 * H, std=STD["gate_down"]))
                w[c + "forget_weight"] = _bf16(n(H, RANK, std=STD["gate_out"]))
                w[c + "gain_weight"] = _bf16(n(H, RANK, std=STD["gate_out"]))
        # w13 rows interleave gate and up; w2 is (hidden, intermediate).
        for name, rows, cols in (("w13", 2 * INTER, H), ("w2", H, INTER)):
            e = p + f"mlp.experts.{name}_weight"
            if i in NVFP4_LAYERS:
                # Two E2M1 codes per byte, one e4m3 scale per 16 values, and
                # amax / denominator per expert. Random codes have rms 2.93.
                w[e] = torch.randint(
                    0, 256, (EXPERTS, rows, cols // 2), generator=g
                ).to(torch.uint8)
                scale = torch.rand(EXPERTS, rows, cols // 16, generator=g) + 0.5
                w[e + "_scale"] = scale.to(torch.float8_e4m3fn)
                amax = (
                    STD[name]
                    * 2688.0
                    / 2.93
                    * (0.7 + 0.6 * torch.rand(EXPERTS, generator=g))
                )
                w[e + "_amax"] = amax
                w[e + "_outer_scale_denominator"] = torch.tensor([2688.0])
            else:
                # e4m3 values, one ue8m0 exponent (2 ** (scale - 127)) per 32.
                scale = torch.randint(
                    121, 124, (EXPERTS, rows, cols // 32), generator=g
                )
                w[e] = n(EXPERTS, rows, cols, std=STD[name] * 32).to(
                    torch.float8_e4m3fn
                )
                w[e + "_scale"] = scale.to(torch.uint8)

    vis, hv = "model.visual.", VIS["hidden"]
    emb = vis + "backbone.embeddings."
    w[emb + "class_embedding"] = _bf16(n(hv))
    w[emb + "cls_position_embedding"] = _bf16(n(hv))
    w[emb + "patch_embedding.weight"] = _bf16(n(hv, 3 * VIS["patch"] ** 2, std=0.2))
    w[emb + "position_embedding"] = _bf16(n(VIS["grid"] ** 2, hv))
    for name in ("backbone.ln_pre",):
        w[vis + name + ".weight"] = _bf16(n(hv, std=0.2, mean=1.0))
        w[vis + name + ".bias"] = _bf16(n(hv, std=0.2))
    for i in range(VIS["layers"]):
        p = f"{vis}backbone.layers.{i}."
        for ln in ("ln_1", "ln_2"):
            w[p + ln + ".weight"] = _bf16(n(hv, std=0.2, mean=1.0))
            w[p + ln + ".bias"] = _bf16(n(hv, std=0.2))
        for ls in ("ls_1", "ls_2"):
            w[p + ls + ".gamma"] = _bf16(n(hv, std=0.2, mean=0.5))
        for name, out_f, in_f in (
            ("self_attn.q_proj", hv, hv),
            ("self_attn.k_proj", hv, hv),
            ("self_attn.v_proj", hv, hv),
            ("self_attn.out_proj", hv, hv),
            ("mlp.c_fc", VIS["mlp"], hv),
            ("mlp.c_proj", hv, VIS["mlp"]),
        ):
            w[p + name + ".weight"] = _bf16(n(out_f, in_f, std=0.2))
            w[p + name + ".bias"] = _bf16(n(out_f, std=0.2))
    w[vis + "proj_in.weight"] = _bf16(n(VIS["proj"], VIS["merge"] ** 2 * hv, std=0.2))
    w[vis + "proj_out.weight"] = _bf16(n(H, VIS["proj"], std=0.2))
    return w


def quant_config(weights: dict[str, torch.Tensor]) -> dict:
    """hf_quant_config.json with the released file's groups, tiny tensor list."""
    experts = re.compile(r"layers\.(\d+)\.mlp\.experts\.w(13|2)_weight$")
    groups: dict[str, list[str]] = {
        "float32": [],
        "bfloat16": [],
        "nvfp4": [],
        "mxfp8": [],
        "unembed": [],
    }
    for name, tensor in weights.items():
        if m := experts.search(name):
            groups["nvfp4" if int(m[1]) in NVFP4_LAYERS else "mxfp8"].append(name)
        elif ".mlp.experts." in name:
            continue
        elif name.endswith("unembed.weight"):
            groups["unembed"].append(name)
        else:
            groups[str(tensor.dtype).removeprefix("torch.")].append(name)
    cast = {
        "required_activation_quantization": None,
        "weight_block_shape": [],
        "weight_transform": "cast",
    }
    return {
        "model": {
            "parameter_groups": [
                {
                    "compute_dtype": "float32",
                    "output_dtype": "float32",
                    **cast,
                    "tensors": sorted(groups["float32"]),
                },
                {
                    "compute_dtype": "bfloat16",
                    "output_dtype": "bfloat16",
                    **cast,
                    "tensors": sorted(groups["bfloat16"]),
                },
                {
                    "compute_dtype": "nvfp4",
                    "output_dtype": "uint8",
                    "required_activation_quantization": {
                        "amax_type": "per_token",
                        "block_size": 16,
                        "format": "nvfp4_e2m1",
                        "rounding": "nearest",
                        "scale_dtype": "float8_e4m3fn",
                        "scale_selection_rule": "fouroversix_mae",
                    },
                    "weight_block_shape": [1, 16],
                    "weight_scale_format": "amax_denominator",
                    "weight_transform": "nvfp4_nearest",
                    "tensors": sorted(groups["nvfp4"]),
                },
                {
                    "compute_dtype": "mxfp8",
                    "output_dtype": "float8_e4m3fn",
                    "required_activation_quantization": {
                        "block_size": 32,
                        "format": "mxfp8_e4m3",
                        "rounding": "nearest",
                        "scale_dtype": "ue8m0",
                    },
                    "weight_block_shape": [32, 32],
                    "weight_transform": "mxfp8_nearest",
                    "tensors": sorted(groups["mxfp8"]),
                },
                {
                    "compute_dtype": "bfloat16",
                    "fwd_output_dtype": "float32",
                    "output_dtype": "bfloat16",
                    **cast,
                    "tensors": groups["unembed"],
                },
            ]
        },
        "quant_method": "inkling2",
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_digest(t: torch.Tensor) -> list:
    """Shape, sum, sum of magnitudes and an order-sensitive sum (fp64)."""
    f = t.detach().double().flatten()
    ramp = torch.arange(f.numel(), dtype=torch.float64) % 251 + 1
    return [
        list(t.shape),
        f.sum().item(),
        f.abs().sum().item(),
        (f * ramp).sum().item(),
    ]


class Recorder:
    """Forward hooks that keep the tensors going in and out of the OPS modules."""

    def __init__(self, root: torch.nn.Module):
        self.tensors: dict[str, torch.Tensor] = {}
        self.index: dict[str, dict] = {}
        self.handles = []
        self.calls: dict[str, int] = {}
        self.by_content: dict[str, str] = {}
        for op, (path, calls) in OPS.items():
            hook = self._hook(op, path, calls)
            self.handles.append(
                root.get_submodule(path).register_forward_hook(hook, with_kwargs=True)
            )

    def _hook(self, op: str, path: str, calls: tuple[int, ...]):
        def hook(module, args, kwargs, output):
            call = self.calls.get(op, 0)
            self.calls[op] = call + 1
            if call not in calls:
                return
            bound = inspect.signature(module.forward).bind(*args, **kwargs)
            extra = bound.arguments.pop("kwargs", {})
            inputs = dict(bound.arguments)
            inputs.update(
                {k: v for k, v in extra.items() if isinstance(v, torch.Tensor)}
            )
            outputs = list(output) if isinstance(output, tuple) else [output]
            key = f"{op}.{call}"
            if "attention_mask" in inputs and key not in KEEP_WEIGHTS:
                outputs[1] = None
            self.index[key] = {
                "module": path,
                "class": type(module).__name__,
                "inputs": {k: self._put(f"{key}.in.{k}", v) for k, v in inputs.items()},
                "outputs": [
                    self._put(f"{key}.out.{i}", v) for i, v in enumerate(outputs)
                ],
            }

        return hook

    def _put(self, key: str, value):
        """A tensor's key in ops.safetensors, or the plain value in a list."""
        if not isinstance(value, torch.Tensor):
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise TypeError(f"{key}: {type(value).__name__}")
            return [value]
        value = value.detach().clone().contiguous()
        if key.endswith((".in.attention_mask", ".in.conv_mask")):
            # HF passes an additive mask; the fixtures keep "may attend".
            value = value == 0 if value.is_floating_point() else value.bool()
        # Equal tensors (one module's output is another's input) are stored once.
        content = _sha256(
            f"{value.dtype}{tuple(value.shape)}".encode() + value.numpy().tobytes()
        )
        key = self.by_content.setdefault(content, key)
        self.tensors.setdefault(key, value)
        return key

    def close(self):
        for handle in self.handles:
            handle.remove()
        missing = [
            f"{op}.{c}"
            for op, (_, calls) in OPS.items()
            for c in calls
            if f"{op}.{c}" not in self.index
        ]
        assert not missing, missing


def reference_problem() -> str | None:
    """Why the importable transformers is not the reference tree, or None."""
    import transformers

    if not hasattr(transformers, "Inkling2LoopForConditionalGeneration"):
        return "transformers has no Inkling 2 classes"
    tree = pathlib.Path(transformers.__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            run = subprocess.run(
                ["git", "-C", str(tree), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return run.stdout.strip() if run.returncode == 0 else None

    # An installed package can sit inside some other repository.
    top = git("rev-parse", "--show-toplevel")
    if top is None or pathlib.Path(top).resolve() != tree:
        return f"transformers at {tree} is not a git checkout"
    head = git("rev-parse", "HEAD")
    if head != TRANSFORMERS_COMMIT:
        return f"transformers at {tree} is at {head}, not {TRANSFORMERS_COMMIT}"
    if git("status", "--porcelain", "--", "src/transformers") != "":
        return f"transformers at {tree} has uncommitted changes"
    return None


def load_reference(directory: pathlib.Path):
    from transformers import Inkling2LoopForConditionalGeneration

    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        directory, dtype=torch.float32, output_loading_info=True
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not info.get(key), (key, info[key])
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model.eval()


class Visits:
    """Hooks on the decoder layers and routers, across every forward."""

    def __init__(self, text_model: torch.nn.Module):
        self.states: list[torch.Tensor] = []
        self.layers: list[int] = []
        self.router_logits: list[torch.Tensor] = []
        self.router_indices: list[torch.Tensor] = []
        self.min_router_gap = float("inf")
        for i, layer in enumerate(text_model.layers):
            layer.register_forward_hook(lambda m, a, out, i=i: self._layer(i, out))
            layer.mlp.gate.register_forward_hook(self._router)

    def _layer(self, i: int, out: torch.Tensor) -> None:
        self.layers.append(i)
        self.states.append(out.detach().clone())

    def _router(self, gate, args, out) -> None:
        logits = out[0]
        self.router_logits.append(logits.detach().clone())
        self.router_indices.append(out[2].detach().clone())
        # The reference ranks sigmoid(logits) + bias; this is the distance
        # between the last expert chosen and the first one left out.
        ranked = (
            (torch.sigmoid(logits) + gate.e_score_correction_bias)
            .sort(-1, descending=True)
            .values
        )
        gap = (ranked[:, gate.top_k - 1] - ranked[:, gate.top_k]).min().item()
        self.min_router_gap = min(self.min_router_gap, gap)

    def take(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """What one forward left: states, router logits and choices by visit."""
        taken = (
            torch.stack(self.states),
            torch.stack(self.router_logits),
            torch.stack(self.router_indices),
            self.layers,
        )
        self.clear()
        return taken

    def clear(self) -> None:
        self.states, self.layers = [], []
        self.router_logits, self.router_indices = [], []


def run_cases(model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    g = torch.Generator().manual_seed(DATA_SEED)
    cases: dict[str, torch.Tensor] = {}
    report: dict = {}
    text_model = model.model.language_model
    visits = Visits(text_model)

    # One-shot prefill past the window (8), the rel extent (8) and n_floor (16).
    ids = torch.randint(2, IMAGE_TOKEN, (1, 20), generator=g)
    recorder = Recorder(model)
    out = model(input_ids=ids, use_cache=False)
    recorder.close()
    states, routers, chosen, layers = visits.take()
    again = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    visits.clear()
    hidden = torch.stack(again.hidden_states)
    assert torch.equal(again.logits, out.logits)
    # output_hidden_states is the normed embedding, visits 0-46, then the
    # final normed state in place of visit 47; the hooks see all 48.
    assert hidden.shape[0] == 49 and torch.equal(hidden[1:48], states[:47])
    assert not torch.equal(hidden[48], states[47])
    loop = list(range(6, 18))
    assert layers == list(range(6)) + loop * 3 + list(range(18, 24))
    report["visit_layers"] = layers
    cases["text.input_ids"] = ids
    cases["text.inputs_embeds"] = hidden[0, 0]
    cases["text.visit_states"] = states[:, 0]
    cases["text.router_logits"] = routers
    cases["text.final_hidden_state"] = hidden[48, 0]
    cases["text.logits"] = out.logits[0]

    # The same ids through HF's cache: prefill 9, then one token at a time,
    # and again as chunks of 7, 6 and 7.
    def cached(sizes):
        cache, logits, start = None, [], 0
        for size in sizes:
            step = model(
                input_ids=ids[:, start : start + size], cache=cache, use_cache=True
            )
            cache, start = step.cache, start + size
            logits.append(step.logits[0])
        return torch.cat(logits)

    cases["text.logits_cached_decode"] = cached([9] + [1] * 11)
    cases["text.logits_chunked_prefill"] = cached([7, 6, 7])
    for name in ("cached_decode", "chunked_prefill"):
        diff = (cases[f"text.logits_{name}"] - cases["text.logits"]).abs().max().item()
        report[f"max_abs_logit_diff_{name}_vs_prefill"] = diff
        assert diff < 1e-5, (name, diff)

    # Greedy continuation through the cache.
    prompt = torch.randint(2, IMAGE_TOKEN, (1, 6), generator=g)
    step = model(input_ids=prompt, use_cache=True)
    generated, margins = [], []
    for _ in range(20):
        top = step.logits[0, -1].topk(2)
        generated.append(top.indices[0].item())
        margins.append((top.values[0] - top.values[1]).item())
        step = model(input_ids=top.indices[:1][None], cache=step.cache, use_cache=True)
    cases["greedy.prompt_ids"] = prompt
    cases["greedy.output_ids"] = torch.tensor([generated])
    report["greedy_min_top2_logit_margin"] = min(margins)
    assert min(margins) > 1e-2, margins
    visits.clear()

    # Two images (2x3 and 1x2 tokens) through the tower, then text + image.
    tile = VIS["patch"] * VIS["merge"]
    grid = torch.tensor([[1, 2, 3], [1, 1, 2]])
    pixels = torch.rand(8, tile, tile, 3, generator=g) * 2 - 1
    features = model.get_image_features(pixels, grid, return_dict=True)
    img = [IMAGE_TOKEN]
    mm_ids = torch.tensor([[5, 9] + img * 6 + [14, 3] + img * 2 + [22, 8, 40]])
    out = model(
        input_ids=mm_ids, pixel_values=pixels, image_grid_thw=grid, use_cache=False
    )
    embeds = text_model.embed_tokens(mm_ids)
    embeds[0, mm_ids[0] == IMAGE_TOKEN] = torch.cat(features.pooler_output)
    cases["image.pixel_values"] = pixels
    cases["image.image_grid_thw"] = grid
    cases["image.vision_last_hidden_state"] = features.last_hidden_state
    cases["image.features"] = torch.cat(features.pooler_output)
    cases["image.input_ids"] = mm_ids
    cases["image.inputs_embeds"] = embeds[0]
    cases["image.visit_states_last_token"] = visits.take()[0][:, 0, -1].contiguous()
    cases["image.logits"] = out.logits[0]
    from_embeds = model(inputs_embeds=embeds, use_cache=False).logits[0]
    assert torch.equal(from_embeds, out.logits[0])

    report["min_router_gap"] = visits.min_router_gap
    assert visits.min_router_gap > 1e-4, visits.min_router_gap
    report["coverage"] = coverage(model, recorder, cases, chosen)
    return recorder.tensors, cases, {**report, "ops": recorder.index}


def coverage(
    model, recorder: Recorder, cases: dict[str, torch.Tensor], chosen: torch.Tensor
) -> dict:
    """Asserts the tiny run reaches both sides of every branch it is for.

    `chosen` is the routers' top_k_index over the visits of the text case.
    """
    cfg = model.config.text_config
    layers = model.model.language_model.layers
    stats = {}

    def arg(op: str, name: str) -> torch.Tensor:
        return recorder.tensors[recorder.index[op]["inputs"][name]]

    def clipped(op: str, module, threshold: float) -> float:
        x = arg(f"{op}.0", "hidden_states")
        scale = getattr(module, "scale", None)
        norm = (x if scale is None else x * scale).norm(dim=-1)
        return (norm > threshold).float().mean().item()

    attn, full = layers[1].self_attn, layers[5].self_attn
    qk, rel = cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold
    stats["clipped_fraction"] = {
        "l2_clip": clipped("l2_clip", attn.k_proj.norm, qk),
        "scaled_l2_clip": clipped("scaled_l2_clip", attn.q_proj.norm, qk),
        "log_scaled_l2_clip": clipped("log_scaled_l2_clip", full.q_proj.norm, qk),
        "rel_l2_clip_sliding": clipped(
            "rel_l2_clip_sliding", attn.relative_position_bias.norm, rel
        ),
        "rel_l2_clip_full": clipped(
            "rel_l2_clip_full", full.relative_position_bias.norm, rel
        ),
    }
    assert all(0.1 < v < 0.9 for v in stats["clipped_fraction"].values()), stats
    positions = arg("attn_full.0", "position_ids")
    stats["positions_past_n_floor"] = int(
        (positions + 1 > cfg.log_scaling_n_floor).sum()
    )
    assert stats["positions_past_n_floor"] >= 4
    for op, window in (("attn_sliding", cfg.sliding_window_size), ("attn_full", None)):
        allowed = arg(f"{op}.0", "attention_mask")[0, 0]
        distance = arg(f"{op}.0", "relative_distance")
        expect = (
            (distance >= 0) if window is None else (distance >= 0) & (distance < window)
        )
        assert torch.equal(allowed, expect), op
    stats["keys_past_window"] = int(
        (arg("attn_sliding.0", "relative_distance") >= cfg.sliding_window_size).sum()
    )
    stats["keys_past_rel_extent"] = int(
        (arg("attn_full.0", "relative_distance") >= cfg.rel_extent).sum()
    )
    assert stats["keys_past_window"] > 0 and stats["keys_past_rel_extent"] > 0
    # The reference's own choices (top-k of sigmoid(logits) + bias): layer 1's
    # call in ops.safetensors, then every visit of the text case.
    stats["experts_chosen_in_router_op"] = int(
        arg("experts_nvfp4.0", "top_k_index").unique().numel()
    )
    assert stats["experts_chosen_in_router_op"] > cfg.num_experts_per_tok
    stats["experts_chosen_in_text_case"] = int(chosen.unique().numel())
    assert stats["experts_chosen_in_text_case"] == EXPERTS
    stats["max_abs_swiglu_gate"] = arg("swiglu.0", "gate").abs().max().item()
    stats["max_abs_swiglu_up"] = arg("swiglu.0", "up").abs().max().item()
    assert stats["max_abs_swiglu_gate"] > 1 and stats["max_abs_swiglu_up"] > 1
    stats["max_abs_logit"] = cases["text.logits"].abs().max().item()
    assert stats["max_abs_logit"] > 1
    stats["max_abs_visit_state"] = cases["text.visit_states"].abs().max().item()
    return stats


def generate(out: pathlib.Path) -> dict:
    if problem := reference_problem():
        raise RuntimeError(problem)
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    weights = native_weights(torch.Generator().manual_seed(WEIGHT_SEED))
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (out / "hf_quant_config.json").write_text(json.dumps(quant_config(weights)) + "\n")
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})

    with torch.no_grad():
        model = load_reference(out)
        ops, cases, report = run_cases(model)
        state = {k: _tensor_digest(v) for k, v in model.state_dict().items()}
    save_file(ops, out / "ops.safetensors")
    save_file(cases, out / "cases.safetensors")

    import transformers

    manifest = {
        "generator": "sm70/tests/models/fixtures/inkling2_tiny/gen_fixture.py",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "weight_seed": WEIGHT_SEED,
        "data_seed": DATA_SEED,
        "dtype": "float32",
        "files": {name: _sha256((out / name).read_bytes()) for name in FILES},
        **report,
        "hf_state_dict": state,
    }
    (out / "manifest.json").write_text(_dump(manifest))
    return manifest


def _dump(manifest: dict) -> str:
    """JSON with one line per entry of the two long tables."""
    lines = []
    for key, value in manifest.items():
        if key in ("ops", "hf_state_dict"):
            rows = ",\n".join(
                f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in value.items()
            )
            lines.append(f" {json.dumps(key)}: {{\n{rows}\n }}")
        else:
            lines.append(f" {json.dumps(key)}: {json.dumps(value)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _tensor_differences(want: pathlib.Path, got: pathlib.Path) -> list[str]:
    """Tensors of two safetensors files that differ: floats by more than
    FLOAT_TOLERANCE, anything else at all."""
    a, b = load_file(want), load_file(got)
    problems = [f"{want.name}: only in one file: {k}" for k in sorted(set(a) ^ set(b))]
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if (x.dtype, x.shape) != (y.dtype, y.shape):
            problems.append(f"{want.name}: {key}: dtype or shape")
        elif not x.is_floating_point():
            if not torch.equal(x, y):
                problems.append(f"{want.name}: {key}: differs")
        elif x.numel():
            diff = (x.double() - y.double()).abs().max().item()
            if not diff <= FLOAT_TOLERANCE:
                problems.append(f"{want.name}: {key}: max abs diff {diff:.3g}")
    return problems


def check(out: pathlib.Path) -> tuple[int, list[str]]:
    """Regenerates elsewhere and compares with `out`: an exit code and a report.

    Byte-identical files are required on the torch version and CPU capability
    of manifest.json. Elsewhere the files written before the forward must be
    identical and the recorded tensors equal within FLOAT_TOLERANCE.
    """
    want = json.loads((out / "manifest.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        new = pathlib.Path(tmp)
        got = generate(new)
        same = {name: got["files"][name] == want["files"][name] for name in FILES}
        report = [
            f"{'same     ' if same[name] else 'DIFFERENT'} {name} {got['files'][name]}"
            for name in FILES
        ]
        if all(same.values()):
            return SAME, report
        env = ("torch_version", "cpu_capability")
        if all(got[key] == want.get(key) for key in env):
            return DIFFERENT, report
        report.append(
            "not the manifest's build: "
            + ", ".join(f"{key} {got[key]} (manifest: {want.get(key)})" for key in env)
        )
        if not (same["config.json"] and same["hf_quant_config.json"]):
            return DIFFERENT, report
        if not same["model.safetensors"]:
            report.append(
                "not reproducible here: this build draws other random weights "
                "(model.safetensors), so no output can be compared"
            )
            return NOT_REPRODUCIBLE, report
        problems = [
            f"manifest.json: {key} differs"
            for key in ("visit_layers", "ops")
            if got[key] != want[key]
        ]
        for name in ("ops.safetensors", "cases.safetensors"):
            problems += _tensor_differences(out / name, new / name)
    if problems:
        return DIFFERENT, report + problems
    report.append(
        f"equal within {FLOAT_TOLERANCE:g}: floats of ops.safetensors and "
        "cases.safetensors; every other tensor is identical"
    )
    return SAME, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=HERE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate elsewhere and compare with --out",
    )
    args = parser.parse_args()
    if problem := reference_problem():
        print(problem, file=sys.stderr)
        return NOT_REFERENCE
    if not args.check:
        generate(args.out)
        return SAME
    code, report = check(args.out)
    print("\n".join(report))
    return code


if __name__ == "__main__":
    sys.exit(main())
