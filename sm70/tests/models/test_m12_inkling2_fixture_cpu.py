# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 tiny fixtures (fixtures/inkling2_tiny/).

CPU only, and without the Hugging Face Inkling 2 classes: the committed files
against manifest.json, the tiny checkpoint against the released checkpoint's
tensor names and quant groups (data/inkling2_mini/), its packed experts
against the weights the reference loaded, and the reference's own cached
decode against its prefill. Regenerating needs the reference tree at the
manifest's commit; that test skips without it.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch

from vllm_sm70.models.inkling2_sm70 import sm70_configs

HERE = pathlib.Path(__file__).resolve().parent
TINY = HERE / "fixtures" / "inkling2_tiny"
MINI = HERE / "data" / "inkling2_mini"

_spec = importlib.util.spec_from_file_location("gen_fixture", TINY / "gen_fixture.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((TINY / "manifest.json").read_text())


@pytest.fixture(scope="module")
def native():
    return safetensors_torch.load_file(TINY / "model.safetensors")


@pytest.fixture(scope="module")
def ops():
    return safetensors_torch.load_file(TINY / "ops.safetensors")


@pytest.fixture(scope="module")
def cases():
    return safetensors_torch.load_file(TINY / "cases.safetensors")


def _pattern(name: str) -> str:
    return re.sub(r"\d+", "N", name)


def _groups(path: pathlib.Path) -> dict[str, str]:
    """Tensor name -> the group of hf_quant_config.json it is listed in."""
    out = {}
    for group in json.loads(path.read_text())["model"]["parameter_groups"]:
        label = f"{group['compute_dtype']}>{group.get('fwd_output_dtype', group['output_dtype'])}"
        out.update(dict.fromkeys(group["tensors"], label))
    return out


def test_files_match_manifest(manifest):
    assert (
        manifest["generator"]
        == "sm70/tests/models/fixtures/inkling2_tiny/gen_fixture.py"
    )
    assert manifest["transformers_commit"] == gen.TRANSFORMERS_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["transformers_commit"])
    assert manifest["transformers_commit"] in (TINY / "PROVENANCE.md").read_text()
    assert (manifest["weight_seed"], manifest["data_seed"]) == (
        gen.WEIGHT_SEED,
        gen.DATA_SEED,
    )
    assert manifest["torch_version"] and manifest["cpu_capability"]
    assert set(manifest["files"]) == set(gen.FILES)
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((TINY / name).read_bytes()).hexdigest() == digest, name
    assert json.loads((TINY / "config.json").read_text()) == gen.CONFIG
    committed = [p for p in TINY.iterdir() if p.is_file()]
    assert sum(p.stat().st_size for p in committed) < 5 * 1024 * 1024


def test_native_names_and_groups_match_released_checkpoint(native):
    released = set(
        json.loads((MINI / "model.safetensors.index.json").read_text())["weight_map"]
    )
    llm = {n for n in released if n.startswith("model.llm.")}
    assert {n for n in native if n.startswith("model.llm.")} == llm
    # The tiny tower has fewer layers; its names are the released ones.
    visual = {n for n in native if n.startswith("model.visual.")}
    assert visual < released
    assert {_pattern(n) for n in visual} == {_pattern(n) for n in released - llm}

    tiny_groups, released_groups = (
        _groups(TINY / "hf_quant_config.json"),
        _groups(MINI / "hf_quant_config.json"),
    )
    assert set(tiny_groups) == {
        n for n in native if not re.search(r"_weight_(scale|amax|outer)", n)
    }
    assert tiny_groups == {n: released_groups[n] for n in tiny_groups}
    # Cast groups store the tensor in the group's dtype.
    for name, group in tiny_groups.items():
        if group.startswith(("float32", "bfloat16")):
            assert str(native[name].dtype) == "torch." + group.split(">")[0], name


def test_packed_expert_layouts(native):
    e, h, i = gen.EXPERTS, gen.H, gen.INTER
    for layer in range(24):
        for name, rows, cols in (("w13", 2 * i, h), ("w2", h, i)):
            p = f"model.llm.layers.{layer}.mlp.experts.{name}_weight"
            weight, scale = native[p], native[p + "_scale"]
            if layer in gen.NVFP4_LAYERS:
                assert (weight.dtype, tuple(weight.shape)) == (
                    torch.uint8,
                    (e, rows, cols // 2),
                )
                assert (scale.dtype, tuple(scale.shape)) == (
                    torch.float8_e4m3fn,
                    (e, rows, cols // 16),
                )
                assert (
                    native[p + "_amax"].dtype,
                    tuple(native[p + "_amax"].shape),
                ) == (torch.float32, (e,))
                assert tuple(native[p + "_outer_scale_denominator"].shape) == (1,)
            else:
                assert (weight.dtype, tuple(weight.shape)) == (
                    torch.float8_e4m3fn,
                    (e, rows, cols),
                )
                assert (scale.dtype, tuple(scale.shape)) == (
                    torch.uint8,
                    (e, rows, cols // 32),
                )
                assert p + "_amax" not in native


def _dequantize(native: dict, prefix: str) -> torch.Tensor:
    """fp32 expert weights from the packed tensors, as the reference loads them."""
    weight, scale = native[prefix], native[prefix + "_scale"].float()
    if prefix + "_amax" not in native:
        return (
            weight.float().unflatten(-1, (-1, 32)) * torch.exp2(scale - 127)[..., None]
        ).flatten(-2)
    lut = torch.tensor(gen.E2M1)
    codes = weight.long()
    values = torch.stack((lut[codes & 0xF], lut[codes >> 4]), dim=-1).flatten(-2)
    values = values.unflatten(-1, (-1, 16)) * scale[..., None]
    factor = native[prefix + "_amax"] / native[prefix + "_outer_scale_denominator"]
    return (values * factor[:, None, None, None]).flatten(-2)


def _assert_digest(manifest: dict, hf_name: str, tensor: torch.Tensor) -> None:
    shape, total, magnitude, ramp = manifest["hf_state_dict"][hf_name]
    got = gen._tensor_digest(tensor.float())
    assert got[0] == shape, hf_name
    assert got[1:] == pytest.approx([total, magnitude, ramp], rel=1e-9, abs=1e-9), (
        hf_name
    )


def test_native_checkpoint_loads_to_the_reference_weights(manifest, native):
    llm, hf = "model.llm.", "model.language_model."
    for layer in range(24):
        e = f"layers.{layer}.mlp.experts."
        w13 = _dequantize(native, f"{llm}{e}w13_weight")
        # Checkpoint rows alternate gate, up; the reference keeps gate rows first.
        _assert_digest(
            manifest,
            f"{hf}{e}gate_up_proj",
            torch.cat((w13[:, 0::2], w13[:, 1::2]), dim=1),
        )
        _assert_digest(
            manifest, f"{hf}{e}down_proj", _dequantize(native, f"{llm}{e}w2_weight")
        )
        with pytest.raises(AssertionError):
            _assert_digest(manifest, f"{hf}{e}gate_up_proj", w13)
    _assert_digest(
        manifest, "lm_head.weight", native[llm + "unembed.weight"][: gen.UNPADDED]
    )
    assert native[llm + "unembed.weight"][gen.UNPADDED :].abs().sum() > 0
    for layer in range(24):
        _assert_digest(
            manifest,
            f"{hf}layers.{layer}.mlp.gate.e_score_correction_bias",
            native[f"{llm}layers.{layer}.mlp.gate.bias"],
        )
    # attn_queries.{i} belongs to layer i + 1; layer 0 has none.
    for i in range(23):
        _assert_digest(
            manifest,
            f"{hf}layers.{i + 1}.attention_input.query",
            native[f"{llm}attn_res.attn_queries.{i}"],
        )
    assert f"{hf}layers.0.attention_input.query" not in manifest["hf_state_dict"]
    # Rows 2l, 2l + 1 of the two message tensors are (query, mlp) and (key, value) of loop layer l.
    q_mlp = native[llm + "middle_cycle_cross_pass_queries"].unflatten(0, (-1, 2))
    k_v = native[llm + "middle_cycle_cross_pass_kv_queries"].unflatten(0, (-1, 2))
    _assert_digest(
        manifest,
        hf + "message_input.query",
        torch.cat((q_mlp[:, :1], k_v, q_mlp[:, 1:]), dim=1),
    )
    gate = "layers.6.controllers.attn."
    _assert_digest(
        manifest,
        f"{hf}layers.6.q_message_gate.message_gate_proj.weight",
        native[f"{llm}{gate}gain_weight"],
    )
    _assert_digest(
        manifest,
        f"{hf}layers.6.q_message_gate.forget_gate_proj.weight",
        native[f"{llm}{gate}forget_weight"],
    )


def test_config_parses_with_the_plugin_classes(manifest):
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    text = config.get_text_config()
    assert (text.num_hidden_layers, text.num_layer_visits) == (24, 48)
    assert text.loop_core_layers == range(6, 18) and text.num_loop_passes == 3
    assert text.n_layers_per_attn_res_block == 6
    assert (text.hidden_size, text.n_routed_experts) == (64, 8)
    assert (text.sliding_window_size, text.rel_extent, text.log_scaling_n_floor) == (
        8,
        8,
        16,
    )
    order = manifest["visit_layers"]
    assert len(order) == 48
    for layer in range(24):
        assert [
            v for v, visited in enumerate(order) if visited == layer
        ] == text.layer_visit_ids(layer)


def test_ops_index_resolves(manifest, ops):
    index = manifest["ops"]
    assert set(index) == {
        f"{op}.{call}" for op, (_, calls) in gen.OPS.items() for call in calls
    }
    used = set()
    for key, entry in index.items():
        assert entry["module"] == gen.OPS[key.rsplit(".", 1)[0]][0]
        for ref in [*entry["inputs"].values(), *entry["outputs"]]:
            if isinstance(ref, str):
                assert ref in ops, (key, ref)
                used.add(ref)
            else:
                assert len(ref) == 1 and not isinstance(ref[0], (list, dict)), (
                    key,
                    ref,
                )
    assert used == set(ops)
    assert all(
        t.dtype in (torch.float32, torch.int64, torch.bool) for t in ops.values()
    )
    assert all(torch.isfinite(t).all() for t in ops.values() if t.is_floating_point())


def test_ops_cover_the_edges_they_are_for(manifest, native, ops, cases):
    index, text = manifest["ops"], gen.CONFIG["text_config"]

    def arg(op, name):
        return ops[index[op]["inputs"][name]]

    def out(op, i=0):
        return ops[index[op]["outputs"][i]]

    # Masks are stored as "may attend": the window for sliding layers, causal for full.
    for op, window in (
        ("attn_sliding.0", text["sliding_window_size"]),
        ("attn_full.0", None),
    ):
        distance, allowed = arg(op, "relative_distance"), arg(op, "attention_mask")
        assert allowed.dtype == torch.bool and distance.shape == (20, 20)
        expect = (
            distance >= 0 if window is None else (distance >= 0) & (distance < window)
        )
        assert torch.equal(allowed[0, 0], expect)
        assert (distance >= 8).any()
        weights = out(op, 1)
        assert (weights[0][:, ~expect] == 0).all() and (weights.sum(-1) < 1).all()
    assert index["attn_sliding_loop.1"]["inputs"]["cache_offset"] == [18]
    assert index["attn_full_loop.2"]["inputs"]["cache_offset"] == [30]
    assert index["attn_res_message.17"]["inputs"]["loop_layer_idx"] == [5]
    # tau > 1 on the last positions only.
    positions = arg("log_scale.0", "position_ids")[0]
    tau = (out("log_scale.0") / arg("log_scale.0", "hidden_states"))[0].mean((1, 2))
    floor, alpha = text["log_scaling_n_floor"], text["log_scaling_alpha"]
    expect = 1 + alpha * torch.log(((positions + 1) / floor).clamp(min=1))
    assert (
        torch.allclose(tau, expect, atol=1e-5)
        and (tau[:16] == 1).all()
        and (tau[16:] > 1.005).all()
    )
    # Both sides of every clip.
    assert all(0.1 < v < 0.9 for v in manifest["coverage"]["clipped_fraction"].values())
    norm = arg("l2_clip.0", "hidden_states").norm(dim=-1)
    clipped = norm > text["qk_l2_clip_threshold"]
    assert clipped.any() and not clipped.all()
    assert torch.equal(
        out("l2_clip.0")[~clipped], arg("l2_clip.0", "hidden_states")[~clipped]
    )
    # The logit cap acts on the stored logits, which are the text case's.
    capped = out("logit_softcap.0")
    assert torch.equal(capped[0], cases["text.logits"])
    raw = arg("logit_softcap.0", "hidden_states")
    assert torch.allclose(capped, 30 * torch.tanh(raw / 30), atol=1e-6)
    assert (raw - capped).abs().max() > 1e-4
    assert manifest["min_router_gap"] > 1e-4
    # The router ranks sigmoid(logits) + bias, the checkpoint's gate.bias.
    chosen = out("router.0", 2)
    assert index["experts_nvfp4.0"]["inputs"]["top_k_index"] == "router.0.out.2"
    scores = torch.sigmoid(out("router.0")) + native["model.llm.layers.1.mlp.gate.bias"]
    top_k = text["num_experts_per_tok"]
    assert torch.equal(
        chosen.sort(-1).values, scores.topk(top_k).indices.sort(-1).values
    )
    assert not torch.equal(
        chosen.sort(-1).values, out("router.0").topk(top_k).indices.sort(-1).values
    )
    coverage = manifest["coverage"]
    assert coverage["experts_chosen_in_router_op"] == chosen.unique().numel() > top_k
    assert coverage["experts_chosen_in_text_case"] == gen.EXPERTS


def test_reference_cached_decode_equals_prefill(manifest, cases):
    logits = cases["text.logits"]
    assert logits.shape == (20, gen.UNPADDED)
    for name in ("cached_decode", "chunked_prefill"):
        diff = (cases[f"text.logits_{name}"] - logits).abs().max().item()
        assert diff < 1e-5, (name, diff)
        assert diff == manifest[f"max_abs_logit_diff_{name}_vs_prefill"]


def test_case_shapes(cases):
    assert cases["text.input_ids"].shape == (1, 20)
    assert cases["text.visit_states"].shape == (48, 20, gen.H)
    assert cases["text.router_logits"].shape == (48, 20, gen.EXPERTS)
    assert cases["greedy.prompt_ids"].shape == (1, 6) and cases[
        "greedy.output_ids"
    ].shape == (1, 20)
    assert cases["greedy.output_ids"].max() < gen.UNPADDED
    grid, ids = cases["image.image_grid_thw"], cases["image.input_ids"][0]
    tokens = int(grid.prod(-1).sum())
    assert (
        tokens
        == int((ids == gen.IMAGE_TOKEN).sum())
        == cases["image.features"].shape[0]
    )
    tile = gen.VIS["patch"] * gen.VIS["merge"]
    assert cases["image.pixel_values"].shape == (tokens, tile, tile, 3)
    assert torch.equal(
        cases["image.inputs_embeds"][ids == gen.IMAGE_TOKEN], cases["image.features"]
    )
    assert cases["image.logits"].shape == (ids.numel(), gen.UNPADDED)
    assert cases["image.visit_states_last_token"].shape == (48, gen.H)
    assert all(torch.isfinite(t).all() for t in cases.values() if t.is_floating_point())


def test_generator_reproduces_the_committed_files():
    if problem := gen.reference_problem():
        pytest.skip(problem)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    run = subprocess.run(
        [sys.executable, str(TINY / "gen_fixture.py"), "--check"],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )
    if run.returncode == gen.NOT_REPRODUCIBLE:
        pytest.skip(run.stdout.strip().splitlines()[-1])
    assert run.returncode == gen.SAME, run.stdout + run.stderr
