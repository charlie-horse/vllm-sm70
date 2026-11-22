# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 non-expert weight loader (weights.py).

CPU, without the Hugging Face Inkling 2 classes: the tiny checkpoint in
released names is loaded at TP1 and at both ranks of TP2 and compared with the
digests of the weights as the reference loaded them (manifest.json
`hf_state_dict`); the released checkpoint's tensor names and config are test
data (data/inkling2_mini/).
"""

import importlib.util
import json
import pathlib
import re

import pytest

pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch
from torch import nn

from vllm_sm70.models.inkling2_sm70 import sm70_configs, weights

HERE = pathlib.Path(__file__).resolve().parent
TINY = HERE / "fixtures" / "inkling2_tiny"
MINI = HERE / "data" / "inkling2_mini"

_spec = importlib.util.spec_from_file_location("gen_fixture", TINY / "gen_fixture.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
HF_DIGESTS = json.loads((TINY / "manifest.json").read_text())["hf_state_dict"]
HF_TEXT = {
    name: digest
    for name, digest in HF_DIGESTS.items()
    if "vision_tower" not in name and ".mlp.experts." not in name
}
NUM_EXPERT_TENSORS = 20 * 8 + 4 * 4

# The parameters transformers keeps in float32 (HF_FP32_MODULES), by the names
# of this loader.
FP32_PARAMS = re.compile(
    r"model\.(attn_res\..*|message_queries"
    r"|layers\.\d+\.(mlp\.gate\.(weight|bias)|(attn|mlp)_sconv\.weight"
    r"|attn\.[kv]_sconv\.weight|controllers\.\w+\.(gain|forget)_weight))"
)


def _config(path: pathlib.Path):
    return sm70_configs.Inkling2LoopSm70MMConfig(**json.loads(path.read_text()))


@pytest.fixture(scope="module")
def tiny_config():
    return _config(TINY / "config.json")


@pytest.fixture(scope="module")
def mini_config():
    return _config(MINI / "config.json")


def _convert(config, tp_size, dtype=torch.float32, native=NATIVE):
    maps = [
        weights.build_weight_map(config, tp_size=tp_size, tp_rank=rank, dtype=dtype)
        for rank in range(tp_size)
    ]
    return maps, [m.convert(native.items()) for m in maps]


def _hf_tensors(maps, converted) -> dict[str, torch.Tensor]:
    """Reference-named tensors from every rank's parameters, shards re-joined."""
    by_hf: dict[str, list[weights.Route]] = {}
    for route in maps[0].routes.values():
        by_hf.setdefault(route.hf_name, []).append(route)
    out = {}
    for hf_name, routes in by_hf.items():
        route = routes[0]
        ranks = [c.params[route.param] for c in converted]
        if len(routes) > 1:
            assert all(torch.equal(ranks[0], r) for r in ranks), hf_name
            out[hf_name] = ranks[0]
            continue
        if route.index is not None:
            ranks = [r[route.index] for r in ranks]
        if route.tp_dim is None:
            assert all(torch.equal(ranks[0], r) for r in ranks), hf_name
            joined = ranks[0]
        else:
            joined = torch.cat(ranks, dim=route.tp_dim)
        shape = list(route.source_shape)
        if route.rows is not None:
            shape[0] = route.rows
        out[hf_name] = joined.reshape(shape)
    return out


@pytest.mark.parametrize("tp_size", [1, 2])
def test_tiny_checkpoint_equals_the_reference_weights(tiny_config, tp_size):
    maps, converted = _convert(tiny_config, tp_size)
    got = _hf_tensors(maps, converted)
    assert got.keys() == HF_TEXT.keys()
    for name, want in HF_TEXT.items():
        digest = gen._tensor_digest(got[name])
        assert digest[0] == want[0], name
        assert digest[1:] == pytest.approx(want[1:], rel=1e-12, abs=1e-12), name
    for result in converted:
        assert len(result.experts) == NUM_EXPERT_TENSORS and not result.visual
        assert all(".mlp.experts." in name for name, _ in result.experts)


def test_tp2_ranks_hold_disjoint_head_shards(tiny_config):
    maps, converted = _convert(tiny_config, 2)
    (single,), (whole,) = _convert(tiny_config, 1)
    assert maps[0].params == maps[1].params
    sharded = {r.param for r in single.routes.values() if r.tp_dim is not None}
    for name, spec in single.params.items():
        halves = [c.params[name] for c in converted]
        if name in sharded:
            assert maps[0].params[name].shape != spec.shape, name
            assert halves[0].numel() * 2 == whole.params[name].numel(), name
            assert not torch.equal(*halves), name
        else:
            assert torch.equal(halves[0], whole.params[name]), name
            assert torch.equal(halves[1], whole.params[name]), name
    # Query-side rows first, then K and V, per rank.
    full = weights.qkvr_sizes(tiny_config, is_local=False)
    assert full == (24, 24, 24, 6, 8, 8)
    assert weights.qkvr_sizes(tiny_config, False, 2) == tuple(n // 2 for n in full)
    qkvr = converted[1].params["model.layers.5.attn.qkvr.weight"]
    assert qkvr.shape == (sum(full) // 2, 64)
    llm = "model.llm.layers.5.attn."
    assert torch.equal(qkvr[:12], NATIVE[llm + "wq_du.weight"][12:].float())
    assert torch.equal(qkvr[-4:], NATIVE[llm + "wv_dv.weight"][4:].float())
    wo = converted[1].params["model.layers.5.attn.wo_ud.weight"]
    assert torch.equal(wo, NATIVE[llm + "wo_ud.weight"][:, 12:].float())
    static = converted[1].params["model.layers.5.attn.k_sconv.static_w"]
    assert torch.equal(static, NATIVE[llm + "k_sconv.static_w"][:, 4:].float())


def test_renames_and_merges(tiny_config):
    (wmap,), (result,) = _convert(tiny_config, 1)
    params, llm = result.params, "model.llm."
    assert params["lm_head.weight"].shape == (72, 64)
    assert torch.equal(
        params["lm_head.weight"], NATIVE[llm + "unembed.weight"][:72].float()
    )
    assert params["model.embed_tokens.weight"].shape == (80, 64)
    # attn_queries.{i} is layer i + 1's; full layers hold q, k, v rows.
    route = wmap.routes[llm + "attn_res.attn_queries.4"]
    assert route.hf_name == "model.language_model.layers.5.attention_input.query"
    assert params["model.attn_res.attn_queries.4"].shape == (3, 64)
    assert params["model.attn_res.attn_queries.3"].shape == (1, 64)
    assert "model.attn_res.attn_queries.23" not in params
    queries = NATIVE[llm + "middle_cycle_cross_pass_queries"]
    kv_queries = NATIVE[llm + "middle_cycle_cross_pass_kv_queries"]
    message = params["model.message_queries"]
    assert message.shape == (12, 4, 64)
    for layer in range(12):
        rows = [
            queries[2 * layer],
            *kv_queries[2 * layer :][:2],
            queries[2 * layer + 1],
        ]
        assert torch.equal(message[layer], torch.stack(rows))
    norm = params["model.layers.11.attn_norm.weight"]
    for i, stream in enumerate("qkv"):
        native = NATIVE[f"{llm}layers.11.{stream}_input_norm.weight"]
        assert torch.equal(norm[i], native.float())
    assert params["model.layers.10.attn_norm.weight"].shape == (1, 64)
    assert params["model.layers.6.controllers.attn.gain_weight"].shape == (64, 2)
    assert "model.layers.5.controllers.attn.gain_weight" not in params


def test_dtype_table(tiny_config):
    assert weights.HF_FP32_MODULES == (
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
    (wmap,), (half,) = _convert(tiny_config, 1, torch.float16)
    _, (single,) = _convert(tiny_config, 1)
    fp32 = {name for name, spec in wmap.params.items() if spec.dtype == torch.float32}
    assert fp32 == {name for name in wmap.params if FP32_PARAMS.fullmatch(name)}
    assert len(fp32) == 23 + 24 + 1 + 1 + 24 * 4 + 20 * 2 + 12 * 8
    for name, spec in wmap.params.items():
        tensor = half.params[name]
        assert tensor.dtype == spec.dtype, name
        assert spec.dtype in (torch.float16, torch.float32), name
        # bf16 -> fp16 is exact for values that are fp16 normals.
        want = single.params[name]
        assert torch.equal(tensor, want.to(spec.dtype)), name
        normal = want.abs() >= 2.0**-14
        assert torch.equal(tensor.float()[normal], want[normal]), name
        assert tensor.isfinite().all(), name
    # Norm weights stay in the model dtype.
    for name in ("model.norm.weight", "model.layers.5.attn_norm.weight"):
        assert wmap.params[name].dtype == torch.float16


def test_dtype_table_matches_the_reference_source():
    spec = None
    try:
        spec = importlib.util.find_spec("transformers.models.inkling2_loop")
    except ModuleNotFoundError:
        pass
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("transformers has no inkling2_loop model")
    source = (
        pathlib.Path(spec.submodule_search_locations[0]) / "modeling_inkling2_loop.py"
    ).read_text()
    body = re.search(r"_keep_in_fp32_modules_strict = \[(.*?)\]", source, re.DOTALL)
    assert tuple(re.findall(r'"([^"]+)"', body.group(1))) == weights.HF_FP32_MODULES


@pytest.mark.parametrize("tp_size", [1, 2])
def test_released_names_map_to_one_parameter_or_a_skip(mini_config, tp_size):
    index = json.loads((MINI / "model.safetensors.index.json").read_text())
    names = list(index["weight_map"])
    assert len(names) == 1244
    wmap = weights.build_weight_map(mini_config, tp_size=tp_size, tp_rank=tp_size - 1)
    owners: dict[str, int] = {}
    routed = set()
    for name in names:
        dest = wmap.classify(name)
        if isinstance(dest, weights.Skip):
            owners[dest.owner] = owners.get(dest.owner, 0) + 1
            assert name.startswith("model.visual.") == (dest.owner == "visual")
            assert (".mlp.experts." in name) == (dest.owner == "experts")
        else:
            assert dest.param in wmap.params
            routed.add(name)
    assert routed == wmap.routes.keys()
    assert owners["experts"] == NUM_EXPERT_TENSORS
    assert owners["visual"] == 1244 - NUM_EXPERT_TENSORS - len(routed)
    assert {r.param for r in wmap.routes.values()} == wmap.params.keys()

    shapes = {name: spec.shape for name, spec in wmap.params.items()}
    assert shapes["lm_head.weight"] == (200058, 2304)
    assert shapes["model.embed_tokens.weight"] == (201024, 2304)
    assert shapes["model.message_queries"] == (12, 4, 2304)
    assert shapes["model.attn_res.attn_queries.22"] == (3, 2304)
    # wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv of the checkpoint.
    full = (3072, 384, 3072, 24, 1024, 1024)
    for layer in (0, 5):
        assert weights.qkvr_sizes(mini_config, layer != 5) == full
        prefix = f"model.layers.{layer}.attn."
        assert shapes[prefix + "qkvr.weight"] == (sum(full) // tp_size, 2304)
        assert shapes[prefix + "wo_ud.weight"] == (2304, 3072 // tp_size)
        assert shapes[prefix + "q_scale"] == (24 // tp_size, 128)
        assert shapes[prefix + "r_scale"] == (24 // tp_size, 16)
        assert shapes[prefix + "rel_logits_proj"] == (16, 1024)
    assert shapes["model.layers.0.attn.k_sconv.weight"] == (1024 // tp_size, 1, 4)
    assert shapes["model.layers.5.attn.k_sconv.static_w"] == (4, 1024 // tp_size)
    assert shapes["model.layers.5.attn.v_sconv.weight_proj"] == (32 // tp_size, 2304)
    assert shapes["model.layers.6.controllers.mlp.down_weight"] == (64, 4608)
    assert wmap.routes["model.llm.layers.6.attn_sconv.weight"].source_shape == (
        2304,
        1,
        4,
    )


def test_rejects_bad_checkpoints(tiny_config):
    wmap = weights.build_weight_map(tiny_config, tp_size=1, tp_rank=0)
    gate = "model.llm.layers.3.mlp.gate.weight"

    missing = {k: v for k, v in NATIVE.items() if k != gate}
    with pytest.raises(
        ValueError,
        match=r"missing checkpoint tensors: \['model.llm.layers.3.mlp.gate.weight'\]",
    ):
        wmap.convert(missing.items())
    for extra in (
        "model.llm.layers.24.mlp.gate.weight",
        "model.llm.layers.24.mlp.experts.w2_weight",
        "model.llm.layers.0.mlp.experts.w3_weight",
        "model.llm.layers.5.attn_norm.weight",
        "mtp.norm.weight",
    ):
        with pytest.raises(ValueError, match="unexpected checkpoint tensor"):
            wmap.convert({**NATIVE, extra: NATIVE[gate]}.items())
    with pytest.raises(ValueError, match="duplicate checkpoint tensor"):
        wmap.convert([*NATIVE.items(), (gate, NATIVE[gate])])
    with pytest.raises(ValueError, match=r"checkpoint shape \(8, 32\) != \(8, 64\)"):
        wmap.convert({**NATIVE, gate: NATIVE[gate][:, :32]}.items())
    with pytest.raises(ValueError, match="torch.uint8 is not a float weight dtype"):
        wmap.convert({**NATIVE, gate: NATIVE[gate].to(torch.uint8)}.items())

    with pytest.raises(ValueError, match="don't split over 4 ranks"):
        weights.build_weight_map(tiny_config, tp_size=4, tp_rank=0)
    with pytest.raises(ValueError, match="TP rank 2"):
        weights.build_weight_map(tiny_config, tp_size=2, tp_rank=2)


class _Bag(nn.Module):
    """A module whose parameters carry the given dotted names."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.tensors = {
            k: nn.Parameter(v, requires_grad=False) for k, v in tensors.items()
        }

    def named_parameters(self, *args, **kwargs):
        return iter(self.tensors.items())


def _bag(wmap, **extra) -> _Bag:
    tensors = {
        name: torch.full(spec.shape, float("nan"), dtype=spec.dtype)
        for name, spec in wmap.params.items()
    }
    return _Bag({**tensors, **extra})


@pytest.mark.parametrize("tp_size", [1, 2])
def test_load_weights_fills_a_model(tiny_config, tp_size):
    rank = tp_size - 1
    wmap = weights.build_weight_map(tiny_config, tp_size=tp_size, tp_rank=rank)
    expert = torch.zeros(2)
    model = _bag(wmap, **{"model.layers.0.mlp.experts.gate_up": expert})
    experts: list[str] = []
    loaded = weights.load_weights(
        model,
        NATIVE.items(),
        tiny_config,
        tp_size=tp_size,
        tp_rank=rank,
        expert_loader=lambda name, tensor: experts.append(name),
    )
    assert loaded == wmap.params.keys()
    assert len(experts) == NUM_EXPERT_TENSORS
    want = wmap.convert(NATIVE.items()).params
    for name, param in model.tensors.items():
        if name in want:
            assert param.dtype == want[name].dtype
            assert not param.isnan().any() and torch.equal(param, want[name]), name
    assert torch.equal(model.tensors["model.layers.0.mlp.experts.gate_up"], expert)


def test_vision_tensors_go_to_the_tower_only_when_built(tiny_config):
    num_visual = sum(name.startswith("model.visual.") for name in NATIVE)
    assert num_visual
    wmap = weights.build_weight_map(
        tiny_config, tp_size=1, tp_rank=0, dtype=torch.float32, with_visual=True
    )
    result = wmap.convert(NATIVE.items())
    assert len(result.visual) == num_visual
    name, tensor = result.visual[0]
    assert tensor is NATIVE["model.visual." + name]

    seen: list[str] = []
    model = _bag(wmap, **{"visual.proj_in.weight": torch.zeros(1)})
    weights.load_weights(
        model,
        NATIVE.items(),
        tiny_config,
        tp_size=1,
        tp_rank=0,
        dtype=torch.float32,
        visual_loader=lambda name, tensor: seen.append(name),
    )
    assert len(seen) == num_visual and "proj_in.weight" in seen
    with pytest.raises(ValueError, match="visual_loader must match with_visual"):
        wmap.load_into(model, NATIVE.items())


def test_load_weights_rejects_a_model_that_differs_from_the_map(tiny_config):
    wmap = weights.build_weight_map(tiny_config, tp_size=1, tp_rank=0)
    kwargs = {"tp_size": 1, "tp_rank": 0}

    model = _bag(wmap)
    del model.tensors["model.norm.weight"]
    with pytest.raises(ValueError, match="missing parameter model.norm.weight"):
        weights.load_weights(model, NATIVE.items(), tiny_config, **kwargs)
    model = _bag(wmap, **{"model.extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="unmapped parameter model.extra"):
        weights.load_weights(model, NATIVE.items(), tiny_config, **kwargs)
    model = _bag(wmap, **{"model.norm.weight": torch.zeros(64, dtype=torch.float16)})
    with pytest.raises(ValueError, match=r"model.norm.weight: \(64,\) torch.float16"):
        weights.load_weights(model, NATIVE.items(), tiny_config, **kwargs)
    model = _bag(wmap, **{"model.norm.weight": torch.zeros(1, 64)})
    with pytest.raises(ValueError, match=r"model.norm.weight: \(1, 64\) torch.float32"):
        weights.load_weights(model, NATIVE.items(), tiny_config, **kwargs)
