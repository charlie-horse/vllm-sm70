# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 torch expert path (moe.py).

CPU, without the Hugging Face Inkling 2 classes: the router and the experts of
the tiny checkpoint (models/fixtures/inkling2_tiny/) are run at TP1 and as the
two rank partials of TP2 and compared with the router and expert recordings of
the reference (ops.safetensors). The recordings hold 20 tokens; M = 64 repeats
them, which the per-token ops allow.
"""

import json
import pathlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch

from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import moe, quant_config, sm70_configs
from vllm_sm70.models.inkling2_sm70.ops import torch_ref

TINY = pathlib.Path(__file__).resolve().parents[1] / "models/fixtures/inkling2_tiny"
NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
OPS = safetensors_torch.load_file(TINY / "ops.safetensors")
FORMATS = quant_config.expert_formats(
    json.loads((TINY / "hf_quant_config.json").read_text())
)
SIZES = [1, 2, 8, 64]
TOL = 1e-5
# Recorded op -> (layer, hidden states, top-k index, top-k weights, output).
CASES = {
    "nvfp4": (
        1,
        "router.0.in.hidden_states",
        "router.0.out.2",
        "router.0.out.1",
        "experts_nvfp4.0.out.0",
    ),
    "mxfp8": (
        23,
        "experts_mxfp8.0.in.hidden_states",
        "experts_mxfp8.0.in.top_k_index",
        "experts_mxfp8.0.in.top_k_weights",
        "experts_mxfp8.0.out.0",
    ),
}
ROUTER_LAYER = 1


@pytest.fixture(scope="module")
def config():
    return sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )


def build(config, layer, *, tp_size=1, tp_rank=0, dtype=torch.float32):
    module = moe.Inkling2Sm70Experts(config, tp_size=tp_size, dtype=dtype)
    layers = ew.load_expert_layers(
        NATIVE, config, tp_size=tp_size, tp_rank=tp_rank, formats=FORMATS
    )
    module.load_expert_layer(layers[layer])
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    module.gate.weight.data.copy_(NATIVE[prefix + "weight"])
    module.gate.bias.data.copy_(NATIVE[prefix + "bias"])
    return module


def rows(m: int) -> torch.Tensor:
    return torch.arange(m) % OPS["router.0.in.hidden_states"].shape[0]


def recorded(case: str, m: int):
    layer, *names = CASES[case]
    return (layer, *(OPS[name][rows(m)] for name in names))


def run(module, x, index, weights) -> torch.Tensor:
    out = torch.full_like(x, float("nan"))
    module.expert_sum(x, index, weights, out)
    return out


def test_fixture_covers_the_recorded_shapes(config):
    text = config.get_text_config()
    x = OPS["router.0.in.hidden_states"]
    assert x.shape == (20, text.hidden_size) and x.dtype == torch.float32
    assert OPS["router.0.out.2"].shape == (20, text.num_experts_per_tok)
    assert {FORMATS[CASES[c][0]] for c in CASES} == {ew.NVFP4, ew.MXFP8}


def test_module_layout(config):
    text = config.get_text_config()
    e, h, i = text.n_routed_experts, text.hidden_size, text.intermediate_size
    one = moe.Inkling2Sm70Experts(config)
    assert one.w13.shape == (e, 2 * i, h) and one.w2.shape == (e, h, i)
    assert one.w13.dtype == one.w2.dtype == torch.float16
    assert {n: (tuple(p.shape), p.dtype) for n, p in one.named_parameters()} == {
        "gate.weight": ((e, h), torch.float32),
        "gate.bias": ((e,), torch.float32),
    }
    assert one.state_dict().keys() == {"gate.weight", "gate.bias"}
    half = moe.Inkling2Sm70Experts(config, tp_size=2)
    assert half.w13.shape == (e, i, h) and half.w2.shape == (e, h, i // 2)
    with pytest.raises(ValueError, match="doesn't split"):
        moe.Inkling2Sm70Experts(config, tp_size=3)


@pytest.mark.parametrize("m", SIZES + [20])
def test_router_matches_fixture(config, m):
    module = build(config, ROUTER_LAYER)
    x = OPS["router.0.in.hidden_states"][rows(m)]
    logits, weights, indices = module.route(x)
    assert logits.dtype == weights.dtype == torch.float32
    assert torch.equal(indices, OPS["router.0.out.2"][rows(m)])
    assert (logits - OPS["router.0.out.0"][rows(m)]).abs().max() <= TOL
    assert (weights - OPS["router.0.out.1"][rows(m)]).abs().max() <= TOL


def test_router_stays_fp32_for_fp16_input(config):
    module = build(config, ROUTER_LAYER, dtype=torch.float16)
    x = OPS["router.0.in.hidden_states"].half()
    logits, weights, indices = module.route(x)
    want = torch_ref.router(
        x.float(), module.gate.weight, module.gate.bias, 2, module.gate.route_scale
    )
    assert logits.dtype == weights.dtype == torch.float32
    assert torch.equal(logits, want[0]) and torch.equal(weights, want[1])
    assert torch.equal(indices, want[2])
    assert module.gate.weight.dtype == module.gate.bias.dtype == torch.float32


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("m", SIZES)
def test_tp1_matches_fixture(config, case, m):
    layer, x, index, weights, want = recorded(case, m)
    got = run(build(config, layer), x, index, weights)
    assert got.dtype == torch.float32
    assert (got - want).abs().max() <= TOL


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("m", SIZES)
def test_tp2_rank_partials_sum_to_fixture(config, case, m):
    layer, x, index, weights, want = recorded(case, m)
    partials = [
        run(build(config, layer, tp_size=2, tp_rank=rank), x, index, weights)
        for rank in range(2)
    ]
    assert all(p.abs().max() > 0 for p in partials)
    assert (partials[0] - partials[1]).abs().max() > 1e-3
    assert (partials[0] + partials[1] - want).abs().max() <= TOL
    tp1 = run(build(config, layer), x, index, weights)
    assert (partials[0] + partials[1] - tp1).abs().max() <= TOL


@pytest.mark.parametrize("m", SIZES)
def test_forward_routes_and_writes_the_caller_buffer(config, m):
    layer, x, _, _, want = recorded("nvfp4", m)
    out = torch.full_like(x, float("nan"))
    data = out.data_ptr()
    partials = []
    for rank in range(2):
        module = build(config, layer, tp_size=2, tp_rank=rank)
        assert module(x, out) is None
        assert out.data_ptr() == data
        partials.append(out.clone())
    assert (partials[0] + partials[1] - want).abs().max() <= TOL
    assert hasattr(moe.Inkling2Sm70Experts.forward, "__wrapped__")


def test_padding_rows_are_zero(config):
    layer, x, _, _, want = recorded("nvfp4", 8)
    module = build(config, layer)
    pad = torch.tensor([False, True, False, False, True, True, False, True])
    _, weights, indices = module.route(x, pad)
    assert (indices[pad] == module.num_experts).all() and (weights[pad] == 0).all()
    out = torch.full_like(x, float("nan"))
    module(x, out, pad)
    assert (out[pad] == 0).all()
    assert (out[~pad] - want[~pad]).abs().max() <= TOL
    module(x, out, torch.ones(8, dtype=torch.bool))
    assert (out == 0).all()


def test_ids_outside_the_expert_range_select_nothing(config):
    layer, x, index, weights, want = recorded("mxfp8", 8)
    module = build(config, layer)
    index = index.clone()
    index[3] = module.num_experts
    index[5, 0] = -1
    got = run(module, x, index, weights)
    keep = torch.ones(8, dtype=torch.bool)
    keep[[3, 5]] = False
    assert (got[3] == 0).all()
    assert (got[keep] - want[keep]).abs().max() <= TOL
    only = index.clone()
    only[:] = module.num_experts
    only[5, 1] = index[5, 1]
    assert (run(module, x, only, weights)[5] - got[5]).abs().max() <= TOL
    assert (got[5] - want[5]).abs().max() > 1e-3


@pytest.mark.parametrize("case", list(CASES))
def test_fp16_rounds_the_fp32_sum_once(config, case):
    layer, x, index, weights, want = recorded(case, 20)
    module = build(config, layer, dtype=torch.float16)
    x = x.half()
    got = run(module, x, index, weights)
    assert got.dtype == torch.float16
    ref = torch_ref.experts(
        x, index, weights, module.w13, module.w2, module.gate_cap, module.up_cap
    )
    assert torch.equal(got, ref)
    assert (got.float() - want).abs().max() <= 2e-2 * want.abs().max()
    partials = [
        run(
            build(config, layer, tp_size=2, tp_rank=rank, dtype=torch.float16),
            x,
            index,
            weights,
        )
        for rank in range(2)
    ]
    total = partials[0].float() + partials[1].float()
    assert (total - got.float()).abs().max() <= 2e-2 * want.abs().max()


def test_load_rejects_another_ranks_shape(config):
    layers = ew.load_expert_layers(NATIVE, config, formats=FORMATS)
    module = moe.Inkling2Sm70Experts(config, tp_size=2, dtype=torch.float32)
    with pytest.raises(ValueError, match="w13 has shape"):
        module.load_expert_layer(layers[0])


def test_soft_caps_must_be_positive(config):
    text = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    ).get_text_config()
    text.mlp_up_softcap = 0.0
    with pytest.raises(ValueError, match="must be positive"):
        moe.Inkling2Sm70Experts(text)
