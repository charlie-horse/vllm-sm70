# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 expert loader (expert_weights.py).

CPU, without the Hugging Face Inkling 2 classes: the packed experts of the
tiny checkpoint (models/fixtures/inkling2_tiny/) are dequantized at TP1 and at
both ranks of TP2 and compared with the test oracle (models/inkling2_ref.py)
bit for bit and with the digests of the weights as the reference loaded them
(manifest.json `hf_state_dict`).
"""

import importlib.util
import json
import pathlib
import random
import sys

import pytest

pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch

from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import quant_config, sm70_configs, weights

MODELS = pathlib.Path(__file__).resolve().parents[1] / "models"
TINY = MODELS / "fixtures" / "inkling2_tiny"

sys.path.insert(0, str(MODELS))
from inkling2_ref import Inkling2Weights

_spec = importlib.util.spec_from_file_location("gen_fixture", TINY / "gen_fixture.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
EXPERTS = {k: v for k, v in NATIVE.items() if ".mlp.experts." in k}
HF_DIGESTS = json.loads((TINY / "manifest.json").read_text())["hf_state_dict"]
FORMATS = quant_config.expert_formats(
    json.loads((TINY / "hf_quant_config.json").read_text())
)
NVFP4_LAYER, MXFP8_LAYER = 3, 21
PREFIX = "model.llm.layers.{}.mlp.experts."


@pytest.fixture(scope="module")
def config():
    return sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )


@pytest.fixture(scope="module")
def oracle():
    return Inkling2Weights(NATIVE, torch.float32)


@pytest.fixture(scope="module")
def full(config):
    return ew.load_expert_layers(EXPERTS, config, formats=FORMATS)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def assert_bit_equal(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(bits(got), bits(want))


def layer_tensors(layer: int) -> dict[str, torch.Tensor]:
    prefix = PREFIX.format(layer)
    return {k: v.clone() for k, v in EXPERTS.items() if k.startswith(prefix)}


def one_layer(config, tensors, **kwargs) -> ew.ExpertLayer:
    """The single layer of `tensors` (the loader is not finished)."""
    loader = ew.Inkling2ExpertLoader(config, **kwargs)
    for name, tensor in tensors.items():
        loader(name, tensor)
    (layer,) = loader.layers.values()
    return layer


def test_formats_match_hf_quant_config(full):
    assert {i: layer.format for i, layer in full.items()} == FORMATS
    assert full[NVFP4_LAYER].format == ew.NVFP4
    assert full[MXFP8_LAYER].format == ew.MXFP8


@pytest.mark.parametrize("layer", range(24))
def test_tp1_bit_equal_to_reference(full, oracle, layer):
    w13, w2 = full[layer].dequantize(torch.float32)
    want13, want2 = oracle.experts(layer)
    assert_bit_equal(w13, want13)
    assert_bit_equal(w2, want2)
    hf = f"model.language_model.layers.{layer}.mlp.experts."
    for got, name in ((w13, "gate_up_proj"), (w2, "down_proj")):
        digest, want = gen._tensor_digest(got), HF_DIGESTS[hf + name]
        assert digest[0] == want[0]
        assert digest[1:] == pytest.approx(want[1:], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("layer", [NVFP4_LAYER, MXFP8_LAYER])
def test_fp16_is_the_rounded_reference(full, oracle, layer):
    w13, w2 = full[layer].dequantize()
    want13, want2 = oracle.experts(layer)
    assert w13.dtype == w2.dtype == torch.float16
    assert_bit_equal(w13, want13.half())
    assert_bit_equal(w2, want2.half())


@pytest.mark.parametrize("layer", [NVFP4_LAYER, MXFP8_LAYER])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_split_then_dequant_equals_dequant_then_split(config, full, layer, dtype):
    w13, w2 = full[layer].dequantize(dtype)
    inter = w2.shape[-1]
    local = inter // 2
    for rank in range(2):
        layers = ew.load_expert_layers(
            EXPERTS, config, tp_size=2, tp_rank=rank, formats=FORMATS
        )
        got13, got2 = layers[layer].dequantize(dtype)
        units = slice(local * rank, local * (rank + 1))
        gate, up = w13[:, :inter], w13[:, inter:]
        assert_bit_equal(got13, torch.cat((gate[:, units], up[:, units]), dim=1))
        assert_bit_equal(got2, w2[:, :, units])


def test_split_on_scale_block_boundaries(config):
    """A rank whose columns cover whole scale blocks, MXFP8 and NVFP4."""
    text = config.get_text_config()
    experts, hidden, inter = text.n_routed_experts, text.hidden_size, 64
    wide = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    wide.get_text_config().intermediate_size = inter
    gen_ = torch.Generator().manual_seed(7)
    p = PREFIX.format(0)

    def u8(*shape, high=256):
        return torch.randint(0, high, shape, generator=gen_, dtype=torch.uint8)

    def e4m3(*shape):
        return (torch.rand(shape, generator=gen_) * 4 - 2).to(torch.float8_e4m3fn)

    mx = {
        p + "w13_weight": e4m3(experts, 2 * inter, hidden),
        p + "w13_weight_scale": u8(experts, 2 * inter, hidden // 32, high=8) + 115,
        p + "w2_weight": e4m3(experts, hidden, inter),
        p + "w2_weight_scale": u8(experts, hidden, inter // 32, high=8) + 115,
    }
    nv = {
        p + "w13_weight": u8(experts, 2 * inter, hidden // 2),
        p + "w13_weight_scale": e4m3(experts, 2 * inter, hidden // 16).abs(),
        p + "w13_weight_amax": torch.rand(experts, generator=gen_) + 0.1,
        p + "w13_weight_outer_scale_denominator": torch.tensor([2688.0]),
        p + "w2_weight": u8(experts, hidden, inter // 2),
        p + "w2_weight_scale": e4m3(experts, hidden, inter // 16).abs(),
        p + "w2_weight_amax": torch.rand(experts, generator=gen_) + 0.1,
        p + "w2_weight_outer_scale_denominator": torch.tensor([2688.0]),
    }
    for tensors in (mx, nv):
        want13, want2 = Inkling2Weights(tensors, torch.float32).experts(0)
        w13, w2 = one_layer(wide, tensors).dequantize(torch.float32)
        assert_bit_equal(w13, want13)
        assert_bit_equal(w2, want2)
        for rank in range(2):
            layer = one_layer(wide, tensors, tp_size=2, tp_rank=rank)
            assert layer.w2.scale.shape[-1] == tensors[p + "w2_weight_scale"].shape[-1]
            got13, got2 = layer.dequantize(torch.float32)
            units = slice(32 * rank, 32 * (rank + 1))
            assert_bit_equal(
                got13,
                torch.cat((w13[:, :inter][:, units], w13[:, inter:][:, units]), dim=1),
            )
            assert_bit_equal(got2, w2[:, :, units])


@pytest.mark.parametrize("tp_size", [1, 2])
def test_nvfp4_codes_reproduce_the_dequantized_weights(config, tp_size):
    lut = torch.tensor(ew.E2M1_VALUES)
    for rank in range(tp_size):
        layer = ew.load_expert_layers(EXPERTS, config, tp_size=tp_size, tp_rank=rank)[
            NVFP4_LAYER
        ]
        dense = layer.dequantize(torch.float32)
        for codes, want in zip(layer.nvfp4_codes(), dense):
            assert codes.codes.dtype == torch.uint8
            assert codes.block_scale.dtype == torch.float8_e4m3fn
            assert codes.global_scale.dtype == torch.float32
            assert codes.codes.shape == want.shape
            assert codes.block_scale.shape == (*want.shape[:2], want.shape[2] // 16)
            assert codes.codes.is_contiguous() and codes.block_scale.is_contiguous()
            assert int(codes.codes.max()) < 16
            scale = codes.block_scale.float().repeat_interleave(16, dim=-1)
            values = lut[codes.codes.int()] * scale
            values = values * codes.global_scale[:, None, None]
            assert_bit_equal(values, want)


def test_nvfp4_global_scale_is_amax_over_denominator(full):
    layer = full[NVFP4_LAYER]
    p = PREFIX.format(NVFP4_LAYER)
    for codes, name in zip(layer.nvfp4_codes(), ("w13", "w2")):
        want = (
            EXPERTS[f"{p}{name}_weight_amax"]
            / EXPERTS[f"{p}{name}_weight_outer_scale_denominator"]
        )
        assert torch.equal(codes.global_scale, want)
        assert codes.name == f"{p}{name}_weight"


def test_mxfp8_layer_has_no_codes(full):
    with pytest.raises(ValueError, match="w13_weight: mxfp8"):
        full[MXFP8_LAYER].nvfp4_codes()


def test_loader_is_the_weight_map_expert_callback(config, full):
    """Tensors as weights.py hands them over, in a shuffled order."""
    stream = weights.build_weight_map(config, tp_size=1, tp_rank=0).convert(
        NATIVE.items()
    )
    assert {name for name, _ in stream.experts} == set(EXPERTS)
    order = list(stream.experts)
    random.Random(3).shuffle(order)
    seen: list[int] = []
    loader = ew.Inkling2ExpertLoader(
        config, formats=FORMATS, sink=lambda layer: seen.append(layer.layer)
    )
    for name, tensor in order:
        loader(name, tensor)
    assert loader.finish() == set(EXPERTS)
    assert sorted(seen) == list(range(24)) and not loader.layers


def test_incomplete_and_unknown_tensors_raise(config):
    loader = ew.Inkling2ExpertLoader(config)
    with pytest.raises(ValueError, match="unexpected expert tensor"):
        loader("model.llm.layers.0.mlp.gate.weight", torch.zeros(1))
    with pytest.raises(ValueError, match="unexpected expert tensor"):
        loader(PREFIX.format(24) + "w2_weight", torch.zeros(1))
    tensors = layer_tensors(NVFP4_LAYER)
    name, tensor = tensors.popitem()
    for item in tensors.items():
        loader(*item)
    with pytest.raises(ValueError, match="duplicate expert tensor"):
        loader(*next(iter(tensors.items())))
    with pytest.raises(ValueError, match=f"layer {NVFP4_LAYER}: incomplete"):
        loader.finish()
    loader(name, tensor)
    with pytest.raises(ValueError, match="after layer 3 was complete"):
        loader(name, tensor)
    with pytest.raises(ValueError, match="no expert tensors for layers"):
        loader.finish()


def test_format_cross_checks(config):
    wrong = dict(FORMATS)
    wrong[NVFP4_LAYER] = ew.MXFP8
    with pytest.raises(ValueError, match="layer 3 experts are nvfp4 .* 'mxfp8'"):
        one_layer(config, layer_tensors(NVFP4_LAYER), formats=wrong)

    p = PREFIX.format(MXFP8_LAYER)
    tensors = layer_tensors(MXFP8_LAYER)
    tensors[p + "w2_weight_scale"] = tensors[p + "w2_weight_scale"].float()
    with pytest.raises(ValueError, match="w2_weight: .* neither NVFP4 nor MXFP8"):
        one_layer(config, tensors)

    # An amax in an MXFP8 layer, before and after the layer's other tensors.
    tensors = layer_tensors(MXFP8_LAYER)
    with pytest.raises(ValueError, match="w13_weight_amax' after layer 21"):
        one_layer(config, tensors | {p + "w13_weight_amax": torch.ones(8)})
    with pytest.raises(ValueError, match="w13_weight: .* neither NVFP4 nor MXFP8"):
        one_layer(config, {p + "w13_weight_amax": torch.ones(8)} | tensors)

    mixed = layer_tensors(MXFP8_LAYER)
    for name, tensor in layer_tensors(NVFP4_LAYER).items():
        if ".w2_weight" in name:
            mixed[name.replace(".3.", f".{MXFP8_LAYER}.")] = tensor
    with pytest.raises(ValueError, match=r"mxfp8 \(w13\) and nvfp4 \(w2\)"):
        one_layer(config, mixed)


def test_shape_and_split_checks(config):
    p = PREFIX.format(NVFP4_LAYER)
    tensors = layer_tensors(NVFP4_LAYER)
    tensors[p + "w13_weight_scale"] = tensors[p + "w13_weight_scale"][..., :2]
    with pytest.raises(ValueError, match=r"w13_weight_scale: checkpoint shape"):
        one_layer(config, tensors)
    with pytest.raises(ValueError, match="doesn't split over 3 ranks"):
        ew.Inkling2ExpertLoader(config, tp_size=3)
    with pytest.raises(ValueError, match="TP rank 2 outside"):
        ew.Inkling2ExpertLoader(config, tp_size=2, tp_rank=2)
    # 32 intermediate units over 4 ranks: 8 columns, half an NVFP4 block.
    layer = one_layer(config, layer_tensors(NVFP4_LAYER), tp_size=4, tp_rank=1)
    full13, full2 = one_layer(config, layer_tensors(NVFP4_LAYER)).dequantize()
    got13, got2 = layer.dequantize()
    assert_bit_equal(got2, full2[:, :, 8:16])
    assert_bit_equal(got13, torch.cat((full13[:, 8:16], full13[:, 40:48]), dim=1))
    with pytest.raises(ValueError, match="w2_weight: columns 8..16 don't align"):
        layer.nvfp4_codes()


def test_planted_nvfp4_overflow_raises_with_tensor_name(config):
    p = PREFIX.format(NVFP4_LAYER)
    tensors = layer_tensors(NVFP4_LAYER)
    tensors[p + "w2_weight_amax"][5] = 3.0e7
    layer = one_layer(config, tensors)
    layer.w13.dequantize()
    with pytest.raises(ValueError) as error:
        layer.dequantize()
    message = str(error.value)
    assert message.startswith(p + "w2_weight (experts 0-7): max |x| = ")
    assert "does not fit in torch.float16" in message
    # fp32 holds the same values.
    layer.dequantize(torch.float32)


def test_planted_mxfp8_overflow_and_flush_raise_with_tensor_name(config):
    p = PREFIX.format(MXFP8_LAYER)
    tensors = layer_tensors(MXFP8_LAYER)
    weight = tensors[p + "w13_weight"].float()
    expert, row, _ = (weight[:, :, :32].abs() >= 1).nonzero()[0].tolist()
    tensors[p + "w13_weight_scale"][expert, row, 0] = 127 + 17
    with pytest.raises(ValueError, match="w13_weight .*does not fit in torch.float16"):
        one_layer(config, tensors).dequantize()

    tensors[p + "w13_weight_scale"][expert, row, 0] = 127 - 26
    with pytest.raises(
        ValueError, match=r"w13_weight \(experts 0-7\): \d+ nonzero values round to"
    ):
        one_layer(config, tensors).dequantize()
    one_layer(config, tensors).dequantize(torch.float32)

    tensors[p + "w13_weight_scale"][expert, row, 0] = 255
    with pytest.raises(ValueError, match="w13_weight .*does not fit"):
        one_layer(config, tensors).dequantize(torch.float32)


def test_fp16_subnormals_are_kept(config):
    p = PREFIX.format(MXFP8_LAYER)
    tensors = layer_tensors(MXFP8_LAYER)
    tensors[p + "w2_weight_scale"].fill_(127 - 12)
    w2 = one_layer(config, tensors).w2.dequantize()
    want = Inkling2Weights(tensors, torch.float32).experts(MXFP8_LAYER)[1]
    subnormal = (w2 != 0) & (w2.abs() < torch.finfo(torch.float16).tiny)
    assert subnormal.any()
    assert torch.equal(w2.float()[subnormal], want[subnormal])


def test_range_check_hook(config):
    seen: list[tuple[str, torch.dtype, torch.dtype]] = []
    layer = one_layer(config, layer_tensors(NVFP4_LAYER))
    layer.dequantize(
        range_check=lambda name, values, cast: seen.append(
            (name, values.dtype, cast.dtype)
        )
    )
    p = PREFIX.format(NVFP4_LAYER)
    assert seen == [
        (p + "w13_weight (experts 0-7)", torch.float32, torch.float16),
        (p + "w2_weight (experts 0-7)", torch.float32, torch.float16),
    ]
    tensors = layer_tensors(NVFP4_LAYER)
    tensors[p + "w2_weight_amax"][0] = 3.0e7
    w2 = one_layer(config, tensors).w2.dequantize(range_check=None)
    assert torch.isinf(w2[0]).any()


def test_chunks_cover_every_expert(config, monkeypatch):
    layer = one_layer(config, layer_tensors(NVFP4_LAYER))
    want = layer.dequantize()
    monkeypatch.setattr(ew, "EXPERTS_PER_CHUNK", 3)
    names: list[str] = []
    got = layer.dequantize(range_check=lambda name, *_: names.append(name))
    for a, b in zip(got, want):
        assert_bit_equal(a, b)
    assert [n.split("(")[1] for n in names[:3]] == [
        "experts 0-2)",
        "experts 3-5)",
        "experts 6-7)",
    ]
