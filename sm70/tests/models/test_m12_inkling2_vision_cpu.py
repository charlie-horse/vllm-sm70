# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 vision encoder (inkling2_sm70/vision.py).

CPU fp32 against the image features recorded in fixtures/inkling2_tiny (two
non-square images). The comparison with a live Hugging Face tower, which adds
a square image, skips without the Inkling 2 classes in transformers.
"""

import importlib.util
import json
import pathlib

import pytest

pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch
import transformers

from vllm_sm70.models.inkling2_sm70 import vision

HERE = pathlib.Path(__file__).resolve().parent
TINY = HERE / "fixtures" / "inkling2_tiny"
MINI = HERE / "data" / "inkling2_mini"
ATOL = 1e-4
PREFIX = "model.visual."

CONFIG = json.loads((TINY / "config.json").read_text())["vision_config"]
NATIVE = {
    name: tensor
    for name, tensor in safetensors_torch.load_file(TINY / "model.safetensors").items()
    if name.startswith(PREFIX)
}
CASES = safetensors_torch.load_file(TINY / "cases.safetensors")
PIXELS, GRID = CASES["image.pixel_values"], CASES["image.image_grid_thw"]
FEATURES = CASES["image.features"]
SIZES = GRID.prod(-1).tolist()

HAS_REFERENCE = hasattr(transformers, "Inkling2LoopForConditionalGeneration")


def build(dtype=torch.float32, **kwargs) -> vision.Inkling2VisionEncoder:
    encoder = vision.Inkling2VisionEncoder(CONFIG, dtype=dtype, **kwargs)
    encoder.load_weights(NATIVE.items())
    return encoder.eval()


@pytest.fixture(scope="module")
def encoder():
    return build()


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape and a.dtype == b.dtype
    return (a - b).abs().max().item()


@torch.inference_mode()
def test_two_image_batch_matches_the_recorded_features(encoder):
    assert max_diff(encoder.encode(PIXELS, GRID), FEATURES) <= ATOL


@torch.inference_mode()
@pytest.mark.parametrize("image", range(len(SIZES)))
def test_single_non_square_image_matches_the_recorded_features(encoder, image):
    rows, cols = GRID[image, 1:].tolist()
    assert rows != cols
    pixels = PIXELS.split(SIZES)[image]
    got = encoder.encode(pixels, GRID[image : image + 1])
    assert max_diff(got, FEATURES.split(SIZES)[image]) <= ATOL


@torch.inference_mode()
def test_forward_normalizes_raw_fp16_pixels_in_fp32(encoder):
    raw = torch.randint(
        0, 256, PIXELS.shape, generator=torch.Generator().manual_seed(3)
    )
    raw[0, :, 5:] = -1  # padding
    raw = raw.to(torch.float16)
    normalized = vision.normalize_pixels(raw)
    assert normalized.dtype == torch.float32
    exact = raw.double().mul(2).div(255).sub(1)
    assert (normalized.double() - exact).abs().max().item() <= 2e-7
    assert normalized.min().item() == pytest.approx(-1 - 2 / 255)
    assert torch.equal(encoder(raw, GRID), encoder.encode(normalized, GRID))


@torch.inference_mode()
def test_merge_order_mutation_fails(encoder, monkeypatch):
    """Tiles cut and positioned column-major: only the 2x2 concat order is wrong."""

    def tile_patches(pixels, merge, patch):
        x = pixels.reshape(-1, merge, patch, merge, patch, pixels.shape[-1])
        return x.permute(0, 3, 1, 2, 4, 5).reshape(-1, patch * patch * pixels.shape[-1])

    def patch_positions(rows, cols, merge, device=None):
        y = torch.arange(rows * merge).view(rows, 1, 1, merge)
        x = torch.arange(cols * merge).view(1, cols, merge, 1)
        shape = (rows, cols, merge, merge)
        return torch.stack((y.expand(shape), x.expand(shape)), dim=-1).reshape(-1, 2)

    monkeypatch.setattr(vision, "tile_patches", tile_patches)
    monkeypatch.setattr(vision, "patch_positions", patch_positions)
    assert max_diff(encoder.encode(PIXELS, GRID), FEATURES) > 10 * ATOL


@torch.inference_mode()
def test_rope_axis_swap_mutation_fails(encoder, monkeypatch):
    """Height angles in the first half of the head, width angles in the second."""
    rope_cos_sin = vision.rope_cos_sin
    monkeypatch.setattr(
        vision, "rope_cos_sin", lambda pos, *args: rope_cos_sin(pos.flip(-1), *args)
    )
    assert max_diff(encoder.encode(PIXELS, GRID), FEATURES) > 10 * ATOL


@torch.inference_mode()
@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_chunked_attention_equals_unchunked(encoder, chunk):
    chunked = build(attn_query_chunk=chunk)
    # Longest image: 2x3 tiles = 25 tokens with the CLS token.
    assert encoder.attn_query_chunk > 25 > chunk
    assert max_diff(chunked.encode(PIXELS, GRID), encoder.encode(PIXELS, GRID)) <= 1e-5


@torch.inference_mode()
def test_empty_batch(encoder):
    out = encoder(PIXELS[:0].half(), GRID[:0])
    assert out.shape == (0, CONFIG["text_hidden_size"]) and out.dtype == torch.float32


@torch.inference_mode()
def test_rejects_malformed_inputs(encoder):
    with pytest.raises(ValueError, match="tiles do not fill"):
        encoder.encode(PIXELS[:-1], GRID)
    with pytest.raises(ValueError, match="expected \\[N, 8, 8, 3\\]"):
        encoder.encode(PIXELS.permute(0, 3, 1, 2), GRID)
    with pytest.raises(ValueError, match="rows, cols"):
        encoder.encode(PIXELS, torch.tensor([[2, 2, 2]]))


def test_every_released_visual_tensor_maps_to_a_parameter():
    config = json.loads((MINI / "config.json").read_text())["vision_config"]
    index = json.loads((MINI / "model.safetensors.index.json").read_text())
    names = {
        n.removeprefix(PREFIX) for n in index["weight_map"] if n.startswith(PREFIX)
    }
    encoder = vision.Inkling2VisionEncoder(config, device="meta")
    params = dict(encoder.named_parameters())
    assert len(names) == 422
    assert names == params.keys()
    assert all(p.dtype == torch.float16 for p in params.values())
    assert encoder.text_hidden_size == 2304
    assert params["backbone.embeddings.patch_embedding.weight"].shape == (1024, 1200)
    assert params["backbone.embeddings.position_embedding"].shape == (32 * 32, 1024)
    assert len(encoder.backbone.layers) == 23


def test_loader_strips_the_prefix_and_casts_bf16_to_fp16():
    assert all(t.dtype == torch.bfloat16 for t in NATIVE.values())
    encoder = vision.Inkling2VisionEncoder(CONFIG)
    loaded = encoder.load_weights(NATIVE.items())
    params = dict(encoder.named_parameters())
    assert loaded == params.keys() == {n.removeprefix(PREFIX) for n in NATIVE}
    for name, param in params.items():
        assert param.dtype == torch.float16
        assert torch.equal(param, NATIVE[PREFIX + name].to(torch.float16)), name


@pytest.mark.parametrize("problem", ["missing", "unexpected", "duplicate", "shape"])
def test_loader_fails_on_a_wrong_key_set(problem):
    weights = list(NATIVE.items())
    name, tensor = weights[-1]
    if problem == "missing":
        weights.pop()
    elif problem == "unexpected":
        weights.append((PREFIX + "backbone.ln_post.weight", tensor))
    elif problem == "duplicate":
        weights.append((name, tensor))
    else:
        weights[-1] = (name, tensor[:-1])
    match = {"missing": "Missing", "shape": "has shape"}.get(problem, "Unexpected")
    with pytest.raises(ValueError, match=match):
        vision.Inkling2VisionEncoder(CONFIG).load_weights(weights)


@torch.inference_mode()
def test_fp16_output_is_finite_and_near_fp32():
    out = build(torch.float16)(((PIXELS + 1) * 127.5).round().half(), GRID)
    want = build()(((PIXELS + 1) * 127.5).round().half(), GRID)
    assert out.dtype == torch.float16 and out.isfinite().all()
    assert max_diff(out.float(), want) <= 2e-2


@pytest.fixture(scope="module")
def reference():
    spec = importlib.util.spec_from_file_location(
        "gen_fixture", TINY / "gen_fixture.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen.load_reference(TINY)


@pytest.mark.skipif(
    not HAS_REFERENCE, reason="needs transformers with the Inkling 2 classes"
)
@torch.inference_mode()
@pytest.mark.parametrize(
    "grid",
    [
        [[1, 2, 2]],
        [[1, 1, 1]],
        [[1, 3, 3]],
        [[1, 1, 3]],
        [[1, 3, 2]],
        [[1, 2, 2], [1, 4, 1]],
    ],
    ids=["square", "one-tile", "square-3", "wide", "tall", "square-and-tall"],
)
def test_matches_a_live_reference_tower(encoder, reference, grid):
    grid = torch.tensor(grid)
    side = CONFIG["native_patch_size"] * CONFIG["adapter_merge_size"]
    generator = torch.Generator().manual_seed(11)
    pixels = torch.rand(int(grid.prod(-1).sum()), side, side, 3, generator=generator)
    pixels = pixels * 2 - 1
    want = reference.get_image_features(pixels, grid, return_dict=True).pooler_output
    assert max_diff(encoder.encode(pixels, grid), torch.cat(want)) <= ATOL
