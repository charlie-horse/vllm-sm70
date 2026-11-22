# SPDX-License-Identifier: Apache-2.0
"""Tests for M12 Inkling 2 image preprocessing and placeholder expansion
(vllm_sm70/models/inkling2_sm70/processing.py).

CPU only: grid and token counts, fp16 raw pixels with -1 padding, alpha on the
checkerboard, and both prompt-update targets (plugin renderer and Jinja).
"""

import io
import json
import math
import pathlib
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import numpy as np
import torch
from PIL import Image
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.processing import (
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)

from vllm_sm70.models.inkling2_sm70.processing import (
    IMAGE_MARKER_ID,
    IMAGE_PAD_ID,
    NATIVE_RENDERER_MODE,
    PATCH_SIZE,
    Inkling2DummyInputsBuilder,
    Inkling2ImageProcessor,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)

DATA = pathlib.Path(__file__).resolve().parent / "data" / "inkling2_mini"
TARGET_MODES = {"marker": NATIVE_RENDERER_MODE, "pad": "hf"}
BOS, END = 200000, 200010


def _raw(pixels) -> torch.Tensor:
    return torch.from_numpy(np.asarray(pixels, dtype=np.float32)).half()


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_processor(tokenizer_mode, vision_config=None):
    mm_config = MultiModalConfig()
    model_config = SimpleNamespace(
        model="inkling2",
        tokenizer_mode=tokenizer_mode,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    if vision_config is not None:
        model_config.hf_config = SimpleNamespace(vision_config=vision_config)
    info = Inkling2ProcessingInfo(InputProcessingContext(model_config, None))
    return Inkling2MultiModalProcessor(info, Inkling2DummyInputsBuilder(info))


def _apply(processor, prompt, images):
    inputs = ProcessorInputs(
        prompt=prompt, mm_data_items=processor.info.parse_mm_data({"image": images})
    )
    return processor.apply(inputs, TimingContext(enabled=False))


def test_grid_covers_image_without_extra_column():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (39, 41, 3), dtype=np.uint8)
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(pixels)])

    assert out["vision_grid_thw"].tolist() == [[1, 1, 2]]
    patches = out["vision_pixels"]
    assert patches.dtype == torch.float16
    assert patches.shape == (2, 40, 40, 3)
    # Row-major cells of raw values; pixels beyond the image are -1.
    assert torch.equal(patches[0, :39], _raw(pixels[:, :40]))
    assert torch.equal(patches[1, :39, :1], _raw(pixels[:, 40:]))
    assert (patches[1, :, 1:] == -1).all()
    assert (patches[:, 39] == -1).all()


# (width, height): multiple of 40, padded, one cell, wide strip, tall strip.
@pytest.mark.parametrize(
    "size", [(120, 80), (130, 95), (1, 1), (40, 40), (2000, 7), (3, 401)]
)
def test_token_count_and_padding(size):
    width, height = size
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(pixels)])

    rows, cols = math.ceil(height / 40), math.ceil(width / 40)
    assert out["vision_grid_thw"].tolist() == [[1, rows, cols]]
    patches = out["vision_pixels"]
    assert patches.dtype == torch.float16
    assert patches.shape == (rows * cols, 40, 40, 3)
    # Reassemble the padded canvas from the row-major cells.
    canvas = patches.view(rows, cols, 40, 40, 3).permute(0, 2, 1, 3, 4)
    canvas = canvas.reshape(rows * 40, cols * 40, 3)
    assert torch.equal(canvas[:height, :width], _raw(pixels))
    assert (canvas[height:] == -1).all()
    assert (canvas[:, width:] == -1).all()


def test_two_images_are_concatenated_in_order():
    first = np.full((40, 80, 3), 7, dtype=np.uint8)
    second = np.full((50, 30, 3), 9, dtype=np.uint8)
    out = Inkling2ImageProcessor().preprocess(
        [Image.fromarray(first), Image.fromarray(second)]
    )

    assert out["vision_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 1]]
    patches = out["vision_pixels"]
    assert patches.shape == (4, 40, 40, 3)
    assert (patches[:2] == 7).all()
    assert (patches[2, :, :30] == 9).all() and (patches[2, :, 30:] == -1).all()
    assert (patches[3, :10, :30] == 9).all() and (patches[3, 10:] == -1).all()


def test_transparency_composited_on_checkerboard():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[0, 0] = (10, 20, 30, 255)
    image = Image.fromarray(rgba, "RGBA")
    expected = np.where(
        (np.arange(16)[None, :] // 8 + np.arange(16)[:, None] // 8) % 2 == 0, 204, 153
    )
    expected = np.repeat(expected[:, :, None], 3, axis=2)
    expected[0, 0] = (10, 20, 30)

    flattened = MediaWithBytes(image.convert("RGB"), _png(image))
    processor = Inkling2ImageProcessor()
    for image_input in (image, _png(image), flattened):
        out = processor.preprocess([image_input])
        assert out["vision_grid_thw"].tolist() == [[1, 1, 1]]
        patches = out["vision_pixels"]
        assert torch.equal(patches[0, :16, :16], _raw(expected))
        assert (patches[0, 16:] == -1).all() and (patches[0, :, 16:] == -1).all()


def test_partial_alpha_blend_rounds_to_nearest():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., :3] = 100
    rgba[..., 3] = 128
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(rgba, "RGBA")])
    # (100*128 + 204*127 + 127) // 255
    assert (out["vision_pixels"][0, :8, :8] == 152).all()


@pytest.mark.parametrize("target", ["marker", "pad"])
def test_placeholder_expansion(target):
    processor = _make_processor(TARGET_MODES[target])
    assert processor.info.image_target == target
    per_image = processor.info.image_prompt_ids
    assert per_image == (
        [IMAGE_MARKER_ID] if target == "marker" else [IMAGE_MARKER_ID, IMAGE_PAD_ID]
    )
    images = [Image.new("RGB", (100, 50)), Image.new("RGBA", (40, 40))]
    out = _apply(processor, [BOS, *per_image, END, BOS, *per_image, END], images)

    ids = out["prompt_token_ids"]
    assert ids == (
        [BOS, IMAGE_MARKER_ID]
        + [IMAGE_PAD_ID] * 6
        + [END, BOS, IMAGE_MARKER_ID, IMAGE_PAD_ID, END]
    )
    assert ids.count(IMAGE_MARKER_ID) == 2
    ranges = out["mm_placeholders"]["image"]
    embed_positions = []
    for r in ranges:
        mask = (
            r.is_embed
            if r.is_embed is not None
            else torch.ones(r.length, dtype=torch.bool)
        )
        embed_positions.append([r.offset + i for i in range(r.length) if bool(mask[i])])
    assert embed_positions == [list(range(2, 8)), [11]]
    assert all(ids[p] == IMAGE_PAD_ID for ps in embed_positions for p in ps)
    data = out["mm_kwargs"].get_data()
    assert data["vision_pixels"].shape == (7, 40, 40, 3)
    assert data["vision_pixels"].dtype == torch.float16
    assert data["vision_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 1]]


@pytest.mark.parametrize("target", ["marker", "pad"])
@pytest.mark.parametrize("size", [(80, 80), (41, 39), (17, 17), (1000, 30)])
def test_each_target_yields_one_marker_and_n_pads(target, size):
    processor = _make_processor(TARGET_MODES[target])
    width, height = size
    num_tokens = math.ceil(height / 40) * math.ceil(width / 40)
    prompt = [BOS, *processor.info.image_prompt_ids, END]
    out = _apply(processor, prompt, [Image.new("RGB", size)])

    assert out["prompt_token_ids"] == (
        [BOS, IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * num_tokens + [END]
    )


def test_pad_target_needs_the_template_pad():
    processor = _make_processor("hf")
    with pytest.raises(ValueError):
        _apply(processor, [BOS, IMAGE_MARKER_ID, END], [Image.new("RGB", (40, 40))])


@pytest.mark.parametrize("target", ["marker", "pad"])
def test_dummy_inputs_expand(target):
    processor = _make_processor(TARGET_MODES[target])
    inputs = processor.get_dummy_inputs(100, {"image": 2}, {})
    out = processor.apply(inputs, TimingContext(enabled=False))

    # isqrt(100) = 10 cells per side.
    assert out["mm_kwargs"].get_data()["vision_grid_thw"].tolist() == [[1, 10, 10]] * 2
    ids = out["prompt_token_ids"]
    assert ids == ([IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * 100) * 2
    marker, pad = "<|content_image|>", "<|unused_200054|>"
    text = processor.dummy_inputs.get_dummy_text({"image": 2})
    assert text == (marker if target == "marker" else marker + pad) * 2


def test_patch_size_from_checkpoint_vision_config():
    vision = json.loads((DATA / "config.json").read_text())["vision_config"]
    processor = _make_processor(NATIVE_RENDERER_MODE, vision)
    assert processor.info.patch_size == PATCH_SIZE == 40
    assert processor.info.image_processor.patch_size == 40

    tiny = _make_processor(
        NATIVE_RENDERER_MODE, {"native_patch_size": 4, "adapter_merge_size": 2}
    )
    assert tiny.info.patch_size == 8
    out = _apply(tiny, [IMAGE_MARKER_ID], [Image.new("RGB", (20, 9), (1, 2, 3))])
    assert out["prompt_token_ids"] == [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * 6
    data = out["mm_kwargs"].get_data()
    assert data["vision_pixels"].shape == (6, 8, 8, 3)
    assert data["vision_grid_thw"].tolist() == [[1, 2, 3]]


def test_no_tokenizer_needed():
    assert Inkling2MultiModalProcessor.requires_tokenizer is False
