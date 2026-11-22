# SPDX-License-Identifier: Apache-2.0
"""Tests for transparent images on the M12 Inkling 2 serve path
(vllm_sm70/models/inkling2_sm70/processing.py, Inkling2MultiModalDataParser).

CPU only. Images go through vLLM's MediaConnector (data and http URLs) or
arrive as PIL images, then through ``parse_mm_data`` and
``BaseMultiModalProcessor.apply``, with and without the processor cache, as
the renderer runs them. The pixels the model receives must equal the
reference compositing (TF models/inkling2/image_processing_inkling2.py
``Inkling2ImageProcessor.convert_to_rgb`` and ``_preprocess``), whatever the
server's image ``media_io_kwargs``.
"""

import base64
import io
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import numpy as np
import torch
from PIL import Image
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.processing import (
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)

from vllm_sm70.models.inkling2_sm70.processing import (
    IMAGE_MARKER_ID,
    NATIVE_RENDERER_MODE,
    PATCH_SIZE,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)

WIDTH, HEIGHT = 70, 45
URL = "http://images.test/rgba.png"


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _rgba() -> Image.Image:
    rng = np.random.default_rng(12)
    pixels = rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
    pixels[:10, :, 3] = 0
    pixels[10:20, :, 3] = 255
    return Image.fromarray(pixels, "RGBA")


def _la() -> Image.Image:
    rng = np.random.default_rng(13)
    return Image.fromarray(
        rng.integers(0, 256, (HEIGHT, WIDTH, 2), dtype=np.uint8), "LA"
    )


def _palette_with_transparency() -> Image.Image:
    rng = np.random.default_rng(14)
    image = Image.fromarray(rng.integers(0, 16, (HEIGHT, WIDTH), dtype=np.uint8), "P")
    image.putpalette(rng.integers(0, 256, 16 * 3, dtype=np.uint8).tolist())
    image.info["transparency"] = 3
    return image


def _opaque() -> Image.Image:
    rng = np.random.default_rng(15)
    return Image.fromarray(
        rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8), "RGB"
    )


IMAGES = {"rgba": _rgba, "la": _la, "p_trns": _palette_with_transparency}


def _reference_pixels(png: bytes) -> torch.Tensor:
    """Transcription of the reference: decode the bytes, keep alpha, blend on
    the 8-pixel checkerboard, pad to the grid with -1, one patch per cell."""
    with Image.open(io.BytesIO(png)) as pil:
        image = pil.convert("RGBA" if pil.has_transparency_data else "RGB")
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    if image.shape[0] == 4:
        height, width = image.shape[-2:]
        rows = torch.arange(height)[:, None] // 8
        columns = torch.arange(width)[None, :] // 8
        background = torch.where((rows + columns) % 2 == 0, 204, 153)
        rgba = image.to(torch.int32)
        alpha = rgba[3:4]
        image = (rgba[:3] * alpha + background * (255 - alpha) + 127) // 255
    channels, height, width = image.shape
    rows, cols = -(-height // PATCH_SIZE), -(-width // PATCH_SIZE)
    padded = torch.full((channels, rows * PATCH_SIZE, cols * PATCH_SIZE), -1.0)
    padded[:, :height, :width] = image.float()
    patches = padded.reshape(channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
    return patches.permute(1, 3, 2, 4, 0).flatten(0, 1)


def _make_processor(media_io_kwargs=None):
    mm_config = MultiModalConfig(media_io_kwargs=media_io_kwargs or {})
    model_config = SimpleNamespace(
        model="inkling2",
        tokenizer_mode=NATIVE_RENDERER_MODE,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    info = Inkling2ProcessingInfo(InputProcessingContext(model_config, None))
    processor = Inkling2MultiModalProcessor(info, Inkling2DummyInputsBuilder(info))
    return processor, model_config


def _served_pixels(processor, images, cache=None) -> torch.Tensor:
    """What renderers/base.py ``_process_multimodal`` hands the model."""
    inputs = ProcessorInputs(
        prompt=[IMAGE_MARKER_ID] * len(images),
        mm_data_items=processor.info.parse_mm_data({"image": images}),
        cache=cache,
    )
    out = processor.apply(inputs, TimingContext(enabled=False))
    return out["mm_kwargs"].get_data()["vision_pixels"].float()


class _FakeConnection:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get_bytes(self, url, **kwargs) -> bytes:
        assert url == URL
        return self.payload


def _fetch(route: str, png: bytes, media_io_kwargs):
    if route == "data_url":
        return MediaConnector(media_io_kwargs).fetch_image(_data_url(png))
    connector = MediaConnector(media_io_kwargs, connection=_FakeConnection(png))
    return connector.fetch_image(URL)


MEDIA_IO_KWARGS = [
    None,
    {"image": {"rgba_background_color": [0, 0, 0]}},
    {"image": {"image_mode": None}},
]


@pytest.mark.parametrize("route", ["data_url", "http"])
@pytest.mark.parametrize("kind", sorted(IMAGES))
@pytest.mark.parametrize(
    "media_io_kwargs", MEDIA_IO_KWARGS, ids=["default", "black_bg", "keep_mode"]
)
@pytest.mark.parametrize("cached", [False, True])
def test_fetched_transparent_image_matches_reference(
    route, kind, media_io_kwargs, cached
):
    png = _png(IMAGES[kind]())
    processor, model_config = _make_processor(media_io_kwargs)
    cache = MultiModalProcessorOnlyCache(model_config) if cached else None
    expected = _reference_pixels(png)
    for _ in range(2 if cached else 1):
        image = _fetch(route, png, media_io_kwargs)
        served = _served_pixels(processor, [image], cache)
        assert torch.equal(served, expected)


def test_vllm_media_io_flattens_alpha_differently():
    # Guards the premise: vLLM's own decode composites on a solid color.
    png = _png(_rgba())
    image = MediaConnector().fetch_image(_data_url(png))
    assert image.media.mode == "RGB"
    flattened = torch.from_numpy(np.array(image.media)).float()
    top_left = _reference_pixels(png)[0]
    assert not torch.equal(flattened[:PATCH_SIZE, :PATCH_SIZE], top_left)


@pytest.mark.parametrize("kind", sorted(IMAGES))
@pytest.mark.parametrize("cached", [False, True])
def test_offline_pil_image_matches_reference(kind, cached):
    image = IMAGES[kind]()
    processor, model_config = _make_processor()
    cache = MultiModalProcessorOnlyCache(model_config) if cached else None
    expected = _reference_pixels(_png(image))
    for _ in range(2 if cached else 1):
        assert torch.equal(_served_pixels(processor, [image], cache), expected)


@pytest.mark.parametrize("route", ["data_url", "http", "pil"])
@pytest.mark.parametrize("cached", [False, True])
def test_opaque_rgb_is_unchanged(route, cached):
    image = _opaque()
    png = _png(image)
    processor, model_config = _make_processor()
    cache = MultiModalProcessorOnlyCache(model_config) if cached else None
    item = image if route == "pil" else _fetch(route, png, None)
    served = _served_pixels(processor, [item], cache)
    raw = torch.from_numpy(np.array(image)).float()
    cells = served.reshape(2, 2, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4)
    full = cells.reshape(2 * PATCH_SIZE, 2 * PATCH_SIZE, 3)
    assert torch.equal(full[:HEIGHT, :WIDTH], raw)
    assert (full[HEIGHT:] == -1).all() and (full[:, WIDTH:] == -1).all()


def test_mixed_batch_keeps_order_and_each_decoding():
    rgba, opaque = _rgba(), _opaque()
    processor, model_config = _make_processor()
    cache = MultiModalProcessorOnlyCache(model_config)
    fetched = MediaConnector().fetch_image(_data_url(_png(rgba)))
    served = _served_pixels(processor, [opaque, fetched], cache)
    expected = torch.cat(
        [_reference_pixels(_png(opaque)), _reference_pixels(_png(rgba))]
    )
    assert torch.equal(served, expected)
