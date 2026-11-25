# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 image preprocessing and vLLM multimodal processing on SM70.

Port of upstream vllm/models/inkling2/processing.py (see PROVENANCE.md): fp16
pixels, the cell size from ``vision_config``, a prompt target per renderer:

- ``inkling2_sm70`` tokenizer mode: the renderer emits one
  ``<|content_image|>`` per image; it becomes the marker followed by
  ``rows * cols`` copies of ``IMAGE_PAD_ID``.
- any other mode (the checkpoint's Jinja template, which needs
  ``--chat-template-content-format openai``): the template emits
  ``<|content_image|><|unused_200054|>``; only the pad is replaced, by
  ``rows * cols`` pads. A pad count other than the image count is rejected.
Only the pad positions receive vision embeddings. An image over
``Inkling2ProcessingInfo.max_image_tokens`` is scaled down to fit, keeping
its shape (kill switch ``VLLM_SM70_INKLING_IMAGE_RESIZE``; off, it raises
ValueError).
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers.feature_extraction_utils import BatchFeature
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from vllm_sm70 import envs
from vllm_sm70.config import Inkling2Config

from .sm70_configs import IMAGE_ENCODER_TOKEN_CAP_ATTR

# <|content_image|>: block marker emitted by the renderer, kept verbatim.
IMAGE_MARKER_ID = 200005
IMAGE_PLACEHOLDER = "<|content_image|>"
# <|unused_200054|>: one per decoder image token, overwritten by embeddings.
IMAGE_PAD_ID = 200054

IMAGE_PAD = "<|unused_200054|>"

# Cell size of Inkling 2 Mini: native_patch_size 20 x adapter_merge_size 2.
PATCH_SIZE = 40
MERGE_SIZE = 2
# Tokenizer mode of the plugin renderer, which emits the marker only.
NATIVE_RENDERER_MODE = "inkling2_sm70"
TARGET_MARKER = "marker"
TARGET_PAD = "pad"

ImageInput: TypeAlias = bytes | Image.Image | MediaWithBytes | np.ndarray


def _composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgba = rgba.astype(np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    blended = rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127
    return (blended // 255).astype(np.uint8)


EXIF_SWITCH = "VLLM_SM70_INKLING_IMAGE_EXIF"


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    # Turn the pixels upright as the EXIF Orientation tag says (phone photos).
    if envs.environment_variables()[EXIF_SWITCH]():
        image = ImageOps.exif_transpose(image)
    if not image.has_transparency_data:
        return np.array(image.convert("RGB"), dtype=np.uint8)
    return _composite_on_checkerboard(np.array(image.convert("RGBA")))


class Inkling2DecodedImage(MediaWithBytes[Image.Image]):
    """RGB media decoded from ``original_bytes`` by ``decode_image``.

    vLLM's ``ImageMediaIO`` flattens transparency on a solid color before the
    processor runs, and its processor cache re-parses the unwrapped media, so
    the parser swaps that media for the reference decoding of the same bytes.
    """


def decode_image(image: ImageInput) -> np.ndarray:
    """Decode an image to HWC uint8 RGB, compositing alpha on a checkerboard.

    Encoded bytes (including `MediaWithBytes.original_bytes`) are preferred
    over decoded media, so no vLLM-side mode conversion leaks into the
    pixels. The EXIF Orientation tag is applied (kill switch
    `VLLM_SM70_INKLING_IMAGE_EXIF`).
    """
    if isinstance(image, Inkling2DecodedImage):
        return np.array(image.media, dtype=np.uint8)
    if isinstance(image, MediaWithBytes):
        image = image.original_bytes
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if data.startswith(b"\xff\xd8") and not data.endswith(b"\xff\xd9"):
            data += b"\xff\xd9"
        with Image.open(io.BytesIO(data)) as pil:
            pil.load()
            return _pil_to_rgb(pil)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return _pil_to_rgb(image)


def _reference_item(item: Any) -> Any:
    if (
        not isinstance(item, MediaWithBytes)
        or isinstance(item, Inkling2DecodedImage)
        or not isinstance(item.media, Image.Image)
    ):
        return item
    rgb = Image.fromarray(decode_image(item.original_bytes), "RGB")
    # Pixels depend on the bytes alone, so they alone key the caches.
    return Inkling2DecodedImage(rgb, item.original_bytes)


RESIZE_SWITCH = "VLLM_SM70_INKLING_IMAGE_RESIZE"


class Inkling2ImageProcessor:
    """Pads each image to a `patch_size` grid and emits one patch per cell.

    Returns `vision_pixels` (fp16 `[sum(rows*cols), P, P, 3]` raw RGB
    values, padding pixels at -1; the encoder normalizes them) and
    `vision_grid_thw`. With `max_tokens`, an image of more cells is scaled
    down (Lanczos filter, same shape) until it fits; with the kill switch
    `VLLM_SM70_INKLING_IMAGE_RESIZE` at "0" it raises ValueError instead.
    """

    def __init__(
        self, patch_size: int = PATCH_SIZE, max_tokens: int | None = None
    ) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.patch_size = patch_size
        self.max_tokens = max_tokens

    def check_size(self, width: int, height: int) -> None:
        rows, cols = self.get_grid(width, height)
        if self.max_tokens is not None and rows * cols > self.max_tokens:
            patch = self.patch_size
            raise ValueError(
                f"Image of {width}x{height} pixels is too large: it takes "
                f"{rows * cols} image tokens ({rows} x {cols} cells of {patch} "
                f"pixels) and this server accepts at most {self.max_tokens} "
                "per image. Images are not resized; send a smaller one."
            )

    def fit_size(self, width: int, height: int) -> tuple[int, int]:
        """The largest size of the same shape whose grid fits `max_tokens`."""
        rows, cols = self.get_grid(width, height)
        if self.max_tokens is None or rows * cols <= self.max_tokens:
            return width, height
        area = self.max_tokens * self.patch_size**2
        scale = min(1.0, math.sqrt(area / (width * height)))
        while True:
            new_w = max(1, math.floor(width * scale))
            new_h = max(1, math.floor(height * scale))
            rows, cols = self.get_grid(new_w, new_h)
            if rows * cols <= self.max_tokens:
                return new_w, new_h
            scale *= 0.99

    def fit(self, pixels: np.ndarray) -> np.ndarray:
        """`pixels` (HWC uint8) scaled down to fit `max_tokens` if needed."""
        height, width = pixels.shape[:2]
        new_w, new_h = self.fit_size(width, height)
        if (new_w, new_h) == (width, height):
            return pixels
        if not envs.environment_variables()[RESIZE_SWITCH]():
            self.check_size(width, height)
        image = Image.fromarray(pixels, "RGB")
        return np.array(image.resize((new_w, new_h), Image.Resampling.LANCZOS))

    def get_grid(self, width: int, height: int) -> tuple[int, int]:
        return math.ceil(height / self.patch_size), math.ceil(width / self.patch_size)

    def preprocess(self, images: Sequence[ImageInput]) -> dict[str, torch.Tensor]:
        patch = self.patch_size
        decoded = []
        for image in images:
            pixels = torch.from_numpy(self.fit(decode_image(image)))
            self.check_size(pixels.shape[1], pixels.shape[0])
            decoded.append(pixels)
        grids = [self.get_grid(pixels.shape[1], pixels.shape[0]) for pixels in decoded]
        out = torch.empty(
            (sum(rows * cols for rows, cols in grids), patch, patch, 3),
            dtype=torch.float16,
        )
        start = 0
        for pixels, (rows, cols) in zip(decoded, grids):
            height, width, _ = pixels.shape
            if (height, width) != (rows * patch, cols * patch):
                padded = pixels.new_full(
                    (rows * patch, cols * patch, 3), -1, dtype=torch.float16
                )
                padded[:height, :width] = pixels
                pixels = padded
            cells = pixels.view(rows, patch, cols, patch, 3)
            out[start : start + rows * cols].view(rows, cols, patch, patch, 3).copy_(
                cells.permute(0, 2, 1, 3, 4)
            )
            start += rows * cols
        return {
            "vision_pixels": out,
            "vision_grid_thw": torch.tensor(
                [[1, rows, cols] for rows, cols in grids], dtype=torch.int64
            ),
        }


class Inkling2MultiModalDataParser(MultiModalDataParser):
    """Gives the processor the reference decoding of every image.

    PIL images keep their mode, so transparency reaches the checkerboard
    step; fetched images (``MediaWithBytes``) become ``Inkling2DecodedImage``.
    """

    def _parse_image_data(self, data: Any) -> Any:
        if data is None or self.is_embeddings(data):
            return super()._parse_image_data(data)
        if isinstance(data, (Image.Image, MediaWithBytes)):
            items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            return super()._parse_image_data(data)
        else:
            items = list(data)
        return ImageProcessorItems([_reference_item(item) for item in items])


class Inkling2ProcessingInfo(BaseProcessingInfo):
    @cached_property
    def patch_size(self) -> int:
        """Pixels per decoder image token side, from `vision_config`."""
        hf_config = getattr(self.ctx.model_config, "hf_config", None)
        vision = getattr(hf_config, "vision_config", None)
        if vision is None:
            return PATCH_SIZE
        if not isinstance(vision, Mapping):
            vision = vision.to_dict()
        return int(vision["native_patch_size"]) * int(vision["adapter_merge_size"])

    @cached_property
    def max_image_tokens(self) -> int:
        """Most image tokens one image may take: the encoder sequence cap
        (`Sm70Config.inkling2.max_image_encoder_tokens`, which the
        engine-config hook leaves on `hf_config`) less the class token, in
        cells of merge * merge patches."""
        hf_config = getattr(self.ctx.model_config, "hf_config", None)
        cap = getattr(hf_config, IMAGE_ENCODER_TOKEN_CAP_ATTR, None)
        if cap is None:
            cap = Inkling2Config().max_image_encoder_tokens
        vision = getattr(hf_config, "vision_config", None)
        merge = MERGE_SIZE
        if vision is not None:
            if not isinstance(vision, Mapping):
                vision = vision.to_dict()
            merge = int(vision["adapter_merge_size"])
        tokens = (int(cap) - 1) // merge**2
        if tokens < 1:
            raise ValueError(
                f"sm70.inkling2.max_image_encoder_tokens={cap} leaves no room "
                f"for one image token ({merge * merge} patches and the class "
                "token)."
            )
        return tokens

    @cached_property
    def image_target(self) -> str:
        """Which prompt token the processor expands, by renderer."""
        mode = getattr(self.ctx.model_config, "tokenizer_mode", None)
        return TARGET_MARKER if mode == NATIVE_RENDERER_MODE else TARGET_PAD

    @cached_property
    def image_prompt_ids(self) -> list[int]:
        """The token ids the renderer emits for one image."""
        if self.image_target == TARGET_MARKER:
            return [IMAGE_MARKER_ID]
        return [IMAGE_MARKER_ID, IMAGE_PAD_ID]

    @cached_property
    def image_processor(self) -> Inkling2ImageProcessor:
        return Inkling2ImageProcessor(self.patch_size, self.max_image_tokens)

    def get_data_parser(self) -> MultiModalDataParser:
        return Inkling2MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {}


class Inkling2DummyInputsBuilder(BaseDummyInputsBuilder[Inkling2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        per_image = IMAGE_PLACEHOLDER
        if self.info.image_target == TARGET_PAD:
            per_image += IMAGE_PAD
        return per_image * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        # Largest square image whose tokens fit in seq_len and the image cap.
        cells = math.isqrt(min(seq_len, self.info.max_image_tokens))
        side = self.info.patch_size * max(1, cells)
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class Inkling2MultiModalProcessor(BaseMultiModalProcessor[Inkling2ProcessingInfo]):
    # Prompts arrive as token ids and the update target is a single token id.
    requires_tokenizer = False

    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> ProcessorInputs:
        # Build marker ids directly: the tokenizer may not parse special text.
        mm_data = self.dummy_inputs.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        return ProcessorInputs(
            prompt=self.info.image_prompt_ids * mm_counts.get("image", 0),
            mm_data_items=self.info.parse_mm_data(mm_data, validate=False),
        )

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )
        images = mm_items.get("image")
        processed = None
        if isinstance(images, ImageProcessorItems) and images.get_count():
            # Raw items keep MediaWithBytes.original_bytes for exact decoding.
            processed = BatchFeature(self.info.image_processor.preprocess(images.data))
        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, processed
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thw = hf_inputs.get("vision_grid_thw", torch.empty((0, 3)))
        return {
            "vision_pixels": MultiModalFieldConfig.flat_from_sizes(
                "image", grid_thw.prod(-1)
            ),
            "vision_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _apply_prompt_updates(self, token_ids, mm_prompt_updates):
        if self.info.image_target == TARGET_PAD:
            # Template mode parses special-token text, so prompt text can
            # spell the pad; an extra pad would take an image's embeddings.
            num_pads = token_ids.count(IMAGE_PAD_ID)
            num_images = len(mm_prompt_updates.get("image", ()))
            if num_pads != num_images:
                raise ValueError(
                    f"The prompt has {num_pads} {IMAGE_PAD} tokens for "
                    f"{num_images} images; text must not contain {IMAGE_PAD}."
                )
        return super()._apply_prompt_updates(token_ids, mm_prompt_updates)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_items = out_mm_kwargs.get("image", [])
        target = (
            [IMAGE_MARKER_ID]
            if self.info.image_target == TARGET_MARKER
            else [IMAGE_PAD_ID]
        )

        def replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_items[item_idx]
            assert item is not None
            grid_thw = item["vision_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod())
            pads = [IMAGE_PAD_ID] * num_tokens
            if target == [IMAGE_PAD_ID]:
                return PromptUpdateDetails.from_seq(pads)
            return PromptUpdateDetails.select_token_id(
                [IMAGE_MARKER_ID] + pads, IMAGE_PAD_ID
            )

        return [
            PromptReplacement(modality="image", target=target, replacement=replacement)
        ]
