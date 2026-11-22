# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 image preprocessing and vLLM multimodal processing on SM70.

Port of upstream vllm/models/inkling2/processing.py (see PROVENANCE.md).
Differences: pixels are fp16, the cell size comes from the checkpoint's
``vision_config``, and the prompt-update target follows the renderer:

- ``inkling2_sm70`` tokenizer mode: the renderer emits one
  ``<|content_image|>`` per image; it becomes the marker followed by
  ``rows * cols`` copies of ``IMAGE_PAD_ID``.
- any other mode (the checkpoint's Jinja template): the template emits
  ``<|content_image|><|unused_200054|>`` for typed image parts, so only the
  pad is replaced, by ``rows * cols`` pads. Typed parts need
  ``--chat-template-content-format openai``.

Only the pad positions receive vision embeddings.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image
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

# <|content_image|>: block marker emitted by the renderer, kept verbatim.
IMAGE_MARKER_ID = 200005
IMAGE_PLACEHOLDER = "<|content_image|>"
# <|unused_200054|>: one per decoder image token, overwritten by embeddings.
IMAGE_PAD_ID = 200054

IMAGE_PAD = "<|unused_200054|>"

# Cell size of Inkling 2 Mini: native_patch_size 20 x adapter_merge_size 2.
PATCH_SIZE = 40
# Tokenizer mode of the plugin renderer, which emits the marker only.
NATIVE_RENDERER_MODE = "inkling2_sm70"
TARGET_MARKER = "marker"
TARGET_PAD = "pad"
# Dummy (profiling) images are at most this many pixels per side.
MAX_DUMMY_IMAGE_SIDE = 2048

ImageInput: TypeAlias = bytes | Image.Image | MediaWithBytes | np.ndarray


def _composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgba = rgba.astype(np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    blended = rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127
    return (blended // 255).astype(np.uint8)


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    if not image.has_transparency_data:
        return np.array(image.convert("RGB"), dtype=np.uint8)
    return _composite_on_checkerboard(np.array(image.convert("RGBA")))


def decode_image(image: ImageInput) -> np.ndarray:
    """Decode an image to HWC uint8 RGB, compositing alpha on a checkerboard.

    Encoded bytes (including `MediaWithBytes.original_bytes`) are preferred
    over decoded media, so no vLLM-side mode conversion or EXIF handling
    leaks into the pixels.
    """
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


class Inkling2ImageProcessor:
    """Pads each image to a `patch_size` grid and emits one patch per cell.

    Returns `vision_pixels` (fp16 `[sum(rows*cols), P, P, 3]` raw RGB
    values, padding pixels at -1; the encoder normalizes them) and
    `vision_grid_thw`.
    """

    def __init__(self, patch_size: int = PATCH_SIZE) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = patch_size

    def get_grid(self, width: int, height: int) -> tuple[int, int]:
        return math.ceil(height / self.patch_size), math.ceil(width / self.patch_size)

    def preprocess(self, images: Sequence[ImageInput]) -> dict[str, torch.Tensor]:
        patch = self.patch_size
        decoded = [torch.from_numpy(decode_image(image)) for image in images]
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
    """Keeps PIL image modes so transparency reaches the checkerboard step."""

    def _parse_image_data(self, data: Any) -> Any:
        if isinstance(data, Image.Image):
            return ImageProcessorItems([data])
        if (
            isinstance(data, Sequence)
            and data
            and all(isinstance(item, Image.Image) for item in data)
        ):
            return ImageProcessorItems(list(data))
        return super()._parse_image_data(data)


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
        return Inkling2ImageProcessor(self.patch_size)

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
        # Largest square image whose tokens still fit in seq_len.
        patch = self.info.patch_size
        side = min(MAX_DUMMY_IMAGE_SIDE, patch * max(1, math.isqrt(seq_len)))
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
