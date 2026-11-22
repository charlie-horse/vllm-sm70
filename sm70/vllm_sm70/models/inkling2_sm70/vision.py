# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 vision encoder in plain torch: a ViT with a CLS token per image
and axial RoPE, then a 2x2 merge projection to the decoder width.

Module names mirror the checkpoint, so `model.visual.<name>` loads into
`Inkling2VisionEncoder.<name>` unchanged.

Shapes: N image tokens (tiles), m merge size, p native patch size, W width.
One tile is `[m * p, m * p, 3]` channels-last; an image of `rows x cols` tiles
holds `rows * cols` tiles in raster order. Inside the encoder the patches of
an image are in block order: tile by tile, the `m * m` patches of a tile in
raster order, so the merge is a plain reshape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

CHECKPOINT_PREFIX = "model.visual."
ROPE_THETA = 10000.0


def normalize_pixels(raw: torch.Tensor) -> torch.Tensor:
    """Raw RGB values (0..255, padding -1) to fp32 `2 * v / 255 - 1`."""
    return raw.float() * (2.0 / 255.0) - 1.0


def tile_patches(pixels: torch.Tensor, merge: int, patch: int) -> torch.Tensor:
    """`[N, m*p, m*p, 3]` tiles to `[N*m*m, p*p*3]` patches in block order."""
    x = pixels.reshape(-1, merge, patch, merge, patch, pixels.shape[-1])
    return x.transpose(2, 3).reshape(-1, patch * patch * pixels.shape[-1])


def patch_positions(
    rows: int, cols: int, merge: int, device: torch.device | None = None
) -> torch.Tensor:
    """int64 `[rows*cols*m*m, 2]` (y, x) of each patch of one image, 0-based,
    in block order."""
    y = torch.arange(rows * merge, device=device).view(rows, 1, merge, 1)
    x = torch.arange(cols * merge, device=device).view(1, cols, 1, merge)
    shape = (rows, cols, merge, merge)
    return torch.stack((y.expand(shape), x.expand(shape)), dim=-1).reshape(-1, 2)


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float = ROPE_THETA
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 cos and sin `[L, head_dim]` of (y, x) positions `[L, 2]`: the
    width angles fill the first half of the head, the height angles the
    second, each angle repeated for one interleaved channel pair."""
    half = head_dim // 2
    exponent = torch.arange(0, half, 2, dtype=torch.float32) / half
    inv_freq = (1.0 / theta**exponent).to(positions.device)
    angles = positions[..., None].float() * inv_freq
    angles = torch.cat((angles[:, 1], angles[:, 0]), dim=-1)
    angles = angles.repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the interleaved channel pairs of `x` `[L, heads, head_dim]` in
    fp32; the result has the dtype of `x`."""
    y = x.float()
    even, odd = y[..., 0::2], y[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return (y * cos[:, None] + rotated * sin[:, None]).to(x.dtype)


def resample_position_table(
    table: torch.Tensor, grid: int, positions: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """fp32 rows `[L, W]` of the `[grid*grid, W]` position table resampled
    bilinearly to `height x width`, taken at (y, x) `positions`."""
    image = table.float().view(1, grid, grid, -1).permute(0, 3, 1, 2)
    image = F.interpolate(
        image, size=(height, width), mode="bilinear", align_corners=False
    )
    return image[0][:, positions[:, 0], positions[:, 1]].transpose(0, 1)


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    y = F.layer_norm(
        x.float(),
        norm.normalized_shape,
        norm.weight.float(),
        norm.bias.float(),
        norm.eps,
    )
    return y.to(x.dtype)


class Inkling2VisionAttention(nn.Module):
    def __init__(self, width: int, num_heads: int, **kw: Any) -> None:
        super().__init__()
        if width % num_heads or (width // num_heads) % 4:
            raise ValueError("vision head dim must be a multiple of 4")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q_proj = nn.Linear(width, width, **kw)
        self.k_proj = nn.Linear(width, width, **kw)
        self.v_proj = nn.Linear(width, width, **kw)
        self.out_proj = nn.Linear(width, width, **kw)

    def forward(
        self,
        hidden: torch.Tensor,
        lengths: list[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_chunk: int,
    ) -> torch.Tensor:
        shape = (-1, self.num_heads, self.head_dim)
        q = apply_rope(self.q_proj(hidden).view(shape), cos, sin)
        k = apply_rope(self.k_proj(hidden).view(shape), cos, sin)
        v = self.v_proj(hidden).view(shape)
        out = []
        # Each image attends only to itself: [1, heads, L, head_dim].
        for qi, ki, vi in zip(*(t.split(lengths) for t in (q, k, v))):
            qi, ki, vi = (t.transpose(0, 1).unsqueeze(0) for t in (qi, ki, vi))
            if qi.shape[2] <= query_chunk:
                o = F.scaled_dot_product_attention(qi, ki, vi)
            else:
                o = torch.cat(
                    [
                        F.scaled_dot_product_attention(part, ki, vi)
                        for part in qi.split(query_chunk, dim=2)
                    ],
                    dim=2,
                )
            out.append(o[0].transpose(0, 1))
        return self.out_proj(torch.cat(out).reshape(hidden.shape))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width: int, intermediate_size: int, **kw: Any) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, intermediate_size, **kw)
        self.c_proj = nn.Linear(intermediate_size, width, **kw)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width: int, **kw: Any) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, **kw))


class Inkling2VisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any], **kw: Any) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, **kw)
        self.self_attn = Inkling2VisionAttention(
            width, config["num_attention_heads"], **kw
        )
        self.ln_2 = nn.LayerNorm(width, eps=eps, **kw)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]), **kw)
        self.ls_1 = Inkling2VisionLayerScale(width, **kw)
        self.ls_2 = Inkling2VisionLayerScale(width, **kw)

    def forward(
        self,
        hidden: torch.Tensor,
        lengths: list[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_chunk: int,
    ) -> torch.Tensor:
        attn = self.self_attn(
            _layer_norm(hidden, self.ln_1), lengths, cos, sin, query_chunk
        )
        hidden = hidden + self.ls_1.gamma * attn
        return hidden + self.ls_2.gamma * self.mlp(_layer_norm(hidden, self.ln_2))


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any], **kw: Any) -> None:
        super().__init__()
        width, grid = config["hidden_size"], config["posemb_grid_size"]
        self.patch_embedding = nn.Linear(
            3 * config["native_patch_size"] ** 2, width, bias=False, **kw
        )
        self.position_embedding = nn.Parameter(torch.empty(grid * grid, width, **kw))
        self.class_embedding = nn.Parameter(torch.empty(width, **kw))
        self.cls_position_embedding = nn.Parameter(torch.empty(width, **kw))


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config: Mapping[str, Any], **kw: Any) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config, **kw)
        self.ln_pre = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"], **kw
        )
        self.layers = nn.ModuleList(
            Inkling2VisionLayer(config, **kw)
            for _ in range(config["num_hidden_layers"])
        )


class Inkling2VisionEncoder(nn.Module):
    """Maps raw `[N, m*p, m*p, 3]` tiles (0..255, padding -1) to `[N,
    text_hidden_size]` decoder embeddings, one row per tile, images in input
    order. Pixel normalization, the position resample, RoPE and the layer
    norms run in fp32; everything else in the parameter dtype.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        text_hidden_size: int | None = None,
        *,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        attn_query_chunk: int = 1024,
    ) -> None:
        super().__init__()
        kw = {"dtype": dtype, "device": device}
        self.patch_size = config["native_patch_size"]
        self.merge_size = config["adapter_merge_size"]
        self.posemb_grid_size = config["posemb_grid_size"]
        self.text_hidden_size = (
            config["text_hidden_size"] if text_hidden_size is None else text_hidden_size
        )
        # Queries per SDPA call for an image with more tokens than this.
        self.attn_query_chunk = attn_query_chunk
        self.backbone = Inkling2VisionBackbone(config, **kw)
        self.proj_in = nn.Linear(
            self.merge_size**2 * config["hidden_size"],
            config["projection_hidden_dim"],
            bias=config["projector_bias"],
            **kw,
        )
        self.proj_out = nn.Linear(
            config["projection_hidden_dim"],
            self.text_hidden_size,
            bias=config["projector_bias"],
            **kw,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.proj_out.weight.dtype

    def forward(
        self, vision_pixels: torch.Tensor, vision_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """`vision_pixels`: raw `[N, m*p, m*p, 3]`; `vision_grid_thw`: int
        `[images, 3]` rows of (1, rows, cols) with `sum(rows * cols) == N`."""
        return self.encode(normalize_pixels(vision_pixels), vision_grid_thw)

    def encode(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """As `forward`, for pixels already normalized to [-1, 1]."""
        m, side = self.merge_size, self.merge_size * self.patch_size
        grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        if pixels.shape[1:] != (side, side, 3):
            raise ValueError(
                f"expected [N, {side}, {side}, 3] tiles, got {pixels.shape}"
            )
        if any(t != 1 or h < 1 or w < 1 for t, h, w in grid):
            raise ValueError(f"expected (1, rows, cols) grids, got {grid}")
        if sum(h * w for _, h, w in grid) != pixels.shape[0]:
            raise ValueError(f"{pixels.shape[0]} tiles do not fill grids {grid}")
        if not grid:
            return pixels.new_empty((0, self.text_hidden_size), dtype=self.dtype)

        backbone, emb, device = self.backbone, self.backbone.embeddings, pixels.device
        patches = emb.patch_embedding(
            tile_patches(pixels.to(self.dtype), m, self.patch_size)
        )
        cls = (emb.class_embedding + emb.cls_position_embedding)[None]
        hidden, rope_positions, lengths = [], [], []
        for (_, h, w), image in zip(
            grid, patches.split([h * w * m * m for _, h, w in grid])
        ):
            positions = patch_positions(h, w, m, device)
            pos = resample_position_table(
                emb.position_embedding, self.posemb_grid_size, positions, h * m, w * m
            )
            hidden += [cls, image + pos.to(image.dtype)]
            # CLS sits at (0, 0); patches are 1-based.
            rope_positions += [positions.new_zeros(1, 2), positions + 1]
            lengths.append(positions.shape[0] + 1)
        hidden = _layer_norm(torch.cat(hidden), backbone.ln_pre)
        head_dim = backbone.layers[0].self_attn.head_dim if backbone.layers else 4
        cos, sin = rope_cos_sin(torch.cat(rope_positions), head_dim)
        for layer in backbone.layers:
            hidden = layer(hidden, lengths, cos, sin, self.attn_query_chunk)

        merged = torch.cat([image[1:] for image in hidden.split(lengths)])
        merged = merged.reshape(-1, m * m * hidden.shape[-1])
        return self.proj_out(F.gelu(self.proj_in(merged)))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint tensors named `model.visual.<parameter>` (the prefix
        is optional), cast to the parameter dtype. Raises on an unexpected,
        misshapen or missing tensor."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name = name.removeprefix(CHECKPOINT_PREFIX)
            if name not in params or name in loaded:
                raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
            if params[name].shape != weight.shape:
                raise ValueError(
                    f"Vision checkpoint tensor {name} has shape {tuple(weight.shape)}, "
                    f"expected {tuple(params[name].shape)}"
                )
            params[name].data.copy_(weight)
            loaded.add(name)
        if missing := sorted(params.keys() - loaded):
            raise ValueError(f"Missing vision checkpoint tensors: {missing}")
        return loaded
