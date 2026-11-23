# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 expert activation fake-quant (emulation of the upstream A4/A8 GEMMs).

What: with ``sm70.inkling2.expert_act_fake_quant`` the expert input and the
  post-activation value of every expert GEMM are quantized and dequantized in
  fp32 and cast back to their dtype, as upstream quantizes them for its
  NVFP4 and MXFP8 GEMMs. NVFP4 layers: per-token amax / 1536, blocks of 16,
  the 6.0- or 4.0-based e4m3 block scale with the smaller absolute error
  (UP vllm/models/inkling2/nvidia/ops/nvfp4.py ``_quantize_row``). MXFP8
  layers: e4m3 values, a power-of-two scale rounded up per 32 values
  (UP ops/mxfp8.py ``_mxfp8_block``). The post-activation value is
  quantized without the route weight (UP ops/activation.py
  ``swiglu_quantize`` multiplies it in first); its per-token amax spans
  this rank's intermediate slice, as upstream's.
Kill-switch: off by default; configuration turns it on.

Every op is elementwise or a fixed-shape reduction: no host sync and no
data-dependent shape, so it runs inside a captured decode graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

import torch

from .expert_weights import MXFP8, MXFP8_BLOCK, NVFP4, NVFP4_BLOCK

FP32_MAX: Final = 3.4028234663852886e38
E4M3_MAX: Final = 448.0
NVFP4_GLOBAL: Final = 1536.0  # 6 (e2m1 max) * 256: the per-token scale's divisor
# Without a checkpoint format map: layers below this one are NVFP4, the rest MXFP8.
NVFP4_LAYERS: Final = 20
# e2m1 rounding bounds, high to low: (bound, value, bound inclusive).
_E2M1_STEPS: Final = (
    (5.0, 4.0, True),
    (3.5, 3.0, False),
    (2.5, 2.0, True),
    (1.75, 1.5, False),
    (1.25, 1.0, True),
    (0.75, 0.5, False),
    (0.25, 0.0, True),
)

__all__ = [
    "ActQuant",
    "apply",
    "install",
    "layer_format",
    "mxfp8_fake_quant",
    "nvfp4_fake_quant",
]


def e4m3_round(v: torch.Tensor) -> torch.Tensor:
    """Non-negative fp32 rounded to e4m3 (nearest even, subnormal step 2**-9,
    saturating at 448)."""
    _, e = torch.frexp(v)
    step = torch.exp2(((e - 1).clamp(min=-6) - 3).float())
    return (torch.round(v / step) * step).clamp(max=E4M3_MAX)


def e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """Non-negative fp32 to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} with
    upstream's tie rule (ops/nvfp4.py ``_e2m1``)."""
    out = torch.full_like(a, 6.0)
    for bound, value, inclusive in _E2M1_STEPS:
        below = a <= bound if inclusive else a < bound
        out = torch.where(below, torch.full_like(a, value), out)
    return out


def _pairwise_sum16(error: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim (16) in upstream's order (ops/nvfp4.py ``_block_error``)."""
    for _ in range(4):
        error = error[..., 0::2] + error[..., 1::2]
    return error[..., 0]


def nvfp4_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """fp32 [M, K] -> its NVFP4 round trip, per-token amax, blocks of 16."""
    m, k = x.shape
    amax = x.abs().amax(1, keepdim=True)
    encoder = (NVFP4_GLOBAL / amax).clamp(max=FP32_MAX)
    decoder = 1.0 / encoder
    outer = amax * (1.0 / NVFP4_GLOBAL)
    blocks = x.view(m, k // NVFP4_BLOCK, NVFP4_BLOCK)
    block_max = blocks.abs().amax(-1)
    candidates, errors = [], []
    for divisor in (6.0, 4.0):
        scale = e4m3_round(
            (block_max * (encoder * (1.0 / divisor))).clamp(max=E4M3_MAX)
        )
        inverse = (1.0 / (scale * decoder)).clamp(max=FP32_MAX)
        q = e2m1_round((blocks * inverse[..., None]).abs()) * torch.sign(blocks)
        rec = (q * scale[..., None]) * outer[..., None]
        candidates.append(rec)
        errors.append(_pairwise_sum16((rec - blocks).abs()))
    choose4 = errors[1] < errors[0]
    return torch.where(choose4[..., None], candidates[1], candidates[0]).view(m, k)


def mxfp8_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """fp32 [M, K] -> its MXFP8 round trip, blocks of 32."""
    m, k = x.shape
    blocks = x.view(m, k // MXFP8_BLOCK, MXFP8_BLOCK)
    raw = blocks.abs().amax(-1) * (1.0 / E4M3_MAX)
    mantissa, exponent = torch.frexp(raw)
    exponent = torch.where(mantissa > 0.5, exponent, exponent - 1)
    scale = torch.where(raw > 0, torch.exp2(exponent.float()), torch.ones_like(raw))
    values = (blocks / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
    return (e4m3_round(values.abs()) * torch.sign(values) * scale[..., None]).view(m, k)


_FAKE_QUANT = {
    NVFP4: (nvfp4_fake_quant, NVFP4_BLOCK),
    MXFP8: (mxfp8_fake_quant, MXFP8_BLOCK),
}


class ActQuant:
    """The fake-quant of one layer's expert activations: [M, K] in any float
    dtype -> a new tensor of that dtype, computed in fp32."""

    def __init__(self, fmt: str) -> None:
        if fmt not in _FAKE_QUANT:
            raise ValueError(f"Inkling 2: no activation fake-quant for {fmt!r}")
        self.format = fmt
        self.fn, self.block = _FAKE_QUANT[fmt]

    def check(self, *widths: int) -> None:
        """Raise ValueError unless every width splits into whole blocks."""
        for width in widths:
            if width % self.block:
                raise ValueError(
                    f"Inkling 2 {self.format} activation fake-quant needs widths "
                    f"that are multiples of {self.block}, got {width}"
                )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x.float()).to(x.dtype)

    def __repr__(self) -> str:
        return f"ActQuant({self.format!r})"


def apply(quant: ActQuant | None, x: torch.Tensor) -> torch.Tensor:
    """`x` itself when `quant` is None, else its fake-quant."""
    return x if quant is None else quant(x)


def layer_format(layer_idx: int, formats: Mapping[int, str] | None) -> str:
    """The checkpoint's expert format of a layer, else NVFP4 below
    ``NVFP4_LAYERS`` and MXFP8 from there on."""
    if formats and layer_idx in formats:
        return formats[layer_idx]
    return NVFP4 if layer_idx < NVFP4_LAYERS else MXFP8


def install(
    layers: Iterable[Any], formats: Mapping[int, str] | None, enabled: bool
) -> None:
    """Set ``layer.mlp.act_quant`` of every decoder layer (None when off)."""
    for layer in layers:
        quant = None
        if enabled:
            quant = ActQuant(layer_format(layer.layer_idx, formats))
            quant.check(layer.mlp.hidden_size, layer.mlp.local_intermediate)
        layer.mlp.act_quant = quant
