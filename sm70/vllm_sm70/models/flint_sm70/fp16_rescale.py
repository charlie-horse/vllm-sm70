# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 range handling for Flint on SM70 (Volta has no BF16). Every rescale is
by a power of two, so it only moves exponents. Two transforms keep FP16
activations finite:

* MoE intermediate rebalance. On massive-activation tokens, silu(gate) * up
  can exceed FP16. ``up`` is divided by 16 and ``down`` is multiplied by 16,
  so the expert output is unchanged.
* Residual-stream shift. From layer 46 on, the stream is carried divided by
  8. Layer 46 scales its residual after the attention add, later layers scale
  their attention output, and every MLP or MoE down projection from 46 on is
  divided by 8 at load. Every RMSNorm is invariant to the scale.

``load_weights`` applies them (``fp16_weight_pipeline``); decoder layers read
their scales from ``stream_scales``.
"""

from collections.abc import Iterable, Iterator

import regex as re
import torch

FP16_MAX = torch.finfo(torch.float16).max

MOE_INTERMEDIATE_SHIFT = 16
_MOE_UP = (".mlp.experts.", ".mlp.shared_expert.")

FP16_STREAM_SHIFT_LAYER = 46
FP16_STREAM_SHIFT = 8
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")

LM_HEAD_SCALE = "lm_head.weight_scale_2"


def stream_scales(layer_idx: int, enabled: bool) -> tuple[float | None, float | None]:
    """Return ``(residual_scale, attn_out_scale)`` for one decoder layer.

    The layer applies ``attn_out_scale`` to the attention output before the
    post-attention norm, and ``residual_scale`` to the residual after it.
    """
    if not enabled:
        return None, None
    if layer_idx == FP16_STREAM_SHIFT_LAYER:
        return 1.0 / FP16_STREAM_SHIFT, None
    if layer_idx > FP16_STREAM_SHIFT_LAYER:
        return None, 1.0 / FP16_STREAM_SHIFT
    return None, None


def rebalance_moe_intermediate(
    weights: Iterable[tuple[str, torch.Tensor]], shift: int = MOE_INTERMEDIATE_SHIFT
) -> Iterator[tuple[str, torch.Tensor]]:
    """Divide MoE up projections by ``shift`` and multiply the matching down
    projections by it.

    A power of two only moves exponents, so this is exact except where an
    E4M3 block scale or an FP16 value leaves the normal range. In NVFP4
    tensors it moves the up projection's E4M3 block scales (``weight_scale``,
    which keeps the gate/up global scales shared) and the down projection's
    FP32 global scale (``weight_scale_2``). In unquantized tensors it moves
    the weights themselves.
    """
    for name, tensor in weights:
        if any(block in name for block in _MOE_UP):
            dense = name.endswith(".weight") and tensor.is_floating_point()
            if ".up_proj." in name and (name.endswith(".weight_scale") or dense):
                tensor = (tensor.float() / shift).to(tensor.dtype)
            elif ".down_proj." in name and (name.endswith(".weight_scale_2") or dense):
                tensor = (tensor.float() * shift).to(tensor.dtype)
        yield name, tensor


def rescale_fp16_stream(
    weights: Iterable[tuple[str, torch.Tensor]],
    start_layer: int = FP16_STREAM_SHIFT_LAYER,
    shift: int = FP16_STREAM_SHIFT,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Divide the MLP/MoE down projections of layers ``>= start_layer`` by
    ``shift``, so their outputs join the rescaled residual stream.
    NVFP4 tensors move the FP32 global scale (``weight_scale_2``);
    unquantized tensors move the weights.
    """
    for name, tensor in weights:
        match = _LAYER_INDEX.search(name)
        if (
            match is not None
            and int(match.group(1)) >= start_layer
            and ".mlp." in name
            and ".down_proj." in name
            and (
                name.endswith(".weight_scale_2")
                or (name.endswith(".weight") and tensor.is_floating_point())
            )
        ):
            tensor = (tensor.float() / shift).to(tensor.dtype)
        yield name, tensor


def check_fp16_range(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Reject BF16/FP32 tensors that would overflow when cast to FP16."""
    for name, tensor in weights:
        if tensor.dtype in (torch.bfloat16, torch.float32) and tensor.numel():
            amax = tensor.detach().abs().amax()
            if not bool(amax <= FP16_MAX):
                raise ValueError(
                    f"{name}: max |x| = {float(amax)} does not fit in FP16 "
                    f"(max {FP16_MAX})"
                )
        yield name, tensor


def fp16_weight_pipeline(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """The three FP16 filters, in order."""
    weights = rebalance_moe_intermediate(weights, MOE_INTERMEDIATE_SHIFT)
    weights = rescale_fp16_stream(weights, FP16_STREAM_SHIFT_LAYER, FP16_STREAM_SHIFT)
    return check_fp16_range(weights)


class LmHeadScale:
    """Captures ``lm_head.weight_scale_2`` from the weight stream and applies
    it after loading.

    The NVFP4 checkpoint stores lm_head dequantized without its global
    scale, which comes separately as a scalar. With BF16/FP32 weights the
    scale multiplies the logits, as upstream does. FP16 can't hold the unscaled
    logits (they are the true logits divided by a scale of order 1e-4), so in
    FP16 the scale is folded into the weight.
    """

    def __init__(self) -> None:
        self.scale: float | None = None

    def intercept(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name == LM_HEAD_SCALE:
                self.scale = float(tensor)
            else:
                yield name, tensor

    def apply(self, lm_head: torch.nn.Module | None, logits_processor) -> None:
        # Pass lm_head=None on non-last PP ranks.
        if self.scale is None or lm_head is None:
            return
        weight = lm_head.weight
        if weight.dtype != torch.float16:
            logits_processor.scale = self.scale
            return
        with torch.no_grad():
            for rows in weight.data.split(8192):
                folded = rows.float().mul_(self.scale)
                if not bool(folded.abs().amax() <= FP16_MAX):
                    raise ValueError(
                        "lm_head.weight times lm_head.weight_scale_2 exceeds "
                        "the FP16 range"
                    )
                rows.copy_(folded)
