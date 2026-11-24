# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: load-time checks.

* ``Fp16Audit`` checks every cast to fp16. A value above the fp16 range
  raises with the tensor name. So does a nonzero value that rounds to zero
  and exceeds ``FLUSH_RELATIVE`` times the tensor's largest |w|, and a
  nonzero tensor whose largest |w| is below the smallest normal fp16 value.
  Other values that round to zero, and subnormals, are counted; a tensor
  with more than ``SUBNORMAL_WARN_SHARE`` subnormal values is logged.
* ``check_hard_coded_config`` refuses a config whose values differ from the
  ones the model code is written for.
* ``walk_process_weights_after_loading`` runs the post-load hook of every
  submodule; vLLM itself calls only attention layers and the top model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger("vllm.sm70.inkling2.fp16_audit")

__all__ = [
    "FLUSH_RELATIVE",
    "FP16_MAX",
    "FP16_MIN_NORMAL",
    "HARD_CODED_CONFIG",
    "SUBNORMAL_WARN_SHARE",
    "Fp16Audit",
    "check_hard_coded_config",
    "walk_process_weights_after_loading",
]

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14
# Largest |w| that rounds to zero in fp16 (half the smallest subnormal).
FP16_FLUSH_MAX = 2.0**-25
FLUSH_RELATIVE = 2.0**-10
SUBNORMAL_WARN_SHARE = 0.01

# Text-config values the model code assumes. ``is_varloop`` is a field of
# ``loop_config``.
HARD_CODED_CONFIG: dict[str, Any] = {
    "attn_res_normalize_values": True,
    "use_value_norm_no_gamma": True,
    "use_gate_bias": True,
    "n_shared_experts": 0,
    "q_bias": False,
    "o_bias": False,
    "is_varloop": False,
}

_MISSING = object()


class Fp16Audit:
    """Running totals of one model load."""

    def __init__(self) -> None:
        self.tensors = 0
        self.flushed = 0
        self.subnormal = 0
        self.max_abs = 0.0
        self.max_name: str | None = None

    def check(
        self, name: str, source: torch.Tensor, cast: torch.Tensor | None = None
    ) -> None:
        """Check ``source`` (the values before the cast) against fp16.
        ``cast`` is its fp16 copy when the caller has one; a ``cast`` of
        another dtype is not checked."""
        if cast is not None and cast.dtype != torch.float16:
            return
        if source.numel() == 0:
            return
        low, high = torch.aminmax(source.detach())
        amax = float(torch.maximum(low.abs(), high.abs()))
        if not amax <= FP16_MAX:
            raise ValueError(
                f"{name}: max |w| = {amax} does not fit in fp16 (max {FP16_MAX})"
            )
        if cast is None:
            cast = source.detach().to(torch.float16)
        zeros = cast.numel() - int(torch.count_nonzero(cast))
        flushed = zeros - (source.numel() - int(torch.count_nonzero(source)))
        if flushed and amax * FLUSH_RELATIVE < FP16_FLUSH_MAX:
            lost = source.detach()[(source != 0) & (cast == 0)].abs().amax()
            if float(lost) > amax * FLUSH_RELATIVE:
                raise ValueError(
                    f"{name}: {flushed} nonzero values up to |w| = "
                    f"{float(lost):g} round to zero in fp16 and the largest "
                    f"|w| is {amax:g}"
                )
        if 0.0 < amax < FP16_MIN_NORMAL:
            raise ValueError(
                f"{name}: max |w| = {amax:g} is below the smallest normal fp16 "
                f"value {FP16_MIN_NORMAL:g}; every value is subnormal or zero "
                "in fp16"
            )
        self.tensors += 1
        self.flushed += flushed
        subnormal = int((cast.abs() < FP16_MIN_NORMAL).sum()) - zeros
        self.subnormal += subnormal
        if subnormal > SUBNORMAL_WARN_SHARE * cast.numel():
            logger.warning(
                "fp16 audit: %s has %d of %d values (%.2f%%) subnormal in fp16.",
                name,
                subnormal,
                cast.numel(),
                100.0 * subnormal / cast.numel(),
            )
        if amax > self.max_abs:
            self.max_abs, self.max_name = amax, name

    def summary(self) -> str:
        return (
            f"fp16 audit: {self.tensors} tensors, max |w| {self.max_abs:g} "
            f"({self.max_name}), {self.flushed} values rounded to zero, "
            f"{self.subnormal} subnormal"
        )


def check_hard_coded_config(config: Any) -> None:
    """Raise ValueError when ``config`` (the model config or its text config)
    differs from ``HARD_CODED_CONFIG``."""
    get_text_config = getattr(config, "get_text_config", None)
    text = get_text_config() if callable(get_text_config) else config
    loop = getattr(text, "loop_config", None) or {}
    problems = []
    for key, want in HARD_CODED_CONFIG.items():
        got = (
            loop.get(key, want) if key == "is_varloop" else getattr(text, key, _MISSING)
        )
        if got is _MISSING or type(got) is not type(want) or got != want:
            shown = "missing" if got is _MISSING else repr(got)
            problems.append(f"{key}={shown} (supported: {want!r})")
    if problems:
        raise ValueError(
            "vllm_sm70: Inkling 2 config values the SM70 model does not "
            "implement: " + ", ".join(problems)
        )


def walk_process_weights_after_loading(model: nn.Module) -> list[str]:
    """Call ``process_weights_after_loading()`` on every submodule of
    ``model`` that defines it, children before parents; returns their names.
    Skipped: ``model`` itself and the attention layers vLLM calls with a
    dtype argument."""
    from vllm.model_executor.layers.attention import is_deferred_attention_layer

    reached = []
    for name, module in reversed(list(model.named_modules())):
        if module is model or is_deferred_attention_layer(module):
            continue
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook):
            hook()
            reached.append(name)
    return reached
