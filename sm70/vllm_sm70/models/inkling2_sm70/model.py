# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (looped) on SM70: the registered model class.

The class only fixes the architecture name and import path; constructing it
raises. It has no vLLM model interface, so vLLM rejects it at ModelConfig
validation before it constructs it.
"""

from __future__ import annotations

from typing import Any

from torch import nn


class Inkling2LoopSm70ForConditionalGeneration(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "vllm_sm70: Inkling2LoopSm70ForConditionalGeneration has no model "
            "body in this build of vllm-sm70; it can't be served."
        )
