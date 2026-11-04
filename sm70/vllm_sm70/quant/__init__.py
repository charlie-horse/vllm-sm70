# SPDX-License-Identifier: Apache-2.0
"""SM70 quantization configs (min capability 70) and linear/MoE methods (K03)."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True
