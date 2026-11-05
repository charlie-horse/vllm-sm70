# SPDX-License-Identifier: Apache-2.0
"""SM70 gated-delta-net kernels and layer (K08). kernels.py: Kai; layer.py: Mara."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True
