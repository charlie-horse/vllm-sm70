# SPDX-License-Identifier: Apache-2.0
"""SM70 custom ops: Python wrappers and fake impls over torch.ops._sm70_C (K02+)."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True
