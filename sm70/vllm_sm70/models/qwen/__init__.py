# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 / 3.8 on SM70 (Mara).

M7 dflash2-fp16-range: ``dflash_sm70`` points the DFlash2 draft architecture
at a subclass that runs the BF16-trained draft in FP16 with BF16 range.
Kill-switch: VLLM_SM70_DFLASH2_EMULATION=0.
"""

_registered = False


def register() -> None:
    """Called by vllm_sm70.models.register(). Idempotent; raises on failure.

    Marks itself done only after every step succeeded, so a retry re-runs it.
    """
    global _registered
    if _registered:
        return
    from vllm_sm70.models.qwen import dflash_sm70

    dflash_sm70.register()
    _registered = True
