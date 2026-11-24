# SPDX-License-Identifier: Apache-2.0
"""SM70 fused-MoE methods and router."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True
    from vllm_sm70 import router

    router.register()  # flint_route fake (provider binds via the platform)
