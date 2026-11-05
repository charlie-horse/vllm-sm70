# SPDX-License-Identifier: Apache-2.0
"""SM70 attention backends (Flash-V100, K05-K07)."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True
