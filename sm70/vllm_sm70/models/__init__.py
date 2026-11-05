# SPDX-License-Identifier: Apache-2.0
"""SM70 model registrations: flint_sm70/, qwen/."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent.

    A failure raises and leaves the hook unregistered, so a retry runs the
    sub-registrations again.
    """
    global _registered
    if _registered:
        return
    from vllm_sm70.models import qwen

    qwen.register()
    _registered = True
