# SPDX-License-Identifier: Apache-2.0
"""SM70 quantization configs (min capability 70) and linear/MoE methods (K03)."""

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent.

    Imports vllm_sm70.quant.configs (M2), which registers the SM70 quant
    configs on import, then runs its checks. A failure raises and leaves
    the hook unregistered, so a retry runs the checks again.
    The same for Inkling 2's "inkling2" method (M12), on any device.
    """
    global _registered
    if _registered:
        return
    from vllm_sm70.quant import configs

    configs.register()
    from vllm_sm70.models.inkling2_sm70 import quant_config as inkling2_quant

    inkling2_quant.register()
    _registered = True
