# SPDX-License-Identifier: Apache-2.0
"""SM70 custom ops: Python wrappers and fake impls over torch.ops._sm70_C.

K02/K03: TurboMind (tm.py). K05: Flash-V100 attention (fv100.py).
K12: Flint NVFP4 MoE M<=2 QPN decode (flint_moe.py).
K18: DFlash2 draft FP16 M<=8 GEMM + compact FP16 top-64 (dflash2.py).
K16: top-k candidate sampler (sampler.py).
"""

import logging

logger = logging.getLogger("vllm.sm70")

_registered = False


def _sm70_c_loaded() -> bool:
    # `hasattr(torch.ops, "_sm70_C")` alone is always True: torch.ops creates
    # a namespace object for any attribute name. The namespace only has ops
    # once vllm_sm70._sm70_C is imported, which _ext.load() does.
    import torch

    from vllm_sm70 import _ext

    return hasattr(torch.ops, "_sm70_C") and _ext.is_loaded()


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent.

    Without _sm70_C (CPU-only build, VLLM_SM70_SKIP_EXT=1) nothing is
    imported, so the upstream paths stay in use.
    """
    global _registered
    if _registered:
        return
    _registered = True
    if not _sm70_c_loaded():
        logger.debug("vllm_sm70.ops: _sm70_C not loaded; TurboMind ops skipped")
        return
    from vllm_sm70.ops import tm  # noqa: F401  (registers the fakes)
    from vllm_sm70.ops import fv100  # noqa: F401  (K05: Flash-V100 fakes)
    from vllm_sm70.ops import qpn8  # noqa: F401  (K09: QPN8 fakes + linear op)
    from vllm_sm70.ops import flint_moe  # noqa: F401  (K12: decode fakes)
    from vllm_sm70.ops import dflash2  # noqa: F401  (K18: DFlash2 fakes)
    from vllm_sm70.ops import sampler  # noqa: F401  (K16: sampler fakes)
