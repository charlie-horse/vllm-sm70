# SPDX-License-Identifier: Apache-2.0
"""SM70 gated-delta-net kernels and layer (K08). kernels.py: Kai; layer.py: Mara."""

import logging

logger = logging.getLogger("vllm.sm70")

_registered = False


def register() -> None:
    """Area hook, called by vllm_sm70.register(). Idempotent.

    Without _sm70_C, or with a build that has no GDN ops, nothing is
    imported and vLLM keeps its upstream GDN kernels.
    """
    global _registered
    if _registered:
        return
    _registered = True
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        logger.debug("vllm_sm70.gdn: _sm70_C not loaded; FlashQLA GDN skipped")
        return
    from vllm_sm70.gdn import kernels  # registers the fakes

    if not kernels.has_op("gdn_forward_vlk_varlen_out"):
        logger.debug("vllm_sm70.gdn: _sm70_C has no GDN ops (built without K08)")
        return
    kernels.register_chunk_gated_delta_rule()
