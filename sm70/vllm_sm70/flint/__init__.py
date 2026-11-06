# SPDX-License-Identifier: Apache-2.0
"""Flint kernels on SM70 (Kai). K15: fused q/k RMSNorm + RoPE + KV write.

This module stays light (no Triton, no vLLM model imports) because the
platform hook imports it during ``check_and_update_config``. The kernel and
its custom op live in ``vllm_sm70.flint.qk_norm_rope_kv``, which the SM70
Flint model imports.
"""

from __future__ import annotations

from typing import Any

# Registered in the "vllm" torch library by qk_norm_rope_kv.py. Prefixed so
# it cannot clash with an upstream op; the fork name was
# vllm::qk_norm_rope_kv_cache_update (FORK qk_norm_rope_kv.py:27-30).
QK_ROPE_KV_OP_NAME = "sm70_qk_norm_rope_kv_cache_update"
# Splitting-op form. The op takes the layer name, like unified_kv_cache_update,
# so it is split out of the compiled graph next to it.
QK_ROPE_KV_SPLITTING_OP = f"vllm::{QK_ROPE_KV_OP_NAME}"
_KV_UPDATE_OP = "vllm::unified_kv_cache_update"


def install_splitting_op(compilation_config: Any, *, enabled: bool) -> bool:
    """Add QK_ROPE_KV_SPLITTING_OP to splitting_ops wherever upstream splits
    out ``vllm::unified_kv_cache_update``. Returns True when it is present.

    Port of FORK config/compilation.py:1172-1174. The fork appended inside
    set_splitting_ops_for_v1; upstream calls that after the platform hook
    (UP config/vllm.py:2067 hook, :2161 set_splitting_ops_for_v1). So when
    splitting_ops is still None and upstream would take its default branch
    (UP compilation.py:1175-1213), the default list is built here with
    upstream's own method; the later call keeps a non-None list as is.
    """
    if not enabled:
        return False
    from vllm.config.compilation import CompilationMode

    cc = compilation_config
    if cc.mode != CompilationMode.VLLM_COMPILE:
        return False
    if cc.splitting_ops is None:
        if cc.use_inductor_graph_partition or cc.pass_config.fuse_attn_quant:
            # Upstream does not split the KV update there either.
            return False
        # all2all_backend="" skips the DeepEP cudagraph check here; upstream's
        # own call later runs it with the real backend.
        cc.set_splitting_ops_for_v1(all2all_backend="", data_parallel_size=1)
    ops = cc.splitting_ops
    if ops is None or _KV_UPDATE_OP not in ops:
        return False
    if QK_ROPE_KV_SPLITTING_OP not in ops:
        ops.append(QK_ROPE_KV_SPLITTING_OP)
    return True
