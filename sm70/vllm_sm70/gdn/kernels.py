# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashQLA SM70 GDN kernels (K08): wrappers, fakes and the prefill route.

Ported from fork cf9492a:
- flash_qla/ops/gated_delta_rule/chunk/sm70/fused_fwd.py
  (chunk_gated_delta_rule_fwd_sm70_vlk_varlen, :269-322;
  gdn_decode_mixed_qkv_global_state_sm70, :325-417)
- vllm/model_executor/layers/fla/ops/sm70/gdn_prefill.py
  (flashqla_sm70_chunk_gated_delta_rule, non-TileLang branch)
- vllm/model_executor/layers/fla/ops/sm70/gdn_decode.py
  (flashqla_decode_rejection / flashqla_decode)
The pybind11 extension, its loader/JIT fallback, the native policy object,
the TileLang branch (K10) and the profiler are gone; the ops live in
torch.ops._sm70_C (sm70/csrc/gdn/gdn_forward.cu).

Routes:
- Prefill: `register_chunk_gated_delta_rule()` installs an out-of-tree
  `ChunkGatedDeltaRule` (CustomOp.register_oot, UP custom_op.py:234) whose
  forward runs the FlashQLA varlen kernel on SM70. Kill switch
  VLLM_SM70_GDN_FLASHQLA_PREFILL=0, read when the op is constructed.
- Non-spec decode: `flashqla_decode(...)`, called by the GDN layer subclass
  (gdn/layer.py) from `_forward_core_decode_non_spec`. Kill switch
  VLLM_SM70_GDN_DECODE_FLASHQLA=0, read on every call.
Both fall back to upstream (Triton/FLA) when a tensor contract is not met.
"""

from __future__ import annotations

import logging
import os

import torch

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

logger = logging.getLogger("vllm.sm70")

NAMESPACE = "_sm70_C"
PREFILL_KILL_SWITCH = "VLLM_SM70_GDN_FLASHQLA_PREFILL"
DECODE_KILL_SWITCH = "VLLM_SM70_GDN_DECODE_FLASHQLA"

# Load _sm70_C before any has_op() check below (see ops/tm.py).
from vllm_sm70 import _ext  # noqa: E402

_ext.load()


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def _op(name: str):
    if not has_op(name):
        raise RuntimeError(
            f"SM70 GDN op {NAMESPACE}::{name} is not available. "
            "Build vllm_sm70._sm70_C with CUDA 12.x for sm_70."
        )
    return getattr(getattr(torch.ops, NAMESPACE), name)


# ---------------------------------------------------------------------------
# Op wrappers and fakes
# ---------------------------------------------------------------------------


# fork gdn_forward.cu gdn_forward_vlk_varlen (pybind, returned [out, state])
def gdn_forward_vlk_varlen_out(
    output: torch.Tensor,
    final_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float,
    gate_is_exp: bool,
) -> None:
    _op("gdn_forward_vlk_varlen_out")(
        output,
        final_state,
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        gate_is_exp,
    )


if has_op("gdn_forward_vlk_varlen_out"):

    @register_fake("_sm70_C::gdn_forward_vlk_varlen_out")
    def _gdn_forward_vlk_varlen_out_fake(
        output: torch.Tensor,
        final_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        scale: float,
        gate_is_exp: bool,
    ) -> None:
        return None


# fork gdn_forward.cu gdn_decode_mixed_qkv_global_state
def gdn_decode_mixed_qkv_global_state(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
) -> None:
    _op("gdn_decode_mixed_qkv_global_state")(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        output,
        scale,
        use_qk_l2norm,
    )


if has_op("gdn_decode_mixed_qkv_global_state"):

    @register_fake("_sm70_C::gdn_decode_mixed_qkv_global_state")
    def _gdn_decode_mixed_qkv_global_state_fake(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        output: torch.Tensor,
        scale: float,
        use_qk_l2norm: bool,
    ) -> None:
        return None


# fork gdn_forward.cu resolve_column_groups_per_block
def gdn_resolve_column_groups_per_block(tokens: int, q_heads: int, v_heads: int) -> int:
    """Columns of V per thread subgroup the kernels pick for this shape."""
    return int(_op("gdn_resolve_column_groups_per_block")(tokens, q_heads, v_heads))


# No fake for ops without Tensor arguments (same rule as ops/tm.py).
NO_FAKE_OPS = frozenset({"gdn_resolve_column_groups_per_block"})


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

_HEAD_DIM = 128
_ACTIVATION_DTYPES = (torch.float16, torch.float32)


def prefill_enabled() -> bool:
    """Kill switch for the FlashQLA prefill route ("0" turns it off)."""
    return os.environ.get(PREFILL_KILL_SWITCH) != "0"


def decode_enabled() -> bool:
    """Kill switch for the FlashQLA decode route ("0" turns it off)."""
    return os.environ.get(DECODE_KILL_SWITCH) != "0"


def _is_sm70(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) == (7, 0)


def prefill_rejection(q, k, v, g, beta, initial_state) -> str | None:
    """Why the FlashQLA prefill cannot take these tensors, or None."""
    if not has_op("gdn_forward_vlk_varlen_out"):
        return "op_missing"
    if not q.is_cuda or not _is_sm70(q.device):
        return "device"
    if q.dtype not in _ACTIVATION_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return "dtype"
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or q.shape[0] != 1:
        return "shape"
    if q.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return "head_dim"
    if v.shape[2] % q.shape[2]:
        return "heads"
    if g.shape != v.shape[:3] or beta.shape != g.shape:
        return "gate_shape"
    if initial_state is not None and initial_state.dtype not in (
        torch.float16,
        torch.float32,
    ):
        return "state_dtype"
    return None


def flashqla_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FlashQLA varlen chunked GDN prefill, same contract as upstream FLA
    `chunk_gated_delta_rule` (UP third_party/flash_linear_attention/ops/
    chunk.py:138): q/k [1, T, H, K], v [1, T, Hv, V], g (log space) and
    beta [1, T, Hv], initial_state [N, Hv, V, K]. Returns (o [1, T, Hv, V],
    final_state [N, Hv, V, K] float32 or None).
    Fork: fla/ops/sm70/gdn_prefill.py flashqla_sm70_chunk_gated_delta_rule.
    """
    if use_qk_l2norm_in_kernel:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    tokens = q.shape[1]
    v_heads, v_dim, k_dim = v.shape[2], v.shape[3], q.shape[3]
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, tokens], device=q.device, dtype=torch.int32)
    elif cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    num_sequences = cu_seqlens.numel() - 1

    output = None
    if core_attn_out is not None:
        candidate = core_attn_out[:tokens].unsqueeze(0)
        if (
            candidate.shape == v.shape
            and candidate.dtype == v.dtype
            and candidate.is_contiguous()
        ):
            output = candidate
    direct = output is not None
    if output is None:
        output = torch.empty_like(v, memory_format=torch.contiguous_format)
    final_state = torch.empty(
        (num_sequences, v_heads, v_dim, k_dim), device=q.device, dtype=torch.float32
    )
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    gdn_forward_vlk_varlen_out(
        output,
        final_state,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state,
        cu_seqlens.contiguous(),
        float(k_dim**-0.5 if scale is None else scale),
        False,
    )
    if core_attn_out is not None and not direct:
        flat = output.reshape(-1)
        core_attn_out.reshape(-1)[: flat.numel()].copy_(flat)
    return output, (final_state if output_final_state else None)


def decode_rejection(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
) -> str | None:
    """Why the FlashQLA decode cannot take these tensors, or None.
    Fork: fla/ops/sm70/gdn_decode.py flashqla_decode_rejection + the checks of
    fused_fwd.py gdn_decode_mixed_qkv_global_state_sm70."""
    if not has_op("gdn_decode_mixed_qkv_global_state"):
        return "op_missing"
    if not mixed_qkv.is_cuda or not _is_sm70(mixed_qkv.device):
        return "device"
    tokens = mixed_qkv.shape[0] if mixed_qkv.ndim == 2 else 0
    if tokens <= 0:
        return "no_tokens"
    dtype = mixed_qkv.dtype
    if dtype not in _ACTIVATION_DTYPES or a.dtype != dtype or b.dtype != dtype:
        return "dtype"
    if out.dtype != dtype or A_log.dtype != torch.float32:
        return "dtype"
    if dt_bias.dtype not in _ACTIVATION_DTYPES:
        return "dtype"
    if state_indices.dtype != torch.int32 or state_indices.ndim != 1:
        return "state_indices"
    if state_indices.shape[0] < tokens:
        return "state_indices"
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) < mixed_qkv.shape[1]:
        return "mixed_qkv_layout"
    if ssm_state.ndim != 4 or ssm_state.shape[2:] != (_HEAD_DIM, _HEAD_DIM):
        return "state_shape"
    if ssm_state.dtype not in _ACTIVATION_DTYPES:
        return "state_dtype"
    v_heads = ssm_state.shape[1]
    if ssm_state.stride()[1:] != (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1):
        return "state_layout"
    if a.shape != (tokens, v_heads) or b.shape != (tokens, v_heads):
        return "gate_shape"
    if A_log.shape != (v_heads,) or dt_bias.shape != (v_heads,):
        return "gate_shape"
    if out.shape != (tokens, v_heads, _HEAD_DIM):
        return "out_shape"
    for t in (a, b, A_log, dt_bias, state_indices, out):
        if not t.is_contiguous():
            return "contiguity"
    qk = mixed_qkv.shape[1] - v_heads * _HEAD_DIM
    if qk <= 0 or qk % (2 * _HEAD_DIM) or v_heads % (qk // (2 * _HEAD_DIM)):
        return "heads"
    return None


def flashqla_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> bool:
    """FlashQLA fused non-spec GDN decode on SM70; one token per request.

    mixed_qkv: [T, 2*H*K + Hv*V], the post-conv1d packed q|k|v rows (row
      stride may be wider, columns dense); a, b: [T, Hv]; A_log: [Hv] fp32;
      dt_bias: [Hv]; ssm_state: [slots, Hv, V, K] (K=V=128), updated in place
      at state_indices[:T] (int32; a slot <= 0 or >= slots writes zeros to
      out and leaves the state alone: slot 0 is vLLM's NULL_BLOCK_ID and -1
      its PAD_SLOT_ID, as in upstream's packed decode);
      out: [T, Hv, V], written. Computes the sigmoid gating, the optional
      q/k L2 norm and the recurrent update in one launch.
    Returns True when it ran; False, touching nothing, when the kill switch
    VLLM_SM70_GDN_DECODE_FLASHQLA=0 is set or a tensor contract is not met,
    in which case the caller runs the upstream decode.
    """
    if not decode_enabled():
        return False
    if state_indices.ndim == 1 and state_indices.shape[0] > mixed_qkv.shape[0]:
        state_indices = state_indices[: mixed_qkv.shape[0]]
    reason = decode_rejection(
        mixed_qkv, a, b, A_log, dt_bias, ssm_state, state_indices, out
    )
    if reason is not None:
        logger.debug("vllm_sm70: FlashQLA GDN decode rejected (%s)", reason)
        return False
    gdn_decode_mixed_qkv_global_state(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        out,
        float(_HEAD_DIM**-0.5 if scale is None else scale),
        bool(use_qk_l2norm),
    )
    return True


# ---------------------------------------------------------------------------
# Out-of-tree ChunkGatedDeltaRule (prefill route)
# ---------------------------------------------------------------------------

_oot_class = None


def register_chunk_gated_delta_rule():
    """Install the SM70 ChunkGatedDeltaRule with vLLM's OOT registry.

    Idempotent. Returns the class. Imports vLLM, so call it from register()
    only. Instances choose FlashQLA at construction when the kill switch is
    not "0" and the current device is SM70; each call still checks its
    tensors and falls back to upstream's forward_native (Triton/FLA).
    """
    global _oot_class
    if _oot_class is not None:
        return _oot_class

    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )

    class Sm70ChunkGatedDeltaRule(ChunkGatedDeltaRule):
        def __init__(self) -> None:
            super().__init__()
            self.sm70_flashqla = (
                prefill_enabled()
                and has_op("gdn_forward_vlk_varlen_out")
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability() == (7, 0)
            )
            if self.sm70_flashqla:
                self.gdn_prefill_backend = "flashqla_sm70"
                self._forward_method = self.forward_flashqla_sm70
                logger.info("vllm_sm70: GDN prefill uses FlashQLA-SM70")

        def forward_flashqla_sm70(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: torch.Tensor | None = None,
            chunk_indices: torch.Tensor | None = None,
            chunk_offsets: torch.Tensor | None = None,
            use_qk_l2norm_in_kernel: bool = True,
            core_attn_out: torch.Tensor | None = None,
            aiter_prefill_metadata: object | None = None,
        ):
            reason = prefill_rejection(q, k, v, g, beta, initial_state)
            if reason is not None:
                logger.debug("vllm_sm70: FlashQLA GDN prefill rejected (%s)", reason)
                return self.forward_native(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    core_attn_out=core_attn_out,
                    aiter_prefill_metadata=aiter_prefill_metadata,
                )
            return flashqla_chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                core_attn_out=core_attn_out,
            )

    name = ChunkGatedDeltaRule.__name__
    if name in op_registry_oot:
        # Someone else (another plugin) already replaced it: leave theirs.
        logger.warning(
            "vllm_sm70: %s already has an out-of-tree class (%s); "
            "FlashQLA GDN prefill not installed",
            name,
            op_registry_oot[name],
        )
        return None
    CustomOp.register_oot(name=name)(Sm70ChunkGatedDeltaRule)
    _oot_class = Sm70ChunkGatedDeltaRule
    return _oot_class
