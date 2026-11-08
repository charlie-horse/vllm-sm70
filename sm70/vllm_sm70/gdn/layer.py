# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen GDN layer subclass (M11): routes non-spec decode and its warmup to K08.

Upstream `QwenGatedDeltaNetAttention` (UP qwen_gdn_linear_attn.py:543) is a
`PluggableLayer`, so `PluggableLayer.register_oot` (UP custom_op.py:83) makes
every `QwenGatedDeltaNetAttention(...)` built by qwen3_5.py / qwen3_next.py
an instance of this subclass. No upstream file is touched.

Overrides (fork cf9492a qwen_gdn_linear_attn.py):
- `_forward_core_decode_non_spec` (UP :1981; fork :4649-4810): upstream's
  conv1d update, then K08 `flashqla_decode` instead of the packed Triton
  decode. Falls back to the packed Triton decode when K08 refuses the tensors.
- `_warmup_prefill_kernels` (UP :1308-1448; fork :3458 + warmup/gdn.py):
  upstream's warmup (its `self.chunk_gated_delta_rule` is K08's OOT op, so
  prefill already warms FlashQLA), then one FlashQLA decode at 1 and 2
  tokens, dense and at the real row stride.
Kill switches are K08's: VLLM_SM70_GDN_DECODE_FLASHQLA=0 makes both
overrides the upstream methods; VLLM_SM70_GDN_FLASHQLA_PREFILL=0 is read by
the prefill op itself.
"""

from __future__ import annotations

import logging

import torch

from vllm_sm70.gdn import kernels

logger = logging.getLogger("vllm.sm70")

_layer_class = None


def _decode_on() -> bool:
    return kernels.decode_enabled() and kernels.has_op(
        "gdn_decode_mixed_qkv_global_state"
    )


def register_qwen_gdn_layer():
    """Install the SM70 Qwen GDN layer with vLLM's OOT registry. Idempotent.

    Returns the class, or None when another plugin already replaced it.
    Imports vLLM, so call it from register() only.
    """
    global _layer_class
    if _layer_class is not None:
        return _layer_class

    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as up

    class Sm70QwenGatedDeltaNetAttention(up.QwenGatedDeltaNetAttention):
        def _forward_core_decode_non_spec(
            self, mixed_qkv, b, a, core_attn_out, attn_metadata
        ):
            if not _decode_on():
                return super()._forward_core_decode_non_spec(
                    mixed_qkv, b, a, core_attn_out, attn_metadata
                )
            n = attn_metadata.num_actual_tokens
            idx = attn_metadata.non_spec_state_indices_tensor[:n]
            conv_state, ssm_state = self.kv_cache[0], self.kv_cache[1]
            if not up.is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            w = self.conv1d.weight
            x = up.causal_conv1d_update(
                mixed_qkv[:n],
                conv_state,
                w.view(w.size(0), w.size(2)),
                self.conv1d.bias,
                self.activation,
                conv_state_indices=idx,
                validate_data=False,
            )
            # split_ba() returns column views; K08 needs dense [T, Hv].
            a, b = a[:n].contiguous(), b[:n].contiguous()
            out = core_attn_out[:n]
            scale = self.head_k_dim**-0.5
            if kernels.flashqla_decode(
                x, a, b, self.A_log, self.dt_bias, ssm_state, idx, out, scale=scale
            ):
                return
            up.fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=x,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=scale,
                initial_state=ssm_state,
                out=out.unsqueeze(1),
                ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
            )

        def _warmup_prefill_kernels(self, qkv_or_qkvz, v_dim):
            first = not self._prefill_kernels_warmed_up
            super()._warmup_prefill_kernels(qkv_or_qkvz, v_dim)
            if first and _decode_on():
                self._sm70_warmup_flashqla_decode(qkv_or_qkvz, v_dim)

        def _sm70_warmup_flashqla_decode(self, qkv_or_qkvz, v_dim):
            qkv = qkv_or_qkvz.shape[-1] - v_dim
            stride = qkv_or_qkvz.stride(0) if qkv_or_qkvz.dim() >= 2 else qkv
            hv = self.num_v_heads // self.tp_size
            dev, dt = qkv_or_qkvz.device, qkv_or_qkvz.dtype
            _, state_dtype = self.get_state_dtype()
            # Slot 0 is NULL_BLOCK_ID (K08 skips it); tokens use slots 1..t.
            state = torch.zeros(
                3, hv, self.head_v_dim, self.head_k_dim, device=dev, dtype=state_dtype
            )
            try:
                for t in (1, 2):
                    for width in sorted({qkv, max(qkv, stride)}):
                        kernels.flashqla_decode(
                            torch.zeros(t, width, device=dev, dtype=dt)[:, :qkv],
                            torch.zeros(t, hv, device=dev, dtype=dt),
                            torch.zeros(t, hv, device=dev, dtype=dt),
                            self.A_log,
                            self.dt_bias,
                            state,
                            torch.arange(1, t + 1, device=dev, dtype=torch.int32),
                            torch.empty(t, hv, self.head_v_dim, device=dev, dtype=dt),
                            scale=self.head_k_dim**-0.5,
                        )
            except Exception:
                logger.warning(
                    "vllm_sm70: FlashQLA GDN decode warmup failed for layer %s",
                    self.prefix,
                    exc_info=True,
                )

    name = up.QwenGatedDeltaNetAttention.__name__
    if name in op_registry_oot:
        logger.warning(
            "vllm_sm70: %s already has an out-of-tree class (%s); "
            "FlashQLA GDN decode not installed",
            name,
            op_registry_oot[name],
        )
        return None
    PluggableLayer.register_oot(name=name)(Sm70QwenGatedDeltaNetAttention)
    _layer_class = Sm70QwenGatedDeltaNetAttention
    return _layer_class
