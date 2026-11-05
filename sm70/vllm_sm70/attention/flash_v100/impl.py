# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttnV100Impl: Flash-V100 forward for vLLM v0.33.0 (K06).

Rebuilt from fork cf9492a vllm/v1/attention/backends/flash_v100/impl.py
(1210 lines). Subclasses TritonAttentionImpl, as the fork did:

* __init__ is upstream's (sinks, sliding window, the fp8/bf16 KV refusals
  below SM89/SM80); the constructor arguments the lean kernels cannot serve
  (ALiBi, soft cap, chunk lookback, non-decoder layers, quantized KV) mark
  the layer for upstream's Triton forward instead.
* do_kv_cache_update is upstream's (triton_reshape_and_cache_flash into the
  LBNHC cache), so forward_includes_kv_cache_update stays False.
* forward splits upstream's logical (blocks, H_kv, block_size, 2*D) cache
  into strided K/V views (no copy), checks them once, runs decode rows then
  prefill groups (dispatch.py), and falls back to the Triton forward for
  per-step features the kernels lack (tensor `causal`, R-SWA, multimodal
  prefix ranges, fused output quantization).
"""

from __future__ import annotations

import logging

import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from vllm_sm70.attention.flash_v100 import dispatch
from vllm_sm70.attention.flash_v100.backend import SUPPORTED_HEAD_SIZES
from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100Metadata
from vllm_sm70.ops import fv100

logger = logging.getLogger("vllm.sm70")

_logged_fallbacks: set[str] = set()


def _log_fallback_once(reason: str) -> None:
    if reason not in _logged_fallbacks:
        _logged_fallbacks.add(reason)
        logger.info("vllm_sm70 Flash-V100: using the Triton forward (%s)", reason)


class FlashAttnV100Impl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fv100_reason = self._unsupported_reason()
        if self._fv100_reason is not None:
            _log_fallback_once(self._fv100_reason)
        self._kv_views_checked = False
        dispatch.init_fused_decode(self)  # K14: counters before any capture

    def _unsupported_reason(self) -> str | None:
        if self.attn_type != AttentionType.DECODER:
            return f"attention type {self.attn_type}"
        if self.head_size not in SUPPORTED_HEAD_SIZES:
            return f"head size {self.head_size}"
        if self.kv_cache_dtype not in ("auto", "float16"):
            return f"kv_cache_dtype {self.kv_cache_dtype}"
        if self.alibi_slopes is not None or self.use_alibi_sqrt:
            return "ALiBi"
        if self.logits_soft_cap:
            return "logits soft cap"
        if self.chunk_lookback != -1:
            return "chunk lookback"
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            return f"sinks dtype {self.sinks.dtype}"
        return None

    def _step_fallback_reason(
        self,
        metadata: FlashAttnV100Metadata,
        query: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> str | None:
        if self._fv100_reason is not None:
            return self._fv100_reason
        if not isinstance(metadata, FlashAttnV100Metadata):
            return "non-Flash-V100 metadata"
        if isinstance(metadata.causal, torch.Tensor):
            return "per-row causal tensor"
        if metadata.rswa_prefix_lens is not None:
            return "R-SWA"
        if metadata.mm_prefix_range_tensor is not None:
            return "multimodal prefix ranges"
        if output_scale is not None or output_block_scale is not None:
            return "fused output quantization"
        if query.dtype != torch.float16:
            return f"query dtype {query.dtype}"
        return None

    def _sinks(self) -> torch.Tensor | None:
        sinks = self.sinks
        if sinks is not None and not sinks.is_contiguous():
            sinks = sinks.contiguous()
        return sinks

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttnV100Metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        reason = self._step_fallback_reason(
            attn_metadata, query, output_scale, output_block_scale
        )
        if reason is not None:
            _log_fallback_once(reason)
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        key_cache, value_cache = dispatch.split_kv_cache(kv_cache, self.head_size)
        if not self._kv_views_checked:
            # Once per layer: strides, alignment, dtype of the LBNHC views.
            fv100.check_kv_cache_views(
                key_cache,
                value_cache,
                head_size=self.head_size,
                num_kv_heads=self.num_kv_heads,
            )
            self._kv_views_checked = True

        dispatch.fused_decode_hook(self, attn_metadata)
        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]
        out = output[:num_tokens]
        sinks = self._sinks()
        window = tuple(self.sliding_window)
        dispatch.run_decode(
            query,
            key_cache,
            value_cache,
            out,
            attn_metadata,
            scale=self.scale,
            window=window,
            sinks=sinks,
        )
        if attn_metadata.prefill_groups:
            dispatch.run_prefill(
                query,
                key_cache,
                value_cache,
                out,
                attn_metadata,
                scale=self.scale,
                causal=bool(attn_metadata.causal),
                window=window,
                sinks=sinks,
            )
        return output
