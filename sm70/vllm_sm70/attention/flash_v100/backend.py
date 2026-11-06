# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLASH_ATTN_V100: the Flash-V100 attention backend for vLLM v0.33.0 (K06).

Rebuilt from fork cf9492a vllm/v1/attention/backends/flash_v100/backend.py
(52 lines) against the v0.33.0 interface (Ivy's section B, sec. 5). Like the
fork it subclasses the Triton backend, so the KV-cache write
(`do_kv_cache_update` -> triton_reshape_and_cache_flash), the metadata base
and the encoder/unsupported-feature fallback are upstream's.

Registered as AttentionBackendEnum.CUSTOM (vllm_sm70.attention.register) and
picked by Sm70CudaPlatform.get_attn_backend_cls on SM70 for fp16 decoder
layers with head size 128 or 256. Kill switch: VLLM_SM70_FLASH_V100=0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

# Human-readable name for logs. get_name() must return the enum member name
# ("CUSTOM"): every upstream Attention layer does
# AttentionBackendEnum[backend.get_name()] (UP attention.py:433), and
# FLASH_ATTN_V100 is not a member.
BACKEND_NAME = "FLASH_ATTN_V100"
ENUM_NAME = "CUSTOM"
SUPPORTED_HEAD_SIZES = (128, 256)


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash-V100 paged attention on Volta (sm_70), fp16 only."""

    # fp16 activations and KV only: the lean kernels (K05) instantiate fp16.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16"]

    # The KV cache is written by upstream's reshape_and_cache (Triton impl).
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return ENUM_NAME

    @staticmethod
    def get_impl_cls():
        from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

        return FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        from vllm_sm70.attention.flash_v100.metadata import (
            FlashAttnV100MetadataBuilder,
        )

        return FlashAttnV100MetadataBuilder

    # -- KV cache layout (sec. 5.1-5.3) ------------------------------------

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        # Generic paged kernels and the BM32 phase prefill need pages that are
        # a multiple of 16 tokens; XQA and decode take any such page size.
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % 16 == 0

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # K and V interleaved per head; the kernels take explicit strides
        # (stride(H) == 2 * head_size), see vllm_sm70.ops.fv100.
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # Replaces the removed get_kv_cache_shape/stride_order hooks: fp16 KV
        # needs no packing beyond upstream's default (K | V per head).
        return spec

    # -- capability flags (sec. 5.8) ---------------------------------------

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return list(SUPPORTED_HEAD_SIZES)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in SUPPORTED_HEAD_SIZES

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return (capability.major, capability.minor) == (7, 0)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        # Non-causal paged prefill (draft models) is compiled (K05).
        return True

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Prefill groups are planned from the CPU query lengths.
        return False
