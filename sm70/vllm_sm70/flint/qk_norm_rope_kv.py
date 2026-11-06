# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused per-head q/k RMSNorm, NeoX RoPE and paged FP16 KV write for SM70.

Port of FORK vllm/models/flint/qk_norm_rope_kv.py (cf9492a); see
PROVENANCE.md next to this file.

The unfused route runs three or four kernels per attention layer: the q and
k RMSNorms, the RoPE (sliding layers only) and the KV cache write. At one or
two decode tokens each of them is launch bound, so this kernel does all of
it in one launch, one program per (token, head).

Numerics follow the unfused route: the norm reduces in FP32 and rounds
``x * rsqrt(mean(x^2) + eps) * w`` to FP16 once, and RoPE reads the rounded
normalized values, computes ``x1 * cos - x2 * sin`` and ``x2 * cos + x1 * sin``
in FP32 and rounds once. The cache receives the same FP16 keys the attention
gets, and V is copied unchanged. The cache layout and the padding rule
(slots below zero are skipped, tokens past ``slot_mapping`` are not written)
are those of upstream ``TritonAttentionImpl.do_kv_cache_update``: the
per-layer cache is logical ``(blocks, Hkv, page, 2 * D)`` (physical LBNHC on
Flash-V100 and TRITON_ATTN), viewed as K and V ``(blocks, page, Hkv, D)``
with explicit strides.

Kill switch: VLLM_SM70_FLINT_QK_ROPE_KV=0 (Sm70Config.flint_qk_rope_kv),
read by the model when it decides the route (``fused_route_rejection``).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from vllm_sm70.flint import QK_ROPE_KV_OP_NAME, QK_ROPE_KV_SPLITTING_OP

OP_NAME = QK_ROPE_KV_OP_NAME
SPLITTING_OP = QK_ROPE_KV_SPLITTING_OP

HEAD_DIM = 128
# RoPE tables the kernel reads (it widens either to FP32). The fork built the
# table in the model dtype; upstream Flint builds it in FP32
# (UP models/flint/model.py:283-291).
COS_SIN_DTYPES = (torch.float16, torch.float32)
# Unquantized FP16 KV. "float16" is the explicit spelling of "auto" for an
# FP16 model.
KV_CACHE_DTYPES = ("auto", "float16")


@triton.jit(do_not_specialize=["num_slots"])
def _qk_norm_rope_kv_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    q_w_ptr,
    k_w_ptr,
    pos_ptr,
    cos_sin_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_ptr,
    qkv_stride,
    q_out_stride,
    k_out_stride,
    cos_sin_stride,
    cos_sin_rows,
    num_slots,
    page_size,
    k_block_stride,
    k_page_stride,
    k_head_stride,
    v_block_stride,
    v_page_stride,
    v_head_stride,
    eps,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    row = qkv_ptr + token * qkv_stride
    # q heads, then k heads, are contiguous in the qkv row.
    src = row + head * HEAD_DIM
    is_q = head < NUM_Q_HEADS
    w_ptr = tl.where(is_q, q_w_ptr, k_w_ptr)

    x = tl.load(src + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / HEAD_DIM + eps)
    w = tl.load(w_ptr + d).to(tl.float32)
    normed = (x * inv * w).to(tl.float16)
    if HAS_ROPE:
        half = HEAD_DIM // 2
        # The rotary partner of channel d is d + D/2 (first half) or
        # d - D/2 (second half); its norm shares this row's variance.
        mate = (d + half) % HEAD_DIM
        xm = tl.load(src + mate).to(tl.float32)
        wm = tl.load(w_ptr + mate).to(tl.float32)
        other = (xm * inv * wm).to(tl.float16).to(tl.float32)
        pos = tl.load(pos_ptr + token)
        pos = tl.where(pos < 0, pos + cos_sin_rows, pos)
        freq = d % half
        cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + freq).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + half + freq).to(tl.float32)
        value = normed.to(tl.float32)
        first = value * cos - other * sin
        second = value * cos + other * sin
        out = tl.where(d < half, first, second).to(tl.float16)
    else:
        out = normed

    if head < NUM_Q_HEADS:
        tl.store(q_out_ptr + token * q_out_stride + head * HEAD_DIM + d, out)
    else:
        kv_head = head - NUM_Q_HEADS
        tl.store(k_out_ptr + token * k_out_stride + kv_head * HEAD_DIM + d, out)
        if STORE_CACHE:
            slot = tl.load(slot_ptr + token, mask=token < num_slots, other=-1)
            slot = slot.to(tl.int64)
            if slot >= 0:
                block = slot // page_size
                offset = slot % page_size
                tl.store(
                    k_cache_ptr
                    + block * k_block_stride
                    + offset * k_page_stride
                    + kv_head * k_head_stride
                    + d,
                    out,
                )
                v = tl.load(row + (NUM_Q_HEADS + NUM_KV_HEADS + kv_head) * HEAD_DIM + d)
                tl.store(
                    v_cache_ptr
                    + block * v_block_stride
                    + offset * v_page_stride
                    + kv_head * v_head_stride
                    + d,
                    v,
                )


def qk_norm_rope_kv(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    epsilon: float,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
) -> None:
    """Normalize, rotate and publish one layer's q/k/v.

    ``qkv`` is ``[tokens, (Hq + 2 * Hkv) * D]`` with unit inner stride.
    ``q_out``/``k_out`` receive the rotated q/k rows. Without
    ``cos_sin_cache`` only the norms run. With ``key_cache``, ``value_cache``
    (``[blocks, page, Hkv, D]`` views, any outer strides) and
    ``slot_mapping``, k and v are also written to the cache.
    """
    head_dim = q_weight.shape[0]
    num_tokens = qkv.shape[0]
    num_q_heads = q_out.shape[1] // head_dim
    num_kv_heads = k_out.shape[1] // head_dim
    assert qkv.dtype == q_out.dtype == k_out.dtype == torch.float16
    assert qkv.shape[1] == (num_q_heads + 2 * num_kv_heads) * head_dim
    assert qkv.stride(1) == q_out.stride(1) == k_out.stride(1) == 1
    assert q_out.shape[0] == k_out.shape[0] == num_tokens
    assert k_weight.shape == (head_dim,)
    assert q_weight.is_contiguous() and k_weight.is_contiguous()
    # Unused pointers get a valid tensor of the right dtype; the constexpr
    # flags compile their loads and stores away.
    rope_args: tuple = (q_weight, 0, 0)
    if cos_sin_cache is not None:
        assert cos_sin_cache.dtype in COS_SIN_DTYPES
        assert cos_sin_cache.shape[1] == head_dim and cos_sin_cache.stride(1) == 1
        assert positions.numel() == num_tokens
        rope_args = (cos_sin_cache, cos_sin_cache.stride(0), cos_sin_cache.shape[0])
    cache_args: tuple = (q_out, q_out, positions, 0, 1, 0, 0, 0, 0, 0, 0)
    if key_cache is not None:
        assert value_cache is not None and slot_mapping is not None
        assert key_cache.dtype == value_cache.dtype == torch.float16
        assert key_cache.ndim == value_cache.ndim == 4
        assert key_cache.shape == value_cache.shape
        assert key_cache.shape[2:] == (num_kv_heads, head_dim)
        assert key_cache.stride(3) == value_cache.stride(3) == 1
        assert slot_mapping.ndim == 1
        cache_args = (
            key_cache,
            value_cache,
            slot_mapping,
            slot_mapping.shape[0],
            key_cache.shape[1],
            *key_cache.stride()[:3],
            *value_cache.stride()[:3],
        )
    if num_tokens == 0:
        return
    cos_sin, cos_sin_stride, cos_sin_rows = rope_args
    k_cache, v_cache, slots, num_slots, page_size, *cache_strides = cache_args
    _qk_norm_rope_kv_kernel[(num_tokens, num_q_heads + num_kv_heads)](
        qkv,
        q_out,
        k_out,
        q_weight,
        k_weight,
        positions,
        cos_sin,
        k_cache,
        v_cache,
        slots,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_stride,
        cos_sin_rows,
        num_slots,
        page_size,
        *cache_strides,
        epsilon,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HAS_ROPE=cos_sin_cache is not None,
        STORE_CACHE=key_cache is not None,
        num_warps=1,
    )


def fused_route_rejection(
    *,
    enabled: bool,
    is_sm70: bool,
    dtype: torch.dtype,
    head_dim: int,
    q_eps: float,
    k_eps: float,
    rotary_emb: object | None,
    kv_cache_dtype: str,
    attn_type: str,
    calculate_kv_scales: bool,
    kv_sharing: bool,
    backend_writes_kv_in_forward: bool,
) -> str | None:
    """Return why a layer keeps the unfused route, or None to fuse.

    Decided once per layer at construction, so the captured forward has no
    shape- or data-dependent branch. ``enabled`` is
    ``sm70_config(vllm_config).flint_qk_rope_kv``.
    """
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    if not enabled:
        return "VLLM_SM70_FLINT_QK_ROPE_KV=0"
    if not is_sm70:
        return "the device is not SM70"
    if dtype != torch.float16:
        return f"the model dtype is {dtype}, not float16"
    if head_dim != HEAD_DIM:
        return f"head_dim is {head_dim}, not {HEAD_DIM}"
    if q_eps != k_eps:
        return "the q and k norms use different epsilons"
    if rotary_emb is not None and not (
        type(rotary_emb) is RotaryEmbedding
        and rotary_emb.is_neox_style
        and rotary_emb.head_size == rotary_emb.rotary_dim == head_dim
        and rotary_emb.cos_sin_cache.dtype in COS_SIN_DTYPES
    ):
        return "the rotary embedding is not full-dimension NeoX in float16/32"
    if kv_cache_dtype not in KV_CACHE_DTYPES:
        return f"kv_cache_dtype is {kv_cache_dtype!r}, not 'auto'"
    if attn_type != "decoder" or calculate_kv_scales:
        return "the layer is not plain decoder attention"
    if kv_sharing:
        return "the layer shares another layer's KV cache"
    if backend_writes_kv_in_forward:
        return "the attention backend writes the KV cache in its forward"
    return None


def _direct_cache_views(
    layer,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the K/V cache views this kernel may write, or None to fall
    back to the backend's own writer."""
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = layer.impl
    # Only the writer whose layout this kernel reproduces: upstream Triton's
    # (also inherited by Flash-V100), unquantized. The gate is on the method,
    # so a backend that overrides do_kv_cache_update keeps its own writer.
    if type(impl).do_kv_cache_update is not TritonAttentionImpl.do_kv_cache_update:
        return None
    if getattr(impl, "_is_per_token_head_quant", False):
        return None
    if getattr(impl, "_set_kv_ptr_range", False):
        return None
    if layer.kv_cache_dtype not in KV_CACHE_DTYPES or kv_cache.ndim != 4:
        return None
    if kv_cache.shape[1] != num_kv_heads or kv_cache.shape[-1] != 2 * head_dim:
        return None
    # As in TritonAttentionImpl.do_kv_cache_update (UP triton_attn.py:828):
    # (B, H, N, 2D) -> (B, N, H, 2D) -> K, V each (B, N, H, D).
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    if (
        key_cache.dtype != torch.float16
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or slot_mapping.ndim != 1
    ):
        return None
    return key_cache, value_cache


def qk_norm_rope_kv_cache_update(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    epsilon: float,
    layer_name: LayerNameType,
) -> None:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    _, layer, kv_cache, slot_mapping = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    head_dim = q_weight.shape[0]
    num_kv_heads = k_out.shape[1] // head_dim
    views = None
    if slot_mapping is not None and kv_cache.numel() > 0:
        views = _direct_cache_views(
            layer, kv_cache, slot_mapping, num_kv_heads, head_dim
        )
    if views is not None:
        qk_norm_rope_kv(
            qkv,
            q_out,
            k_out,
            q_weight,
            k_weight,
            positions,
            cos_sin_cache,
            epsilon,
            key_cache=views[0],
            value_cache=views[1],
            slot_mapping=slot_mapping,
        )
        return
    qk_norm_rope_kv(
        qkv, q_out, k_out, q_weight, k_weight, positions, cos_sin_cache, epsilon
    )
    if slot_mapping is not None:
        # A cache layout this kernel does not write: keep the backend's writer.
        v_start = q_out.shape[1] + k_out.shape[1]
        layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            layer,
            k_out.view(-1, num_kv_heads, head_dim),
            qkv[:, v_start:].view(-1, num_kv_heads, head_dim),
            kv_cache,
            slot_mapping,
        )


def qk_norm_rope_kv_cache_update_fake(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    epsilon: float,
    layer_name: LayerNameType,
) -> None:
    return None


def attention_after_fused_write(
    attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``attn(q, k, v)`` without the KV-cache write, for a layer whose cache
    the op above has already written.

    Replaces FORK ``Attention.forward(..., kv_cache_updated=True)``
    (FORK attention.py:535,583,604), which upstream does not have. Mirrors UP
    Attention.forward (UP attention.py:490-581) for an unquantized KV cache:
    no query quant, no unified_kv_cache_update. q/k are the op's q_out/k_out,
    so the attention is ordered after the op by data dependency.
    """
    num_tokens = q.shape[0]
    output = torch.empty(
        (num_tokens, attn.num_heads * attn.head_size_v),
        dtype=q.dtype,
        device=q.device,
    )
    query = q.view(-1, attn.num_heads, attn.head_size)
    key = k.view(-1, attn.num_kv_heads, attn.head_size)
    value = v.view(-1, attn.num_kv_heads, attn.head_size_v)
    out = output.view(-1, attn.num_heads, attn.head_size_v)
    if attn.use_direct_call:
        from vllm.model_executor.layers.attention.attention import (
            unified_attention_with_output,
        )

        unified_attention_with_output(query, key, value, out, attn.layer_name)
    else:
        torch.ops.vllm.unified_attention_with_output(
            query, key, value, out, _encode_layer_name(attn.layer_name)
        )
    return output


def _register() -> None:
    # Idempotent: a re-import (tests, reload) must not define the op twice.
    if hasattr(torch.ops.vllm, OP_NAME):
        return
    # q_out/k_out order this op before attention, which consumes them; the KV
    # write it makes through the forward context is ordered by that
    # dependency.
    direct_register_custom_op(
        op_name=OP_NAME,
        op_func=qk_norm_rope_kv_cache_update,
        fake_impl=qk_norm_rope_kv_cache_update_fake,
        mutates_args=["q_out", "k_out"],
    )


_register()
