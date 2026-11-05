# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused per-head q/k RMSNorm, NeoX RoPE and paged FP16 KV write for SM70.

Ported from charlie-horse/vLLM cf9492a ``vllm/models/flint/qk_norm_rope_kv.py``.
The kernel (fork :33-125) and the host wrapper (fork :128-215) are unchanged.
Three things differ from the fork, all for UP v0.33.0 (4ba787e):

1. The op is ``torch.ops.vllm.sm70_flint_qk_norm_rope_kv_cache_update`` (fork
   name ``qk_norm_rope_kv_cache_update``, fork :27). Like UP
   ``unified_kv_cache_update``
   (UP vllm/model_executor/layers/attention/attention.py:729-767), it
   **returns a 0-element dummy**, which the model passes to
   ``torch.ops.vllm.unified_attention_with_output(kv_cache_dummy_dep=...)``
   (UP :772-800). That data edge orders the cache write before attention under
   torch.compile, so the fork-only ``Attention.forward(kv_cache_updated=True)``
   (FORK vllm/model_executor/layers/attention/attention.py:535) is not needed.
2. Cache views follow UP's Triton layout: ``kv_cache.transpose(1, 2).split(D,
   -1)`` (UP vllm/v1/attention/backends/triton_attn.py:827-828), instead of
   the fork's ``unbind(1)`` of
   ``[blocks, 2, page, Hkv, D]``.
3. It isn't added to ``splitting_ops`` (the fork patched
   config/compilation.py:1174). On torch >= 2.11 the layer name is an opaque
   ``LayerName`` that is hoisted as a graph input (UP utils/torch_utils.py:
   940-990), so piecewise graphs are still shared across layers. If GPU test
   T4 shows per-layer recompiles, add ``QUALIFIED_OP_NAME`` to
   ``--compilation-config splitting_ops`` (user-settable, UP
   config/compilation.py:502), together with UP's ``_attention_ops`` and the
   two kv-update ops.

Numerics follow the unfused route. The norm reduces in FP32 and rounds
``x * rsqrt(mean(x^2) + eps) * w`` to FP16 once. RoPE reads the rounded
normalized values, computes ``x1 * cos - x2 * sin`` and ``x2 * cos + x1 * sin``
in FP32 and rounds once. The cache receives the same FP16 keys the attention
gets, and V is copied unchanged. Slots below zero (padding) are skipped, as
``triton_reshape_and_cache_flash`` does.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

OP_NAME = "sm70_flint_qk_norm_rope_kv_cache_update"
QUALIFIED_OP_NAME = f"vllm::{OP_NAME}"  # for an optional splitting_ops entry


# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:33-125
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
    # Verbatim from FORK vllm/models/flint/qk_norm_rope_kv.py:65-125.
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


# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:128-215
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
    """Normalize, rotate and publish one layer's q/k/v (fork :128-215).

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
        assert cos_sin_cache.dtype == torch.float16
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


# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:218-265
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
    kv_sharing: bool,
    backend_writes_kv_in_forward: bool,
) -> str | None:
    """Return why a layer keeps the unfused route, or None to fuse.

    Port of fork :218-265. The fork's ``calculate_kv_scales`` argument is
    gone because UP's ``Attention`` has no such attribute (grep of UP
    attention/attention.py is empty). ``enabled`` now comes from
    ``sm70_config(vllm_config).flint_qk_rope_kv``. Decided once per layer at
    construction, so the captured forward has no shape- or data-dependent
    branch.
    """
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    if not enabled:
        return "VLLM_SM70_FLINT_QK_ROPE_KV=0 or Sm70Config.flint_qk_rope_kv=False"
    if not is_sm70:
        return "the device is not SM70"
    if dtype != torch.float16:
        return f"the model dtype is {dtype}, not float16"
    if head_dim != 128:
        return f"head_dim is {head_dim}, not 128"
    if q_eps != k_eps:
        return "the q and k norms use different epsilons"
    if rotary_emb is not None and not (
        type(rotary_emb) is RotaryEmbedding
        and rotary_emb.is_neox_style
        and rotary_emb.head_size == rotary_emb.rotary_dim == head_dim
        and rotary_emb.cos_sin_cache.dtype == torch.float16
    ):
        return "the rotary embedding is not full-dimension NeoX in float16"
    if kv_cache_dtype != "auto":
        return f"kv_cache_dtype is {kv_cache_dtype!r}, not 'auto'"
    if attn_type != "decoder":
        return "the layer is not plain decoder attention"
    if kv_sharing:
        return "the layer shares another layer's KV cache"
    if backend_writes_kv_in_forward:
        return "the attention backend writes the KV cache in its forward"
    return None


def triton_cache_views(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[blocks, Hkv, page, 2*D]``
    (UP vllm/v1/attention/backends/triton_attn.py:598-599) -> K and V as
    ``[blocks, page, Hkv, D]`` views, exactly as
    ``TritonAttentionImpl.do_kv_cache_update`` builds them (UP :827-828).
    Not in the fork: its cache was ``[blocks, 2, page, Hkv, D]`` (fork :282-284)."""
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    return key_cache, value_cache


def qk_norm_rope_kv_reference(
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
    """Pure-torch oracle for ``qk_norm_rope_kv`` (CPU and GPU), with the
    kernel's rounding points (fork :65-125): FP32 norm, one FP16 rounding,
    RoPE in FP32 on the rounded values, one FP16 rounding. The unfused route
    (RMSNorm + NeoX RotaryEmbedding + triton_reshape_and_cache_flash) is
    expected to match it."""
    head_dim = q_weight.shape[0]
    hq = q_out.shape[1] // head_dim
    hkv = k_out.shape[1] // head_dim
    n = qkv.shape[0]
    heads = qkv[:, : (hq + hkv) * head_dim].float().reshape(n, hq + hkv, head_dim)
    w = torch.cat(
        [q_weight.float().expand(hq, head_dim), k_weight.float().expand(hkv, head_dim)]
    )
    inv = torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + epsilon)
    normed = (heads * inv * w).half()
    out = normed
    if cos_sin_cache is not None:
        half = head_dim // 2
        pos = positions.long()
        pos = torch.where(pos < 0, pos + cos_sin_cache.shape[0], pos)
        cos = cos_sin_cache[pos, :half].float()[:, None, :]
        sin = cos_sin_cache[pos, half:].float()[:, None, :]
        x1, x2 = normed[..., :half].float(), normed[..., half:].float()
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).half()
    q_out.copy_(out[:, :hq].reshape(n, -1))
    k_out.copy_(out[:, hq:].reshape(n, -1))
    if key_cache is None:
        return
    page = key_cache.shape[1]
    v = qkv[:, (hq + hkv) * head_dim :].reshape(n, hkv, head_dim)
    for t in range(min(n, slot_mapping.shape[0])):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        key_cache[slot // page, slot % page] = out[t, hq:]
        value_cache[slot // page, slot % page] = v[t]


# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:268-294
def _direct_cache_views(
    layer, kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the K/V cache views this kernel may write, or None to fall back
    to the backend's own writer.

    Port of fork :268-294, rewritten for UP's Triton layout (UP
    triton_attn.py:798-843). The kernel only replaces the plain FP16 writer:
    not per-token-head quant (:811), not FP8 (:829), and not the ROCm pointer
    range (:832, `_set_kv_ptr_range`).
    """
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = layer.impl
    if type(impl).do_kv_cache_update is not TritonAttentionImpl.do_kv_cache_update:
        return None  # e.g. Kai's CUSTOM Flash-V100 backend: use its writer
    if getattr(impl, "_is_per_token_head_quant", False) or getattr(
        impl, "_set_kv_ptr_range", False
    ):
        return None
    if layer.kv_cache_dtype != "auto" or kv_cache.ndim != 4:
        return None
    key_cache, value_cache = triton_cache_views(kv_cache, impl.head_size)
    head_dim = impl.head_size
    if (
        key_cache.dtype != torch.float16
        or key_cache.shape[2:] != (num_kv_heads, head_dim)
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or slot_mapping.ndim != 1
    ):
        return None
    return key_cache, value_cache


# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:297-345
def sm70_flint_qk_norm_rope_kv_cache_update(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    epsilon: float,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Op body (fork :297-345). Returns the dummy dependency tensor."""
    from vllm.model_executor.layers.attention.attention import get_attention_context

    _, layer, kv_cache, slot_mapping = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    head_dim = q_weight.shape[0]
    num_kv_heads = k_out.shape[1] // head_dim
    views = None
    if slot_mapping is not None and kv_cache.numel() > 0:
        views = _direct_cache_views(layer, kv_cache, slot_mapping, num_kv_heads)
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
        return qkv.new_empty(0)
    qk_norm_rope_kv(
        qkv, q_out, k_out, q_weight, k_weight, positions, cos_sin_cache, epsilon
    )
    if slot_mapping is not None and kv_cache.numel() > 0:
        # A cache layout this kernel doesn't write: keep the backend's writer.
        v_start = q_out.shape[1] + k_out.shape[1]
        layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            layer,
            k_out.view(-1, num_kv_heads, head_dim),
            qkv[:, v_start:].view(-1, num_kv_heads, head_dim),
            kv_cache,
            slot_mapping,
        )
    return qkv.new_empty(0)


# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:348-359
def sm70_flint_qk_norm_rope_kv_cache_update_fake(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    epsilon: float,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


# q_out/k_out are mutated. The returned dummy orders the cache write before
# unified_attention_with_output (same pattern as UP
# vllm/model_executor/layers/attention/attention.py:762-767).
# ported from FORK vllm/models/flint/qk_norm_rope_kv.py:362-369
direct_register_custom_op(
    op_name=OP_NAME,
    op_func=sm70_flint_qk_norm_rope_kv_cache_update,
    fake_impl=sm70_flint_qk_norm_rope_kv_cache_update_fake,
    mutates_args=["q_out", "k_out"],
)
