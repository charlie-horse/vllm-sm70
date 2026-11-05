# SPDX-License-Identifier: Apache-2.0
"""K05 GPU: Flash-V100 vs torch SDPA for the Flint and Qwen shapes.

Every case reads K/V through upstream v0.33.0's KV layout (LBNHC: one
[blocks, block_size, H_kv, 2*D] buffer, K and V interleaved per head, so a
K or V view has stride(2) == 2*D), split by fv100.split_lbnhc_kv_cache.
The reference is fp32 torch.nn.functional.scaled_dot_product_attention with
the causal/window mask and the learned sink as an additive bias
(_fv100_ref.sdpa_reference).

  Flint: D128, GQA 8 (TP2 16q/2kv; TP1 32q/4kv), window 513 or full, sinks.
  Qwen3.6/3.8 full-attention layers: D256, GQA 6 (12q/2kv) and 8 (8q/1kv),
  no window, no sinks; decode through both the XQA and the scalar kernel.
Batch {1, 4} x context {512, 4096}.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _fv100_ref import (  # noqa: E402
    FULL,
    SWA513,
    decode_reference,
    paged_kv,
    prefill_reference,
    sinks_for,
)

pytestmark = pytest.mark.gpu

DECODE_ATOL, DECODE_RTOL = 2e-3, 2e-2
PREFILL_ATOL, PREFILL_RTOL = 1e-2, 2e-2

BATCH = pytest.mark.parametrize("batch", [1, 4])
CONTEXT = pytest.mark.parametrize("context", [512, 4096])
FLINT_HEADS = pytest.mark.parametrize(
    "heads", [(16, 2), (32, 4)], ids=["flint-tp2", "flint-tp1"]
)
FLINT_WINDOW = pytest.mark.parametrize("window", [SWA513, FULL], ids=["swa513", "full"])
QWEN_HEADS = pytest.mark.parametrize(
    "heads", [(12, 2), (8, 1)], ids=["gqa6-qwen38", "gqa8-qwen36"]
)


def _lengths(batch: int, context: int) -> list[int]:
    """Distinct per-row lengths, the longest equal to `context`."""
    return [context, context - 17, context // 2 + 3, context - 1][:batch]


def _query(shape, device, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen).to(device, torch.float16)


def _lbnhc(fv100, seq_lens, num_kv_heads, head_dim, block_size, device, seed):
    k, v, table, _ = paged_kv(
        seq_lens, num_kv_heads, head_dim, block_size, device, seed, layout="lbnhc"
    )
    assert k.stride(2) == 2 * head_dim and v.stride(2) == 2 * head_dim
    assert k.data_ptr() + head_dim * k.element_size() == v.data_ptr()
    fv100.check_kv_cache_views(k, v, head_size=head_dim, num_kv_heads=num_kv_heads)
    return k, v, table


@BATCH
@CONTEXT
@FLINT_HEADS
@FLINT_WINDOW
@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_flint_decode_vs_sdpa(fv100_gpu, batch, context, heads, window, block_size):
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    lens_list = _lengths(batch, context)
    k, v, table = _lbnhc(fv100, lens_list, num_kv_heads, 128, block_size, device, 1)
    q = _query((batch, num_heads, 128), device, 2)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=device)
    sinks = sinks_for(num_heads, device, 3)
    scale = 128**-0.5
    out = fv100.flash_attn_decode_paged(
        q,
        k,
        v,
        table,
        lens,
        softmax_scale=scale,
        window_size=window,
        max_seq_len_hint=max(lens_list),
        sinks=sinks,
    )
    torch.cuda.synchronize()
    ref = decode_reference(
        q, k, v, table, lens, scale=scale, window=window, sinks=sinks
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=DECODE_ATOL, rtol=DECODE_RTOL)


@BATCH
@CONTEXT
@FLINT_HEADS
@FLINT_WINDOW
@pytest.mark.parametrize("q_len", [1, 64])
@torch.inference_mode()
def test_flint_prefill_vs_sdpa(fv100_gpu, batch, context, heads, window, q_len):
    """A q_len chunk at the end of each row (q_len 1: a one-token chunk)."""
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    lens_list = _lengths(batch, context)
    k, v, table = _lbnhc(fv100, lens_list, num_kv_heads, 128, 16, device, 4)
    q = _query((batch, q_len, num_heads, 128), device, 5)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=device)
    sinks = sinks_for(num_heads, device, 6)
    scale = 128**-0.5
    out = fv100.flash_attn_prefill_paged(
        q,
        k,
        v,
        table,
        lens,
        softmax_scale=scale,
        causal=True,
        window_size=window,
        sinks=sinks,
    )
    torch.cuda.synchronize()
    ref = prefill_reference(
        q, k, v, table, lens, scale=scale, causal=True, window=window, sinks=sinks
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=PREFILL_ATOL, rtol=PREFILL_RTOL)


@BATCH
@CONTEXT
@QWEN_HEADS
@pytest.mark.parametrize("kernel", ["xqa", "scalar"])
@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_qwen_d256_decode_vs_sdpa(fv100_gpu, batch, context, heads, kernel, block_size):
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    lens_list = _lengths(batch, context)
    k, v, table = _lbnhc(fv100, lens_list, num_kv_heads, 256, block_size, device, 7)
    q = _query((batch, num_heads, 256), device, 8)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=device)
    scale = 256**-0.5
    if kernel == "xqa":
        assert fv100.xqa_supported(q, k)
        decode = fv100.flash_attn_decode_paged_xqa
    else:
        decode = fv100.flash_attn_decode_paged
    out = decode(
        q, k, v, table, lens, softmax_scale=scale, max_seq_len_hint=max(lens_list)
    )
    torch.cuda.synchronize()
    ref = decode_reference(q, k, v, table, lens, scale=scale, window=FULL, sinks=None)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=DECODE_ATOL, rtol=DECODE_RTOL)


@QWEN_HEADS
@pytest.mark.parametrize("partition_size", [256, 512, 1024])
@torch.inference_mode()
def test_qwen_d256_xqa_every_partition(fv100_gpu, heads, partition_size):
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    lens_list = [4096, 1, 1025, 3000]
    k, v, table = _lbnhc(fv100, lens_list, num_kv_heads, 256, 16, device, 9)
    q = _query((4, num_heads, 256), device, 10)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=device)
    scale = 256**-0.5
    out = fv100.flash_attn_decode_paged_xqa(
        q,
        k,
        v,
        table,
        lens,
        softmax_scale=scale,
        max_seq_len_hint=4096,
        partition_size_hint=partition_size,
    )
    torch.cuda.synchronize()
    ref = decode_reference(q, k, v, table, lens, scale=scale, window=FULL, sinks=None)
    torch.testing.assert_close(out.float(), ref, atol=DECODE_ATOL, rtol=DECODE_RTOL)


@BATCH
@CONTEXT
@QWEN_HEADS
@pytest.mark.parametrize("q_len", [1, 8, 64])
@pytest.mark.parametrize("block_size", [16, 784])
@torch.inference_mode()
def test_qwen_d256_prefill_vs_sdpa(fv100_gpu, batch, context, heads, q_len, block_size):
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    lens_list = _lengths(batch, context)
    k, v, table = _lbnhc(fv100, lens_list, num_kv_heads, 256, block_size, device, 11)
    q = _query((batch, q_len, num_heads, 256), device, 12)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=device)
    scale = 256**-0.5
    out = fv100.flash_attn_prefill_paged(
        q, k, v, table, lens, softmax_scale=scale, causal=True
    )
    torch.cuda.synchronize()
    ref = prefill_reference(
        q, k, v, table, lens, scale=scale, causal=True, window=FULL, sinks=None
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=PREFILL_ATOL, rtol=PREFILL_RTOL)


@torch.inference_mode()
def test_lbnhc_and_separate_layouts_agree(fv100_gpu):
    """Same K/V values in both layouts give bit-identical outputs."""
    device, fv100 = fv100_gpu
    lens_list = [3000, 700]
    k, v, table = _lbnhc(fv100, lens_list, 2, 128, 16, device, 13)
    k_sep, v_sep = k.contiguous(), v.contiguous()
    q = _query((2, 16, 128), device, 14)
    lens = torch.tensor(lens_list, dtype=torch.int32, device=device)
    sinks = sinks_for(16, device, 15)
    kw = dict(
        softmax_scale=128**-0.5, window_size=SWA513, max_seq_len_hint=3000, sinks=sinks
    )
    a = fv100.flash_attn_decode_paged(q, k, v, table, lens, **kw).clone()
    b = fv100.flash_attn_decode_paged(q, k_sep, v_sep, table, lens, **kw)
    assert torch.equal(a, b)
