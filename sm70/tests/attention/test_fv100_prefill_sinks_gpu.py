# SPDX-License-Identifier: Apache-2.0
"""K05 GPU: learned sinks on the Flash-V100 paged prefill (Flint shapes).

Port of fork cf9492a tests/kernels/attention/test_sm70_flash_v100_sinks_prefill.py
(GPU cases, :140-325). The fork's CPU cases check fork backend wiring
(prefill.py executors) and are not ported; K06 owns the backend. The kernel
is reached through vllm_sm70.ops.fv100 instead of flash_attn_v100, and the
reference is the shared SDPA one (_fv100_ref.py). Head size 64 of the fork's
last case is not built in K05 (lean subset: 128/256).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _fv100_ref import FULL, SWA513, sdpa_reference  # noqa: E402

pytestmark = pytest.mark.gpu

ATOL = 2e-2
RTOL = 2e-2


def _sinks(num_heads, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    sinks = torch.randn(num_heads, generator=gen) * 2.0
    sinks[0] = 10.0  # dominates the denominator
    sinks[1] = -30.0  # negligible
    return sinks.to(device)


def _paged_cache(k_lin, v_lin, page, seed):
    """Scatter linear [N, Hkv, D] K/V into shuffled pages with spares."""
    n, num_kv_heads, head_dim = k_lin.shape
    used = -(-n // page)
    num_blocks = used + 3
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=gen).to(k_lin.device)
    shape = (num_blocks, page, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, dtype=k_lin.dtype, device=k_lin.device)
    v_cache = torch.randn(shape, dtype=k_lin.dtype, device=k_lin.device)
    for j in range(used):
        rows = slice(j * page, min((j + 1) * page, n))
        k_cache[int(perm[j]), : rows.stop - rows.start] = k_lin[rows]
        v_cache[int(perm[j]), : rows.stop - rows.start] = v_lin[rows]
    block_table = perm[:used].to(torch.int32)[None]
    return k_cache, v_cache, block_table


def _random_qkv(q_len, kv_len, num_heads, num_kv_heads, head_dim, seed, device):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device, torch.float16)

    return (
        randn(q_len, num_heads, head_dim),
        randn(kv_len, num_kv_heads, head_dim),
        randn(kv_len, num_kv_heads, head_dim),
    )


@torch.inference_mode()
@pytest.mark.parametrize("page", [16, 256])
@pytest.mark.parametrize(
    "q_len,kv_len",
    [
        (40, 150),  # fresh prefill inside one 176-token KV tile
        (300, 300),  # fresh prefill over two KV tiles
        (96, 1400),  # chunk over a prefix, eight KV tiles
        (1, 777),  # one-row chunk over a prefix
    ],
)
@pytest.mark.parametrize("window", [FULL, SWA513], ids=["full", "swa513"])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(32, 4), (16, 2)])
def test_paged_prefill_sinks_match_reference(
    fv100_gpu, num_heads, num_kv_heads, window, q_len, kv_len, page
):
    device, fv100 = fv100_gpu
    head_dim = 128
    scale = head_dim**-0.5
    q, k_lin, v_lin = _random_qkv(
        q_len, kv_len, num_heads, num_kv_heads, head_dim, q_len + kv_len, device
    )
    k_cache, v_cache, block_table = _paged_cache(k_lin, v_lin, page, seed=page)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    sinks = _sinks(num_heads, device)

    def run(**kwargs):
        return fv100.flash_attn_prefill_paged(
            q[None],
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            **kwargs,
        )[0]

    out = run(sinks=sinks)
    out_default = run()
    out_none = run(sinks=None)
    out_neg_inf = run(sinks=torch.full_like(sinks, float("-inf")))
    torch.cuda.synchronize()

    kwargs = dict(scale=scale, causal=True, window=window)
    ref = sdpa_reference(q, k_lin, v_lin, sinks=sinks, **kwargs)
    ref_none = sdpa_reference(q, k_lin, v_lin, sinks=None, **kwargs)
    assert not torch.allclose(ref, ref_none, atol=ATOL, rtol=RTOL), "vacuous sinks"

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
    # Without sinks the kernel is unchanged; a -inf sink adds exactly zero.
    assert torch.equal(out_none, out_default)
    assert torch.equal(out_neg_inf, out_default)
    torch.testing.assert_close(out_default.float(), ref_none, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
@pytest.mark.parametrize("window", [FULL, SWA513], ids=["full", "swa513"])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(32, 4), (16, 2)])
def test_chunked_prefill_with_sinks_matches_single_pass(
    fv100_gpu, num_heads, num_kv_heads, window
):
    """Chunks over a growing cached prefix reproduce one full prefill."""
    device, fv100 = fv100_gpu
    head_dim, page, total = 128, 16, 1500
    scale = head_dim**-0.5
    q, k_lin, v_lin = _random_qkv(
        total, total, num_heads, num_kv_heads, head_dim, 7, device
    )
    k_cache, v_cache, block_table = _paged_cache(k_lin, v_lin, page, seed=3)
    sinks = _sinks(num_heads, device, seed=1)
    chunks = []
    for start, end in ((0, 700), (700, 1300), (1300, 1333), (1333, total)):
        seq_lens = torch.tensor([end], dtype=torch.int32, device=device)
        chunks.append(
            fv100.flash_attn_prefill_paged(
                q[None, start:end],
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                softmax_scale=scale,
                causal=True,
                window_size=window,
                sinks=sinks,
            )[0]
        )
    torch.cuda.synchronize()
    ref = sdpa_reference(
        q, k_lin, v_lin, scale=scale, causal=True, window=window, sinks=sinks
    )
    torch.testing.assert_close(torch.cat(chunks).float(), ref, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
@pytest.mark.parametrize("window", [FULL, SWA513], ids=["full", "swa513"])
def test_batched_requests_index_sinks_per_head(fv100_gpu, window):
    """Two requests with different prefixes and page tables in one launch."""
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads, head_dim, page, q_len = 16, 2, 128, 16, 64
    scale = head_dim**-0.5
    lengths = (900, 200)
    torch.manual_seed(13)
    sinks = _sinks(num_heads, device, seed=2)
    width = -(-max(lengths) // page)
    shape = (2 * width + 1, page, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    queries, tables, refs = [], [], []
    for b, kv_len in enumerate(lengths):
        count = -(-kv_len // page)
        # The requests' pages interleave in one pool.
        pages = (torch.arange(count) * 2 + b + 1).to(torch.int32)
        table = torch.zeros(width, dtype=torch.int32)
        table[:count] = pages
        tables.append(table)
        index = pages.long().to(device)
        q = torch.randn(q_len, num_heads, head_dim, dtype=torch.float16, device=device)
        queries.append(q)
        refs.append(
            sdpa_reference(
                q,
                k_cache[index].flatten(0, 1)[:kv_len],
                v_cache[index].flatten(0, 1)[:kv_len],
                scale=scale,
                causal=True,
                window=window,
                sinks=sinks,
            )
        )
    out = fv100.flash_attn_prefill_paged(
        torch.stack(queries),
        k_cache,
        v_cache,
        torch.stack(tables).to(device),
        torch.tensor(lengths, dtype=torch.int32, device=device),
        softmax_scale=scale,
        causal=True,
        window_size=window,
        sinks=sinks,
    )
    torch.cuda.synchronize()
    for b, ref in enumerate(refs):
        torch.testing.assert_close(out[b].float(), ref, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
@pytest.mark.parametrize("head_dim,num_heads,num_kv_heads", [(256, 8, 2)])
def test_d256_sinks_use_the_generic_kernel(
    fv100_gpu, head_dim, num_heads, num_kv_heads
):
    """Sinks keep D256 on the generic kernel's epilogue (not low-SMEM/BM32)."""
    device, fv100 = fv100_gpu
    scale = head_dim**-0.5
    q, k_lin, v_lin = _random_qkv(
        100, 300, num_heads, num_kv_heads, head_dim, head_dim, device
    )
    k_cache, v_cache, block_table = _paged_cache(k_lin, v_lin, 16, seed=5)
    sinks = _sinks(num_heads, device, seed=3)
    out = fv100.flash_attn_prefill_paged(
        q[None],
        k_cache,
        v_cache,
        block_table,
        torch.tensor([300], dtype=torch.int32, device=device),
        softmax_scale=scale,
        causal=True,
        sinks=sinks,
    )[0]
    torch.cuda.synchronize()
    ref = sdpa_reference(
        q, k_lin, v_lin, scale=scale, causal=True, window=FULL, sinks=sinks
    )
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
