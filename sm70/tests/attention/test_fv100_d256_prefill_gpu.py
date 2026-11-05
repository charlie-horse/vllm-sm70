# SPDX-License-Identifier: Apache-2.0
"""K05 GPU: D256 paged prefill over long prefixes and every kept D256 route.

Port of fork cf9492a tests/kernels/attention/test_sm70_flash_v100_splitkv_n.py.
The fork test exercised the external FA2 split-D/split-KV operators
(torch.ops._vllm_fa2_C.sm70_d256_splitd_*), which K05 does not carry
(Tier B drops the zhinianqin FA2 stack). Its shapes, a short chunk over a
very long prefix (Q64 over KV 70016, Q512 over KV 33024) at Hq6/Hkv1/D256,
now run through the lean Flash-V100 paged prefill, checked against an fp32
reference instead of against the unsplit dense operator.

The second test pins each D256 route the lean build keeps (launcher in
fused_mha_forward_paged.cu): BM32 phase (M >= 32, page % 16 == 0, no
window), low-SMEM contiguous (page 16), low-SMEM stride-268 with the
software pipeline (page 784), low-SMEM generic (other pages, or any page
with a window) and the generic kernel (M == 1).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _fv100_ref import FULL, paged_kv, prefill_reference  # noqa: E402

pytestmark = pytest.mark.gpu

HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5


def _run_case(
    fv100_gpu,
    *,
    q_len,
    kv_len,
    heads,
    page,
    causal=True,
    window=FULL,
    seed=0,
    atol,
    rtol=0.0,
):
    device, fv100 = fv100_gpu
    num_heads, num_kv_heads = heads
    k_cache, v_cache, block_table, _ = paged_kv(
        [kv_len], num_kv_heads, HEAD_DIM, page, device, seed
    )
    gen = torch.Generator().manual_seed(seed + 1)
    q = torch.randn((1, q_len, num_heads, HEAD_DIM), generator=gen).to(
        device, torch.float16
    )
    lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    out = fv100.flash_attn_prefill_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        lens,
        softmax_scale=SCALE,
        causal=causal,
        window_size=window,
    )
    torch.cuda.synchronize()
    ref = prefill_reference(
        q,
        k_cache,
        v_cache,
        block_table,
        lens,
        scale=SCALE,
        causal=causal,
        window=window,
        sinks=None,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("heads", [(6, 1), (8, 1)], ids=["gqa6", "gqa8"])
@pytest.mark.parametrize("page", [16, 784])
@pytest.mark.parametrize("query_len,kv_len", [(64, 70016), (512, 33024)])
@torch.inference_mode()
def test_d256_short_chunk_over_long_prefix(fv100_gpu, heads, page, query_len, kv_len):
    _run_case(
        fv100_gpu,
        q_len=query_len,
        kv_len=kv_len,
        heads=heads,
        page=page,
        seed=query_len,
        atol=5e-3,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "q_len,page,window",
    [
        (64, 16, FULL),  # BM32 phase
        (40, 32, FULL),  # BM32 phase, partial last query tile
        (8, 16, FULL),  # low-SMEM contiguous (page 16)
        (8, 784, FULL),  # low-SMEM stride-268 + software pipeline (page 784)
        (8, 32, FULL),  # low-SMEM generic
        (1, 16, FULL),  # generic kernel (M == 1)
        (64, 16, (127, 0)),  # low-SMEM with a window (window disables BM32)
    ],
    ids=[
        "bm32",
        "bm32-tail",
        "lowsmem-p16",
        "lowsmem-p784",
        "lowsmem-p32",
        "m1",
        "window",
    ],
)
@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["qwen38-tp2", "qwen36-tp2"])
@torch.inference_mode()
def test_d256_each_kept_route_matches_reference(fv100_gpu, q_len, page, window, heads):
    _run_case(
        fv100_gpu,
        q_len=q_len,
        kv_len=3000,
        heads=heads,
        page=page,
        window=window,
        seed=q_len + page,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("page", [16, 784])
@torch.inference_mode()
def test_d256_non_causal_bm32(fv100_gpu, page):
    """Non-causal (draft-style) prefill keeps the BM32 non-causal instance."""
    _run_case(
        fv100_gpu,
        q_len=64,
        kv_len=1000,
        heads=(8, 1),
        page=page,
        causal=False,
        seed=page,
        atol=1e-2,
        rtol=1e-2,
    )
