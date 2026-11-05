# SPDX-License-Identifier: Apache-2.0
"""K15 GPU checks: fused q/k RMSNorm + RoPE + FP16 paged KV write vs the
unfused upstream ops, on an SM70 GPU.

Port of FORK tests/kernels/core/test_sm70_flint_qk_norm_rope_kv.py (197).
Changes: the cache is upstream v0.33.0's per-layer layout (logical
(blocks, Hkv, page, 2D), physical LBNHC, viewed with transpose(1, 2).split
as in TritonAttentionImpl.do_kv_cache_update) instead of the fork's
[blocks, 2, page, Hkv, D] + unbind(1); an FP32 RoPE table is also checked
(upstream Flint builds it in FP32); shapes are Flint TP2 (16/2) and TP1
(32/4). The kernel is pure Triton, so _sm70_C is not needed.

Run from the repo root: pytest -m gpu sm70/tests/flint
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

D = 128
EPS = 1e-6
K_FILL = 13.0
V_FILL = 17.0
SHAPES = [
    pytest.param(16, 2, id="flint-tp2-hq16-hkv2"),
    pytest.param(32, 4, id="flint-tp1-hq32-hkv4"),
]


@pytest.fixture
def sm70():
    if not torch.cuda.is_available():
        pytest.skip("requires an SM70 GPU")
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (7, 0):
            torch.cuda.set_device(index)
            from vllm_sm70.flint import qk_norm_rope_kv as mod

            return mod
    pytest.skip("requires an SM70 GPU")


def _rope(dtype, max_pos):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    with set_current_vllm_config(VllmConfig()):
        return RotaryEmbedding(D, D, max_pos, 10000.0, True, dtype).cuda()


def _kv_cache(blocks, page, hkv):
    # Upstream per-layer cache: logical (B, H, N, 2D), physical LBNHC.
    kv = torch.empty(blocks, page, hkv, 2 * D, device="cuda", dtype=torch.float16)
    kv = kv.permute(0, 2, 1, 3)
    kv[..., :D].fill_(K_FILL)
    kv[..., D:].fill_(V_FILL)
    return kv


def _views(kv):
    return kv.transpose(1, 2).split(D, dim=-1)


def _reference_norm(x, w):
    # vllm rms_norm_kernel: FP32 variance, fp16(x * rsqrt(var + eps) * w).
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * inv * w.float()).half()


def _reference_rope(x, cos_sin, positions):
    # The compiled NeoX RoPE: FP32 arithmetic on FP16 inputs, one rounding.
    cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).half()


def _make(tokens, hq, hkv, blocks, page, seed):
    torch.manual_seed(seed)
    qkv = torch.randn(tokens, (hq + 2 * hkv) * D, device="cuda").half() * 3
    qw = (torch.randn(D, device="cuda") * 0.5 + 1).half()
    kw = (torch.randn(D, device="cuda") * 0.5 + 1).half()
    kv = _kv_cache(blocks, page, hkv)
    perm = torch.randperm(blocks * page, device="cuda")[:tokens]
    slots = perm.to(torch.int64)
    if tokens > 1:
        slots[-1] = -1  # CUDA-graph padding row: must not touch the cache.
    return qkv, qw, kw, kv, slots


def _unfused(qkv, qw, kw, positions, cos_sin, kv, slots, hq, hkv):
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    tokens = qkv.shape[0]
    q_raw, k_raw, v = qkv.split([hq * D, hkv * D, hkv * D], dim=-1)
    q = torch.empty(tokens, hq, D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv, D, device="cuda", dtype=torch.float16)
    ops.rms_norm(q, q_raw.reshape(tokens, hq, D), qw, EPS)
    ops.rms_norm(k, k_raw.reshape(tokens, hkv, D), kw, EPS)
    if cos_sin is not None:
        q = _reference_rope(q, cos_sin, positions)
        k = _reference_rope(k, cos_sin, positions)
    kc, vc = _views(kv)
    scale = torch.ones(1, device="cuda")
    triton_reshape_and_cache_flash(
        k, v.reshape(tokens, hkv, D), kc, vc, slots, "auto", scale, scale
    )
    return q.reshape(tokens, hq * D), k.reshape(tokens, hkv * D)


def _assert_fp16_close(actual, expected):
    # Same formula; only the FP32 reduction order and FMA contraction differ.
    # Allow one FP16 ulp, and require nearly every element to match exactly.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)
    exact = (actual == expected).float().mean().item()
    assert exact > 0.97, exact


@pytest.mark.parametrize("tokens", [1, 2, 7, 300])
@pytest.mark.parametrize("hq,hkv", SHAPES)
@pytest.mark.parametrize("rope_dtype", [None, torch.float16, torch.float32])
@pytest.mark.parametrize("page", [16, 32])
@torch.inference_mode()
def test_fused_matches_unfused(sm70, tokens, hq, hkv, rope_dtype, page):
    blocks = 64
    qkv, qw, kw, kv_ref, slots = _make(tokens, hq, hkv, blocks, page, tokens)
    kv = kv_ref.clone()  # preserve_format keeps the LBNHC strides
    assert kv.stride() == kv_ref.stride()
    positions = torch.randint(0, 262144, (tokens,), device="cuda")
    cos_sin = None
    if rope_dtype is not None:
        cos_sin = _rope(rope_dtype, 262144).cos_sin_cache
    exp_q, exp_k = _unfused(qkv, qw, kw, positions, cos_sin, kv_ref, slots, hq, hkv)

    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    kc, vc = _views(kv)
    sm70.qk_norm_rope_kv(qkv, q, k, qw, kw, positions, cos_sin, EPS, kc, vc, slots)
    _assert_fp16_close(q, exp_q)
    _assert_fp16_close(k, exp_k)
    # V is copied bit for bit; untouched slots keep their fill.
    assert torch.equal(kv[..., D:], kv_ref[..., D:])
    written = torch.zeros(blocks * page, dtype=torch.bool, device="cuda")
    written[slots[slots >= 0]] = True
    k_flat = kc.reshape(blocks * page, hkv * D)
    assert torch.all(k_flat[~written] == K_FILL)
    ref_k_flat = _views(kv_ref)[0].reshape(blocks * page, hkv * D)
    _assert_fp16_close(k_flat, ref_k_flat)

    # Layout is bit-identical: the stock writer fed the fused k/v produces
    # exactly the fused cache.
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    layout = _kv_cache(blocks, page, hkv)
    lk, lv = _views(layout)
    scale = torch.ones(1, device="cuda")
    v = qkv[:, (hq + hkv) * D :]
    triton_reshape_and_cache_flash(
        k.view(tokens, hkv, D),
        v.reshape(tokens, hkv, D),
        lk,
        lv,
        slots,
        "auto",
        scale,
        scale,
    )
    assert torch.equal(layout, kv)


@torch.inference_mode()
def test_fused_matches_eager_modules(sm70):
    """Looser check against the eager PyTorch modules (which round each op)."""
    tokens, hq, hkv = 2, 16, 2
    qkv, qw, kw, kv, slots = _make(tokens, hq, hkv, 8, 16, 7)
    positions = torch.tensor([5, 4097], device="cuda")
    rope = _rope(torch.float16, 262144)
    q_raw, k_raw, _ = qkv.split([hq * D, hkv * D, hkv * D], dim=-1)
    q_ref, k_ref = rope.forward_native(
        positions,
        _reference_norm(q_raw.reshape(tokens, hq, D), qw).reshape(tokens, -1),
        _reference_norm(k_raw.reshape(tokens, hkv, D), kw).reshape(tokens, -1),
    )
    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    sm70.qk_norm_rope_kv(qkv, q, k, qw, kw, positions, rope.cos_sin_cache, EPS)
    torch.testing.assert_close(q, q_ref, rtol=4e-3, atol=8e-3)
    torch.testing.assert_close(k, k_ref, rtol=4e-3, atol=8e-3)


@torch.inference_mode()
def test_graph_replay_reads_live_inputs(sm70):
    tokens, hq, hkv = 2, 16, 2
    qkv, qw, kw, kv, slots = _make(tokens, hq, hkv, 8, 16, 3)
    slots[-1] = 9
    positions = torch.tensor([0, 1], device="cuda")
    rope = _rope(torch.float16, 4096)
    q = torch.empty(tokens, hq * D, device="cuda", dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, device="cuda", dtype=torch.float16)
    kc, vc = _views(kv)

    def run():
        sm70.qk_norm_rope_kv(
            qkv, q, k, qw, kw, positions, rope.cos_sin_cache, EPS, kc, vc, slots
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    qkv.zero_()
    slots.fill_(-1)
    kv[..., :D].fill_(K_FILL)
    kv[..., D:].fill_(V_FILL)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.count_nonzero(q) == torch.count_nonzero(k) == 0
    assert torch.all(kv[..., :D] == K_FILL) and torch.all(kv[..., D:] == V_FILL)
