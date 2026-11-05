# SPDX-License-Identifier: Apache-2.0
"""K15 CPU checks of the fused q/k RMSNorm + RoPE + FP16 KV write.

The kernel runs through Triton's interpreter (fixture ``interp``), so this
checks the numerics and the KV-cache addressing on upstream v0.33.0's cache
layout without a GPU:
- against a float reference of the unfused route (norm, then NeoX RoPE), at
  the Flint TP2 (Hq 16 / Hkv 2) and TP1 (32 / 4) shapes, head dim 128;
- bit-identical cache bytes to upstream's own writer
  (triton_reshape_and_cache_flash, interpreted) fed the fused k/v, on the
  per-layer cache as upstream hands it over: logical (blocks, Hkv, page, 2D)
  with physical LBNHC (Flash-V100 and TRITON_ATTN) or LBHNC;
- the op's routing: direct write for TritonAttentionImpl and
  FlashAttnV100Impl (inherited writer), the backend's writer otherwise.

The fork's GPU version of these checks is test_k15_qk_norm_rope_kv_gpu.py.
The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent they show as skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

D = 128
EPS = 1e-6
K_FILL = 13.0
V_FILL = 17.0
MAX_POS = 4096
# Flint: 32 query heads, 4 KV heads. TP2 is the shipped config.
SHAPES = [
    pytest.param(16, 2, id="flint-tp2-hq16-hkv2"),
    pytest.param(32, 4, id="flint-tp1-hq32-hkv4"),
]


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu",
            "-k",
            "not test_interpreted_suite",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 100, summary
    assert "skipped" not in summary, summary


def _cos_sin(dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    freqs = torch.outer(torch.arange(MAX_POS, dtype=torch.float), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def _kv_cache(blocks, page, hkv, layout):
    """Per-layer cache as upstream passes it: logical (B, H, N, 2D)."""
    if layout == "LBNHC":
        kv = torch.empty(blocks, page, hkv, 2 * D, dtype=torch.float16)
        kv = kv.permute(0, 2, 1, 3)
    else:  # LBHNC
        kv = torch.empty(blocks, hkv, page, 2 * D, dtype=torch.float16)
    kv[..., :D].fill_(K_FILL)
    kv[..., D:].fill_(V_FILL)
    return kv


def _views(kv):
    return kv.transpose(1, 2).split(D, dim=-1)


def _make(tokens, hq, hkv, blocks, page, seed):
    gen = torch.Generator().manual_seed(seed)
    qkv = (torch.randn(tokens, (hq + 2 * hkv) * D, generator=gen) * 3).half()
    qw = (torch.randn(D, generator=gen) * 0.5 + 1).half()
    kw = (torch.randn(D, generator=gen) * 0.5 + 1).half()
    slots = torch.randperm(blocks * page, generator=gen)[:tokens].to(torch.int64)
    if tokens > 1:
        slots[-1] = -1  # CUDA-graph padding row: must not touch the cache.
    positions = torch.randint(0, MAX_POS, (tokens,), generator=gen)
    return qkv, qw, kw, slots, positions


def _ref_norm(x, w):
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * inv * w.float()).half()


def _ref_rope(x, cos_sin, positions):
    cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).half()


def _reference(qkv, qw, kw, positions, cos_sin, hq, hkv):
    tokens = qkv.shape[0]
    q_raw, k_raw, _ = qkv.split([hq * D, hkv * D, hkv * D], dim=-1)
    q = _ref_norm(q_raw.view(tokens, hq, D), qw)
    k = _ref_norm(k_raw.view(tokens, hkv, D), kw)
    if cos_sin is not None:
        q = _ref_rope(q, cos_sin, positions)
        k = _ref_rope(k, cos_sin, positions)
    return q.reshape(tokens, hq * D), k.reshape(tokens, hkv * D)


def _assert_fp16_close(actual, expected):
    # Same formula; only the FP32 reduction order differs. One FP16 ulp, and
    # nearly every element exact (FORK test_sm70_flint_qk_norm_rope_kv.py:76).
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)
    exact = (actual == expected).float().mean().item()
    assert exact > 0.97, exact


def _outputs(tokens, hq, hkv):
    q = torch.empty(tokens, hq * D, dtype=torch.float16)
    k = torch.empty(tokens, hkv * D, dtype=torch.float16)
    return q, k


@pytest.mark.parametrize("tokens", [1, 2, 7, 19])
@pytest.mark.parametrize("hq,hkv", SHAPES)
@pytest.mark.parametrize("rope", [None, torch.float16, torch.float32])
@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC"])
def test_fused_matches_reference_and_upstream_writer(
    interp, tokens, hq, hkv, rope, page, layout
):
    blocks = 8
    qkv, qw, kw, slots, positions = _make(tokens, hq, hkv, blocks, page, tokens)
    cos_sin = None if rope is None else _cos_sin(rope)
    exp_q, exp_k = _reference(qkv, qw, kw, positions, cos_sin, hq, hkv)

    kv = _kv_cache(blocks, page, hkv, layout)
    kc, vc = _views(kv)
    q, k = _outputs(tokens, hq, hkv)
    interp.qk.qk_norm_rope_kv(qkv, q, k, qw, kw, positions, cos_sin, EPS, kc, vc, slots)
    _assert_fp16_close(q, exp_q)
    _assert_fp16_close(k, exp_k)

    # Upstream's writer fed the fused k and the raw v produces the very same
    # cache bytes: same layout, same padding rule, V copied bit for bit.
    ref_kv = _kv_cache(blocks, page, hkv, layout)
    rk, rv = _views(ref_kv)
    scale = torch.ones(1)
    v = qkv[:, (hq + hkv) * D :]
    interp.reshape_and_cache(
        k.view(tokens, hkv, D),
        v.view(tokens, hkv, D),
        rk,
        rv,
        slots,
        "auto",
        scale,
        scale,
    )
    assert torch.equal(kv, ref_kv)

    # Untouched slots keep their fill; written slots hold the fused k.
    k_flat = kc.reshape(blocks * page, hkv * D)
    written = torch.zeros(blocks * page, dtype=torch.bool)
    written[slots[slots >= 0]] = True
    assert torch.all(k_flat[~written] == K_FILL)
    live = slots >= 0
    assert torch.equal(k_flat[slots[live]], k[live])


def test_negative_positions_wrap_like_indexing(interp):
    tokens, hq, hkv = 2, 16, 2
    qkv, qw, kw, _, _ = _make(tokens, hq, hkv, 4, 16, 5)
    positions = torch.tensor([-1, 7])
    cos_sin = _cos_sin(torch.float16)
    exp_q, exp_k = _reference(qkv, qw, kw, positions, cos_sin, hq, hkv)
    q, k = _outputs(tokens, hq, hkv)
    interp.qk.qk_norm_rope_kv(qkv, q, k, qw, kw, positions, cos_sin, EPS)
    _assert_fp16_close(q, exp_q)
    _assert_fp16_close(k, exp_k)


def test_tokens_past_slot_mapping_are_not_written(interp):
    """Upstream's writer launches one program per slot_mapping row; rows of
    qkv beyond it (padding) never reach the cache."""
    tokens, hq, hkv, blocks, page = 3, 16, 2, 4, 16
    qkv, qw, kw, _, positions = _make(tokens, hq, hkv, blocks, page, 9)
    slots = torch.tensor([5], dtype=torch.int64)
    kv = _kv_cache(blocks, page, hkv, "LBNHC")
    kc, vc = _views(kv)
    q, k = _outputs(tokens, hq, hkv)
    interp.qk.qk_norm_rope_kv(qkv, q, k, qw, kw, positions, None, EPS, kc, vc, slots)
    k_flat = kc.reshape(blocks * page, hkv * D)
    assert torch.equal(k_flat[5], k[0])
    keep = torch.ones(blocks * page, dtype=torch.bool)
    keep[5] = False
    assert torch.all(k_flat[keep] == K_FILL)


def test_zero_tokens_is_a_no_op(interp):
    qkv = torch.empty(0, 20 * D, dtype=torch.float16)
    q, k = _outputs(0, 16, 2)
    w = torch.ones(D, dtype=torch.float16)
    interp.qk.qk_norm_rope_kv(
        qkv, q, k, w, w, torch.empty(0, dtype=torch.long), None, EPS
    )


# -- the custom op: routing between the direct write and the backend writer --


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, layer, key, value, kv_cache, slot_mapping):
        self.calls.append((key.clone(), value.clone(), kv_cache, slot_mapping))


def _impl(cls, **attrs):
    impl = object.__new__(cls)
    for name, value in attrs.items():
        setattr(impl, name, value)
    return impl


def _run_op(
    interp,
    monkeypatch,
    impl,
    *,
    kv_cache_dtype="auto",
    kv=None,
    slots=None,
    tokens=2,
    hq=16,
    hkv=2,
):
    import vllm.model_executor.layers.attention.attention as attention_mod

    blocks, page = 4, 16
    qkv, qw, kw, default_slots, positions = _make(tokens, hq, hkv, blocks, page, 3)
    if kv is None:
        kv = _kv_cache(blocks, page, hkv, "LBNHC")
    if slots is None:
        slots = default_slots
    layer = SimpleNamespace(impl=impl, kv_cache_dtype=kv_cache_dtype)
    monkeypatch.setattr(
        attention_mod,
        "get_attention_context",
        lambda name: (None, layer, kv, slots),
    )
    q, k = _outputs(tokens, hq, hkv)
    interp.qk.qk_norm_rope_kv_cache_update(
        qkv, q, k, qw, kw, positions, _cos_sin(torch.float16), EPS, "layers.0.attn"
    )
    return SimpleNamespace(qkv=qkv, q=q, k=k, kv=kv, slots=slots, hq=hq, hkv=hkv)


def _triton_impl_classes():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    classes = [pytest.param(TritonAttentionImpl, id="TRITON_ATTN")]
    try:
        from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl
    except Exception as exc:  # pragma: no cover - import must work on CPU
        pytest.fail(f"FlashAttnV100Impl did not import: {exc}")
    classes.append(pytest.param(FlashAttnV100Impl, id="FLASH_ATTN_V100"))
    return classes


@pytest.mark.parametrize("impl_cls", _triton_impl_classes())
def test_op_writes_cache_directly_on_triton_writers(interp, monkeypatch, impl_cls):
    recorder = _Recorder()
    impl = _impl(impl_cls)
    impl.do_kv_cache_update = recorder  # would only be used on fallback
    out = _run_op(interp, monkeypatch, impl)
    assert recorder.calls == []
    kc, vc = _views(out.kv)
    live = out.slots >= 0
    k_flat = kc.reshape(-1, out.hkv * D)
    v_flat = vc.reshape(-1, out.hkv * D)
    assert torch.equal(k_flat[out.slots[live]], out.k[live])
    v = out.qkv[:, (out.hq + out.hkv) * D :]
    assert torch.equal(v_flat[out.slots[live]], v[live])


def test_op_keeps_backend_writer_for_other_backends(interp, monkeypatch):
    class OtherImpl:
        def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
            raise AssertionError("replaced per instance below")

    recorder = _Recorder()
    impl = OtherImpl()
    impl.do_kv_cache_update = recorder
    out = _run_op(interp, monkeypatch, impl)
    assert len(recorder.calls) == 1
    key, value, kv_cache, slot_mapping = recorder.calls[0]
    assert kv_cache is out.kv and slot_mapping is out.slots
    assert torch.equal(key, out.k.view(-1, out.hkv, D))
    v = out.qkv[:, (out.hq + out.hkv) * D :]
    assert torch.equal(value, v.reshape(-1, out.hkv, D))
    # The kernel itself left the cache alone.
    assert torch.all(out.kv[..., :D] == K_FILL)
    assert torch.all(out.kv[..., D:] == V_FILL)


@pytest.mark.parametrize(
    "case",
    [
        dict(attrs=dict(_is_per_token_head_quant=True)),
        dict(attrs=dict(_set_kv_ptr_range=True)),
        dict(kv_cache_dtype="fp8_e4m3"),
    ],
    ids=["per-token-head-quant", "rocm-ptr-range", "fp8-kv"],
)
def test_op_falls_back_when_the_cache_is_not_plain_fp16(interp, monkeypatch, case):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    recorder = _Recorder()
    impl = _impl(TritonAttentionImpl, **case.get("attrs", {}))
    impl.do_kv_cache_update = recorder
    _run_op(
        interp,
        monkeypatch,
        impl,
        kv_cache_dtype=case.get("kv_cache_dtype", "auto"),
    )
    assert len(recorder.calls) == 1


def test_op_skips_the_cache_while_profiling(interp, monkeypatch):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    recorder = _Recorder()
    impl = _impl(TritonAttentionImpl)
    impl.do_kv_cache_update = recorder
    import vllm.model_executor.layers.attention.attention as attention_mod

    tokens, hq, hkv = 2, 16, 2
    qkv, qw, kw, _, positions = _make(tokens, hq, hkv, 4, 16, 1)
    layer = SimpleNamespace(impl=impl, kv_cache_dtype="auto")
    empty = torch.empty(0, dtype=torch.float16)
    monkeypatch.setattr(
        attention_mod, "get_attention_context", lambda n: (None, layer, empty, None)
    )
    q, k = _outputs(tokens, hq, hkv)
    interp.qk.qk_norm_rope_kv_cache_update(
        qkv, q, k, qw, kw, positions, None, EPS, "layers.0.attn"
    )
    assert recorder.calls == []
    exp_q, exp_k = _reference(qkv, qw, kw, positions, None, hq, hkv)
    _assert_fp16_close(q, exp_q)
    _assert_fp16_close(k, exp_k)


def test_custom_op_schema_mutates_only_q_and_k_out():
    import vllm_sm70.flint.qk_norm_rope_kv as mod

    op = getattr(torch.ops.vllm, mod.OP_NAME).default
    written = [
        a.name
        for a in op._schema.arguments
        if a.alias_info is not None and a.alias_info.is_write
    ]
    assert written == ["q_out", "k_out"]
    assert len(op._schema.returns) == 0


def test_attention_after_fused_write_skips_the_kv_update(monkeypatch):
    import vllm.model_executor.layers.attention.attention as attention_mod
    import vllm_sm70.flint.qk_norm_rope_kv as mod

    seen = []

    def fake_attention(query, key, value, output, layer_name, **kwargs):
        seen.append((query.shape, key.shape, value.shape, output.shape, layer_name))
        output.fill_(1.0)

    def no_kv_update(*args, **kwargs):
        raise AssertionError("the fused route must not write the cache again")

    monkeypatch.setattr(attention_mod, "unified_attention_with_output", fake_attention)
    monkeypatch.setattr(attention_mod, "unified_kv_cache_update", no_kv_update)
    attn = SimpleNamespace(
        num_heads=16,
        num_kv_heads=2,
        head_size=D,
        head_size_v=D,
        layer_name="layers.0.attn",
        use_direct_call=True,
    )
    tokens = 3
    q = torch.zeros(tokens, 16 * D, dtype=torch.float16)
    k = torch.zeros(tokens, 2 * D, dtype=torch.float16)
    v = torch.zeros(tokens, 2 * D, dtype=torch.float16)
    out = mod.attention_after_fused_write(attn, q, k, v)
    assert out.shape == (tokens, 16 * D) and torch.all(out == 1.0)
    assert seen == [
        (
            (tokens, 16, D),
            (tokens, 2, D),
            (tokens, 2, D),
            (tokens, 16, D),
            "layers.0.attn",
        )
    ]
