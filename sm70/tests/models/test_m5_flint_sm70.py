# SPDX-License-Identifier: Apache-2.0
"""Tests for M5 flint-sm70-model (vllm_sm70/models/flint_sm70/).

CPU: fp16_rescale against frozen copies of the fork's math; a torch
  oracle of K15's qk_norm_rope_kv (vllm_sm70/flint) against the unfused
  route and against upstream's Triton KV-cache layout through K15's
  _direct_cache_views (Flint's fused-route admission included);
  FlintSm70ModelConfig refusals; registry wiring at cc 7.0 vs 8.0; and the
  VLLM_SM70_FLINT_QK_ROPE_KV kill switch.
GPU (``-m gpu``): GPU test plan T1-T8. Tests marked "2 GPUs" need both V100s
  and the ``vllm`` service stopped; they run in the 00:00-06:00 window.
  Paths come from env:
    FLINT_SM70_CHECKPOINT  the ModelOpt NVFP4 Flint checkpoint dir
    FLINT_SM70_GOLDEN      JSON of fork cf9492a greedy records (T5); c4130:
                           ~/vllm-sm70-work/bea/golden/flint_cf9492a_greedy64.json
    FLINT_SM70_T4_SEQS_DIR, FLINT_SM70_T4_ENVELOPE, FLINT_SM70_FORK_SCORES_DIR,
    FLINT_SM70_OURS_NOISE  teacher-forced T4a-T4d/T7 inputs (see the T4 section
                           and docs/m5-calibration-nov7.md)
    M5_MUTATION            planted bug for sm70/scripts/m5_mutation_check.sh
"""

import functools
import importlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from types import SimpleNamespace

import pytest

# Deterministic TurboMind kernel choice for every engine these tests start
# (Quinn/Astro, Nov 6): small-shape autotune picks kernels per instance by
# timing. Set before any engine spawns; M5_ALLOW_TUNING=1 opts out.
if os.environ.get("M5_ALLOW_TUNING") != "1":
    for _q in ("NVFP4", "FP8", "AWQ", "MXFP4"):
        os.environ.setdefault(f"VLLM_SM70_{_q}_TUNE_SMALL_SHAPES", "0")

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch

import vllm.platforms
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.models import flint_sm70
from vllm_sm70.models.flint_sm70 import fp16_rescale as fr
from vllm_sm70.flint import QK_ROPE_KV_OP_NAME, QK_ROPE_KV_SPLITTING_OP
from vllm_sm70.flint import qk_norm_rope_kv as qk  # K15 (#54)

# --------------------------------------------------------------------------
# Frozen copies of the fork's FP16 math (FORK cf9492a
# vllm/models/flint/model.py:95-107 and :696-753), kept verbatim as the
# oracle for fp16_rescale.py.
# --------------------------------------------------------------------------
_F_MOE_UP = (".mlp.experts.", ".mlp.shared_expert.")
_F_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_F_FP16_MAX = torch.finfo(torch.float16).max


def _fork_rebalance_moe_intermediate(
    weights: Iterable[tuple[str, torch.Tensor]], shift: int
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if any(block in name for block in _F_MOE_UP):
            dense = name.endswith(".weight") and tensor.is_floating_point()
            if ".up_proj." in name and (name.endswith(".weight_scale") or dense):
                tensor = (tensor.float() / shift).to(tensor.dtype)
            elif ".down_proj." in name and (name.endswith(".weight_scale_2") or dense):
                tensor = (tensor.float() * shift).to(tensor.dtype)
        yield name, tensor


def _fork_rescale_fp16_stream(
    weights: Iterable[tuple[str, torch.Tensor]], start_layer: int, shift: int
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        match = _F_LAYER_INDEX.search(name)
        if (
            match is not None
            and int(match.group(1)) >= start_layer
            and ".mlp." in name
            and ".down_proj." in name
            and (
                name.endswith(".weight_scale_2")
                or (name.endswith(".weight") and tensor.is_floating_point())
            )
        ):
            tensor = (tensor.float() / shift).to(tensor.dtype)
        yield name, tensor


def _fork_check_fp16_range(weights):
    for name, tensor in weights:
        if tensor.dtype in (torch.bfloat16, torch.float32) and tensor.numel():
            amax = tensor.detach().abs().amax()
            if not bool(amax <= _F_FP16_MAX):
                raise ValueError(f"{name}: does not fit in FP16")
        yield name, tensor


def _checkpoint_stream(seed=0):
    """Names in Flint's checkpoint layout, NVFP4 and BF16 mixed, across layers
    below, at and above the shift layer."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for layer in (0, 45, 46, 47, 48):
        for block in ("mlp.experts.3", "mlp.shared_expert", "mlp"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"model.layers.{layer}.{block}.{proj}"
                out.append(
                    (
                        f"{base}.weight",
                        torch.randint(0, 255, (8, 8), generator=g).to(torch.uint8),
                    )
                )
                out.append(
                    (
                        f"{base}.weight_scale",
                        (torch.rand(8, 1, generator=g) * 400 + 1).to(
                            torch.float8_e4m3fn
                        ),
                    )
                )
                out.append(
                    (f"{base}.weight_scale_2", torch.rand((), generator=g) * 1e-3)
                )
                out.append(
                    (f"{base}.bf16.weight", torch.randn(8, 8, generator=g).bfloat16())
                )
        out.append(
            (
                f"model.layers.{layer}.self_attn.o_proj.weight_scale_2",
                torch.rand((), generator=g),
            )
        )
    return out


def _same(a, b):
    assert [n for n, _ in a] == [n for n, _ in b]
    for (n, x), (_, y) in zip(a, b):
        assert x.dtype == y.dtype, n
        assert torch.equal(
            x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x,
            y.view(torch.uint8) if y.dtype == torch.float8_e4m3fn else y,
        ), n


def test_fp16_pipeline_equals_fork_math():
    stream = _checkpoint_stream()
    fork = list(
        _fork_check_fp16_range(
            _fork_rescale_fp16_stream(
                _fork_rebalance_moe_intermediate(iter(stream), 16), 46, 8
            )
        )
    )
    ours = list(fr.fp16_weight_pipeline(iter(stream)))
    _same(ours, fork)
    assert (
        fr.MOE_INTERMEDIATE_SHIFT,
        fr.FP16_STREAM_SHIFT_LAYER,
        fr.FP16_STREAM_SHIFT,
    ) == (16, 46, 8)


@pytest.mark.parametrize(
    "fn, fork_fn, args",
    [
        (fr.rebalance_moe_intermediate, _fork_rebalance_moe_intermediate, (16,)),
        (fr.rescale_fp16_stream, _fork_rescale_fp16_stream, (46, 8)),
    ],
)
def test_each_fp16_filter_equals_fork(fn, fork_fn, args):
    stream = _checkpoint_stream(seed=1)
    _same(list(fn(iter(stream), *args)), list(fork_fn(iter(stream), *args)))


def test_check_fp16_range_rejects_overflow():
    bad = [("model.layers.0.mlp.down_proj.weight", torch.tensor([7e4]).bfloat16())]
    with pytest.raises(ValueError, match="does not fit in FP16"):
        list(fr.check_fp16_range(bad))
    list(fr.check_fp16_range([("x", torch.tensor([6e4]).bfloat16())]))


@pytest.mark.parametrize(
    "layer, expected",
    [
        (0, (None, None)),
        (45, (None, None)),
        (46, (1 / 8, None)),
        (47, (None, 1 / 8)),
        (48, (None, 1 / 8)),
    ],
)
def test_stream_scales(layer, expected):
    """Residual /8 at layer 46, attention output /8 above it
    (FORK vllm/models/flint/model.py:402-408)."""
    assert fr.stream_scales(layer, True) == expected
    assert fr.stream_scales(layer, False) == (None, None)


def test_lm_head_scale_folds_in_fp16_and_scales_logits_otherwise():
    """FORK vllm/models/flint/model.py:647-668."""
    sc = fr.LmHeadScale()
    rest = list(
        sc.intercept(
            iter(
                [
                    ("lm_head.weight_scale_2", torch.tensor(1e-4)),
                    ("lm_head.weight", torch.ones(2)),
                ]
            )
        )
    )
    assert [n for n, _ in rest] == ["lm_head.weight"] and sc.scale == pytest.approx(
        1e-4
    )
    head = SimpleNamespace(
        weight=torch.nn.Parameter(torch.full((4, 3), 2.0).half(), requires_grad=False)
    )
    lp = SimpleNamespace(scale=1.0)
    sc.apply(head, lp)
    assert lp.scale == 1.0
    torch.testing.assert_close(
        head.weight.float(), torch.full((4, 3), 2e-4), rtol=1e-3, atol=0
    )
    head32 = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    )
    sc.apply(head32, lp)
    assert lp.scale == pytest.approx(1e-4) and torch.equal(
        head32.weight, torch.ones(2, 2)
    )
    big = fr.LmHeadScale()
    big.scale = 1e3
    with pytest.raises(ValueError, match="exceeds the FP16 range"):
        big.apply(
            SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.full((1, 1), 100.0).half(), requires_grad=False
                )
            ),
            lp,
        )
    fr.LmHeadScale().apply(None, lp)  # non-last PP rank: no-op


# --------------------------------------------------------------------------
# K15 qk_norm_rope_kv: torch oracle and the upstream Triton cache layout
# --------------------------------------------------------------------------
HQ, HKV, D, PAGE = 4, 2, 128, 16


def _qk_oracle(
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
    """Pure-torch oracle with K15 ``qk.qk_norm_rope_kv``'s signature (CPU and
    GPU) and the kernel's rounding points (FORK
    vllm/models/flint/qk_norm_rope_kv.py:65-125): FP32 norm, one FP16
    rounding, RoPE in FP32 on the rounded values, one FP16 rounding. The
    unfused route (RMSNorm + NeoX RotaryEmbedding +
    triton_reshape_and_cache_flash) is expected to match it."""
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


def _cos_sin(max_pos=64, dtype=torch.float16):
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    freqs = torch.outer(torch.arange(max_pos).float(), inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)


def _inputs(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(n, (HQ + 2 * HKV) * D, generator=g).half()
    qw = (torch.rand(D, generator=g) + 0.5).half()
    kw = (torch.rand(D, generator=g) + 0.5).half()
    pos = torch.randint(0, 64, (n,), generator=g)
    return qkv, qw, kw, pos


@pytest.mark.parametrize("rope", [True, False])
def test_reference_matches_unfused_route(rope):
    """The fused route's numerics
    (FORK vllm/models/flint/qk_norm_rope_kv.py:10-15) equal the unfused route:
    per-head RMSNorm rounded to fp16, then NeoX RoPE on the rounded values."""
    qkv, qw, kw, pos = _inputs()
    n = qkv.shape[0]
    cs = _cos_sin() if rope else None
    q = torch.empty(n, HQ * D, dtype=torch.half)
    k = torch.empty(n, HKV * D, dtype=torch.half)
    _qk_oracle(qkv, q, k, qw, kw, pos, cs, 1e-6)

    def norm(x, w):
        x = x.float()
        return (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        ).half()

    qh = norm(qkv[:, : HQ * D].reshape(n, HQ, D), qw)
    kh = norm(qkv[:, HQ * D : (HQ + HKV) * D].reshape(n, HKV, D), kw)
    if rope:
        cos, sin = cs[pos, : D // 2].float(), cs[pos, D // 2 :].float()

        def neox(x):
            x1, x2 = x[..., : D // 2].float(), x[..., D // 2 :].float()
            c, s = cos[:, None], sin[:, None]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).half()

        qh, kh = neox(qh), neox(kh)
    assert torch.equal(q, qh.reshape(n, -1)) and torch.equal(k, kh.reshape(n, -1))


def _triton_impl():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl.head_size = D
    impl._is_per_token_head_quant = False
    impl._set_kv_ptr_range = False
    return impl


def _cache_views(kv):
    """The K/V views K15's op writes for a TRITON_ATTN (or Flash-V100) layer:
    ``qk._direct_cache_views`` on a [B, Hkv, N, 2*D] layer cache."""
    layer = SimpleNamespace(impl=_triton_impl(), kv_cache_dtype="auto")
    slots = torch.zeros(1, dtype=torch.long, device=kv.device)
    views = qk._direct_cache_views(layer, kv, slots, kv.shape[1], D)
    assert views is not None
    return views


def test_cache_write_uses_upstream_triton_layout():
    """kv_cache is [blocks, Hkv, page, 2*D]
    (UP vllm/v1/attention/backends/triton_attn.py:598-599); K goes to
    [..., :D] and V to [..., D:], at slot = block*page + offset. Padded
    slots (-1) are skipped."""
    qkv, qw, kw, pos = _inputs(n=4)
    kv_cache = torch.zeros(3, HKV, PAGE, 2 * D, dtype=torch.half)
    slots = torch.tensor([0, 17, -1, 47])
    layer = SimpleNamespace(impl=_triton_impl(), kv_cache_dtype="auto")
    views = qk._direct_cache_views(layer, kv_cache, slots, HKV, D)
    assert views is not None
    for v in views:
        assert v.shape == (3, PAGE, HKV, D) and v.data_ptr() >= kv_cache.data_ptr()
    q = torch.empty(4, HQ * D, dtype=torch.half)
    k = torch.empty(4, HKV * D, dtype=torch.half)
    _qk_oracle(qkv, q, k, qw, kw, pos, _cos_sin(), 1e-6, *views, slots)
    vv = qkv[:, (HQ + HKV) * D :].reshape(4, HKV, D)
    kk = k.reshape(4, HKV, D)
    written = torch.zeros_like(kv_cache, dtype=torch.bool)
    for t, s in enumerate(slots.tolist()):
        if s < 0:
            continue
        b, off = divmod(s, PAGE)
        assert torch.equal(kv_cache[b, :, off, :D], kk[t])
        assert torch.equal(kv_cache[b, :, off, D:], vv[t])
        written[b, :, off] = True
    assert not kv_cache[~written].any()  # nothing else touched


def _layer_cache(layout_name, blocks, layers=2, layer=1, device="cpu"):
    """A per-layer [B, H, N, 2*D] KV view with UP's physical stride order.

    UP allocates one buffer per KV-cache group in ``KVCacheLayout`` stride
    order over logical [L, B, H, N, C] (UP vllm/v1/kv_cache_layout.py:15-28)
    and hands each layer a logical [B, H, N, C] view
    (vllm/v1/kv_cache_interface.py:418-427, create_kv_cache_views). TRITON_ATTN
    accepts any layout: it doesn't override supported_kv_cache_layouts, so the
    base returns None (vllm/v1/attention/backend.py:371-375). Permuted layouts
    such as LBNHC reach do_kv_cache_update as non-contiguous views."""
    from vllm.v1.kv_cache_layout import KVCacheLayout

    order = KVCacheLayout[layout_name].stride_order
    logical = (layers, blocks, HKV, PAGE, 2 * D)
    physical = torch.zeros([logical[i] for i in order], dtype=torch.half, device=device)
    inverse = [order.index(i) for i in range(5)]
    return physical.permute(*inverse)[layer]


LAYOUTS = ["LBHNC", "LBNHC", "BLHNC", "BLNHC"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_cache_write_honours_permuted_layouts(layout):
    """The op takes block/page/head strides from the views and needs only a
    unit inner stride (D), so a permuted (LBNHC-style) layer view gets the
    same logical cache as the identity layout, -1 padding included."""
    qkv, qw, kw, pos = _inputs(n=6, seed=3)
    slots = torch.tensor([0, 19, -1, 47, 5, -1])
    ref = _layer_cache("LBHNC", 3)
    cache = _layer_cache(layout, 3)
    if layout != "LBHNC":
        assert not cache.is_contiguous()
    layer = SimpleNamespace(impl=_triton_impl(), kv_cache_dtype="auto")
    for kv in (ref, cache):
        views = qk._direct_cache_views(layer, kv, slots, HKV, D)
        assert views is not None  # admitted: the kernel honours the strides
        q = torch.empty(6, HQ * D, dtype=torch.half)
        k = torch.empty(6, HKV * D, dtype=torch.half)
        _qk_oracle(qkv, q, k, qw, kw, pos, _cos_sin(), 1e-6, *views, slots)
    assert torch.equal(cache, ref)
    assert ref.any()


def test_direct_cache_views_refuses_other_layouts():
    kv = torch.zeros(2, HKV, PAGE, 2 * D, dtype=torch.half)
    slots = torch.tensor([0])
    impl = _triton_impl()
    ok = SimpleNamespace(impl=impl, kv_cache_dtype="auto")
    assert qk._direct_cache_views(ok, kv, slots, HKV, D) is not None
    fp8 = SimpleNamespace(impl=impl, kv_cache_dtype="fp8")
    assert qk._direct_cache_views(fp8, kv, slots, HKV, D) is None
    other = SimpleNamespace(impl=_OtherImpl(), kv_cache_dtype="auto")
    assert qk._direct_cache_views(other, kv, slots, HKV, D) is None  # CUSTOM backend
    assert qk._direct_cache_views(ok, kv.float(), slots, HKV, D) is None
    impl._is_per_token_head_quant = True
    assert qk._direct_cache_views(ok, kv, slots, HKV, D) is None


class _OtherImpl:
    """A backend with its own KV writer (e.g. Kai's CUSTOM Flash-V100)."""

    head_size = D

    def do_kv_cache_update(self, *args):
        pass


def _neox_rope():
    """A real RotaryEmbedding instance without the CustomOp constructor (which
    needs a current vllm config): the admission check reads only its type,
    neox flag, sizes and cache dtype."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rope = RotaryEmbedding.__new__(RotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.head_size = rope.rotary_dim = 128
    rope.is_neox_style = True
    rope.cos_sin_cache = _cos_sin(16)
    return rope


def _rejection(**kw):
    args = dict(
        enabled=True,
        is_sm70=True,
        dtype=torch.float16,
        head_dim=128,
        q_eps=1e-6,
        k_eps=1e-6,
        rotary_emb=_neox_rope(),
        kv_cache_dtype="auto",
        attn_type="decoder",
        calculate_kv_scales=False,
        kv_sharing=False,
        backend_writes_kv_in_forward=False,
    )
    args.update(kw)
    return qk.fused_route_rejection(**args)


def test_fused_route_admission():
    assert _rejection() is None
    assert _rejection(rotary_emb=None) is None  # full-attention layers
    assert "VLLM_SM70_FLINT_QK_ROPE_KV=0" in _rejection(enabled=False)
    assert "not SM70" in _rejection(is_sm70=False)
    assert "bfloat16" in _rejection(dtype=torch.bfloat16)
    assert "kv_cache_dtype" in _rejection(kv_cache_dtype="fp8")
    assert "writes the KV cache" in _rejection(backend_writes_kv_in_forward=True)
    assert "different epsilons" in _rejection(k_eps=1e-5)


def test_kill_switch_disables_fusion(monkeypatch):
    from vllm_sm70.config import clear_cache, sm70_config

    clear_cache()
    assert sm70_config(None).flint_qk_rope_kv is True
    monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", "0")
    assert sm70_config(None).flint_qk_rope_kv is False
    reason = _rejection(enabled=sm70_config(None).flint_qk_rope_kv)
    assert reason is not None and "VLLM_SM70_FLINT_QK_ROPE_KV=0" in reason


def test_op_registered_under_sm70_name():
    assert qk.OP_NAME == QK_ROPE_KV_OP_NAME
    assert QK_ROPE_KV_OP_NAME.startswith("sm70_")
    assert hasattr(torch.ops.vllm, QK_ROPE_KV_OP_NAME)


def test_fused_route_runs_k15_op_and_attention():
    """M5's fused route calls K15's registered op (the one the platform hook
    splits out and the mutation hooks patch), then K15's
    attention_after_fused_write; M5 keeps no copy of either."""
    import inspect

    from vllm_sm70.models.flint_sm70 import model as m

    src = inspect.getsource(m.FlintSm70Attention._forward_fused)
    assert "_QK_ROPE_KV_OP(" in src
    assert m._QK_ROPE_KV_OP is getattr(torch.ops.vllm, QK_ROPE_KV_OP_NAME)
    assert "attention_after_fused_write(self.attn, q, k, v)" in src
    assert QK_ROPE_KV_SPLITTING_OP == f"vllm::{QK_ROPE_KV_OP_NAME}"
    assert m.fused_route_rejection is qk.fused_route_rejection
    assert m.attention_after_fused_write is qk.attention_after_fused_write
    assert _mutation_module().TARGET == qk.__name__
    assert importlib.util.find_spec("vllm_sm70.models.flint_sm70.qk_norm_rope_kv") is None


# --------------------------------------------------------------------------
# FlintSm70ModelConfig and registry wiring
# --------------------------------------------------------------------------
class _FakeCuda(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"
    capability: DeviceCapability | None = None

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return cls.capability


@pytest.fixture
def device(monkeypatch):
    def _set(major, minor, ext=True):
        cls = type("Dev", (_FakeCuda,), {})
        cls.capability = DeviceCapability(major, minor)
        monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
        sm70 = (major, minor) == (7, 0)
        monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: sm70)
        monkeypatch.setattr(_ext, "is_loaded", lambda: ext)
        sm70_platform.is_sm70_ready.cache_clear()

    yield _set
    sm70_platform.is_sm70_ready.cache_clear()


def _vllm_config(
    dtype=torch.float16, backend=None, kv="auto", head_dim=128, spec=None, block_size=16
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype, hf_text_config=SimpleNamespace(head_dim=head_dim)
        ),
        attention_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(cache_dtype=kv, block_size=block_size),
        speculative_config=spec,
    )


@pytest.fixture
def fv100_ops(monkeypatch):
    """Control whether the Flash-V100 ops count as built (CPU tests)."""
    from vllm_sm70 import attention

    from vllm_sm70.attention.flash_v100 import BACKEND_PATH
    from vllm_sm70.ops import fv100

    state = {"ok": True, "sinks": True, "custom": BACKEND_PATH}
    monkeypatch.setattr(attention, "_ops_available", lambda: state["ok"])
    monkeypatch.setattr(fv100, "supports_sinks", lambda: state["sinks"])
    monkeypatch.setattr(attention, "custom_backend_path", lambda: state["custom"])
    monkeypatch.delenv(attention.KILL_SWITCH, raising=False)
    return state


def test_config_refuses_off_sm70(device):
    from vllm_sm70.models.flint_sm70.config import FlintSm70ModelConfig

    device(8, 0)
    with pytest.raises(ValueError, match="compute capability 7.0"):
        FlintSm70ModelConfig.verify_and_update_config(_vllm_config())


def test_config_refuses_non_fp16(device):
    from vllm_sm70.models.flint_sm70.config import FlintSm70ModelConfig

    device(7, 0)
    with pytest.raises(ValueError, match="--dtype half"):
        FlintSm70ModelConfig.verify_and_update_config(_vllm_config(torch.bfloat16))


def test_config_selects_flash_v100_with_sinks(device, fv100_ops):
    """fork config.py:375-399: Flash-V100 (CUSTOM) when its sink route admits."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70.models.flint_sm70.config import FlintSm70ModelConfig

    device(7, 0)
    vc = _vllm_config()
    FlintSm70ModelConfig.verify_and_update_config(vc)
    assert vc.attention_config.backend == AttentionBackendEnum.CUSTOM


@pytest.mark.parametrize(
    "case, kw, why",
    [
        ("kill switch", {}, "VLLM_SM70_FLASH_V100=0"),
        ("ops missing", {}, "ops are missing"),
        ("head 256", {"head_dim": 256}, "head_dim=256"),
        ("spec decode", {"spec": object()}, "speculative"),
        ("sink ops missing", {}, "sink ops are missing"),
        ("custom overridden", {}, "CUSTOM is overridden"),
        ("block size 8", {"block_size": 8}, "multiple of 16"),
    ],
)
def test_config_falls_back_to_triton(device, fv100_ops, monkeypatch, case, kw, why):
    """fork config.py:276-305, 400-405: TRITON_ATTN when the sink route rejects."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70.models.flint_sm70 import config as fcfg

    device(7, 0)
    if case == "kill switch":
        monkeypatch.setenv("VLLM_SM70_FLASH_V100", "0")
    if case == "ops missing":
        fv100_ops["ok"] = False
    if case == "sink ops missing":
        fv100_ops["sinks"] = False
    if case == "custom overridden":
        fv100_ops["custom"] = "somebody.else.Backend"
    vc = _vllm_config(**kw)
    assert why in fcfg.flash_v100_sink_route_rejection(vc)
    fcfg.FlintSm70ModelConfig.verify_and_update_config(vc)
    assert vc.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


def test_config_refuses_fp8_kv_and_keeps_explicit_backend(device, fv100_ops):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70.models.flint_sm70.config import FlintSm70ModelConfig

    device(7, 0)
    # FP8 KV: Flash-V100's sink route rejects, then TRITON refuses fp8 on SM70.
    with pytest.raises(ValueError, match="FP16 KV cache"):
        FlintSm70ModelConfig.verify_and_update_config(_vllm_config(kv="fp8_e4m3"))
    explicit = _vllm_config(backend=AttentionBackendEnum.FLEX_ATTENTION)
    FlintSm70ModelConfig.verify_and_update_config(explicit)
    assert explicit.attention_config.backend == AttentionBackendEnum.FLEX_ATTENTION
    explicit = _vllm_config(backend=AttentionBackendEnum.TRITON_ATTN)
    FlintSm70ModelConfig.verify_and_update_config(explicit)
    assert explicit.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


@pytest.fixture
def registry(monkeypatch):
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    saved_model = ModelRegistry.models.pop(flint_sm70.ARCH, None)
    saved_cfg = MODELS_CONFIG_MAP.pop(flint_sm70.ARCH, None)
    monkeypatch.setattr(flint_sm70, "_registered", False)
    yield ModelRegistry, MODELS_CONFIG_MAP
    ModelRegistry.models.pop(flint_sm70.ARCH, None)
    MODELS_CONFIG_MAP.pop(flint_sm70.ARCH, None)
    if saved_model is not None:
        ModelRegistry.models[flint_sm70.ARCH] = saved_model
    if saved_cfg is not None:
        MODELS_CONFIG_MAP[flint_sm70.ARCH] = saved_cfg


def test_register_at_cc70(device, registry):
    from vllm_sm70.models.flint_sm70.config import FlintSm70ModelConfig

    reg, cfg_map = registry
    device(7, 0)
    assert flint_sm70.register() is True
    entry = reg.models[flint_sm70.ARCH]
    assert (entry.module_name, entry.class_name) == (
        "vllm_sm70.models.flint_sm70",
        "FlintSm70ForCausalLM",
    )
    assert cfg_map[flint_sm70.ARCH] is FlintSm70ModelConfig
    assert flint_sm70.register() is True  # idempotent
    # Upstream's own Flint entry is untouched.
    assert reg.models["FlintForCausalLM"].module_name == "vllm.models.flint"


def test_not_registered_at_cc80(device, registry):
    reg, cfg_map = registry
    device(8, 0)
    assert flint_sm70.register() is False
    assert flint_sm70.ARCH not in reg.models and flint_sm70.ARCH not in cfg_map
    assert flint_sm70._registered is False


def test_register_raises_without_ext_then_retries(device, registry):
    reg, _ = registry
    device(7, 0, ext=False)
    with pytest.raises(RuntimeError, match="_sm70_C is not loaded"):
        flint_sm70.register()
    assert flint_sm70._registered is False and flint_sm70.ARCH not in reg.models
    device(7, 0, ext=True)
    assert flint_sm70.register() is True


def test_register_raises_if_registry_ignores_it(device, registry, monkeypatch):
    reg, _ = registry
    device(7, 0)
    monkeypatch.setattr(reg, "register_model", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="upstream registry semantics changed"):
        flint_sm70.register()
    assert flint_sm70._registered is False


def test_model_class_surface():
    """Static checks that need no engine: the class shape vLLM relies on."""
    from vllm.model_executor.models.interfaces import supports_pp

    from vllm_sm70.models.flint_sm70 import model as m

    cls = m.FlintSm70ForCausalLM
    assert supports_pp(cls)
    assert cls.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    assert cls.hf_to_vllm_mapper.orig_to_new_stacked[".mlp.up_proj"] == (
        ".mlp.gate_up_proj",
        1,
    )
    from vllm.models.flint.configs import FlintConfig

    assert m.FlintConfig is FlintConfig  # reuses upstream's config class


# --------------------------------------------------------------------------
# GPU plan (c4130). "1 GPU": one V100. "2 GPUs": TP2, both V100s, service
# stopped, 00:00-06:00 window only.
# --------------------------------------------------------------------------
def _gpu_ok(n=1) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= n
        and torch.cuda.get_device_capability() == (7, 0)
    )


one_gpu = pytest.mark.skipif(not _gpu_ok(1), reason="needs a V100")
two_gpus = pytest.mark.skipif(not _gpu_ok(2), reason="needs 2 V100 (TP2)")
CKPT = os.environ.get("FLINT_SM70_CHECKPOINT")
needs_ckpt = pytest.mark.skipif(CKPT is None, reason="set FLINT_SM70_CHECKPOINT")
ARCH_OVERRIDE = {"architectures": [flint_sm70.ARCH]}
PROMPTS = [
    "def quicksort(xs):",
    "The capital of France is",
    "Explain the CAP theorem in one sentence.",
    "1, 1, 2, 3, 5, 8,",
    "SELECT name FROM users WHERE",
    "Translate to German: good morning",
    "Write a haiku about V100 GPUs.",
    "The derivative of x**3 is",
]


def _llm(**kw):
    from vllm import LLM

    args = dict(
        model=CKPT,
        tensor_parallel_size=2,
        dtype="half",
        hf_overrides=ARCH_OVERRIDE,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
    )
    args.update(kw)
    return LLM(**args)


def _greedy(llm, n=64, prompts=PROMPTS):
    from vllm import SamplingParams

    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=n))
    return [list(o.outputs[0].token_ids) for o in outs]


def _gpus_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _shutdown(llm, settle_mib=2048, timeout_s=120):
    """Release an LLM's GPUs before the next one starts (upstream's
    VllmRunner.__exit__ recipe): stop the engine core, drop the object,
    tear down the process groups, then wait for the memory to come back."""
    import gc

    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    try:
        llm.llm_engine.engine_core.shutdown()
    finally:
        del llm
        gc.collect()
        torch._dynamo.reset()
        cleanup_dist_env_and_memory()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = _gpus_used_mib()
        if not used or max(used) < settle_mib:
            return
        time.sleep(2)
    raise RuntimeError(f"GPU memory not released after shutdown: {used} MiB")


# T5 INFO: a first mismatch counts as a near-tie when the golden token is our
# #2 within this many nats.
T4_NEAR_TIE = 0.1


def _greedy_once(n=64, prompts=PROMPTS, **llm_kw):
    """Build an LLM, run greedy decoding, always shut it down."""
    llm = _llm(**llm_kw)
    try:
        return _greedy(llm, n, prompts)
    finally:
        _shutdown(llm)


# T2 contract for qk_norm_rope_kv (Astro, Nov 5, from Bea's t2_diag):
# bitwise at decode sizes; at prefill sizes q/k/cached K within atol+rtol of
# the torch-GPU oracle and no less accurate than torch-GPU vs a CPU FP32
# oracle; cached V and unwritten slots exact.
T2_BITWISE_MAX_N = 17
T2_ATOL = 2e-3
T2_RTOL = 1e-3
# Mean-error slack vs torch-GPU: Bea's diag had the kernel at 287 vs 281 (q)
# and 120 vs 117 (k) mismatches against the CPU oracle, i.e. equal within
# noise; a strict <= on the mean would fail on that noise alone.
T2_MEAN_SLACK = 1.25


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in FP16 ulps (monotone int16 ordering of the bits)."""

    def order(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((order(a) - order(b)).abs().max()) if a.numel() else 0


def test_max_ulp_helper():
    one = torch.tensor([1.0], dtype=torch.half)
    nxt = torch.nextafter(one.float(), torch.tensor([2.0])).half()  # rounds to 1
    up = (one.view(torch.int16) + 1).view(torch.half)
    assert _max_ulp(one, one) == 0 and _max_ulp(one, nxt) == 0
    assert _max_ulp(one, up) == 1
    z, nz = torch.tensor([0.0], dtype=torch.half), torch.tensor([-0.0], dtype=torch.half)
    assert _max_ulp(z, nz) == 0
    tiny = torch.tensor([0.0], dtype=torch.half).view(torch.int16) + 1
    assert _max_ulp(tiny.view(torch.half), (-tiny.view(torch.half))) == 2


@pytest.mark.gpu
@one_gpu
@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("n", [1, 2, 17, 8192])
def test_t2_triton_kernel_bitwise_vs_reference(rope, n):
    """T2 (1 GPU): the Triton kernel vs the torch oracle.

    Decode sizes (n <= T2_BITWISE_MAX_N): bitwise equal on q, k and the
    whole UP-layout K/V cache.

    Prefill sizes: the oracle's CUDA ``mean(-1)`` reduces in a different
    order than the kernel's per-row sum, and with RoPE that shows up as a
    few large-ulp differences on near-zero outputs (Bea's diag, Nov 5:
    |delta| <= 1.95e-3 on 28 of 4.2M q values at n=8192). Contract (Astro):
    - q, k and the cached K within T2_ATOL + T2_RTOL of the torch-GPU oracle;
    - the kernel is no less accurate than torch-GPU itself: against a CPU
      FP32 oracle (same math and rounding points), the kernel's max abs
      error <= torch-GPU's, and its mean abs error <= T2_MEAN_SLACK x
      torch-GPU's;
    - the cached V half (a plain copy) and every unwritten slot (padded
      tokens, unused slots) bitwise equal, and unwritten slots still zero."""
    g = torch.Generator().manual_seed(n)
    qkv = torch.randn(n, (HQ + 2 * HKV) * D, generator=g).half().cuda()
    qw = (torch.rand(D, generator=g) + 0.5).half().cuda()
    kw = (torch.rand(D, generator=g) + 0.5).half().cuda()
    pos = torch.randint(0, 64, (n,), generator=g).cuda()
    cs = _cos_sin().cuda() if rope else None
    blocks = n // PAGE + 2
    slots = torch.randperm(blocks * PAGE, generator=g)[:n].cuda()
    slots[::5] = -1

    def run(fn, device):
        to = (lambda t: None if t is None else t.to(device))
        kv = torch.zeros(blocks, HKV, PAGE, 2 * D, dtype=torch.half, device=device)
        q = torch.empty(n, HQ * D, dtype=torch.half, device=device)
        k = torch.empty(n, HKV * D, dtype=torch.half, device=device)
        fn(to(qkv), q, k, to(qw), to(kw), to(pos), to(cs), 1e-6,
           *_cache_views(kv), to(slots))
        return q, k, kv

    tq, tk, tkv = run(qk.qk_norm_rope_kv, "cuda")
    rq, rk, rkv = run(_qk_oracle, "cuda")
    if n <= T2_BITWISE_MAX_N:
        for a, b in zip((tq, tk, tkv), (rq, rk, rkv)):
            assert torch.equal(a, b)
        return
    cq, ck, ckv = run(_qk_oracle, "cpu")
    pairs = {
        "q": (tq, rq, cq),
        "k": (tk, rk, ck),
        "cached K": (tkv[..., :D], rkv[..., :D], ckv[..., :D]),
    }
    for name, (tri, gpu, cpu) in pairs.items():
        torch.testing.assert_close(
            tri, gpu, atol=T2_ATOL, rtol=T2_RTOL, msg=f"{name} vs torch-GPU oracle"
        )
        e_tri = (tri.float().cpu() - cpu.float()).abs()
        e_gpu = (gpu.float().cpu() - cpu.float()).abs()
        print(
            f"T2 n={n} rope={rope} {name}: vs CPU oracle max {e_tri.max():.3e} "
            f"(torch-GPU {e_gpu.max():.3e}), mean {e_tri.mean():.3e} "
            f"(torch-GPU {e_gpu.mean():.3e}), ulp vs GPU {_max_ulp(tri, gpu)}"
        )
        assert e_tri.max() <= e_gpu.max(), f"{name}: kernel max error > torch-GPU's"
        assert e_tri.mean() <= T2_MEAN_SLACK * e_gpu.mean() + 1e-12, (
            f"{name}: kernel mean error > {T2_MEAN_SLACK} x torch-GPU's"
        )
    assert torch.equal(tkv[..., D:], rkv[..., D:]), "cached V must be exact"
    written = torch.zeros(blocks * PAGE, dtype=torch.bool, device="cuda")
    written[slots[slots >= 0]] = True
    untouched = ~written.view(blocks, PAGE)[:, None, :].expand(blocks, HKV, PAGE)
    assert torch.equal(tkv[untouched], rkv[untouched]), "unwritten slots changed"
    assert not tkv[untouched].any(), "kernel wrote to a padded/unused slot"

@pytest.mark.gpu
@one_gpu
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("rope", [True, False])
def test_t2b_permuted_cache_matches_upstream_writer(layout, rope):
    """T2b (1 GPU): on a permuted UP layer view, our fused op's cache write is
    byte-identical to UP's own TRITON_ATTN writer,
    TritonAttentionImpl.do_kv_cache_update
    (UP vllm/v1/attention/backends/triton_attn.py:798-843, which calls
    triton_reshape_and_cache_flash,
    vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:377), fed the
    same K and V, with -1 padded slots. UP's writer is a Triton kernel with no
    CPU path, hence gpu."""
    from vllm.v1.attention.backend import AttentionType

    n = 37
    g = torch.Generator().manual_seed(7)
    qkv = torch.randn(n, (HQ + 2 * HKV) * D, generator=g).half().cuda()
    qw = (torch.rand(D, generator=g) + 0.5).half().cuda()
    kw = (torch.rand(D, generator=g) + 0.5).half().cuda()
    pos = torch.randint(0, 64, (n,), generator=g).cuda()
    cs = _cos_sin().cuda() if rope else None
    blocks = 5
    slots = torch.randperm(blocks * PAGE, generator=g)[:n].cuda()
    slots[::4] = -1
    ours = _layer_cache(layout, blocks, device="cuda")
    q = torch.empty(n, HQ * D, dtype=torch.half, device="cuda")
    k = torch.empty(n, HKV * D, dtype=torch.half, device="cuda")
    qk.qk_norm_rope_kv(
        qkv, q, k, qw, kw, pos, cs, 1e-6, *_cache_views(ours), slots
    )
    impl = _triton_impl()
    impl.attn_type = AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    one = torch.ones((), device="cuda")
    upstream = _layer_cache(layout, blocks, device="cuda")
    impl.do_kv_cache_update(
        SimpleNamespace(_k_scale=one, _v_scale=one),
        k.view(n, HKV, D),
        qkv[:, (HQ + HKV) * D :].view(n, HKV, D),
        upstream,
        slots,
    )
    torch.cuda.synchronize()
    assert torch.equal(ours, upstream)


def _tiny_flint_dir(tmp_path):
    """A tiny random-weight Flint config (load_format=dummy fills weights)."""
    from vllm.models.flint.configs import FlintConfig

    cfg = FlintConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[0],
        layer_types=["sliding_attention", "full_attention"] * 2,
        no_rope_layer_indices=[1, 3],
        sliding_window=64,
        max_position_embeddings=1024,
        torch_dtype="float16",
    )
    cfg.architectures = [flint_sm70.ARCH]
    cfg.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.gpu
@one_gpu
def test_t1_tiny_random_forward(tmp_path):
    """T1 (1 GPU): a tiny unquantized FlintSm70 with dummy weights builds,
    compiles and decodes with TP1, fused and unfused, giving the same tokens."""
    from vllm import LLM, SamplingParams

    path = _tiny_flint_dir(tmp_path)
    ids = [[1, 2, 3, 4, 5], [7, 8, 9]]
    toks = []
    for fused in ("1", "0"):
        os.environ["VLLM_SM70_FLINT_QK_ROPE_KV"] = fused
        llm = LLM(
            model=path,
            load_format="dummy",
            dtype="half",
            skip_tokenizer_init=True,
            max_model_len=512,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            seed=0,
        )
        outs = llm.generate(
            [{"prompt_token_ids": p} for p in ids],
            SamplingParams(temperature=0, max_tokens=16),
        )
        toks.append([list(o.outputs[0].token_ids) for o in outs])
        _shutdown(llm)
    os.environ.pop("VLLM_SM70_FLINT_QK_ROPE_KV", None)
    assert all(len(t) == 16 for t in toks[0])
    assert toks[0] == toks[1]


# The route line comes from init_logger("vllm.sm70.flint"), under vLLM's
# configured "vllm" logger (it used to be "vllm_sm70.models.flint_sm70.model",
# which vLLM's default config silently drops). T3 still passes an explicit
# logging config that routes both "vllm" and "vllm_sm70" to stdout, so the
# check does not depend on vLLM's default handler or log level.
_T3_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(levelname)s %(name)s: %(message)s"}},
    "handlers": {
        "out": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        name: {"handlers": ["out"], "level": "INFO", "propagate": False}
        for name in ("vllm", "vllm_sm70")
    },
}


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_t3_checkpoint_loads(capfd, monkeypatch, tmp_path):
    """T3 (2 GPUs): the NVFP4 checkpoint loads at TP2 fp16 and every layer
    takes the fused route (Flash-V100 by default, TRITON_ATTN with
    VLLM_SM70_FLASH_V100=0).

    The route line is logged in the TP worker processes, and the vllm logger
    does not propagate, so caplog never sees it. capfd reads the inherited
    stdout/stderr file descriptors, which the workers write to, and
    _T3_LOGGING_CONFIG makes the "vllm_sm70" logger write there."""
    from vllm.config.logging import LoggingConfig

    cfg = tmp_path / "logging.json"
    cfg.write_text(json.dumps(_T3_LOGGING_CONFIG))
    # Passed explicitly (EngineArgs.logging_config); the engine core and the
    # TP workers apply it. The env vars cover anything built from defaults.
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", str(cfg))
    toks = _greedy_once(
        enforce_eager=True,
        logging_config=LoggingConfig(
            configure_logging=True, pylogging_config_file=str(cfg)
        ),
    )
    out, err = capfd.readouterr()
    logs = out + err
    sys.stdout.write(logs)  # keep the worker log in the pytest report
    assert "Flint SM70: fused q/k norm, RoPE and KV-cache write." in logs
    assert "Flint SM70: unfused q/k norm/RoPE because" not in logs
    assert all(len(t) == 64 for t in toks)


# T4a-T4d/T7: teacher-forced closeness, anchored on the fork (Astro, Nov 6-7).
# All scoring goes through sm70/scripts/m5_teacher_forced.py (the same code
# calibrates the fork), one sequence at a time (M=1 in eager and in replay).
# Metrics per continuation token: "decode" (forced decode: every token is a
# real decode step, raw logprob of the forced token) and "prefill"
# (prompt_logprobs over the whole sequence).
#   FLINT_SM70_T4_SEQS_DIR      setK.json from `m5_teacher_forced.py gen` (>= 3 sets)
#   FLINT_SM70_T4_ENVELOPE      `m5_teacher_forced.py envelope` over the fork's
#                               scores: per pair/metric, anchor = max over sets
#   FLINT_SM70_FORK_SCORES_DIR  the fork's scores,
#                               setK/{eager,eager2,graphs,unfused}.json (the
#                               envelope's input; T4d reads graphs.json)
#   FLINT_SM70_OURS_NOISE       optional: ours_eager_vs_eager envelope JSON.
#                               T4c writes it (two separate eager instances);
#                               T4a/T4d/T7 print it on every gate line.
# Every score comes from its own engine instance (one LLM build per score).
# Gate (Quinn; factors PROVISIONAL, constants at the top of
# m5_teacher_forced.py): per set, ours mean <= 1.5x anchor mean AND ours max
# <= 1.5x anchor max; each line also shows count > 0.1 nats, our own
# eager-vs-eager instance noise and the fork's. docs/m5-calibration-nov7.md
# has the commands.
T4_SEQS_DIR = os.environ.get("FLINT_SM70_T4_SEQS_DIR")
T4_ENVELOPE = os.environ.get("FLINT_SM70_T4_ENVELOPE")
FORK_SCORES_DIR = os.environ.get("FLINT_SM70_FORK_SCORES_DIR")
OURS_NOISE = os.environ.get("FLINT_SM70_OURS_NOISE")
T4_ENVELOPE_FACTOR = 1.5
# Mutation check (sm70/scripts/m5_mutation_check.sh): which planted bug is on.
M5_MUTATION = os.environ.get("M5_MUTATION")
MUTATIONS_DIR = pathlib.Path(__file__).resolve().parent / "m5_mutations"
needs_seqs = pytest.mark.skipif(
    T4_SEQS_DIR is None, reason="set FLINT_SM70_T4_SEQS_DIR (m5_teacher_forced.py gen)"
)
needs_envelope = pytest.mark.skipif(
    T4_SEQS_DIR is None or T4_ENVELOPE is None,
    reason="set FLINT_SM70_T4_SEQS_DIR and FLINT_SM70_T4_ENVELOPE (fork calibration)",
)
needs_fork_scores = pytest.mark.skipif(
    FORK_SCORES_DIR is None, reason="set FLINT_SM70_FORK_SCORES_DIR (fork scores)"
)


@functools.lru_cache(maxsize=None)
def _tf_module():
    """sm70/scripts/m5_teacher_forced.py, imported under its own name so the
    forcing processor's FQCN resolves in the engine core and TP workers."""
    scripts = pathlib.Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    mod = importlib.import_module("m5_teacher_forced")
    mod.ensure_importable()
    return mod


@functools.lru_cache(maxsize=None)
def _mutation_module():
    spec = importlib.util.spec_from_file_location(
        "m5_mutation", MUTATIONS_DIR / "m5_mutation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mutated(**llm_kw):
    """LLM kwargs for the MUTATED side of T4d/T7. Hook mutations are already
    in every process (sitecustomize); the config mutation rope_theta_1p01 is
    applied here, to this side only."""
    if M5_MUTATION == "rope_theta_1p01":
        override = _mutation_module().rope_theta_override(CKPT)
        print(f"M5_MUTATION rope_theta_1p01: hf_overrides {json.dumps(override)}")
        llm_kw["hf_overrides"] = {**ARCH_OVERRIDE, **override}
    return llm_kw


def _seq_sets():
    return _tf_module().load_seq_sets(T4_SEQS_DIR)


def _envelope(seq_sets):
    """The fork envelope, checked to be calibrated on exactly these sequences."""
    tf = _tf_module()
    with open(T4_ENVELOPE) as f:
        env = json.load(f)
    want = {k: v["seqs_sha256"] for k, v in env["sets"].items()}
    got = {str(k): tf.seqs_sha256(s) for k, s in seq_sets.items()}
    assert got == want, (
        "FLINT_SM70_T4_SEQS_DIR is not the sequence set the envelope was "
        f"calibrated on: ours {got} vs envelope {want}"
    )
    assert len(want) >= tf.MIN_SETS, f"envelope has {len(want)} sets (< {tf.MIN_SETS})"
    return env


def _score_sets(seq_sets, decode=True, prefill=True, greedy_prompts=(), **llm_kw):
    """Build one LLM with the scoring setup (forcing processor, raw logprobs,
    max_num_seqs=1, no prefix caching), score every set, optionally run free
    greedy (64 tokens, one prompt at a time), always shut down."""
    tf = _tf_module()
    llm = _llm(**{**tf.scoring_llm_kwargs(1), **llm_kw})
    try:
        scores = {
            k: tf.score_seqs(llm, seqs, decode, prefill) for k, seqs in seq_sets.items()
        }
        greedy = [_greedy(llm, 64, [p])[0] for p in greedy_prompts]
    finally:
        _shutdown(llm)
    return scores, greedy


def _ours_noise():
    """Our eager-vs-eager (two engine instances) envelope from T4c, if any."""
    if OURS_NOISE is None or not os.path.exists(OURS_NOISE):
        print(f"ours_eager_vs_eager: not available ({OURS_NOISE!r}; run t4c first)")
        return None
    with open(OURS_NOISE) as f:
        return json.load(f)


def _noise_stats(env, metric, k=None):
    """eager_vs_eager stats for one set (k) or the anchor (k=None)."""
    e = ((env or {}).get("pairs", {}).get(_tf_module().NOISE_PAIR) or {}).get(metric)
    if e is None:
        return None
    return e["anchor"] if k is None else e["per_set"].get(str(k))


def _gate_sets(tag, label, anchor, metric, a_by_set, b_by_set, ours_noise=None,
               fork_env=None):
    """Gate every set's |dlogprob| stats (a vs b) against one anchor; print a
    verdict line per set (count > 0.1, our instance noise and the fork's
    included); return the breaches."""
    tf = _tf_module()
    assert tf.FACTOR == T4_ENVELOPE_FACTOR
    failures = []
    for k in sorted(a_by_set):
        ours = tf.stats(tf.metric(a_by_set[k], metric), tf.metric(b_by_set[k], metric))
        line, bad = tf.gate_line(tag, label, metric, k, ours, anchor,
                                 _noise_stats(ours_noise, metric, k),
                                 _noise_stats(fork_env, metric))
        print(line)
        failures += [f"{label} {metric} set{k}: {b}" for b in bad]
    return failures


def _print_top1(tag, name, scores):
    for k, s in sorted(scores.items()):
        bad, n = _tf_module().top1_disagreements(s)
        print(f"{tag} info {name} set{k}: raw top-1 != forced token at {bad}/{n} steps")


# ---- CPU checks of the harness -------------------------------------------
def _batch(added=(), removed=(), moved=(), size=2):
    from vllm.v1.sample.logits_processor import BatchUpdate

    return BatchUpdate(batch_size=size, removed=list(removed), added=list(added),
                       moved=list(moved))


def test_force_processor_masks_and_tracks_state():
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import MoveDirectionality

    tf = _tf_module()
    proc = tf.ForceRefLogitsProcessor(None, torch.device("cpu"), False)
    assert proc.is_argmax_invariant() is False  # applied for greedy requests too
    ref = [5, 2, 7]
    out0, out1 = [], []
    proc.update_state(_batch(added=[
        (0, SamplingParams(temperature=0, max_tokens=3, extra_args={tf.FORCE_KEY: ref}),
         [1, 2], out0),
        (1, SamplingParams(temperature=0, max_tokens=3), [3], out1),
    ]))
    for step in range(3):
        logits = torch.randn(2, 10)
        logits[0, 9] = 100.0  # the model's own top-1 is never the reference
        free = logits[1].clone()
        res = proc.apply(logits)
        assert int(res[0].argmax()) == ref[step]
        assert torch.isinf(res[0]).sum() == 9 and res[0, ref[step]] == 0.0
        assert torch.equal(res[1], free)  # unforced row untouched
        out0.append(ref[step])  # the runner appends the sampled id
        out1.append(int(free.argmax()))
        proc.update_state(None)
    logits = torch.randn(2, 10)
    assert torch.equal(proc.apply(logits.clone()), logits)  # past the reference
    # A swap moves the state with the request; a removal drops it.
    out2 = [4]
    proc.update_state(_batch(
        added=[(1, SamplingParams(extra_args={tf.FORCE_KEY: [8, 3]}), [1], out2)],
        removed=[0], moved=[(0, 1, MoveDirectionality.SWAP)],
    ))
    assert set(proc.forced) == {0}
    assert int(proc.apply(torch.randn(2, 10))[0].argmax()) == 3  # step 1 of [8, 3]
    proc.update_state(_batch(removed=[0], size=0))
    assert proc.forced == {}


def test_force_processor_validate_params():
    from vllm import SamplingParams

    tf = _tf_module()
    tf.ForceRefLogitsProcessor.validate_params(SamplingParams())
    tf.ForceRefLogitsProcessor.validate_params(
        SamplingParams(max_tokens=2, extra_args={tf.FORCE_KEY: [1, 2]})
    )
    for bad in ([], [1, -2], "12", [1.5]):
        with pytest.raises(ValueError):
            tf.ForceRefLogitsProcessor.validate_params(
                SamplingParams(max_tokens=1, extra_args={tf.FORCE_KEY: bad})
            )
    with pytest.raises(ValueError, match="max_tokens"):
        tf.ForceRefLogitsProcessor.validate_params(
            SamplingParams(max_tokens=3, extra_args={tf.FORCE_KEY: [1, 2]})
        )
    assert tf.FORCE_FQCN.split(":") == ["m5_teacher_forced", "ForceRefLogitsProcessor"]
    kw = tf.scoring_llm_kwargs(1)
    assert kw["logprobs_mode"] == "raw_logprobs" and kw["max_num_seqs"] == 1
    assert str(tf.SCRIPTS_DIR) in os.environ["PYTHONPATH"].split(os.pathsep)


class _FakeLLM:
    """Stands in for LLM.generate: forced decode returns the reference with
    per-step logprob dicts; prefill returns prompt_logprobs."""

    def __init__(self, flip_at=None):
        self.flip_at = flip_at
        self.calls = 0

    def generate(self, prompts, params):
        self.calls += 1
        params = params if isinstance(params, list) else [params] * len(prompts)
        outs = []
        for p, sp in zip(prompts, params):
            ids = p["prompt_token_ids"]
            if sp.prompt_logprobs:
                plp = [None] + [{t: SimpleNamespace(logprob=-0.5 * i, rank=1)}
                                for i, t in enumerate(ids[1:], 1)]
                outs.append(SimpleNamespace(prompt_logprobs=plp, outputs=[]))
                continue
            ref = list(sp.extra_args["m5_force"])
            toks = list(ref)
            if self.flip_at is not None:
                toks[self.flip_at] += 1
            steps = [{t: SimpleNamespace(logprob=-1.0 - j, rank=2),
                      99: SimpleNamespace(logprob=-0.1, rank=1)} for j, t in enumerate(toks)]
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=toks, logprobs=steps)]))
        return outs


def test_score_seqs_decode_and_prefill_records():
    tf = _tf_module()
    seqs = [{"prompt": "a", "start": 2, "ids": [1, 2, 3, 4, 5]},
            {"prompt": "b", "start": 1, "ids": [7, 8, 9]}]
    llm = _FakeLLM()
    res = tf.score_seqs(llm, seqs)
    assert llm.calls == 3  # one generate per sequence (M=1) + one prefill call
    assert res["seqs_sha256"] == tf.seqs_sha256(seqs)
    assert res["seqs"][0]["decode"] == [-1.0, -2.0, -3.0]
    assert res["seqs"][0]["decode_top1"] == [99, 99, 99]
    assert res["seqs"][1]["prefill"] == [-0.5, -1.0]
    assert tf.top1_disagreements(res) == (5, 5)
    with pytest.raises(tf.ForcingError, match="first difference at 1"):
        tf.score_seqs(_FakeLLM(flip_at=1), seqs, prefill=False)


def _write_scores(root, k, name, decode, prefill, sha="s"):
    d = root / f"set{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({
        "seqs_sha256": sha, "seqs": [{"decode": decode, "prefill": prefill}]}))


def test_envelope_aggregates_max_over_sets(tmp_path):
    tf = _tf_module()
    base = [-1.0] * 10
    # per set k: graphs off by dg[k] at every token (decode), one 0.3 spike in
    # prefill; unfused off by du[k] (decode) and 2*du[k] (prefill).
    dg, du = {0: 0.01, 1: 0.04, 2: 0.02}, {0: 0.002, 1: 0.001, 2: 0.005}
    for k in range(3):
        _write_scores(tmp_path, k, "eager", base, base, sha=f"h{k}")
        _write_scores(tmp_path, k, "eager2", [x - 0.2 * (k == 2) for x in base], base,
                      sha=f"h{k}")
        _write_scores(tmp_path, k, "graphs", [x - dg[k] for x in base],
                      [x - (0.3 if i == k else 0.0) for i, x in enumerate(base)], sha=f"h{k}")
        _write_scores(tmp_path, k, "unfused", [x - du[k] for x in base],
                      [x - 2 * du[k] for x in base], sha=f"h{k}")
    env = tf.build_envelope(tmp_path)
    assert env["sets"] == {str(k): {"seqs_sha256": f"h{k}"} for k in range(3)}
    eg = env["pairs"]["eager_vs_graphs"]
    assert eg["decode"]["anchor"]["mean"] == pytest.approx(0.04)  # max over sets
    assert eg["decode"]["anchor"]["max"] == pytest.approx(0.04)
    assert eg["decode"]["per_set"]["0"]["mean"] == pytest.approx(0.01)
    assert eg["prefill"]["anchor"]["max"] == pytest.approx(0.3)
    assert eg["prefill"]["anchor"]["mean"] == pytest.approx(0.03)
    assert eg["prefill"]["per_set"]["1"]["over_0_1"] == 1
    fu = env["pairs"]["fused_vs_unfused"]
    assert fu["decode"]["anchor"]["mean"] == pytest.approx(0.005)
    assert fu["prefill"]["anchor"]["max"] == pytest.approx(0.01)
    assert fu["decode"]["per_set"]["2"]["n"] == 10
    ee = env["pairs"]["eager_vs_eager"]
    assert ee["decode"]["anchor"]["max"] == pytest.approx(0.2)
    assert ee["decode"]["per_set"]["2"]["over_0_1"] == 10
    assert ee["prefill"]["anchor"] == {"mean": 0.0, "max": 0.0, "over_0_1": 0}
    only = tf.build_envelope(tmp_path, pairs=["eager_vs_eager"])
    assert list(only["pairs"]) == ["eager_vs_eager"] and only["gate"]["provisional"]
    with pytest.raises(ValueError, match=">= 4 prompt sets"):
        tf.build_envelope(tmp_path, min_sets=4)
    _write_scores(tmp_path, 1, "unfused", base, base, sha="other")
    with pytest.raises(ValueError, match="different sequences"):
        tf.build_envelope(tmp_path)
    (tmp_path / "set1" / "graphs.json").unlink()
    with pytest.raises(FileNotFoundError):
        tf.build_envelope(tmp_path)
    with pytest.raises(ValueError, match="unknown pairs"):
        tf.envelope_from_scores({}, ["nope"])


def test_stats_refuses_misaligned_and_flags_nan():
    tf = _tf_module()
    with pytest.raises(ValueError, match="sequence 0"):
        tf.stats([[1.0, 2.0]], [[1.0]])
    s = tf.stats([[-1.0, float("-inf")]], [[float("nan"), float("-inf")]])
    assert s["max"] == float("inf") and s["over_0_1"] == 1


def test_envelope_gate():
    tf = _tf_module()
    anchor = {"mean": 0.02, "max": 0.4, "over_0_1": 3}
    assert tf.gate_failures({"mean": 0.029, "max": 0.59}, anchor) == []
    mean_bad = tf.gate_failures({"mean": 0.05, "max": 0.1}, anchor)
    assert len(mean_bad) == 1 and mean_bad[0].startswith("mean")
    max_bad = tf.gate_failures({"mean": 0.01, "max": 0.61}, anchor)
    assert len(max_bad) == 1 and max_bad[0].startswith("max")
    assert len(tf.gate_failures({"mean": 1.0, "max": float("inf")}, anchor)) == 2
    line, bad = tf.gate_line("T4a", "eager_vs_graphs", "decode", 0,
                             {"n": 512, "mean": 0.01, "max": 0.2, "over_0_1": 4}, anchor)
    assert not bad and line.startswith("T4a eager_vs_graphs decode set0: ")
    assert ">0.1=4/512" in line and line.endswith("PASS")


def test_gate_sets_prints_every_set_and_collects(capsys):
    tf = _tf_module()
    anchor = {"mean": 0.02, "max": 0.4, "over_0_1": 0}
    a = {k: {"seqs": [{"decode": [-1.0] * 100}]} for k in range(3)}
    b = {0: {"seqs": [{"decode": [-1.01] * 100}]},
         1: {"seqs": [{"decode": [-1.05] * 100}]},  # mean breach
         2: {"seqs": [{"decode": [-1.0] * 99 + [-1.7]}]}}  # max breach
    fails = _gate_sets("T7", "fused_vs_unfused", anchor, "decode", a, b)
    out = capsys.readouterr().out
    assert [ln.split(":")[0] for ln in out.splitlines()] == [
        f"T7 fused_vs_unfused decode set{k}" for k in range(3)]
    assert len(fails) == 2 and "set1: mean" in fails[0] and "set2: max" in fails[1]
    assert tf.FACTOR == T4_ENVELOPE_FACTOR


def test_prompt_sets_are_disjoint():
    tf = _tf_module()
    sets = tf.PROMPT_SETS
    assert len(sets) >= 3 and all(len(p) == 8 for p in sets.values())
    assert sets[0] == PROMPTS
    flat = [p for ps in sets.values() for p in ps]
    assert len(flat) == len(set(flat))


# ---- mutation hook (m5_mutations/) ----------------------------------------
def _qk_call(n, slots=None, blocks=3):
    qkv, qw, kw, pos = _inputs(n=n, seed=11)
    kv = torch.zeros(blocks, HKV, PAGE, 2 * D, dtype=torch.half)
    q = torch.empty(n, HQ * D, dtype=torch.half)
    k = torch.empty(n, HKV * D, dtype=torch.half)
    if slots is None:
        slots = torch.arange(n) * 3
    return qkv, q, k, qw, kw, pos, kv, slots


def _run_registered_op(monkeypatch, n, slots=None):
    """Call K15's torch.ops.vllm.<QK_ROPE_KV_OP_NAME> on CPU with a fake
    attention context whose layer takes the direct cache-write branch.

    direct_register_custom_op registers the module's Python op body under the
    platform's dispatch key: CPU on a host without the SM70 platform, CUDA
    where it is active (c4130 with CUDA_VISIBLE_DEVICES="", Felix). For the
    latter, register the same body for CPU for the duration of the call, so
    the call still goes through the registered op's dispatch."""
    import vllm.model_executor.layers.attention.attention as attention
    from vllm.utils.torch_utils import _encode_layer_name

    qkv, q, k, qw, kw, pos, kv, slots = _qk_call(n, slots)
    layer = SimpleNamespace(impl=_triton_impl(), kv_cache_dtype="auto")
    monkeypatch.setattr(attention, "get_attention_context",
                        lambda name: (None, layer, kv, slots))
    op = getattr(torch.ops.vllm, QK_ROPE_KV_OP_NAME)
    cpu_lib = None
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"vllm::{QK_ROPE_KV_OP_NAME}", "CPU"
    ):
        cpu_lib = torch.library.Library("vllm", "IMPL")
        cpu_lib.impl(QK_ROPE_KV_OP_NAME, qk.qk_norm_rope_kv_cache_update, "CPU")
    try:
        op(qkv, q, k, qw, kw, pos, _cos_sin(), 1e-6, _encode_layer_name("m5.layer"))
    finally:
        if cpu_lib is not None:
            cpu_lib._destroy()
    return q, k, kv, (qkv, qw, kw, pos, slots)


def _reference(n, qkv, qw, kw, pos, slots, blocks=3):
    kv = torch.zeros(blocks, HKV, PAGE, 2 * D, dtype=torch.half)
    q = torch.empty(n, HQ * D, dtype=torch.half)
    k = torch.empty(n, HKV * D, dtype=torch.half)
    _qk_oracle(qkv, q, k, qw, kw, pos, _cos_sin(), 1e-6, *_cache_views(kv), slots)
    return q, k, kv

def test_mutation_fired_marker_only_when_a_call_changes(capfd):
    """Quinn: a mutation must be shown to have FIRED. The qk wrapper marks
    fired only on a call it changed (<= 8 tokens), once per process."""
    import inspect

    mm = _mutation_module()
    mm._FIRED.clear()
    calls = []

    def fake_op(qkv, q_out, k_out, q_weight, k_weight, positions, cos_sin_cache,
                epsilon, key_cache=None, value_cache=None, slot_mapping=None):
        calls.append(positions.clone())

    wrapped = mm.make_wrapper(fake_op, "rope_pos_decode")
    cs = torch.zeros(64, 8)
    big = torch.zeros(9, 4)
    wrapped(big, None, None, None, None, torch.arange(9), cs, 1e-6)
    assert "fired" not in capfd.readouterr().err  # prefill-sized call: untouched
    small = torch.zeros(2, 4)
    wrapped(small, None, None, None, None, torch.tensor([3, 4]), cs, 1e-6)
    wrapped(small, None, None, None, None, torch.tensor([5, 6]), cs, 1e-6)
    err = capfd.readouterr().err
    assert err.count("M5_MUTATION rope_pos_decode fired in pid") == 1
    assert calls[1].tolist() == [4, 5] and calls[0].tolist() == list(range(9))


def test_mutation_moe_decode_scale_scales_k12_output():
    """moe_decode_scale wraps flint_moe_decode_out (looked up by M4 as a
    module attribute at call time) and scales its output by 1.01."""
    import types

    mm = _mutation_module()
    mm._FIRED.discard("moe_decode_scale")
    fake = types.ModuleType(mm.MOE_TARGET)

    def flint_moe_decode_out(out, *rest):
        out.fill_(2.0)
        return out

    fake.flint_moe_decode_out = flint_moe_decode_out
    mm.install(fake, "moe_decode_scale")
    out = fake.flint_moe_decode_out(torch.zeros(2, 4), None)
    assert torch.allclose(out, torch.full((2, 4), 2.0 * mm.MOE_DECODE_SCALE))
    mm.install(fake, "moe_decode_scale")  # idempotent: no double scaling
    out = fake.flint_moe_decode_out(torch.zeros(1, 4))
    assert torch.allclose(out, torch.full((1, 4), 2.0 * mm.MOE_DECODE_SCALE))
    assert mm._target_for("moe_decode_scale") == (mm.MOE_TARGET, "flint_moe_decode_out")
    assert mm._target_for("kv_slot_decode")[0] == mm.TARGET


@pytest.mark.parametrize("n", [4, 9])
def test_mutation_rope_pos_decode_reaches_registered_op(monkeypatch, n):
    """The wrapper on the module global changes what the REGISTERED op
    computes: positions + 1 at <= 8 tokens, untouched above."""
    mm = _mutation_module()
    monkeypatch.setattr(qk, "qk_norm_rope_kv",
                        mm.make_wrapper(_qk_oracle, "rope_pos_decode"))
    q, k, kv, (qkv, qw, kw, pos, slots) = _run_registered_op(monkeypatch, n)
    want_pos = pos + 1 if n <= mm.DECODE_MAX_TOKENS else pos
    rq, rk, rkv = _reference(n, qkv, qw, kw, want_pos, slots)
    assert torch.equal(q, rq) and torch.equal(k, rk) and torch.equal(kv, rkv)
    cq, _, _ = _reference(n, qkv, qw, kw, pos, slots)
    assert torch.equal(q, cq) == (n > mm.DECODE_MAX_TOKENS)


@pytest.mark.parametrize("n", [4, 9])
def test_mutation_kv_slot_decode_reaches_registered_op(monkeypatch, n):
    mm = _mutation_module()
    monkeypatch.setattr(qk, "qk_norm_rope_kv",
                        mm.make_wrapper(_qk_oracle, "kv_slot_decode"))
    slots = torch.tensor([0, 17, -1, 47 if n > 8 else 30] + [20 + i for i in range(n - 4)])
    q, k, kv, (qkv, qw, kw, pos, slots) = _run_registered_op(monkeypatch, n, slots)
    total = 3 * PAGE
    want = slots.where(slots < 0, (slots + 1) % total) if n <= mm.DECODE_MAX_TOKENS else slots
    rq, rk, rkv = _reference(n, qkv, qw, kw, pos, want)
    assert torch.equal(q, rq) and torch.equal(k, rk) and torch.equal(kv, rkv)
    if n <= mm.DECODE_MAX_TOKENS:
        assert kv[0, :, 0].abs().sum() == 0 and kv[0, :, 1].abs().sum() > 0  # 0 -> 1
        assert kv[0, :, 0, :D].abs().sum() == 0


def test_mutation_install_wraps_the_triton_function(monkeypatch):
    mm = _mutation_module()
    original = qk.qk_norm_rope_kv
    monkeypatch.setattr(qk, "qk_norm_rope_kv", original)
    mm.install(qk, "kv_slot_decode")
    assert qk.qk_norm_rope_kv._m5_original is original
    mm.install(qk, "rope_pos_decode")  # re-install wraps the original, not the wrapper
    assert qk.qk_norm_rope_kv._m5_original is original
    assert qk.qk_norm_rope_kv._m5_mutation == "rope_pos_decode"
    # The op body resolves the name in the module globals at call time.
    assert qk.qk_norm_rope_kv_cache_update.__globals__ is vars(qk)
    with pytest.raises(ValueError, match="known"):
        mm.install_import_hook("no_such_bug")


@pytest.mark.parametrize("mutation", ["kv_slot_decode", ""])
def test_mutation_sitecustomize_in_fresh_process(tmp_path, mutation):
    """A fresh interpreter (as a TP worker is) with the mutations dir first on
    PYTHONPATH: the shadowed sitecustomize still runs, and importing the qk
    module installs the wrapper iff M5_MUTATION is set."""
    shadowed = tmp_path / "shadowed"
    shadowed.mkdir()
    (shadowed / "sitecustomize.py").write_text(
        "import os\nos.environ['M5_SHADOWED_RAN'] = '1'\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(MUTATIONS_DIR), str(shadowed)] + [p for p in sys.path if p])
    env["M5_MUTATION"] = mutation
    code = (
        "import os, json\n"
        "from vllm_sm70.flint import qk_norm_rope_kv as qk\n"
        "print(json.dumps([os.environ.get('M5_SHADOWED_RAN'),"
        " getattr(qk.qk_norm_rope_kv, '_m5_mutation', None),"
        " getattr(getattr(qk.qk_norm_rope_kv, '_m5_original', None), '__name__', None)]))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                       text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    ran, installed, orig = json.loads(r.stdout.strip().splitlines()[-1])
    assert ran == "1"
    if mutation:
        assert (installed, orig) == (mutation, "qk_norm_rope_kv")
        assert f"M5_MUTATION {mutation} installed in pid" in r.stderr
    else:
        assert installed is None


def test_rope_theta_override(tmp_path):
    mm = _mutation_module()

    def ov(cfg):
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return mm.rope_theta_override(str(tmp_path))

    flat = ov({"rope_theta": 1e6, "rope_parameters": {"rope_type": "default",
                                                      "rope_theta": 1e6, "x": 1}})
    assert flat == {"rope_theta": pytest.approx(1.01e6),
                    "rope_parameters": {"rope_type": "default",
                                        "rope_theta": pytest.approx(1.01e6), "x": 1}}
    nested = ov({"rope_parameters": {"sliding_attention": {"rope_theta": 1e4},
                                     "full_attention": {"rope_theta": 1e6}}})
    assert nested == {"rope_parameters": {
        "sliding_attention": {"rope_theta": pytest.approx(1.01e4)},
        "full_attention": {"rope_theta": pytest.approx(1.01e6)}}}
    legacy = ov({"rope_theta": 5e5})
    assert legacy["rope_parameters"]["rope_theta"] == pytest.approx(5.05e5)
    assert legacy["rope_theta"] == pytest.approx(5.05e5)
    default = ov({})
    assert default["rope_theta"] == pytest.approx(10100.0)


def test_skipif_markers_are_boolean():
    for mark in (needs_seqs, needs_envelope, needs_fork_scores, needs_ckpt):
        assert isinstance(mark.args[0], bool)


# ---- GPU gates ---------------------------------------------------------------
@pytest.mark.gpu
@two_gpus
@needs_ckpt
@needs_envelope
def test_t4a_teacher_forced_eager_vs_graphs():
    """T4a (2 GPUs): our eager vs our compiled + CUDA graphs, teacher-forced
    on every prompt set, decode AND prefill metrics.

    Bea's t4_bisect (Nov 6): Inductor codegen accounts for all of the
    eager/compiled prefill difference (graphs_only = 0.0), and free decode
    under graph replay diverges too, so the decode metric (real decode steps,
    replayed graphs) is gated as well as prefill. Gate per set and metric:
    mean <= 1.5x and max <= 1.5x the fork's eager_vs_graphs anchor."""
    seq_sets = _seq_sets()
    env = _envelope(seq_sets)
    eager, _ = _score_sets(seq_sets, enforce_eager=True)
    graphs, _ = _score_sets(seq_sets)
    _print_top1("T4a", "eager", eager)
    _print_top1("T4a", "graphs", graphs)
    tf, noise = _tf_module(), _ours_noise()
    pair = tf.GATE_ANCHOR_PAIR["T4a"]
    failures = []
    for m in tf.METRICS:
        anchor = env["pairs"][pair][m]["anchor"]
        failures += _gate_sets("T4a", "eager_vs_graphs", anchor, m, eager, graphs,
                               noise, env)
    assert not failures, f"T4a outside {T4_ENVELOPE_FACTOR}x the fork anchor: {failures}"


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@pytest.mark.parametrize("max_num_seqs", [1, 2])
def test_t4b_graph_replay_exact(max_num_seqs):
    """T4b (2 GPUs): CUDA-graph replay is exact when the batch shape matches.

    Mode NONE + cudagraph FULL takes Inductor out; at max_num_seqs 1 and 2
    every decode step has M <= 2 (no capture padding beyond the batch), which
    covers the K11/K12 M<=2 paths. Greedy tokens must equal eager's for all
    256 tokens: any flip is a capture/replay bug, not numerics."""
    eager = _greedy_once(256, enforce_eager=True, max_num_seqs=max_num_seqs)
    graphs = _greedy_once(
        256, max_num_seqs=max_num_seqs,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL"},
    )
    for i, (e, g) in enumerate(zip(eager, graphs)):
        pos = next((j for j, (a, b) in enumerate(zip(e, g)) if a != b), None)
        print(f"T4b max_num_seqs={max_num_seqs} prompt {i}: "
              + ("identical" if pos is None else f"differs at {pos}"))
    assert eager == graphs


def _t4c_report(s1, s2, g1, g2, fork_env=None):
    """T4c verdict lines and failures: forced-decode |dlogprob| stats per set
    (vs the fork's eager_vs_eager anchor), bitwise check, free-run first
    divergence per prompt."""
    tf = _tf_module()
    failures = []
    for k in sorted(s1):
        for m in tf.METRICS:
            if not all(m in r for s in (s1[k], s2[k]) for r in s["seqs"]):
                continue
            st = tf.stats(tf.metric(s1[k], m), tf.metric(s2[k], m))
            exact = tf.metric(s1[k], m) == tf.metric(s2[k], m)
            print(f"T4c ours_eager_vs_eager {m} set{k}: {tf._brief('ours', st)} | "
                  f"{tf._brief('fork_eager_vs_eager', _noise_stats(fork_env, m))} -> "
                  + ("bitwise equal" if exact else "DIFFERS"))
            if m == "decode" and not exact:
                failures.append(f"set{k} decode: {st['over_0_1']}/{st['n']} > {tf.OVER}, "
                                f"max {st['max']:.4g}")
    for i, (x, y) in enumerate(zip(g1, g2)):
        pos = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), None)
        if pos is None and len(x) != len(y):
            pos = min(len(x), len(y))
        print(f"T4c free-run prompt {i}: "
              + ("identical" if pos is None else f"first divergence at token {pos}"))
    return failures


def test_t4c_report_flags_decode_differences(capsys):
    a = {0: {"seqs": [{"decode": [-1.0, -2.0], "prefill": [-1.0, -2.0]}]}}
    b = {0: {"seqs": [{"decode": [-1.0, -2.5], "prefill": [-1.0, -2.0]}]}}
    assert _t4c_report(a, a, [[1, 2]], [[1, 2]]) == []
    fails = _t4c_report(a, b, [[1, 2, 3]], [[1, 4, 3]])
    out = capsys.readouterr().out
    assert len(fails) == 1 and "1/2 > 0.1" in fails[0]
    assert "T4c ours_eager_vs_eager decode set0: ours mean=0.25 max=0.5" in out
    assert "T4c free-run prompt 0: first divergence at token 1" in out


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@needs_seqs
def test_t4c_eager_repeatable_across_engine_instances():
    """T4c (2 GPUs): eager vs eager across TWO SEPARATE ENGINE INSTANCES
    (build, score, shut down, build again: new engine core, new workers).

    Reports the forced-decode (and prefill) |dlogprob| stats per set and the
    free-run greedy first divergence per prompt (64 tokens, one prompt at a
    time). Gate: forced-decode logprobs bitwise identical. Bea (Nov 6): two
    instances gave different free-run tokens on 7/8 prompts with real
    logprob shifts (suspected startup autotuning, e.g. TurboMind kernel
    selection by timing), so this is EXPECTED to fail on today's build: a
    finding, not a test bug. It also writes our eager_vs_eager envelope to
    FLINT_SM70_OURS_NOISE (before the gate), which T4a/T4d/T7 print as
    ours_eager_vs_eager on every line."""
    seq_sets = _seq_sets()
    s1, g1 = _score_sets(seq_sets, greedy_prompts=PROMPTS, enforce_eager=True)
    s2, g2 = _score_sets(seq_sets, greedy_prompts=PROMPTS, enforce_eager=True)
    tf = _tf_module()
    noise = tf.envelope_from_scores({"eager": s1, "eager2": s2}, [tf.NOISE_PAIR], min_sets=1)
    if OURS_NOISE is not None:
        with open(OURS_NOISE, "w") as f:
            json.dump(noise, f, indent=1)
        print(f"T4c wrote ours_eager_vs_eager -> {OURS_NOISE}")
    fork_env = None
    if T4_ENVELOPE is not None:
        with open(T4_ENVELOPE) as f:
            fork_env = json.load(f)
    failures = _t4c_report(s1, s2, g1, g2, fork_env)
    assert not failures, (
        "engine-instance nondeterminism: two separate eager engine instances "
        f"give different forced-decode logprobs: {failures}"
    )


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@needs_envelope
@needs_fork_scores
def test_t4d_graphs_vs_fork_graphs():
    """T4d (2 GPUs): cross-stack anchor. OUR graphs vs the FORK's graphs on
    the same sequences, decode metric.

    T4a compares our two modes with each other, so a bug present in both
    (e.g. a wrong RoPE position or KV slot in the shared fused op) cancels
    out there. T4d compares our replayed decode against the fork's (its
    graphs.json from FLINT_SM70_FORK_SCORES_DIR, the envelope's input). Gate
    per set: mean <= 1.5x and max <= 1.5x the fork's own eager_vs_graphs
    decode anchor, i.e. ours may differ from the fork no more than the
    fork's two modes differ from each other (plus 50%). Mutated side under
    m5_mutation_check.sh: ours."""
    seq_sets = _seq_sets()
    env = _envelope(seq_sets)
    tf = _tf_module()
    fork = tf.load_scores(FORK_SCORES_DIR, "graphs")
    assert sorted(fork) == sorted(seq_sets), f"fork sets {sorted(fork)}"
    for k, seqs in seq_sets.items():
        assert fork[k]["seqs_sha256"] == tf.seqs_sha256(seqs), f"fork set{k}: other sequences"
        assert fork[k].get("mode") in (None, "graphs"), f"fork set{k}: {fork[k].get('mode')}"
    ours, _ = _score_sets(seq_sets, prefill=False, **_mutated())
    _print_top1("T4d", "ours graphs", ours)
    anchor = env["pairs"][tf.GATE_ANCHOR_PAIR["T4d"]]["decode"]["anchor"]
    failures = _gate_sets("T4d", "ours_graphs_vs_fork_graphs", anchor, "decode", ours, fork,
                          _ours_noise(), env)
    assert not failures, (
        f"T4d outside {T4_ENVELOPE_FACTOR}x the fork's eager_vs_graphs anchor: {failures}")


# Earliest fork self-divergence in Bea's golden is token 40 (prefix-cache path).
GOLDEN_EXACT_TOKENS = 32


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@pytest.mark.skipif(
    os.environ.get("FLINT_SM70_GOLDEN") is None, reason="set FLINT_SM70_GOLDEN"
)
def test_t5_golden_tokens_vs_fork():
    """T5 (2 GPUs): greedy tokens match the fork cf9492a golden.

    Golden file (Bea, captured from live fork Flint, raw /v1/completions,
    temperature 0, max_tokens 64): a JSON list of {"prompt",
    "prompt_token_ids", "token_ids", "stable_across_2_runs", ...}.

    Each prompt is generated alone (as the golden was captured).

    Gate: prompts whose golden was stable across the fork's own two runs
    must match the first GOLDEN_EXACT_TOKENS tokens exactly. Prompts the
    fork itself could not reproduce are INFO only: we print the common
    prefix and, at the first mismatch, whether the golden token is our #2
    and our top-2 gap (one-sided near-tie check; the golden has no
    logprobs). There must be at least one stable prompt."""
    from vllm import SamplingParams

    with open(os.environ["FLINT_SM70_GOLDEN"]) as f:
        golden = json.load(f)
    assert any(g.get("stable_across_2_runs") for g in golden), "no stable prompt"
    # One request at a time, as the golden was captured (raw /v1/completions
    # on live fork Flint, one prompt per request): decode then runs at M=1
    # like the fork did (K11 GEMV, K12 decode). Batching the prompts changes
    # M per step and with it the kernel route and the greedy result (Bea,
    # Nov 6: the 8K prompt matches 16/16 alone, diverges at 10 batched).
    llm = _llm()
    try:
        sp = SamplingParams(temperature=0, max_tokens=64, logprobs=2)
        outs = [
            llm.generate([{"prompt_token_ids": g["prompt_token_ids"]}], sp)[0]
            for g in golden
        ]
    finally:
        _shutdown(llm)
    failures = []
    for g, o in zip(golden, outs):
        c = o.outputs[0]
        ours, ref = list(c.token_ids), g["token_ids"]
        common = next(
            (i for i, (a, b) in enumerate(zip(ours, ref)) if a != b),
            min(len(ours), len(ref)),
        )
        stable = bool(g.get("stable_across_2_runs"))
        note = ""
        if common < min(len(ours), len(ref)):
            lp = {t: x.logprob for t, x in c.logprobs[common].items()}
            top2 = sorted(lp, key=lp.get, reverse=True)[:2]
            gold = ref[common]
            if gold in top2 and len(top2) == 2:
                gap = abs(lp[top2[0]] - lp[top2[1]])
                kind = "near-tie" if gap <= T4_NEAR_TIE else "confident"
                note = f"; at {common}: golden is our #2, gap {gap:.4f} nats ({kind})"
            else:
                note = f"; at {common}: golden {gold} not in our top-2 {top2}"
        print(
            f"T5 {'GATE' if stable else 'INFO'} prompt {g['prompt'][:40]!r}: "
            f"common prefix {common}/64 (fork stable={stable}){note}"
        )
        if stable and common < GOLDEN_EXACT_TOKENS:
            failures.append((g["prompt"][:40], common))
    assert not failures, f"diverged before {GOLDEN_EXACT_TOKENS} tokens: {failures}"


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_t6_fp16_headroom_8k():
    """T6 (2 GPUs): an 8K-token prompt decodes with finite logprobs (the
    layer 44-48 residual stays in FP16 range thanks to the rescale)."""
    from vllm import SamplingParams

    llm = _llm(enforce_eager=True)
    prompt = " ".join(["the quick brown fox"] * 2000)
    out = llm.generate(
        [prompt], SamplingParams(temperature=0, max_tokens=8, logprobs=5)
    )[0].outputs[0]
    for step in out.logprobs:
        assert all(torch.isfinite(torch.tensor(lp.logprob)) for lp in step.values())


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@needs_envelope
def test_t7_unfused_path_matches(monkeypatch):
    """T7 (2 GPUs): VLLM_SM70_FLINT_QK_ROPE_KV=0 (unfused + upstream
    Attention) vs the fused route, both eager, teacher-forced on every set,
    decode and prefill. Gate per set and metric: mean <= 1.5x and max <=
    1.5x the fork's own fused_vs_unfused anchor. Mutated side under
    m5_mutation_check.sh: fused (hook mutations patch the fused op only)."""
    from vllm_sm70.config import clear_cache

    seq_sets = _seq_sets()
    env = _envelope(seq_sets)
    fused, _ = _score_sets(seq_sets, enforce_eager=True, **_mutated())
    monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", "0")
    clear_cache()
    try:
        unfused, _ = _score_sets(seq_sets, enforce_eager=True)
    finally:
        monkeypatch.delenv("VLLM_SM70_FLINT_QK_ROPE_KV")
        clear_cache()
    tf, noise = _tf_module(), _ours_noise()
    failures = []
    for m in tf.METRICS:
        anchor = env["pairs"][tf.GATE_ANCHOR_PAIR["T7"]][m]["anchor"]
        failures += _gate_sets("T7", "fused_vs_unfused", anchor, m, fused, unfused,
                               noise, env)
    assert not failures, f"T7 outside {T4_ENVELOPE_FACTOR}x the fork anchor: {failures}"
