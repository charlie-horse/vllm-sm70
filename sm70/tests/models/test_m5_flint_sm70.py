# SPDX-License-Identifier: Apache-2.0
"""Tests for M5 flint-sm70-model (vllm_sm70/models/flint_sm70/).

CPU: fp16_rescale against frozen copies of the fork's math; the
  qk_norm_rope_kv torch oracle against the unfused route and against
  upstream's Triton KV-cache layout (fused-route admission included);
  FlintSm70ModelConfig refusals; registry wiring at cc 7.0 vs 8.0; and the
  VLLM_SM70_FLINT_QK_ROPE_KV kill switch.
GPU (``-m gpu``): GPU test plan T1-T8. Tests marked "2 GPUs" need both V100s
  and the ``vllm`` service stopped; they run in the 00:00-06:00 window.
  Paths come from env:
    FLINT_SM70_CHECKPOINT  the ModelOpt NVFP4 Flint checkpoint dir
    FLINT_SM70_GOLDEN      JSON of fork cf9492a greedy records (T5); c4130:
                           ~/vllm-sm70-work/bea/golden/flint_cf9492a_greedy64.json
"""

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch

import vllm.platforms
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.models import flint_sm70
from vllm_sm70.models.flint_sm70 import fp16_rescale as fr
from vllm_sm70.models.flint_sm70 import qk_norm_rope_kv as qk

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
# qk_norm_rope_kv: torch oracle and the upstream Triton cache layout
# --------------------------------------------------------------------------
HQ, HKV, D, PAGE = 4, 2, 128, 16


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
    qk.qk_norm_rope_kv_reference(qkv, q, k, qw, kw, pos, cs, 1e-6)

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


def test_cache_write_uses_upstream_triton_layout():
    """kv_cache is [blocks, Hkv, page, 2*D]
    (UP vllm/v1/attention/backends/triton_attn.py:598-599); K goes to
    [..., :D] and V to [..., D:], at slot = block*page + offset. Padded
    slots (-1) are skipped."""
    qkv, qw, kw, pos = _inputs(n=4)
    kv_cache = torch.zeros(3, HKV, PAGE, 2 * D, dtype=torch.half)
    slots = torch.tensor([0, 17, -1, 47])
    layer = SimpleNamespace(impl=_triton_impl(), kv_cache_dtype="auto")
    views = qk._direct_cache_views(layer, kv_cache, slots, HKV)
    assert views is not None
    for v in views:
        assert v.shape == (3, PAGE, HKV, D) and v.data_ptr() >= kv_cache.data_ptr()
    q = torch.empty(4, HQ * D, dtype=torch.half)
    k = torch.empty(4, HKV * D, dtype=torch.half)
    qk.qk_norm_rope_kv_reference(
        qkv, q, k, qw, kw, pos, _cos_sin(), 1e-6, *views, slots
    )
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
        views = qk._direct_cache_views(layer, kv, slots, HKV)
        assert views is not None  # admitted: the kernel honours the strides
        q = torch.empty(6, HQ * D, dtype=torch.half)
        k = torch.empty(6, HKV * D, dtype=torch.half)
        qk.qk_norm_rope_kv_reference(
            qkv, q, k, qw, kw, pos, _cos_sin(), 1e-6, *views, slots
        )
    assert torch.equal(cache, ref)
    assert ref.any()


def test_direct_cache_views_refuses_other_layouts():
    kv = torch.zeros(2, HKV, PAGE, 2 * D, dtype=torch.half)
    slots = torch.tensor([0])
    impl = _triton_impl()
    ok = SimpleNamespace(impl=impl, kv_cache_dtype="auto")
    assert qk._direct_cache_views(ok, kv, slots, HKV) is not None
    fp8 = SimpleNamespace(impl=impl, kv_cache_dtype="fp8")
    assert qk._direct_cache_views(fp8, kv, slots, HKV) is None
    other = SimpleNamespace(impl=_OtherImpl(), kv_cache_dtype="auto")
    assert qk._direct_cache_views(other, kv, slots, HKV) is None  # CUSTOM backend
    assert qk._direct_cache_views(ok, kv.float(), slots, HKV) is None
    impl._is_per_token_head_quant = True
    assert qk._direct_cache_views(ok, kv, slots, HKV) is None


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
    assert qk.OP_NAME == "sm70_flint_qk_norm_rope_kv_cache_update"
    assert hasattr(torch.ops.vllm, qk.OP_NAME)


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


def _vllm_config(dtype=torch.float16, backend=None, kv="auto"):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype),
        attention_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(cache_dtype=kv),
    )


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


def test_config_selects_triton_and_refuses_fp8_kv(device):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70.models.flint_sm70.config import FlintSm70ModelConfig

    device(7, 0)
    vc = _vllm_config()
    FlintSm70ModelConfig.verify_and_update_config(vc)
    assert vc.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    with pytest.raises(ValueError, match="FP16 KV cache"):
        FlintSm70ModelConfig.verify_and_update_config(_vllm_config(kv="fp8_e4m3"))
    explicit = _vllm_config(backend=AttentionBackendEnum.FLEX_ATTENTION)
    FlintSm70ModelConfig.verify_and_update_config(explicit)
    assert explicit.attention_config.backend == AttentionBackendEnum.FLEX_ATTENTION


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
           *qk.triton_cache_views(kv, D), to(slots))
        return q, k, kv

    tq, tk, tkv = run(qk.qk_norm_rope_kv, "cuda")
    rq, rk, rkv = run(qk.qk_norm_rope_kv_reference, "cuda")
    if n <= T2_BITWISE_MAX_N:
        for a, b in zip((tq, tk, tkv), (rq, rk, rkv)):
            assert torch.equal(a, b)
        return
    cq, ck, ckv = run(qk.qk_norm_rope_kv_reference, "cpu")
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
        qkv, q, k, qw, kw, pos, cs, 1e-6, *qk.triton_cache_views(ours, D), slots
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
    takes the fused route on TRITON_ATTN.

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


# T4a/T7: teacher-forced closeness, anchored on the fork (Astro, Nov 6).
# FLINT_SM70_T4_SEQS: fixed sequences from `m5_teacher_forced.py gen`.
# FLINT_SM70_T4_ENVELOPE: the fork's own stats from `m5_teacher_forced.py
# envelope` (pairs "eager_vs_graphs" and "fused_vs_unfused").
T4_SEQS = os.environ.get("FLINT_SM70_T4_SEQS")
T4_ENVELOPE = os.environ.get("FLINT_SM70_T4_ENVELOPE")
T4_ENVELOPE_FACTOR = 1.5
needs_envelope = pytest.mark.skipif(
    T4_SEQS is None or T4_ENVELOPE is None,
    reason="set FLINT_SM70_T4_SEQS and FLINT_SM70_T4_ENVELOPE (fork calibration)",
)


def _tf_module():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "m5_teacher_forced.py"
    spec = importlib.util.spec_from_file_location("m5_teacher_forced", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _teacher_forced_scores(seqs, **llm_kw):
    """Per-sequence logprobs of each continuation token (prompt_logprobs)."""
    from vllm import SamplingParams

    llm = _llm(**llm_kw)
    try:
        outs = llm.generate(
            [{"prompt_token_ids": s["ids"]} for s in seqs],
            SamplingParams(max_tokens=1, prompt_logprobs=1),
        )
        return [
            [o.prompt_logprobs[i][s["ids"][i]].logprob
             for i in range(s["start"], len(s["ids"]))]
            for s, o in zip(seqs, outs)
        ]
    finally:
        _shutdown(llm)


def _assert_within_fork_envelope(pair, a, b):
    """mean and p99 |dlogprob| <= T4_ENVELOPE_FACTOR x the fork's own."""
    tf = _tf_module()
    ours = tf.stats(a, b)
    with open(T4_ENVELOPE) as f:
        fork = json.load(f)[pair]
    print(f"{pair}: ours {json.dumps(ours)} fork {json.dumps(fork)} "
          f"(limit {T4_ENVELOPE_FACTOR}x mean and p99)")
    for key in ("mean", "p99"):
        assert ours[key] <= T4_ENVELOPE_FACTOR * fork[key] + 1e-6, (
            f"{pair}: {key} {ours[key]:.4f} > {T4_ENVELOPE_FACTOR} x fork "
            f"{fork[key]:.4f}"
        )


def test_envelope_gate(monkeypatch, tmp_path):
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"eager_vs_graphs": {"mean": 0.02, "p99": 0.4}}))
    monkeypatch.setitem(globals(), "T4_ENVELOPE", str(env))
    base = [[-1.0] * 100]
    near = [[-1.0 - (0.5 if i == 0 else 0.02) for i in range(100)]]  # mean ~0.0248, p99 0.02
    _assert_within_fork_envelope("eager_vs_graphs", base, near)
    far = [[-1.0 - 0.05 for _ in range(100)]]  # mean 0.05 > 1.5 x 0.02
    with pytest.raises(AssertionError, match="mean"):
        _assert_within_fork_envelope("eager_vs_graphs", base, far)
    spiky = [[-1.0 - (0.9 if i < 3 else 0.0) for i in range(100)]]  # p99 0.9 > 0.6
    with pytest.raises(AssertionError, match="p99"):
        _assert_within_fork_envelope("eager_vs_graphs", base, spiky)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@needs_envelope
def test_t4a_teacher_forced_eager_vs_graphs():
    """T4a (2 GPUs): eager vs compiled + CUDA graphs on fixed sequences.

    Bea's t4_bisect (Nov 6): Inductor codegen accounts for all of the
    teacher-forced difference (graphs_only = 0.0), so this is the Inductor
    numeric envelope. Gate: within 1.5x of the fork's own eager-vs-graphs
    mean and p99 on the same sequences."""
    with open(T4_SEQS) as f:
        seqs = json.load(f)["seqs"]
    eager = _teacher_forced_scores(seqs, enforce_eager=True)
    graphs = _teacher_forced_scores(seqs)
    _assert_within_fork_envelope("eager_vs_graphs", eager, graphs)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_t4b_graph_replay_exact_at_one_seq():
    """T4b (2 GPUs): CUDA-graph replay is exact when M cannot change.

    max_num_seqs=1 means every decode step has M=1 in both runs (no capture
    padding), and mode NONE + cudagraph FULL takes Inductor out. Greedy
    tokens must then be identical to eager for all 256 tokens: any flip is a
    capture/replay bug, not numerics."""
    eager = _greedy_once(256, enforce_eager=True, max_num_seqs=1)
    graphs = _greedy_once(
        256, max_num_seqs=1,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL"},
    )
    for i, (e, g) in enumerate(zip(eager, graphs)):
        pos = next((j for j, (a, b) in enumerate(zip(e, g)) if a != b), None)
        print(f"T4b prompt {i}: " + ("identical" if pos is None else f"differs at {pos}"))
    assert eager == graphs

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
    Attention) vs the fused route, both eager, teacher-forced on fixed
    sequences. Gate: within 1.5x of the fork's own fused-vs-unfused mean and
    p99 (pair "fused_vs_unfused" in the envelope)."""
    with open(T4_SEQS) as f:
        seqs = json.load(f)["seqs"]
    fused = _teacher_forced_scores(seqs, enforce_eager=True)
    monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", "0")
    unfused = _teacher_forced_scores(seqs, enforce_eager=True)
    _assert_within_fork_envelope("fused_vs_unfused", fused, unfused)
