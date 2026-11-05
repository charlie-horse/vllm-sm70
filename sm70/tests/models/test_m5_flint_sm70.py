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
    FLINT_SM70_GOLDEN      JSON of fork cf9492a greedy token ids (T5)
"""

import json
import os
import re
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
    """Fork model.py:402-408: residual /8 at 46, attention output /8 above."""
    assert fr.stream_scales(layer, True) == expected
    assert fr.stream_scales(layer, False) == (None, None)


def test_lm_head_scale_folds_in_fp16_and_scales_logits_otherwise():
    """Fork model.py:647-668."""
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
    """The fused route's numerics (fork qk_norm_rope_kv.py:10-15) equal the
    unfused route: per-head RMSNorm rounded to fp16, then NeoX RoPE on the
    rounded values."""
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
    """kv_cache is [blocks, Hkv, page, 2*D] (UP triton_attn.py:598-599); K goes
    to [..., :D] and V to [..., D:], at slot = block*page + offset. Padded
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


one_gpu = pytest.mark.skipif("not _gpu_ok(1)", reason="needs a V100")
two_gpus = pytest.mark.skipif("not _gpu_ok(2)", reason="needs 2 V100 (TP2)")
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


@pytest.mark.gpu
@one_gpu
@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("n", [1, 2, 17, 8192])
def test_t2_triton_kernel_bitwise_vs_reference(rope, n):
    """T2 (1 GPU): the Triton kernel equals the torch oracle bitwise on q, k
    and the UP-layout K/V cache, padded slots included."""
    g = torch.Generator().manual_seed(n)
    qkv = torch.randn(n, (HQ + 2 * HKV) * D, generator=g).half().cuda()
    qw = (torch.rand(D, generator=g) + 0.5).half().cuda()
    kw = (torch.rand(D, generator=g) + 0.5).half().cuda()
    pos = torch.randint(0, 64, (n,), generator=g).cuda()
    cs = _cos_sin().cuda() if rope else None
    blocks = n // PAGE + 2
    slots = torch.randperm(blocks * PAGE, generator=g)[:n].cuda()
    slots[::5] = -1
    res = []
    for fn in (qk.qk_norm_rope_kv, qk.qk_norm_rope_kv_reference):
        kv = torch.zeros(blocks, HKV, PAGE, 2 * D, dtype=torch.half, device="cuda")
        q = torch.empty(n, HQ * D, dtype=torch.half, device="cuda")
        k = torch.empty(n, HKV * D, dtype=torch.half, device="cuda")
        fn(qkv, q, k, qw, kw, pos, cs, 1e-6, *qk.triton_cache_views(kv, D), slots)
        res.append((q, k, kv))
    for a, b in zip(*res):
        assert torch.equal(a, b)


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
        del llm
    os.environ.pop("VLLM_SM70_FLINT_QK_ROPE_KV", None)
    assert all(len(t) == 16 for t in toks[0])
    assert toks[0] == toks[1]


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_t3_checkpoint_loads(caplog):
    """T3 (2 GPUs): the NVFP4 checkpoint loads at TP2 fp16 and every layer
    takes the fused route on TRITON_ATTN."""
    with caplog.at_level("INFO"):
        llm = _llm(enforce_eager=True)
    assert "fused q/k norm, RoPE and KV-cache write" in caplog.text
    assert all(len(t) == 64 for t in _greedy(llm))


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_t4_compiled_graphs_match_eager():
    """T4 (2 GPUs): eager vs compiled + FULL_AND_PIECEWISE CUDA graphs give
    identical greedy tokens (no splitting op needed for the fused op)."""
    eager = _greedy(_llm(enforce_eager=True), 256)
    graphs = _greedy(_llm(), 256)
    assert eager == graphs


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@pytest.mark.skipif(
    os.environ.get("FLINT_SM70_GOLDEN") is None, reason="set FLINT_SM70_GOLDEN"
)
def test_t5_golden_tokens_vs_fork():
    """T5 (2 GPUs): the first 64 greedy tokens equal the fork cf9492a run
    (fork sampler and fused router off) for PROMPTS."""
    with open(os.environ["FLINT_SM70_GOLDEN"]) as f:
        golden = json.load(f)
    assert _greedy(_llm()) == [g[:64] for g in golden]


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
def test_t7_unfused_path_matches(monkeypatch):
    """T7 (2 GPUs): VLLM_SM70_FLINT_QK_ROPE_KV=0 (unfused + upstream Attention)
    gives the same greedy tokens as the fused route."""
    fused = _greedy(_llm(enforce_eager=True))
    monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", "0")
    assert _greedy(_llm(enforce_eager=True)) == fused
