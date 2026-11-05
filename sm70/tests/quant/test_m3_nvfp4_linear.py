# SPDX-License-Identifier: Apache-2.0
"""Tests for M3 nvfp4-dense-turbomind (vllm_sm70/quant/nvfp4_linear.py).

CPU tests mock ``vllm_sm70.ops.tm`` with pure-torch references:
  prepare packs 8 E2M1 codes per int32 along N and returns meta =
  (32*K, N), the layout the fork's GEMV admission checks (FORK cf9492a
  sm70_turbomind.py:621-622). gemm/gemv dequantize (E2M1 x fp16 combined
  scale) and matmul in fp32.
What the CPU tests can't show, and the ``gpu`` tests below cover: Kai's
  real TurboMind tile layout and k_ld/q_ld values, the GEMV's 2^shift
  prescale numerics, FP16 accumulation error, and CUDA-graph capture.
"""

import importlib
import sys
import types

import pytest

# Collected without vllm/torch installed (e.g. plain `pytest sm70/tests`): skip.
pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch
import vllm.model_executor.layers.quantization as me_quant
import vllm.platforms
from vllm.model_executor.layers.quantization import modelopt as up_modelopt
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform

E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
FLINT_QUANT_CFG = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["model.embed_tokens", "*.mlp.gate", "lm_head"],
    },
}


# --------------------------------------------------------------------------
# Mock of Kai's vllm_sm70.ops.tm
# --------------------------------------------------------------------------
def _pack_n8(codes: torch.Tensor) -> torch.Tensor:
    k, n = codes.shape
    c = codes.to(torch.int32).reshape(k, n // 8, 8)
    out = torch.zeros((k, n // 8), dtype=torch.int32)
    for i in range(8):
        out |= c[:, :, i] << (4 * i)
    return out


def _unpack_n8(w: torch.Tensor) -> torch.Tensor:
    parts = [(w >> (4 * i)) & 15 for i in range(8)]
    return torch.stack(parts, dim=-1).reshape(w.shape[0], -1).long()


def _dequant_tm(w, s, group_size):
    return E2M1[_unpack_n8(w)] * s.float().repeat_interleave(group_size, dim=0)


def make_mock_tm(gemv: bool = True):
    tm = types.ModuleType("vllm_sm70.ops.tm")
    tm.calls = []

    # Exact Kai contract (FORK torch_bindings.cpp:654-660, 948-961;
    # vllm/_sm70/linear.py:350, 1763, 1805): no global_scale argument anywhere.
    def nvfp4_sm70_prepare(qweight, scales, group_size=16, interleave_gated_silu=False):
        assert qweight.dtype == torch.uint8 and scales.dtype == torch.float16
        tm.calls.append(("prepare", qweight, scales, group_size, interleave_gated_silu))
        k, n = qweight.shape
        meta = torch.tensor([32 * k, n], dtype=torch.int64)
        return [_pack_n8(qweight), scales.clone(), meta]

    def nvfp4_gemm_sm70_out(out, x, w, s, group_size, k_ld, q_ld, gated_silu=False):
        assert gated_silu is False
        tm.calls.append(("gemm", x.shape[0]))
        out.copy_((x.float() @ _dequant_tm(w, s, group_size)).to(out.dtype))

    def nvfp4_tm_gemv_dispatch_sm70_out(
        out, x, w, s, group_size, k_ld, q_ld, scale_shift=14
    ):
        shift = scale_shift
        # The real op runs its GEMV for M <= 2 and TurboMind above (FORK
        # sm70_provider.py:42).
        tm.calls.append(
            ("gemv" if x.shape[0] <= 2 else "dispatch_gemm", x.shape[0], shift)
        )
        out.copy_((x.float() @ _dequant_tm(w, s, group_size)).to(out.dtype))

    tm.nvfp4_sm70_prepare = nvfp4_sm70_prepare
    tm.nvfp4_gemm_sm70_out = nvfp4_gemm_sm70_out
    if gemv:
        tm.nvfp4_tm_gemv_dispatch_sm70_out = nvfp4_tm_gemv_dispatch_sm70_out
    tm.has_op = lambda name: callable(getattr(tm, name, None))
    return tm


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    """vLLM parameters read the TP rank at construction (UP parameter.py:63-64)."""
    import vllm.model_executor.parameter as param_mod

    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture
def mock_tm(monkeypatch):
    def _install(gemv: bool = True):
        tm = make_mock_tm(gemv)
        ops = types.ModuleType("vllm_sm70.ops")
        ops.tm = tm
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", tm)
        return tm

    return _install


# --------------------------------------------------------------------------
# Fake device (same scheme as test_m2_quant_gates.py)
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
    """device(major, minor, ops=True): fake the current device and K01's
    readiness inputs (same scheme as test_m2_quant_gates.py)."""

    def _set(major: int, minor: int, ops: bool = True):
        cls = type("Dev", (_FakeCuda,), {})
        cls.capability = DeviceCapability(major, minor)
        monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
        sm70 = (major, minor) == (7, 0)
        monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: sm70)
        monkeypatch.setattr(_ext, "is_loaded", lambda: ops)
        sm70_platform.is_sm70_ready.cache_clear()

    yield _set
    sm70_platform.is_sm70_ready.cache_clear()


@pytest.fixture
def restore_registries():
    saved_q = dict(me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
    saved_b = dict(up_modelopt.LINEAR_METHOD_BUILDERS)
    yield
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.clear()
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.update(saved_q)
    up_modelopt.LINEAR_METHOD_BUILDERS.clear()
    up_modelopt.LINEAR_METHOD_BUILDERS.update(saved_b)


@pytest.fixture
def nvl(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", raising=False)
    import vllm_sm70.quant.nvfp4_linear as nvl

    return nvl


def _quant_config():
    from vllm_sm70.quant.configs import Sm70ModelOptNvFp4Config

    return Sm70ModelOptNvFp4Config.from_config(
        {k: (dict(v) if isinstance(v, dict) else v) for k, v in FLINT_QUANT_CFG.items()}
    )


def _loader(param, loaded):
    param.data.copy_(loaded)


def _make_layer(method, k, parts):
    layer = torch.nn.Module()
    method.create_weights(
        layer, k, list(parts), k, sum(parts), torch.float16, weight_loader=_loader
    )
    return layer


def _random_nvfp4(n, k, seed=0, global_scale=1e-3):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    block = (torch.rand((n, k // 16), generator=g) * 400 + 1).to(torch.float8_e4m3fn)
    return packed, block, torch.tensor(global_scale, dtype=torch.float32)


def _fill(layer, packed, block, gscale):
    layer.weight.data.copy_(packed)
    layer.weight_scale.data.copy_(block)
    layer.weight_scale_2.data.fill_(float(gscale))


def _reference(x, packed, block, gscale, fp16_scale: bool):
    """ModelOpt NVFP4 semantics, independent of the module under test:
    W[n, k] = e2m1(nibble k of row n; low nibble = even k)
              * block[n, k//16] * global."""
    n = packed.shape[0]
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(n, -1).long()
    scale = block.float() * gscale
    if (
        fp16_scale
    ):  # TurboMind folds block*global into FP16 (FORK sm70_turbomind.py:508-515)
        scale = scale.half().float()
    w = E2M1[codes] * scale.repeat_interleave(16, dim=1)
    return x.float() @ w.t()


# --------------------------------------------------------------------------
# CPU tests
# --------------------------------------------------------------------------
def test_create_weights_matches_upstream_w4a16(nvl, monkeypatch):
    """Flint TP2 qkv (K=2048, N=2048+256+256): same params as UP's own
    ModelOptLinearMethod for W4A16_NVFP4 (kernel selection stubbed out)."""
    cfg = _quant_config()
    parts = [2048, 256, 256]
    ours = _make_layer(nvl.ModelOptNvFp4Sm70LinearMethod(cfg, "q"), 2048, parts)

    monkeypatch.setattr(up_modelopt, "select_linear_kernel", lambda *a, **k: object())
    monkeypatch.setattr(up_modelopt, "expose_input_quant_key", lambda *a, **k: None)
    spec, ctx, fmt = up_modelopt.resolve("W4A16_NVFP4", cfg, "q")
    theirs = _make_layer(up_modelopt.ModelOptLinearMethod(spec, ctx, fmt), 2048, parts)

    def sig(layer):
        return {
            n: (type(p), p.dtype, tuple(p.shape)) for n, p in layer.named_parameters()
        }

    assert sig(ours) == sig(theirs)
    s = sig(ours)
    assert s["weight"] == (ModelWeightParameter, torch.uint8, (2560, 1024))
    assert s["weight_scale"] == (ModelWeightParameter, torch.float8_e4m3fn, (2560, 128))
    assert s["weight_scale_2"] == (PerTensorScaleParameter, torch.float32, (3,))
    assert s["input_scale"][1:] == (torch.float32, (3,))  # dropped after load
    assert ours.logical_widths == parts
    assert ours.output_size_per_partition == 2560
    assert (
        "ModelOptNvFp4Sm70LinearMethod"
        in __import__(
            "vllm.model_executor.layers.linear", fromlist=["x"]
        ).WEIGHT_LOADER_V2_SUPPORTED
    )


def test_process_weights_calls_prepare(nvl, mock_tm):
    tm = mock_tm()
    method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "o")
    layer = _make_layer(method, 512, [2048])
    packed, block, g = _random_nvfp4(2048, 512)
    _fill(layer, packed, block, g)
    method.process_weights_after_loading(layer)

    ((name, qweight, scales, gs, inter),) = [c for c in tm.calls if c[0] == "prepare"]
    assert qweight.shape == (512, 2048) and qweight.dtype == torch.uint8
    assert scales.shape == (32, 2048) and scales.dtype == torch.float16
    assert (gs, inter) == (16, False)
    # codes [K, N]: element (k, n) is nibble k of checkpoint row n.
    assert int(qweight[1, 3]) == int(packed[3, 0] >> 4)
    assert int(qweight[0, 3]) == int(packed[3, 0] & 15)
    assert torch.equal(scales, (block.float().t() * g).half())
    assert not hasattr(layer, "input_scale") and not hasattr(layer, "weight_scale_2")
    assert layer.weight.numel() == 0 and layer.weight_scale.numel() == 0
    st = getattr(layer, nvl.STATE_ATTR)
    assert (st.k_ld, st.q_ld, st.output_size) == (32 * 512, 2048, 2048)


def test_global_scale_folded_once(nvl, mock_tm):
    """The fp16 scales handed to prepare already include weight_scale_2
    (FORK sm70_turbomind.py:506-513), and nothing applies it again: scaling the
    global scale by 4 scales the scales by exactly 4 and the output by 4."""
    tm = mock_tm()
    packed, block, g = _random_nvfp4(512, 256, seed=3, global_scale=2.0**-10)
    x = torch.randn(2, 256).half()
    outs, scales = [], []
    for gs in (g, g * 4):
        method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "f")
        layer = _make_layer(method, 256, [512])
        _fill(layer, packed, block, gs)
        tm.calls.clear()
        method.process_weights_after_loading(layer)
        ((_, _, sc, _, _),) = [c for c in tm.calls if c[0] == "prepare"]
        scales.append(sc)
        assert torch.equal(sc, (block.t().float() * gs).half())
        assert torch.equal(getattr(layer, nvl.STATE_ATTR).scales, sc)
        outs.append(method.apply(layer, x).float())
    assert torch.equal(scales[1].float(), scales[0].float() * 4)
    torch.testing.assert_close(outs[1], outs[0] * 4, rtol=1e-3, atol=1e-3)
    ref = _reference(x, packed, block, g, fp16_scale=True)
    torch.testing.assert_close(outs[0], ref, rtol=2e-3, atol=2e-3)


def test_unloaded_scale_is_detected(nvl, mock_tm):
    mock_tm()
    method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "o")
    layer = _make_layer(method, 512, [2048])  # weight_scale stays NaN
    with pytest.raises(RuntimeError, match="never loaded"):
        method.process_weights_after_loading(layer)


def _prepared(nvl, k, n, seed=0):
    method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "p")
    layer = _make_layer(method, k, [n])
    data = _random_nvfp4(n, k, seed)
    _fill(layer, *data)
    method.process_weights_after_loading(layer)
    return method, layer, data


@pytest.mark.parametrize("m", [1, 2, 3, 17])
def test_gemv_route_when_enabled_and_present(nvl, mock_tm, m):
    tm = mock_tm(gemv=True)
    method, layer, _ = _prepared(nvl, 512, 2048)  # Flint shared-expert down (512, 2048)
    assert getattr(layer, nvl.STATE_ATTR).tm_gemv
    tm.calls.clear()
    method.apply(layer, torch.randn(m, 512).half())
    # One opaque dispatch call; the op runs the GEMV for M <= 2 itself.
    assert tm.calls[0][0] == ("gemv" if m <= 2 else "dispatch_gemm")


def test_gemm_when_gemv_op_absent(nvl, mock_tm):
    tm = mock_tm(gemv=False)
    method, layer, _ = _prepared(nvl, 512, 2048)
    assert not getattr(layer, nvl.STATE_ATTR).tm_gemv
    tm.calls.clear()
    method.apply(layer, torch.randn(1, 512).half())
    assert tm.calls == [("gemm", 1)]


def test_kill_switch_forces_gemm(nvl, mock_tm, monkeypatch):
    tm = mock_tm(gemv=True)
    monkeypatch.setenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", "0")
    method, layer, _ = _prepared(nvl, 512, 2048)
    assert method.gemv_enabled is False
    tm.calls.clear()
    method.apply(layer, torch.randn(2, 512).half())
    assert tm.calls == [("gemm", 2)]


def test_non_flint_shape_uses_gemm(nvl, mock_tm):
    tm = mock_tm(gemv=True)
    method, layer, _ = _prepared(nvl, 256, 512)
    tm.calls.clear()
    method.apply(layer, torch.randn(1, 256).half())
    assert tm.calls == [("gemm", 1)]


@pytest.mark.parametrize("has_gemv", [False, True])
def test_gemv_decision_with_real_c4130_meta(nvl, monkeypatch, has_gemv):
    """Reproduces the c4130 f31b6aa decision for the Flint TP2 "o" shape
    (K=N=2048, in MODELOPT_NVFP4_GEMV_SHAPES, FORK sm70_turbomind.py:569-579)
    with the meta Kai's real nvfp4_sm70_prepare returned there: k_ld=65536
    (=32*K) and q_ld=2048 (=N), with fp16 scales and the default config. Every
    admission check passes except the op probe, so the GEMV is selected only
    when K11's nvfp4_tm_gemv_dispatch_sm70_out is registered."""
    monkeypatch.delenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", raising=False)
    k = n = 2048
    state = nvl.SM70TurboMindLinearState(
        weight=torch.zeros((k, n // 8), dtype=torch.int32),
        scales=torch.full((k // 16, n), 2.0**-10, dtype=torch.float16),
        group_size=16,
        k_ld=65536,
        q_ld=2048,
        output_size=n,
        padded_output_size=n,
    )
    method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "o_proj")
    assert method.gemv_enabled is True  # the sm70_config gate is open
    assert (k, n) in nvl.MODELOPT_NVFP4_GEMV_SHAPES
    probed = []

    def has_tm_op(op):
        probed.append(op)
        return has_gemv or op != nvl.GEMV_OP

    monkeypatch.setattr(nvl, "has_tm_op", has_tm_op)
    admitted = nvl.admit_modelopt_nvfp4_gemv(state, enabled=method.gemv_enabled)
    assert probed == [nvl.GEMV_OP]  # reached the last check: only the op decides
    assert admitted is has_gemv and state.tm_gemv is has_gemv
    assert state.provider == (nvl.GEMV_OP if has_gemv else nvl.GEMM_OP)
    assert state.tm_gemv_scale_shift == 14


def test_scale_shift(nvl):
    assert nvl._tm_gemv_scale_shift(torch.tensor([1.0]).half()) == 14
    assert nvl._tm_gemv_scale_shift(torch.tensor([8.0]).half()) == 12
    assert nvl._tm_gemv_scale_shift(torch.tensor([60000.0]).half()) == 0


@pytest.mark.parametrize("k, n, m", [(512, 2048, 1), (256, 512, 5), (64, 48, 3)])
def test_numerics_vs_reference_dequant(nvl, mock_tm, k, n, m):
    """Self-consistency of unpack/format/pad against an independent ModelOpt
    dequant. n=48 exercises the 32-column padding and crop."""
    mock_tm()
    method, layer, (packed, block, g) = _prepared(nvl, k, n, seed=k + n)
    x = torch.randn(m, k).half()
    bias = torch.randn(n).half()
    y = method.apply(layer, x, bias)
    assert y.shape == (m, n) and y.dtype == torch.float16
    ref16 = _reference(x, packed, block, g, fp16_scale=True) + bias.float()
    torch.testing.assert_close(y.float(), ref16, rtol=2e-3, atol=2e-3)
    ref32 = _reference(x, packed, block, g, fp16_scale=False) + bias.float()
    torch.testing.assert_close(y.float(), ref32, rtol=2e-2, atol=2e-2)


def test_3d_input_restored(nvl, mock_tm):
    mock_tm()
    method, layer, _ = _prepared(nvl, 256, 512)
    assert method.apply(layer, torch.randn(2, 3, 256).half()).shape == (2, 3, 512)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_non_fp16_activation_raises_type_error(nvl, mock_tm, dtype):
    """apply() declares a TypeError for non-FP16 activations and raises it
    before any op runs (no silent cast, no kernel call)."""
    tm = mock_tm()
    method, layer, _ = _prepared(nvl, 256, 512)
    tm.calls.clear()
    with pytest.raises(TypeError, match=r"needs float16 activations.*--dtype half"):
        method.apply(layer, torch.randn(1, 256).to(dtype))
    assert tm.calls == []
    assert "TypeError" in nvl.apply_prepared_linear.__doc__
    assert "TypeError" in nvl.ModelOptNvFp4Sm70LinearMethod.apply.__doc__


def test_online_fp8_on_sm70_is_refused_by_upstream(
    device, restore_registries, monkeypatch, tmp_path
):
    """Online ``--quantization fp8`` (checkpoint without a quant config) on
    SM70, with M2 + M3 installed. Upstream rewrites it to
    OnlineQuantizationConfig (UP 4ba787e weight_utils.py:355-363), whose min
    capability is 75 (online/base.py:209-212). It therefore never reaches
    Sm70Fp8Config's gate, and VllmConfig refuses cc 7.0 (config/vllm.py:926-936).
    Nothing picks a kernel silently."""
    from types import SimpleNamespace

    from vllm.config.vllm import VllmConfig
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.model_loader.weight_utils import get_quant_config

    monkeypatch.delenv("VLLM_SM70_FP8_TURBOMIND", raising=False)
    device(7, 0)
    configs = _reload_configs(monkeypatch)
    assert configs.REGISTERED and configs.M3_INSTALLED
    assert configs.Sm70Fp8Config.get_min_capability() == 70  # M2 gate is open
    model_config = SimpleNamespace(
        quantization="fp8",
        quantization_config=None,
        hf_config=SimpleNamespace(),  # unquantized checkpoint
        hf_overrides={},
        model=str(tmp_path),
        revision=None,
        dtype=torch.float16,
    )
    load_config = SimpleNamespace(download_dir=None)
    built = get_quant_config(model_config, load_config)
    assert type(built) is OnlineQuantizationConfig
    assert not isinstance(built, configs.Sm70Fp8Config)
    assert built.get_min_capability() == 75
    with pytest.raises(
        ValueError, match="Minimum capability: 75. Current capability: 70"
    ):
        VllmConfig._get_quantization_config(model_config, load_config)


def test_fp8_config_refuses_online_construction():
    """The other route, an Fp8Config built for a non-fp8 checkpoint, raises
    in upstream's constructor (UP 4ba787e fp8.py:106-112)."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    with pytest.raises(ValueError, match="no longer supports online"):
        Fp8Config(is_checkpoint_fp8_serialized=False)


def test_missing_ops_raise(nvl, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops", None)  # import fails
    method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "o")
    layer = _make_layer(method, 256, [512])
    _fill(layer, *_random_nvfp4(512, 256))
    with pytest.raises(RuntimeError, match="vllm_sm70.ops.tm"):
        method.process_weights_after_loading(layer)


def test_group_size_must_be_16(nvl):
    cfg = _quant_config()
    cfg.group_size = 32
    with pytest.raises(ValueError, match="group_size 16"):
        nvl.ModelOptNvFp4Sm70LinearMethod(cfg, "o")


def _reload_configs(monkeypatch, env=None):
    if env is None:
        monkeypatch.delenv("VLLM_SM70_QUANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_QUANT", env)
    up_modelopt.LINEAR_METHOD_BUILDERS.pop("W4A16_NVFP4", None)
    import vllm_sm70.quant.configs as configs

    return importlib.reload(configs)


def test_hook_installs_on_sm70(device, restore_registries, monkeypatch, nvl):
    device(7, 0)
    configs = _reload_configs(monkeypatch)
    assert configs.M3_INSTALLED and nvl.builder_installed()
    configs.register()  # includes the M3 D4 check
    method = up_modelopt.build_linear_method(_quant_config(), "W4A16_NVFP4", "x")
    assert type(method) is nvl.ModelOptNvFp4Sm70LinearMethod
    up_modelopt.LINEAR_METHOD_BUILDERS["W4A16_NVFP4"] = lambda c, p: None
    with pytest.raises(RuntimeError, match="replaced after M3"):
        configs.register()


@pytest.mark.parametrize("cap, ops", [((8, 0), True), ((7, 5), True), ((7, 0), False)])
def test_hook_not_installed_off_sm70(device, restore_registries, monkeypatch, cap, ops):
    device(*cap, ops=ops)
    configs = _reload_configs(monkeypatch)
    assert not configs.M3_INSTALLED
    assert "W4A16_NVFP4" not in up_modelopt.LINEAR_METHOD_BUILDERS


def test_hook_respects_quant_kill(device, restore_registries, monkeypatch):
    device(7, 0)
    configs = _reload_configs(monkeypatch, "0")
    assert not configs.REGISTERED and not configs.M3_INSTALLED


def test_builder_falls_through_at_cc80(device, restore_registries, nvl):
    """Builder installed (e.g. imported on an SM70 host) but asked on cc 8.0:
    upstream's generic method, exactly as with no builder."""
    nvl.install_linear_builder()
    device(8, 0)
    method = up_modelopt.build_linear_method(_quant_config(), "W4A16_NVFP4", "x")
    assert type(method) is up_modelopt.ModelOptLinearMethod
    assert (
        method.spec.weight == up_modelopt.kNvfp4Static
        and method.spec.activation is None
    )
    nvl.uninstall_linear_builder()
    assert "W4A16_NVFP4" not in up_modelopt.LINEAR_METHOD_BUILDERS


def test_builder_chains_existing(device, restore_registries, nvl):
    sentinel = object()
    up_modelopt.LINEAR_METHOD_BUILDERS["W4A16_NVFP4"] = lambda c, p: sentinel
    nvl.install_linear_builder()
    nvl.install_linear_builder()  # idempotent
    device(8, 0)
    assert (
        up_modelopt.build_linear_method(_quant_config(), "W4A16_NVFP4", "x") is sentinel
    )
    nvl.uninstall_linear_builder()
    nvl._upstream_builder = None


# --------------------------------------------------------------------------
# GPU tests (c4130 only): real Kai ops via vllm_sm70.ops.tm
# --------------------------------------------------------------------------
def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        return False
    try:
        from vllm_sm70.ops import tm  # noqa: F401
    except ImportError:
        return False
    return True


# Boolean, not a string: a string condition is evaluated in the decorated
# function's globals, which are torch's when @torch.inference_mode() wraps it.
needs_sm70 = pytest.mark.skipif(
    not _gpu_ready(), reason="needs an SM70 GPU and vllm_sm70.ops"
)


def _has_k11_gemv() -> bool:
    try:
        from vllm_sm70.ops import tm
    except ImportError:
        return False
    return tm.has_op("nvfp4_tm_gemv_dispatch_sm70_out")


# K11 (Phase 4) ships the GEMV op; main's K03 build does not
# (sm70/csrc/turbomind/PROVENANCE.md:258). Without it, admission refuses the
# layer and the GEMM serves every M, which is correct behaviour.
needs_k11_gemv = pytest.mark.skipif(
    "not _has_k11_gemv()",
    reason="needs _sm70_C::nvfp4_tm_gemv_dispatch_sm70_out (K11, not in K03)",
)


# Flint TP2 per-rank (K, N), FORK sm70_turbomind.py:567-579
FLINT_SHAPES = [
    (2048, 2560),
    (2048, 2048),
    (2048, 1024),
    (512, 2048),
    (2048, 6144),
    (3072, 2048),
]


def _gpu_prepared(nvl, k, n, seed, monkeypatch, gemv=True):
    if not gemv:
        monkeypatch.setenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", "0")
    method = nvl.ModelOptNvFp4Sm70LinearMethod(_quant_config(), "g")
    with torch.device("cuda"):
        layer = _make_layer(method, k, [n])
    data = _random_nvfp4(n, k, seed)
    _fill(layer, *(t.cuda() for t in data))
    method.process_weights_after_loading(layer)
    return method, layer, data


@pytest.mark.gpu
@needs_sm70
@pytest.mark.parametrize("k, n", FLINT_SHAPES + [(2048, 48 * 32 + 16)])
@pytest.mark.parametrize("m", [1, 2, 3, 16, 128, 1024])
@pytest.mark.parametrize("gemv", [True, False])
def test_gpu_matches_reference(nvl, monkeypatch, k, n, m, gemv):
    method, layer, (packed, block, g) = _gpu_prepared(
        nvl, k, n, k ^ n, monkeypatch, gemv
    )
    st = getattr(layer, nvl.STATE_ATTR)
    assert st.tm_gemv == (
        gemv and (k, n) in nvl.MODELOPT_NVFP4_GEMV_SHAPES and nvl.has_tm_op(nvl.GEMV_OP)
    )
    x = (torch.randn(m, k) * 0.5).half()
    y = method.apply(layer, x.cuda()).float().cpu()
    ref = _reference(x, packed, block, g, fp16_scale=True)
    # FP16 TurboMind accumulation: loose bound vs fp32 reference.
    torch.testing.assert_close(y, ref, rtol=3e-2, atol=3e-2 * float(ref.abs().mean()))


@pytest.mark.gpu
@needs_sm70
@needs_k11_gemv
@pytest.mark.parametrize("k, n", FLINT_SHAPES)
@pytest.mark.parametrize("m", [1, 2])
def test_gpu_gemv_equals_gemm(nvl, monkeypatch, k, n, m):
    x = torch.randn(m, k).half().cuda()
    m1, l1, _ = _gpu_prepared(nvl, k, n, 7, monkeypatch, gemv=True)
    y1 = m1.apply(l1, x)
    monkeypatch.setenv("VLLM_SM70_MODELOPT_NVFP4_GEMV", "0")
    m2, l2, _ = _gpu_prepared(nvl, k, n, 7, monkeypatch, gemv=False)
    y2 = m2.apply(l2, x)
    torch.testing.assert_close(y1, y2, rtol=1e-2, atol=1e-2)


def _check_strided_rows(nvl, monkeypatch, gemv: bool) -> None:
    """Port of FORK cf9492a tests/kernels/quantization/
    test_sm70_nvfp4_tm_gemv.py:157-165. A row-strided (non-contiguous) M=2
    activation on the Flint "o" shape (K=N=2048, one of the six admitted
    shapes, FORK sm70_turbomind.py:569-579) must give exactly the result of
    its contiguous copy. flatten_linear_input passes 2-D views through
    unchanged, so the op sees the row stride."""
    method, layer, _ = _gpu_prepared(nvl, 2048, 2048, 1, monkeypatch, gemv=gemv)
    assert (2048, 2048) in nvl.MODELOPT_NVFP4_GEMV_SHAPES
    assert getattr(layer, nvl.STATE_ATTR).tm_gemv is gemv
    k = 2048
    wide = torch.randn(2, k + 64, device="cuda").half()
    strided = wide[:, :k]
    assert not strided.is_contiguous()
    out = method.apply(layer, strided)
    assert torch.equal(out, method.apply(layer, strided.clone()))


@pytest.mark.gpu
@needs_sm70
@torch.inference_mode()
def test_strided_activation_rows_gemm(nvl, monkeypatch):
    """Strided rows on the TurboMind GEMM path (GEMV killed). Always runs."""
    _check_strided_rows(nvl, monkeypatch, gemv=False)


@pytest.mark.gpu
@needs_sm70
@needs_k11_gemv
@torch.inference_mode()
def test_strided_activation_rows(nvl, monkeypatch):
    """Strided rows on the M<=2 GEMV. Skips without K11's op, as the fork's
    test did (FORK test_sm70_nvfp4_tm_gemv.py:29-37, ``_requires_gemv``)."""
    _check_strided_rows(nvl, monkeypatch, gemv=True)


@pytest.mark.gpu
@needs_sm70
def test_gpu_cuda_graph_capture(nvl, monkeypatch):
    method, layer, _ = _gpu_prepared(nvl, 2048, 2560, 3, monkeypatch)
    x = torch.randn(1, 2048).half().cuda()
    eager = method.apply(layer, x).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = method.apply(layer, x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
