# SPDX-License-Identifier: Apache-2.0
"""Tests for M2 sm70-quant-gates (vllm_sm70/quant/configs.py). CPU only.

What: checks the capability gate for each (capability, SM70 ops) case,
  registry identity, the VLLM_SM70_QUANT kill switch, unchanged upstream
  behaviour at cc 8.0, and that upstream's own checks accept SM70: the
  capability gate (UP 4ba787e config/vllm.py:926-936) and quant-method
  detection (config/model.py:1382-1450). Fixtures use the real Flint and
  Qwen FP8 checkpoint shapes.
Why: the gate is the first thing an SM70 boot hits, and an upstream rebase
  can silently undo the override (D4).
Kill-switch: VLLM_SM70_QUANT is covered below.

The device is faked in two places. ``vllm.platforms._current_platform``
feeds upstream's own capability check, which upstream resolves lazily
(vllm/platforms/__init__.py:302-321). K01's ``vllm_sm70.platform.is_sm70_ready``
gets its inputs patched: ``all_visible_gpus_are_sm70`` (NVML) and
``vllm_sm70._ext.is_loaded`` (the _sm70_C extension).
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization as me_quant
import vllm.platforms
from vllm.config.model import ModelConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

import vllm_sm70.quant as sm70_quant
from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform

# Flint config.json quantization_config (Theo, pinned checkpoint; FORK
# cf9492a vllm/models/flint/convert_gguf.py:114-123).
FLINT_QUANT_CFG = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["model.embed_tokens", "*.mlp.gate", "lm_head"],
    },
}
# Qwen3.6/3.8 FP8 quantization_config.
QWEN_FP8_QUANT_CFG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
CASES = {
    "fp8": (Fp8Config, QWEN_FP8_QUANT_CFG),
    "modelopt_fp4": (ModelOptNvFp4Config, FLINT_QUANT_CFG),
}


class _FakeCuda(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"
    capability: DeviceCapability | None = None

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return cls.capability


class _FakeCpu(_FakeCuda):
    _enum = PlatformEnum.CPU
    device_name = "cpu"
    device_type = "cpu"


@pytest.fixture
def device(monkeypatch):
    """device(major, minor, ops=True, cuda=True): fake the current device."""

    monkeypatch.delenv("VLLM_SM70_FP8_TURBOMIND", raising=False)

    def _set(major: int, minor: int, ops: bool = True, cuda: bool = True):
        cls = type("Dev", (_FakeCuda if cuda else _FakeCpu,), {})
        cls.capability = DeviceCapability(major, minor)
        monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
        sm70 = cuda and (major, minor) == (7, 0)
        monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: sm70)
        monkeypatch.setattr(_ext, "is_loaded", lambda: ops)
        sm70_platform.is_sm70_ready.cache_clear()

    yield _set
    sm70_platform.is_sm70_ready.cache_clear()


@pytest.fixture
def load_configs(monkeypatch):
    """Import vllm_sm70.quant.configs fresh under the given env, then put
    upstream's registry of customized quant configs back."""
    saved = dict(me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    def _load(quant_env: str | None = None):
        if quant_env is None:
            monkeypatch.delenv("VLLM_SM70_QUANT", raising=False)
        else:
            monkeypatch.setenv("VLLM_SM70_QUANT", quant_env)
        for name in ("fp8", "modelopt_fp4"):
            me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop(name, None)
        import vllm_sm70.quant.configs as configs

        return importlib.reload(configs)

    yield _load
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.clear()
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.update(saved)


def _ours(configs, name):
    return configs.SM70_QUANT_CONFIGS[name]


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize(
    "cap, ops, cuda, sm70",
    [
        ((7, 0), True, True, True),
        ((7, 0), False, True, False),  # extension not loaded
        ((7, 0), True, False, False),  # not CUDA
        ((7, 5), True, True, False),
        ((8, 0), True, True, False),
        ((6, 0), True, True, False),
    ],
)
def test_gate_value(device, load_configs, name, cap, ops, cuda, sm70):
    configs = load_configs()
    device(*cap, ops=ops, cuda=cuda)
    upstream_cls, _ = CASES[name]
    assert sm70_platform.is_sm70_ready() is sm70
    got = _ours(configs, name).get_min_capability()
    assert got == (70 if sm70 else upstream_cls.get_min_capability())


@pytest.mark.parametrize("name", list(CASES))
def test_cc80_identical_to_upstream(device, load_configs, name):
    configs = load_configs()
    device(8, 0)
    upstream_cls, raw = CASES[name]
    ours = _ours(configs, name)
    assert ours.get_min_capability() == upstream_cls.get_min_capability() == 75
    a, b = ours.from_config(dict(raw)), upstream_cls.from_config(dict(raw))
    assert a.get_supported_act_dtypes() == b.get_supported_act_dtypes()
    assert a.get_name() == b.get_name()


def test_registry_identity_and_register(load_configs):
    configs = load_configs()
    assert configs.REGISTERED
    assert me_quant.get_quantization_config("fp8") is configs.Sm70Fp8Config
    assert (
        me_quant.get_quantization_config("modelopt_fp4")
        is configs.Sm70ModelOptNvFp4Config
    )
    assert issubclass(configs.Sm70Fp8Config, Fp8Config)
    assert issubclass(configs.Sm70ModelOptNvFp4Config, ModelOptNvFp4Config)
    configs.register()  # D4 asserts pass


def test_register_detects_overwritten_slot(load_configs):
    configs = load_configs()
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["fp8"] = Fp8Config
    with pytest.raises(RuntimeError, match="'fp8' resolves to"):
        configs.register()


def test_reimport_is_idempotent(load_configs):
    load_configs()
    import vllm_sm70.quant.configs as configs

    configs = importlib.reload(configs)
    assert me_quant.QUANTIZATION_METHODS.count("fp8") == 1
    assert me_quant.QUANTIZATION_METHODS.count("modelopt_fp4") == 1
    assert me_quant.get_quantization_config("fp8") is configs.Sm70Fp8Config
    configs.register()


def test_kill_switch_keeps_upstream(device, load_configs):
    configs = load_configs("0")
    device(7, 0)
    assert not configs.REGISTERED
    assert me_quant.get_quantization_config("fp8") is Fp8Config
    assert me_quant.get_quantization_config("modelopt_fp4") is ModelOptNvFp4Config
    configs.register()  # no-op, no raise
    for name, (upstream_cls, raw) in CASES.items():
        with pytest.raises(ValueError, match="not supported for the current GPU"):
            _capability_check(name, raw)


@pytest.mark.parametrize("value", ["1", "yes"])
def test_kill_switch_other_values_keep_enabled(load_configs, value):
    # Kill-only: only "0" disables; anything else is ignored (vllm_sm70/envs.py).
    assert load_configs(value).REGISTERED


def _capability_check(name, raw):
    """Run upstream's VllmConfig._get_quantization_config (config/vllm.py
    :917-950) with get_quant_config stubbed to build from ``raw``."""
    from vllm.model_executor.model_loader import weight_utils

    model_config = SimpleNamespace(
        quantization=name,
        dtype=torch.float16,
        model="stub",
        hf_config=None,
        revision=None,
    )
    built = me_quant.get_quantization_config(name).from_config(dict(raw))
    mp = pytest.MonkeyPatch()
    with mp.context() as m:
        m.setattr(weight_utils, "get_quant_config", lambda mc, lc: built)
        return VllmConfig._get_quantization_config(model_config, None)


@pytest.mark.parametrize("name", list(CASES))
def test_upstream_capability_check_passes_on_sm70(device, load_configs, name):
    configs = load_configs()
    device(7, 0)
    cfg = _capability_check(name, CASES[name][1])
    assert type(cfg) is _ours(configs, name)


@pytest.mark.parametrize("name", list(CASES))
def test_upstream_capability_check_refuses_without_ops(device, load_configs, name):
    load_configs()
    device(7, 0, ops=False)
    with pytest.raises(ValueError, match="Minimum capability: 75"):
        _capability_check(name, CASES[name][1])


@pytest.mark.parametrize(
    "raw, expected", [(FLINT_QUANT_CFG, "modelopt_fp4"), (QWEN_FP8_QUANT_CFG, "fp8")]
)
def test_detection_lands_on_our_subclass(load_configs, raw, expected):
    """Upstream's quant-method detection (config/model.py:1382-1450) maps
    each checkpoint shape to our slot, and from_config builds our class."""
    configs = load_configs()
    stub = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(quantization_config=dict(raw)),
        hf_config=None,
        allow_deprecated_quantization=False,
    )
    ModelConfig._verify_quantization(stub)
    assert stub.quantization == expected
    built = me_quant.get_quantization_config(stub.quantization).from_config(dict(raw))
    assert type(built) is _ours(configs, expected)
    if expected == "modelopt_fp4":
        assert built.quant_method == "W4A16_NVFP4"
        assert built.group_size == 16
    else:
        assert built.weight_block_size == [128, 128]


@pytest.mark.parametrize("name", list(CASES))
def test_pickled_config_keeps_our_class(load_configs, name):
    """The engine core unpickles vllm_config.quant_config before it loads
    plugins (UP v1/engine/core.py:1462-1464), so the class must survive the
    round trip."""
    import pickle

    configs = load_configs()
    built = _ours(configs, name).from_config(dict(CASES[name][1]))
    assert type(pickle.loads(pickle.dumps(built))) is _ours(configs, name)


@pytest.fixture
def fresh_quant_area(monkeypatch):
    monkeypatch.setattr(sm70_quant, "_registered", False)
    yield


def test_area_register_runs_checks_and_is_idempotent(
    load_configs, fresh_quant_area, monkeypatch
):
    configs = load_configs()
    calls = []
    real = configs.register
    monkeypatch.setattr(configs, "register", lambda: (calls.append(1), real()))
    sm70_quant.register()
    sm70_quant.register()
    assert calls == [1]
    assert sm70_quant._registered


def test_area_register_retries_after_failure(
    load_configs, fresh_quant_area, monkeypatch
):
    """K03's import_kernels() retries a failed area once."""
    configs = load_configs()
    real = configs.register
    outcomes = iter([RuntimeError("simulated failure"), None])
    calls = []

    def flaky():
        calls.append(1)
        err = next(outcomes)
        if err is not None:
            raise err
        real()

    monkeypatch.setattr(configs, "register", flaky)
    with pytest.raises(RuntimeError, match="simulated failure"):
        sm70_quant.register()
    assert not sm70_quant._registered
    sm70_quant.register()  # retry succeeds
    assert sm70_quant._registered
    sm70_quant.register()  # then a no-op
    assert calls == [1, 1]


def test_area_register_raises_on_broken_override(load_configs, fresh_quant_area):
    """Fail-fast: a broken override raises out of quant.register() (never
    caught and logged here), and the area stays unregistered."""
    load_configs()
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["fp8"] = Fp8Config
    with pytest.raises(RuntimeError, match="'fp8' resolves to"):
        sm70_quant.register()
    assert not sm70_quant._registered


def test_kill_switch_area_register_is_noop(load_configs, fresh_quant_area):
    configs = load_configs("0")
    sm70_quant.register()
    assert not configs.REGISTERED
    assert me_quant.get_quantization_config("fp8") is Fp8Config


def test_platform_module_does_not_import_quant():
    """vllm_sm70.platform must not pull in the quant configs (re-entrancy)."""
    import subprocess
    import sys

    code = (
        "import sys, vllm_sm70.platform; "
        "assert 'vllm_sm70.quant.configs' not in sys.modules; "
        "assert 'vllm.model_executor.layers.quantization' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fp8_turbomind_kill_restores_upstream_fp8_gate(device, load_configs, monkeypatch):
    """cc 7.0 + _sm70_C loaded + VLLM_SM70_FP8_TURBOMIND=0: the fp8 gate is
    upstream's 75 and upstream refuses; modelopt_fp4 (Flint) keeps 70."""
    configs = load_configs()
    device(7, 0)
    monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", "0")
    assert sm70_platform.is_sm70_ready()
    assert _ours(configs, "fp8").get_min_capability() == 75
    with pytest.raises(ValueError, match="Minimum capability: 75"):
        _capability_check("fp8", QWEN_FP8_QUANT_CFG)
    assert _ours(configs, "modelopt_fp4").get_min_capability() == 70
    cfg = _capability_check("modelopt_fp4", FLINT_QUANT_CFG)
    assert type(cfg) is _ours(configs, "modelopt_fp4")


@pytest.mark.parametrize("value", [None, "1", "yes"])
def test_fp8_turbomind_other_values_keep_sm70_gate(device, load_configs, monkeypatch, value):
    configs = load_configs()
    device(7, 0)
    if value is None:
        monkeypatch.delenv("VLLM_SM70_FP8_TURBOMIND", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", value)
    assert _ours(configs, "fp8").get_min_capability() == 70
