# SPDX-License-Identifier: Apache-2.0
"""Tests for inkling2-sm70-quant (vllm_sm70/models/inkling2_sm70/quant_config.py).

CPU only: upstream's quant-method detection (UP config/model.py:1382-1450)
resolves the Inkling 2 Mini config to "inkling2" at faked cc 7.0 and 8.0 and
still resolves Flint to "modelopt_fp4" and Qwen to "fp8"; upstream's
capability and dtype check accepts fp16; registry identity, pickling and the
VLLM_SM70_QUANT kill switch; the expert formats read from
hf_quant_config.json (data/inkling2_mini/ and the tiny fixture).
"""

import copy
import importlib
import json
import pathlib
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch
import vllm.model_executor.layers.quantization as me_quant
import vllm.platforms
from vllm.config.model import ModelConfig
from vllm.config.vllm import VllmConfig
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

import vllm_sm70.quant as sm70_quant
import vllm_sm70.quant.configs as sm70_quant_configs
from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.models import inkling2_sm70

MODELS = pathlib.Path(__file__).resolve().parents[1] / "models"
MINI = MODELS / "data" / "inkling2_mini"
TINY = MODELS / "fixtures" / "inkling2_tiny"

# As tests/quant/test_m2_quant_gates.py.
FLINT_QUANT_CFG = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["model.embed_tokens", "*.mlp.gate", "lm_head"],
    },
}
QWEN_FP8_QUANT_CFG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
EXPECTED_FORMATS = {
    **dict.fromkeys(range(20), "nvfp4"),
    **dict.fromkeys(range(20, 24), "mxfp8"),
}


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
    def _set(major, minor):
        cls = type("Dev", (_FakeCuda,), {})
        cls.capability = DeviceCapability(major, minor)
        monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
        sm70 = (major, minor) == (7, 0)
        monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: sm70)
        monkeypatch.setattr(_ext, "is_loaded", lambda: True)
        sm70_platform.is_sm70_ready.cache_clear()

    yield _set
    sm70_platform.is_sm70_ready.cache_clear()


@pytest.fixture
def load(monkeypatch):
    """Import quant_config fresh under the given VLLM_SM70_QUANT, then put
    upstream's registry and vllm_sm70.quant.configs' classes back."""
    saved = dict(me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
    saved_names = list(me_quant.QUANTIZATION_METHODS)
    configs_state = dict(vars(sm70_quant_configs))

    def _load(quant_env: str | None = None):
        if quant_env is None:
            monkeypatch.delenv("VLLM_SM70_QUANT", raising=False)
        else:
            monkeypatch.setenv("VLLM_SM70_QUANT", quant_env)
        me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop("inkling2", None)
        if "inkling2" in me_quant.QUANTIZATION_METHODS:
            me_quant.QUANTIZATION_METHODS.remove("inkling2")
        from vllm_sm70.models.inkling2_sm70 import quant_config

        return importlib.reload(quant_config)

    yield _load
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.clear()
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.update(saved)
    me_quant.QUANTIZATION_METHODS[:] = saved_names
    vars(sm70_quant_configs).update(configs_state)
    # Leave the registry naming the module's current class for later tests.
    _load()


def _hf_config(path):
    from vllm.transformers_utils.config import get_config

    inkling2_sm70.register_configs()
    return get_config(str(path), trust_remote_code=False)


def _detect(quant_cfg, hf_config, user_quant=None):
    stub = SimpleNamespace(
        quantization=user_quant,
        model_arch_config=SimpleNamespace(quantization_config=copy.deepcopy(quant_cfg)),
        hf_config=hf_config,
        allow_deprecated_quantization=False,
    )
    ModelConfig._verify_quantization(stub)
    return stub.quantization


@pytest.mark.parametrize("cap", [(7, 0), (8, 0)])
@pytest.mark.parametrize("path", [MINI, TINY], ids=["mini", "tiny"])
def test_inkling_config_resolves_to_inkling2(device, load, cap, path):
    qc = load()
    device(*cap)
    hf_config = _hf_config(path)
    assert hf_config.model_type == "inkling2_loop"
    quant_cfg = hf_config.quantization_config
    assert quant_cfg == {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    assert _detect(quant_cfg, hf_config) == "inkling2"
    assert me_quant.get_quantization_config("inkling2") is qc.Inkling2Sm70QuantConfig


def test_resolves_with_overridden_architectures(load):
    load()
    hf_config = _hf_config(MINI)
    hf_config.architectures = [inkling2_sm70.ARCH]
    assert _detect(hf_config.quantization_config, hf_config) == "inkling2"


@pytest.mark.parametrize("cap", [(7, 0), (8, 0)])
@pytest.mark.parametrize(
    "raw, model_type, expected",
    [
        (FLINT_QUANT_CFG, "flint", "modelopt_fp4"),
        (QWEN_FP8_QUANT_CFG, "qwen3_moe", "fp8"),
    ],
)
def test_flint_and_qwen_unchanged(device, load, cap, raw, model_type, expected):
    load()
    # Fresh, so the registry and the module name the same class objects.
    configs = importlib.reload(sm70_quant_configs)
    device(*cap)
    for hf_config in (
        None,
        SimpleNamespace(model_type=model_type, architectures=["X"]),
    ):
        assert _detect(raw, hf_config) == expected
    cls = me_quant.get_quantization_config(expected)
    assert cls is configs.SM70_QUANT_CONFIGS[expected]
    assert type(cls.from_config(copy.deepcopy(raw))) is cls
    configs.register()


def test_class_contract(load):
    qc = load()
    cls = qc.Inkling2Sm70QuantConfig
    cfg = cls.from_config({"quant_method": "modelopt", "quant_algo": "NVFP4"})
    assert type(cfg) is cls
    assert cfg.get_name() == "inkling2"
    assert cls.get_min_capability() == 70
    assert cfg.get_supported_act_dtypes() == [torch.float16]
    assert cls.get_config_filenames() == ["hf_quant_config.json"]
    assert cfg.get_quant_method(torch.nn.Linear(2, 2), "model.llm.layers.0.mlp") is None
    assert cfg.original_config == {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    assert cls.override_quantization_method({}, None) is None
    assert cls.override_quantization_method({}, None, hf_config=None) is None
    other = SimpleNamespace(model_type="inkling2")
    assert cls.override_quantization_method({}, None, hf_config=other) is None


def _quant_config(dtype):
    """Upstream's VllmConfig._get_quantization_config (UP config/vllm.py
    :917-950) with the real get_quant_config on the Mini config.json."""
    model_config = SimpleNamespace(
        quantization="inkling2",
        quantization_config=None,
        dtype=dtype,
        model=str(MINI),
        hf_config=_hf_config(MINI),
        hf_overrides={},
        revision=None,
    )
    return VllmConfig._get_quantization_config(model_config, None)


@pytest.mark.parametrize("cap", [(7, 0), (8, 0)])
def test_upstream_capability_and_dtype_check(device, load, cap):
    qc = load()
    device(*cap)
    assert type(_quant_config(torch.float16)) is qc.Inkling2Sm70QuantConfig
    with pytest.raises(ValueError, match="bfloat16 is not supported for quantization"):
        _quant_config(torch.bfloat16)


def test_pickle_and_reimport(load):
    qc = load()
    built = qc.Inkling2Sm70QuantConfig.from_config({"quant_method": "modelopt"})
    assert type(pickle.loads(pickle.dumps(built))) is qc.Inkling2Sm70QuantConfig
    qc = importlib.reload(qc)
    assert me_quant.QUANTIZATION_METHODS.count("inkling2") == 1
    assert me_quant.get_quantization_config("inkling2") is qc.Inkling2Sm70QuantConfig
    qc.register()


def test_custom_method_is_probed_before_builtins(load):
    load()
    assert "inkling2" in me_quant.QUANTIZATION_METHODS
    from typing import get_args

    assert "inkling2" not in get_args(me_quant.QuantizationMethods)


def test_register_detects_overwritten_slot(load):
    qc = load()
    me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["inkling2"] = (
        sm70_quant_configs.Sm70Fp8Config
    )
    with pytest.raises(RuntimeError, match="'inkling2' resolves to"):
        qc.register()


def test_register_restores_missing_slot(load):
    qc = load()
    del me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["inkling2"]
    qc.register()
    assert me_quant.get_quantization_config("inkling2") is qc.Inkling2Sm70QuantConfig
    me_quant.QUANTIZATION_METHODS.remove("inkling2")
    del me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["inkling2"]
    qc.register()
    assert me_quant.get_quantization_config("inkling2") is qc.Inkling2Sm70QuantConfig
    assert me_quant.QUANTIZATION_METHODS.count("inkling2") == 1


def test_kill_switch_keeps_upstream(load):
    qc = load("0")
    assert not qc.REGISTERED
    assert "inkling2" not in me_quant.QUANTIZATION_METHODS
    qc.register()  # no-op
    hf_config = _hf_config(MINI)
    assert _detect(hf_config.quantization_config, hf_config) == "modelopt_fp4"


def test_quant_area_hook_checks_inkling2(load, monkeypatch):
    """vllm_sm70.quant.register() runs in every process, before the models
    area and outside its cc 7.0 gate."""
    qc = load()
    calls = []
    monkeypatch.setattr(sm70_quant, "_registered", False)
    monkeypatch.setattr(sm70_quant_configs, "register", lambda: None)
    monkeypatch.setattr(qc, "register", lambda: calls.append(1))
    sm70_quant.register()
    assert calls == [1]


@pytest.mark.parametrize("path", [MINI, TINY], ids=["mini", "tiny"])
def test_expert_formats_from_test_data(load, path):
    qc = load()
    raw = json.loads((path / "hf_quant_config.json").read_text(encoding="utf-8"))
    assert qc.expert_formats(raw) == EXPECTED_FORMATS
    assert qc.mxfp8_layers(raw) == frozenset({20, 21, 22, 23})
    assert qc.load_expert_formats(path) == EXPECTED_FORMATS
    assert qc.load_expert_formats(str(path), "main") == EXPECTED_FORMATS


def test_load_expert_formats_missing_file(load, tmp_path):
    assert load().load_expert_formats(tmp_path) is None


def _groups(*groups):
    return {
        "model": {
            "parameter_groups": [{"compute_dtype": c, "tensors": t} for c, t in groups]
        }
    }


def _expert(layer, which):
    return f"model.llm.layers.{layer}.mlp.experts.{which}_weight"


@pytest.mark.parametrize(
    "raw, match",
    [
        (_groups(("int8", [])), "Unsupported Inkling 2 compute dtype"),
        (
            _groups(("bfloat16", [_expert(0, "w13")])),
            "expert weight with compute dtype",
        ),
        (
            _groups(("nvfp4", [_expert(0, "w13")]), ("mxfp8", [_expert(0, "w2")])),
            "one format",
        ),
        (_groups(("nvfp4", [_expert(0, "w13")])), "one format"),
        (
            _groups(
                ("nvfp4", [_expert(0, "w13"), _expert(0, "w2")]),
                ("mxfp8", [_expert(0, "w2")]),
            ),
            "both nvfp4 and mxfp8",
        ),
    ],
)
def test_expert_formats_rejects(load, raw, match):
    with pytest.raises(ValueError, match=match):
        load().expert_formats(raw)


def test_expert_formats_ignores_other_tensors(load):
    raw = _groups(
        ("float32", ["model.llm.layers.0.mlp.gate.bias"]),
        (
            "mxfp8",
            [
                _expert(3, "w13"),
                _expert(3, "w2"),
                "model.llm.layers.3.mlp.experts.w2_weight_scale",
            ],
        ),
    )
    assert load().expert_formats(raw) == {3: "mxfp8"}
