# SPDX-License-Identifier: Apache-2.0
"""Tests for M12 inkling2-sm70-model registration (vllm_sm70/models/inkling2_sm70/).

CPU only: registry wiring at faked cc 7.0 vs 8.0, the config classes in
vLLM's _CONFIG_REGISTRY on both, the vendored configs.py against the sha256
in PROVENANCE.md, and the Inkling 2 Mini config.json (data/inkling2_mini/)
parsed through the registered classes.
"""

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch

import vllm.platforms
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.models import flint_sm70, inkling2_sm70

PKG = pathlib.Path(inkling2_sm70.__file__).resolve().parent
DATA = pathlib.Path(__file__).resolve().parent / "data" / "inkling2_mini"
CONFIG_TYPES = ("inkling2", "inkling2_loop", "inkling2_model")


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


def _raw(mapping) -> dict:
    """Entries as stored (LazyConfigDict keeps class names as strings)."""
    return {k: dict.__getitem__(mapping, k) for k in mapping}


@pytest.fixture
def registry(monkeypatch):
    """Both registries without the Inkling 2 entries; restored afterwards."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    saved_model = ModelRegistry.models.pop(inkling2_sm70.ARCH, None)
    saved_cfg = {
        k: dict.pop(_CONFIG_REGISTRY, k) for k in CONFIG_TYPES if k in _CONFIG_REGISTRY
    }
    monkeypatch.setattr(inkling2_sm70, "_registered", False)
    monkeypatch.delitem(MODELS_CONFIG_MAP, inkling2_sm70.ARCH, raising=False)
    yield ModelRegistry, _CONFIG_REGISTRY
    ModelRegistry.models.pop(inkling2_sm70.ARCH, None)
    MODELS_CONFIG_MAP.pop(inkling2_sm70.ARCH, None)
    if saved_model is not None:
        ModelRegistry.models[inkling2_sm70.ARCH] = saved_model
    for k in CONFIG_TYPES:
        dict.pop(_CONFIG_REGISTRY, k, None)
    _CONFIG_REGISTRY.update(saved_cfg)


def _assert_configs_registered(cfg_reg):
    from vllm_sm70.models.inkling2_sm70 import configs, sm70_configs

    assert cfg_reg["inkling2"] is sm70_configs.Inkling2Sm70MMConfig
    assert cfg_reg["inkling2_loop"] is sm70_configs.Inkling2LoopSm70MMConfig
    assert cfg_reg["inkling2_model"] is configs.Inkling2ModelConfig


def test_register_at_cc70(device, registry):
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    reg, cfg_reg = registry
    device(7, 0)
    models, cfg_map, cfgs = dict(reg.models), dict(MODELS_CONFIG_MAP), _raw(cfg_reg)
    assert inkling2_sm70.register() is True
    entry = reg.models[inkling2_sm70.ARCH]
    assert (entry.module_name, entry.class_name) == (
        "vllm_sm70.models.inkling2_sm70",
        "Inkling2LoopSm70ForConditionalGeneration",
    )
    _assert_configs_registered(cfg_reg)
    assert inkling2_sm70.register() is True  # idempotent
    # Nothing else moved: Flint, Qwen and every other entry are the same objects.
    assert {k: v for k, v in reg.models.items() if k != inkling2_sm70.ARCH} == models
    hook = MODELS_CONFIG_MAP[inkling2_sm70.ARCH]
    assert hook is inkling2_sm70.Inkling2Sm70ModelConfig
    assert dict(MODELS_CONFIG_MAP) == {**cfg_map, inkling2_sm70.ARCH: hook}
    assert {k: v for k, v in _raw(cfg_reg).items() if k not in CONFIG_TYPES} == cfgs
    assert reg.models["FlintForCausalLM"].module_name == "vllm.models.flint"


def test_not_registered_at_cc80_but_configs_are(device, registry):
    reg, cfg_reg = registry
    device(8, 0)
    models, cfgs = dict(reg.models), _raw(cfg_reg)
    assert inkling2_sm70.register() is False
    assert inkling2_sm70.ARCH not in reg.models
    assert inkling2_sm70._registered is False
    assert dict(reg.models) == models
    _assert_configs_registered(cfg_reg)
    assert {k: v for k, v in _raw(cfg_reg).items() if k not in CONFIG_TYPES} == cfgs


def test_register_raises_without_ext_then_retries(device, registry):
    reg, cfg_reg = registry
    device(7, 0, ext=False)
    with pytest.raises(RuntimeError, match="_sm70_C is not loaded"):
        inkling2_sm70.register()
    assert inkling2_sm70._registered is False
    assert inkling2_sm70.ARCH not in reg.models
    _assert_configs_registered(cfg_reg)  # before the gate
    device(7, 0, ext=True)
    assert inkling2_sm70.register() is True
    assert inkling2_sm70.ARCH in reg.models


def test_register_raises_if_registry_ignores_it(device, registry, monkeypatch):
    reg, _ = registry
    device(7, 0)
    monkeypatch.setattr(reg, "register_model", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="upstream registry semantics changed"):
        inkling2_sm70.register()
    assert inkling2_sm70._registered is False


def test_register_configs_keeps_existing_keys(registry, caplog):
    _, cfg_reg = registry

    class Theirs:
        pass

    cfg_reg["inkling2"] = Theirs
    with caplog.at_level("WARNING", logger="vllm.sm70"):
        inkling2_sm70.register_configs()
    assert dict.__getitem__(cfg_reg, "inkling2") is Theirs
    assert "already registered" in caplog.text
    first = _raw(cfg_reg)
    caplog.clear()
    del cfg_reg["inkling2"]
    inkling2_sm70.register_configs()
    inkling2_sm70.register_configs()  # idempotent, silent
    _assert_configs_registered(cfg_reg)
    assert not caplog.text
    assert {k: v for k, v in _raw(cfg_reg).items() if k != "inkling2"} == {
        k: v for k, v in first.items() if k != "inkling2"
    }


def test_models_area_calls_inkling2_beside_flint(monkeypatch):
    from vllm_sm70 import models
    from vllm_sm70.models import qwen

    calls = []
    for name, mod in (("flint", flint_sm70), ("inkling2", inkling2_sm70), ("qwen", qwen)):
        monkeypatch.setattr(mod, "register", lambda n=name: calls.append(n))
    monkeypatch.setattr(models, "_registered", False)
    models.register()
    assert calls == ["flint", "inkling2", "qwen"]


def test_stub_model_refuses_construction():
    cls = inkling2_sm70.Inkling2LoopSm70ForConditionalGeneration
    assert cls.__name__ == inkling2_sm70.ARCH
    assert inkling2_sm70.LAZY_TARGET == f"{cls.__module__.rpartition('.')[0]}:{cls.__name__}"
    with pytest.raises(NotImplementedError, match="no model body"):
        cls(vllm_config=None, prefix="")
    with pytest.raises(AttributeError):
        inkling2_sm70.NoSuchName  # noqa: B018


def test_vendored_configs_match_recorded_sha256():
    text = (PKG / "PROVENANCE.md").read_text(encoding="utf-8")
    recorded = re.findall(r"\b[0-9a-f]{64}\b", text)
    assert len(recorded) == 1, recorded
    assert "c32251b5d192f0f72f9ca8c5595d1d43f1dd37b8" in text
    got = hashlib.sha256((PKG / "configs.py").read_bytes()).hexdigest()
    assert got == recorded[0]


def test_mini_config_parses(registry):
    from vllm.config.model import _get_head_dtype
    from vllm.transformers_utils.config import get_config
    from vllm_sm70.models.inkling2_sm70 import sm70_configs

    inkling2_sm70.register_configs()
    cfg = get_config(str(DATA), trust_remote_code=False)
    assert type(cfg) is sm70_configs.Inkling2LoopSm70MMConfig
    text = cfg.get_text_config()
    assert text.num_hidden_layers == 24 and text.num_loop_passes == 3
    assert text.num_layer_visits == 48
    assert list(text.loop_core_layers) == list(range(6, 18))
    global_layers = sorted(set(range(text.num_hidden_layers)) - set(text.local_layer_ids))
    assert global_layers == [5, 11, 17, 23]
    assert [text.layer_visit_ids(i) for i in (5, 6, 17, 18, 23)] == [
        [5], [6, 18, 30], [17, 29, 41], [42], [47]
    ]
    assert (text.vocab_size, text.padded_vocab_size) == (200058, 201024)
    # The head follows the model dtype, whatever --dtype resolves to.
    assert cfg.head_dtype == sm70_configs.HEAD_DTYPE == "model"
    for dtype in (torch.float16, torch.bfloat16):
        assert _get_head_dtype(cfg, dtype, "generate") is dtype
    # The serve-time architecture override applies on top of the checkpoint's.
    over = get_config(
        str(DATA),
        trust_remote_code=False,
        hf_overrides_kw={"architectures": [inkling2_sm70.ARCH]},
    )
    assert over.architectures == [inkling2_sm70.ARCH]
    assert over.head_dtype == "model"


def test_explicit_head_dtype_wins(registry):
    from vllm_sm70.models.inkling2_sm70 import sm70_configs

    raw = json.loads((DATA / "config.json").read_text(encoding="utf-8"))
    cfg = sm70_configs.Inkling2LoopSm70MMConfig(**raw, head_dtype="float32")
    assert cfg.head_dtype == "float32"
    plain = {"text_config": {"num_hidden_layers": 2, "local_layer_ids": [0]}}
    assert sm70_configs.Inkling2Sm70MMConfig(**plain).head_dtype == "model"


def test_metadata_fixtures_are_the_checkpoint_layout():
    quant = json.loads((DATA / "hf_quant_config.json").read_text(encoding="utf-8"))
    index = json.loads((DATA / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert isinstance(quant, dict) and quant
    assert len(index["weight_map"]) > 1000


def test_registration_imports_no_model_triton_or_cuda():
    """In a fresh process without GPUs. vLLM's own config and renderer
    registry modules (which the registration needs) are imported first: they
    already pull in torch and, through it, Triton, so the check is on what
    registration adds."""
    code = (
        "import sys, torch, vllm.transformers_utils.config as c\n"
        "import vllm.renderers.registry, vllm.tokenizers.registry\n"
        "before = set(sys.modules)\n"
        "from vllm_sm70.models import inkling2_sm70\n"
        "assert inkling2_sm70.register() is False\n"
        "added = set(sys.modules) - before\n"
        "heavy = ('triton', 'vllm.model_executor', 'vllm.models', 'vllm.v1',\n"
        "         'vllm_sm70.ops', 'vllm_sm70.attention', 'vllm_sm70._sm70_C')\n"
        "bad = sorted(m for m in added if m.startswith(heavy))\n"
        "assert not bad, bad\n"
        "assert 'vllm_sm70.models.inkling2_sm70.model' not in sys.modules\n"
        "assert 'vllm_sm70.models.inkling2_sm70.renderer' not in sys.modules\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert type(c._CONFIG_REGISTRY['inkling2_loop']).__name__ == 'type'\n"
        "import vllm_sm70\n"
        "vllm_sm70.register()\n"
        "assert vllm_sm70.registered_areas().get('models') == 'ok', (\n"
        "    vllm_sm70.registered_areas())\n"
        "assert 'vllm_sm70.models.inkling2_sm70.model' not in sys.modules\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600, env=env)
