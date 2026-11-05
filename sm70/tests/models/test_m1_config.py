# SPDX-License-Identifier: Apache-2.0
"""M1 sm70-config: Sm70Config defaults, parsing, kill switches, hash.

CPU only. Pure pydantic: vllm is never imported, except by the VLLM_SM70_QUANT
path check (M2 registration is skipped), which is skipped when vLLM's Python
source is not importable.

The kill switches use K01's ``vllm_sm70.envs.kill_switch``: only "0" acts,
and the environment can never turn a feature on.
"""

import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vllm_sm70 import envs
from vllm_sm70.config import (
    GRAPH_FIELDS,
    KILL_SWITCHES,
    Sm70Config,
    clear_cache,
    compute_hash,
    dflash2_verifier_enabled,
    fp8_qpn8_enabled,
    fp8_turbomind_enabled,
    require_fp8_turbomind,
    sm70_config,
)

PKG = pathlib.Path(__file__).resolve().parents[2] / "vllm_sm70"

EFFECTIVE_DEFAULTS = {
    "quant_backend": "auto",
    "nvfp4_gemv": True,
    "flint_moe_qpn_decode": True,
    "fp8_moe_decode": True,
    "flint_qk_rope_kv": True,
    "fp8_qpn8": None,
    "dflash2_verifier": "auto",
    "shared_moe_overlap": False,
}

# Our sections of K01's registry.
OURS = {**envs._SECTIONS["quant"], **envs._SECTIONS["models"]}

# Switches that are registered but whose path has not landed yet. Each must
# name the patch that wires it, and the no-dead-switch test fails as soon
# as the name is consumed, so that this entry gets removed.
PENDING: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in OURS:
        monkeypatch.delenv(name, raising=False)
    clear_cache()
    yield
    clear_cache()


def _vc(sm70=None):
    additional = None if sm70 is None else {"sm70": sm70}
    return SimpleNamespace(additional_config=additional, kernel_config=None)


# --- defaults and parsing ------------------------------------------------


def test_defaults_are_frozen_effective_values():
    assert Sm70Config().model_dump() == EFFECTIVE_DEFAULTS
    assert sm70_config(None).model_dump() == EFFECTIVE_DEFAULTS


def test_none_additional_config_returns_defaults():
    assert sm70_config(SimpleNamespace(additional_config=None)) == Sm70Config()
    vc2 = SimpleNamespace(additional_config={"other": 1})
    assert sm70_config(vc2) == Sm70Config()


def test_additional_config_parse_and_cache():
    vc = _vc({"quant_backend": "turbomind", "flint_qk_rope_kv": False, "fp8_qpn8": True})
    cfg = sm70_config(vc)
    assert cfg.quant_backend == "turbomind"
    assert cfg.flint_qk_rope_kv is False
    assert cfg.fp8_qpn8 is True
    assert sm70_config(vc) is cfg  # cached per vllm_config


def test_kernel_config_fallback():
    vc = SimpleNamespace(
        additional_config=None,
        kernel_config=SimpleNamespace(sm70={"nvfp4_gemv": False}),
    )
    assert sm70_config(vc).nvfp4_gemv is False


def test_unknown_key_rejected():
    with pytest.raises(Exception, match="extra|Extra"):
        sm70_config(_vc({"skinny_moe": False}))
    for bad in ("cutlass", "marlin"):  # Route-M dropped for v1
        with pytest.raises(Exception):
            sm70_config(_vc({"quant_backend": bad}))
    with pytest.raises(Exception, match="extra|Extra"):
        sm70_config(_vc({"flint_fusions": False}))  # old name


def test_frozen():
    with pytest.raises(Exception):
        Sm70Config().nvfp4_gemv = False  # type: ignore[misc]


def test_config_imports_no_vllm():
    code = (
        "import sys, vllm_sm70.config; "
        "bad = [m for m in sys.modules if m == 'vllm' or m.startswith('vllm.')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- kill switches -------------------------------------------------------


def test_registered_names_are_exactly_ours():
    assert set(envs._SECTIONS["quant"]) == {
        "VLLM_SM70_QUANT",
        "VLLM_SM70_FP8_TURBOMIND",
        "VLLM_SM70_MODELOPT_NVFP4_GEMV",
        "VLLM_SM70_FP8_MOE_DECODE",
        "VLLM_SM70_FP8_QPN8",
    }
    assert set(envs._SECTIONS["models"]) == {
        "VLLM_SM70_FLINT_MOE_QPN_DECODE",
        "VLLM_SM70_FLINT_QK_ROPE_KV",
        "VLLM_SM70_DFLASH2_VERIFIER",
        "VLLM_SM70_DFLASH2_EMULATION",
        "VLLM_SM70_SHARED_MOE_OVERLAP",
    }
    # one kill per field, except quant_backend (env-only refusal)
    assert {f for f, _ in KILL_SWITCHES.values()} == set(Sm70Config.model_fields) - {
        "quant_backend"
    }
    assert set(KILL_SWITCHES) <= set(OURS)
    for name in (  # M10 and Kai own these; they must not be ours
        "VLLM_SM70_FLINT_FUSED_ROUTER",
        "VLLM_SM70_FLINT_SAMPLER",
        "VLLM_SM70_FLINT_ATTN_FUSED_DECODE",
        "VLLM_SM70_FLINT_AR_RMSNORM",
    ):
        assert name not in OURS


def _field_off(name):
    field, killed = KILL_SWITCHES[name]
    cfg = sm70_config(_vc({}))
    assert getattr(cfg, field) == killed
    assert {k: v for k, v in cfg.model_dump().items() if k != field} == {
        k: v for k, v in EFFECTIVE_DEFAULTS.items() if k != field
    }


def _fp8_refuses(name):
    assert not fp8_turbomind_enabled()
    with pytest.raises(RuntimeError, match="VLLM_SM70_FP8_TURBOMIND=0"):
        require_fp8_turbomind()
    assert sm70_config(_vc({})).quant_backend == "auto"  # no Marlin fallback


def _quant_registration_skipped(name):
    pytest.importorskip("vllm.envs")
    import importlib

    import vllm.model_executor.layers.quantization as me_quant
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )

    saved = dict(me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
    try:
        for slot in ("fp8", "modelopt_fp4"):
            me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop(slot, None)
        import vllm_sm70.quant.configs as configs

        configs = importlib.reload(configs)
        assert not configs.REGISTERED
        assert me_quant.get_quantization_config("fp8") is Fp8Config
        assert me_quant.get_quantization_config("modelopt_fp4") is ModelOptNvFp4Config
    finally:
        me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.clear()
        me_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.update(saved)


def _dflash2_emulation_off(name):
    pytest.importorskip("vllm.envs")
    from unittest import mock

    import torch
    from vllm.model_executor.models import ModelRegistry

    import vllm_sm70.platform as sm70_platform
    from vllm_sm70.models.qwen import dflash_sm70

    bf16_draft = SimpleNamespace(dtype=torch.bfloat16)
    assert not dflash_sm70.emulation_enabled(bf16_draft, torch.float16, capability=70)
    before = ModelRegistry.models[dflash_sm70.DRAFT_ARCH]
    with mock.patch.object(sm70_platform, "is_sm70_ready", lambda: True):
        assert dflash_sm70.register() is False
    assert ModelRegistry.models[dflash_sm70.DRAFT_ARCH] is before


def _getter_only(name):
    # TODO: replace with a path test in the patch named in PENDING.
    assert getattr(envs, name) is False


def _path_off(name):
    if name in KILL_SWITCHES:
        _field_off(name)
        if name == "VLLM_SM70_MODELOPT_NVFP4_GEMV":
            # TODO(M3): also assert the NVFP4 W4A16 GEMV is not selected.
            pass
    elif name == "VLLM_SM70_FP8_TURBOMIND":
        _fp8_refuses(name)
    elif name == "VLLM_SM70_QUANT":
        _quant_registration_skipped(name)
    elif name == "VLLM_SM70_DFLASH2_EMULATION":
        _dflash2_emulation_off(name)
    elif name in PENDING:
        _getter_only(name)
    else:
        raise AssertionError(f"{name}: no path check; add one")


@pytest.mark.parametrize("name", sorted(OURS))
def test_every_kill_switch_turns_its_path_off(monkeypatch, name):
    assert getattr(envs, name) is True
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    _path_off(name)


@pytest.mark.parametrize("value", [None, "", "1", "true", "on", "yes", "off", " 0"])
@pytest.mark.parametrize("name", sorted(OURS))
def test_non_zero_values_keep_enabled(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert getattr(envs, name) is True
    assert sm70_config(_vc({})).model_dump() == EFFECTIVE_DEFAULTS
    if name == "VLLM_SM70_FP8_TURBOMIND":
        require_fp8_turbomind()


@pytest.mark.parametrize("value", ["1", "true", "on"])
def test_env_cannot_enable(monkeypatch, value):
    for name in OURS:
        monkeypatch.setenv(name, value)
    off = {
        "nvfp4_gemv": False,
        "flint_moe_qpn_decode": False,
        "fp8_moe_decode": False,
        "flint_qk_rope_kv": False,
        "fp8_qpn8": False,
        "dflash2_verifier": "off",
        "shared_moe_overlap": False,
        "quant_backend": "turbomind",
    }
    assert sm70_config(_vc(off)).model_dump() == off


@pytest.mark.parametrize("name", sorted(KILL_SWITCHES))
def test_kill_switch_overrides_explicit_enable(monkeypatch, name):
    field, killed = KILL_SWITCHES[name]
    enabled = {"dflash2_verifier": "on", "fp8_qpn8": True}.get(field, True)
    monkeypatch.setenv(name, "0")
    assert getattr(sm70_config(_vc({field: enabled})), field) == killed


def test_no_dead_switch():
    """Every name in our sections is read by plugin code, or is PENDING.

    "Read" means ``envs.<NAME>`` or the name as a string literal (e.g. a
    KILL_SWITCHES key); mentions in comments and docstrings don't count.
    """
    sources = {
        p: p.read_text()
        for p in PKG.rglob("*.py")
        if p.name != "envs.py" or p.parent != PKG
    }
    consumed = {
        name
        for name in OURS
        if any(
            re.search(rf"(envs\.{name}\b|[\"']{name}[\"'])", text)
            for text in sources.values()
        )
    }
    dead = set(OURS) - consumed - set(PENDING)
    assert not dead, f"registered but never read: {sorted(dead)}"
    stale = set(PENDING) & consumed
    assert not stale, f"now consumed; drop from PENDING: {sorted(stale)}"
    assert set(PENDING) <= set(OURS)


# --- hash and auto rules -------------------------------------------------


def test_hash_changes_for_every_graph_field():
    base = compute_hash(Sm70Config())
    flips = {
        "quant_backend": "turbomind",
        "flint_qk_rope_kv": False,
        "fp8_qpn8": True,
        "dflash2_verifier": "off",
        "flint_moe_qpn_decode": False,
        "fp8_moe_decode": False,
        "nvfp4_gemv": False,
        "shared_moe_overlap": True,  # the fork hashes it too
    }
    assert set(flips) == set(GRAPH_FIELDS) == set(Sm70Config.model_fields)
    hashes = {compute_hash(Sm70Config(**{k: v})) for k, v in flips.items()}
    assert base not in hashes and len(hashes) == len(flips)


def test_compute_hash_requires_an_argument():
    with pytest.raises(TypeError):
        compute_hash()  # type: ignore[call-arg]
    assert compute_hash(None) == compute_hash(Sm70Config())  # effective defaults


# A non-default config value for each field that has a kill switch, so the
# kill changes the effective value even where the default is already "off".
_ENABLED = {
    "nvfp4_gemv": True,
    "flint_moe_qpn_decode": True,
    "fp8_moe_decode": True,
    "flint_qk_rope_kv": True,
    "fp8_qpn8": True,
    "dflash2_verifier": "on",
    "shared_moe_overlap": True,
}


@pytest.mark.parametrize("name", sorted(KILL_SWITCHES))
def test_graph_field_kill_switch_changes_hash(monkeypatch, name):
    field, _ = KILL_SWITCHES[name]
    assert field in GRAPH_FIELDS
    vc = _vc({field: _ENABLED[field]})
    base = compute_hash(vc)
    clear_cache()
    monkeypatch.setenv(name, "0")
    vc2 = _vc({field: _ENABLED[field]})
    assert compute_hash(vc2) != base
    assert compute_hash(sm70_config(vc2)) == compute_hash(vc2)


def test_hash_of_defaults_sees_kill_switch(monkeypatch):
    base = compute_hash(None)
    monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", "0")
    assert compute_hash(None) != base


def test_auto_resolution_rules():
    cfg = Sm70Config()
    assert dflash2_verifier_enabled(cfg, verifier_qualified=True)  # Qwen3.8
    assert not dflash2_verifier_enabled(cfg, verifier_qualified=False)  # Qwen3.6
    assert fp8_qpn8_enabled(cfg, verifier_qualified=True)
    assert not fp8_qpn8_enabled(cfg, verifier_qualified=False)
    assert not fp8_qpn8_enabled(Sm70Config(fp8_qpn8=False), verifier_qualified=True)
    assert not dflash2_verifier_enabled(
        Sm70Config(dflash2_verifier="off"), verifier_qualified=True
    )


def test_fp8_moe_stage_constants_match_fork_resolution():
    """FORK cf9492a Sm70MoEFormatConfig.resolve (vllm/config/sm70_moe.py
    :319-452) with the unit's env (nothing set)."""
    from vllm_sm70.config import CONSTANTS

    assert CONSTANTS["MOE_SINGLE_TOKEN_REDUCE"] == "weighted"  # :393-397
    assert CONSTANTS["MOE_SINGLE_TOKEN_W2"] == "indexed"  # :381-391
    assert CONSTANTS["MOE_SINGLE_TOKEN_W13"] == ("dense",)  # :371-379
    assert CONSTANTS["FP8_MOE_LEGACY_COMPACT"] is True
    assert CONSTANTS["FP8_MOE_COMPACT_EXACT_LAYOUT"] is True  # :344-349
    assert CONSTANTS["FP8_MOE_COMPACT_NATIVE_UNPERMUTE"] is False
    assert CONSTANTS["FP8_MOE_COMPACT_DECOMPOSED"] is False
    assert CONSTANTS["FP8_MOE_BATCHED_GEMM"] is True
    assert CONSTANTS["FP8_MOE_W13_PER_EXPERT"] is False
    assert CONSTANTS["FP8_MOE_W2_PER_EXPERT"] is False
    assert CONSTANTS["FP8_MOE_PERMUTE_WITH_SCRATCH"] is True
