# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 DSpark draft routing (inkling2_sm70/draft.py), on CPU.

The staged draft directory from the Inkling 2 Mini config.json
(data/inkling2_mini/) and a placeholder mtp.safetensors, parsed back through
vLLM's config loader; the engine check the config hook runs; the speculator
class the U6 platform hook returns; the kill switch.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_sm70.models import inkling2_sm70
from vllm_sm70.models.inkling2_sm70 import draft

DATA = pathlib.Path(__file__).resolve().parent / "data" / "inkling2_mini"
SWITCH = "VLLM_SM70_INKLING_DSPARK"


@pytest.fixture(autouse=True)
def switch_unset(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)


@pytest.fixture
def target(tmp_path) -> pathlib.Path:
    directory = tmp_path / "target"
    directory.mkdir()
    shutil.copy(DATA / "config.json", directory / "config.json")
    (directory / draft.DRAFT_WEIGHTS).write_bytes(b"weights")
    return directory


@pytest.fixture
def draft_hf_config():
    return draft.draft_config(DATA)


def _target_hf_config():
    from vllm_sm70.models.inkling2_sm70.sm70_configs import Inkling2LoopSm70MMConfig

    return Inkling2LoopSm70MMConfig(**json.loads((DATA / "config.json").read_text()))


def _spec(hf_config, method="dspark", k=7):
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        draft_model_config=SimpleNamespace(hf_config=hf_config),
    )


# --- the staged directory ------------------------------------------------------


def test_draft_config_from_the_mini_checkpoint(draft_hf_config):
    c = draft_hf_config
    assert c.model_type == draft.DRAFT_MODEL_TYPE
    assert c.architectures == [draft.DRAFT_ARCH]
    assert c.target_layer_ids == c.eagle_aux_hidden_state_layer_ids == [5, 23, 41, 48]
    assert (c.draft_block_size, c.dspark_block_size) == (8, 7)
    assert c.sample_from_anchor is False and c.is_causal is False
    assert (c.hidden_size, c.num_hidden_layers, c.vocab_size) == (2304, 3, 200058)


def test_stage_writes_config_and_links_weights(target, tmp_path):
    from vllm.transformers_utils.config import get_config

    inkling2_sm70.register_configs()
    out = draft.stage_draft_dir(target, tmp_path / "draft")
    link = out / draft.DRAFT_WEIGHTS
    assert (
        link.is_symlink() and link.resolve() == (target / draft.DRAFT_WEIGHTS).resolve()
    )
    assert sorted(p.name for p in out.iterdir()) == ["config.json", draft.DRAFT_WEIGHTS]
    # vLLM's loader parses it to the same config as from_target.
    loaded = get_config(str(out), trust_remote_code=False)
    assert type(loaded).__name__ == "Inkling2DSparkConfig"
    want = draft.draft_config(target).to_dict()
    got = type(loaded).from_pretrained(out).to_dict()
    for d in (want, got):
        d.pop("_name_or_path")
    assert got == want
    assert loaded.architectures == [draft.DRAFT_ARCH]
    assert loaded.eagle_aux_hidden_state_layer_ids == [5, 23, 41, 48]


def test_stage_is_idempotent_and_relinks(target, tmp_path):
    out = draft.stage_draft_dir(target, tmp_path / "draft")
    config = out / "config.json"
    os.utime(config, (1, 1))
    draft.stage_draft_dir(target, out)
    assert config.stat().st_mtime == 1  # unchanged, not rewritten
    link = out / draft.DRAFT_WEIGHTS
    link.unlink()
    link.symlink_to(tmp_path)
    draft.stage_draft_dir(target, out)
    assert link.resolve() == (target / draft.DRAFT_WEIGHTS).resolve()


def test_stage_needs_the_weights_and_an_mtp_config(target, tmp_path):
    (target / draft.DRAFT_WEIGHTS).unlink()
    with pytest.raises(FileNotFoundError, match="no mtp.safetensors"):
        draft.stage_draft_dir(target, tmp_path / "draft")
    (target / draft.DRAFT_WEIGHTS).write_bytes(b"weights")
    raw = json.loads((target / "config.json").read_text())
    del raw["mtp_config"]
    (target / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no mtp_config"):
        draft.stage_draft_dir(target, tmp_path / "draft")


def test_cli(target, tmp_path, capsys):
    assert draft.main([]) == 2
    assert draft.main([str(target), str(tmp_path / "d")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "d")


# --- the engine check and the speculator class --------------------------------


def test_rejection_accepts_the_staged_draft(draft_hf_config):
    assert draft.speculative_rejection(None, None) is None
    target = _target_hf_config()
    assert draft.speculative_rejection(_spec(draft_hf_config), target) is None


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"method": "eagle3"}, "only with method dspark"),
        ({"method": "mtp"}, "only with method dspark"),
        ({"k": 3}, "drafts 7 tokens per step"),
    ],
)
def test_rejection_refuses(draft_hf_config, kw, match):
    reason = draft.speculative_rejection(_spec(draft_hf_config, **kw), None)
    assert reason is not None and match in reason


def test_rejection_refuses_other_drafts_and_visits(draft_hf_config, monkeypatch):
    other = SimpleNamespace(model_type="qwen3")
    assert "only with method dspark" in draft.speculative_rejection(_spec(other), None)
    target = _target_hf_config()
    monkeypatch.setattr(draft_hf_config, "target_layer_ids", [5, 23, 41, 47])
    assert "restage it" in draft.speculative_rejection(_spec(draft_hf_config), target)


def test_kill_switch(draft_hf_config, monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    spec = _spec(draft_hf_config)
    assert f"{SWITCH}=0" in draft.speculative_rejection(spec, None)
    assert draft.speculator_cls(SimpleNamespace(speculative_config=spec)) is None
    assert draft.speculative_rejection(None, None) is None


def test_speculator_cls(draft_hf_config, monkeypatch):
    from vllm.utils import import_utils

    resolved = []
    monkeypatch.setattr(
        import_utils,
        "resolve_obj_by_qualname",
        lambda name: resolved.append(name) or "cls",
    )
    for spec in (
        None,
        _spec(draft_hf_config, method="eagle3"),
        _spec(SimpleNamespace(model_type="qwen3")),
    ):
        assert draft.speculator_cls(SimpleNamespace(speculative_config=spec)) is None
    assert resolved == []
    cfg = SimpleNamespace(speculative_config=_spec(draft_hf_config))
    assert draft.speculator_cls(cfg) == "cls"
    assert resolved == [draft.SPECULATOR_TARGET]
