# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 DSpark feature capture, on CPU.

The tiny checkpoint (fixtures/inkling2_tiny) gets the Inkling 2 Mini
``mtp_config`` at its own hidden size; the draft config comes from
``Inkling2DSparkConfig.from_target`` and the aux layers from upstream's V2
helpers (``get_eagle3_aux_layers_from_config``,
``set_eagle3_aux_hidden_state_layers``). The captured tensors are compared
with the dense oracle (inkling2_ref.py) in each engine scenario and in
verify-shaped steps of 8 tokens per request: visits 5, 23 and 41 against the
oracle's input of that visit (the output of the visit before it in the same
block, i.e. upstream's AttnRes partial ``stack[num_blocks]``), id 48 against
the final-norm output.
"""

from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx
from inkling2_ref import Inkling2Oracle
from test_m12_inkling2_model_cpu import Engine, _have_vllm

pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

TINY = HERE / "fixtures" / "inkling2_tiny"
MINI = HERE / "data" / "inkling2_mini" / "config.json"
LEN_A, LEN_B, PREFIX = 23, 19, 8
TOL = 1e-4
TARGET_IDS = (5, 23, 41, 48)
VERIFY_ROWS = 8


def _target_config():
    """The tiny config with the Mini mtp_config at the tiny hidden size."""
    from vllm_sm70.models.inkling2_sm70.sm70_configs import Inkling2LoopSm70MMConfig

    raw = json.loads((TINY / "config.json").read_text())
    mtp = dict(json.loads(MINI.read_text())["mtp_config"])
    mtp["hidden_size"] = raw["text_config"]["hidden_size"]
    raw["mtp_config"] = mtp
    return Inkling2LoopSm70MMConfig(**raw)


@pytest.fixture(scope="module")
def draft_hf_config():
    from vllm_sm70.models.inkling2_sm70.configs import Inkling2DSparkConfig

    return Inkling2DSparkConfig.from_target(_target_config())


@pytest.fixture(scope="module")
def spec(draft_hf_config):
    return SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
    )


@pytest.fixture(scope="module")
def engine(spec) -> Engine:
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        requires_aux_hidden_states,
        set_eagle3_aux_hidden_state_layers,
    )

    engine = Engine()
    assert requires_aux_hidden_states(spec)
    set_eagle3_aux_hidden_state_layers(engine.model, spec)
    return engine


@pytest.fixture(scope="module")
def oracle() -> Inkling2Oracle:
    return Inkling2Oracle.from_checkpoint(TINY)


@pytest.fixture(scope="module")
def sequences() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(12)
    vocab = json.loads((TINY / "config.json").read_text())["text_config"][
        "unpadded_vocab_size"
    ]
    a = torch.randint(0, vocab, (LEN_A,), generator=generator)
    b = torch.randint(0, vocab, (LEN_B,), generator=generator)
    b[:PREFIX] = a[:PREFIX]
    return {"a": a, "b": b}


@pytest.fixture(scope="module")
def expected(oracle, sequences) -> dict[str, torch.Tensor]:
    """Per request [T, 4 * H]: the oracle's states at the target ids."""
    out = {}
    for req_id, ids in sequences.items():
        ref = oracle.forward(ids[None])
        states = [ref.visit_states[visit - 1, 0] for visit in TARGET_IDS[:-1]]
        out[req_id] = torch.cat([*states, ref.final_hidden_state[0]], dim=1)
    return out


def _aux_forward(engine: Engine, sequences):
    model = engine.model

    def forward(rank, batch):
        with torch.no_grad():
            final, aux = model(batch.take(sequences), batch.positions)
        assert len(aux) == len(TARGET_IDS) and aux[-1] is final
        return torch.cat(aux, dim=1).float()

    return forward


def _assert_aux(got, want, hidden, where):
    assert got.shape == want.shape, where
    diff = (got - want).abs().unflatten(1, (len(TARGET_IDS), hidden)).amax((0, 2))
    for target_id, err in zip(TARGET_IDS, diff.tolist()):
        assert err <= TOL, f"{where}: id {target_id} is off by {err:.3e}"


def test_draft_config_reads_the_mini_target_ids(draft_hf_config, oracle):
    assert tuple(draft_hf_config.target_layer_ids) == TARGET_IDS
    assert tuple(draft_hf_config.eagle_aux_hidden_state_layer_ids) == TARGET_IDS
    assert draft_hf_config.dspark_block_size == VERIFY_ROWS - 1
    num_visits = len(oracle.visit_layers)
    assert TARGET_IDS[-1] == num_visits == _target_config().text_config.num_layer_visits
    # Each captured visit continues the block of the visit before it, so its
    # input is that visit's output (the partial block), with no pass restart
    # or block boundary in between.
    for visit in TARGET_IDS[:-1]:
        layer = oracle.visit_layers[visit]
        assert layer % oracle.block != 0
        assert oracle.visit_layers[visit - 1] == layer - 1


def test_upstream_helpers_set_the_target_ids(engine, spec):
    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    assert supports_eagle3(engine.model)
    assert tuple(get_eagle3_aux_layers_from_config(spec)) == TARGET_IDS
    assert engine.model.model.aux_hidden_state_layers == TARGET_IDS


def test_default_aux_layers_are_the_checkpoint_ids(engine, monkeypatch):
    model = engine.model
    monkeypatch.setattr(
        model.config, "mtp_config", {"target_layer_ids": [5, 23]}, raising=False
    )
    assert model.get_eagle3_default_aux_hidden_state_layers() == (5, 23)
    monkeypatch.setattr(model.config, "mtp_config", None)
    with pytest.raises(ValueError, match="no mtp_config.target_layer_ids"):
        model.get_eagle3_default_aux_hidden_state_layers()


@pytest.mark.parametrize(
    "name", ("prefill", "decode", "chunk1", "chunk7", "resume", "interleaved")
)
def test_aux_capture_matches_oracle(engine, sequences, expected, name):
    plans = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX, chunks=(1, 7))
    engine.reset()
    engine.sequences = sequences
    got = engine.harness.run(plans[name], _aux_forward(engine, sequences))
    assert got, name
    for req_id, rows in got.items():
        first = PREFIX if (name, req_id) == ("resume", "b") else 0
        want = expected[req_id][first : first + rows.shape[0]]
        _assert_aux(rows, want, engine.hidden, f"{name}/{req_id}")


@pytest.mark.parametrize("metadata", [None, "full"])
def test_aux_capture_in_verify_shaped_steps(engine, sequences, expected, metadata):
    """Two requests, then steps of VERIFY_ROWS tokens each (a DSpark verify
    batch), with and without padding."""
    engine.reset()
    engine.sequences = sequences
    harness = engine.harness
    harness.add_request("a")
    harness.add_request("b")
    pad = (
        {}
        if metadata is None
        else {"pad_tokens": 3, "pad_reqs": 1, "metadata": metadata}
    )
    forward = _aux_forward(engine, sequences)
    rows = {"a": [], "b": []}
    for schedule in (
        [("a", 7), ("b", 3)],
        [("a", VERIFY_ROWS), ("b", VERIFY_ROWS)],
        [("a", VERIFY_ROWS), ("b", VERIFY_ROWS)],
    ):
        out = harness.step(schedule, forward, **pad)[0]
        for req_id, part in harness.batch.split(out).items():
            rows[req_id].append(part)
    for req_id, parts in rows.items():
        got = torch.cat(parts)
        _assert_aux(got, expected[req_id][: got.shape[0]], engine.hidden, req_id)
