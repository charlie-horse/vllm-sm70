# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 model body at TP1 (model.py, loop.py, layers.py), on CPU.

The registered model class is built on the tiny checkpoint
(fixtures/inkling2_tiny), loaded through its own ``load_weights`` and driven
by the paged engine harness (inkling2_harness.py). Every visit's residual
stream, the final state and the logits are compared with the dense oracle
(inkling2_ref.py) in each engine scenario: prefill, prefill + decode, chunked
prefill, prefix resume and two interleaved requests. Also: greedy ids, the
soft cap in the returned logits, the aux hidden-state capture, the per-visit
dump, batch padding and the refusals of the constructor.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx
from inkling2_ref import Inkling2Oracle

TINY = HERE / "fixtures" / "inkling2_tiny"
LEN_A, LEN_B, PREFIX = 23, 19, 8
BLOCK_SIZE = 8
TOL = 1e-4
AUX_VISITS = (5, 23, 41)


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


def _raw_config() -> dict:
    return json.loads((TINY / "config.json").read_text())


def _weights():
    from safetensors.torch import load_file

    return load_file(TINY / "model.safetensors").items()


class Engine:
    """The model on one harness. `run` frees every request first."""

    def __init__(self, dtype=torch.float32, *, sm70=None, head_dtype=None, tp_size=1):
        from vllm.config import set_current_vllm_config

        from vllm_sm70.models.inkling2_sm70 import model as model_module
        from vllm_sm70.models.inkling2_sm70.sm70_configs import (
            Inkling2LoopSm70MMConfig,
        )

        self.harness = hx.PagedHarness(tp_size, block_size=BLOCK_SIZE, dtype=dtype)
        config = self.harness.vllm_config(0)
        model_config = config.model_config
        model_config.hf_config = Inkling2LoopSm70MMConfig(**_raw_config())
        model_config.head_dtype = dtype if head_dtype is None else head_dtype
        model_config.model = str(TINY)
        config.parallel_config.pipeline_parallel_size = 1
        config.additional_config = None if sm70 is None else {"sm70": sm70}
        with self.harness.tp_rank(0), set_current_vllm_config(config):
            self.model = model_module.Inkling2LoopSm70ForConditionalGeneration(
                vllm_config=config
            )
        self.loaded = self.model.load_weights(_weights())
        self.harness.bind()
        self.states: list[torch.Tensor] = []
        self.sequences: dict[str, torch.Tensor] = {}
        self.num_visits = len(self.model.model.visits)
        self.hidden = self.model.text_config.hidden_size

    def forward(self, rank: int, batch: hx.HarnessBatch) -> torch.Tensor:
        """[P, visits * H + H + vocab]: every visit's state, the final state
        and the logits of each batch token."""
        body = self.model.model
        hook, states = body.visit_hook, []

        def record(visit, layer, pass_idx, state):
            assert (layer, pass_idx) == body.visits[visit] and visit == len(states)
            states.append(state.clone())
            if hook is not None:
                hook(visit, layer, pass_idx, state)

        body.visit_hook = record
        try:
            with torch.no_grad():
                final = self.model(batch.take(self.sequences), batch.positions)
                logits = self.model.compute_logits(final)
        finally:
            body.visit_hook = hook
        self.states = states
        return torch.cat(
            [torch.stack(states, 1).flatten(1).float(), final.float(), logits], dim=1
        )

    def reset(self) -> None:
        for req_id in list(self.harness.requests):
            self.harness.free_request(req_id)

    def run(self, plan, sequences, **step_kwargs) -> dict[str, torch.Tensor]:
        self.reset()
        self.sequences = sequences
        return self.harness.run(plan, self.forward, **step_kwargs)


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine()


@pytest.fixture(scope="module")
def oracle() -> Inkling2Oracle:
    return Inkling2Oracle.from_checkpoint(TINY)


@pytest.fixture(scope="module")
def sequences() -> dict[str, torch.Tensor]:
    """Token ids of requests a and b; b starts with a's first PREFIX ids."""
    generator = torch.Generator().manual_seed(12)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    a = torch.randint(0, vocab, (LEN_A,), generator=generator)
    b = torch.randint(0, vocab, (LEN_B,), generator=generator)
    b[:PREFIX] = a[:PREFIX]
    return {"a": a, "b": b}


@pytest.fixture(scope="module")
def expected(oracle, sequences):
    """Per request: the oracle's output of the whole sequence, and the same
    as rows [T, visits * H + H + vocab]."""
    out = {}
    for req_id, ids in sequences.items():
        ref = oracle.forward(ids[None])
        rows = torch.cat(
            [
                ref.visit_states[:, 0].transpose(0, 1).flatten(1),
                ref.final_hidden_state[0],
                ref.logits[0],
            ],
            dim=1,
        )
        out[req_id] = (ref, rows)
    return out


def _assert_rows(engine, got, want, where):
    """Compare [n, visits * H + H + vocab] rows part by part."""
    assert got.shape == want.shape, where
    split = engine.num_visits * engine.hidden
    diff = (got - want).abs()
    per_visit = diff[:, :split].unflatten(1, (engine.num_visits, -1)).amax((0, 2))
    worst = int(per_visit.argmax())
    assert per_visit[worst] <= TOL, (
        f"{where}: visit {worst} {engine.model.model.visits[worst]} is off by "
        f"{per_visit[worst]:.3e}; first visit over: "
        f"{int((per_visit > TOL).nonzero()[0])}"
    )
    assert diff[:, split : split + engine.hidden].max() <= TOL, f"{where}: final state"
    assert diff[:, split + engine.hidden :].max() <= TOL, f"{where}: logits"


# --- build and load ---------------------------------------------------------


def test_is_a_text_generation_model_and_loads_every_parameter(engine):
    from vllm.model_executor.models.interfaces_base import is_text_generation_model

    from vllm_sm70.models import inkling2_sm70

    model = engine.model
    assert type(model).__name__ == inkling2_sm70.ARCH
    assert is_text_generation_model(type(model))
    assert engine.loaded == {name for name, _ in model.named_parameters()}
    text = model.text_config
    assert model.lm_head.weight.shape == (text.unpadded_vocab_size, text.hidden_size)
    assert model.logits_processor.org_vocab_size == text.unpadded_vocab_size
    assert model.logits_processor.soft_cap == text.final_logit_softcapping
    # Every visit has owners of its own, and the experts are loaded.
    body = model.model
    assert len(set(body.visits)) == engine.num_visits == text.num_layer_visits
    assert set(body.caches.attn) == set(body.caches.conv) == set(body.visits)
    for layer in body.layers:
        assert (
            torch.isfinite(layer.mlp.w13).all() and torch.isfinite(layer.mlp.w2).all()
        )


def test_embed_input_ids_applies_embed_norm(engine, oracle, sequences):
    ids = sequences["a"]
    got = engine.model.embed_input_ids(ids)
    assert torch.allclose(got, oracle.embed(ids), atol=1e-6, rtol=0)
    raw = torch.nn.functional.embedding(ids, engine.model.model.embed_tokens.weight)
    assert not torch.allclose(got, raw, atol=1e-3)


# --- the engine scenarios against the oracle ---------------------------------

SCENARIOS = ("prefill", "decode", "chunk1", "chunk3", "chunk7", "resume", "interleaved")


# Every step padded as a graph capture size is: metadata of a FULL graph and
# of a PIECEWISE graph (inkling2_harness.py).
PADDING = {
    "exact": {},
    "full": {"pad_tokens": 3, "pad_reqs": 1, "metadata": "full"},
    "unpadded": {"pad_tokens": 3, "pad_reqs": 1, "metadata": "unpadded"},
}


@pytest.mark.parametrize("padding", PADDING)
@pytest.mark.parametrize("name", SCENARIOS)
def test_visits_and_logits_match_oracle(engine, sequences, expected, name, padding):
    plans = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)
    assert PREFIX % engine.harness.prefix_alignment == 0
    got = engine.run(plans[name], sequences, **PADDING[padding])
    assert got, name
    for req_id, rows in got.items():
        first = PREFIX if (name, req_id) == ("resume", "b") else 0
        want = expected[req_id][1][first : first + rows.shape[0]]
        _assert_rows(engine, rows, want, f"{name}/{req_id}")
    if name == "resume":
        assert got["a"].shape[0] == PREFIX and got["b"].shape[0] == LEN_B - PREFIX
    elif name == "interleaved":
        assert got["a"].shape[0] == LEN_A and got["b"].shape[0] == LEN_B
    else:
        assert got["a"].shape[0] == LEN_A


@pytest.mark.parametrize("metadata", ["full", "unpadded"])
def test_padding_tokens_and_requests_do_not_change_real_rows(
    engine, sequences, expected, metadata
):
    engine.reset()
    engine.sequences = sequences
    harness = engine.harness
    harness.add_request("a")
    harness.add_request("b")
    rows = {"a": [], "b": []}
    for schedule in ([("a", 5), ("b", 3)], [("a", 1), ("b", 1)], [("a", 4), ("b", 1)]):
        out = harness.step(
            schedule, engine.forward, pad_tokens=3, pad_reqs=1, metadata=metadata
        )[0]
        assert out.shape[0] == harness.batch.num_padded
        assert torch.isfinite(out).all()
        num_tokens = harness.batch.num_tokens
        slots = harness.metadata[0]["model.layers.0.attn"].slot_mapping
        assert slots.shape[0] == num_tokens + (3 if metadata == "full" else 0)
        for req_id, part in harness.batch.split(out).items():
            rows[req_id].append(part)
    for req_id, parts in rows.items():
        got = torch.cat(parts)
        _assert_rows(
            engine, got, expected[req_id][1][: got.shape[0]], f"padded/{req_id}"
        )


def test_greedy_ids_match_oracle(engine, oracle, sequences):
    prompt, new_tokens = sequences["a"][:9], 8
    want = oracle.greedy(prompt[None], new_tokens)[0]
    engine.reset()
    harness = engine.harness
    harness.add_request("g")
    ids = prompt.clone()
    count = len(prompt)
    got = []
    for _ in range(new_tokens):
        engine.sequences = {"g": ids}
        out = harness.step({"g": count}, engine.forward)[0]
        token = out[-1, -engine.model.lm_head.weight.shape[0] :].argmax()
        got.append(int(token))
        ids = torch.cat([ids, token[None]])
        count = 1
    assert got == want.tolist()


# --- head --------------------------------------------------------------------


def test_logits_are_fp32_and_include_the_soft_cap(engine, expected):
    model = engine.model
    cap = model.text_config.final_logit_softcapping
    final = expected["a"][0].final_hidden_state[0]
    # Large states push the raw logits past the cap.
    hidden = final * 400.0
    raw = torch.nn.functional.linear(hidden, model.lm_head.weight).float()
    assert raw.abs().max() > 2 * cap
    logits = model.compute_logits(hidden)
    assert logits.dtype == torch.float32
    assert logits.shape == (final.shape[0], model.text_config.unpadded_vocab_size)
    assert logits.abs().max() <= cap
    capped = cap * torch.tanh(raw / cap)
    assert torch.allclose(logits, capped, atol=1e-5, rtol=0)
    logprobs = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(logprobs, torch.log_softmax(capped, -1), atol=1e-5, rtol=0)
    assert (logprobs - torch.log_softmax(raw, -1)).abs().max() > 1.0


def test_head_gemm_runs_in_the_model_dtype_then_casts():
    from types import SimpleNamespace

    from vllm_sm70.models.inkling2_sm70.model import Inkling2Sm70LogitsProcessor

    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(11, 16, generator=generator).half()
    hidden = torch.randn(5, 16, generator=generator).half()
    head = SimpleNamespace(weight=weight)
    logits = Inkling2Sm70LogitsProcessor._apply_head(None, head, hidden, None)
    assert logits.dtype == torch.float32
    assert torch.equal(logits, torch.nn.functional.linear(hidden, weight).float())


# --- aux hidden states and the visit dump --------------------------------------


def test_no_capture_without_aux_layers(engine, sequences, monkeypatch):
    body = engine.model.model
    assert body.aux_hidden_state_layers == ()

    def fail(state):
        raise AssertionError("captured a hidden state with no aux layer set")

    monkeypatch.setattr(body, "_capture", fail)
    engine.reset()
    engine.sequences = sequences
    engine.harness.add_request("a")
    batch_out = []

    def forward(rank, batch):
        with torch.no_grad():
            out = engine.model(batch.take(sequences), batch.positions)
        batch_out.append(out)
        return out

    engine.harness.step({"a": 6}, forward)
    assert isinstance(batch_out[0], torch.Tensor)


def test_aux_layers_return_visit_inputs_and_final_state(engine, sequences, expected):
    model = engine.model
    ref = expected["a"][0]
    num_visits = engine.num_visits
    model.set_aux_hidden_state_layers((*AUX_VISITS, num_visits))
    try:
        engine.reset()
        engine.harness.add_request("a")
        result = []

        def forward(rank, batch):
            with torch.no_grad():
                result.append(model(batch.take(sequences), batch.positions))
            return result[-1][0]

        engine.harness.step({"a": LEN_A}, forward)
    finally:
        model.set_aux_hidden_state_layers(())
    final, aux = result[0]
    assert len(aux) == len(AUX_VISITS) + 1
    assert aux[-1] is final
    assert torch.allclose(final, ref.final_hidden_state[0], atol=TOL, rtol=0)
    for visit, state in zip(AUX_VISITS, aux):
        want = ref.visit_states[visit - 1, 0]
        assert torch.allclose(state, want, atol=TOL, rtol=0), visit


def test_visit_dump_writes_every_visit(tmp_path, sequences):
    from vllm_sm70.config import clear_cache

    clear_cache()
    engine = Engine(sm70={"inkling2": {"visit_dump_dir": str(tmp_path)}})
    try:
        engine.sequences = sequences
        engine.harness.add_request("a")
        for step, count in enumerate((4, 1)):
            engine.harness.step({"a": count}, engine.forward)
            folder = tmp_path / "rank0" / f"forward{step:06d}"
            files = sorted(folder.iterdir())
            assert len(files) == engine.num_visits
            for visit, (path, state) in enumerate(zip(files, engine.states)):
                layer, pass_idx = engine.model.model.visits[visit]
                assert path.name == (
                    f"visit{visit:02d}_layer{layer:02d}_pass{pass_idx}.pt"
                )
                assert torch.equal(torch.load(path), state)
    finally:
        clear_cache()


def test_no_dump_hook_by_default(engine):
    assert engine.model.model.visit_hook is None


# --- other dtypes and refusals -------------------------------------------------


def test_fp16_model_loads_and_runs(sequences):
    """With the reference's softmax-weight rounding (`hf_rounding`), the fp16
    model is bit-equal to the fp16 oracle in a prefill: every visit, the
    final state and the logits."""
    engine = Engine(torch.float16)
    for name, param in engine.model.named_parameters():
        spec = engine.model.model.weight_map.params[name]
        assert param.dtype == spec.dtype, name
    assert engine.model.lm_head.weight.dtype == torch.float16
    ids = sequences["a"]
    ref = Inkling2Oracle.from_checkpoint(TINY, dtype=torch.float16).forward(ids[None])
    want = torch.cat(
        [
            ref.visit_states[:, 0].transpose(0, 1).flatten(1).float(),
            ref.final_hidden_state[0].float(),
            ref.logits[0].float(),
        ],
        dim=1,
    )
    for layer in engine.model.model.layers:
        layer.attn.hf_rounding = True
    for padding in PADDING.values():
        got = engine.run(hx.prefill_plan("a", LEN_A), sequences, **padding)["a"]
        assert engine.states[0].dtype == torch.float16
        assert torch.equal(got, want)
    for layer in engine.model.model.layers:
        layer.attn.hf_rounding = False
    got = engine.run(hx.decode_plan("a", 12, 9), sequences)["a"]
    assert got.shape[0] == 12 and torch.isfinite(got).all()
    cap = engine.model.text_config.final_logit_softcapping
    assert got[:, -engine.model.lm_head.weight.shape[0] :].abs().max() <= cap


def test_out_of_range_expert_block_fails_the_fp16_load(monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import expert_weights

    values = expert_weights.PackedExpertWeight._chunk_values
    monkeypatch.setattr(
        expert_weights.PackedExpertWeight,
        "_chunk_values",
        lambda self, experts: values(self, experts) * 1e6,
    )
    with pytest.raises(ValueError, match=r"experts\.w13_weight.*fit in fp16"):
        Engine(torch.float16)


def test_post_load_hook_reaches_submodule_hooks(engine):
    calls = []
    mlp = engine.model.model.layers[0].mlp
    mlp.process_weights_after_loading = lambda: calls.append("mlp")
    try:
        engine.model.process_weights_after_loading()
    finally:
        del mlp.process_weights_after_loading
    assert calls == ["mlp"]


def test_refuses_head_dtype_other_than_the_model_dtype():
    with pytest.raises(ValueError, match="head_dtype"):
        Engine(torch.float16, head_dtype=torch.float32)
