# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 model body at TP2 (model.py, loop.py, layers.py), on CPU.

Two rank modules of the registered model class are built on the tiny
checkpoint (fixtures/inkling2_tiny); each loads its own shards through its
``load_weights``. The paged engine harness (inkling2_harness.py) runs them in
lockstep and sums the tensors they all-reduce. Every visit's residual stream,
the final state and the logits are compared with a TP1 run of the same class
in each engine scenario. Also: the shards against the TP1 parameters, the
number and place of the all-reduces, the replicated conv state, and the
per-rank partial maxima of the audit.
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

TINY = HERE / "fixtures" / "inkling2_tiny"
LEN_A, LEN_B, PREFIX = 23, 19, 8
BLOCK_SIZE = 8
TOL = 5e-5


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
    """One model per TP rank on one harness. `run` frees every request
    first and returns rank 0's rows; `rank_rows` keeps every rank's rows of
    each step."""

    def __init__(self, tp_size: int, dtype=torch.float32, *, sm70=None):
        from vllm.config import set_current_vllm_config

        from vllm_sm70.models.inkling2_sm70 import model as model_module
        from vllm_sm70.models.inkling2_sm70.sm70_configs import (
            Inkling2LoopSm70MMConfig,
        )

        self.tp_size = tp_size
        self.harness = hx.PagedHarness(tp_size, block_size=BLOCK_SIZE, dtype=dtype)
        self.models = []
        self.loaded = []
        for rank in range(tp_size):
            config = self.harness.vllm_config(rank)
            model_config = config.model_config
            model_config.hf_config = Inkling2LoopSm70MMConfig(**_raw_config())
            model_config.head_dtype = dtype
            model_config.model = str(TINY)
            config.parallel_config.pipeline_parallel_size = 1
            config.additional_config = None if sm70 is None else {"sm70": sm70}
            with self.harness.tp_rank(rank), set_current_vllm_config(config):
                model = model_module.Inkling2LoopSm70ForConditionalGeneration(
                    vllm_config=config
                )
            self.loaded.append(model.load_weights(_weights()))
            self.models.append(model)
        self.harness.bind()
        self.sequences: dict[str, torch.Tensor] = {}
        self.rank_rows: list[list[torch.Tensor]] = []
        self.num_visits = len(self.models[0].model.visits)
        self.hidden = self.models[0].text_config.hidden_size

    def forward(self, rank: int, batch: hx.HarnessBatch) -> torch.Tensor:
        """[P, visits * H + H + vocab]: every visit's state, the final state
        and the logits of each batch token, as rank `rank` computes them."""
        model = self.models[rank]
        body = model.model
        hook, states = body.visit_hook, []

        def record(visit, layer, pass_idx, state):
            states.append(state.clone())
            if hook is not None:
                hook(visit, layer, pass_idx, state)

        body.visit_hook = record
        try:
            with torch.no_grad():
                final = model(batch.take(self.sequences), batch.positions)
                logits = model.compute_logits(final)
        finally:
            body.visit_hook = hook
        return torch.cat(
            [torch.stack(states, 1).flatten(1).float(), final.float(), logits], dim=1
        )

    def _combine(self, results: list[torch.Tensor]) -> torch.Tensor:
        self.rank_rows.append(results)
        return results[0]

    def reset(self) -> None:
        for req_id in list(self.harness.requests):
            self.harness.free_request(req_id)
        self.rank_rows = []

    def run(self, plan, sequences) -> dict[str, torch.Tensor]:
        self.reset()
        self.sequences = sequences
        return self.harness.run(plan, self.forward, combine=self._combine)


@pytest.fixture(scope="module")
def tp1() -> Engine:
    return Engine(1)


@pytest.fixture(scope="module")
def tp2() -> Engine:
    return Engine(2)


@pytest.fixture(scope="module")
def sequences() -> dict[str, torch.Tensor]:
    """Token ids of requests a and b; b starts with a's first PREFIX ids."""
    generator = torch.Generator().manual_seed(12)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    a = torch.randint(0, vocab, (LEN_A,), generator=generator)
    b = torch.randint(0, vocab, (LEN_B,), generator=generator)
    b[:PREFIX] = a[:PREFIX]
    return {"a": a, "b": b}


# --- build and per-rank loading ------------------------------------------------


def test_each_rank_loads_every_parameter_of_its_own(tp2):
    for rank, (model, loaded) in enumerate(zip(tp2.models, tp2.loaded)):
        assert (model.tp_size, model.tp_rank) == (2, rank)
        assert loaded == {name for name, _ in model.named_parameters()}
        assert model.model.weight_map.tp_rank == rank


def test_attention_shards_are_the_head_slices_of_tp1(tp1, tp2):
    """Rank r holds the r-th half of the q heads with their rel, gate and
    sink rows, of the KV heads and of the o_proj columns."""
    text = tp1.models[0].text_config
    local = set(text.local_layer_ids)
    for index, whole in enumerate(tp1.models[0].model.layers):
        is_local = index in local
        heads = text.swa_num_attention_heads if is_local else text.num_attention_heads
        kv_heads = (
            text.swa_num_key_value_heads if is_local else text.num_key_value_heads
        )
        head_dim = text.swa_head_dim if is_local else text.head_dim
        assert whole.attn.sizes == (
            heads * head_dim,
            heads * text.d_rel,
            heads * head_dim,
            heads,
            kv_heads * head_dim,
            kv_heads * head_dim,
        )
        shards = [m.model.layers[index].attn for m in tp2.models]
        for shard in shards:
            assert shard.num_heads == heads // 2
            assert shard.sizes == tuple(size // 2 for size in whole.attn.sizes)
        parts = [shard.qkvr.weight.split(shard.sizes) for shard in shards]
        for site, want in enumerate(whole.attn.qkvr.weight.split(whole.attn.sizes)):
            got = torch.cat([part[site] for part in parts])
            assert torch.equal(got, want), (index, site)
        assert torch.equal(
            torch.cat([shard.wo_ud.weight for shard in shards], dim=1),
            whole.attn.wo_ud.weight,
        )
        for name in ("q_scale", "r_scale"):
            want = getattr(whole.attn, name)
            got = torch.cat([getattr(shard, name) for shard in shards])
            assert want.shape[0] == heads and torch.equal(got, want), (index, name)
        # One relative table for all heads.
        for shard in shards:
            assert torch.equal(shard.rel_logits_proj, whole.attn.rel_logits_proj)
        for rank, model in enumerate(tp2.models):
            owner = model.model.caches.attn[index, 0]
            assert owner.num_kv_heads == kv_heads // 2, (index, rank)


def test_expert_shards_are_the_intermediate_slices_of_tp1(tp1, tp2):
    for index, whole in enumerate(tp1.models[0].model.layers):
        shards = [m.model.layers[index].mlp for m in tp2.models]
        gate, up = whole.mlp.w13.chunk(2, dim=1)
        got = [shard.w13.chunk(2, dim=1) for shard in shards]
        assert torch.equal(torch.cat([part[0] for part in got], dim=1), gate)
        assert torch.equal(torch.cat([part[1] for part in got], dim=1), up)
        assert torch.equal(torch.cat([s.w2 for s in shards], dim=2), whole.mlp.w2)
        for shard in shards:
            assert torch.equal(shard.gate.weight, whole.mlp.gate.weight)
            assert torch.equal(shard.gate.bias, whole.mlp.gate.bias)


def test_other_parameters_are_whole_on_every_rank(tp1, tp2):
    whole = dict(tp1.models[0].named_parameters())
    sharded = (".attn.qkvr.", ".attn.wo_ud.", ".attn.q_scale", ".attn.r_scale")
    sharded += (".k_sconv.", ".v_sconv.", "lm_head.")
    replicated = [name for name in whole if not any(s in name for s in sharded)]
    assert "model.layers.0.attn_sconv.weight" in replicated
    assert "model.layers.0.mlp_sconv.weight" in replicated
    for model in tp2.models:
        params = dict(model.named_parameters())
        assert set(params) == set(whole)
        for name in replicated:
            assert torch.equal(params[name], whole[name]), name


def test_lm_head_rows_are_split_over_the_ranks(tp1, tp2):
    """Each rank holds half of the vocabulary rows; the vocabulary attributes
    are those of a VocabParallelEmbedding without padding."""
    whole = tp1.models[0].lm_head
    heads = [model.lm_head for model in tp2.models]
    assert whole.tp_size == 1
    assert torch.equal(torch.cat([h.weight for h in heads]), whole.weight)
    rows = whole.weight.shape[0] // 2
    for rank, head in enumerate(heads):
        assert (head.tp_size, head.tp_rank) == (2, rank)
        assert head.org_vocab_size == head.org_vocab_size_padded == 2 * rows
        assert head.num_embeddings_per_partition == rows
        assert head.shard_indices.org_vocab_start_index == rank * rows
        assert head.shard_indices.num_org_vocab_padding == 0


# --- the engine scenarios against TP1 ------------------------------------------

SCENARIOS = ("prefill", "decode", "chunk1", "chunk3", "chunk7", "resume", "interleaved")


def _assert_rows(engine, got, want, where):
    """Compare [n, visits * H + H + vocab] rows part by part."""
    assert got.shape == want.shape, where
    assert torch.isfinite(got).all(), where
    split = engine.num_visits * engine.hidden
    diff = (got - want).abs()
    per_visit = diff[:, :split].unflatten(1, (engine.num_visits, -1)).amax((0, 2))
    worst = int(per_visit.argmax())
    assert per_visit[worst] <= TOL, (
        f"{where}: visit {worst} {engine.models[0].model.visits[worst]} is off by "
        f"{per_visit[worst]:.3e}; first visit over: "
        f"{int((per_visit > TOL).nonzero()[0])}"
    )
    assert diff[:, split : split + engine.hidden].max() <= TOL, f"{where}: final state"
    assert diff[:, split + engine.hidden :].max() <= TOL, f"{where}: logits"


@pytest.mark.parametrize("name", SCENARIOS)
def test_visits_and_logits_match_tp1(tp1, tp2, sequences, name):
    plan = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)[name]
    assert PREFIX % tp2.harness.prefix_alignment == 0
    want = tp1.run(plan, sequences)
    got = tp2.run(plan, sequences)
    assert set(got) == set(want) and got, name
    for req_id, rows in got.items():
        _assert_rows(tp2, rows, want[req_id], f"{name}/{req_id}")
    # The residual stream is replicated: both ranks hold the same values.
    for step, (rank0, rank1) in enumerate(tp2.rank_rows):
        assert torch.equal(rank0, rank1), (name, step)


@pytest.mark.parametrize("metadata", ["full", "unpadded"])
def test_padding_does_not_change_real_rows_at_tp2(tp1, tp2, sequences, metadata):
    rows = {}
    for engine in (tp1, tp2):
        engine.reset()
        engine.sequences = sequences
        harness = engine.harness
        harness.add_request("a")
        harness.add_request("b")
        parts = {"a": [], "b": []}
        for schedule in ([("a", 5), ("b", 3)], [("a", 1), ("b", 1)]):
            out = harness.step(
                schedule, engine.forward, pad_tokens=3, pad_reqs=1, metadata=metadata
            )
            assert all(torch.isfinite(o).all() for o in out)
            for req_id, part in harness.batch.split(out[0]).items():
                parts[req_id].append(part)
        rows[engine.tp_size] = {r: torch.cat(p) for r, p in parts.items()}
    for req_id, got in rows[2].items():
        _assert_rows(tp2, got, rows[1][req_id], f"padded/{req_id}")


# --- all-reduce placement --------------------------------------------------------


def test_two_all_reduces_per_visit_feed_the_replicated_convs(
    tp2, sequences, monkeypatch
):
    """Per visit a rank all-reduces its o_proj partial, then its expert
    partial, and nothing else (the logits all-gather once); the short convs
    store the reduced tensors, so the ranks' conv streams hold the same
    values."""
    from vllm_sm70.models.inkling2_sm70 import layers
    from vllm_sm70.models.inkling2_sm70.cache import STREAM_ATTN, STREAM_MLP

    group = tp2.harness._group
    reduce, gather = group.all_reduce, group.all_gather
    calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    partials: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    conv_inputs: dict[int, list[torch.Tensor]] = {0: [], 1: []}

    def counted(tensor):
        rank = group.rank_in_group
        out = reduce(tensor)
        calls.append((rank, tensor, out))
        return out

    gathers: list[tuple[int, torch.Tensor]] = []

    def logits_gather(tensor, dim=-1):
        gathers.append((group.rank_in_group, tensor))
        return gather(tensor, dim)

    short_conv = layers.short_conv

    def conv(x, weight, owner, stream, out):
        conv_inputs[group.rank_in_group].append(x)
        short_conv(x, weight, owner, stream, out)

    monkeypatch.setattr(group, "all_reduce", counted)
    monkeypatch.setattr(group, "all_gather", logits_gather)
    monkeypatch.setattr(layers, "short_conv", conv)
    tp2.reset()
    tp2.sequences = sequences
    tp2.harness.add_request("a")
    for model in tp2.models:
        rank = model.tp_rank
        model.model.partial_audit = _Recorder(partials[rank])
    try:
        tp2.harness.step({"a": 9}, tp2.forward)
    finally:
        for model in tp2.models:
            model.model.partial_audit = None
    visits = tp2.num_visits
    assert len(calls) == 2 * 2 * visits
    # The body all-gathers nothing; the sharded head's logits once per rank,
    # in the model dtype.
    rows = tp2.models[0].lm_head.weight.shape[0]
    assert sorted(rank for rank, _ in gathers) == [0, 1]
    for _, sent in gathers:
        assert sent.shape == (9, rows)
        assert sent.dtype == tp2.models[0].lm_head.weight.dtype
    for rank in (0, 1):
        own = [(sent, out) for r, sent, out in calls if r == rank]
        assert len(own) == len(partials[rank]) == len(conv_inputs[rank]) == 2 * visits
        for (sent, out), partial, conv_input in zip(
            own, partials[rank], conv_inputs[rank]
        ):
            assert sent is partial
            assert conv_input is out
    for (_, sent0, out0), (_, sent1, out1) in zip(
        [c for c in calls if c[0] == 0], [c for c in calls if c[0] == 1]
    ):
        assert torch.equal(out0, sent0 + sent1) and torch.equal(out0, out1)
        assert not torch.equal(sent0, sent1)
    # The conv state of the hidden streams is the same on both ranks.
    for name, cache0 in tp2.harness.caches[0].items():
        if ".conv_state" not in name:
            continue
        owner = tp2.harness.owners[0][name]
        cache1 = tp2.harness.caches[1][name]
        for stream in (STREAM_ATTN, STREAM_MLP):
            offset, width = owner.stream_ranges[stream]
            part0 = cache0[..., offset : offset + width]
            part1 = cache1[..., offset : offset + width]
            assert torch.equal(part0.isnan(), part1.isnan()), name
            assert torch.equal(part0.nan_to_num(), part1.nan_to_num()), name


class _Recorder:
    """A partial audit that keeps the tensors it is shown."""

    def __init__(self, seen: list[torch.Tensor]) -> None:
        self.seen = seen

    def record(self, visit: int, site: str, partial: torch.Tensor) -> None:
        assert len(self.seen) == 2 * visit + ("attn", "mlp").index(site)
        self.seen.append(partial)

    def save(self) -> None:
        pass


# --- the audit of the per-rank partials -------------------------------------------


def test_audit_reports_the_partial_maxima_of_each_rank(tmp_path, sequences):
    from vllm_sm70.config import clear_cache
    from vllm_sm70.models.inkling2_sm70.loop import PartialAudit

    clear_cache()
    try:
        engine = Engine(2, sm70={"inkling2": {"visit_dump_dir": str(tmp_path)}})
    finally:
        clear_cache()
    audits = [model.model.partial_audit for model in engine.models]
    recorded: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for rank, audit in enumerate(audits):
        assert isinstance(audit, PartialAudit)
        assert audit.maxima.shape == (engine.num_visits, 2)
        assert audit.maxima.dtype == torch.float32
        assert not audit.maxima.any()
        assert "model.partial_audit.maxima" not in engine.models[rank].state_dict()
        record = audit.record

        def keep(visit, site, partial, record=record, seen=recorded[rank]):
            seen.append(partial.abs().max())
            record(visit, site, partial)

        audit.record = keep
    engine.sequences = sequences
    engine.harness.add_request("a")
    engine.harness.step({"a": 7}, engine.forward)
    first = [audit.maxima.clone() for audit in audits]
    for rank, audit in enumerate(audits):
        want = torch.stack(recorded[rank]).reshape(engine.num_visits, 2)
        assert torch.equal(audit.maxima, want)
        assert torch.isfinite(audit.maxima).all() and (audit.maxima > 0).all()
        report = audit.report()
        assert list(report) == ["attn", "mlp"]
        assert report["attn"] == float(want[:, 0].max())
        assert report["mlp"] == float(want[:, 1].max())
        saved = torch.load(tmp_path / f"rank{rank}" / "partial_maxima.pt")
        assert torch.equal(saved, audit.maxima)
    assert not torch.equal(audits[0].maxima, audits[1].maxima)
    # The maxima run over the forwards.
    engine.harness.step({"a": 5}, engine.forward)
    for rank, audit in enumerate(audits):
        assert (audit.maxima >= first[rank]).all()
        assert (audit.maxima > first[rank]).any()
        saved = torch.load(tmp_path / f"rank{rank}" / "partial_maxima.pt")
        assert torch.equal(saved, audit.maxima)
        # Each rank dumps its visits beside its maxima.
        assert len(list((tmp_path / f"rank{rank}" / "forward000001").iterdir())) == (
            engine.num_visits
        )


def test_audit_keeps_a_non_finite_partial():
    from vllm_sm70.models.inkling2_sm70.loop import PartialAudit

    audit = PartialAudit(3)
    audit.record(1, "mlp", torch.tensor([[1.0, -4.0]], dtype=torch.float16))
    audit.record(1, "mlp", torch.tensor([[2.0, 3.0]], dtype=torch.float16))
    audit.record(2, "attn", torch.tensor([[1.0, float("inf")]]))
    audit.record(0, "attn", torch.tensor([[float("nan"), 1.0]]))
    audit.record(0, "attn", torch.tensor([[5.0, 1.0]]))
    assert audit.maxima[1].tolist() == [0.0, 4.0]
    assert audit.maxima[2, 0] == float("inf")
    assert audit.maxima[0, 0].isnan()
    audit.save()  # no directory: nothing is written


def test_no_audit_by_default(tp2):
    assert all(model.model.partial_audit is None for model in tp2.models)


# --- fp16 and refusals ---------------------------------------------------------------


def test_fp16_ranks_run_and_agree(sequences):
    engine = Engine(2, torch.float16)
    got = engine.run(hx.decode_plan("a", 12, 9), sequences)["a"]
    assert got.shape[0] == 12 and torch.isfinite(got).all()
    for rank0, rank1 in engine.rank_rows:
        assert torch.equal(rank0, rank1)
    for model in engine.models:
        assert model.model.layers[0].mlp.w13.dtype == torch.float16


def test_refuses_a_tp_size_that_does_not_split_the_heads():
    with pytest.raises(ValueError, match="split over 4 ranks"):
        Engine(4)
