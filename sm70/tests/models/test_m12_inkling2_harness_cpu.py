# SPDX-License-Identifier: Apache-2.0
"""Self-test of the M12 Inkling 2 CPU engine harness (inkling2_harness.py).

Needs upstream v0.33.0's Python importable (skipped otherwise). A toy module
writes a code of (token, position, visit, head, column) into every cache and
reads its window back through the block table; the sums are compared with a
closed form in every scenario at TP1 and at simulated TP2, and injected
faults have to show.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx

TINY = HERE / "fixtures" / "inkling2_tiny" / "config.json"
MINI = HERE / "data" / "inkling2_mini" / "config.json"
LEN_A, LEN_B, PREFIX = 23, 19, 8


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


def text_config(path: pathlib.Path = TINY):
    from vllm_sm70.models.inkling2_sm70.configs import Inkling2ModelConfig

    return Inkling2ModelConfig(**json.loads(path.read_text())["text_config"])


def code(token, position, visit, head, column):
    """An integer in 1..251, exact in fp16."""
    return (token * 31 + position * 17 + visit * 13 + head * 7 + column * 3) % 251 + 1


class Toy:
    """One rank. Output [P, 2 * visits] float64: per visit the sum over the
    attention window and over the conv window of every head's values, all
    ranks included."""

    def __init__(self, config, vllm_config, tp_size: int, rank: int):
        from vllm_sm70.models.inkling2_sm70.cache import build_cache_owners

        self.rank = rank
        self.owners = build_cache_owners(
            config, vllm_config=vllm_config, tp_size=tp_size
        )
        self.visits = list(self.owners.attn)
        self.wrong_slot = False  # fault: write one slot too far

    def columns(self, owner):
        ranges = getattr(owner, "stream_ranges", None)
        if ranges is None:
            return torch.arange(owner.kv_cache.shape[3])
        return torch.cat(
            [torch.arange(start, start + width) for start, width in ranges]
        )

    def window(self, owner, md) -> int | None:
        return getattr(md, "sliding_window", owner.sliding_window)

    def visit(self, owner, visit: int, tokens):
        from vllm_sm70.models.inkling2_sm70.backend import cache_metadata

        md = cache_metadata(owner)
        cache = owner.kv_cache
        block_size = md.block_size
        assert block_size == cache.shape[2]
        heads = self.rank * owner.num_kv_heads + torch.arange(owner.num_kv_heads)
        columns = self.columns(owner)
        real = md.slot_mapping >= 0
        slots = (md.slot_mapping[real] + int(self.wrong_slot)) % cache[
            :, 0, :, 0
        ].numel()
        positions = md.positions[real]
        values = code(
            tokens[real][:, None, None],
            positions[:, None, None],
            visit,
            heads[None, :, None],
            columns[None, None, :],
        ).to(cache.dtype)
        # Indexed dims: [columns, tokens]; the head dim follows.
        cache[slots // block_size, :, slots % block_size, columns[:, None]] = (
            values.permute(2, 0, 1)
        )

        window = self.window(owner, md)
        span = int(positions.max()) + 1 if window is None else window
        keys = positions[:, None] - torch.arange(span)[None, :]
        allowed = keys >= 0
        keys = keys.clamp(min=0)
        rows = md.seq_idx[real].long()[:, None]
        blocks = md.block_table[rows, keys // block_size].long()
        read = cache[blocks, :, keys % block_size][..., columns].double().sum((2, 3))
        out = torch.zeros(real.shape[0], dtype=torch.float64)
        out[real] = torch.where(allowed, read, 0.0).sum(1)
        return out

    def forward(self, batch, tokens_of):
        from vllm.distributed import tensor_model_parallel_all_reduce

        tokens = batch.take(tokens_of)
        out = []
        for index, visit in enumerate(self.visits):
            out.append(self.visit(self.owners.attn[visit], 2 * index, tokens))
            out.append(self.visit(self.owners.conv[visit], 2 * index + 1, tokens))
        return tensor_model_parallel_all_reduce(torch.stack(out, dim=1))


def build(tp_size: int, config=None, **kwargs):
    config = config or text_config()
    kwargs.setdefault("block_size", 4)
    kwargs.setdefault("dtype", torch.float16)
    harness = hx.PagedHarness(tp_size, **kwargs)
    toys = []
    for rank in range(tp_size):
        with harness.tp_rank(rank):
            toys.append(Toy(config, harness.vllm_config(rank), tp_size, rank))
    harness.bind()
    return harness, toys


def sequences(prefix: int = PREFIX):
    a = torch.arange(LEN_A) * 5 % 37 + 1
    b = torch.cat([a[:prefix], torch.arange(LEN_B - prefix) * 7 % 41 + 40])
    return {"a": a, "b": b}


def expected(toy: Toy, tp_size: int, tokens, first: int = 0):
    """Closed form of Toy's output for positions first.. of one sequence."""
    out = torch.zeros(len(tokens) - first, 2 * len(toy.visits), dtype=torch.float64)
    for index, visit in enumerate(toy.visits):
        for kind, owner in enumerate((toy.owners.attn[visit], toy.owners.conv[visit])):
            heads = torch.arange(owner.num_kv_heads * tp_size)
            columns = toy.columns(owner)
            per_key = code(
                tokens[:, None, None],
                torch.arange(len(tokens))[:, None, None],
                2 * index + kind,
                heads[None, :, None],
                columns[None, None, :],
            ).sum((1, 2))
            window = owner.sliding_window
            for position in range(first, len(tokens)):
                low = 0 if window is None else max(0, position - window + 1)
                out[position - first, 2 * index + kind] = per_key[
                    low : position + 1
                ].sum()
    return out


def run(harness, toys, plan, tokens, **kwargs):
    return harness.run(
        plan, lambda rank, batch: toys[rank].forward(batch, tokens), **kwargs
    )


SCENARIOS = ["prefill", "decode", "chunk1", "chunk3", "chunk7", "resume", "interleaved"]


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_toy_reads_back_what_it_wrote(tp_size, scenario):
    harness, toys = build(tp_size)
    tokens = sequences()
    assert harness.prefix_alignment == 4
    plan = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)[scenario]
    got = run(harness, toys, plan, tokens)
    want = {"a": expected(toys[0], tp_size, tokens["a"])}
    if scenario == "resume":
        want = {
            "a": want["a"][:PREFIX],
            "b": expected(toys[0], tp_size, tokens["b"], PREFIX),
        }
        assert harness.requests["b"].first == PREFIX
    elif scenario == "interleaved":
        want["b"] = expected(toys[0], tp_size, tokens["b"])
        assert not harness.requests
    assert set(got) == set(want)
    for req_id in want:
        assert torch.equal(got[req_id], want[req_id]), req_id


def test_scenarios_cover_the_plan():
    plans = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)
    assert list(plans) == SCENARIOS

    def steps(name):
        return [dict(args[0]) for op, *args in plans[name] if op == "step"]

    assert steps("prefill") == [{"a": LEN_A}]
    assert steps("decode") == [{"a": LEN_A // 2}] + [{"a": 1}] * (LEN_A - LEN_A // 2)
    assert [s["a"] for s in steps("chunk7")] == [7, 7, 7, 2]
    assert steps("resume") == [{"a": PREFIX}, {"b": LEN_B - PREFIX}]
    mixed = steps("interleaved")
    assert sum(s.get("a", 0) for s in mixed) == LEN_A
    assert sum(s.get("b", 0) for s in mixed) == LEN_B
    assert any(len(s) == 2 and max(s.values()) > 1 for s in mixed)
    assert any(list(s) == ["b", "a"] for s in mixed) and {"b": 1} in mixed


def test_tp2_partials_are_summed_per_rank_and_by_run():
    harness, toys = build(2)
    tokens = sequences()
    harness.add_request("a")
    full = harness.step(
        {"a": LEN_A}, lambda rank, batch: toys[rank].forward(batch, tokens)
    )
    assert torch.equal(full[0], full[1])

    harness2, toys2 = build(2)
    data = {**tokens, "c": tokens["a"]}

    def partial(rank, batch):
        from vllm.distributed import get_tensor_model_parallel_rank, parallel_state

        assert get_tensor_model_parallel_rank() == rank
        assert parallel_state.get_tp_group().world_size == 2
        toy = toys2[rank]
        visit = toy.visits[0]
        return toy.visit(toy.owners.attn[visit], 0, batch.take(data))

    got = harness2.run(hx.prefill_plan("a", LEN_A), partial, combine="sum")
    rank0 = harness2.run(hx.prefill_plan("c", 5), partial, combine=lambda r: r[0])
    assert torch.equal(got["a"], full[0][:, 0])
    assert not torch.equal(2 * rank0["c"], got["a"][:5])


def test_state_is_restored_and_rank_errors_surface():
    from vllm import forward_context
    from vllm.distributed import parallel_state

    harness, _ = build(2)
    harness.add_request("a")
    assert parallel_state._TP is None
    assert forward_context._forward_context is None

    def uneven(rank, batch):
        from vllm.distributed import tensor_model_parallel_all_reduce

        if rank == 0:
            tensor_model_parallel_all_reduce(torch.ones(1))
        return torch.zeros(batch.num_padded)

    harness.check_writes = False
    with pytest.raises(RuntimeError, match="collective"):
        harness.step({"a": 2}, uneven)

    def broken(rank, batch):
        if rank == 1:
            raise KeyError("rank 1 failed")
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(torch.ones(1))

    with pytest.raises(KeyError, match="rank 1 failed"):
        harness.step({"a": 2}, broken)
    assert parallel_state._TP is None
    assert forward_context._forward_context is None
    assert harness.requests["a"].computed == 0


def test_metadata_comes_from_the_builders_and_matches_the_tables():
    from vllm_sm70.models.inkling2_sm70 import backend

    harness, toys = build(2)
    tokens = sequences()

    def forward(rank, batch):
        return toys[rank].forward(batch, tokens)

    harness.add_request("a")
    harness.add_request("b")
    harness.step({"a": 6}, forward)
    harness.step({"a": 5, "b": 1}, forward, pad_tokens=3, pad_reqs=1)
    batch = harness.batch
    assert batch.req_ids == ("b", "a")  # the one-token request first
    assert (batch.num_tokens, batch.num_padded, batch.num_reqs_padded) == (6, 9, 3)
    assert batch.positions.tolist() == [0, 6, 7, 8, 9, 10, 0, 0, 0]
    assert batch.query_start_loc.tolist() == [0, 1, 6, 6]
    assert batch.rows("a") == slice(1, 6)

    tables = set()
    for rank in range(2):
        owners = toys[rank].owners
        for name, md in harness.metadata[rank].items():
            owner = harness.owners[rank][name]
            is_attn = isinstance(md, backend.Inkling2AttentionMetadata)
            assert is_attn == (name in {o.prefix for o in owners.attn.values()})
            assert isinstance(md, backend.Inkling2SconvMetadata)
            assert md.num_actual_tokens == 6
            assert md.seq_idx.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0]
            assert md.query_start.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0]
            assert md.positions.tolist() == batch.positions.tolist()
            assert md.slot_mapping[6:].tolist() == [-1] * 3
            assert md.block_table.shape[0] == 3
            assert md.block_table[2].tolist() == [0] * md.block_table.shape[1]
            real = md.slot_mapping[:6]
            block = md.block_table[
                md.seq_idx[:6].long(), md.positions[:6] // md.block_size
            ]
            assert (
                block * md.block_size + md.positions[:6] % md.block_size
            ).tolist() == (real.tolist())
            assert owner.kv_cache is harness.caches[rank][name]
            assert owner.kv_cache.shape[1] == owner.num_kv_heads
            assert owner.kv_cache.dtype == torch.float16
            if is_attn:
                assert md.sliding_window == owner.sliding_window
                assert (md.num_decodes, md.num_prefills) == (1, 2)
                assert md.seq_lens.tolist() == [1, 11, 0]
            if rank == 0:
                tables.add(tuple(md.block_table.flatten().tolist()))
    # No two owners share a block table.
    assert len(tables) == len(harness.tables) == 96
    assert harness.metadata[0].keys() == harness.metadata[1].keys()


def test_sliding_window_owners_lose_blocks_before_the_window():
    harness, toys = build(1)
    tokens = sequences()
    run(harness, toys, hx.chunked_plan("a", LEN_A, 3), tokens)
    owners = toys[0].owners
    local = owners.attn[0, 0]
    full = next(o for o in owners.attn.values() if not o.is_local)
    conv = owners.conv[0, 0]
    assert local.sliding_window == 8 and conv.sliding_window == 4
    # The last step started at 21 computed tokens.
    assert harness.tables[local.prefix].rows["a"][:3] == [0, 0, 0]
    assert 0 not in harness.tables[local.prefix].rows["a"][3:]
    assert harness.tables[conv.prefix].rows["a"][:4] == [0, 0, 0, 0]
    assert 0 not in harness.tables[full.prefix].rows["a"]
    assert harness.caches[0][local.prefix][0].isnan().all()
    live = set(harness.tables[local.prefix].rows["a"]) - {0}
    for block in range(harness.num_blocks):
        assert bool(harness.caches[0][local.prefix][block].isnan().all()) == (
            block not in live
        )


def _mismatch(harness, toys, tokens, tamper):
    harness.add_request("a")
    harness.add_request("b")

    def forward(rank, batch):
        return toys[rank].forward(batch, tokens)

    harness.step({"a": 6, "b": 5}, forward)
    out = harness.step({"a": 3, "b": 4}, forward, tamper=tamper)[0]
    want = torch.cat(
        [
            expected(toys[0], harness.tp_size, tokens[r], 0)[s : s + n]
            for r, s, n in (("a", 6, 3), ("b", 5, 4))
        ]
    )
    same = (out == want).all(0)
    return [i for i in range(same.shape[0]) if not same[i]]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_cross_wired_block_table_is_detected(tp_size):
    tokens = sequences(prefix=0)
    harness, toys = build(tp_size)
    assert _mismatch(harness, toys, tokens, None) == []

    # The two requests' rows of one owner's block table, swapped on one rank.
    harness, toys = build(tp_size)
    full = next(o for o in toys[0].owners.attn.values() if not o.is_local)
    column = 2 * toys[0].visits.index((full.layer_idx, full.pass_idx))

    def swap_rows(rank, metadata):
        if rank == tp_size - 1:
            md = metadata[full.prefix]
            md.block_table = md.block_table.flip(0)

    assert _mismatch(harness, toys, tokens, swap_rows) == [column]

    # One visit handed the block table of another visit of the same layer.
    harness, toys = build(tp_size)
    first, second = toys[0].owners.conv[6, 0], toys[0].owners.conv[6, 1]
    columns = [2 * toys[0].visits.index(v) + 1 for v in ((6, 0), (6, 1))]

    def swap_owners(rank, metadata):
        a, b = metadata[first.prefix], metadata[second.prefix]
        a.block_table, b.block_table = b.block_table, a.block_table

    assert _mismatch(harness, toys, tokens, swap_owners) == columns


def test_stray_and_missing_writes_fail_the_step():
    tokens = sequences()
    harness, toys = build(1)
    harness.add_request("a")
    toys[0].wrong_slot = True
    with pytest.raises(hx.HarnessError, match="outside the slot mapping"):
        harness.step({"a": 5}, lambda rank, batch: toys[rank].forward(batch, tokens))

    harness, toys = build(1)
    harness.add_request("a")
    owner = toys[0].owners.attn[0, 0]

    def half(rank, batch):
        out = toys[rank].forward(batch, tokens)
        owner.kv_cache[harness.tables[owner.prefix].rows["a"][0], :, 1] = float("nan")
        return out

    with pytest.raises(hx.HarnessError, match="were not written"):
        harness.step({"a": 5}, half)

    def null_block(rank, batch):
        owner.kv_cache[0, :, 0] = 1.0
        return toys[rank].forward(batch, tokens)

    harness, toys = build(1)
    harness.add_request("a")
    owner = toys[0].owners.attn[0, 0]
    with pytest.raises(hx.HarnessError, match="null block"):
        harness.step({"a": 5}, null_block)


def test_request_rules():
    harness, toys = build(1)
    tokens = sequences()
    run(harness, toys, hx.prefill_plan("a", 14) + hx.step({"a": 1}), tokens)
    with pytest.raises(hx.HarnessError, match="multiple of 4"):
        harness.add_request("b", resume_from="a", prefix_len=6)
    with pytest.raises(hx.HarnessError, match="not computed"):
        harness.add_request("b", resume_from="a", prefix_len=16)
    with pytest.raises(hx.HarnessError, match="exists"):
        harness.add_request("a")
    # The donor's second step freed the conv window of a 4-token prefix.
    with pytest.raises(hx.HarnessError, match="ran past the prefix"):
        harness.add_request("b", resume_from="a", prefix_len=4)
    with pytest.raises(ValueError, match="once per step"):
        harness.step([("a", 1), ("a", 1)], None)
    with pytest.raises(hx.HarnessError, match="already"):
        harness.bind()


def test_mini_geometry_at_tp2():
    """The released config at block size 32: 96 owners per rank, conv blocks
    of 4 tokens, and a conv cache allocated with larger blocks."""
    config = text_config(MINI)
    tokens = sequences()
    harness, toys = build(2, config, block_size=32, num_blocks=8)
    got = run(harness, toys, hx.decode_plan("a", LEN_A, 20), tokens)
    assert torch.equal(got["a"], expected(toys[0], 2, tokens["a"]))
    assert harness.prefix_alignment == 32
    conv = toys[0].owners.conv[0, 0]
    assert conv.kv_cache.shape == (8, 4, 4, 2048) and conv.cache_block_size == 4

    harness, toys = build(1, block_size=4, conv_block_size=8)
    assert toys[0].owners.conv[0, 0].cache_block_size == 8
    got = run(harness, toys, hx.chunked_plan("a", LEN_A, 3), tokens)
    assert torch.equal(got["a"], expected(toys[0], 1, tokens["a"]))
