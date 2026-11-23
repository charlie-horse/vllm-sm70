# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2: when a DSpark engine speculates (inkling2_sm70/spec_policy.py).

CPU only, no model. The batch-size schedule the engine-config hook sets
from ``sm70.inkling2``, as v0.33's scheduler reads it (its dense lookup);
the decode steps the cache builders treat as decode-only under it; the
draft's graph manager built without it; and the inherited propose, which
for a step that verifies no drafts stores the draft's context K/V and runs
no block pass. The hook itself: test_m12_inkling2_config_cpu.py; the
target and draft across switching steps: test_m12_inkling2_verify_graph_cpu.py
and test_m12_inkling2_dspark_cpu.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_sm70.config import Inkling2Config
from vllm_sm70.models.inkling2_sm70 import spec_policy

K = 7
Q = K + 1


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.delenv(spec_policy.SWITCH, raising=False)


@pytest.mark.parametrize(
    ("max_num_seqs", "max_reqs", "tokens", "want"),
    [
        (4, 1, 7, [(1, 1, 7), (2, 4, 0)]),
        (8, 2, 7, [(1, 2, 7), (3, 8, 0)]),
        (4, 1, 5, [(1, 1, 5), (2, 4, 0)]),
        (4, 4, 7, None),  # every step speculates
        (4, 9, 7, None),
        (4, 4, 5, [(1, 4, 5)]),
        (1, 1, 7, None),
        (None, 2, 7, [(1, 2, 7), (3, 3, 0)]),
    ],
)
def test_schedule(max_num_seqs, max_reqs, tokens, want):
    assert spec_policy.schedule(K, max_num_seqs, max_reqs, tokens) == want


@pytest.mark.parametrize("tokens", [0, K + 1])
def test_schedule_refuses_verifying_more_than_the_drafts(tokens):
    with pytest.raises(ValueError, match="dspark_spec_tokens"):
        spec_policy.schedule(K, 4, 1, tokens)


@pytest.mark.parametrize(
    ("max_num_seqs", "max_reqs", "tokens", "want"),
    [
        (4, 1, 7, [0, 7, 0, 0, 0]),
        (8, 3, 5, [0, 5, 5, 5, 0, 0, 0, 0, 0]),
        (2, 2, 5, [0, 5, 5]),
    ],
)
def test_the_scheduler_reads_k_by_scheduled_requests(
    max_num_seqs, max_reqs, tokens, want
):
    """v0.33's dense lookup (UP v1/spec_decode/dynamic/utils.py), which the
    scheduler indexes by the scheduled requests of a step."""
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    schedule = spec_policy.schedule(K, max_num_seqs, max_reqs, tokens)
    assert build_dynamic_sd_schedule_lookup(schedule, max_num_seqs, K) == want


def _engine(schedule=None, max_num_seqs=4):
    spec = SimpleNamespace(
        num_speculative_tokens=K,
        parallel_drafting=False,
        num_speculative_tokens_per_batch_size=schedule,
    )
    return SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        use_v2_model_runner=True,
    )


def test_apply_sets_the_default_schedule():
    config = _engine()
    want = [(1, 1, K), (2, 4, 0)]
    assert spec_policy.apply(config, Inkling2Config()) == want
    assert config.speculative_config.num_speculative_tokens_per_batch_size == want


def test_apply_keeps_a_given_schedule_and_obeys_the_switch(monkeypatch):
    given = [(1, 2, 3), (3, 4, 0)]
    config = _engine(given)
    assert spec_policy.apply(config, Inkling2Config(dspark_spec_tokens=5)) == given
    assert config.speculative_config.num_speculative_tokens_per_batch_size is given
    monkeypatch.setenv(spec_policy.SWITCH, "0")
    config = _engine()
    assert spec_policy.apply(config, Inkling2Config()) is None
    assert config.speculative_config.num_speculative_tokens_per_batch_size is None
    assert spec_policy.apply(SimpleNamespace(speculative_config=None), None) is None


# --- decode-only steps under a schedule ---------------------------------------


def _common(query_lens, pad_reqs=0, pad_tokens=0):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    lens = list(query_lens) + [0] * pad_reqs
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    num_tokens = int(qsl[-1]) + pad_tokens
    seq_lens = torch.tensor([40 + n for n in query_lens] + [0] * pad_reqs)
    return CommonAttentionMetadata(
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens.int(),
        num_reqs=len(lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=torch.zeros(len(lens), 4, dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
    )


def _graph_engine(schedule):
    from vllm.config.compilation import CUDAGraphMode

    config = _engine(schedule)
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=64
    )
    config.model_config = SimpleNamespace(enforce_eager=False)
    return config


def test_decode_query_lens():
    from vllm_sm70.models.inkling2_sm70.backend import decode_query_lens

    assert decode_query_lens(_engine()) == (Q,)
    assert decode_query_lens(_engine([(1, 1, K), (2, 4, 0)])) == (1, Q)
    assert decode_query_lens(_engine([(1, 1, 5), (2, 4, 0)])) == (1, 6)
    # Counts above num_speculative_tokens are cut to it, as upstream does.
    assert decode_query_lens(_engine([(1, 2, 9), (3, 4, 3)])) == (4, Q)
    assert decode_query_lens(SimpleNamespace(speculative_config=None)) == (1,)


@pytest.mark.parametrize(
    ("schedule", "query_lens", "pad_reqs", "want"),
    [
        ([(1, 1, K), (2, 4, 0)], [1, 1], 2, True),
        ([(1, 1, K), (2, 4, 0)], [Q], 3, True),
        ([(1, 1, K), (2, 4, 0)], [Q, 1], 0, False),
        ([(1, 1, K), (2, 4, 0)], [6], 0, False),
        ([(1, 1, 5), (2, 4, 0)], [6], 1, True),
        ([(1, 1, 5), (2, 4, 0)], [Q], 0, False),
        ([(1, 1, 5), (2, 4, 0)], [1] * 65, 0, False),  # above the capture cap
        (None, [1, 1], 0, False),  # static speculation: no graph of q = 1
    ],
)
def test_steps_of_each_tier_are_decode_only(schedule, query_lens, pad_reqs, want):
    from vllm_sm70.models.inkling2_sm70.backend import replays_decode_graph

    q = max(query_lens)
    cm = _common(query_lens, pad_reqs=pad_reqs, pad_tokens=pad_reqs * q)
    assert replays_decode_graph(_graph_engine(schedule), cm) is want


# --- the speculator -----------------------------------------------------------


def test_the_draft_graph_manager_is_built_without_the_schedule():
    """Every draft step runs the whole block: v0.33's graph manager would
    otherwise capture the schedule's tiers as draft query lengths."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    from vllm_sm70.models.inkling2_sm70.speculator import Inkling2DSparkSm70Speculator

    schedule = [(1, 1, K), (2, 4, 0)]
    spec = Inkling2DSparkSm70Speculator.__new__(Inkling2DSparkSm70Speculator)
    spec.vllm_config = _engine(schedule)
    seen = []

    def init(self, mode):
        seen.append(
            self.vllm_config.speculative_config.num_speculative_tokens_per_batch_size
        )
        if mode == "boom":
            raise RuntimeError(mode)

    with mock.patch.object(DFlashSpeculator, "init_cudagraph_manager", init):
        spec.init_cudagraph_manager("full")
        with pytest.raises(RuntimeError):
            spec.init_cudagraph_manager("boom")
    assert seen == [None, None]
    held = spec.vllm_config.speculative_config.num_speculative_tokens_per_batch_size
    assert held is schedule


def test_a_step_without_drafts_stores_the_context_only(monkeypatch):
    """The inherited propose (UP v1/worker/gpu/spec_decode/dflash/
    speculator.py:546-553) with num_speculative_tokens 0: the target's
    features and the draft's context K/V of every scheduled token are
    stored; f_rnn, the block pass and its graph are not touched."""
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    from vllm_sm70.models.inkling2_sm70.speculator import Inkling2DSparkSm70Speculator

    assert Inkling2DSparkSm70Speculator.propose is DFlashSpeculator.propose
    prepared = []
    monkeypatch.setattr(
        dflash, "prepare_dflash_inputs", lambda *args: prepared.append(args)
    )
    fake = mock.MagicMock()
    fake.pcp_manager = None
    fake.draft_kv_cache_group_id = 0
    fake.draft_kv_cache_group_ids = [0]
    fake.num_query_per_req = Q
    fake.max_model_len = 1024
    fake.draft_tokens = torch.zeros(4, K, dtype=torch.int64)
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=2,
        seq_lens_cpu_upper_bound=torch.tensor([40, 50, 0, 0]),
    )
    hidden, aux = torch.zeros(2, 4), [torch.zeros(2, 4)]
    out = Inkling2DSparkSm70Speculator.propose(
        fake,
        batch,
        {},
        {},
        hidden,
        aux,
        torch.ones(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.int64),
        num_speculative_tokens=0,
    )
    assert out.shape == (2, K)
    assert len(prepared) == 1
    fake._store_hidden_states.assert_called_once_with(2, hidden, aux)
    fake._precompute_context_kv.assert_called_once_with(0, 2, False)
    fake.prepare_context_anchor.assert_not_called()
    fake._generate_draft.assert_not_called()
    fake.query_cudagraph_manager.run_fullgraph.assert_not_called()


# --- the graphs v0.33's managers capture under the hook's sizes ---------------


def _manager(sizes, schedule, max_num_seqs):
    """v0.33's CudaGraphManager of a query length of Q, as the target's (with
    the schedule) or the draft's (the speculator hides it) is built."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as cu

    spec = SimpleNamespace(num_speculative_tokens_per_batch_size=schedule)
    spec.uses_dynamic_speculative_decoding = lambda: bool(
        spec.num_speculative_tokens_per_batch_size
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=sizes[-1]
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        speculative_config=spec,
        num_speculative_tokens=K,
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
    )
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    with (
        mock.patch.object(cu, "get_pp_group", lambda: pp),
        mock.patch.object(cu.current_platform, "get_global_graph_pool", lambda: None),
    ):
        manager = cu.CudaGraphManager(
            config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, Q
        )
    return {
        (d.uniform_token_count, d.num_reqs)
        for descs in manager._capture_descs.values()
        for d in descs
    }


@pytest.mark.parametrize(
    ("max_num_seqs", "max_reqs", "tokens"),
    [(4, 1, 7), (8, 2, 7), (4, 1, 5), (4, 4, 5), (16, 3, 6)],
)
def test_every_decode_step_has_a_graph(max_num_seqs, max_reqs, tokens):
    """Each step of n requests gets a target graph of its q and at least n
    requests, and each drafting step a draft graph of the whole block; the
    draft has no graph of another query length."""
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    from vllm_sm70.models.inkling2_sm70.config import (
        DECODE_GRAPH_MAX_BATCH,
        _schedule_capture_sizes,
    )

    schedule = spec_policy.schedule(K, max_num_seqs, max_reqs, tokens)
    reqs = min(max_num_seqs, DECODE_GRAPH_MAX_BATCH)
    sizes = _schedule_capture_sizes(schedule, Q, reqs)
    target = _manager(sizes, schedule, max_num_seqs)
    draft = _manager(sizes, None, max_num_seqs)
    lookup = build_dynamic_sd_schedule_lookup(schedule, max_num_seqs, K)
    for n in range(1, reqs + 1):
        q = 1 + lookup[n]
        assert any(tq == q and tn >= n for tq, tn in target), (n, q, target)
        if lookup[n]:
            assert any(dn >= n for _, dn in draft), (n, draft)
    assert {q for q, _ in draft} == {Q}
    assert {q for q, _ in target} == {1 + lookup[n] for n in range(1, reqs + 1)}
