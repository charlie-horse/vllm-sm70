# SPDX-License-Identifier: Apache-2.0
"""CPU checks: the greedy vocab-parallel top-1 fast path on the sampler hook.

* Identity: on a fake 2-rank vocab split (real ParallelLMHead shards, the
  real LogitsProcessor, a stand-in all-gather), every rank's token equals
  ``argmax`` over the gathered, trimmed logits, as upstream's
  ``Sampler.greedy_sample`` returns on CUDA: random logits, exact ties
  across and inside ranks, a padded last shard whose padding would
  otherwise win, and the Flint TP2 vocabulary.
* The hook: all-greedy batches take the greedy route; every case upstream's
  Sampler would do more than argmax for returns None; switch=0 (env or
  Sm70Config) restores the upstream path, and with VLLM_SM70_FLINT_SAMPLER=0
  too there is no hook at all.
No GPU and no built _sm70_C: a fake "cuda:0" that reports sm_70 (the
capability query is patched) drives the admission branches.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_sm70 import sample, sample_greedy, sample_v1
from vllm_sm70.sample_v1 import GREEDY, TOPK, Sm70HiddenStateSampler

GREEDY_SWITCH = "VLLM_SM70_GREEDY_TOKEN_FASTPATH"
SAMPLER_SWITCH = "VLLM_SM70_FLINT_SAMPLER"
FLINT_VOCAB = 150144


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


requires_vllm = pytest.mark.skipif(
    not _have("vllm.v1.outputs"), reason="vLLM Python source not importable"
)


# ------------------------------------------------- real LM head, fake 2-rank TP


@pytest.fixture
def vllm_env(monkeypatch):
    """What LogitsProcessor and ParallelLMHead read, without an engine: a
    current config with only head_dtype, and a stand-in TP group."""
    if not _have("vllm.model_executor.layers.logits_processor"):
        pytest.skip("vLLM Python source not importable")
    import vllm.config.vllm as vc
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(
        vc,
        "_current_vllm_config",
        SimpleNamespace(model_config=SimpleNamespace(head_dtype=None)),
    )

    def set_tp(world_size, rank):
        monkeypatch.setattr(
            ps, "_TP", SimpleNamespace(world_size=world_size, rank_in_group=rank)
        )

    set_tp(1, 0)
    return set_tp


def _shards(vllm_env, weight, tp_size, *, poison_padding=False):
    """One ParallelLMHead per rank, loaded from the full [vocab, hidden]
    weight as the model loader does (padding rows zero-filled)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    vocab, hidden = weight.shape
    heads = []
    for rank in range(tp_size):
        vllm_env(tp_size, rank)
        head = ParallelLMHead(vocab, hidden, params_dtype=weight.dtype)
        head.weight_loader(head.weight, weight)
        pad = head.shard_indices.num_org_vocab_padding
        if poison_padding and pad:
            # Padding rows that would win the argmax if they were not masked.
            head.weight.data[-pad:] = 64.0
        heads.append(head)
    return heads


def _logits_processor(vocab):
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    return LogitsProcessor(vocab)


def _reference(lp, heads, hidden):
    """Upstream's full path: per-rank GEMM, all-gather along vocab, trim to
    org_vocab_size, FP32, argmax (Sampler.greedy_sample on CUDA)."""
    parts = [lp._apply_head(h, hidden, None) for h in heads]
    logits = torch.cat(parts, dim=-1)[..., : lp.org_vocab_size]
    return logits.to(torch.float32).argmax(dim=-1).to(torch.int32)


def _fast_path(monkeypatch, vllm_env, lp, heads, hidden):
    """Every rank's fast-path token, with a stand-in for the pair all-gather."""
    import vllm.model_executor.layers.logits_processor as lp_mod

    tp_size = len(heads)
    pairs: list[torch.Tensor] = []

    def record(t, dim=-1):
        pairs.append(t.clone())
        return torch.cat([t] * tp_size, dim=dim)

    monkeypatch.setattr(lp_mod, "tensor_model_parallel_all_gather", record)
    for rank, head in enumerate(heads):
        vllm_env(tp_size, rank)
        sample_greedy.greedy_tokens(lp, head, hidden)
    assert len(pairs) == (tp_size if tp_size > 1 else 0)
    assert all(p.dtype == torch.float32 and p.shape[-1] == 2 for p in pairs)

    def exchange(t, dim=-1):
        return torch.cat(pairs, dim=dim)

    monkeypatch.setattr(lp_mod, "tensor_model_parallel_all_gather", exchange)
    out = []
    for rank, head in enumerate(heads):
        vllm_env(tp_size, rank)
        tokens = sample_greedy.greedy_tokens(lp, head, hidden)
        assert tokens.dtype == torch.int32
        out.append(tokens)
    return out


def _check(monkeypatch, vllm_env, weight, hidden, tp_size=2, **kw):
    heads = _shards(vllm_env, weight, tp_size, **kw)
    lp = _logits_processor(weight.shape[0])
    for head in heads:
        assert sample_greedy.lm_head_reject_reason(lp, head) is None
    want = _reference(lp, heads, hidden)
    for got in _fast_path(monkeypatch, vllm_env, lp, heads, hidden):
        assert torch.equal(got, want), (got, want)
    return want


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("vocab", [512, 509, 1000])
@pytest.mark.parametrize("rows", [1, 2, 5])
def test_two_rank_split_equals_gathered_argmax(
    monkeypatch, vllm_env, dtype, vocab, rows
):
    g = torch.Generator().manual_seed(vocab * 10 + rows)
    weight = torch.randn(vocab, 32, generator=g).to(dtype)
    hidden = torch.randn(rows, 32, generator=g).to(dtype)
    _check(monkeypatch, vllm_env, weight, hidden)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tp_sizes(monkeypatch, vllm_env, tp_size):
    g = torch.Generator().manual_seed(tp_size)
    weight = torch.randn(777, 16, generator=g)
    hidden = torch.randn(3, 16, generator=g)
    _check(monkeypatch, vllm_env, weight, hidden, tp_size=tp_size)


def test_masked_padding_never_wins(monkeypatch, vllm_env):
    # 509 pads to 512 (64-aligned): rank 1 holds 253 tokens and 3 padding
    # rows, poisoned to dominate. Upstream trims them; the fast path masks them.
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(509, 16, generator=g)
    hidden = torch.rand(4, 16, generator=g) + 0.5  # positive: poison wins unmasked
    want = _check(
        monkeypatch, vllm_env, weight, hidden, poison_padding=True
    )
    assert int(want.max()) < 509


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ties_take_the_first_index(monkeypatch, vllm_env, dtype):
    """Exact ties at the maximum: across ranks (rank 0 wins) and inside a
    rank (the lower id wins), as argmax over the gathered row."""
    g = torch.Generator().manual_seed(1)
    vocab, hidden_size = 512, 16
    weight = torch.randn(vocab, hidden_size, generator=g) * 0.1
    hidden = torch.ones(3, hidden_size)
    top = torch.full((hidden_size,), 2.0)
    # row 0: tie across ranks 300 (rank 1) and 100 (rank 0)
    # rows 1-2: also a tie inside rank 1 (400 and 260), below the first pair
    for t in (100, 300):
        weight[t] = top
    hidden[1:, 0] = -50.0  # makes rows 1-2 prefer the next pair
    for t in (260, 400):
        weight[t] = top
        weight[t, 0] = -3.0
    weight = weight.to(dtype)
    hidden = hidden.to(dtype)
    want = _check(monkeypatch, vllm_env, weight, hidden)
    assert want.tolist() == [100, 260, 260]


def test_flint_tp2_vocabulary(monkeypatch, vllm_env):
    """Flint's vocabulary splits evenly at TP2 (75072 per rank, no padding)."""
    g = torch.Generator().manual_seed(2)
    weight = torch.randn(FLINT_VOCAB, 8, generator=g).to(torch.float16)
    hidden = torch.randn(2, 8, generator=g).to(torch.float16)
    weight[FLINT_VOCAB - 1] = 40.0  # last token of the last rank
    hidden[1] = 1.0
    want = _check(monkeypatch, vllm_env, weight, hidden)
    assert int(want[1]) == FLINT_VOCAB - 1


# ----------------------------------------------------------- LM-head admission


def _lp(**kw):
    fields = dict(
        logits_as_input=False,
        soft_cap=None,
        scale=1.0,
        org_vocab_size=512,
        get_top_tokens=lambda *a, **k: None,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def _head(**kw):
    fields = dict(
        num_added_embeddings=0,
        org_vocab_size=512,
        org_vocab_size_padded=512,
        shard_indices=object(),
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "want,lp,head",
    [
        (None, _lp(), _head()),
        (None, _lp(soft_cap=30.0, scale=0.5), _head()),  # same per-element ops
        ("logits_as_input", _lp(logits_as_input=True), _head()),
        ("logits_processor", _lp(get_top_tokens=None), _head()),
        ("scale", _lp(scale=0.0), _head()),
        ("scale", _lp(scale=-1.0), _head()),
        ("added_vocab", _lp(), _head(num_added_embeddings=16)),
        ("vocab_size", _lp(org_vocab_size=500), _head()),
        (
            "vocab_size",
            _lp(org_vocab_size=1 << 25),
            _head(org_vocab_size=1 << 25, org_vocab_size_padded=1 << 25),
        ),
        ("lm_head", _lp(), _head(shard_indices=None)),
    ],
)
def test_lm_head_reject_reason(want, lp, head):
    assert sample_greedy.lm_head_reject_reason(lp, head) == want


def test_device_reject_reason(monkeypatch):
    assert sample_greedy.device_reject_reason(None) == "non_cuda"
    assert sample_greedy.device_reject_reason(torch.device("cpu")) == "non_cuda"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    assert sample_greedy.device_reject_reason(torch.device("cuda", 0)) == "non_sm70"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 0))
    assert sample_greedy.device_reject_reason(torch.device("cuda", 0)) is None


# ---------------------------------------------------------- the sampler hook


def _metadata(rows=1, **overrides):
    fields = dict(
        all_random=False,
        all_greedy=True,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=None,
        top_k=None,
        top_p=None,
        generators={},
        logitsprocs=SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
        thinking_budget_state_holder=None,
        output_token_ids=[],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _Runner:
    def __init__(self, rows=1):
        import numpy as np

        self.device = torch.device("cuda", 0)
        top = np.full(64, 0, dtype=np.int32)
        self.input_batch = SimpleNamespace(num_reqs=rows, top_k_cpu=top)


def _model(opt_in=True):
    m = SimpleNamespace(lm_head=_head(), logits_processor=_lp())
    if opt_in:
        m.supports_sm70_topk_sampler = True
    return m


def _sched(structured=False):
    return SimpleNamespace(has_structured_output_requests=structured)


@pytest.fixture
def fake_sm70(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(sample, "ops_available", lambda: True)
    monkeypatch.setattr(sample, "prepare", lambda device: None)
    monkeypatch.delenv(GREEDY_SWITCH, raising=False)
    monkeypatch.delenv(SAMPLER_SWITCH, raising=False)
    sample._is_sm70.cache_clear()
    yield
    sample._is_sm70.cache_clear()


@pytest.fixture
def hook(fake_sm70):
    def make(rows=1, vllm_config=None, **kw):
        h = Sm70HiddenStateSampler(**kw)
        runner = _Runner(rows)
        h.prepare(runner, vllm_config)
        h._test_runner = runner  # keep the weakref alive
        return h

    return make


@pytest.fixture
def greedy_calls(monkeypatch):
    """Replace the vocab-parallel top-1 with a recorder."""
    calls = []

    def fake(logits_processor, lm_head, hidden):
        calls.append(hidden.shape[0])
        return torch.arange(hidden.shape[0], dtype=torch.int32) + 7

    monkeypatch.setattr(sample_greedy, "greedy_tokens", fake)
    return calls


@pytest.fixture
def topk_calls(monkeypatch):
    calls = []

    def fake(logits_processor, lm_head, hidden, metadata, *, vocab_size):
        calls.append(hidden.shape[0])
        return torch.zeros(hidden.shape[0], dtype=torch.int32)

    monkeypatch.setattr(sample, "sample_lm_head_candidates", fake)
    return calls


def _try(h, rows=1, model=None, metadata=None, sched=None, spec=None):
    return h.try_sample(
        model or _model(),
        torch.empty(rows, 8, dtype=torch.float16),
        metadata or _metadata(rows),
        sched or _sched(),
        spec,
    )


@requires_vllm
@pytest.mark.parametrize("rows", [1, 2, 8, 20])
def test_all_greedy_batch_takes_the_fast_path(hook, greedy_calls, topk_calls, rows):
    from vllm.v1.outputs import SamplerOutput

    out = _try(hook(rows=rows), rows=rows)
    assert isinstance(out, SamplerOutput)
    assert out.logprobs_tensors is None
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.sampled_token_ids.shape == (rows, 1)
    assert out.sampled_token_ids[:, 0].tolist() == [7 + i for i in range(rows)]
    assert greedy_calls == [rows] and topk_calls == []


@requires_vllm
def test_inactive_processors_and_argmax_invariant_ones_are_admitted(
    hook, greedy_calls
):
    # min-p / top-k / top-p cannot change a greedy token; empty min-tokens
    # and logit-bias processors do nothing.
    procs = SimpleNamespace(
        argmax_invariant=[object()],
        non_argmax_invariant=[SimpleNamespace(min_toks={}), SimpleNamespace(biases={})],
    )
    assert _try(hook(), metadata=_metadata(logitsprocs=procs)) is not None
    assert greedy_calls == [1]


@requires_vllm
def test_random_batches_still_go_to_k16(hook, greedy_calls, topk_calls):
    meta = _metadata(
        all_random=True,
        all_greedy=False,
        temperature=torch.ones(1),
        top_k=torch.full((1,), 20, dtype=torch.int32),
        top_p=torch.full((1,), 0.95),
    )
    h = hook()
    h._test_runner.input_batch.top_k_cpu[0] = 20
    assert h.route(meta) == TOPK
    assert _try(h, metadata=meta) is not None
    assert greedy_calls == [] and topk_calls == [1]


def _thinking(active):
    return SimpleNamespace(has_tracked_requests=lambda: active)


NONE_CASES = {
    "model_not_opted_in": dict(model=_model(opt_in=False)),
    "spec_decode": dict(spec=object()),
    "structured_output": dict(sched=_sched(structured=True)),
    "thinking_budget": dict(
        metadata=_metadata(thinking_budget_state_holder=_thinking(True))
    ),
    "min_tokens_active": dict(
        metadata=_metadata(
            logitsprocs=SimpleNamespace(
                argmax_invariant=[],
                non_argmax_invariant=[
                    SimpleNamespace(min_toks={0: (3, [], set(), False)})
                ],
            )
        )
    ),
    "logit_bias_active": dict(
        metadata=_metadata(
            logitsprocs=SimpleNamespace(
                argmax_invariant=[],
                non_argmax_invariant=[SimpleNamespace(biases={0: {5: 1.0}})],
            )
        )
    ),
    "custom_logits_processor": dict(
        metadata=_metadata(
            logitsprocs=SimpleNamespace(
                argmax_invariant=[], non_argmax_invariant=[object()]
            )
        )
    ),
    "logprobs": dict(metadata=_metadata(max_num_logprobs=1)),
    "logprob_token_ids": dict(metadata=_metadata(logprob_token_ids={0: [1]})),
    "penalties": dict(metadata=_metadata(no_penalties=False)),
    "allowed_token_ids": dict(metadata=_metadata(allowed_token_ids_mask=object())),
    "bad_words": dict(metadata=_metadata(bad_words_token_ids={0: [[1]]})),
    "lm_head_added_vocab": dict(
        model=SimpleNamespace(
            supports_sm70_topk_sampler=True,
            lm_head=_head(num_added_embeddings=8),
            logits_processor=_lp(),
        )
    ),
}


@pytest.mark.parametrize("case", sorted(NONE_CASES))
def test_upstream_path_cases(hook, greedy_calls, topk_calls, case):
    assert _try(hook(), **NONE_CASES[case]) is None
    assert greedy_calls == [] and topk_calls == []


def test_hidden_rows_must_match_batch(hook, greedy_calls):
    assert _try(hook(rows=1), rows=2, metadata=_metadata(1)) is None
    assert greedy_calls == []


def test_capture_returns_none(hook, greedy_calls, monkeypatch):
    h = hook()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _try(h) is None
    assert greedy_calls == []


def test_non_sm70_returns_none(hook, greedy_calls, monkeypatch):
    h = hook()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert _try(h) is None
    assert greedy_calls == []


@requires_vllm
def test_fp64_gumbel_does_not_matter_for_greedy(hook, greedy_calls):
    h = hook(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(use_fp64_gumbel=True))
    )
    assert _try(h) is not None
    assert greedy_calls == [1]


@requires_vllm
def test_missing_k16_ops_do_not_matter_for_greedy(hook, greedy_calls, monkeypatch):
    h = hook()
    monkeypatch.setattr(sample, "ops_available", lambda: False)
    assert _try(h) is not None


def test_rejections_are_logged_once_per_route(hook, monkeypatch):
    lines = []
    monkeypatch.setattr(
        sample_v1.logger, "info", lambda msg, *args: lines.append(msg % args)
    )
    h = hook()
    for _ in range(3):
        _try(h, metadata=_metadata(max_num_logprobs=1))
    assert lines == ["SM70 greedy top-1 fast path rejected: logprobs"]


@pytest.mark.parametrize(
    "want,kw",
    [
        ("spec_decode", dict(spec_decode_metadata=object())),
        ("logprobs", dict(sampling_metadata=_metadata(max_num_logprobs=0))),
        ("penalties", dict(sampling_metadata=_metadata(no_penalties=False))),
        ("model", dict(model=_model(opt_in=False))),
        (None, {}),
    ],
)
def test_reject_reasons_name_each_case(hook, want, kw):
    args = dict(
        model=_model(),
        sample_hidden_states=torch.empty(1, 8, dtype=torch.float16),
        sampling_metadata=_metadata(),
        scheduler_output=_sched(),
        spec_decode_metadata=None,
    )
    args.update(kw)
    assert hook().reject_reason(**args) == want


@requires_vllm
def test_hook_end_to_end_equals_compute_logits_argmax(fake_sm70, vllm_env):
    """Real LogitsProcessor and ParallelLMHead (TP1) behind the hook: the
    token equals argmax(compute_logits) for each row."""
    g = torch.Generator().manual_seed(3)
    weight = torch.randn(509, 16, generator=g)
    (head,) = _shards(vllm_env, weight, 1)
    lp = _logits_processor(509)
    model = SimpleNamespace(
        supports_sm70_topk_sampler=True, lm_head=head, logits_processor=lp
    )
    hidden = torch.randn(3, 16, generator=g)
    h = Sm70HiddenStateSampler()
    runner = _Runner(3)
    h.prepare(runner, None)
    out = h.try_sample(model, hidden, _metadata(3), _sched(), None)
    want = lp(head, hidden).argmax(dim=-1).to(torch.int32)
    assert torch.equal(out.sampled_token_ids[:, 0], want)


# --------------------------------------------------- switch = 0 / Sm70Config


def test_switch_zero_restores_the_upstream_path(hook, greedy_calls, monkeypatch):
    monkeypatch.setenv(GREEDY_SWITCH, "0")
    h = hook()
    assert not h.greedy_enabled
    # The all-greedy batch now goes to the top-k sampler route, which rejects
    # it (temperature 0) -> None: compute_logits + the upstream Sampler.
    assert h.route(_metadata()) == TOPK
    assert h.reject_reason(_model(), torch.empty(1, 8), _metadata(), _sched(), None)
    assert _try(h) is None
    assert greedy_calls == []


def test_non_zero_values_keep_it_on(hook, monkeypatch):
    for value in ("", "1", "false", "00"):
        monkeypatch.setenv(GREEDY_SWITCH, value)
        assert hook().greedy_enabled


def test_sm70_config_can_turn_it_off(hook, greedy_calls):
    from vllm_sm70.config import clear_cache

    clear_cache()
    vc = SimpleNamespace(
        additional_config={"sm70": {"greedy_token_fastpath": False}},
        kernel_config=None,
        model_config=None,
    )
    h = hook(vllm_config=vc)
    assert not h.greedy_enabled and h.route(_metadata()) == TOPK
    assert _try(h) is None and greedy_calls == []
    clear_cache()


def test_constructor_flag_off_stays_off(hook, greedy_calls):
    h = hook(greedy=False)
    assert not h.greedy_enabled
    assert _try(h) is None and greedy_calls == []


def test_k16_off_greedy_on_has_no_topk_route(hook, greedy_calls, topk_calls):
    h = hook(topk=False)
    random = _metadata(all_greedy=False, all_random=True)
    assert h.route(random) is None
    assert h.reject_reason(_model(), torch.empty(1, 8), random, _sched(), None) == (
        "disabled"
    )
    assert _try(h, metadata=random) is None and topk_calls == []


def test_registered_in_sampler_section_and_sm70_config():
    from vllm_sm70 import envs
    from vllm_sm70.config import GRAPH_FIELDS, KILL_SWITCHES, Sm70Config

    assert GREEDY_SWITCH in envs._SECTIONS["sampler"]
    assert KILL_SWITCHES[GREEDY_SWITCH] == ("greedy_token_fastpath", False)
    assert Sm70Config().greedy_token_fastpath is True
    assert "greedy_token_fastpath" not in GRAPH_FIELDS


@pytest.fixture
def platform_cls(monkeypatch):
    if not _have("vllm.platforms.cuda"):
        pytest.skip("vLLM Python source not importable")
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    from vllm_sm70 import platform

    return platform.Sm70CudaPlatform


@pytest.mark.parametrize(
    "k16,greedy,hook_exists",
    [(None, None, True), ("0", None, True), (None, "0", True), ("0", "0", False)],
)
def test_platform_getter(platform_cls, monkeypatch, k16, greedy, hook_exists):
    for name, value in ((SAMPLER_SWITCH, k16), (GREEDY_SWITCH, greedy)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    h = platform_cls.get_hidden_state_sampler()
    assert (h is not None) is hook_exists
    if h is not None:
        assert h._topk is (k16 is None)
        assert h.greedy_enabled is (greedy is None)


@pytest.mark.parametrize(
    "k16,greedy,prepared",
    [(None, None, True), ("0", None, True), (None, "0", True), ("0", "0", False)],
)
def test_kernel_warmup_prepares_when_either_is_on(
    platform_cls, monkeypatch, k16, greedy, prepared
):
    from vllm_sm70 import warmup

    for name, value in ((SAMPLER_SWITCH, k16), (GREEDY_SWITCH, greedy)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    seen = []
    monkeypatch.setattr(warmup, "run", lambda worker: None)
    monkeypatch.setattr(sample_v1, "prepare_worker", seen.append)
    worker = SimpleNamespace()
    assert platform_cls.kernel_warmup(worker) is None
    assert seen == ([worker] if prepared else [])


def test_greedy_only_hook_does_not_create_the_k16_counter(fake_sm70, monkeypatch):
    created = []
    monkeypatch.setattr(sample, "prepare", lambda device: created.append(device))
    h = Sm70HiddenStateSampler(topk=False)
    runner = _Runner()
    h.prepare(runner, None)
    assert h.prepared and created == []
    Sm70HiddenStateSampler().prepare(runner, None)
    assert created == [torch.device("cuda", 0)]
