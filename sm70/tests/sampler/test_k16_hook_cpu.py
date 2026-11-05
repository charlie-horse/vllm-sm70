# SPDX-License-Identifier: Apache-2.0
"""K16 CPU checks: the U4 hook consumer, the platform wiring, the shape
guards for Flint TP2, R4, and the op bindings (no GPU, no built _sm70_C).

The hook tests drive Sm70HiddenStateSampler.try_sample with stand-in runner,
model and metadata objects. A fake "cuda:0" device that reports sm_70 (the
capability query is patched) lets every admission branch run on the CPU;
the kernel call itself is replaced where a case is admitted.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pathlib
import re
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
SM70 = HERE.parents[1]
CSRC = SM70 / "csrc" / "sampler"
OPS_PY = SM70 / "vllm_sm70" / "ops" / "sampler.py"
sys.path.insert(0, str(HERE.parent / "ops"))
from _schemas import sm70_schemas
from vllm_sm70 import sample, sample_v1
from vllm_sm70.sample_v1 import Sm70HiddenStateSampler

EXPECTED_OPS = {"sm70_local_top32_half", "sm70_sample_topk_candidates"}
VOCAB = 150144  # Flint


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


requires_vllm = pytest.mark.skipif(
    not _have("vllm.v1.outputs"), reason="vLLM Python source not importable"
)


# ---------------------------------------------------------------- stand-ins


def _metadata(rows=1, **overrides):
    fields = dict(
        all_random=True,
        all_greedy=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=torch.ones(rows),
        top_k=torch.full((rows,), 20, dtype=torch.int32),
        top_p=torch.full((rows,), 0.95),
        generators={},
        logitsprocs=SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
        thinking_budget_state_holder=None,
        output_token_ids=[],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _Runner:
    def __init__(self, rows=1, top_k=20):
        import numpy as np

        self.device = torch.device("cuda", 0)
        top = np.full(8, VOCAB, dtype=np.int32)
        top[:rows] = top_k
        self.input_batch = SimpleNamespace(num_reqs=rows, top_k_cpu=top)


def _model(opt_in=True):
    m = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(0, dtype=torch.float16)),
        logits_processor=SimpleNamespace(org_vocab_size=VOCAB),
    )
    if opt_in:
        m.supports_sm70_topk_sampler = True
    return m


def _sched(structured=False):
    return SimpleNamespace(has_structured_output_requests=structured)


@pytest.fixture
def fake_sm70(monkeypatch):
    """A 'cuda:0' that reports sm_70 with the K16 ops present; no capture."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(sample, "ops_available", lambda: True)
    monkeypatch.setattr(sample, "prepare", lambda device: None)
    sample._is_sm70.cache_clear()
    yield
    sample._is_sm70.cache_clear()


@pytest.fixture
def hook(fake_sm70):
    def make(rows=1, top_k=20, fp64=False):
        h = Sm70HiddenStateSampler()
        runner = _Runner(rows, top_k)
        h.prepare(
            runner,
            SimpleNamespace(model_config=SimpleNamespace(use_fp64_gumbel=fp64)),
        )
        h._test_runner = runner  # keep the weakref alive
        return h

    return make


@pytest.fixture
def kernel(monkeypatch):
    """Replace the device sampling with a recorder returning token ids."""
    calls = []

    def fake(logits_processor, lm_head, hidden, metadata, *, vocab_size):
        calls.append((hidden.shape[0], vocab_size))
        return torch.arange(hidden.shape[0], dtype=torch.int32)

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


# ------------------------------------------------------------ U4 contract


def test_try_sample_signature_matches_u4_call():
    """patches/0005 calls try_sample(model, sample_hidden_states,
    sampling_metadata, scheduler_output, spec_decode_metadata)."""
    params = list(inspect.signature(Sm70HiddenStateSampler.try_sample).parameters)
    assert params == [
        "self",
        "model",
        "sample_hidden_states",
        "sampling_metadata",
        "scheduler_output",
        "spec_decode_metadata",
    ]


def test_patched_runner_calls_in_that_order():
    if not _have("vllm.v1.worker.gpu_model_runner"):
        pytest.skip("vLLM not importable")
    spec = importlib.util.find_spec("vllm.v1.worker.gpu_model_runner")
    src = pathlib.Path(spec.origin).read_text()
    if "_hidden_state_sampler.try_sample(" not in src:
        pytest.skip("vLLM without patches/0005 (U4)")
    call = src[src.index("_hidden_state_sampler.try_sample(") :]
    call = call[: call.index(")")]
    args = [a.strip() for a in call.split("(", 1)[1].split(",") if a.strip()]
    assert args == [
        "self.model",
        "sample_hidden_states",
        "self.input_batch.sampling_metadata",
        "scheduler_output",
        "spec_decode_metadata",
    ]


# ------------------------------------------------------------ admitted path


@requires_vllm
@pytest.mark.parametrize("rows", [1, 2])
def test_admitted_batch_returns_sampler_output(hook, kernel, rows):
    from vllm.v1.outputs import SamplerOutput

    out = _try(hook(rows=rows), rows=rows)
    assert isinstance(out, SamplerOutput)
    assert out.logprobs_tensors is None
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.sampled_token_ids.shape == (rows, 1)
    assert kernel == [(rows, VOCAB)]


@requires_vllm
def test_inactive_processors_are_admitted(hook, kernel):
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    min_p = object.__new__(MinPLogitsProcessor)
    min_p.min_p_count = 0
    procs = SimpleNamespace(
        argmax_invariant=[min_p],
        non_argmax_invariant=[
            SimpleNamespace(min_toks={}),
            SimpleNamespace(biases={}),
        ],
    )
    assert _try(hook(), metadata=_metadata(logitsprocs=procs)) is not None


@requires_vllm
def test_kernel_refusal_falls_back(hook, monkeypatch):
    monkeypatch.setattr(sample, "sample_lm_head_candidates", lambda *a, **k: None)
    assert _try(hook()) is None


# ------------------------------------------------------- the None (upstream) cases


def test_not_prepared_returns_none(fake_sm70, kernel):
    assert _try(Sm70HiddenStateSampler()) is None
    assert kernel == []


def test_dead_runner_returns_none(fake_sm70, kernel):
    h = Sm70HiddenStateSampler()
    h.prepare(_Runner(), None)  # the runner is collected right away
    import gc

    gc.collect()
    assert _try(h) is None
    assert kernel == []


def _thinking(active):
    return SimpleNamespace(has_tracked_requests=lambda: active)


NONE_CASES = {
    # runner/model state
    "model_not_opted_in": dict(model=_model(opt_in=False)),
    "model_without_lm_head": dict(
        model=SimpleNamespace(supports_sm70_topk_sampler=True, lm_head=None)
    ),
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
    # Penalties need output_token_ids, which async scheduling refreshes only
    # in _sample (update_async_output_token_ids), after try_sample.
    "penalties_async": dict(metadata=_metadata(no_penalties=False)),
    # sampling params the kernel does not reproduce exactly
    "greedy": dict(metadata=_metadata(all_random=False, all_greedy=True)),
    "mixed_greedy_random": dict(metadata=_metadata(all_random=False)),
    "logprobs": dict(metadata=_metadata(max_num_logprobs=5)),
    "logprob_token_ids": dict(metadata=_metadata(logprob_token_ids={0: [1]})),
    "allowed_token_ids": dict(metadata=_metadata(allowed_token_ids_mask=object())),
    "bad_words": dict(metadata=_metadata(bad_words_token_ids={0: [[1]]})),
    "no_top_k": dict(metadata=_metadata(top_k=None)),
    "no_temperature": dict(metadata=_metadata(temperature=None)),
}


@pytest.mark.parametrize("case", sorted(NONE_CASES))
def test_upstream_path_cases(hook, kernel, case):
    assert _try(hook(), **NONE_CASES[case]) is None
    assert kernel == []


@pytest.mark.parametrize("top_k", [0, 21, 50, VOCAB])
def test_top_k_outside_1_to_20_returns_none(hook, kernel, top_k):
    assert _try(hook(top_k=top_k)) is None
    assert kernel == []


def test_three_rows_return_none(hook, kernel):
    assert _try(hook(rows=3), rows=3) is None
    assert kernel == []


def test_hidden_rows_must_match_batch(hook, kernel):
    # e.g. a spec-decode-shaped hidden batch without spec metadata
    assert _try(hook(rows=1), rows=2, metadata=_metadata(1)) is None
    assert kernel == []


def test_two_seeded_rows_return_none(hook, kernel):
    meta = _metadata(2, generators={0: object()})
    assert _try(hook(rows=2), rows=2, metadata=meta) is None
    assert kernel == []


def test_fp64_gumbel_returns_none(hook, kernel):
    assert _try(hook(fp64=True)) is None
    assert kernel == []


def test_capture_returns_none(hook, kernel, monkeypatch):
    h = hook()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _try(h) is None
    assert kernel == []


@requires_vllm
def test_active_min_p_returns_none(hook, kernel):
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    min_p = object.__new__(MinPLogitsProcessor)
    min_p.min_p_count = 1
    procs = SimpleNamespace(argmax_invariant=[min_p], non_argmax_invariant=[])
    assert _try(hook(), metadata=_metadata(logitsprocs=procs)) is None
    assert kernel == []


def test_non_sm70_returns_none(hook, kernel, monkeypatch):
    h = hook()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert _try(h) is None
    assert kernel == []


@requires_vllm
def test_missing_ops_returns_none(hook, kernel, monkeypatch):
    h = hook()
    monkeypatch.setattr(sample, "ops_available", lambda: False)
    assert _try(h) is None
    assert kernel == []


def test_rejections_are_logged_once(hook, kernel, monkeypatch):
    lines = []
    monkeypatch.setattr(
        sample_v1.logger, "info", lambda msg, *args: lines.append(msg % args)
    )
    h = hook()
    for _ in range(3):
        _try(h, spec=object())
    assert lines == ["SM70 top-k candidate sampler rejected: spec_decode"]


@pytest.mark.parametrize(
    "want,kw",
    [
        ("spec_decode", dict(spec_decode_metadata=object())),
        ("structured_output", dict(scheduler_output=_sched(True))),
        ("penalties", dict(sampling_metadata=_metadata(no_penalties=False))),
        ("model", dict(model=_model(opt_in=False))),
        ("top_k", dict(sampling_metadata=_metadata(top_k=None))),
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


# -------------------------------------------------------- prepare (R4) / warmup


def test_prepare_creates_counter_only_on_sm70_with_ops(monkeypatch):
    created = []
    monkeypatch.setattr(sample, "prepare", lambda device: created.append(device))
    monkeypatch.setattr(sample, "ops_available", lambda: True)
    monkeypatch.setattr(sample, "_is_sm70", lambda index: index == 0)
    h = Sm70HiddenStateSampler()
    runner = _Runner()
    h.prepare(runner, None)
    assert created == [torch.device("cuda", 0)] and h.prepared
    runner.device = torch.device("cuda", 1)
    Sm70HiddenStateSampler().prepare(runner, None)
    runner.device = torch.device("cpu")
    Sm70HiddenStateSampler().prepare(runner, None)
    assert created == [torch.device("cuda", 0)]


def test_prepare_refuses_during_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(sample, "_OVERFLOW_MONITORS", {})
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        sample.prepare(torch.device("cuda", 0))
    assert sample._OVERFLOW_MONITORS == {}


def test_sampling_never_creates_the_counter():
    """R4: only prepare() constructs _OverflowMonitor; the per-call path
    looks it up and runs uncounted without it."""
    tree = ast.parse(inspect.getsource(sample))
    creators = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_OverflowMonitor"
                ):
                    creators.add(fn.name)
    assert creators == {"prepare"}
    src = inspect.getsource(sample.sample_topk_candidates)
    assert "_overflow_monitor(" in src and "None if monitor is None" in src


@requires_vllm
def test_sampler_does_not_create_device_state_in_hot_path(hook, kernel, monkeypatch):
    h = hook()  # prepare() ran here, as in kernel_warmup
    created = []
    monkeypatch.setattr(sample, "prepare", lambda device: created.append(device))
    _try(h)
    _try(h)
    assert created == []


def test_prepare_worker_binds_only_our_hook(fake_sm70):
    ours = Sm70HiddenStateSampler()
    runner = _Runner()
    runner._hidden_state_sampler = ours
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(use_fp64_gumbel=True)),
    )
    sample_v1.prepare_worker(worker)
    assert ours.prepared and ours._fp64_gumbel
    runner._hidden_state_sampler = None
    sample_v1.prepare_worker(worker)  # no hook: nothing to do
    sample_v1.prepare_worker(SimpleNamespace())  # V2 worker shape: no-op


# ------------------------------------------------- platform wiring / kill switch


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


def test_getter_returns_a_new_hook_by_default(platform_cls, monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FLINT_SAMPLER", raising=False)
    a = platform_cls.get_hidden_state_sampler()
    b = platform_cls.get_hidden_state_sampler()
    assert isinstance(a, Sm70HiddenStateSampler) and a is not b


@pytest.mark.parametrize("value,on", [("0", False), ("1", True), ("", True)])
def test_kill_switch(platform_cls, monkeypatch, value, on):
    monkeypatch.setenv("VLLM_SM70_FLINT_SAMPLER", value)
    assert (platform_cls.get_hidden_state_sampler() is not None) is on


def test_kernel_warmup_prepares_the_runner_hook(platform_cls, monkeypatch):
    seen = []
    monkeypatch.setattr(sample_v1, "prepare_worker", lambda w: seen.append(w))
    monkeypatch.delenv("VLLM_SM70_FLINT_SAMPLER", raising=False)
    worker = object()
    assert platform_cls.kernel_warmup(worker) is None
    assert seen == [worker]
    monkeypatch.setenv("VLLM_SM70_FLINT_SAMPLER", "0")
    platform_cls.kernel_warmup(worker)
    assert seen == [worker]


def test_kernel_warmup_survives_prepare_failure(platform_cls, monkeypatch):
    def boom(worker):
        raise RuntimeError("test")

    monkeypatch.setattr(sample_v1, "prepare_worker", boom)
    monkeypatch.delenv("VLLM_SM70_FLINT_SAMPLER", raising=False)
    assert platform_cls.kernel_warmup(object()) is None


def test_kill_switch_is_registered():
    from vllm_sm70 import envs

    assert set(envs._SECTIONS["sampler"]) == {"VLLM_SM70_FLINT_SAMPLER"}


# --------------------------------------- shape guards: Flint TP2, vocab shard, k=20


def _lm_head(org_vocab, tp_size, *, dtype=torch.float16, added=0, bias=None):
    padded = -(-org_vocab // 64) * 64
    return SimpleNamespace(
        weight=torch.empty(0, dtype=dtype),
        bias=bias,
        num_added_embeddings=added,
        tp_size=tp_size,
        org_vocab_size=org_vocab,
        org_vocab_size_padded=padded,
        num_embeddings_per_partition=padded // tp_size,
    )


def _lp(org_vocab, **kw):
    fields = dict(
        logits_as_input=False,
        soft_cap=None,
        scale=1.0,
        head_dtype=None,
        org_vocab_size=org_vocab,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_flint_vocab_admitted_at_every_tp(tp_size):
    assert (
        sample.lm_head_reject_reason(_lp(VOCAB), _lm_head(VOCAB, tp_size), tp_size)
        is None
    )


def test_flint_tp2_shard_geometry():
    """TP2: two 75072-logit shards, 37 selection tiles each (<= 128), 64
    gathered candidates (<= 128), k=20 leaves 12 spare per rank."""
    from vllm_sm70.ops import sampler as ops

    head = _lm_head(VOCAB, 2)
    assert head.num_embeddings_per_partition == 75072
    assert -(-75072 // ops.TILE) == 37 <= ops.MAX_TILES
    assert 2 * ops.LOCAL_K <= ops.MAX_CANDIDATES
    assert sample.MAX_TOP_K == 20 and ops.LOCAL_K - sample.MAX_TOP_K == 12
    assert 1 <= sample.MAX_ROWS <= ops.MAX_KERNEL_ROWS


@pytest.mark.parametrize("org_vocab", [150000, 151936, 248320])
def test_padded_vocab_tp2_admitted(org_vocab):
    """The last rank's shard has fewer valid logits than columns."""
    assert (
        sample.lm_head_reject_reason(_lp(org_vocab), _lm_head(org_vocab, 2), 2) is None
    )


@pytest.mark.parametrize(
    "lp,head,tp,reason",
    [
        (
            _lp(VOCAB, soft_cap=30.0),
            _lm_head(VOCAB, 2),
            2,
            "logits_processor_transform",
        ),
        (_lp(VOCAB, scale=0.5), _lm_head(VOCAB, 2), 2, "logits_processor_transform"),
        (
            _lp(VOCAB, logits_as_input=True),
            _lm_head(VOCAB, 2),
            2,
            "logits_processor_transform",
        ),
        (_lp(VOCAB, head_dtype=torch.float32), _lm_head(VOCAB, 2), 2, "head_dtype"),
        (_lp(VOCAB), _lm_head(VOCAB, 2, dtype=torch.bfloat16), 2, "lm_head_dtype"),
        (_lp(VOCAB), _lm_head(VOCAB, 2, bias=object()), 2, "lm_head_bias"),
        (_lp(VOCAB), _lm_head(VOCAB, 2, added=16), 2, "added_vocab"),
        (_lp(VOCAB), _lm_head(VOCAB, 4), 2, "lm_head_group"),
        (_lp(VOCAB), _lm_head(VOCAB, 8), 8, "tp_size"),
        (_lp(1000), _lm_head(VOCAB, 2), 2, "vocab_size"),
        # one rank would hold > 262144 logits (128 tiles of 2048)
        (_lp(600000), _lm_head(600000, 2), 2, "shard_shape"),
        (_lp(300000), _lm_head(300000, 1), 1, "shard_shape"),
        # the last rank keeps fewer than 32 valid logits
        (_lp(64 * 3 + 8), _lm_head(64 * 3 + 8, 4), 4, "shard_shape"),
    ],
)
def test_lm_head_rejections(lp, head, tp, reason):
    assert sample.lm_head_reject_reason(lp, head, tp) == reason


def test_wire_guards():
    meta = _metadata(2)
    wire = torch.zeros(4, 64)
    for bad in (
        torch.zeros(4, 64, dtype=torch.float16),
        torch.zeros(4, 63),
        torch.zeros(3, 64),
        torch.zeros(64, 4).t(),
    ):
        assert (
            sample.sample_topk_candidates(bad, meta, tp_size=2, vocab_size=VOCAB)
            is None
        )
    assert (
        sample.sample_topk_candidates(wire, meta, tp_size=8, vocab_size=VOCAB) is None
    )
    assert sample.sample_topk_candidates(wire, meta, tp_size=2, vocab_size=16) is None
    assert (
        sample.sample_topk_candidates(wire, _metadata(1), tp_size=2, vocab_size=VOCAB)
        is None
    )
    meta64 = _metadata(2, top_k=torch.full((2,), 20, dtype=torch.int64))
    assert (
        sample.sample_topk_candidates(wire, meta64, tp_size=2, vocab_size=VOCAB) is None
    )


def test_shard_supported_rejects_cpu_and_layout():
    x = torch.zeros(2, 75072, dtype=torch.float16)
    assert not sample.shard_supported(x, vocab_start=0, num_valid=75072)


# ----------------------------------------------------------- op bindings


def test_exactly_the_two_ops_are_defined():
    assert set(sm70_schemas(area="sampler")) == EXPECTED_OPS


def test_schemas_are_the_fork_schemas():
    schemas = sm70_schemas(area="sampler")
    assert schemas["sm70_local_top32_half"] == (
        "sm70_local_top32_half(Tensor logits, int vocab_start, int num_valid) -> Tensor"
    )
    assert schemas["sm70_sample_topk_candidates"] == (
        "sm70_sample_topk_candidates(Tensor wire, Tensor temperature, "
        "Tensor top_k, Tensor? top_p, Generator? generator, int vocab_size, "
        "int tp_size, Tensor(a!)? overflow) -> Tensor"
    )


def test_every_op_has_a_cuda_impl_only():
    text = (CSRC / "sampler_ops.cpp").read_text()
    impls = {}
    for block in re.finditer(
        r"TORCH_LIBRARY_IMPL\(\s*_sm70_C\s*,\s*(\w+)\s*,\s*m\s*\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    ):
        for op in re.findall(r'm\.impl\("(\w+)"', block.group(2)):
            impls.setdefault(op, set()).add(block.group(1))
    assert impls == {op: {"CUDA"} for op in EXPECTED_OPS}


def test_every_op_has_a_matching_fake():
    text = OPS_PY.read_text()
    assert set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', text)) == EXPECTED_OPS
    defs = {
        n.name: [a.arg for a in n.args.args]
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.FunctionDef)
    }
    for name, schema in sm70_schemas(area="sampler").items():
        args = schema[schema.index("(") + 1 : schema.rindex(") ->")]
        names = [a.strip().split(" ")[-1] for a in args.split(",")]
        assert defs[f"_{name}_fake"] == names, name


def test_kernel_constants_match_python():
    from vllm_sm70.ops import sampler as ops

    cu = (CSRC / "topk_candidates_sampler.cu").read_text()

    def const(name):
        return int(re.search(rf"constexpr int {name} = (\w+);", cu).group(1))

    assert const("kLocalK") == ops.LOCAL_K
    assert const("kSelectThreads") * const("kTileItems") == ops.TILE
    assert const("kMaxTiles") == ops.MAX_TILES
    assert const("kMaxCandidates") == ops.MAX_CANDIDATES
    assert const("kMaxRows") == ops.MAX_KERNEL_ROWS


def test_no_fork_namespace_or_sidecar_left():
    bad = []
    for path in sorted(CSRC.iterdir()):
        if path.suffix not in (".cu", ".cpp", ".cmake"):
            continue
        code = re.sub(r"//[^\n]*|#[^\n]*", " ", path.read_text())
        for pat in (r"TORCH_LIBRARY\w*\(\s*_C\b", r"\bgetenv\b", r"PYBIND11_MODULE"):
            if re.search(pat, code):
                bad.append(f"{path.name}: {pat}")
    for py in (OPS_PY, SM70 / "vllm_sm70" / "sample.py"):
        text = py.read_text()
        for pat in (r"torch\.ops\._C\b", r"vllm\._sm70_ops", r"current_platform"):
            if re.search(pat, text):
                bad.append(f"{py.name}: {pat}")
    assert not bad, bad


def test_sources_cmake_and_hook():
    text = (CSRC / "sources.cmake").read_text()
    listed = set(re.findall(r'CMAKE_CURRENT_LIST_DIR\}/([\w.]+)"', text))
    assert listed == {"topk_candidates_sampler.cu", "sampler_ops.cpp"}
    assert "use_fast_math" not in re.sub(r"#[^\n]*", "", text)
    top = (SM70 / "CMakeLists.txt").read_text()
    assert 'include("${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler/sources.cmake")' in top


@pytest.fixture
def ops_with_fakes():
    """ops/sampler.py with its fakes registered against stub schemas."""
    lib = None
    if not hasattr(torch.ops._sm70_C, "sm70_local_top32_half"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in sm70_schemas(area="sampler").items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.ops.sampler as mod

    mod = importlib.reload(mod)
    yield mod
    if lib is not None:
        lib._destroy()


def test_fakes_on_meta(ops_with_fakes):
    ops = ops_with_fakes
    logits = torch.empty(2, 75072, dtype=torch.float16, device="meta")
    wire = ops.local_top32_half(logits, 0, 75072)
    assert wire.shape == (2, 64) and wire.dtype == torch.float32
    gathered = torch.empty(4, 64, dtype=torch.float32, device="meta")
    out = ops.sample_topk_candidates(
        gathered,
        torch.empty(2, device="meta"),
        torch.empty(2, dtype=torch.int32, device="meta"),
        None,
        None,
        VOCAB,
        2,
        None,
    )
    assert out.shape == (2,) and out.dtype == torch.int32


def test_provenance_lists_every_kept_function():
    text = (CSRC / "PROVENANCE.md").read_text()
    for name in (
        "half_order",
        "local_top32_tile_kernel",
        "local_top32_merge_kernel",
        "native_exponential_for_index",
        "sample_topk_candidates_kernel",
        "sm70_local_top32_half",
        "sm70_sample_topk_candidates",
        "metadata_reject_reason",
        "argmax_invariant_reject_reason",
        "admission_reject_reason",
        "lm_head_topk_candidates",
        "sample_lm_head_candidates",
        "local_topk_candidates",
        "shard_supported",
        "sample_topk_candidates",
        "_OverflowMonitor",
    ):
        assert f"`{name}`" in text, name
