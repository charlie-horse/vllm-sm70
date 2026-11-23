# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2: verify steps of speculative decoding as FULL decode graphs.

With a speculative config of K draft tokens, a decode step holds q = K + 1
query tokens per request. On the tiny checkpoint (fixtures/inkling2_tiny),
driven by the paged engine harness under FULL_DECODE_ONLY:

- a verify step with the metadata a replayed FULL graph gets (padded
  requests and slot mapping, the runner's block-table width) is decode-only,
  and its real rows equal the eager route of the same step
  (VLLM_SM70_INKLING_DECODE_GRAPH=0: the ops of mixed batches) within 1e-5
  in fp32 and 4e-3 in fp16 of the largest value (one fp16 ulp of the visit
  states after 48 visits; the attention and conv ops alone are within 2e-3,
  ops/test_m12_inkling2_triton_*_cpu.py), across steps whose tail drafts
  are rejected (the next step recomputes their positions) and sliding
  windows and conv taps that reach into the same step's rows. The tiny
  model's router scores are saturated sigmoids with near-ties, so in fp16
  each router's bias pins its top-k: a rounding difference of the two
  routes then cannot flip an expert;
- a padded verify step has no host sync and launches the same ops on the
  same allocation sizes at other positions and request counts;
- only uniform steps of q tokens per request (zero-token padding requests
  last) are decode-only; the builders declare UNIFORM_BATCH graphs and a
  reorder threshold of q with a speculative config, and keep
  UNIFORM_SINGLE_TOKEN_DECODE and 1 without one;
- the engine-config hook counts the capture cap in requests of q tokens;
- with a dynamic speculative-decoding schedule (spec_policy.py), steps of
  q = 1 and of q = K + 1 are both decode-only, and requests that switch
  between them mid-stream, after rejected drafts, match the eager route.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inkling2_sync_guard import no_host_sync
from test_m12_inkling2_model_cpu import Engine, _have_vllm, _raw_config

pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

K = 7
Q = K + 1
PROMPT_A, PROMPT_B = 9, 6
# Drafts rejected at the end of each verify step, per request.
REJECTED = ({"a": 5, "b": 0}, {"a": 0, "b": 7}, {"a": 3, "b": 2}, {"a": 7, "b": 4})
PADDED_REQS = 4
TABLE_WIDTH = 16
SWITCH = "VLLM_SM70_INKLING_DECODE_GRAPH"
FULL = "FULL_DECODE_ONLY"
TOLS = {torch.float32: 1e-5, torch.float16: 4e-3}


def _spec_config(config) -> None:
    """A speculative config of K drafts on the V2 runner, and the capture cap
    the engine-config hook sets for it."""
    from vllm_sm70.models.inkling2_sm70.config import DECODE_GRAPH_MAX_BATCH

    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=K, parallel_drafting=False
    )
    config.use_v2_model_runner = True
    config.compilation_config.max_cudagraph_capture_size = DECODE_GRAPH_MAX_BATCH * Q


@pytest.fixture(scope="module")
def sequences():
    generator = torch.Generator().manual_seed(31)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    length = max(PROMPT_A, PROMPT_B) + Q * len(REJECTED)
    return {
        "a": torch.randint(0, vocab, (length,), generator=generator),
        "b": torch.randint(0, vocab, (length,), generator=generator),
    }


@pytest.fixture(scope="module", params=[torch.float32, torch.float16])
def engine(request):
    from vllm_sm70.models.inkling2_sm70.moe import Inkling2Sm70Gate

    engine = Engine(request.param, cudagraph_mode=FULL)
    _spec_config(engine.harness.vllm_config(0))
    if request.param == torch.float16:
        generator = torch.Generator().manual_seed(5)
        for module in engine.model.modules():
            if isinstance(module, Inkling2Sm70Gate):
                # Sigmoid scores lie in (0, 1): bias steps of 1 fix the top-k.
                experts = module.bias.shape[0]
                module.bias.copy_(torch.randperm(experts, generator=generator))
    return engine


def _fixed_width(rank, metadata):
    """The block tables at the runner's static width (zero-filled)."""
    for md in metadata.values():
        width = md.block_table.shape[1]
        assert width <= TABLE_WIDTH
        md.block_table = torch.nn.functional.pad(
            md.block_table, (0, TABLE_WIDTH - width)
        )


def _verify(engine, sequences, req_ids, *, padded, guard=None, steps=None):
    """Prefill each request, then verify steps of Q tokens per request, each
    followed by the rejection of REJECTED of them. `padded`: the metadata of
    a FULL graph of PADDED_REQS requests (decode-only), else the unpadded
    eager step. Returns the real rows of each verify step; with `guard` (a
    list) the model body runs under ``no_host_sync`` and appends each step's
    op trace."""
    engine.reset()
    engine.sequences = sequences
    harness = engine.harness
    prompts = {"a": PROMPT_A, "b": PROMPT_B}
    for req_id in req_ids:
        harness.add_request(req_id)
        harness.step({req_id: prompts[req_id]}, engine.forward)
    body = engine.model.model
    forward = body.forward

    def guarded(positions, inputs_embeds):
        trace: list = []
        with no_host_sync(trace):
            out = forward(positions, inputs_embeds)
        guard.append(trace)
        return out

    outputs = []
    try:
        if guard is not None:
            body.forward = guarded
        for rejected in REJECTED[:steps]:
            schedule = [(req_id, Q) for req_id in req_ids]
            pad = PADDED_REQS - len(req_ids) if padded else 0
            out = harness.step(
                schedule,
                engine.forward,
                pad_tokens=pad * Q,
                pad_reqs=pad,
                metadata="full",
                tamper=_fixed_width if padded else None,
            )[0]
            for md in harness.metadata[0].values():
                assert md.decode_only is padded
            assert torch.isfinite(out).all()
            outputs.append(out[: harness.batch.num_tokens].clone())
            for req_id in req_ids:
                harness.requests[req_id].computed -= rejected[req_id]
    finally:
        if guard is not None:
            del body.forward
    return outputs


@pytest.mark.parametrize("req_ids", [("a",), ("a", "b")], ids=["one", "two"])
def test_graph_shaped_verify_matches_the_eager_route(
    engine, sequences, req_ids, monkeypatch
):
    monkeypatch.setenv(SWITCH, "0")
    want = _verify(engine, sequences, req_ids, padded=False)
    monkeypatch.delenv(SWITCH)
    got = _verify(engine, sequences, req_ids, padded=True)
    tol = TOLS[engine.harness.dtype]
    assert len(got) == len(REJECTED)
    for step, (g, w) in enumerate(zip(got, want)):
        assert g.shape == (len(req_ids) * Q, w.shape[1])
        err = ((g - w).abs().max() / w.abs().max()).item()
        assert err <= tol, (step, err)


def test_padded_verify_step_is_static(engine, sequences):
    """No host sync, and one op trace for every verify step at one padded
    size."""
    traces: list = []
    _verify(engine, sequences, ("a",), padded=True, guard=traces)
    _verify(engine, sequences, ("a", "b"), padded=True, guard=traces, steps=2)
    assert len(traces) == len(REJECTED) + 2
    assert all(len(trace) > 1000 for trace in traces)
    for step, trace in enumerate(traces[1:], 1):
        assert trace == traces[0], f"verify step {step} launches other ops"


# Steps of a schedule [(1, 1, K), (2, max, 0)]: q per request, then the
# drafts rejected per request.
SWITCHING = (
    (Q, {"a": 5, "b": 0}),
    (1, {}),
    (1, {}),
    (Q, {"a": 7, "b": 3}),
    (1, {}),
    (Q, {"a": 0, "b": 7}),
)
SCHEDULE = [(1, 1, K), (2, 8, 0)]


def _switching(engine, sequences, *, padded):
    """Prefill a and b, then the steps of SWITCHING; returns each step's
    real rows."""
    engine.reset()
    engine.sequences = sequences
    harness = engine.harness
    for req_id, prompt in (("a", PROMPT_A), ("b", PROMPT_B)):
        harness.add_request(req_id)
        harness.step({req_id: prompt}, engine.forward)
    outputs = []
    for q, rejected in SWITCHING:
        pad = PADDED_REQS - 2 if padded else 0
        out = harness.step(
            [("a", q), ("b", q)],
            engine.forward,
            pad_tokens=pad * q,
            pad_reqs=pad,
            metadata="full",
            tamper=_fixed_width if padded else None,
        )[0]
        for md in harness.metadata[0].values():
            assert md.decode_only is padded
        assert torch.isfinite(out).all()
        outputs.append(out[: harness.batch.num_tokens].clone())
        for req_id, n in rejected.items():
            harness.requests[req_id].computed -= n
    return outputs


def test_switching_steps_match_the_eager_route(engine, sequences, monkeypatch):
    spec = engine.harness.vllm_config(0).speculative_config
    monkeypatch.setattr(
        spec, "num_speculative_tokens_per_batch_size", SCHEDULE, raising=False
    )
    monkeypatch.setenv(SWITCH, "0")
    want = _switching(engine, sequences, padded=False)
    monkeypatch.delenv(SWITCH)
    got = _switching(engine, sequences, padded=True)
    tol = TOLS[engine.harness.dtype]
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g.shape == (2 * SWITCHING[step][0], w.shape[1])
        err = ((g - w).abs().max() / w.abs().max()).item()
        assert err <= tol, (step, err)


# --- the builders' rule ------------------------------------------------------------


def _common(query_lens, *, pad_reqs=0, pad_tokens=0):
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


def _engine_config(spec: bool, cap: int):
    from vllm.config.compilation import CUDAGraphMode

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=cap,
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=None,
    )
    if spec:
        _spec_config(config)
        config.compilation_config.max_cudagraph_capture_size = cap
    return config


@pytest.mark.parametrize(
    ("query_lens", "pad_reqs", "want"),
    [
        ([Q], 0, True),
        ([Q, Q], 2, True),
        ([Q] * 8, 0, True),
        ([Q] * 9, 0, False),  # above the capture cap
        ([1, 1], 0, False),  # single-token decode: no graph of q = 1
        ([Q, 1], 0, False),
        ([1, Q], 0, False),
        ([Q, 3], 0, False),
    ],
)
def test_only_uniform_verify_steps_replay(query_lens, pad_reqs, want):
    from vllm_sm70.models.inkling2_sm70.backend import replays_decode_graph

    config = _engine_config(True, 8 * Q)
    cm = _common(query_lens, pad_reqs=pad_reqs, pad_tokens=pad_reqs * Q)
    assert replays_decode_graph(config, cm) is want


@pytest.mark.parametrize(
    ("query_lens", "pad_reqs", "want"),
    [([1, 1], 0, True), ([1], 1, True), ([1] * 9, 0, False), ([2, 1], 0, False)],
)
def test_spec_off_keeps_the_single_token_rule(query_lens, pad_reqs, want):
    from vllm_sm70.models.inkling2_sm70.backend import replays_decode_graph

    cm = _common(query_lens, pad_reqs=pad_reqs, pad_tokens=pad_reqs)
    assert replays_decode_graph(_engine_config(False, 8), cm) is want


@pytest.mark.parametrize("spec", [False, True], ids=["spec-off", "spec-on"])
def test_builders_declare_uniform_batch_with_a_speculative_config(
    engine, sequences, spec, monkeypatch
):
    from vllm.v1.attention.backend import AttentionCGSupport

    from vllm_sm70.models.inkling2_sm70 import backend

    config = _engine_config(spec, 8)
    for builder_cls in (
        backend.Inkling2AttentionMetadataBuilder,
        backend.Inkling2SconvMetadataBuilder,
    ):
        support = builder_cls.get_cudagraph_support(config, None)
        want = (
            AttentionCGSupport.UNIFORM_BATCH
            if spec
            else AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        )
        assert support == want
        monkeypatch.setenv(SWITCH, "0")
        assert builder_cls.get_cudagraph_support(config, None) == (
            AttentionCGSupport.NEVER
        )
        monkeypatch.delenv(SWITCH)
    # The engine's builders (spec on) order verify requests as decodes.
    _verify(engine, sequences, ("a",), padded=True, steps=1)
    assert engine.harness._builders[0]
    for builder in engine.harness._builders[0].values():
        assert builder.reorder_batch_threshold == Q


@pytest.mark.parametrize(
    ("max_num_seqs", "k", "want"), [(16, 7, 64), (3, 7, 24), (16, 0, 8), (3, 0, 3)]
)
def test_hook_caps_the_capture_in_requests(max_num_seqs, k, want):
    from vllm_sm70.models.inkling2_sm70 import config as icfg

    compilation = SimpleNamespace(
        cudagraph_capture_sizes=None, max_cudagraph_capture_size=None
    )
    scheduler = SimpleNamespace(max_num_seqs=max_num_seqs)
    icfg._cap_decode_graph_batch(compilation, scheduler, k + 1)
    assert compilation.max_cudagraph_capture_size == want
    compilation.max_cudagraph_capture_size = icfg.DECODE_GRAPH_MAX_BATCH * (k + 1) + 1
    with pytest.raises(ValueError, match="FULL decode graphs of at most"):
        icfg._cap_decode_graph_batch(compilation, scheduler, k + 1)


def test_decode_query_len():
    from vllm_sm70.models.inkling2_sm70.backend import decode_query_len

    assert decode_query_len(SimpleNamespace(speculative_config=None)) == 1
    spec = SimpleNamespace(num_speculative_tokens=K, parallel_drafting=True)
    v2 = SimpleNamespace(speculative_config=spec, use_v2_model_runner=True)
    assert decode_query_len(v2) == Q
    v1 = SimpleNamespace(speculative_config=spec, use_v2_model_runner=False)
    assert decode_query_len(v1) == 1 + 2 * K
