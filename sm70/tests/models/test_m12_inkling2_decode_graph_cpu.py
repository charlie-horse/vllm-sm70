# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2: decode-only batches are ready for a FULL CUDA graph.

On the tiny checkpoint (fixtures/inkling2_tiny), driven by the paged engine
harness with the metadata a FULL graph gets (padded slot mapping, padded
requests):

- at one padded size the real rows of a decode step do not depend on the
  padding or on the other requests: bit-identical whether the step pads
  one request or holds two; against the same step unpadded they differ only
  by the GEMM rounding of another row count (CPU: <= 1.5e-6 in fp32, one
  fp16 ulp in fp16);
- a padded decode step has no host sync and its op trace (every op with the
  shapes and dtypes of its outputs, ``inkling2_sync_guard``) depends on the
  padded batch size and the block-table width alone: steps at other
  positions and with another number of real requests launch the same ops
  on the same allocation sizes, as a replayed graph does;
- VLLM_SM70_INKLING_DECODE_GRAPH=0 takes the ops of mixed batches again;
- the TurboMind workspace refuses to grow inside a capture.

The engine-config hook's mode selection is in
test_m12_inkling2_config_cpu.py; capture and replay on a GPU in
test_m12_inkling2_decode_graph_gpu.py.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inkling2_sync_guard import HostSync, no_host_sync
from test_m12_inkling2_model_cpu import Engine, _have_vllm, _raw_config

pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

PROMPT_A, PROMPT_B = 9, 6
DECODE_STEPS = 5
# A capture size and a block-table width as the runner's: fixed per engine.
PADDED = 4
TABLE_WIDTH = 16
SWITCH = "VLLM_SM70_INKLING_DECODE_GRAPH"


@pytest.fixture(scope="module")
def sequences():
    generator = torch.Generator().manual_seed(23)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    length = max(PROMPT_A, PROMPT_B) + DECODE_STEPS
    return {
        "a": torch.randint(0, vocab, (length,), generator=generator),
        "b": torch.randint(0, vocab, (length,), generator=generator),
    }


@pytest.fixture(scope="module", params=[torch.float32, torch.float16])
def engine(request):
    return Engine(request.param)


def _fixed_width(rank, metadata):
    """The block tables at the runner's static width (zero-filled)."""
    for md in metadata.values():
        width = md.block_table.shape[1]
        assert width <= TABLE_WIDTH
        md.block_table = torch.nn.functional.pad(
            md.block_table, (0, TABLE_WIDTH - width)
        )


def _decode(engine, sequences, req_ids, *, padded, guard=None, steps=DECODE_STEPS):
    """Prefill each request unpadded, then `steps` decode steps of all of
    them, padded to `padded` tokens (FULL-graph metadata) or not. Returns
    the real rows of each decode step and, with `guard` (a list), runs the
    decode steps' model body under ``no_host_sync`` and appends each step's
    op trace to it."""
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
        for _ in range(steps):
            schedule = [(req_id, 1) for req_id in req_ids]
            pad = padded - len(req_ids) if padded else 0
            out = harness.step(
                schedule,
                engine.forward,
                pad_tokens=pad,
                pad_reqs=pad,
                metadata="full",
                tamper=_fixed_width,
            )[0]
            decode_only = os.environ.get(SWITCH) != "0"
            for md in harness.metadata[0].values():
                assert md.decode_only is decode_only
            assert torch.isfinite(out).all()
            outputs.append(out[: harness.batch.num_tokens].clone())
    finally:
        if guard is not None:
            del body.forward
    return outputs


def test_padding_and_other_rows_do_not_change_a_padded_row(engine, sequences):
    alone = _decode(engine, sequences, ("a",), padded=PADDED)
    shared = _decode(engine, sequences, ("a", "b"), padded=PADDED)
    for step, (one, two) in enumerate(zip(alone, shared)):
        assert torch.equal(one[0], two[0]), f"decode step {step}"


@pytest.mark.parametrize("req_ids", [("a",), ("a", "b")], ids=["one", "two"])
def test_padded_decode_rows_match_unpadded(engine, sequences, req_ids):
    want = _decode(engine, sequences, req_ids, padded=0)
    got = _decode(engine, sequences, req_ids, padded=PADDED)
    tol = 1e-5 if engine.harness.dtype == torch.float32 else 4e-3
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=tol, atol=tol)


def test_padded_decode_step_is_static(engine, sequences):
    """No host sync, and one op trace for every step at one padded size."""
    traces: list = []
    _decode(engine, sequences, ("a",), padded=PADDED, guard=traces)
    _decode(engine, sequences, ("a", "b"), padded=PADDED, guard=traces, steps=2)
    assert len(traces) == DECODE_STEPS + 2
    assert all(len(trace) > 1000 for trace in traces)
    for step, trace in enumerate(traces[1:], 1):
        assert trace == traces[0], f"decode step {step} launches other ops"


def test_kill_switch_takes_the_mixed_batch_ops(engine, sequences, monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    with pytest.raises(HostSync):
        _decode(engine, sequences, ("a",), padded=PADDED, guard=[], steps=1)
    want = _decode(engine, sequences, ("a",), padded=0, steps=2)
    monkeypatch.delenv(SWITCH)
    got = _decode(engine, sequences, ("a",), padded=0, steps=2)
    tol = 1e-4 if engine.harness.dtype == torch.float32 else 2e-2
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=tol)


def test_turbomind_workspace_does_not_grow_in_a_capture(monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import moe_tm

    ws = object.__new__(moe_tm.Workspace)
    ws.device, ws.rows, ws.buffers = torch.device("cuda"), 8, {}
    ws.widths = {"permuted_input": 4}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    ws.reserve(8)
    with pytest.raises(RuntimeError, match="inside a CUDA graph capture"):
        ws.reserve(16)
    assert ws.rows == 8
