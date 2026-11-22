# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2: a decode step captured in a CUDA graph on an SM70 GPU.

The tiny checkpoint (fixtures/inkling2_tiny) in fp16 at TP1 on the GPU,
driven by the paged engine harness. Prefills run eagerly. The model body of
a decode step padded to PADDED tokens is captured once in a
``torch.cuda.CUDAGraph`` over static buffers: the input ids, and per cache
owner the block table (at a fixed width), slot mapping and positions; the
builders' own index buffers are static already. Every later decode step
copies its metadata into those buffers and replays the graph, first with
one real request and then with two. Each replay is compared with the same
step run eagerly just before it (same caches, same metadata; a step writes
the same cache slots with the same values each time). Before the capture
the step runs under CUDA sync debugging at "error".

    pytest -m gpu sm70/tests/models/test_m12_inkling2_decode_graph_gpu.py
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

pytestmark = pytest.mark.gpu

PADDED = 4
TABLE_WIDTH = 16
PROMPT_A, PROMPT_B = 9, 6
STEPS_ONE, STEPS_TWO = 4, 3
TOL = 2e-3


@pytest.fixture
def engine(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    for name in (
        "VLLM_SM70_INKLING_DECODE_GRAPH",
        "VLLM_SM70_INKLING_TRITON_SCONV",
    ):
        monkeypatch.delenv(name, raising=False)
    from test_m12_inkling2_model_cpu import Engine

    from vllm_sm70.models.inkling2_sm70 import sconv

    assert sconv.on_triton(torch.zeros(1, device="cuda"))
    return Engine(torch.float16, device="cuda", cudagraph_mode="FULL_DECODE_ONLY")


class _StaticMetadata:
    """A tamper hook for the harness: copies each owner's block table, slot
    mapping and positions into buffers that live as long as the graph."""

    def __init__(self) -> None:
        self.buffers: dict[str, dict[str, torch.Tensor]] = {}

    def __call__(self, rank: int, metadata: dict) -> None:
        for name, md in metadata.items():
            assert md.decode_only is True
            table = torch.nn.functional.pad(
                md.block_table, (0, TABLE_WIDTH - md.block_table.shape[1])
            )
            fresh = {
                "block_table": table,
                "slot_mapping": md.slot_mapping,
                "positions": md.positions,
            }
            static = self.buffers.setdefault(
                name, {key: value.clone() for key, value in fresh.items()}
            )
            for key, value in fresh.items():
                assert static[key].shape == value.shape, (name, key)
                static[key].copy_(value)
                setattr(md, key, static[key])


def _assert_close(got, want, what):
    assert torch.isfinite(got).all(), what
    diff = (got.float() - want.float()).abs().max().item()
    assert diff <= TOL * max(1.0, want.float().abs().max().item()), (what, diff)


def test_decode_step_replays_from_a_cuda_graph(engine):
    generator = torch.Generator().manual_seed(31)
    length = max(PROMPT_A, PROMPT_B) + STEPS_ONE + STEPS_TWO + 1
    vocab = engine.model.text_config.unpadded_vocab_size
    sequences = {
        r: torch.randint(0, vocab, (length,), generator=generator).cuda()
        for r in ("a", "b")
    }
    engine.reset()
    engine.sequences = sequences
    harness, model = engine.harness, engine.model
    static_metadata = _StaticMetadata()
    static_ids = torch.zeros(PADDED, dtype=torch.long, device="cuda")
    graph: dict = {}
    replays = []

    def eager(rank, batch):
        with torch.no_grad():
            return model(batch.take(sequences), batch.positions)

    def prefill(req_id, prompt):
        harness.add_request(req_id)
        harness.step({req_id: prompt}, eager)

    def decode_step(rank, batch):
        ids = batch.take(sequences)
        static_ids.copy_(ids)
        with torch.no_grad():
            if not graph:
                model(static_ids, batch.positions)  # compiles the kernels
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    model(static_ids, batch.positions)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                g = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    model(static_ids, batch.positions)  # warm-up, same writes
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(g):
                    graph["out"] = model(static_ids, batch.positions)
                graph["graph"] = g
            want = model(ids, batch.positions).clone()
            graph["graph"].replay()
            torch.cuda.synchronize()
            got = graph["out"].clone()
        n = batch.num_tokens
        _assert_close(got[:n], want[:n], f"decode step {len(replays)}")
        logits_got = model.compute_logits(got[:n])
        logits_want = model.compute_logits(want[:n])
        _assert_close(logits_got, logits_want, f"logits {len(replays)}")
        replays.append(n)
        return got

    def run_decodes(req_ids, steps):
        pad = PADDED - len(req_ids)
        for _ in range(steps):
            harness.step(
                [(r, 1) for r in req_ids],
                decode_step,
                pad_tokens=pad,
                pad_reqs=pad,
                metadata="full",
                tamper=static_metadata,
            )

    prefill("a", PROMPT_A)
    run_decodes(("a",), STEPS_ONE)
    prefill("b", PROMPT_B)
    run_decodes(("a", "b"), STEPS_TWO)
    assert replays == [1] * STEPS_ONE + [2] * STEPS_TWO
