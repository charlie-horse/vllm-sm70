# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 model body on the merged visit kernels (visit_fusion.py), on CPU.

The model of test_m12_inkling2_model_cpu.py (FULL_DECODE_ONLY) runs a prefill
and decode steps of two requests through Triton's interpreter, with the
merged kernels and with VLLM_SM70_INKLING_VISIT_FUSION=0:
- with the visit hook (each layer output runs its own conv and add,
  ``conv_residual``) every visit's state, the final state and the logits agree;
- without it a layer output stays pending into the next attention input mix
  or the final norm, and the final state and the logits agree;
- the Triton launches of a decode step drop by the count the merges remove
  per visit; the prefill step launches the same kernels on both routes.
The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent they show as skipped.
"""

from __future__ import annotations

import collections
import os
import pathlib
import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx
import test_m12_inkling2_model_cpu as body

pytestmark = pytest.mark.skipif(not body._have_vllm(), reason="vLLM not importable")

SWITCH = "VLLM_SM70_INKLING_VISIT_FUSION"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
LENGTH, PROMPT = 6, 4
TOL = 1e-4


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, not skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    for name in (
        SWITCH,
        "VLLM_SM70_INKLING_TRITON_NORMS",
        "VLLM_SM70_INKLING_TRITON_SCONV",
        "VLLM_SM70_INKLING_FUSED_SMALL_OPS",
        "VLLM_SM70_INKLING_FUSED_ROUTING",
    ):
        env.pop(name, None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", __file__, "-q", "-p",
            "no:cacheprovider", "-m", "not gpu", "-k", "not test_interpreted_suite",
        ],
        env=env, capture_output=True, text=True, timeout=3600, check=False,
    )  # fmt: skip
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    assert re.search(r"\b2 passed", summary), summary
    assert "skipped" not in summary, summary


class _Launches:
    """Triton launches by kernel name, per forward."""

    def __init__(self, monkeypatch) -> None:
        from triton.runtime.interpreter import InterpretedFunction

        self.steps: list[collections.Counter] = []
        real = InterpretedFunction.run

        def run(fn, *args, **kwargs):
            if self.steps:
                self.steps[-1][fn.fn.__name__] += 1
            return real(fn, *args, **kwargs)

        monkeypatch.setattr(InterpretedFunction, "run", run)


def _run(engine, launches: _Launches, hook: bool) -> torch.Tensor:
    """[tokens, ...] rows of both requests: every visit's state (with the
    hook), the final state and the logits."""
    model = engine.model
    forward = engine.forward

    def counted(rank, batch):
        launches.steps.append(collections.Counter())
        if hook:
            return forward(rank, batch)
        input_ids = batch.take(engine.sequences)
        with torch.no_grad():
            final = model(input_ids, batch.positions)
            logits = model.compute_logits(final)
        return torch.cat([final.float(), logits], dim=1)

    generator = torch.Generator().manual_seed(5)
    vocab = body._raw_config()["text_config"]["unpadded_vocab_size"]
    sequences = {
        name: torch.randint(0, vocab, (LENGTH,), generator=generator)
        for name in ("a", "b")
    }
    plan = (
        hx.add("a") + hx.add("b") + hx.step({"a": PROMPT, "b": PROMPT})
        + hx.step({"a": 1, "b": 1}) * (LENGTH - PROMPT)
    )  # fmt: skip
    engine.reset()
    engine.sequences = sequences
    rows = engine.harness.run(plan, counted)
    return torch.cat([rows["a"], rows["b"]])


def _removed_per_decode_step(engine, hook: bool) -> int:
    """Launches the merges remove from a decode step of the tiny model."""
    model = engine.model.model
    visits = model.visits
    loop_visits = sum(1 for _, pass_idx in visits if pass_idx)
    # Per visit: the router GEMV, query_rel (prologue), the attention conv
    # (into the MLP mix) and residual_add (conv and add in one launch); per
    # loop visit two gate launches for each of its two gated mixes.
    removed = 4 * len(visits) + 4 * loop_visits
    if not hook:
        # A pending output's conv and add run inside the next attention
        # input mix (or the final norm), except where the loop reads it: at
        # the loop entry and at each pass restart.
        core = model.config.loop_core_layers
        reads = sum(1 for layer, _ in visits[1:] if layer == core.start)
        removed += len(visits) - reads
    return removed


@interpreted
@pytest.mark.parametrize("hook", [True, False], ids=["visit_hook", "pending"])
def test_merged_visits_match_the_unmerged_kernels(monkeypatch, hook):
    engine = body.Engine(cudagraph_mode="FULL_DECODE_ONLY")
    launches = _Launches(monkeypatch)
    monkeypatch.delenv(SWITCH, raising=False)
    got = _run(engine, launches, hook)
    merged = launches.steps
    launches.steps = []
    monkeypatch.setenv(SWITCH, "0")
    want = _run(engine, launches, hook)
    unmerged = launches.steps

    assert got.shape == want.shape
    assert float((got - want).abs().max()) <= TOL
    # The prefill step is not decode-only: same launches on both routes.
    assert merged[0] == unmerged[0]
    removed = _removed_per_decode_step(engine, hook)
    for fused, plain in zip(merged[1:], unmerged[1:]):
        assert sum(plain.values()) - sum(fused.values()) == removed, (fused, plain)
        assert fused["_conv_mix_kernel"] and fused["_prologue_kernel"]
        assert fused["_router_route_kernel"] and fused["_gate_blend_kernel"]
        for name in ("_conv_mix_kernel", "_prologue_kernel", "_gate_blend_kernel"):
            assert plain[name] == 0
