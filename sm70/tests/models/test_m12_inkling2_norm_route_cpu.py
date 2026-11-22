# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 model body on the Triton norm route (inkling2_sm70/norms.py),
on CPU.

The model of test_m12_inkling2_model_cpu.py runs a prefill and two decode
steps on the tiny checkpoint twice: on the vendored kernels through Triton's
interpreter with VLLM_SM70_INKLING_TRITON_NORMS=1, and without it on the torch
ops. Every visit's state, the final state and the logits agree, and the
torch route makes no kernel call.

The interpreted case runs in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent it shows as skipped.
"""

from __future__ import annotations

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

SWITCH = "VLLM_SM70_INKLING_TRITON_NORMS"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
LENGTH, PROMPT = 7, 5
TOL = 1e-4


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted case to pass, not skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    env.pop(SWITCH, None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu",
            "-k",
            "not test_interpreted_suite",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    assert re.search(r"\b2 passed", summary), summary
    assert "skipped" not in summary, summary


def _kernel_calls(monkeypatch) -> dict[str, int]:
    from vllm_sm70.models.inkling2_sm70 import norms

    calls: dict[str, int] = {}

    def counted(name, real):
        def wrapper(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return real(*args, **kwargs)

        return wrapper

    for module, name in (
        (norms.norm_ops, "rms_norm"),
        (norms.attn_res_ops, "attn_res"),
        (norms.loop_ops, "pair_rms_norm"),
        (norms.loop_ops, "blend_rms_norm"),
    ):
        monkeypatch.setattr(module, name, counted(name, getattr(module, name)))
    return calls


def _run(engine) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5)
    vocab = body._raw_config()["text_config"]["unpadded_vocab_size"]
    ids = torch.randint(0, vocab, (LENGTH,), generator=generator)
    return engine.run(hx.decode_plan("a", LENGTH, PROMPT), {"a": ids})["a"]


def test_cpu_model_stays_on_torch_without_the_interpreter(monkeypatch):
    if INTERPRETED:
        monkeypatch.setenv(SWITCH, "0")
    calls = _kernel_calls(monkeypatch)
    rows = _run(body.Engine())
    assert rows.shape[0] == LENGTH and torch.isfinite(rows).all()
    assert calls == {}


def test_model_on_interpreted_kernels_matches_the_torch_route(monkeypatch):
    if not INTERPRETED:
        pytest.skip("runs in the TRITON_INTERPRET=1 child (test_interpreted_suite)")
    engine = body.Engine()
    calls = _kernel_calls(monkeypatch)
    monkeypatch.setenv(SWITCH, "1")
    got = _run(engine)
    visits = engine.model.model.visits
    steps = 1 + LENGTH - PROMPT
    loop_visits = sum(1 for _, pass_idx in visits if pass_idx)
    # Per forward: the embedding norm and one bank entry per block start.
    starts = sum(
        1 for layer, _ in visits if engine.model.model.layers[layer].starts_block
    )
    assert calls["rms_norm"] == steps * (1 + starts + 1)
    # Per visit the attention mix and the MLP mix, a message mix per loop
    # visit, and the final mix; the first layer has no attention mix.
    assert calls["attn_res"] == steps * (2 * len(visits) - 1 + loop_visits + 1)
    assert calls["pair_rms_norm"] == calls["blend_rms_norm"] == steps * 2 * loop_visits
    used = dict(calls)
    monkeypatch.setenv(SWITCH, "0")
    want = _run(engine)
    assert calls == used
    assert got.shape == want.shape
    split = engine.num_visits * engine.hidden
    diff = (got - want).abs()
    per_visit = diff[:, :split].unflatten(1, (engine.num_visits, -1)).amax((0, 2))
    assert per_visit.max() <= TOL, (int(per_visit.argmax()), float(per_visit.max()))
    assert diff[:, split:].max() <= TOL
