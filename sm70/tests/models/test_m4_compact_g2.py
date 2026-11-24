# SPDX-License-Identifier: Apache-2.0
"""The compact MoE slot route (VLLM_SM70_FLINT_MOE_COMPACT) on vs off,
teacher-forced, at TP2: the correctness gate for that route.

Same harness and limits as test_m10_flint_router.py and T4a in
test_m5_flint_sm70.py: teacher-forced decode logprobs
(sm70/scripts/m5_teacher_forced.py) of "on" vs "off" within 1.5x the fork's
eager_vs_graphs anchor per set (mean and max). max_num_seqs 4 makes decode
steps at M = 3..4 (M*top_k = 18..24 slots) take the compact route; M <= 2
stays on the MoE decode kernel. Free-running greedy equality is not gated:
compact changes the grouping, not the math, but near-tie flips diverge
free runs.

Needs 2 GPUs, FLINT_SM70_CHECKPOINT, FLINT_SM70_T4_SEQS_DIR and
FLINT_SM70_T4_ENVELOPE. Otherwise it is skipped.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch

COMPACT_SWITCH = "VLLM_SM70_FLINT_MOE_COMPACT"
# vllm_sm70/quant/nvfp4_moe.py logs one of these (info_once) at method init.
COMPACT_ON_LOG = "compact slot groups enabled"
COMPACT_OFF_LOG = "compact slot groups DISABLED"


def _gpu_ok(n=1) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= n
        and torch.cuda.get_device_capability() == (7, 0)
    )


two_gpus = pytest.mark.skipif(not _gpu_ok(2), reason="needs 2 V100 (TP2)")
CKPT = os.environ.get("FLINT_SM70_CHECKPOINT")
needs_ckpt = pytest.mark.skipif(CKPT is None, reason="set FLINT_SM70_CHECKPOINT")
T4_READY = bool(
    os.environ.get("FLINT_SM70_T4_SEQS_DIR")
    and os.environ.get("FLINT_SM70_T4_ENVELOPE")
)


@functools.cache
def _m5_tests():
    """test_m5_flint_sm70.py (T4 harness helpers: sequence sets, fork envelope,
    scoring LLM, per-set gate), loaded by path under a private name."""
    path = pathlib.Path(__file__).with_name("test_m5_flint_sm70.py")
    spec = importlib.util.spec_from_file_location("_m5_t4_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@pytest.mark.skipif(
    not T4_READY,
    reason="set FLINT_SM70_T4_SEQS_DIR and FLINT_SM70_T4_ENVELOPE (fork calibration)",
)
def test_g2c_ckpt_tp2_compact_on_off_within_envelope(monkeypatch, capfd):
    """Teacher-forced decode, compact route on vs off, batched (max_num_seqs 4),
    within 1.5x the fork's eager_vs_graphs anchor per set. Each arm's worker
    log must show its route decision (enabled / DISABLED)."""
    m5 = _m5_tests()
    tf = m5._tf_module()
    seq_sets = m5._seq_sets()
    env = m5._envelope(seq_sets)
    arms = {}
    for name, value in (("on", None), ("off", "0")):
        if value is None:
            monkeypatch.delenv(COMPACT_SWITCH, raising=False)
        else:
            monkeypatch.setenv(COMPACT_SWITCH, value)
        capfd.readouterr()
        llm = m5._llm(**tf.scoring_llm_kwargs(4))
        try:
            scores = {
                k: tf.score_seqs(llm, seqs, True, False, one_at_a_time=False)
                for k, seqs in seq_sets.items()
            }
        finally:
            m5._shutdown(llm)
        out, err = capfd.readouterr()
        sys.stdout.write(out + err)
        arms[name] = (scores, out + err)
    on_logs, off_logs = arms["on"][1], arms["off"][1]
    assert COMPACT_ON_LOG in on_logs and COMPACT_OFF_LOG not in on_logs
    assert COMPACT_OFF_LOG in off_logs and COMPACT_ON_LOG not in off_logs
    pair = tf.GATE_ANCHOR_PAIR["T4a"]
    anchor = env["pairs"][pair]["decode"]["anchor"]
    failures = m5._gate_sets(
        "G2c",
        "compact_on_vs_off",
        anchor,
        "decode",
        arms["on"][0],
        arms["off"][0],
        None,
        env,
    )
    assert not failures, f"G2c outside 1.5x the fork {pair} anchor: {failures}"
