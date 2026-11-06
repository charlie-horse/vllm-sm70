# SPDX-License-Identifier: Apache-2.0
"""TurboMind pre-capture warmup on SM70 (marked gpu).

- After the warmup, a warmed shape captured into a CUDA graph dispatches with
  kReuse; an unwarmed one gets kDefault (the control). Read from
  TM_GEMM_TRACE, so each case runs in a fresh process (_warmup_gpu_probe.py).
- TP2 (two GPUs, gloo): rank 0 tunes, rank 1 imports, both end with the same
  dispatch-cache digest.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

PROBE = pathlib.Path(__file__).with_name("_warmup_gpu_probe.py")
TRACE = re.compile(r"\[TM_GEMM_TRACE\] desc=(\S+) policy=(\S+)")


def _env(**extra):
    env = dict(os.environ)
    for name in [k for k in env if k.startswith(("TM_GEMM_", "VLLM_SM70_"))]:
        if name != "VLLM_SM70_PLUGIN_STRICT":
            env.pop(name)
    env.update(extra)
    return env


def test_warmed_shape_reuses_in_capture(warmup_gpu):
    proc = subprocess.run(
        [sys.executable, str(PROBE), "capture"],
        env=_env(TM_GEMM_TRACE="1", TM_GEMM_TRACE_LIMIT="0"),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    err = proc.stderr
    assert "@@CAPTURE@@" in err and "@@END@@" in err, err[-4000:]
    during = err.split("@@CAPTURE@@", 1)[1].split("@@END@@", 1)[0]
    policies = [p for _, p in TRACE.findall(during)]
    # One TurboMind dispatch per apply: M=4 (warmed), then M=5 (not warmed).
    assert len(policies) == 2, during
    assert policies[0].startswith("reuse"), during
    assert policies[1].startswith("default"), during
    # And the warmup itself measured before capture.
    before = err.split("@@CAPTURE@@", 1)[0]
    measured = [p for _, p in TRACE.findall(before) if p.startswith("measure")]
    assert measured, before[-4000:]


def test_tp2_ranks_share_rank0_dispatch_cache(warmup_gpu, tmp_path):
    if warmup_gpu.cuda.device_count() < 2:
        pytest.skip("requires two SM70 GPUs")
    store = tmp_path / "store"
    procs = [
        subprocess.Popen(
            [sys.executable, str(PROBE), "tp2", str(rank), str(store)],
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err[-4000:]
        outs.append(json.loads(out.strip().splitlines()[-1]))
    r0, r1 = outs
    for r in outs:
        assert r["mode"] == "coordinated" and r["source"] == 0
        assert r["agree"], outs
        assert r["counts"]["nvfp4_dense"] > 0
    assert r0["digests"] == r1["digests"] and r0["digests"][0] is not None
    assert r1["imported"] > 0


def test_lut_pins_specs_across_processes(warmup_gpu, tmp_path):
    """Run 1 exports a LUT; runs 2 and 3 (fresh processes) import it, time no
    kernel and give bitwise-identical outputs (T4c at op level)."""
    lut = tmp_path / "tm-{arch}.bin"

    def probe(**env):
        proc = subprocess.run(
            [sys.executable, str(PROBE), "lut"],
            env=_env(TM_GEMM_TRACE="1", TM_GEMM_TRACE_LIMIT="0", **env),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        policies = [p for _, p in TRACE.findall(proc.stderr)]
        return json.loads(proc.stdout.strip().splitlines()[-1]), policies

    _, first_policies = probe(
        VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES="1",
        VLLM_SM70_GEMM_LUT_EXPORT_PATH=str(lut),
    )
    assert any(p.startswith("measure") for p in first_policies)
    written = tmp_path / "tm-sm70.bin"
    assert written.exists() and pathlib.Path(str(written) + ".plan.json").exists()
    runs = [
        probe(VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES="1", VLLM_SM70_GEMM_LUT_PATH=str(lut))
        for _ in range(2)
    ]
    for out, policies in runs:
        assert out["lut_records"] > 0
        assert not any(p.startswith("measure") for p in policies), policies
    # Runs 2 and 3 only: run 1 times M=3 and M=48 after its export (they are
    # not in the plan); with the LUT those take the imported LowerBound spec.
    assert runs[0][0]["out_sha"] == runs[1][0]["out_sha"]
