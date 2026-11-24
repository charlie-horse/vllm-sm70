# SPDX-License-Identifier: Apache-2.0
"""GPU check (marked gpu): the cache-plumbing stub (inkling2_stub.py) on
a real engine at TP2.

Needs two cc 7.0 GPUs with about ``INKLING2_STUB_KV_BYTES`` (default 4 GiB)
free on each. Each case starts an engine in a fresh process: the Inkling 2
Mini config.json, dummy weights, fp16, hybrid KV cache manager, prefix
caching, block size 32, breakable CUDA graphs, on the V1 or the V2 runner.
The engine-config hook of the real model runs for the stub.

    pytest -m gpu sm70/tests/models/test_m12_inkling2_cache_plumbing_gpu.py -s

Every generated token must equal the host-side prediction; a read of another
request's or another position's record changes it. Each case prints its
result (pool size, peak blocks per KV group, preemptions) for the record.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

HERE = pathlib.Path(__file__).resolve().parent
STUB = HERE / "inkling2_stub.py"
MINI = HERE / "data" / "inkling2_mini" / "config.json"
TP = 2
BLOCK = 32
KV_BYTES = int(os.environ.get("INKLING2_STUB_KV_BYTES", str(4 << 30)))
# Conv-state bytes one request may reserve per GPU at B = 2048.
CONV_LIMIT = 2_000_000_000


@pytest.fixture(scope="module")
def sm70_pair():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if not torch.cuda.is_available() or torch.cuda.device_count() < TP:
        pytest.skip("requires two SM70 GPUs")
    if any(torch.cuda.get_device_capability(i) != (7, 0) for i in range(TP)):
        pytest.skip("requires two SM70 GPUs")


def _run(tmp_path, runner, **options):
    options = {
        "config": str(MINI),
        "work_dir": str(tmp_path),
        "tp": TP,
        "block_size": BLOCK,
        "max_model_len": 16384,
        "kv_bytes": KV_BYTES,
        **options,
    }
    env = dict(
        os.environ,
        VLLM_USE_BREAKABLE_CUDAGRAPH="1",
        VLLM_USE_V2_MODEL_RUNNER="1" if runner == "v2" else "0",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE.parents[1]), *filter(None, [env.get("PYTHONPATH")])]
    )
    proc = subprocess.run(
        [sys.executable, str(STUB), json.dumps(options)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-6000:]
    result = json.loads(lines[-1].removeprefix("RESULT "))
    print(f"\n{runner} {json.dumps(result)}")
    return result


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("batched", [512, 2048])
def test_chunked_prefill_concurrency_and_prefix_hits(
    sm70_pair, tmp_path, runner, batched
):
    result = _run(tmp_path, runner, mode="main", batched_tokens=batched)
    cases = {row["case"]: row for row in result["cases"]}
    assert all(row["ok"] for row in cases.values()), cases
    assert cases["prefix_resume"]["cached_tokens"] > 0
    assert cases["prefix_repeat"]["cached_tokens"] > 0
    assert result["kv_cache_layout"] == "LBNHC"
    groups = result["groups"]
    assert len(groups) == 12 and all(g["layers"] == 8 for g in groups)
    windowed = [g for g in groups if g["window"] is not None]
    assert sorted((g["window"], g["block_size"]) for g in windowed) == sorted(
        [(1024, 32)] * 5 + [(4, 4)] * 6
    )
    # In-flight reservation: cdiv(window - 1 + B, block) + 1, within a block.
    for group in windowed:
        assert group["cap"] - 1 <= group["peak"] <= group["cap"], group
    conv_bytes = sum(
        g["peak"] * g["page_bytes"] * g["layers"] for g in windowed if g["window"] == 4
    )
    if batched == 2048:
        assert conv_bytes <= CONV_LIMIT


@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_preempted_requests_recompute_their_caches(sm70_pair, tmp_path, runner):
    result = _run(tmp_path, runner, mode="preempt", batched_tokens=512)
    assert result["preemptions"] > 0
    assert all(row["ok"] for row in result["cases"]), result["cases"]
