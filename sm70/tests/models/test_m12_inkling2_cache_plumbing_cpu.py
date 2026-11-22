# SPDX-License-Identifier: Apache-2.0
"""M12 CPU checks: the cache-plumbing stub (inkling2_stub.py) under vLLM's
real scheduler, hybrid KV cache manager and V1 model runner.

Each case starts an engine in a fresh process on the CPU platform, with
torch stand-ins for the three compiled pieces a torch-only install lacks
(slot-mapping kernel, greedy argmax, CPU memory setup). The model is the
tiny fixture config with 16-wide heads: 48 K/V caches (window 8) and 48
conv states (window 4), block size 16, 64 tokens per batch. Every generated
token must equal the host-side prediction.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

HERE = pathlib.Path(__file__).resolve().parent
STUB = HERE / "inkling2_stub.py"
TINY = HERE / "fixtures" / "inkling2_tiny" / "config.json"
BATCHED = 64


def _run(tmp_path, **options):
    config = json.loads(TINY.read_text())
    config.pop("quantization_config", None)
    config["text_config"].update(
        head_dim=16, swa_head_dim=16, vocab_size=4096, unpadded_vocab_size=4096
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    options = {
        "config": str(tmp_path / "config.json"),
        "work_dir": str(tmp_path),
        "cpu": True,
        "block_size": 16,
        "batched_tokens": BATCHED,
        "max_model_len": 2048,
        "kv_bytes": 64 << 20,
        **options,
    }
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE.parents[1]), *filter(None, [env.get("PYTHONPATH")])]
    )
    proc = subprocess.run(
        [sys.executable, str(STUB), json.dumps(options)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1500,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-4000:]
    return json.loads(lines[-1].removeprefix("RESULT "))


def _cases(result):
    return {row["case"]: row for row in result["cases"]}


def test_reference_matches_a_hand_computation():
    sys.path.insert(0, str(HERE))
    try:
        import inkling2_stub as stub
    finally:
        sys.path.remove(str(HERE))
    import numpy as np

    ref = stub.Reference.__new__(stub.Reference)
    ref.plan = [(stub.SLIDING, 4), (stub.FULL, 0), (stub.CONV, 3)]
    ref.span = 100
    tokens = [7, 9, 11]
    mixed = [
        int(stub.Reference.mix(np.array([t]), np.array([i]))[0])
        for i, t in enumerate(tokens)
    ]
    h = [sum(mixed[: i + 1]) & stub.MASK for i in range(3)]
    # Reads of position 2: sliding p-1, p-3 (none), p-1-((14+0)%3) = -1 (none);
    # full p-1, 0, (62+1)%2 = 1; conv p-1, p-2.
    digest = (
        h[2] + 3 * h[1] + (19 * h[1] + 21 * h[0] + 23 * h[1]) + (35 * h[1] + 37 * h[0])
    )
    assert (
        int(ref.next_tokens(tokens)[2]) == stub.TOKEN_BASE + (digest & stub.MASK) % 100
    )


def test_chunked_prefill_concurrency_and_prefix_hits(tmp_path):
    result = _run(tmp_path, mode="main")
    cases = _cases(result)
    assert set(cases) == {
        "chunked",
        "concurrent_short",
        "concurrent_long",
        "prefix_base",
        "prefix_resume",
        "prefix_repeat",
    }
    assert all(row["ok"] for row in cases.values()), cases
    assert cases["chunked"]["prompt_tokens"] > 2 * BATCHED
    assert cases["prefix_base"]["cached_tokens"] == 0
    assert cases["prefix_resume"]["cached_tokens"] > 0
    assert cases["prefix_repeat"]["cached_tokens"] > 0
    assert result["kv_cache_layout"] == "LBNHC"
    assert result["preemptions"] == 0
    groups = result["groups"]
    assert len(groups) == 12 and all(g["layers"] == 8 for g in groups)
    windowed = [g for g in groups if g["window"] is not None]
    assert sorted((g["window"], g["block_size"]) for g in windowed) == sorted(
        [(8, 16)] * 5 + [(4, 4)] * 6
    )
    # In-flight reservation: cdiv(window - 1 + B, block) + 1, within a block.
    for group in windowed:
        assert group["cap"] - 1 <= group["peak"] <= group["cap"], group


def test_preempted_requests_recompute_their_caches(tmp_path):
    result = _run(tmp_path, mode="preempt")
    assert result["preemptions"] > 0
    assert all(row["ok"] for row in result["cases"]), result["cases"]


def test_a_read_through_another_requests_blocks_changes_the_output(tmp_path):
    result = _run(tmp_path, mode="main", fault=True)
    cases = _cases(result)
    assert not cases["concurrent_short"]["ok"]
    assert not cases["concurrent_long"]["ok"]
