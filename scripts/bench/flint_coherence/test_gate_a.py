"""Dry-run tests for Gate A against the local mock server (no real server)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_a  # noqa: E402
import mock_server  # noqa: E402


@pytest.fixture(scope="module")
def mock():
    srv = mock_server.serve(0)
    yield srv
    srv.shutdown()


def run_gate(port, out, *extra, limit=6, conc="1,4"):
    cmd = [
        sys.executable,
        os.path.join(HERE, "gate_a.py"),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--model",
        "flint",
        "--concurrency",
        conc,
        "--out",
        str(out),
        "--timeout",
        "120",
        *extra,
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    res = None
    if os.path.exists(os.path.join(out, "results.json")):
        with open(os.path.join(out, "results.json")) as fh:
            res = json.load(fh)
    return p, res


def by_name(res):
    return {c["name"]: c for c in res["checks"]}


def test_fixtures_shape():
    meta, fx = gate_a.load_jsonl(os.path.join(HERE, "fixtures", "teacher_forced.jsonl"))
    assert meta["tools"][0]["function"]["name"] == "bash"
    assert 25 <= len(fx) <= 35
    pts = [f["ref_prompt_tokens"] for f in fx]
    assert min(pts) >= 2000 and max(pts) <= 32000
    assert max(pts) > 24000 and min(pts) < 3000
    assert len({f["repo"] for f in fx}) >= 8
    assert len({f["id"] for f in fx}) == len(fx)
    for f in fx:
        assert f["target"]["role"] == "assistant"
        assert f["context"][0]["role"] == "system"
    size = os.path.getsize(os.path.join(HERE, "fixtures", "teacher_forced.jsonl"))
    assert size < 5e6
    _, gp = gate_a.load_jsonl(os.path.join(HERE, "fixtures", "greedy_prompts.jsonl"))
    short = [p for p in gp if p["kind"] != "long"]
    longs = [p for p in gp if p["kind"] == "long"]
    assert len(short) == 20 and len(longs) == 4
    assert {p["kind"] for p in short} == {"code", "math", "prose"}
    assert all(500 <= p["max_tokens"] <= 2000 for p in short)
    for p in longs:
        assert p["expect_substring"] in p["prompt"]


def test_metrics_flag_gibberish():
    p = {"kind": "code", "id": "x"}
    good = {
        "reasoning": "",
        "content": "ok\n```python\nx = 1\n```\n",
        "token_ids": list(range(200)),
        "finish_reason": "stop",
    }
    m = gate_a.text_metrics(p, good)
    assert m["code_blocks"] == 1 and m["code_parse_errors"] == 0
    bad = dict(good, content="```python\ndef f(:\n```\n" + "的是" * 300)
    m = gate_a.text_metrics(p, bad)
    assert m["code_parse_errors"] == 1 and m["non_ascii"] > 0.5
    loop = dict(good, content="the cat sat on the mat and then " * 200)
    m = gate_a.text_metrics({"kind": "prose"}, loop)
    assert m["max_ngram_rep"] > 100 and m["zlib_ratio"] > 5
    assert gate_a.lcp([1, 2, 3], [1, 2, 4]) == 2


def test_healthy_server_passes(mock, tmp_path):
    mock_server.State.bug = False
    p, res = run_gate(mock.server_port, tmp_path / "ok", conc="1,4,16")
    assert p.returncode == 0, p.stdout + p.stderr
    assert res["pass"]
    assert all(c["pass"] for c in res["checks"])
    names = by_name(res)
    assert "A1.c16.delta_vs_c1" in names and "A2.c4.prefix_vs_c1" in names
    assert os.path.exists(tmp_path / "ok" / "summary.md")
    assert "Gate A: PASS" in (tmp_path / "ok" / "summary.md").read_text()


def test_regression_is_caught(mock, tmp_path):
    mock_server.State.bug = True
    try:
        p, res = run_gate(mock.server_port, tmp_path / "bug")
    finally:
        mock_server.State.bug = False
    assert p.returncode == 1, p.stdout + p.stderr
    names = by_name(res)
    assert names["A1.c1.mean_nll"]["pass"]
    assert names["A1.c1.late_vs_early"]["pass"]
    assert not names["A1.c4.delta_vs_c1"]["pass"]
    assert not names["A1.c4.late_vs_early"]["pass"]
    v = names["A1.c4.mean_nll"]["value"]
    assert v["mean_64p"] - v["mean_0_63"] > 1.0
    assert not names["A2.c4.prefix_vs_c1"]["pass"]
    assert not names["A2.c4.non_ascii"]["pass"]
    assert "Gate A: FAIL" in (tmp_path / "bug" / "summary.md").read_text()


def test_reference_comparison(mock, tmp_path):
    mock_server.State.bug = False
    run_gate(mock.server_port, tmp_path / "ref", limit=4, conc="1")
    p, res = run_gate(
        mock.server_port,
        tmp_path / "cur",
        "--reference",
        str(tmp_path / "ref"),
        limit=4,
        conc="1",
    )
    assert p.returncode == 0, p.stdout + p.stderr
    names = by_name(res)
    assert names["A1.c1.delta_vs_reference"]["pass"]
    assert names["A2.c1.prefix_vs_reference"]["pass"]


def test_unreachable_server_exits_2(tmp_path):
    p, _ = run_gate(9, tmp_path / "down", limit=1)
    assert p.returncode == 2
