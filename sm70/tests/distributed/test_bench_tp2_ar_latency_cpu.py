# SPDX-License-Identifier: Apache-2.0
"""CPU checks for sm70/tools/bench_tp2_ar_latency.py's pure-Python logic:
the us/call arithmetic, merging both ranks' samples, and argument checks.
The GPU part runs on c4130: python sm70/tools/bench_tp2_ar_latency.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench_tp2_ar_latency.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_tp2_ar_latency", TOOL)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves string annotations through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def test_import_does_not_need_torch(bench):
    # The module body must stay importable without CUDA/vLLM (spawned
    # children re-import it before they set their device).
    assert bench.DEFAULT_CALLS == 98 and bench.HIDDEN == 2048


def test_per_call_us(bench):
    # 20 replays of a 98-call graph in 24.5 ms -> 12.5 us per call.
    assert bench.per_call_us(24.5, 20, 98) == pytest.approx(12.5)
    with pytest.raises(ValueError):
        bench.per_call_us(1.0, 0, 98)


def test_summarize_merges_ranks_in_route_order(bench):
    S = bench.Sample
    samples = [
        S("fused", 1, 1, 98, 20, [13.0, 12.0, 14.0]),
        S("fused", 1, 0, 98, 20, [12.5, 12.4, 12.6]),
        S("noop", 1, 0, 98, 20, [2.0]),
        S("noop", 1, 1, 98, 20, [2.2]),
        S("fused", 2, 0, 98, 20, [13.0]),
        S("fused", 2, 1, 98, 20, [], "RuntimeError: boom"),
        S("nccl", 4, 0, 98, 20, [30.0]),  # rank 1 missing
    ]
    rows = bench.summarize(samples)
    assert [(r.route, r.batch) for r in rows] == [
        ("noop", 1),
        ("nccl", 4),
        ("fused", 1),
        ("fused", 2),
    ]
    fused1 = rows[2]
    assert fused1.rank0_us == 12.5 and fused1.rank1_us == 13.0
    assert fused1.min_us == 12.0
    assert fused1.step_us == pytest.approx(13.0 * 98)
    assert rows[3].error == "RuntimeError: boom" and rows[3].rank0_us is None
    assert "rank(s) [1]" in rows[1].error
    table = bench.format_table(rows, 98)
    assert "ERROR: RuntimeError: boom" in table and "x98" in table


def test_parse_args(bench):
    args = bench.parse_args([])
    assert args.batches == [1, 2, 4] and args.calls == 98
    assert list(args.routes) == list(bench.ROUTES)
    with pytest.raises(SystemExit):
        bench.parse_args(["--batches", "17"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--calls", "0"])
