#!/usr/bin/env python3
"""Throughput + paired comparison of a run against a baseline (same ids).

Usage::

    .venv/bin/python -I scripts/throughput.py --run runs/slice50_s20 \\
        [--baseline runs] [--ids slice50_ids.txt] [--attempt 1] [--json]

Per run (all done instances of attA in that run dir):
  wall          first run.log timestamp -> last done_ marker mtime
  agg tok/s     sum completion tokens / wall
  stream tok/s  sum completion tokens / sum agent_s (per-sequence speed)
  median agent_s per task
Paired (ids in --ids, scored = done and not infra in both runs): pass@1 of
each run, discordant counts b (baseline only) / c (run only), McNemar exact
two-sided p = min(1, 2 * P[X <= min(b, c)]), X ~ Binom(b + c, 1/2).
Classification comes from analyze.py (same directory or ../../scripts).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_analyze():
    for p in (HERE / "analyze.py", HERE.parent.parent / "scripts" / "analyze.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("analyze", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("analyze.py not found")


TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def run_wall(att_dir: Path, ids: set[str]) -> tuple[float | None, str, str]:
    starts, ends = [], []
    for iid in ids:
        log = att_dir / "log" / f"{iid}.run.log"
        done = att_dir / "eval" / f"done_{iid}"
        if log.exists():
            with open(log, errors="replace") as f:
                m = TS_RE.match(f.readline())
            if m:
                starts.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                              .timestamp())
        if done.exists():
            ends.append(done.stat().st_mtime)
    if not starts or not ends:
        return None, "-", "-"
    t0, t1 = min(starts), max(ends)
    iso = lambda t: dt.datetime.fromtimestamp(t).strftime("%F %T")  # noqa: E731
    return t1 - t0, iso(t0), iso(t1)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def run_summary(A, runs: Path, attempt: int, only: set[str] | None) -> dict:
    recs = [r for r in A.collect(runs, [attempt], False, False)
            if only is None or r["id"] in only]
    ids = {r["id"] for r in recs}
    wall, t0, t1 = run_wall(runs / f"att{attempt}", ids)
    tok = sum(r["completion_tokens"] or 0 for r in recs)
    agent = [r["agent_s"] for r in recs if isinstance(r["agent_s"], (int, float))]
    return dict(
        runs=str(runs), done=len(recs), wall_s=wall, start=t0, end=t1,
        completion_tokens=tok,
        agg_tok_s=tok / wall if wall else None,
        stream_tok_s=tok / sum(agent) if agent and sum(agent) else None,
        median_agent_s=statistics.median(agent) if agent else None,
        sum_agent_h=sum(agent) / 3600 if agent else None,
        recs={r["id"]: r for r in recs},
    )


def paired(A, base: dict, run: dict, ids: list[str]) -> dict:
    ok = lambda r: r is not None and r["category"] not in A.INFRA  # noqa: E731
    pairs = [(base["recs"].get(i), run["recs"].get(i)) for i in ids]
    pairs = [(b, r) for b, r in pairs if ok(b) and ok(r)]
    res = lambda r: r["category"] == "resolved"  # noqa: E731
    kb = sum(res(b) for b, _ in pairs)
    kr = sum(res(r) for _, r in pairs)
    b_only = sorted(b["id"] for b, r in pairs if res(b) and not res(r))
    r_only = sorted(r["id"] for b, r in pairs if res(r) and not res(b))
    n = len(pairs)
    return dict(
        n_pairs=n, n_ids=len(ids),
        baseline_resolved=kb, run_resolved=kr,
        baseline_pass_at_1=kb / n if n else None,
        run_pass_at_1=kr / n if n else None,
        baseline_only=len(b_only), run_only=len(r_only),
        baseline_only_ids=b_only, run_only_ids=r_only,
        mcnemar_exact_p=mcnemar_exact(len(b_only), len(r_only)),
    )


def fmt(x, nd=1):
    return "-" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run dir holding attA/")
    ap.add_argument("--baseline", default=None, help="baseline run dir")
    ap.add_argument("--ids", default=None, help="id list for the paired test")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--baseline-attempt", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    A = load_analyze()
    root = Path(args.run).resolve()
    base_dir = Path(args.baseline) if args.baseline else root.parent
    ids = ([x.strip() for x in open(args.ids) if x.strip()] if args.ids
           else None)
    run = run_summary(A, root, args.attempt, None)
    if ids is None:
        ids = sorted(run["recs"])
    base = run_summary(A, base_dir, args.baseline_attempt, None)
    base_ids = {k: v for k, v in base["recs"].items() if k in set(ids)}
    agent_b = [r["agent_s"] for r in base_ids.values()
               if isinstance(r["agent_s"], (int, float))]
    tok_b = sum(r["completion_tokens"] or 0 for r in base_ids.values())
    base_slice = dict(
        done=len(base_ids), completion_tokens=tok_b,
        median_agent_s=statistics.median(agent_b) if agent_b else None,
        stream_tok_s=tok_b / sum(agent_b) if agent_b and sum(agent_b) else None,
        sum_agent_h=sum(agent_b) / 3600 if agent_b else None,
    )
    P = paired(A, base, run, ids)
    out = dict(run={k: v for k, v in run.items() if k != "recs"},
               baseline_full={k: v for k, v in base.items() if k != "recs"},
               baseline_same_ids=base_slice, paired=P)
    if args.json:
        print(json.dumps(out, indent=1))
        return
    L = ["| run | done | wall h | agg compl tok/s | stream tok/s "
         "| median agent s | sum agent h |", "|---|---|---|---|---|---|---|"]
    for name, s in (("run", run), ("baseline (full run)", base),
                    ("baseline (same ids)", base_slice)):
        wall = s.get("wall_s")
        L.append(f"| {name} | {s['done']} | {fmt(wall / 3600 if wall else None, 2)} "
                 f"| {fmt(s.get('agg_tok_s'))} | {fmt(s['stream_tok_s'])} "
                 f"| {fmt(s['median_agent_s'])} | {fmt(s['sum_agent_h'], 2)} |")
    p = P
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    L += ["", f"Paired on {p['n_pairs']}/{p['n_ids']} ids (scored in both):",
          f"- baseline pass@1 {pct(p['baseline_pass_at_1'])} "
          f"({p['baseline_resolved']}), run pass@1 {pct(p['run_pass_at_1'])} "
          f"({p['run_resolved']})",
          f"- discordant: baseline-only {p['baseline_only']}, "
          f"run-only {p['run_only']}; McNemar exact p = "
          f"{p['mcnemar_exact_p']:.4g}"]
    for k in ("baseline_only_ids", "run_only_ids"):
        if p[k]:
            L.append(f"- {k}: {', '.join(p[k])}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
