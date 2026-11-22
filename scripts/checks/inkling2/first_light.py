#!/usr/bin/env python3
"""Inkling 2 first light: a running server against the HF goldens.

Replays the short text cases of a golden run file teacher-forced (one
/v1/completions request per case with prompt_logprobs; it starts no server)
and compares:

  logprobs  top-1 agreement, flips with a reference top-1/top-2 gap at or
            above --gap-max, a mean top-20 truncated KL (over the reference's
            listed tokens; not S5's full-vocabulary KL), and the
            |logprob delta| of the reference top-1 (max and p99), pooled over
            the output positions of all cases.
  states    the per-visit dump of `sm70.inkling2.visit_dump_dir`
            (<dir>/rank<r>/forward<n>/visit<v>_layer<l>_pass<p>.pt) against
            <states-dir>/<id>.states.npz at the case's golden
            `state_positions`: relative L2 per position, median over the
            positions, per visit; the first failing visit is reported.

The server must run with --max-num-seqs 1 and nothing else may send requests:
the forwards a request adds to the dump directory are taken as that request's
token rows, in order. Run it where the dump directory is readable. Give each
server start a fresh dump directory: a restarted server numbers its forwards
from 0 again and overwrites the old ones, which the script refuses.

Exit status 1 when a threshold fails or a file is missing.

Usage:
  first_light.py --base-url http://127.0.0.1:8000 --model NAME \\
      --goldens GOLDENS_DIR [--dump-dir DIR] [--out report.json]
  first_light.py --goldens GOLDENS_DIR --captured run.json   # compare only
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "greedy"))
import teacher_forced

# Defaults: the suggested short-context thresholds of the fp16 noise-floor
# measurement on the HF reference (each at least 3x the floor).
MIN_AGREEMENT = 0.98
GAP_MAX = 0.5
MAX_REAL_FLIPS = 0
MAX_MEAN_KL = 0.002
MAX_ABS_DELTA = 0.75
P99_ABS_DELTA = 0.25
VISIT_TOLS = "0-6:2e-3,7-17:2.5e-2,18-47:9e-2"
TOP = 20
MAX_PROMPT_TOKENS = 512
RUN_FILE = "run-text.json"

VISIT_FILE = re.compile(r"visit(\d+)_layer(\d+)_pass(\d+)\.pt")


# --- logprobs -------------------------------------------------------------


def position_kl(ref_top: list, cand_top: list) -> tuple[float, int]:
    """(sum over the reference's listed tokens of p * (log p - log q), how
    many of them the candidate does not list): a top-N truncated KL, not
    S5's full-vocabulary metric. A token the candidate does not list gets the
    candidate's smallest listed logprob."""
    cand = {int(tok): lp for tok, lp in cand_top}
    floor = min(cand.values())
    kl, missing = 0.0, 0
    for tok, lp in ref_top:
        cand_lp = cand.get(int(tok))
        if cand_lp is None:
            cand_lp, missing = floor, missing + 1
        kl += math.exp(lp) * (lp - cand_lp)
    return kl, missing


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile, as numpy's default."""
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def compare_logprobs(
    ref: dict,
    cand: dict,
    *,
    min_agreement: float = MIN_AGREEMENT,
    gap_max: float = GAP_MAX,
    max_real_flips: int = MAX_REAL_FLIPS,
    max_mean_kl: float = MAX_MEAN_KL,
    max_abs_delta: float = MAX_ABS_DELTA,
    p99_abs_delta: float = P99_ABS_DELTA,
) -> dict:
    """Run files `ref` (golden) and `cand` (served) with the same cases."""
    base = teacher_forced.compare_runs(ref, cand, 0.0, gap_max)
    cand_cases = {c["id"]: c for c in cand["cases"]}
    kls, deltas, missing, no_delta = [], [], 0, 0
    for case in ref["cases"]:
        for r, c in zip(case["positions"], cand_cases[case["id"]]["positions"]):
            kl, lost = position_kl(r["top"], c["top"])
            kls.append(kl)
            missing += lost
            ref_top, ref_lp = r["top"][0]
            cand_lp = {int(tok): lp for tok, lp in c["top"]}.get(int(ref_top))
            if cand_lp is None:
                no_delta += 1
            else:
                deltas.append(abs(cand_lp - ref_lp))
    positions = base["positions"]
    report = {
        "positions": positions,
        "agreement": base["agreement"],
        "flips": base["flips"],
        "real_flips": base["real_flips"],
        "mean_kl": sum(kls) / len(kls) if kls else None,
        "max_abs_delta": max(deltas, default=None),
        "p99_abs_delta": quantile(deltas, 0.99) if deltas else None,
        "positions_without_delta": no_delta,
        "kl_tokens_missing": missing,
        "thresholds": {
            "min_agreement": min_agreement,
            "gap_max": gap_max,
            "max_real_flips": max_real_flips,
            "max_mean_kl": max_mean_kl,
            "max_abs_delta": max_abs_delta,
            "p99_abs_delta": p99_abs_delta,
        },
        "cases": base["cases"],
    }
    failures = []
    if not positions:
        failures.append("no positions")
    if base["agreement"] < min_agreement:
        failures.append(f"top-1 agreement {base['agreement']:.4f} < {min_agreement}")
    if base["real_flips"] > max_real_flips:
        failures.append(
            f"{base['real_flips']} flips with a reference gap >= {gap_max} "
            f"(allowed {max_real_flips})"
        )
    # A position whose reference top-1 the candidate does not list has a
    # delta above every listed one.
    if no_delta:
        failures.append(
            f"{no_delta} positions where the reference top-1 is outside the "
            "served top list"
        )
    if kls and report["mean_kl"] > max_mean_kl:
        failures.append(f"mean KL {report['mean_kl']:.5f} > {max_mean_kl}")
    if deltas and report["max_abs_delta"] > max_abs_delta:
        failures.append(
            f"max |delta logprob| {report['max_abs_delta']:.3f} > {max_abs_delta}"
        )
    if deltas and report["p99_abs_delta"] > p99_abs_delta:
        failures.append(
            f"p99 |delta logprob| {report['p99_abs_delta']:.3f} > {p99_abs_delta}"
        )
    report["failures"] = failures
    report["pass"] = not failures
    return report


# --- per-visit states -----------------------------------------------------


def parse_visit_tols(spec: str) -> list[tuple[int, int, float]]:
    """ "0-6:2e-3,7-17:2.5e-2" -> [(0, 6, 2e-3), (7, 17, 2.5e-2)]."""
    bands = []
    for part in spec.split(","):
        visits, _, tol = part.partition(":")
        first, _, last = visits.partition("-")
        bands.append((int(first), int(last or first), float(tol)))
    return bands


def visit_tol(bands: list[tuple[int, int, float]], visit: int) -> float:
    for first, last, tol in bands:
        if first <= visit <= last:
            return tol
    raise ValueError(f"no tolerance for visit {visit}")


def compare_states(
    golden: dict, port: dict, positions: list[int], bands: list[tuple[int, int, float]]
) -> dict:
    """`golden`: visit_NN -> [n, H], rows at the sequence positions
    `positions`. `port`: visit index -> [T, H] over the whole sequence.
    Per visit, in order: the median over the positions of the relative L2
    of a row, against the visit's tolerance."""
    import numpy as np

    index = np.asarray(positions, dtype=np.int64)
    visits, first_failing = [], None
    names = sorted(golden)
    for name in names:
        visit = int(name.rpartition("_")[2])
        tol = visit_tol(bands, visit)
        ref = np.asarray(golden[name], dtype=np.float64)
        entry = {"visit": visit, "tol": tol}
        rows = port.get(visit)
        if rows is None:
            entry.update(ok=False, error="no dump of this visit")
        elif ref.shape[0] != index.size:
            entry.update(
                ok=False,
                error=f"{ref.shape[0]} golden rows for {index.size} positions",
            )
        elif index.size and int(index.max()) >= rows.shape[0]:
            entry.update(
                ok=False,
                error=f"position {int(index.max())} outside {rows.shape[0]} rows",
            )
        else:
            got = np.asarray(rows, dtype=np.float64)[index]
            nonfinite = int((~np.isfinite(got)).sum())
            norm = np.linalg.norm(ref, axis=-1)
            diff = np.linalg.norm(got - ref, axis=-1)
            rel = np.where(norm > 0, diff / np.where(norm > 0, norm, 1.0), diff)
            median = float(np.median(rel)) if rel.size else float("inf")
            entry.update(
                median_relative_l2=median,
                p90_relative_l2=float(np.quantile(rel, 0.9)) if rel.size else None,
                max_relative_l2=float(rel.max()) if rel.size else None,
                nonfinite=nonfinite,
                ok=nonfinite == 0 and median <= tol,
            )
        if not entry["ok"] and first_failing is None:
            first_failing = visit
        visits.append(entry)
    extra = sorted(set(port) - {int(n.rpartition("_")[2]) for n in names})
    return {
        "positions": len(positions),
        "visits": visits,
        "first_failing_visit": first_failing,
        "visits_without_golden": extra,
        "pass": bool(visits) and first_failing is None,
    }


def forward_dirs(dump_dir: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """rank<r> -> its forward directories, in forward order."""
    return {
        rank.name: sorted(p for p in rank.glob("forward*") if p.is_dir())
        for rank in sorted(dump_dir.glob("rank*"))
        if rank.is_dir()
    }


class DumpReused(Exception):
    """The dump directory holds forwards of an earlier server start."""


def reused_ranks(dump_dir: pathlib.Path) -> list[str]:
    """Ranks whose forwards were not written in forward order: a later
    server start overwrote the low forward numbers of an earlier one."""
    reused = []
    for rank, forwards in forward_dirs(dump_dir).items():
        last = -1
        for forward in forwards:
            times = [p.stat().st_mtime_ns for p in forward.glob("visit*.pt")]
            if not times:
                continue
            if min(times) < last:
                reused.append(rank)
                break
            last = max(times)
    return reused


def check_fresh(dump_dir: pathlib.Path) -> None:
    """Raise DumpReused when `dump_dir` holds an earlier server's forwards."""
    reused = reused_ranks(dump_dir)
    if reused:
        raise DumpReused(
            f"{dump_dir} ({', '.join(reused)}) holds forwards of an earlier "
            "server start, which the running server overwrites; start the "
            "server with a fresh, empty sm70.inkling2.visit_dump_dir per start"
        )


def load_forwards(forwards: list[pathlib.Path], num_tokens: int) -> dict:
    """visit index -> [num_tokens, H]: the rows of a request's forwards, in
    order. One forward may carry padded rows after the request's; several
    forwards (a chunked prefill) must add up to the request."""
    import numpy as np
    import torch

    per_visit: dict[int, list] = {}
    for forward in forwards:
        for path in sorted(forward.iterdir()):
            match = VISIT_FILE.fullmatch(path.name)
            if match:
                state = torch.load(path, map_location="cpu")
                per_visit.setdefault(int(match[1]), []).append(state.float().numpy())
    out = {}
    for visit, chunks in per_visit.items():
        if len(chunks) != len(forwards):
            raise ValueError(
                f"visit {visit}: {len(chunks)} dumps in {len(forwards)} forwards"
            )
        rows = np.concatenate(chunks, axis=0)
        if rows.shape[0] < num_tokens or (
            len(forwards) > 1 and rows.shape[0] != num_tokens
        ):
            raise ValueError(
                f"visit {visit}: {rows.shape[0]} rows in {len(forwards)} "
                f"forwards for a request of {num_tokens} tokens"
            )
        out[visit] = rows[:num_tokens]
    return out


def new_forwards(
    before: dict[str, list[pathlib.Path]],
    dump_dir: pathlib.Path,
    num_visits: int,
    settle: float,
) -> dict[str, list[pathlib.Path]]:
    """The forwards written since `before`, once every rank has all visits
    of each of them (waits up to `settle` seconds)."""
    deadline = time.monotonic() + settle
    while True:
        now = forward_dirs(dump_dir)
        new = {
            rank: [p for p in dirs if p not in set(before.get(rank, []))]
            for rank, dirs in now.items()
        }
        complete = all(
            dirs and all(len(list(d.glob("visit*.pt"))) >= num_visits for d in dirs)
            for dirs in new.values()
        )
        if (new and complete) or time.monotonic() >= deadline:
            return new
        time.sleep(0.2)


def load_golden_states(path: pathlib.Path) -> dict:
    import numpy as np

    with np.load(path) as data:
        return {name: data[name] for name in data.files if name.startswith("visit_")}


# --- driver ----------------------------------------------------------------


def select_cases(run: dict, ids: list[str] | None, max_prompt_tokens: int) -> list:
    """The replayable cases: named ones, or every case with a full
    generation and a prompt of at most `max_prompt_tokens` tokens."""
    cases = []
    for case in run["cases"]:
        if ids is not None:
            keep = case["id"] in ids
        else:
            keep = (
                not case.get("prefill_only")
                and len(case["prompt_ids"]) <= max_prompt_tokens
            )
        if keep:
            cases.append(case)
    if ids is not None and len(cases) != len(set(ids)):
        missing = sorted(set(ids) - {c["id"] for c in cases})
        raise ValueError(f"cases not in the run file: {missing}")
    return cases


def request(base_url: str, model: str, case: dict, top: int, timeout: float) -> dict:
    body = teacher_forced.build_body(model, case, top)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def run_cases(args: argparse.Namespace, cases: list[dict], bands: list) -> dict:
    """Send every case; returns {"run": the served run file, "states": per
    case and rank the state comparison}."""
    dump_dir = pathlib.Path(args.dump_dir) if args.dump_dir else None
    states_dir = pathlib.Path(args.states_dir or pathlib.Path(args.goldens) / "cases")
    served, states = [], {}
    if dump_dir is not None:
        check_fresh(dump_dir)
    for case in cases:
        before = forward_dirs(dump_dir) if dump_dir else {}
        response = request(args.base_url, args.model, case, args.top, args.timeout)
        served.append(
            {
                "id": case["id"],
                "prompt_ids": case["prompt_ids"],
                "output_ids": case["output_ids"],
                "positions": teacher_forced.positions_from_response(response, case),
            }
        )
        print(case["id"], len(case["output_ids"]), "positions", flush=True)
        if dump_dir is None:
            continue
        golden_path = states_dir / f"{case['id']}.states.npz"
        if not golden_path.is_file():
            states[case["id"]] = {"pass": False, "error": f"no {golden_path}"}
            continue
        golden = load_golden_states(golden_path)
        num_tokens = len(case["prompt_ids"]) + len(case["output_ids"])
        positions = [p for p in case["state_positions"] if p < num_tokens]
        kept = [i for i, p in enumerate(case["state_positions"]) if p < num_tokens]
        golden = {name: value[kept] for name, value in golden.items()}
        ranks = {}
        new = new_forwards(before, dump_dir, len(golden), args.settle)
        if not all(new.values()):
            check_fresh(dump_dir)
        if not new:
            ranks["rank0"] = {"pass": False, "error": f"no dump under {dump_dir}"}
        for rank, forwards in new.items():
            if not forwards:
                ranks[rank] = {
                    "pass": False,
                    "error": f"no dump of this request under {dump_dir / rank}",
                }
                continue
            try:
                port = load_forwards(forwards, num_tokens)
            except ValueError as error:
                ranks[rank] = {"pass": False, "error": str(error)}
                continue
            ranks[rank] = compare_states(golden, port, positions, bands)
            ranks[rank]["forwards"] = [p.name for p in forwards]
        states[case["id"]] = {
            "pass": all(r["pass"] for r in ranks.values()),
            "ranks": ranks,
        }
    return {"run": {"cases": served}, "states": states}


def print_report(logprobs: dict, states: dict) -> None:
    for case in logprobs["cases"]:
        print(
            f"{case['id']}: {case['agree']}/{case['positions']} top-1, "
            f"max |delta| {case['max_abs_delta']}"
        )
        for flip in case["flips"]:
            kind = "near tie" if flip["near_tie"] else "REAL"
            print(
                f"  flip at {flip['pos']}: {flip['ref_top1']} -> "
                f"{flip['cand_top1']}, gap {flip['gap']} ({kind})"
            )
    print(
        f"logprobs: agreement {logprobs['agreement']:.4f} over "
        f"{logprobs['positions']} positions, {logprobs['real_flips']} real flips, "
        f"mean KL {logprobs['mean_kl']}, max |delta| {logprobs['max_abs_delta']}, "
        f"p99 {logprobs['p99_abs_delta']}: "
        f"{'pass' if logprobs['pass'] else 'FAIL'}"
    )
    for failure in logprobs["failures"]:
        print(f"  {failure}")
    for case_id, result in states.items():
        if "error" in result:
            print(f"states {case_id}: {result['error']}: FAIL")
            continue
        for rank, report in result["ranks"].items():
            if "error" in report:
                print(f"states {case_id} {rank}: {report['error']}: FAIL")
            elif report["pass"]:
                last = report["visits"][-1]
                print(
                    f"states {case_id} {rank}: {len(report['visits'])} visits "
                    f"within tolerance (last visit median "
                    f"{last['median_relative_l2']:.3e}): pass"
                )
            else:
                visit = report["first_failing_visit"]
                entry = next(v for v in report["visits"] if v["visit"] == visit)
                detail = entry.get("error") or (
                    f"median relative L2 {entry['median_relative_l2']:.3e} "
                    f"(tol {entry['tol']}), {entry['nonfinite']} non-finite"
                )
                print(
                    f"states {case_id} {rank}: first failing visit {visit}: "
                    f"{detail}: FAIL"
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--goldens", required=True, help="directory of the run files")
    ap.add_argument("--run-file", default=RUN_FILE)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--captured", help="a served run file: compare, send nothing")
    ap.add_argument("--cases", help="comma-separated case ids (default: short ones)")
    ap.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--dump-dir", help="the server's sm70.inkling2.visit_dump_dir")
    ap.add_argument("--states-dir", help="golden .states.npz (default GOLDENS/cases)")
    ap.add_argument("--settle", type=float, default=10.0)
    ap.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT)
    ap.add_argument("--gap-max", type=float, default=GAP_MAX)
    ap.add_argument("--max-real-flips", type=int, default=MAX_REAL_FLIPS)
    ap.add_argument("--max-mean-kl", type=float, default=MAX_MEAN_KL)
    ap.add_argument("--max-abs-delta", type=float, default=MAX_ABS_DELTA)
    ap.add_argument("--p99-abs-delta", type=float, default=P99_ABS_DELTA)
    ap.add_argument("--visit-tols", default=VISIT_TOLS)
    ap.add_argument("--save-run", help="write the served run file here")
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)
    if not args.captured and not (args.base_url and args.model):
        ap.error("give --base-url and --model, or --captured")

    with open(pathlib.Path(args.goldens) / args.run_file) as f:
        golden_run = json.load(f)
    ids = args.cases.split(",") if args.cases else None
    cases = select_cases(golden_run, ids, args.max_prompt_tokens)
    if not cases:
        print("no case selected: FAIL")
        return 1
    bands = parse_visit_tols(args.visit_tols)
    if args.captured:
        with open(args.captured) as f:
            served = json.load(f)
        wanted = {c["id"] for c in cases}
        served = {"cases": [c for c in served["cases"] if c["id"] in wanted]}
        states = {}
    else:
        try:
            result = run_cases(args, cases, bands)
        except DumpReused as error:
            print(f"{error}: FAIL")
            return 1
        served, states = result["run"], result["states"]
        if args.save_run:
            with open(args.save_run, "w") as f:
                json.dump(served, f)
    try:
        logprobs = compare_logprobs(
            {"cases": cases},
            served,
            min_agreement=args.min_agreement,
            gap_max=args.gap_max,
            max_real_flips=args.max_real_flips,
            max_mean_kl=args.max_mean_kl,
            max_abs_delta=args.max_abs_delta,
            p99_abs_delta=args.p99_abs_delta,
        )
    except ValueError as error:
        print(f"{error}: FAIL")
        return 1
    print_report(logprobs, states)
    ok = logprobs["pass"] and all(s["pass"] for s in states.values())
    print("first light:", "pass" if ok else "FAIL")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"pass": ok, "logprobs": logprobs, "states": states}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
