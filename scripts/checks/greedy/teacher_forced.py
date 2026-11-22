#!/usr/bin/env python3
"""Teacher-forced comparison of a served model against a reference.

A run file is {"cases": [{"id", "prompt_ids", "output_ids", "positions"}]}.
"positions" has one entry per output token: {"top": [[token id, logprob],
...]} with the most likely token first and at least the top 2. A reference
run comes from the HF model or from another served configuration.

  capture   Send prompt_ids + output_ids of every case to /v1/completions
            with prompt_logprobs and write a run file.
  compare   REF CAND run files: per-position top-1 agreement, the logprob
            delta of the reference top-1, and every flip classified by the
            reference top-1/top-2 gap (near tie below --gap-max).
  states    REF CAND .npz files of per-visit states: relative L2 per visit,
            naming the first visit above --tol.

compare and states exit 1 on failure.

Usage:
  teacher_forced.py capture --base-url URL --model NAME --cases hf.json --out a.json
  teacher_forced.py compare hf.json a.json      # served vs HF
  teacher_forced.py compare a.json b.json       # cold vs cached, TP1 vs TP2, ...
  teacher_forced.py states hf.npz port.npz --tol 1e-2
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

MIN_AGREEMENT = 0.99
GAP_MAX = 0.5  # the near-tie rule of scripts/bench/flint_coherence/near_tie.py
TOL_SHORT = 1e-2
TOL_LONG = 2e-2


def build_body(model: str, case: dict, top: int = 2) -> dict:
    return {
        "model": model,
        "prompt": [*case["prompt_ids"], *case["output_ids"]],
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": top,
        "add_special_tokens": False,
    }


def positions_from_response(response: dict, case: dict) -> list[dict]:
    """The output-token entries of the response's prompt_logprobs."""
    logprobs = response["choices"][0]["prompt_logprobs"]
    start, n = len(case["prompt_ids"]), len(case["output_ids"])
    if len(logprobs) != start + n:
        raise ValueError(
            f"{case['id']}: {len(logprobs)} prompt_logprobs for {start + n} tokens"
        )
    if start == 0:
        raise ValueError(f"{case['id']}: the first token has no logprobs")
    positions = []
    for entry in logprobs[start:]:
        top = sorted(
            ([int(tok), val["logprob"]] for tok, val in entry.items()),
            key=lambda pair: -pair[1],
        )
        positions.append({"top": top})
    return positions


def capture(base_url: str, model: str, cases: list[dict], top: int, timeout: float):
    out = []
    for case in cases:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(build_body(model, case, top)).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as r:
            response = json.loads(r.read())
        out.append(
            {
                "id": case["id"],
                "prompt_ids": case["prompt_ids"],
                "output_ids": case["output_ids"],
                "positions": positions_from_response(response, case),
            }
        )
        print(case["id"], len(case["output_ids"]), "positions", flush=True)
    return {"cases": out}


def compare_case(ref: dict, cand: dict, gap_max: float = GAP_MAX) -> dict:
    for key in ("prompt_ids", "output_ids"):
        if ref[key] != cand[key]:
            raise ValueError(f"{ref['id']}: {key} differ between the two runs")
    if len(ref["positions"]) != len(cand["positions"]):
        raise ValueError(f"{ref['id']}: position counts differ")
    agree, flips, deltas = 0, [], []
    for pos, (r, c) in enumerate(zip(ref["positions"], cand["positions"])):
        ref_top, ref_lp = r["top"][0]
        cand_lp = dict(map(tuple, c["top"])).get(ref_top)
        if cand_lp is not None:
            deltas.append(cand_lp - ref_lp)
        cand_top = c["top"][0][0]
        if cand_top == ref_top:
            agree += 1
            continue
        gap = ref_lp - r["top"][1][1] if len(r["top"]) > 1 else None
        flips.append(
            {
                "pos": pos,
                "ref_top1": ref_top,
                "cand_top1": cand_top,
                "gap": gap,
                "near_tie": gap is not None and gap < gap_max,
                "cand_is_ref_top2": len(r["top"]) > 1 and cand_top == r["top"][1][0],
            }
        )
    return {
        "id": ref["id"],
        "positions": len(ref["positions"]),
        "agree": agree,
        "flips": flips,
        "max_abs_delta": max((abs(d) for d in deltas), default=None),
        "mean_abs_delta": sum(abs(d) for d in deltas) / len(deltas) if deltas else None,
        "delta_missing": len(ref["positions"]) - len(deltas),
    }


def compare_runs(
    ref: dict,
    cand: dict,
    min_agreement: float = MIN_AGREEMENT,
    gap_max: float = GAP_MAX,
) -> dict:
    cand_cases = {c["id"]: c for c in cand["cases"]}
    ref_ids = [c["id"] for c in ref["cases"]]
    if len(set(ref_ids)) != len(ref_ids) or len(cand_cases) != len(cand["cases"]):
        raise ValueError("duplicate case ids")
    if set(ref_ids) != set(cand_cases):
        raise ValueError(
            f"case ids differ: {sorted(set(ref_ids) ^ set(cand_cases), key=str)}"
        )
    cases = [compare_case(c, cand_cases[c["id"]], gap_max) for c in ref["cases"]]
    total = sum(c["positions"] for c in cases)
    agree = sum(c["agree"] for c in cases)
    flips = [f for c in cases for f in c["flips"]]
    real = sum(not f["near_tie"] for f in flips)
    agreement = agree / total if total else 0.0
    return {
        "positions": total,
        "agreement": agreement,
        "min_agreement": min_agreement,
        "gap_max": gap_max,
        "flips": len(flips),
        "near_tie_flips": len(flips) - real,
        "real_flips": real,
        "pass": total > 0 and agreement >= min_agreement and real == 0,
        "cases": cases,
    }


def relative_l2(ref, cand) -> float:
    import numpy as np

    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if ref.shape != cand.shape:
        return float("inf")
    norm = float(np.linalg.norm(ref))
    diff = float(np.linalg.norm(cand - ref))
    if norm == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / norm


def compare_states(ref: dict, cand: dict, tol: float = TOL_SHORT) -> dict:
    """Relative L2 per visit, in the reference's visit order."""
    visits, first_failing = [], None
    for name, value in ref.items():
        rel = relative_l2(value, cand[name]) if name in cand else float("inf")
        ok = rel <= tol
        if not ok and first_failing is None:
            first_failing = name
        visits.append({"visit": name, "relative_l2": rel, "ok": ok})
    for name in cand:
        if name not in ref:
            first_failing = first_failing or name
            visits.append({"visit": name, "relative_l2": float("inf"), "ok": False})
    return {
        "tol": tol,
        "visits": visits,
        "first_failing": first_failing,
        "pass": bool(visits) and first_failing is None,
    }


def _load_states(path: str) -> dict:
    import numpy as np

    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--base-url", required=True)
    cap.add_argument("--model", required=True)
    cap.add_argument("--cases", required=True, help="a run file, e.g. the HF one")
    cap.add_argument("--out", required=True)
    cap.add_argument("--top", type=int, default=2)
    cap.add_argument("--timeout", type=float, default=600)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("ref")
    cmp_.add_argument("cand")
    cmp_.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT)
    cmp_.add_argument("--gap-max", type=float, default=GAP_MAX)
    cmp_.add_argument("--out")
    st = sub.add_parser("states")
    st.add_argument("ref")
    st.add_argument("cand")
    tol = st.add_mutually_exclusive_group()
    tol.add_argument("--tol", type=float, default=TOL_SHORT)
    tol.add_argument(
        "--long", action="store_true", help=f"long-context tolerance {TOL_LONG}"
    )
    st.add_argument("--out")
    args = ap.parse_args(argv)

    if args.mode == "capture":
        with open(args.cases) as f:
            cases = json.load(f)["cases"]
        run = capture(args.base_url, args.model, cases, args.top, args.timeout)
        with open(args.out, "w") as f:
            json.dump(run, f)
        return 0

    if args.mode == "compare":
        with open(args.ref) as f:
            ref = json.load(f)
        with open(args.cand) as f:
            cand = json.load(f)
        report = compare_runs(ref, cand, args.min_agreement, args.gap_max)
        for case in report["cases"]:
            print(
                f"{case['id']}: {case['agree']}/{case['positions']} top-1, "
                f"max |delta| {case['max_abs_delta']}, "
                f"{case['delta_missing']} without a delta"
            )
            for flip in case["flips"]:
                kind = "near tie" if flip["near_tie"] else "REAL"
                print(
                    f"  flip at {flip['pos']}: {flip['ref_top1']} -> "
                    f"{flip['cand_top1']}, gap {flip['gap']} ({kind})"
                )
        print(
            f"agreement {report['agreement']:.4f} over {report['positions']} "
            f"positions, {report['near_tie_flips']} near-tie and "
            f"{report['real_flips']} real flips: "
            f"{'pass' if report['pass'] else 'FAIL'}"
        )
    else:
        tol = TOL_LONG if args.long else args.tol
        report = compare_states(_load_states(args.ref), _load_states(args.cand), tol)
        for visit in report["visits"]:
            if not visit["ok"]:
                print(f"{visit['visit']}: relative L2 {visit['relative_l2']:.3e}")
        if report["pass"]:
            print(f"{len(report['visits'])} visits within {tol}: pass")
        else:
            print(f"first failing visit: {report['first_failing']} (tol {tol}): FAIL")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
