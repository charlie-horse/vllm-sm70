"""Check that greedy divergence points are near-ties.

Re-runs each A2 prompt once at concurrency 1 with ``logprobs`` and
``top_logprobs=2`` (decode-side only; no ``prompt_logprobs``), up to just past
the furthest divergence seen in earlier gate_a runs. For every divergence index
``d`` where this run still matches the reference prefix, reports the top-1 vs
top-2 logprob gap at ``d``. Small gaps (< --gap-max) mean a benign near-tie
flip; a large gap at a divergence point would point at corruption.

Usage:
  near_tie.py --base-url URL --model flint --ref RUN/a2/a2_outputs.jsonl \
      --variants RUN/a2/a2_outputs.jsonl RUN/a2_repeat/a2_outputs.jsonl \
      --out RUN/near_tie
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def post(base: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prefix_len(a: list[int], b: list[int]) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="flint")
    ap.add_argument("--prompts", default=os.path.join(HERE, "fixtures",
                                                       "greedy_prompts.jsonl"))
    ap.add_argument("--ref", required=True, help="a2_outputs.jsonl (c1 = ref)")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-max", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    prompts = {p["id"]: p for p in load_jsonl(args.prompts) if "id" in p}
    ref = {o["id"]: o for o in load_jsonl(args.ref) if o["concurrency"] == 1}
    divs: dict[str, list[tuple[str, int]]] = {pid: [] for pid in ref}
    for vi, path in enumerate(args.variants):
        for o in load_jsonl(path):
            if path == args.ref and o["concurrency"] == 1:
                continue
            tag = f"v{vi}.c{o['concurrency']}"
            d = prefix_len(ref[o["id"]]["token_ids"], o["token_ids"])
            if d < len(ref[o["id"]]["token_ids"]):
                divs[o["id"]].append((tag, d))

    rows, all_gaps, recs = [], [], []
    for pid in sorted(ref):
        p = prompts[pid]
        if not divs[pid]:
            continue
        n = min(p["max_tokens"], max(d for _, d in divs[pid]) + 4)
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": p["prompt"]}],
            "max_tokens": n,
            "temperature": 0.0,
            "seed": args.seed,
            "return_token_ids": True,
            "logprobs": True,
            "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        r = post(args.base_url, "/chat/completions", body, args.timeout)
        ch = r["choices"][0]
        toks = ch.get("token_ids") or []
        lp = (ch.get("logprobs") or {}).get("content") or []
        gaps = []
        for e in lp:
            top = e.get("top_logprobs") or []
            gaps.append(top[0]["logprob"] - top[1]["logprob"]
                        if len(top) >= 2 else None)
        all_gaps += [g for g in gaps if g is not None]
        m = prefix_len(ref[pid]["token_ids"], toks)
        for tag, d in divs[pid]:
            ok = m >= d and d < len(gaps) and gaps[d] is not None
            row = {"id": pid, "variant": tag, "div": d, "this_run_prefix": m,
                   "gap": round(gaps[d], 4) if ok else None,
                   "top2": ([(t["token"], round(t["logprob"], 4))
                             for t in lp[d]["top_logprobs"]] if ok else None)}
            rows.append(row)
        recs.append({"id": pid, "n_logprobs": len(lp), "n_tokens": len(toks),
                     "secs": round(time.time() - t0, 2)})
        print(f"{pid}: {len(lp)} lp, prefix vs ref {m}, "
              f"{time.time() - t0:.1f}s", flush=True)

    checked = [r for r in rows if r["gap"] is not None]
    near = [r for r in checked if r["gap"] < args.gap_max]
    frac_small = (sum(g < args.gap_max for g in all_gaps) / len(all_gaps)
                  if all_gaps else float("nan"))
    summary = {
        "divergences": len(rows), "checked": len(checked),
        "near_tie": len(near), "gap_max": args.gap_max,
        "gap_at_div_median": (statistics.median(r["gap"] for r in checked)
                              if checked else None),
        "gap_at_div_max": max((r["gap"] for r in checked), default=None),
        "positions_scored": len(all_gaps),
        "frac_all_positions_below_gap_max": round(frac_small, 4),
        "pass": bool(checked) and len(near) == len(checked),
    }
    with open(os.path.join(args.out, "near_tie.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows, "requests": recs}, f,
                  indent=1)
    print(json.dumps(summary, indent=1))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
