#!/usr/bin/env python3
"""Analyze a SWE-bench Verified run of `flint` (mini-swe-agent 2.4.6, swebench 5.0.2).

README
======
Usage::

    /home/user/swe/.venv/bin/python /home/user/swe/scripts/analyze.py \
        [--runs /home/user/swe/runs] [--out /home/user/swe/runs/analysis] \
        [--attempts 1,2] [--boot 10000] [--timeout-is-model] [--include-pending]

Writes ``<out>.md`` and ``<out>.json`` and prints the markdown to stdout (``--json``
prints the JSON instead). Stdlib + numpy only; read-only on the run directories.

Inputs (per attempt A, instance ID), produced by scripts/run_one.sh:
  runs/attA/agent/ID/preds.json              {ID: {model_patch, model_name_or_path}}
  runs/attA/agent/ID/ID/ID.traj.json         mini-swe-agent trajectory
  runs/attA/agent/ID/exit_statuses_*.yaml    {instances_by_exit_status: {...}}
  runs/attA/eval/done_ID                     {"id","attempt","agent_s","eval_s"}
  runs/attA/eval/<model>.attA_ID.json        swebench run report
  runs/attA/eval/logs/run_evaluation/attA_ID/<model>/ID/{report.json,run_instance.log}
  runs/attA/log/ID.{agent,eval}.log          stdout/stderr of agent and harness

Trajectory keys used: info.exit_status, info.submission, info.exception_str,
info.traceback, info.model_stats.api_calls, messages[*].role
(system/user/assistant/tool/exit), messages[*].extra.response.usage
{prompt_tokens, completion_tokens}, messages[*].extra.timestamp. Format-error
turns are role=user messages with extra.interrupt_type=FormatError that still
carry extra.response (billed), so tokens are summed over every message with a
usage block; "turns" counts assistant messages, "api_calls" counts usage blocks.

An instance is counted only once done_ID exists (otherwise "pending", skipped).

Categories (exactly one per instance, first match wins):
  resolved               swebench report says resolved.
  eval_infra_error       patch present but eval did not produce a usable verdict:
                         report missing/empty, swebench infra_failure (environment
                         tier: docker daemon, OOM, network), harness log shows image
                         pull/build failure, docker error, 429/rate limit, or the
                         test run timed out (use --timeout-is-model to count test
                         timeouts as wrong_patch instead).
  runner_infra_error     done_ID carries an "infra" field (r440 run_one.sh writes
                         "pull_failed" | "api_failed"); takes precedence over all.
  api_infra_error        endpoint failures that are not the model's fault: exit
                         status APIConnectionError/APIError/Timeout/5xx/
                         RateLimitError, or any 429 / rate-limit text.
  agent_infra_error      agent never ran because the sandbox failed (docker run /
                         image pull error, no trajectory). Excluded like eval infra.
  context_overflow       API rejected the prompt for context length (400 "maximum
                         context length", ContextWindowExceededError, max_tokens too
                         large for remaining context, ...).
  wrong_patch            non-empty patch, eval completed, not resolved (detail says
                         if the patch failed to apply).
  no_patch_or_step_limit empty submission with LimitsExceeded / TimeExceeded /
                         Submitted-but-empty / agent timeout (6h `timeout`).
  format_tool_error      RepeatedFormatError, FormatError, or a non-context model API
                         exception (e.g. non-context BadRequestError).
  other:<exit_status>    anything else (inspect manually).
Infra categories are excluded from the pass@1 denominator and reported separately.

Statistics: pass@1 per attempt = resolved / (done - infra). Across attempts: mean
of per-attempt pass@1 with (a) a bootstrap 95% CI resampling instances (each
replicate = mean over attempts of per-attempt rates over the resampled
instances), (b) a normal approximation over per-instance mean scores
(SE = sd(p_i)/sqrt(n)). With a single attempt the Wilson score interval is used.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INFRA = ("runner_infra_error", "api_infra_error", "eval_infra_error",
         "agent_infra_error")
CATEGORIES = (
    "resolved",
    "wrong_patch",
    "no_patch_or_step_limit",
    "context_overflow",
    "format_tool_error",
    "eval_infra_error",
    "agent_infra_error",
    "api_infra_error",
    "runner_infra_error",
)

CONTEXT_RE = re.compile(
    r"maximum context length|context length|ContextWindowExceeded|context window"
    r"|longer than the maximum model length|prompt is too long"
    r"|max_tokens' or 'max_completion_tokens' is too large"
    r"|input tokens exceed|too many tokens",
    re.I,
)
# Transient endpoint failures (run_one.sh API_RE + rate limits): never the model's
# fault -> api_infra_error (excluded). Matched on the exit status / exception text.
API_INFRA_STATUSES = {
    "APIConnectionError", "APIError", "Timeout", "InternalServerError",
    "ServiceUnavailableError", "BadGatewayError", "APIResponseValidationError",
    "RateLimitError",
}
RATE_LIMIT_RE = re.compile(r"RateLimitError|\b429\b|Too Many Requests|rate limit", re.I)
API_ERR_RE = re.compile(
    r"BadRequestError|APIConnectionError|APIError|InternalServerError|Timeout"
    r"|ServiceUnavailable|litellm\.|openai\.",
)
FORMAT_STATUSES = {"RepeatedFormatError", "FormatError"}
LIMIT_STATUSES = {"LimitsExceeded", "TimeExceeded", "Submitted", "StepLimitExceeded"}
AGENT_INFRA_RE = re.compile(
    r"docker run|Error response from daemon|Cannot connect to the Docker daemon"
    r"|pull access denied|manifest unknown|toomanyrequests|No space left on device"
    r"|Failed to start container|CalledProcessError",
)
EVAL_INFRA_RE = re.compile(
    r"Error building image|BuildImageError|pull access denied|manifest unknown"
    r"|toomanyrequests|429|Too Many Requests|Error response from daemon"
    r"|Cannot connect to the Docker daemon|No space left on device"
    r"|docker\.errors\.\w+|EvaluationError|ConnectionError|ReadTimeout",
)
TIMEOUT_RE = re.compile(r"Timeout error: \d+ seconds exceeded|Test timed out", re.I)


# ----------------------------------------------------------------------------- io
def read_json(path: Path):
    try:
        text = path.read_text(errors="replace").strip()
        return json.loads(text) if text else None
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path | None, limit: int = 2_000_000) -> str:
    if path is None:
        return ""
    try:
        with open(path, errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def first(paths) -> Path | None:
    paths = sorted(paths)
    return Path(paths[0]) if paths else None


def repo_of(iid: str) -> str:
    owner, rest = iid.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def exit_status_from_yaml(agent_dir: Path, iid: str) -> str | None:
    """Parse exit_statuses_*.yaml without needing PyYAML (simple 2-level map)."""
    for y in sorted(agent_dir.glob("exit_statuses_*.yaml"), reverse=True):
        status = None
        for line in read_text(y).splitlines():
            m = re.match(r"^\s{4}(['\"]?)(.+?)\1:\s*$", line)
            if m:
                status = m.group(2)
            elif status and re.match(
                    rf"^\s*-\s*['\"]?{re.escape(iid)}['\"]?\s*$", line):
                return status
    return None


# ------------------------------------------------------------------ trajectory
def traj_stats(traj: dict | None) -> dict:
    out = dict(turns=0, api_calls=0, prompt_tokens=0, completion_tokens=0,
               max_prompt_tokens=0, traj_wall_s=None)
    if not traj:  # no trajectory: token stats unknown, keep them out of summaries
        return {k: (None if k != "turns" else 0) for k in out}
    stamps = []
    for msg in traj.get("messages", []):
        if msg.get("role") == "assistant":
            out["turns"] += 1
        extra = msg.get("extra") or {}
        if isinstance(extra.get("timestamp"), (int, float)):
            stamps.append(extra["timestamp"])
        resp = extra.get("response")
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if not isinstance(usage, dict):
            continue
        out["api_calls"] += 1
        pt = usage.get("prompt_tokens") or 0
        out["prompt_tokens"] += pt
        out["completion_tokens"] += usage.get("completion_tokens") or 0
        out["max_prompt_tokens"] = max(out["max_prompt_tokens"], pt)
    if len(stamps) >= 2:
        out["traj_wall_s"] = stamps[-1] - stamps[0]
    return out


# ------------------------------------------------------------------- eval side
def eval_info(att_dir: Path, a: int, iid: str) -> dict:
    run_id = f"att{a}_{iid}"
    ev = att_dir / "eval"
    run_report_p = first(glob.glob(str(ev / f"*.{run_id}.json")))
    run_report = read_json(run_report_p) if run_report_p else None
    inst_dir = first(glob.glob(str(ev / "logs/run_evaluation" / run_id / "*" / iid)))
    inst_report = read_json(inst_dir / "report.json") if inst_dir else None
    if isinstance(inst_report, dict):
        inst_report = inst_report.get(iid)
    logs = "\n".join([
        read_text(inst_dir / "run_instance.log") if inst_dir else "",
        read_text(att_dir / "log" / f"{iid}.eval.log"),
    ])
    test_out = read_text(inst_dir / "test_output.txt") if inst_dir else ""
    return dict(run_report=run_report, inst_report=inst_report, logs=logs,
                test_out=test_out, ran=(att_dir / "log" / f"{iid}.eval.log").exists())


# -------------------------------------------------------------- classification
def classify(rec: dict, ev: dict, timeout_is_model: bool) -> tuple[str, str]:
    status = rec["exit_status"] or ""
    if rec.get("infra"):  # runner-declared infra (done_ID "infra"): always wins
        return "runner_infra_error", str(rec["infra"])
    patch = rec["patch"]
    rr, ir = ev["run_report"], ev["inst_report"]
    resolved = bool(ir and ir.get("resolved")) or bool(
        rr and rec["id"] in (rr.get("resolved_ids") or []))
    if resolved:
        return "resolved", ""
    err_text = " ".join([status, rec.get("exception_str") or "",
                         rec.get("traceback") or "", rec.get("last_msgs") or ""])
    if patch.strip():
        reasons = (rr or {}).get("failure_reasons") or {}
        if ir is None:
            m = EVAL_INFRA_RE.search(ev["logs"])
            why = m.group(0) if m else ("eval_not_run" if not ev["ran"]
                                         else "report_missing")
            return "eval_infra_error", why
        if ir.get("infra_failure") or (rr and rec["id"] in
                                       (rr.get("infra_failure_ids") or [])):
            return "eval_infra_error", ir.get("infra_failure_reason") or reasons.get(
                rec["id"], "infra_failure")
        timed_out = TIMEOUT_RE.search(ev["logs"] + ev["test_out"]) or reasons.get(
            rec["id"]) == "tests_timed_out"
        if timed_out and not timeout_is_model:
            return "eval_infra_error", "tests_timed_out"
        if not ir.get("patch_successfully_applied"):
            if EVAL_INFRA_RE.search(ev["logs"]):
                return "eval_infra_error", EVAL_INFRA_RE.search(ev["logs"]).group(0)
            return "wrong_patch", "patch_apply_failed_or_no_test_output"
        return "wrong_patch", reasons.get(rec["id"], "")
    # ---- empty patch: why did the agent stop? Clean agent exits first, then
    # exception text (traj info / exit message / agent stdout log, never tool output).
    if status in LIMIT_STATUSES:
        return "no_patch_or_step_limit", status
    if status in FORMAT_STATUSES:
        return "format_tool_error", status
    exc_text = err_text + " " + rec["agent_log"][-20_000:]
    if CONTEXT_RE.search(exc_text):
        return "context_overflow", status or CONTEXT_RE.search(exc_text).group(0)
    if not rec["has_traj"] and AGENT_INFRA_RE.search(exc_text):
        return "agent_infra_error", status or AGENT_INFRA_RE.search(exc_text).group(0)
    if not rec["has_traj"] and (rec.get("agent_s") or 0) >= 6 * 3600 - 60:
        return "no_patch_or_step_limit", "agent_timeout_6h"
    if status in API_INFRA_STATUSES:
        return "api_infra_error", status
    if RATE_LIMIT_RE.search(err_text):
        return "api_infra_error", status or "rate_limited_429"
    if API_ERR_RE.search(exc_text):
        return "format_tool_error", status or API_ERR_RE.search(exc_text).group(0)
    return f"other:{status or 'no_exit_status'}", ""


# ------------------------------------------------------------------- collecting
def collect(runs: Path, attempts: list[int] | None, include_pending: bool,
            timeout_is_model: bool) -> list[dict]:
    recs = []
    for att_dir in sorted(runs.glob("att*")):
        m = re.fullmatch(r"att(\d+)", att_dir.name)
        if not m or (attempts and int(m.group(1)) not in attempts):
            continue
        a = int(m.group(1))
        ids = {p.name for p in (att_dir / "agent").glob("*") if p.is_dir()}
        ids |= {p.name[5:] for p in (att_dir / "eval").glob("done_*")}
        for iid in sorted(ids):
            done = read_json(att_dir / "eval" / f"done_{iid}")
            if done is None and not include_pending:
                continue
            agent_dir = att_dir / "agent" / iid
            traj = read_json(agent_dir / iid / f"{iid}.traj.json")
            preds = read_json(agent_dir / "preds.json") or {}
            info = (traj or {}).get("info", {})
            patch = (preds.get(iid) or {}).get("model_patch")
            if patch is None:
                patch = info.get("submission")
            msgs = (traj or {}).get("messages", [])
            last = " ".join(
                str(x.get("content", ""))[:4000] + json.dumps(x.get("extra", {}))[:4000]
                for x in msgs[-2:] if x.get("role") == "exit")
            rec = dict(
                id=iid, attempt=a, repo=repo_of(iid), done=done is not None,
                infra=(done or {}).get("infra"),
                exit_status=info.get("exit_status")
                or exit_status_from_yaml(agent_dir, iid),
                exception_str=info.get("exception_str"),
                traceback=info.get("traceback"), last_msgs=last,
                patch=patch or "", has_traj=traj is not None,
                agent_log=read_text(att_dir / "log" / f"{iid}.agent.log")[-200_000:],
                agent_s=(done or {}).get("agent_s"),
                eval_s=(done or {}).get("eval_s"),
            )
            rec.update(traj_stats(traj))
            if rec["agent_s"] is None:
                rec["agent_s"] = rec["traj_wall_s"]
            if not rec["done"]:
                rec["category"], rec["detail"] = "pending", ""
            else:
                ev = eval_info(att_dir, a, iid)
                rec["category"], rec["detail"] = classify(rec, ev, timeout_is_model)
            calls = rec["api_calls"]
            rec["mean_completion_per_turn"] = (
                rec["completion_tokens"] / calls if calls else None)
            if not rec["has_traj"]:
                rec["turns"] = None
            for k in ("agent_log", "last_msgs", "traceback"):
                rec.pop(k, None)
            rec["patch_chars"] = len(rec.pop("patch"))
            if rec["exception_str"]:
                rec["exception_str"] = rec["exception_str"][:300]
            recs.append(rec)
    return recs


# -------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pass_stats(recs: list[dict], n_boot: int, seed: int = 0) -> dict:
    scored = [r for r in recs if r["done"] and r["category"] not in INFRA]
    attempts = sorted({r["attempt"] for r in recs})
    ids = sorted({r["id"] for r in scored})
    per_att = {}
    for a in attempts:
        rs = [r for r in recs if r["attempt"] == a and r["done"]]
        k = sum(r["category"] == "resolved" for r in rs)
        infra = sum(r["category"] in INFRA for r in rs)
        n = len(rs) - infra
        per_att[a] = dict(resolved=k, attempted=n, infra_excluded=infra,
                          done=len(rs), pass_at_1=k / n if n else None,
                          wilson95=wilson(k, n) if n else None)
    out = dict(per_attempt=per_att, n_instances=len(ids), n_attempts=len(attempts))
    if not ids:
        return out
    idx = {iid: i for i, iid in enumerate(ids)}
    M = np.full((len(ids), len(attempts)), np.nan)
    for r in scored:
        M[idx[r["id"]], attempts.index(r["attempt"])] = r["category"] == "resolved"
    att_rates = [v["pass_at_1"] for v in per_att.values() if v["pass_at_1"] is not None]
    mean = float(np.mean(att_rates))
    out["mean_pass_at_1"] = mean
    if len(att_rates) > 1:
        out["attempt_sd"] = float(np.std(att_rates, ddof=1))
    k_tot = int(np.nansum(M))
    n_tot = int(np.sum(~np.isnan(M)))
    if len(attempts) == 1:
        out["ci95_method"] = "wilson"
        out["ci95"] = wilson(k_tot, n_tot)
        return out
    rng = np.random.default_rng(seed)
    obs = ~np.isnan(M)
    V = np.nan_to_num(M)
    boots = []
    for _ in range(0, n_boot, 1000):  # chunks of 1000 replicates
        idx_b = rng.integers(0, len(ids), (min(1000, n_boot), len(ids)))
        k = V[idx_b].sum(axis=1)  # (B, attempts)
        n = obs[idx_b].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(n > 0, k / n, np.nan)
        cnt = (~np.isnan(rates)).sum(axis=1)
        boots.append(np.nansum(rates, axis=1) / np.where(cnt, cnt, np.nan))
    boots = np.concatenate(boots)[:n_boot]
    boots = boots[~np.isnan(boots)]
    out["ci95_method"] = "bootstrap_over_instances"
    out["ci95"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    p_i = np.nanmean(M, axis=1)
    se = float(np.std(p_i, ddof=1) / math.sqrt(len(p_i))) if len(p_i) > 1 else 0.0
    out["normal95"] = (max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se))
    out["normal_se"] = se
    return out


def summarize(vals: list) -> dict:
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return dict(n=0, mean=None, median=None, p90=None)
    return dict(n=len(v), mean=float(np.mean(v)), median=float(statistics.median(v)),
                p90=float(np.percentile(v, 90)))


COST_KEYS = ("turns", "api_calls", "prompt_tokens", "completion_tokens",
             "mean_completion_per_turn", "max_prompt_tokens", "agent_s", "eval_s")


def build(recs: list[dict], n_boot: int) -> dict:
    done = [r for r in recs if r["done"]]
    repos = defaultdict(lambda: Counter())
    for r in done:
        c = repos[r["repo"]]
        if r["category"] in INFRA:
            c["infra"] += 1
        else:
            c["attempted"] += 1
            c["resolved"] += r["category"] == "resolved"
    return dict(
        pass_at_1=pass_stats(recs, n_boot),
        categories=dict(Counter(r["category"] for r in recs)),
        per_repo={k: dict(v, rate=(v["resolved"] / v["attempted"]
                                   if v["attempted"] else None))
                  for k, v in sorted(repos.items())},
        cost={k: summarize([r.get(k) for r in done]) for k in COST_KEYS},
        instances=recs,
    )


# ---------------------------------------------------------------------- markdown
def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def to_markdown(S: dict) -> str:
    P = S["pass_at_1"]
    L = ["# flint SWE-bench Verified analysis", "", "## pass@1", "",
         "| attempt | resolved | attempted | infra excl. | pass@1 | Wilson 95% |",
         "|---|---|---|---|---|---|"]
    for a, v in P["per_attempt"].items():
        w = v["wilson95"]
        L.append(f"| {a} | {v['resolved']} | {v['attempted']} | {v['infra_excluded']} "
                 f"| {pct(v['pass_at_1'])} | "
                 f"{'-' if not w else f'{pct(w[0])} – {pct(w[1])}'} |")
    if "mean_pass_at_1" in P:
        ci = P["ci95"]
        L += ["", f"**Mean pass@1: {pct(P['mean_pass_at_1'])}** "
              f"(95% CI {pct(ci[0])} – {pct(ci[1])}, {P['ci95_method']}; "
              f"{P['n_instances']} instances × {P['n_attempts']} attempt(s))"]
        if "normal95" in P:
            n = P["normal95"]
            L.append(f"Normal approx: {pct(n[0])} – {pct(n[1])} "
                     f"(SE {100 * P['normal_se']:.2f} pp); "
                     f"attempt-to-attempt SD {100 * P.get('attempt_sd', 0):.2f} pp")
    L += ["", "## Per repo", "", "| repo | resolved | attempted | rate | infra |",
          "|---|---|---|---|---|"]
    for repo, v in S["per_repo"].items():
        L.append(f"| {repo} | {v.get('resolved', 0)} | {v.get('attempted', 0)} "
                 f"| {pct(v['rate'])} | {v.get('infra', 0)} |")
    infra = [r for r in S["instances"] if r["category"] in INFRA]
    L += ["", "## Infra-excluded (not in denominator)", ""]
    L += [f"- att{r['attempt']} {r['id']}: {r['category']} ({r['detail'] or '?'})"
          for r in infra] or ["- none"]
    L += ["", "## Categories", "", "| category | count |", "|---|---|"]
    for c, n in sorted(S["categories"].items(), key=lambda x: -x[1]):
        L.append(f"| {c} | {n} |")
    L += ["", "## Cost (done instances)", "", "| metric | n | mean | median | p90 |",
          "|---|---|---|---|---|"]
    for k, v in S["cost"].items():
        L.append(f"| {k} | {v['n']} | {fmt(v['mean'])} | {fmt(v['median'])} "
                 f"| {fmt(v['p90'])} |")
    L += ["", "## Instances", "",
          "| att | instance | category | detail | exit | turns | prompt tok | "
          "compl tok | compl/turn | peak ctx | agent s | eval s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in S["instances"]:
        L.append(
            f"| {r['attempt']} | {r['id']} | {r['category']} | {r['detail'] or ''} "
            f"| {r['exit_status'] or ''} | {fmt(r['turns'])} "
            f"| {fmt(r['prompt_tokens'])} "
            f"| {fmt(r['completion_tokens'])} "
            f"| {fmt(r['mean_completion_per_turn'], 0)} "
            f"| {fmt(r['max_prompt_tokens'])} | {fmt(r['agent_s'])} "
            f"| {fmt(r['eval_s'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="/home/user/swe/runs")
    ap.add_argument("--out", default=None,
                    help="output prefix (default <runs>/analysis)")
    ap.add_argument("--attempts", default=None, help="comma list, e.g. 1,2")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--timeout-is-model", action="store_true",
                    help="count eval test timeouts as wrong_patch, not infra")
    ap.add_argument("--include-pending", action="store_true",
                    help="also list instances without done_ID (not scored)")
    ap.add_argument("--json", action="store_true", help="print JSON to stdout")
    args = ap.parse_args()
    runs = Path(args.runs)
    attempts = [int(x) for x in args.attempts.split(",")] if args.attempts else None
    recs = collect(runs, attempts, args.include_pending, args.timeout_is_model)
    S = build(recs, args.boot)
    md = to_markdown(S)
    out = Path(args.out) if args.out else runs / "analysis"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.md").write_text(md)
    Path(f"{out}.json").write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps(S, indent=1, default=str) if args.json else md)
    if not args.json:
        print(f"_written: {out}.md, {out}.json_", file=sys.stderr)


if __name__ == "__main__":
    main()
