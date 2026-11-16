#!/usr/bin/env python3
"""Gate A: output-coherence checks for a vLLM OpenAI-compatible server.

A1  teacher-forced NLL of known-good assistant turns (SWE-bench trajectories
    from the seqs=2 server), rendered with the server's own chat template via
    ``/tokenize`` and scored with ``/v1/completions`` ``prompt_logprobs=1``,
    ``max_tokens=1``. Run at every concurrency level in fixed waves (the same
    requests are in flight together at each level).
A2  greedy generation (temperature 0, fixed seed) of 24 prompts through
    ``/v1/chat/completions`` with a pool of N concurrent requests. Compared
    with the concurrency-1 output (exact token prefix) and screened for
    gibberish (zlib ratio, n-gram repetition, non-ASCII ratio, python
    ``ast`` parse of fenced code).

Stdlib only. Exit status: 0 overall PASS, 1 FAIL, 2 could not run.

    .venv/bin/python gate_a.py --base-url http://HOST:8000/v1 --model flint \\
        --concurrency 1,4,16 --out runs/<name>
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EARLY = 64  # the regression shows once a sequence passes ~64 tokens


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Client:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base = base_url.rstrip("/")
        self.root = re.sub(r"/v1$", "", self.base)
        self.model = model
        self.timeout = timeout

    def _req(self, url: str, body: dict | None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from None

    def get(self, path: str, root: bool = False) -> dict:
        return self._req((self.root if root else self.base) + path, None)

    def post(self, path: str, body: dict, root: bool = False) -> dict:
        return self._req((self.root if root else self.base) + path, body)

    def tokenize_chat(self, messages, tools, add_generation_prompt, kw=None):
        body = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "add_special_tokens": False,
        }
        if tools:
            body["tools"] = tools
        if kw:
            body["chat_template_kwargs"] = kw
        return self.post("/tokenize", body, root=True)["tokens"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> tuple[dict, list[dict]]:
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
            else:
                rows.append(r)
    return meta, rows


def wire_message(m: dict) -> dict:
    """Trajectory message -> request message (reasoning under both keys)."""
    out = {k: v for k, v in m.items() if k != "reasoning"}
    if m.get("reasoning"):
        out["reasoning"] = m["reasoning"]
        out["reasoning_content"] = m["reasoning"]
    return out


def lcp(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------------------------------------------------------------------
# A1: teacher-forced NLL
# ---------------------------------------------------------------------------


def a1_prepare(cl: Client, fixtures: list[dict], tools, workers: int) -> list:
    def one(fx):
        ctx = [wire_message(m) for m in fx["context"]]
        full = ctx + [wire_message(fx["target"])]
        try:
            p = cl.tokenize_chat(ctx, tools, True)
            f = cl.tokenize_chat(full, tools, False)
        except Exception as e:  # noqa: BLE001
            return {"id": fx["id"], "error": f"tokenize: {e}"}
        start = lcp(p, f)
        return {
            "id": fx["id"],
            "repo": fx.get("repo"),
            "ids": f,
            "start": start,
            "n_ctx": len(p),
            "n_asst": len(f) - start,
            "gen_prompt_mismatch": len(p) - start,
            "ref_prompt_tokens": fx.get("ref_prompt_tokens"),
            "ref_completion_tokens": fx.get("ref_completion_tokens"),
        }

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, fixtures))


def _lp(entry: dict | None, tok: int) -> float | None:
    if not entry:
        return None
    v = entry.get(str(tok), entry.get(tok))
    if v is None:
        return None
    return v["logprob"] if isinstance(v, dict) else float(v)


def a1_score(cl: Client, prep: dict) -> dict:
    t0 = time.time()
    body = {
        "model": cl.model,
        "prompt": prep["ids"],
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 1,
    }
    try:
        resp = cl.post("/completions", body)
        plp = resp["choices"][0]["prompt_logprobs"]
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "secs": time.time() - t0}
    ids, start = prep["ids"], prep["start"]
    nll, missing = [], 0
    for j in range(max(start, 1), len(ids)):
        v = _lp(plp[j] if j < len(plp) else None, ids[j])
        if v is None or math.isnan(v):
            missing += 1
            continue
        nll.append(-max(v, -1e4))
    return {"nll": nll, "missing": missing, "secs": time.time() - t0}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def a1_stats(nll: list[float]) -> dict:
    return {
        "n": len(nll),
        "mean": _mean(nll),
        "mean_0_63": _mean(nll[:EARLY]),
        "mean_64p": _mean(nll[EARLY:]),
        "n_64p": max(0, len(nll) - EARLY),
    }


def run_a1(cl, fixtures, tools, levels, args, log) -> dict:
    log(f"A1: tokenizing {len(fixtures)} fixtures")
    preps = a1_prepare(cl, fixtures, tools, min(8, max(levels)))
    ok = [p for p in preps if "error" not in p]
    per_level: dict[int, dict[str, dict]] = {}
    for c in levels:
        log(f"A1: c={c} ({len(ok)} requests, waves of {c})")
        res: dict[str, dict] = {}
        t0 = time.time()
        for w in range(0, len(ok), c):
            wave = ok[w : w + c]
            barrier = threading.Barrier(len(wave))

            def go(p, barrier=barrier):
                barrier.wait()
                return a1_score(cl, p)

            with ThreadPoolExecutor(len(wave)) as ex:
                outs = list(ex.map(go, wave))
            for p, o in zip(wave, outs):
                if "nll" in o:
                    s = a1_stats(o["nll"])
                    s.update(missing=o["missing"], secs=round(o["secs"], 2))
                    res[p["id"]] = s
                else:
                    res[p["id"]] = {"error": o["error"]}
            errs = [o for o in outs if "error" in o]
            if errs and len(errs) == len(outs) and _is_down(errs[0]["error"]):
                log(f"A1: server unreachable, aborting: {errs[0]['error']}")
                break
        per_level[c] = res
        log(f"A1: c={c} done in {time.time() - t0:.1f}s")
    return {"prepared": [_strip(p) for p in preps], "per_level": per_level}


def _is_down(err: str) -> bool:
    return any(
        s in err
        for s in (
            "Connection refused",
            "Connection reset",
            "Remote end closed",
            "timed out",
            "URLError",
        )
    )


def _strip(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "ids"}


def a1_checks(a1: dict, levels, args) -> list[dict]:
    checks = []
    base = levels[0]
    pl = a1["per_level"]
    prep = {p["id"]: p for p in a1["prepared"]}
    errs = [p["id"] for p in a1["prepared"] if "error" in p]
    checks.append(_chk("A1.tokenize", not errs, f"{len(errs)} errors", errs))
    for c in levels:
        rows = pl.get(c, {})
        bad = [k for k, v in rows.items() if "error" in v or v.get("missing")]
        bad += [k for k in prep if k not in rows and "error" not in prep[k]]
        checks.append(
            _chk(f"A1.c{c}.requests", not bad, f"{len(bad)} failed/missing", bad)
        )
        tok = [v for v in rows.values() if "mean" in v]
        tot = sum(v["n"] for v in tok)
        mean = sum(v["mean"] * v["n"] for v in tok) / tot if tot else math.inf
        e_n = sum(min(v["n"], EARLY) for v in tok)
        l_n = sum(v["n_64p"] for v in tok)
        early = (
            sum(v["mean_0_63"] * min(v["n"], EARLY) for v in tok) / e_n
            if e_n
            else math.nan
        )
        late = (
            sum(v["mean_64p"] * v["n_64p"] for v in tok if v["n_64p"]) / l_n
            if l_n
            else math.nan
        )
        checks.append(
            _chk(
                f"A1.c{c}.mean_nll",
                mean < args.nll_max,
                f"{mean:.4f} nats/tok over {tot} tok (threshold < {args.nll_max}); "
                f"pos 0-63 {early:.4f}, pos 64+ {late:.4f}",
                value={"mean": mean, "mean_0_63": early, "mean_64p": late},
            )
        )
        checks.append(
            _chk(
                f"A1.c{c}.late_vs_early",
                late - early < args.late_margin,
                f"pos64+ - pos0-63 = {late - early:+.4f} "
                f"(threshold < {args.late_margin})",
            )
        )
        if c == base:
            continue
        deltas, worst = {}, []
        for k, v in rows.items():
            b = pl.get(base, {}).get(k)
            if not b or "mean" not in b or "mean" not in v:
                continue
            d = v["mean"] - b["mean"]
            dl = v["mean_64p"] - b["mean_64p"] if v["n_64p"] else 0.0
            deltas[k] = {"d": d, "d_64p": dl}
            if abs(d) >= args.nll_tol:
                worst.append(k)
        mx = max((abs(x["d"]) for x in deltas.values()), default=math.nan)
        mxl = max((abs(x["d_64p"]) for x in deltas.values()), default=math.nan)
        checks.append(
            _chk(
                f"A1.c{c}.delta_vs_c{base}",
                not worst and len(deltas) == len(pl.get(base, {})),
                f"max abs dNLL {mx:.4f} (pos64+ {mxl:.4f}) over {len(deltas)} "
                f"fixtures; {len(worst)} >= {args.nll_tol}",
                worst,
                value=deltas,
            )
        )
    return checks


# ---------------------------------------------------------------------------
# A2: greedy
# ---------------------------------------------------------------------------

FENCE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S)


def a2_request(cl: Client, p: dict, args) -> dict:
    body = {
        "model": cl.model,
        "messages": [{"role": "user", "content": p["prompt"]}],
        "max_tokens": p["max_tokens"],
        "temperature": 0.0,
        "seed": args.seed,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
    }
    t0 = time.time()
    try:
        r = cl.post("/chat/completions", body)
    except Exception as e:  # noqa: BLE001
        return {"id": p["id"], "error": str(e), "secs": time.time() - t0}
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    out = {
        "id": p["id"],
        "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "token_ids": ch.get("token_ids"),
        "finish_reason": ch.get("finish_reason"),
        "usage": r.get("usage"),
        "secs": round(time.time() - t0, 2),
    }
    if out["token_ids"] is None:  # older server: retokenize the text
        text = out["reasoning"] + out["content"]
        try:
            out["token_ids"] = cl.post(
                "/tokenize",
                {"model": cl.model, "prompt": text, "add_special_tokens": False},
                root=True,
            )["tokens"]
            out["token_ids_source"] = "retokenized"
        except Exception:  # noqa: BLE001
            out["token_ids"] = []
    return out


def text_metrics(p: dict, o: dict) -> dict:
    text = o["reasoning"] + o["content"]
    raw = text.encode("utf-8")
    m = {
        "n_tokens": len(o["token_ids"] or []),
        "chars": len(text),
        "finish_reason": o["finish_reason"],
        "zlib_ratio": len(raw) / max(1, len(zlib.compress(raw, 9))),
        "non_ascii": sum(ord(ch) > 127 for ch in text) / max(1, len(text)),
    }
    words = text.split()
    n = 8
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    m["max_ngram_rep"] = max(grams.values(), default=0)
    if p["kind"] == "code":
        blocks = FENCE.findall(o["content"])
        errs = 0
        for b in blocks:
            try:
                ast.parse(b)
            except SyntaxError:
                errs += 1
        m["code_blocks"] = len(blocks)
        m["code_parse_errors"] = errs
    if p.get("expect_substring"):
        m["answer_found"] = p["expect_substring"] in o["content"]
    return m


def run_a2(cl, prompts, levels, args, log) -> dict:
    per_level = {}
    for c in levels:
        log(f"A2: c={c} ({len(prompts)} prompts, pool of {c})")
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            outs = list(ex.map(lambda p: a2_request(cl, p, args), prompts))
        per_level[c] = {o["id"]: o for o in outs}
        ntok = sum(len(o.get("token_ids") or []) for o in outs)
        dt = time.time() - t0
        log(f"A2: c={c} done in {dt:.1f}s ({ntok} tok, {ntok / dt:.1f} tok/s)")
    return {"per_level": per_level}


def a2_checks(a2: dict, prompts, levels, args, ref: dict | None) -> tuple:
    checks, table = [], {}
    base = levels[0]
    pl = a2["per_level"]
    for c in levels:
        fails = {
            k: [] for k in ("err", "prefix", "zlib", "ngram", "ascii", "code", "short")
        }
        for p in prompts:
            o = pl[c].get(p["id"])
            if not o or "error" in o:
                fails["err"].append(p["id"])
                table[(c, p["id"])] = {"error": (o or {}).get("error", "missing")}
                continue
            m = text_metrics(p, o)
            if c != base:
                b = pl[base].get(p["id"])
                if b and "error" not in b:
                    ref_ids = b["token_ids"] or []
                    m["prefix_match"] = lcp(ref_ids, o["token_ids"] or [])
                    need = min(args.prefix_min, len(ref_ids))
                    m["prefix_need"] = need
                    if m["prefix_match"] < need:
                        fails["prefix"].append(p["id"])
            if m["chars"] >= 1000 and not (
                args.zlib_min <= m["zlib_ratio"] <= args.zlib_max
            ):
                fails["zlib"].append(p["id"])
            if m["max_ngram_rep"] > args.ngram_max:
                fails["ngram"].append(p["id"])
            if m["non_ascii"] > args.non_ascii_max:
                fails["ascii"].append(p["id"])
            if p["kind"] == "code" and (
                m["code_parse_errors"]
                or (m["code_blocks"] == 0 and o["finish_reason"] == "stop")
            ):
                fails["code"].append(p["id"])
            if p["kind"] != "long" and m["n_tokens"] < args.min_tokens:
                fails["short"].append(p["id"])
            table[(c, p["id"])] = m
        checks.append(
            _chk(
                f"A2.c{c}.requests",
                not fails["err"],
                f"{len(fails['err'])} errors",
                fails["err"],
            )
        )
        if c != base:
            pm = [table[(c, p["id"])].get("prefix_match") for p in prompts]
            pm = [x for x in pm if x is not None]
            checks.append(
                _chk(
                    f"A2.c{c}.prefix_vs_c{base}",
                    not fails["prefix"],
                    f"min prefix {min(pm, default=0)} tok, median "
                    f"{sorted(pm)[len(pm) // 2] if pm else 0}; need "
                    f">= min({args.prefix_min}, len(c{base})); "
                    f"{len(fails['prefix'])} short",
                    fails["prefix"],
                )
            )
        checks.append(
            _chk(
                f"A2.c{c}.zlib_ratio",
                not fails["zlib"],
                f"ratio must be in [{args.zlib_min}, {args.zlib_max}]",
                fails["zlib"],
            )
        )
        checks.append(
            _chk(
                f"A2.c{c}.ngram_repetition",
                not fails["ngram"],
                f"max 8-word n-gram count <= {args.ngram_max}",
                fails["ngram"],
            )
        )
        checks.append(
            _chk(
                f"A2.c{c}.non_ascii",
                not fails["ascii"],
                f"non-ASCII char ratio <= {args.non_ascii_max}",
                fails["ascii"],
            )
        )
        checks.append(
            _chk(
                f"A2.c{c}.code_parses",
                not fails["code"],
                "fenced python blocks parse with ast",
                fails["code"],
            )
        )
        checks.append(
            _chk(
                f"A2.c{c}.min_length",
                not fails["short"],
                f">= {args.min_tokens} output tokens (non-long prompts)",
                fails["short"],
            )
        )
    if ref is not None:
        rb = ref.get("a2", {})
        short = []
        for p in prompts:
            o, r = pl[base].get(p["id"]), rb.get(p["id"])
            if not o or not r or "error" in o or "error" in r:
                short.append(p["id"])
                continue
            n = lcp(r["token_ids"] or [], o["token_ids"] or [])
            table.setdefault((base, p["id"]), {})["prefix_vs_ref"] = n
            if n < min(args.prefix_min, len(r["token_ids"] or [])):
                short.append(p["id"])
        checks.append(
            _chk(
                f"A2.c{base}.prefix_vs_reference",
                not short,
                f"{len(short)} prompts diverge from reference before "
                f"min({args.prefix_min}, len)",
                short,
            )
        )
    return checks, table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _chk(name, ok, detail, failing=None, value=None) -> dict:
    d = {"name": name, "pass": bool(ok), "detail": detail}
    if failing:
        d["failing"] = failing
    if value is not None:
        d["value"] = value
    return d


def a1_ref_checks(a1: dict, ref: dict, base: int, tol: float) -> list[dict]:
    cur = a1["per_level"].get(base, {})
    rb = ref.get("a1", {})
    bad, mx = [], 0.0
    for k, v in cur.items():
        r = rb.get(k)
        if not r or "mean" not in r or "mean" not in v:
            bad.append(k)
            continue
        d = abs(v["mean"] - r["mean"])
        mx = max(mx, d)
        if d >= tol:
            bad.append(k)
    return [
        _chk(
            f"A1.c{base}.delta_vs_reference",
            not bad,
            f"max abs dNLL {mx:.4f}; {len(bad)} >= {tol} or missing",
            bad,
        )
    ]


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def load_reference(d: str) -> dict:
    ref: dict = {"a1": {}, "a2": {}}
    p1, p2 = os.path.join(d, "a1_nll.jsonl"), os.path.join(d, "a2_outputs.jsonl")
    if os.path.exists(p1):
        rows = _read_jsonl(p1)
        base = min((r["concurrency"] for r in rows), default=None)
        ref["a1"] = {r["id"]: r for r in rows if r["concurrency"] == base}
    if os.path.exists(p2):
        rows = _read_jsonl(p2)
        b2 = min((r["concurrency"] for r in rows), default=None)
        ref["a2"] = {r["id"]: r for r in rows if r["concurrency"] == b2}
    return ref


def write_outputs(out, results, a1, a2, a2_table, levels) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    if a1:
        with open(os.path.join(out, "a1_nll.jsonl"), "w") as fh:
            for c, rows in a1["per_level"].items():
                for k, v in rows.items():
                    fh.write(json.dumps({"concurrency": c, "id": k, **v}) + "\n")
    if a2:
        with open(os.path.join(out, "a2_outputs.jsonl"), "w") as fh:
            for c, rows in a2["per_level"].items():
                for k, v in rows.items():
                    m = a2_table.get((c, k), {})
                    fh.write(
                        json.dumps(
                            {"concurrency": c, **v, "metrics": m}, ensure_ascii=False
                        )
                        + "\n"
                    )
    lines = [
        f"# Gate A: {'PASS' if results['pass'] else 'FAIL'}",
        "",
        (
            f"- server: `{results['base_url']}` model `{results['model']}` "
            f"version `{results.get('server_version')}`"
        ),
        (
            f"- concurrency: {levels}; started {results['started']}, "
            f"took {results['elapsed_s']:.0f}s"
        ),
        f"- thresholds: {json.dumps(results['thresholds'])}",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for ch in results["checks"]:
        fl = ch.get("failing")
        extra = f" — failing: {', '.join(map(str, fl[:6]))}" if fl else ""
        if fl and len(fl) > 6:
            extra += f" (+{len(fl) - 6})"
        lines.append(
            f"| {ch['name']} | {'PASS' if ch['pass'] else 'FAIL'} | "
            f"{ch['detail']}{extra} |"
        )
    if a1:
        lines += [
            "",
            "## A1 per fixture (mean NLL nats/token)",
            "",
            "| fixture | ctx tok | ref ctx | asst tok | "
            + " | ".join(f"c{c} all / 0-63 / 64+" for c in levels)
            + " |",
            "|---|---|---|---|" + "---|" * len(levels),
        ]
        for p in a1["prepared"]:
            if "error" in p:
                lines.append(f"| {p['id']} | tokenize error | | |" + " |" * len(levels))
                continue
            cells = []
            for c in levels:
                v = a1["per_level"].get(c, {}).get(p["id"], {})
                cells.append(
                    "error"
                    if "mean" not in v
                    else f"{v['mean']:.3f} / {v['mean_0_63']:.3f} / {v['mean_64p']:.3f}"
                )
            lines.append(
                f"| {p['id']} | {p['n_ctx']} | {p['ref_prompt_tokens']} | "
                f"{p['n_asst']} | " + " | ".join(cells) + " |"
            )
    if a2:
        lines += [
            "",
            "## A2 per prompt",
            "",
            "| prompt | c | tok | finish | prefix | zlib | ngram | nonascii | code |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for (c, k), m in sorted(a2_table.items(), key=lambda x: (x[0][1], x[0][0])):
            if "error" in m:
                lines.append(f"| {k} | {c} | error | | | | | | |")
                continue
            code = (
                f"{m['code_blocks']}b/{m['code_parse_errors']}err"
                if "code_blocks" in m
                else ""
            )
            if "answer_found" in m:
                code = f"answer {'found' if m['answer_found'] else 'missing'}"
            lines.append(
                f"| {k} | {c} | {m['n_tokens']} | {m['finish_reason']} | "
                f"{m.get('prefix_match', '-')} | {m['zlib_ratio']:.2f} | "
                f"{m['max_ngram_rep']} | {m['non_ascii']:.3f} | {code} |"
            )
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def estimate(fixtures, prompts, levels, decode_tps, prefill_tps) -> str:
    a1_tok = sum(
        (f.get("ref_prompt_tokens") or 0) + (f.get("ref_completion_tokens") or 0)
        for f in fixtures
    )
    a2_out = sum(p["max_tokens"] for p in prompts)
    a2_in = sum(p.get("approx_prompt_tokens", 100) for p in prompts)
    a1_s = len(levels) * a1_tok / prefill_tps
    c1 = a2_out / decode_tps + a2_in / prefill_tps
    # concurrent levels: bounded below by the longest request at c1 speed
    longest = max(p["max_tokens"] for p in prompts) / decode_tps
    par = sum(max(longest, c1 / min(c, 4)) for c in levels if c > 1)
    return (
        f"A1 {len(levels)}x{a1_tok} prefill tok ~{a1_s / 60:.1f} min at "
        f"{prefill_tps} tok/s; A2 <= {a2_out} decode tok per level: c1 "
        f"~{c1 / 60:.1f} min at {decode_tps} tok/s, other levels ~"
        f"{par / 60:.1f} min (upper bound if every max_tokens is used)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True, help="e.g. http://h:8000/v1")
    ap.add_argument("--model", default="flint")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default=os.path.join(HERE, "fixtures"))
    ap.add_argument(
        "--reference",
        default=None,
        help="out dir of a gate_a run on a reference server",
    )
    ap.add_argument("--skip-a1", action="store_true")
    ap.add_argument("--skip-a2", action="store_true")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N fixtures/prompts (smoke)",
    )
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--thinking",
        choices=["off", "on"],
        default="off",
        help="A2 chat_template_kwargs.enable_thinking",
    )
    ap.add_argument("--nll-max", type=float, default=1.0)
    ap.add_argument("--nll-tol", type=float, default=0.02)
    ap.add_argument("--ref-nll-tol", type=float, default=0.05)
    ap.add_argument("--late-margin", type=float, default=0.5)
    ap.add_argument("--prefix-min", type=int, default=256)
    ap.add_argument("--zlib-min", type=float, default=1.4)
    ap.add_argument("--zlib-max", type=float, default=5.0)
    ap.add_argument("--ngram-max", type=int, default=6)
    ap.add_argument("--non-ascii-max", type=float, default=0.05)
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument(
        "--plan",
        action="store_true",
        help="print the request plan and runtime estimate only",
    )
    ap.add_argument("--decode-tps", type=float, default=178.0)
    ap.add_argument("--prefill-tps", type=float, default=2500.0)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    tf_meta, fixtures = load_jsonl(os.path.join(args.fixtures, "teacher_forced.jsonl"))
    _, prompts = load_jsonl(os.path.join(args.fixtures, "greedy_prompts.jsonl"))
    if args.limit:
        fixtures, prompts = fixtures[: args.limit], prompts[: args.limit]
    est = estimate(fixtures, prompts, levels, args.decode_tps, args.prefill_tps)
    thresholds = {
        "a1_mean_nll_max": args.nll_max,
        "a1_delta_tol": args.nll_tol,
        "a1_late_minus_early_max": args.late_margin,
        "a2_prefix_min": args.prefix_min,
        "a2_zlib": [args.zlib_min, args.zlib_max],
        "a2_ngram8_max": args.ngram_max,
        "a2_non_ascii_max": args.non_ascii_max,
        "a2_min_tokens": args.min_tokens,
        "a2_thinking": args.thinking,
    }
    print(
        f"Gate A: {len(fixtures)} A1 fixtures, {len(prompts)} A2 prompts, "
        f"concurrency {levels}"
    )
    print(f"Thresholds: {json.dumps(thresholds)}")
    print(f"Estimate: {est}")
    if args.plan:
        return 0

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cl = Client(args.base_url, args.model, args.timeout)
    try:
        models = [m["id"] for m in cl.get("/models")["data"]]
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach server: {e}", file=sys.stderr)
        return 2
    if args.model not in models:
        print(f"model {args.model!r} not served (have {models})", file=sys.stderr)
        return 2
    try:
        version = cl.get("/version", root=True).get("version")
    except Exception:  # noqa: BLE001
        version = None
    ref = load_reference(args.reference) if args.reference else None

    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    checks, a1, a2, table = [], None, None, {}
    if not args.skip_a1:
        a1 = run_a1(cl, fixtures, tf_meta.get("tools"), levels, args, log)
        checks += a1_checks(a1, levels, args)
        if ref and ref.get("a1"):
            checks += a1_ref_checks(a1, ref, levels[0], args.ref_nll_tol)
        rend = [p for p in a1["prepared"] if "error" not in p]
        mism = [
            p["id"]
            for p in rend
            if p["ref_prompt_tokens"] and p["n_ctx"] != p["ref_prompt_tokens"]
        ]
        log(
            f"A1: rendered ctx == trajectory prompt_tokens for "
            f"{len(rend) - len(mism)}/{len(rend)} fixtures (informational)"
        )
        a1["render_mismatch"] = mism
    if not args.skip_a2:
        a2 = run_a2(cl, prompts, levels, args, log)
        c2, table = a2_checks(a2, prompts, levels, args, ref)
        checks += c2
    results = {
        "pass": all(c["pass"] for c in checks) and bool(checks),
        "base_url": args.base_url,
        "model": args.model,
        "server_version": version,
        "concurrency": levels,
        "started": started,
        "elapsed_s": time.time() - t0,
        "thresholds": thresholds,
        "estimate": est,
        "checks": checks,
        "a1": a1
        and {
            "prepared": a1["prepared"],
            "render_mismatch": a1["render_mismatch"],
            "per_level": a1["per_level"],
        },
        "a2_metrics": {f"c{c}/{k}": m for (c, k), m in table.items()},
    }
    if results["a1"]:  # keep results.json small: per-token lists stay out
        for rows in results["a1"]["per_level"].values():
            for v in rows.values():
                v.pop("nll", None)
    write_outputs(args.out, results, a1, a2, table, levels)
    for ch in checks:
        print(f"  {'PASS' if ch['pass'] else 'FAIL'}  {ch['name']}: {ch['detail']}")
    print(
        f"Gate A overall: {'PASS' if results['pass'] else 'FAIL'} "
        f"-> {os.path.join(args.out, 'summary.md')}"
    )
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
