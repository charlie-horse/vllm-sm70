#!/usr/bin/env python3
"""Build the Gate A fixtures.

Two files are written to ``fixtures/``:

* ``teacher_forced.jsonl``: ~30 (context, assistant turn) pairs taken from
  resolved SWE-bench instances of a known-good run (mini-swe-agent 2.4.6
  trajectories). Contexts are spread log-uniformly over 2K-32K prompt tokens
  (the server's own ``usage.prompt_tokens`` for that turn), one pair per
  instance, round-robin over repos, preferring turns that carry reasoning.
* ``greedy_prompts.jsonl``: 20 hand-written code/math/prose prompts asking
  for 500-2000 output tokens, plus 4 long-context prompts (~8K/16K/24K/32K
  tokens) built from this repo's own source with a question about it.

The trajectories live on a remote host. Nothing is written there: this file
is piped to the remote interpreter (stdlib only) which streams slim records
to stdout::

    .venv/bin/python build_fixtures.py --ssh r440 \\
        --remote-python '~/flint-swebench/.venv/bin/python' \\
        --runs-dir '~/flint-swebench/runs/att1'

``--runs-dir`` may also be a local copy (omit ``--ssh``).
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# mini-swe-agent 2.4.6 tool schema (minisweagent/models/utils/actions_toolcall.py)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

# ---------------------------------------------------------------------------
# Remote side (stdlib only; runs on the host that has the trajectories)
# ---------------------------------------------------------------------------


def _resolved_ids(runs_dir: str) -> set[str]:
    ids: set[str] = set()
    for f in glob.glob(os.path.join(runs_dir, "eval", "*.json")):
        try:
            with open(f) as fh:
                ids.update(json.load(fh).get("resolved_ids") or [])
        except (OSError, ValueError):
            continue
    return ids


def _traj_path(runs_dir: str, iid: str) -> str:
    return os.path.join(runs_dir, "agent", iid, iid, f"{iid}.traj.json")


def _usage(msg: dict) -> dict:
    resp = (msg.get("extra") or {}).get("response") or {}
    return resp.get("usage") or {}


def _reasoning(msg: dict) -> str | None:
    for key in ("reasoning_content", "reasoning"):
        val = msg.get(key)
        if isinstance(val, str) and val:
            return val
    val = (msg.get("provider_specific_fields") or {}).get("reasoning")
    return val if isinstance(val, str) and val else None


def _slim(msg: dict) -> dict:
    out = {"role": msg["role"], "content": msg.get("content")}
    if msg["role"] == "assistant":
        r = _reasoning(msg)
        if r:
            out["reasoning"] = r
        tcs = msg.get("tool_calls") or []
        if tcs:
            out["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tcs
            ]
    elif msg["role"] == "tool":
        out["tool_call_id"] = msg.get("tool_call_id")
    return out


def remote_candidates(runs_dir: str) -> None:
    runs_dir = os.path.expanduser(runs_dir)
    resolved = _resolved_ids(runs_dir)
    for iid in sorted(resolved):
        try:
            with open(_traj_path(runs_dir, iid)) as fh:
                msgs = json.load(fh)["messages"]
        except (OSError, ValueError, KeyError):
            continue
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            u = _usage(m)
            if not u.get("prompt_tokens"):
                continue
            rec = {
                "id": iid,
                "turn": i,
                "pt": u["prompt_tokens"],
                "ct": u.get("completion_tokens") or 0,
                "r": bool(_reasoning(m)),
                "tc": len(m.get("tool_calls") or []),
            }
            sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.write(json.dumps({"n_resolved": len(resolved)}) + "\n")


def remote_pairs(runs_dir: str, picks: str) -> None:
    runs_dir = os.path.expanduser(runs_dir)
    for item in picks.split(","):
        iid, turn = item.rsplit(":", 1)
        turn = int(turn)
        with open(_traj_path(runs_dir, iid)) as fh:
            msgs = json.load(fh)["messages"]
        target = msgs[turn]
        u = _usage(target)
        rec = {
            "id": iid,
            "turn": turn,
            "ref_prompt_tokens": u.get("prompt_tokens"),
            "ref_completion_tokens": u.get("completion_tokens"),
            "context": [_slim(m) for m in msgs[:turn]],
            "target": _slim(target),
        }
        sys.stdout.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Local side
# ---------------------------------------------------------------------------


def _run_remote(args, mode_args: list[str]) -> list[dict]:
    script = os.path.abspath(__file__)
    if args.ssh:
        remote = f"{args.remote_python} -I - " + " ".join(
            shlex.quote(a) for a in mode_args
        )
        cmd = ["ssh", "-o", "ConnectTimeout=20", args.ssh, remote]
        with open(script, "rb") as fh:
            out = subprocess.run(
                cmd, stdin=fh, capture_output=True, check=True, timeout=1800
            ).stdout
    else:
        cmd = [sys.executable, "-I", script, *mode_args]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return [json.loads(line) for line in out.decode().splitlines() if line]


def select(cands: list[dict], n: int, lo: int, hi: int, seed: int) -> list[dict]:
    """Pick ``n`` turns, log-uniform over [lo, hi] prompt tokens.

    One turn per instance; repos are visited round-robin inside each bin so
    the set stays varied; turns with reasoning and >=160 output tokens (so
    positions 64+ are well populated) are preferred.
    """
    rng = random.Random(seed)
    pool = [
        c
        for c in cands
        if lo <= c["pt"] <= hi and 160 <= c["ct"] <= 4000 and c["tc"] > 0
    ]
    rng.shuffle(pool)
    pool.sort(key=lambda c: not c["r"])  # stable: reasoning turns first
    edges = [lo * (hi / lo) ** (k / n) for k in range(n + 1)]
    used_ids: set[str] = set()
    repo_count: dict[str, int] = {}
    picks = []
    for k in range(n):
        in_bin = [
            c
            for c in pool
            if edges[k] <= c["pt"] < edges[k + 1] and c["id"] not in used_ids
        ]
        if not in_bin:
            continue
        # least-used repo first, reasoning preferred (pool order is stable)
        in_bin.sort(
            key=lambda c: (repo_count.get(c["id"].split("__")[0], 0), not c["r"])
        )
        c = in_bin[0]
        picks.append(c)
        used_ids.add(c["id"])
        repo = c["id"].split("__")[0]
        repo_count[repo] = repo_count.get(repo, 0) + 1
    return picks


def _chars(msgs: list[dict]) -> int:
    n = 0
    for m in msgs:
        n += len(m.get("content") or "") + len(m.get("reasoning") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"]["arguments"])
    return n


# ---- greedy prompts --------------------------------------------------------

GREEDY = [
    # (kind, max_tokens, prompt)
    (
        "code",
        1200,
        (
            "Write a complete Python module implementing an LRU cache "
            "class with get, put, delete and resize methods, O(1) operations using a "
            "dict and a doubly linked list, full docstrings and type hints. Then add "
            "a unittest.TestCase with at least eight tests. Put all code in a single "
            "```python fenced block."
        ),
    ),
    (
        "code",
        1500,
        (
            "Write a Python implementation of Dijkstra's algorithm and "
            "A* search on a 2D grid with obstacles, including a function that parses "
            "a grid from a multi-line string, reconstructs the path and prints the "
            "grid with the path marked. Include a __main__ demo. Use one ```python "
            "fenced block."
        ),
    ),
    (
        "code",
        1000,
        (
            "Write a small tokenizer and recursive-descent parser in "
            "Python for arithmetic expressions with +, -, *, /, unary minus, "
            "parentheses and variables, that evaluates them against a dict of "
            "variable values and raises clear errors. Use one ```python block."
        ),
    ),
    (
        "code",
        2000,
        (
            "Implement in Python a minimal key-value store with a "
            "write-ahead log: put/get/delete, crash recovery by replaying the log, "
            "periodic compaction into a snapshot file, and a CLI using argparse. "
            "Explain the design briefly, then give all code in one ```python block."
        ),
    ),
    (
        "code",
        800,
        (
            "Write a Python function that merges overlapping intervals, "
            "another that inserts an interval into a sorted non-overlapping list, "
            "and a third that computes the total covered length. Include doctests. "
            "Use one ```python fenced block."
        ),
    ),
    (
        "code",
        1200,
        (
            "Write an asyncio-based Python rate limiter (token bucket) "
            "usable as an async context manager, plus a demo that runs 50 fake HTTP "
            "requests with at most 5 per second and prints timing statistics. Use "
            "one ```python fenced block."
        ),
    ),
    (
        "code",
        1500,
        (
            "Write a Python class implementing a trie that supports "
            "insert, exact search, prefix search returning all completions sorted "
            "by frequency, deletion, and serialization to and from JSON. Add tests "
            "with pytest-style asserts in the same file. One ```python block."
        ),
    ),
    (
        "code",
        1000,
        (
            "Write a Python script that reads a CSV file of sales "
            "(date, region, product, units, price), computes monthly revenue per "
            "region, the top three products per month, and month-over-month growth, "
            "using only the standard library. One ```python fenced block."
        ),
    ),
    (
        "math",
        1200,
        (
            "Prove that there are infinitely many primes congruent to "
            "3 modulo 4. Then explain why the same elementary argument does not "
            "directly work for primes congruent to 1 modulo 4, and sketch how one "
            "can handle that case. Be rigorous and detailed."
        ),
    ),
    (
        "math",
        1500,
        (
            "Solve step by step: a fair die is rolled repeatedly until "
            "the running total is at least 10. Compute the probability distribution "
            "of the final total and its expected value. Show the recursion, a table "
            "of intermediate values, and check that the probabilities sum to one."
        ),
    ),
    (
        "math",
        1000,
        (
            "Derive the closed form of the sum of k^3 for k from 1 to "
            "n by three different methods: induction, telescoping with binomial "
            "coefficients, and a geometric argument. Explain each step."
        ),
    ),
    (
        "math",
        2000,
        (
            "Explain the singular value decomposition from first "
            "principles: state the theorem, prove existence, compute the SVD of the "
            "matrix [[3, 2, 2], [2, 3, -2]] by hand showing every step, and explain "
            "low-rank approximation and the Eckart-Young theorem."
        ),
    ),
    (
        "math",
        800,
        (
            "A tank holds 1000 litres of brine with 50 kg of salt. Fresh "
            "water flows in at 10 L/min and the well-mixed solution flows out at "
            "the same rate. Set up and solve the differential equation, find when "
            "the salt drops below 5 kg, and discuss the model's assumptions."
        ),
    ),
    (
        "math",
        1200,
        (
            "Compute the eigenvalues and eigenvectors of the 4x4 "
            "matrix with 2 on the diagonal and -1 on the first sub- and "
            "super-diagonals. Then generalize to the n x n case and prove the "
            "formula for the eigenvalues."
        ),
    ),
    (
        "prose",
        1000,
        (
            "Write a detailed explanation, for a new engineer, of how "
            "a CPU cache hierarchy works: cache lines, associativity, write-back "
            "versus write-through, coherence protocols such as MESI, and how these "
            "affect the performance of real programs. Use concrete examples."
        ),
    ),
    (
        "prose",
        1500,
        (
            "Write a short story of about 1200 words about a "
            "lighthouse keeper on a remote island who receives a letter that was "
            "posted forty years earlier. Give it a clear beginning, middle and end."
        ),
    ),
    (
        "prose",
        800,
        (
            "Compare and contrast TCP and QUIC: connection setup, "
            "head-of-line blocking, congestion control, encryption, and deployment "
            "challenges. Write it as a structured technical memo with headings."
        ),
    ),
    (
        "prose",
        1200,
        (
            "Explain how the transformer architecture works, "
            "covering tokenization, embeddings, attention, multi-head attention, "
            "positional encodings, feed-forward layers, layer normalization and the "
            "key-value cache used during inference. Be precise."
        ),
    ),
    (
        "prose",
        600,
        (
            "Write a persuasive essay arguing that cities should "
            "invest in protected bicycle lanes. Address three common objections "
            "and respond to each with evidence-based reasoning."
        ),
    ),
    (
        "prose",
        2000,
        (
            "Write a thorough history of the development of the C "
            "programming language from BCPL and B to C23, covering the key people, "
            "the standardization process, and the most important features added in "
            "each standard revision."
        ),
    ),
]

LONG_TARGETS = [8000, 16000, 24000, 32000]
LONG_SOURCES = [
    "vllm/v1/core",
    "vllm/v1/worker",
    "vllm/entrypoints/openai",
    "vllm/model_executor/layers",
]


def _py_files(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(os.path.join(REPO_ROOT, root)):
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _defs(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not n.name.startswith("__")
    ]


def long_prompt(target_tokens: int, src_root: str, cpt: float) -> dict:
    budget = int(target_tokens * cpt) - 1500
    parts, used, chosen = [], 0, []
    for path in _py_files(src_root):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        rel = os.path.relpath(path, REPO_ROOT)
        block = f"===== FILE: {rel} =====\n{src}\n"
        if used + len(block) > budget:
            if used > budget * 0.8:
                break
            continue
        parts.append(block)
        chosen.append((rel, src))
        used += len(block)
    corpus = "".join(parts)
    # A function defined exactly once, in a file near the middle of the text.
    mid = len(chosen) // 2
    order = sorted(range(len(chosen)), key=lambda k: abs(k - mid))
    pick = None
    for k in order:
        for name in _defs(chosen[k][1]):
            if len(name) > 6 and corpus.count(f"def {name}(") == 1:
                pick = (chosen[k][0], name)
                break
        if pick:
            break
    assert pick, "no unique function found"
    rel, name = pick
    question = (
        "Below are several Python source files from a code base.\n\n"
        f"{corpus}\n"
        "Question: in which of the files above is the function "
        f"`{name}` defined? Give the file path on the first line, then explain "
        "in a few sentences what the function does and how it is used."
    )
    return {
        "id": f"long_{target_tokens // 1000}k",
        "kind": "long",
        "max_tokens": 512,
        "approx_prompt_tokens": target_tokens,
        "prompt": question,
        "expect_substring": rel,
    }


def build_greedy(cpt: float) -> list[dict]:
    recs = []
    for k, (kind, mt, prompt) in enumerate(GREEDY):
        recs.append(
            {"id": f"g{k:02d}_{kind}", "kind": kind, "max_tokens": mt, "prompt": prompt}
        )
    for t, root in zip(LONG_TARGETS, LONG_SOURCES):
        recs.append(long_prompt(t, root, cpt))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", default="~/flint-swebench/runs/att1")
    ap.add_argument("--ssh", default=None, help="host holding the runs dir")
    ap.add_argument("--remote-python", default="~/flint-swebench/.venv/bin/python")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--min-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures"))
    ap.add_argument("--chars-per-token", type=float, default=None)
    ap.add_argument("--greedy-only", action="store_true")
    # internal modes, executed on the trajectory host
    ap.add_argument("--emit-candidates", action="store_true")
    ap.add_argument("--emit-pairs", default=None)
    args = ap.parse_args()

    if args.emit_candidates:
        return remote_candidates(args.runs_dir)
    if args.emit_pairs:
        return remote_pairs(args.runs_dir, args.emit_pairs)

    os.makedirs(args.out, exist_ok=True)
    cpt = args.chars_per_token
    if not args.greedy_only:
        rows = _run_remote(args, ["--emit-candidates", "--runs-dir", args.runs_dir])
        n_resolved = rows[-1].get("n_resolved")
        cands = [r for r in rows if "id" in r]
        picks = select(cands, args.n, args.min_tokens, args.max_tokens, args.seed)
        print(f"resolved={n_resolved} candidates={len(cands)} picked={len(picks)}")
        spec = ",".join(f"{p['id']}:{p['turn']}" for p in picks)
        pairs = _run_remote(args, ["--emit-pairs", spec, "--runs-dir", args.runs_dir])
        pairs.sort(key=lambda p: p["ref_prompt_tokens"])
        ratios = []
        path = os.path.join(args.out, "teacher_forced.jsonl")
        with open(path, "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "_meta": {
                            "source": "SWE-bench Verified att1, seqs=2 server",
                            "sampling": "T=1.0 top_p=0.95 top_k=20, not exact",
                            "tools": [BASH_TOOL],
                        }
                    }
                )
                + "\n"
            )
            for p in pairs:
                p["repo"] = p["id"].split("__")[0]
                ratios.append(_chars(p["context"]) / p["ref_prompt_tokens"])
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        cpt = cpt or sorted(ratios)[len(ratios) // 2] * 0.97
        pts = [p["ref_prompt_tokens"] for p in pairs]
        cts = [p["ref_completion_tokens"] for p in pairs]
        repos = sorted({p["repo"] for p in pairs})
        nr = sum(1 for p in pairs if p["target"].get("reasoning"))
        print(
            f"wrote {path}: {len(pairs)} pairs, {os.path.getsize(path) / 1e6:.2f}"
            f" MB, prompt tokens {min(pts)}-{max(pts)} (median "
            f"{sorted(pts)[len(pts) // 2]}), completion tokens {min(cts)}-"
            f"{max(cts)}, {nr} with reasoning, {len(repos)} repos: " + ", ".join(repos)
        )
    cpt = cpt or 3.3
    recs = build_greedy(cpt)
    path = os.path.join(args.out, "greedy_prompts.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({"_meta": {"chars_per_token": round(cpt, 3)}}))
        fh.write("\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"wrote {path}: {len(recs)} prompts "
        f"({os.path.getsize(path) / 1e6:.2f} MB, chars/token {cpt:.2f})"
    )


if __name__ == "__main__":
    main()
