# SPDX-License-Identifier: Apache-2.0
"""Quality / speed probe for the Qwen Phase 2 matrix (stdlib only).

Talks to a running `vllm serve` over the OpenAI-compatible HTTP API:

  probe.py health  --url URL
  probe.py run     --url URL --model NAME --row ROW --out results.json
                   --max-model-len N [--spec] [--no-gate]
  probe.py summary --out-dir DIR
  probe.py ab-run     --url URL --model NAME --label on|off --out ab.json
  probe.py ab-compare --on on/ab.json --off off/ab.json --out ab_summary.json

Phases of `run`: math (10 problems, exact numeric answer), code (5 tiny
functions, asserts executed in a subprocess with a timeout), needle in a
haystack at ~max-model-len (3 trials), no-gibberish at concurrency 20, and
single-stream decode tok/s at ~4K context. For --spec rows it also diffs the
vllm:spec_decode_* counters from /metrics around each phase.

`ab-run` / `ab-compare` are the M8 QPN8 A/B (m8_qpn8_ab.sh): decode tok/s at
concurrency 1..4 (~1K-token prompt, 256 forced tokens, median of 3 rounds)
and greedy outputs for 5 fixed prompts, compared on vs off.

Quality requests are greedy (temperature 0) and send
chat_template_kwargs.enable_thinking=false explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 127.0.0.1 must never go through an HTTP(S)_PROXY from the environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
REQ_TIMEOUT = 900

GATES = {"math": 9, "code": 5, "needle": 3}  # minimum correct; gibberish: 0 flagged

# --------------------------------------------------------------------------
# Fixed problem sets
# --------------------------------------------------------------------------
MATH = [
    ("What is 17 multiplied by 23?", 391),
    ("What is the sum of all integers from 1 to 100 inclusive?", 5050),
    ("A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
     "How many kilometres does it travel in total?", 270),
    ("How many prime numbers are there that are less than 50?", 15),
    ("What is 2^10 + 3^5?", 1267),
    ("What is the remainder when 1234567 is divided by 9?", 1),
    ("A rectangle has a perimeter of 46 cm and a length of 15 cm. "
     "What is its area in square centimetres?", 120),
    ("What is the greatest common divisor of 84 and 126?", 42),
    ("Pens cost 3 dollars each. For every 4 pens you pay for, you get 1 more pen "
     "free. What is the least number of dollars you must pay to get exactly 15 pens?", 36),
    ("What is the positive solution of x^2 - 5x - 84 = 0?", 12),
]
MATH_SUFFIX = ("\nWork it out briefly, then give the final answer on the last line "
               "in the form 'ANSWER: <number>'.")

CODE = [
    ("is_palindrome",
     "Write a Python function is_palindrome(s) that returns True if the string s "
     "reads the same forwards and backwards when only alphanumeric characters are "
     "kept and case is ignored, else False.",
     "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
     "assert is_palindrome('hello') is False\n"
     "assert is_palindrome('') is True\n"),
    ("fizzbuzz",
     "Write a Python function fizzbuzz(n) that returns a list of strings for the "
     "numbers 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
     "'FizzBuzz' for multiples of both, otherwise the number as a string.",
     "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
     "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
     "assert len(fizzbuzz(30)) == 30\n"),
    ("second_largest",
     "Write a Python function second_largest(nums) that returns the second largest "
     "distinct value in the list nums, or None if there is no such value.",
     "assert second_largest([3, 1, 4, 4, 2]) == 3\n"
     "assert second_largest([5, 5]) is None\n"
     "assert second_largest([1]) is None\n"
     "assert second_largest([-1, -2]) == -2\n"),
    ("word_count",
     "Write a Python function word_count(text) that splits text on whitespace, "
     "lowercases each word, and returns a dict mapping each word to its count.",
     "assert word_count('the cat and the hat') == {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}\n"
     "assert word_count('A a B') == {'a': 2, 'b': 1}\n"
     "assert word_count('') == {}\n"),
    ("flatten",
     "Write a Python function flatten(lst) that flattens arbitrarily nested lists "
     "into a single flat list, keeping the order of elements.",
     "assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]\n"
     "assert flatten([]) == []\n"
     "assert flatten([[[]], 'ab']) == ['ab']\n"),
]
CODE_SUFFIX = ("\nReply with only the function in a single ```python code block. "
               "Use only the standard library.")

GIBBERISH_PROMPTS = [
    "Explain how a rainbow forms.",
    "Describe the water cycle in a few paragraphs.",
    "What are the main differences between TCP and UDP?",
    "Give a short history of the printing press.",
    "Explain what a hash table is and how collisions are handled.",
    "Write a short story about a cat who learns to sail.",
    "Summarise the causes of the First World War.",
    "How does photosynthesis work?",
    "Explain the difference between a virus and a bacterium.",
    "Write a polite email declining a meeting invitation.",
    "What is the Pythagorean theorem and why is it true?",
    "Describe how a bill becomes a law in a parliamentary system.",
    "Explain recursion to a beginner programmer with an example.",
    "What are the health benefits of regular exercise?",
    "Describe the plot of a heist movie you invent.",
    "Explain how vaccines train the immune system.",
    "List some tips for writing clear technical documentation.",
    "How do airplanes stay in the air?",
    "Explain supply and demand with a simple example.",
    "Describe a walk through a forest in autumn.",
]

# Typographic characters a normal English answer may contain.
_OK_NON_ASCII = set("‘’“”–—…•·°×→≤≥é")


# --------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU: sm70/tests/scripts/test_qwen_phase2_probe.py)
# --------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*[*$\\(\s]*(-?\d[\d,]*(?:\.\d+)?)", re.I)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_numeric_answer(text: str) -> float | None:
    """Last 'ANSWER: <n>' in text, else the last number; None if no number."""
    m = _ANSWER_RE.findall(text)
    raw = m[-1] if m else (_NUM_RE.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def extract_code(text: str, func_name: str) -> str:
    """The fenced block defining func_name, else the first fenced block, else text."""
    blocks = re.findall(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", text, re.S | re.I)
    for b in blocks:
        if re.search(rf"\bdef\s+{re.escape(func_name)}\s*\(", b):
            return b
    return blocks[0] if blocks else text


def run_code_check(code: str, asserts: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Execute code + asserts in an isolated subprocess (python -I, empty env)."""
    with tempfile.TemporaryDirectory(prefix="qp2-code-") as d:
        path = Path(d, "check.py")
        path.write_text(code + "\n\n" + asserts + "\nprint('QP2_OK')\n")
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, env={},
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        ok = r.returncode == 0 and "QP2_OK" in r.stdout
        return ok, "" if ok else (r.stderr.strip().splitlines() or ["rc=%d" % r.returncode])[-1]


def gibberish_reasons(text: str) -> list[str]:
    """Why an English answer looks broken; [] if it looks fine."""
    reasons = []
    t = text.strip()
    if not t:
        return ["empty"]
    if "�" in t:
        reasons.append("replacement-char")
    bad = sum(1 for c in t if ord(c) > 127 and c not in _OK_NON_ASCII)
    if bad / len(t) > 0.15:
        reasons.append(f"non-ascii {bad / len(t):.2f}")
    if re.search(r"(\S)\1{39,}", t):
        reasons.append("char-run")
    words = t.split()
    if len(words) >= 20:
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        ratio = len(set(grams)) / len(grams)
        if ratio < 0.5:
            reasons.append(f"repeated-4grams {ratio:.2f}")
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 8:
        reasons.append(f"word-run {best}")
    return reasons


_ADJ = ["quiet", "amber", "distant", "ancient", "silver", "narrow", "busy", "pale",
        "broad", "gentle", "rusty", "hollow", "bright", "muddy", "steep", "calm"]
_NOUN = ["farmer", "river", "lantern", "merchant", "bridge", "orchard", "sailor",
         "library", "garden", "miller", "harbour", "clock", "wagon", "teacher", "mill"]
_VERB = ["passed", "watched", "repaired", "described", "visited", "painted",
         "measured", "followed", "ignored", "admired", "crossed", "counted"]
_PLACE = ["the old market", "the northern hills", "the town square", "the coast",
          "the valley road", "the east gate", "the quiet lake", "the station"]


def filler_sentences(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} the "
            f"{rng.choice(_ADJ)} {rng.choice(_NOUN)} near {rng.choice(_PLACE)}."
            for _ in range(n)]


def needle_prompt(n_sentences: int, depth: float, seed: int) -> tuple[str, str]:
    """(prompt, expected passphrase) with the needle at depth in [0, 1]."""
    rng = random.Random(1000 + seed)
    color = ["crimson", "cobalt", "emerald"][seed % 3]
    code = f"{rng.choice(['TANGERINE', 'MARIGOLD', 'GRANITE', 'PELICAN'])}-{rng.randint(1000, 9999)}"
    sents = filler_sentences(n_sentences, seed)
    sents.insert(int(depth * len(sents)),
                 f"The secret passphrase for the {color} vault is {code}.")
    prompt = (" ".join(sents) + f"\n\nWhat is the secret passphrase for the {color} "
              "vault? Reply with the passphrase only.")
    return prompt, code


def parse_prom(text: str) -> dict[str, float]:
    """Prometheus text -> {name or name{position=..}: value}, summed over other labels."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode"):
            continue
        m = re.match(r"([^{\s]+)(\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        pos = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{pos.group(1)}]" if pos else name
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    def d(k):
        return after.get(k, 0.0) - before.get(k, 0.0)

    drafts = d("vllm:spec_decode_num_drafts_total")
    draft_toks = d("vllm:spec_decode_num_draft_tokens_total")
    acc = d("vllm:spec_decode_num_accepted_tokens_total")
    pos_keys = sorted((k for k in after if k.startswith(
        "vllm:spec_decode_num_accepted_tokens_per_pos_total[")),
        key=lambda k: int(k.rsplit("[", 1)[1].rstrip("]")))
    return {
        "drafts": drafts, "draft_tokens": draft_toks, "accepted_tokens": acc,
        "acceptance_rate": acc / draft_toks if draft_toks else None,
        "mean_acceptance_length": 1 + acc / drafts if drafts else None,
        "per_position_rate": [d(k) / drafts for k in pos_keys] if drafts else [],
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _http(url: str, body: dict | None = None, timeout: float = REQ_TIMEOUT):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return _OPENER.open(req, timeout=timeout)


def get_json(url, body=None):
    with _http(url, body) as r:
        return json.loads(r.read())


class Client:
    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def chat(self, content: str, max_tokens: int, **extra) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}, **extra}
        r = get_json(self.base + "/v1/chat/completions", body)
        lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
        return {"text": r["choices"][0]["message"].get("content") or "",
                "finish_reason": r["choices"][0].get("finish_reason"),
                "usage": r.get("usage", {}),
                "tokens": [t.get("token") for t in lp] or None}

    def count_tokens(self, prompt: str) -> int | None:
        try:
            return int(get_json(self.base + "/tokenize",
                                {"model": self.model, "prompt": prompt})["count"])
        except Exception:
            return None

    def spec_metrics(self) -> dict[str, float]:
        try:
            with _http(self.base + "/metrics", timeout=30) as r:
                return parse_prom(r.read().decode())
        except (urllib.error.URLError, OSError):
            return {}

    def stream_timed(self, content: str, max_tokens: int) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        t_first = t_last = None
        usage = {}
        with _http(self.base + "/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    if (ch.get("delta") or {}).get("content"):
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
        n = usage.get("completion_tokens", 0)
        dec = (n - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first else None
        return {"ttft_s": (t_first - t0) if t_first else None, "decode_tok_s": dec,
                "completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens")}


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------
def phase_math(c: Client) -> dict:
    def one(i):
        q, exp = MATH[i]
        try:
            r = c.chat(q + MATH_SUFFIX, 1024)
        except Exception as e:  # noqa: BLE001
            return {"i": i, "ok": False, "error": repr(e)}
        got = parse_numeric_answer(r["text"])
        return {"i": i, "expected": exp, "got": got, "ok": got is not None and got == exp,
                "finish_reason": r["finish_reason"], "text": r["text"][-400:]}

    with ThreadPoolExecutor(len(MATH)) as ex:
        items = list(ex.map(one, range(len(MATH))))
    return {"correct": sum(x["ok"] for x in items), "total": len(MATH), "items": items}


def phase_code(c: Client) -> dict:
    def one(i):
        name, q, asserts = CODE[i]
        try:
            r = c.chat(q + CODE_SUFFIX, 768)
        except Exception as e:  # noqa: BLE001
            return {"name": name, "ok": False, "error": repr(e)}
        code = extract_code(r["text"], name)
        ok, err = run_code_check(code, asserts)
        return {"name": name, "ok": ok, "error": err, "code": code[:1500]}

    with ThreadPoolExecutor(len(CODE)) as ex:
        items = list(ex.map(one, range(len(CODE))))
    return {"correct": sum(x["ok"] for x in items), "total": len(CODE), "items": items}


def size_haystack(c: Client, target_tokens: int, seed: int) -> tuple[int, str]:
    """Number of filler sentences whose needle prompt is <= target_tokens."""
    sample = 200
    t = c.count_tokens(" ".join(filler_sentences(sample, seed)))
    method = "tokenize"
    per = (t / sample) if t else None
    if not per:
        method, per = "estimate", 14.0  # ~11-12 tokens/sentence; err on the short side
    n = int(target_tokens / per)
    for _ in range(6):  # shrink until the full prompt fits
        prompt, _ = needle_prompt(n, 0.5, seed)
        cnt = c.count_tokens(prompt)
        if cnt is None or cnt <= target_tokens:
            break
        n = int(n * target_tokens / cnt) - 5
    return n, method


def phase_needle(c: Client, max_model_len: int) -> dict:
    target = max_model_len - 768  # room for the chat template and the answer
    n, method = size_haystack(c, target, 0)
    items = []
    for trial, depth in enumerate((0.1, 0.5, 0.9)):
        prompt, code = needle_prompt(n, depth, trial)
        try:
            r = c.chat(prompt, 48)
            ok = code.lower() in r["text"].lower()
            items.append({"trial": trial, "depth": depth, "expected": code, "ok": ok,
                          "prompt_tokens": r["usage"].get("prompt_tokens"),
                          "text": r["text"][:200]})
        except Exception as e:  # noqa: BLE001
            items.append({"trial": trial, "depth": depth, "ok": False, "error": repr(e)})
    return {"correct": sum(x["ok"] for x in items), "total": 3, "sentences": n,
            "sizing": method, "target_tokens": target, "items": items}


def phase_gibberish(c: Client) -> dict:
    def one(i):
        try:
            r = c.chat(GIBBERISH_PROMPTS[i], 256)
        except Exception as e:  # noqa: BLE001
            return {"i": i, "reasons": [f"error {e!r}"], "text": ""}
        return {"i": i, "reasons": gibberish_reasons(r["text"]), "text": r["text"][:300]}

    with ThreadPoolExecutor(20) as ex:
        items = list(ex.map(one, range(20)))
    flagged = [x for x in items if x["reasons"]]
    return {"flagged": len(flagged), "total": 20, "flagged_items": flagged,
            "samples": [x["text"] for x in items[:3]]}


def phase_speed(c: Client, ctx_tokens: int = 4096, max_tokens: int = 512) -> dict:
    n, _ = size_haystack(c, ctx_tokens - 128, 7)
    notes = " ".join(filler_sentences(n, 7))
    prompt = (notes + "\n\nUsing the notes above as loose inspiration, write a long, "
              "detailed story about a lighthouse keeper.")
    c.chat("Say hello.", 8)  # warm-up
    runs = [c.stream_timed(prompt, max_tokens) for _ in range(3)]
    tps = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
    ttft = [r["ttft_s"] for r in runs if r["ttft_s"]]
    return {"decode_tok_s_median": statistics.median(tps) if tps else None,
            "ttft_s_median": statistics.median(ttft) if ttft else None,
            "prompt_tokens": runs[0]["prompt_tokens"], "max_tokens": max_tokens, "runs": runs}


def cmd_run(a) -> int:
    c = Client(a.url, a.model)
    res: dict = {"row": a.row, "model": a.model, "spec": a.spec, "gated": a.gate,
                 "max_model_len": a.max_model_len, "started": time.strftime("%FT%T"),
                 "phases": {}, "spec_decode": {}}
    phases = [("math", phase_math), ("code", phase_code),
              ("needle", lambda c: phase_needle(c, a.max_model_len)),
              ("gibberish", phase_gibberish), ("speed", phase_speed)]
    spec0 = c.spec_metrics() if a.spec else None
    for name, fn in phases:
        before = c.spec_metrics() if a.spec else None
        t = time.time()
        try:
            out = fn(c)
        except Exception as e:  # noqa: BLE001
            out = {"error": repr(e)}
        out["seconds"] = round(time.time() - t, 1)
        res["phases"][name] = out
        if a.spec:
            res["spec_decode"][name] = spec_delta(before, c.spec_metrics())
        print(f"  {name}: {one_phase(name, out)}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    if a.spec:
        res["spec_decode"]["overall"] = spec_delta(spec0, c.spec_metrics())
    res["status"], res["fail_reasons"] = verdict(res)
    res["finished"] = time.strftime("%FT%T")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(summary_line(res), flush=True)
    return 0 if res["status"] in ("PASS", "INFO") else 1


def one_phase(name: str, out: dict) -> str:
    if "error" in out:
        return f"ERROR {out['error']}"
    if name == "gibberish":
        return f"{out['flagged']}/{out['total']} flagged"
    if name == "speed":
        return f"decode {out['decode_tok_s_median']} tok/s, ttft {out['ttft_s_median']} s"
    return f"{out.get('correct')}/{out.get('total')}"


def verdict(res: dict) -> tuple[str, list[str]]:
    p, why = res.get("phases", {}), []
    for k, need in GATES.items():
        if (p.get(k, {}).get("correct") or 0) < need:
            why.append(f"{k} {p.get(k, {}).get('correct')}<{need}")
    if p.get("gibberish", {}).get("flagged", 1) != 0:
        why.append(f"gibberish {p.get('gibberish', {}).get('flagged')} flagged")
    if res.get("spec"):
        overall = res.get("spec_decode", {}).get("overall", {})
        if not overall.get("drafts"):
            why.append("no spec-decode drafts in /metrics")
        elif not overall.get("accepted_tokens"):
            why.append("spec-decode drafts but 0 accepted tokens in /metrics")
    if res.get("error"):
        why.append(res["error"])
    if not res.get("gated", True):
        return "INFO", why
    return ("FAIL" if why else "PASS"), why


def _fmt(v, f="{:.2f}"):
    return "n/a" if v is None else f.format(v)


def summary_line(res: dict) -> str:
    p = res.get("phases", {})
    s = f"row{res.get('row')} {res.get('status', 'FAIL')}"
    if res.get("error") and not p:
        return f"{s} {res['error']}"
    g = lambda k: f"{p.get(k, {}).get('correct', 'n/a')}/{p.get(k, {}).get('total', '?')}"  # noqa: E731
    s += (f" math {g('math')} code {g('code')} needle {g('needle')}"
          f" gibberish {p.get('gibberish', {}).get('flagged', 'n/a')}/20"
          f" decode4k {_fmt(p.get('speed', {}).get('decode_tok_s_median'), '{:.1f}')} tok/s")
    if res.get("spec"):
        o = res.get("spec_decode", {}).get("overall", {})
        sp = res.get("spec_decode", {}).get("speed", {})
        s += (f" accept_len {_fmt(o.get('mean_acceptance_length'))}"
              f" (decode-phase {_fmt(sp.get('mean_acceptance_length'))})"
              f" accept_rate {_fmt(o.get('acceptance_rate'))}")
    if res.get("fail_reasons") and res.get("status") != "PASS":
        s += " [" + "; ".join(res["fail_reasons"]) + "]"
    return s


def cmd_summary(a) -> int:
    rc = 0
    rows = sorted(Path(a.out_dir).glob("row*/results.json"))
    if not rows:
        print(f"no row*/results.json under {a.out_dir}")
        return 1
    for f in rows:
        res = json.loads(f.read_text())
        if "status" not in res:
            res["status"], res["fail_reasons"] = verdict(res)
            if res["status"] == "PASS":  # phases ran but the probe never finished
                res["status"], res["fail_reasons"] = "FAIL", ["probe incomplete"]
        print(summary_line(res))
        rc |= res["status"] == "FAIL"
    return int(rc)


# --------------------------------------------------------------------------
# M8 QPN8 A/B
# --------------------------------------------------------------------------
AB_CONCURRENCY = (1, 2, 3, 4)
AB_PROMPT_TOKENS = 1024
AB_DECODE_TOKENS = 256
AB_ROUNDS = 3
AB_MATCH_TOKENS = 32
AB_RATIO_MIN = 0.98  # QPN8 / M6 decode tok/s; below this is a regression
AB_GREEDY_PROMPTS = [
    "List the first ten prime numbers and explain how you found them.",
    "Explain in plain words what a CUDA graph is.",
    "Write a haiku about a lighthouse in winter.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Translate into French: The quick brown fox jumps over the lazy dog.",
]


def ab_prompt(c: Client) -> str:
    n, _ = size_haystack(c, AB_PROMPT_TOKENS - 64, 11)
    return (" ".join(filler_sentences(n, 11)) + "\n\nUsing the notes above as loose "
            "inspiration, write a long, detailed story about a lighthouse keeper.")


def ab_round(c: Client, prompt: str, conc: int) -> dict:
    """`conc` identical streams at once; aggregate decode tok/s = sum of the
    per-stream decode rates (each stream decodes AB_DECODE_TOKENS)."""
    with ThreadPoolExecutor(conc) as ex:
        runs = list(ex.map(lambda _: c.stream_timed(prompt, AB_DECODE_TOKENS), range(conc)))
    rates = [r["decode_tok_s"] for r in runs]
    ok = all(rates) and all(r["completion_tokens"] == AB_DECODE_TOKENS for r in runs)
    return {"aggregate_tok_s": sum(rates) if ok else None, "runs": runs}


def ab_speed(c: Client) -> dict:
    prompt = ab_prompt(c)
    c.chat("Say hello.", 8)  # warm-up
    out = {"prompt_tokens": c.count_tokens(prompt), "by_concurrency": {}}
    for conc in AB_CONCURRENCY:
        c.stream_timed(prompt, 16)  # settle the prefix cache / graphs
        rounds = [ab_round(c, prompt, conc) for _ in range(AB_ROUNDS)]
        agg = [r["aggregate_tok_s"] for r in rounds if r["aggregate_tok_s"]]
        out["by_concurrency"][str(conc)] = {
            "decode_tok_s_median": statistics.median(agg) if len(agg) == AB_ROUNDS else None,
            "rounds": rounds,
        }
    return out


def ab_greedy(c: Client) -> list[dict]:
    items = []
    for i, prompt in enumerate(AB_GREEDY_PROMPTS):
        r = c.chat(prompt, AB_MATCH_TOKENS, ignore_eos=True, logprobs=True)
        items.append({"i": i, "text": r["text"], "tokens": r.get("tokens"),
                      "completion_tokens": r["usage"].get("completion_tokens")})
    return items


def cmd_ab_run(a) -> int:
    c = Client(a.url, a.model)
    res: dict = {"label": a.label, "model": a.model, "started": time.strftime("%FT%T")}
    try:
        res["speed"] = ab_speed(c)
        res["greedy"] = ab_greedy(c)
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    res["finished"] = time.strftime("%FT%T")
    Path(a.out).write_text(json.dumps(res, indent=2))
    for conc, v in res.get("speed", {}).get("by_concurrency", {}).items():
        print(f"  {a.label} c={conc}: {_fmt(v['decode_tok_s_median'], '{:.1f}')} tok/s",
              flush=True)
    if "error" in res:
        print(f"  {a.label}: ERROR {res['error']}", flush=True)
        return 1
    return 0


def ab_compare(on: dict, off: dict) -> dict:
    """QPN8 (on) vs M6 (off): per-concurrency ratio and greedy-output match."""
    lines, why = [], []
    for err in (on.get("error"), off.get("error")):
        if err:
            why.append(err)
    get = lambda r, c: (r.get("speed", {}).get("by_concurrency", {})  # noqa: E731
                        .get(str(c), {}).get("decode_tok_s_median"))
    for conc in AB_CONCURRENCY:
        t_on, t_off = get(on, conc), get(off, conc)
        ratio = t_on / t_off if t_on and t_off else None
        lines.append(f"c={conc} qpn8 {_fmt(t_on, '{:.1f}')} m6 {_fmt(t_off, '{:.1f}')} "
                     f"tok/s ratio {_fmt(ratio, '{:.3f}')}")
        if ratio is None:
            why.append(f"c={conc} no tok/s")
        elif ratio < AB_RATIO_MIN:
            why.append(f"c={conc} ratio {ratio:.3f} < {AB_RATIO_MIN}")
    g_on, g_off = on.get("greedy") or [], off.get("greedy") or []
    mismatched = []
    if len(g_on) != len(AB_GREEDY_PROMPTS) or len(g_off) != len(AB_GREEDY_PROMPTS):
        why.append("greedy outputs missing")
    else:
        for x, y in zip(g_on, g_off):
            same = (x["tokens"] == y["tokens"]) if x.get("tokens") and y.get("tokens") \
                else x["text"] == y["text"]
            if not same:
                mismatched.append(x["i"])
        if mismatched:
            why.append(f"greedy outputs differ for prompts {mismatched}")
    lines.append(f"greedy first {AB_MATCH_TOKENS} tokens: "
                 f"{len(AB_GREEDY_PROMPTS) - len(mismatched)}/{len(AB_GREEDY_PROMPTS)} match")
    return {"status": "FAIL" if why else "PASS", "lines": lines, "fail_reasons": why,
            "greedy_mismatch": mismatched}


def cmd_ab_compare(a) -> int:
    load = lambda f: json.loads(Path(f).read_text()) if Path(f).is_file() else {  # noqa: E731
        "error": f"missing {f}"}
    res = ab_compare(load(a.on), load(a.off))
    for line in res["lines"]:
        print(line)
    print(f"M8 QPN8 A/B {res['status']}" + (
        " [" + "; ".join(res["fail_reasons"]) + "]" if res["fail_reasons"] else ""))
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0 if res["status"] == "PASS" else 1


def cmd_health(a) -> int:
    try:
        with _http(a.url.rstrip("/") + "/health", timeout=5) as r:
            return 0 if r.status == 200 else 1
    except (urllib.error.URLError, OSError):
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("health")
    h.add_argument("--url", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--row", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-model-len", type=int, required=True)
    r.add_argument("--spec", action="store_true")
    r.add_argument("--no-gate", dest="gate", action="store_false")
    s = sub.add_parser("summary")
    s.add_argument("--out-dir", required=True)
    b = sub.add_parser("ab-run")
    b.add_argument("--url", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--label", required=True)
    b.add_argument("--out", required=True)
    m = sub.add_parser("ab-compare")
    m.add_argument("--on", required=True)
    m.add_argument("--off", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"health": cmd_health, "run": cmd_run, "summary": cmd_summary,
            "ab-run": cmd_ab_run, "ab-compare": cmd_ab_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
