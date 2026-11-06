#!/usr/bin/env python3
"""Teacher-forced logprob comparison for Flint (M5 T4a/T7), runnable on
either stack: this repo's vllm-sm70 build or the fork (cf9492a). It imports
only vllm, so it calibrates the fork's own envelope with the same code.

Why teacher forcing: Flint's greedy output flips on last-bit differences
(MoE top-6/256 routing), so free-running token equality between two valid
numerical paths is not a correctness signal. Scoring FIXED token sequences
gives each path identical inputs at every position; the per-token |dlogprob|
is then the path's own numeric difference, with no cascade.

Subcommands (each LLM build runs in this process; TP workers spawn, so all
work happens under the __main__ guard):
  gen      --model M --out seqs.json [--n 64]        eager greedy continuations
  score    --model M --seqs seqs.json --out s.json --mode eager|graphs
           [--env K=V ...] [--llm-kwargs JSON]       per-token logprobs
  compare  A.json B.json [--out stats.json]          mean/p99/max |dlogprob|
  envelope --pair NAME A.json B.json --out env.json  add a pair's stats to an
           envelope file (run on the fork to calibrate T4a/T7)

Common LLM args match the M5 tests: TP2, fp16, max_model_len 8192,
max_num_seqs 4. Pass --hf-overrides for this repo's FlintSm70ForCausalLM;
omit it on the fork (its FlintForCausalLM is the default architecture).
"""
import argparse
import json
import math
import os

PROMPTS = [
    "def quicksort(xs):",
    "The capital of France is",
    "Explain the CAP theorem in one sentence.",
    "1, 1, 2, 3, 5, 8,",
    "SELECT name FROM users WHERE",
    "Translate to German: good morning",
    "Write a haiku about V100 GPUs.",
    "The derivative of x**3 is",
]


def _llm(a, **extra):
    from vllm import LLM

    kw = dict(
        model=a.model, tensor_parallel_size=a.tp, dtype="half",
        max_model_len=8192, max_num_seqs=4,
    )
    if a.hf_overrides:
        kw["hf_overrides"] = json.loads(a.hf_overrides)
    kw.update(extra)
    return LLM(**kw)


def _close(llm):
    import gc

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 - best effort; memory is checked by the caller
        pass
    del llm
    gc.collect()


def cmd_gen(a):
    from vllm import SamplingParams

    llm = _llm(a, enforce_eager=True)
    try:
        tok = llm.get_tokenizer()
        outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=a.n))
        seqs = []
        for p, o in zip(PROMPTS, outs):
            ids = tok(p)["input_ids"]
            seqs.append({"prompt": p, "start": len(ids),
                         "ids": list(ids) + list(o.outputs[0].token_ids)})
    finally:
        _close(llm)
    json.dump({"seqs": seqs}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(seqs)} sequences")


def cmd_score(a):
    from vllm import SamplingParams

    for kv in a.env or []:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    extra = json.loads(a.llm_kwargs) if a.llm_kwargs else {}
    if a.mode == "eager":
        extra["enforce_eager"] = True
    seqs = json.load(open(a.seqs))["seqs"]
    llm = _llm(a, **extra)
    try:
        outs = llm.generate(
            [{"prompt_token_ids": s["ids"]} for s in seqs],
            SamplingParams(max_tokens=1, prompt_logprobs=1),
        )
        scores = [
            [o.prompt_logprobs[i][s["ids"][i]].logprob for i in range(s["start"], len(s["ids"]))]
            for s, o in zip(seqs, outs)
        ]
    finally:
        _close(llm)
    json.dump({"mode": a.mode, "env": a.env or [], "llm_kwargs": extra,
               "scores": scores}, open(a.out, "w"))
    print(f"wrote {a.out}: {sum(map(len, scores))} scored tokens")


def stats(sa, sb):
    d = sorted(abs(x - y) for xa, xb in zip(sa, sb) for x, y in zip(xa, xb))
    if not d:
        raise SystemExit("no scored tokens to compare")
    p99 = d[min(len(d) - 1, math.ceil(0.99 * len(d)) - 1)]
    return {"n": len(d), "mean": sum(d) / len(d), "p99": p99, "max": d[-1],
            "over_0_1": sum(x > 0.1 for x in d)}


def cmd_compare(a):
    s = stats(json.load(open(a.a))["scores"], json.load(open(a.b))["scores"])
    print(json.dumps(s))
    if a.out:
        json.dump(s, open(a.out, "w"))


def cmd_envelope(a):
    env = json.load(open(a.out)) if os.path.exists(a.out) else {}
    env[a.pair] = stats(json.load(open(a.a))["scores"], json.load(open(a.b))["scores"])
    json.dump(env, open(a.out, "w"), indent=1)
    print(f"{a.pair}: {json.dumps(env[a.pair])} -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "score"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--tp", type=int, default=2)
        p.add_argument("--hf-overrides", default=None)
    sub.choices["gen"].add_argument("--out", required=True)
    sub.choices["gen"].add_argument("--n", type=int, default=64)
    sp = sub.choices["score"]
    sp.add_argument("--seqs", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--mode", choices=["eager", "graphs"], required=True)
    sp.add_argument("--env", action="append", help="K=V, repeatable")
    sp.add_argument("--llm-kwargs", default=None, help="JSON merged into LLM(...)")
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--out")
    ep = sub.add_parser("envelope")
    ep.add_argument("--pair", required=True)
    ep.add_argument("a")
    ep.add_argument("b")
    ep.add_argument("--out", required=True)
    a = ap.parse_args()
    {"gen": cmd_gen, "score": cmd_score, "compare": cmd_compare,
     "envelope": cmd_envelope}[a.cmd](a)


if __name__ == "__main__":
    main()
