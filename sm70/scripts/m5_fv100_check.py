#!/usr/bin/env python3
"""Flint attention check: Flash-V100 (sinks) against TRITON_ATTN, batched.

Kai's GPU check 3 for the M5 Flash-V100 default (Nov 6): greedy outputs on
the M5 prompts at batch sizes 2/3/4/8/20, every batch run twice, on the
default backend (Flash-V100) and with VLLM_SM70_FLASH_V100=0 (TRITON_ATTN).

  gen      --model M [--hf-overrides J] --backend fv100|triton --out F.json
           one engine (graphs mode, TP2, max_num_seqs 20); per batch size N,
           N requests cycling through the prompts, run twice.
  compare  FV100.json TRITON.json

What counts. Flint's greedy tokens flip on last-bit differences (MoE top-6
of 256 routing), so two valid backends may diverge after a shared prefix.
compare prints each request's common prefix for eyeballing and FAILs only on:
  * a run-to-run difference inside one engine (same batch, same backend);
  * a degenerate Flash-V100 output (empty, or the last 32 tokens use fewer
    than 4 distinct ids) where Triton's output for that request is not;
  * a Flash-V100 output that diverges from Triton at token 0.
Read the printed texts as well: gibberish that is not repetitive is a FAIL.

Tuning is off (as in every M5 tool) unless M5_ALLOW_TUNING=1.
"""
import argparse
import json
import os
import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import m5_teacher_forced as tf  # vllm-free at import

BATCH_SIZES = (2, 3, 4, 8, 20)
RUNS = 2
MAX_TOKENS = 64
REPEAT_WINDOW = 32
REPEAT_MIN_DISTINCT = 4


def _degenerate(ids: list[int]) -> bool:
    if not ids:
        return True
    tail = ids[-REPEAT_WINDOW:]
    return len(tail) >= REPEAT_WINDOW and len(set(tail)) < REPEAT_MIN_DISTINCT


def cmd_gen(a) -> None:
    if a.backend == "triton":
        os.environ["VLLM_SM70_FLASH_V100"] = "0"
    else:
        os.environ.pop("VLLM_SM70_FLASH_V100", None)
    print(f"backend: {a.backend} VLLM_SM70_FLASH_V100="
          f"{os.environ.get('VLLM_SM70_FLASH_V100', '<unset>')}", flush=True)
    from vllm import LLM, SamplingParams

    kw = {
        "model": a.model, "tensor_parallel_size": 2, "dtype": "half",
        "max_model_len": 8192, "max_num_seqs": max(BATCH_SIZES),
        "gpu_memory_utilization": a.gpu_util, "enable_prefix_caching": False,
    }
    if a.hf_overrides:
        kw["hf_overrides"] = json.loads(a.hf_overrides)
    llm = LLM(**kw)
    prompts = tf.PROMPT_SETS[0]
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    out = {"backend": a.backend, "prompts": prompts, "batches": {}}
    try:
        for n in BATCH_SIZES:
            batch = [prompts[i % len(prompts)] for i in range(n)]
            runs = []
            for _ in range(RUNS):
                res = llm.generate(batch, sp, use_tqdm=False)
                runs.append([
                    {"prompt": p, "token_ids": list(r.outputs[0].token_ids),
                     "text": r.outputs[0].text}
                    for p, r in zip(batch, res)
                ])
            out["batches"][str(n)] = runs
            print(f"batch {n}: done", flush=True)
    finally:
        tf._close(llm)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"-> {a.out}")


def _common(x: list[int], y: list[int]) -> int:
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                min(len(x), len(y)))


def cmd_compare(a) -> int:
    fv = json.loads(pathlib.Path(a.fv100).read_text())
    tr = json.loads(pathlib.Path(a.triton).read_text())
    fails = []
    for n in map(str, BATCH_SIZES):
        for name, data in (("fv100", fv), ("triton", tr)):
            r0, r1 = data["batches"][n][0], data["batches"][n][1]
            for i, (x, y) in enumerate(zip(r0, r1)):
                if x["token_ids"] != y["token_ids"]:
                    fails.append(f"{name} batch {n} req {i}: run 1 != run 2 "
                                 f"at token {_common(x['token_ids'], y['token_ids'])}")
        for i, (f, t) in enumerate(zip(fv["batches"][n][0], tr["batches"][n][0])):
            fi, ti = f["token_ids"], t["token_ids"]
            c = _common(fi, ti)
            flag = ""
            if _degenerate(fi) and not _degenerate(ti):
                flag = "  DEGENERATE"
                fails.append(f"batch {n} req {i}: Flash-V100 output degenerate")
            if c == 0 and fi != ti:
                flag += "  FIRST-TOKEN"
                fails.append(f"batch {n} req {i}: diverges at token 0")
            print(f"batch {n:>2} req {i:>2} {f['prompt'][:32]!r:36} "
                  f"common {c:>2}/{MAX_TOKENS}{flag}")
            if a.show or flag:
                print(f"    fv100 : {f['text'][:160]!r}")
                print(f"    triton: {t['text'][:160]!r}")
    for line in fails:
        print("FAIL " + line)
    print("FV100 CHECK " + ("PASS" if not fails else f"FAIL ({len(fails)})")
          + " (also read the texts for gibberish: --show)")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--hf-overrides")
    g.add_argument("--backend", choices=("fv100", "triton"), required=True)
    g.add_argument("--gpu-util", type=float, default=0.85)
    g.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("fv100")
    c.add_argument("triton")
    c.add_argument("--show", action="store_true", help="print every text pair")
    a = p.parse_args(argv)
    if a.cmd == "gen":
        tf._deterministic_kernels()
        cmd_gen(a)
        return 0
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
