#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch-config sweep of the Inkling 2 weight-streaming GEMVs on one V100.

Times, under CUDA-graph replay with the weights rotated through copies
larger than L2 (cold weights, as in a decode step):

  dense   ops/small_m_gemv.py ``segment_gemv`` for each (BLOCK_N, BLOCK_K,
          num_warps) candidate against cuBLAS F.linear, at the Inkling 2
          Mini TP2 shapes: qkvr (one input, and three inputs as a gated
          layer), wo, the LM head shard; 1, 2, 4 and 8 rows
  nvfp4   ops/nvfp4_gemv.py for each (BLOCK_K8, num_warps) candidate
          against the TurboMind compact stage (+ the SwiGLU kernel for W13),
          8, 16, 32 and 64 one-row slots of distinct random experts

and prints µs, GB/s and % of 900 GB/s per row, then the best candidate per
row bucket and stage: the entries for small_m_gemv.CONFIGS and
nvfp4_gemv.CONFIGS. Needs an SM70 GPU; the nvfp4 part needs _sm70_C.

  python sm70/tools/bench_inkling2_gemv.py [--part dense|nvfp4|all] [--iters 50]
"""

# ruff: noqa: B023 (each closure is timed before its loop variables move on)
from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn.functional as F

PEAK = 900e9
HIDDEN, LOCAL, EXPERTS = 2304, 576, 128
QKVR = (1536, 192, 1536, 12, 512, 512)
DENSE = {  # name: (rows, K, inputs)
    "qkvr_local": (sum(QKVR), HIDDEN, 1),
    "qkvr_gated": (sum(QKVR), HIDDEN, 3),
    "wo": (HIDDEN, 1536, 1),
    "lm_head": (100512, HIDDEN, 1),  # vocab 201024 over TP2
}
DENSE_CANDIDATES = [
    (n, k, w)
    for n, k, w in itertools.product((8, 16, 32), (32, 64, 128, 256), (2, 4, 8))
    if n * k <= 4096
]
NVFP4_CANDIDATES = list(itertools.product((4, 8, 16, 32), (1, 2, 4, 8)))
COLD_BYTES = 64 << 20


def timed(fn, iters):
    """µs per call of `fn(i)` replayed in one CUDA graph of `iters` calls."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for i in range(2):
            fn(i)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(iters):
                fn(i)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def report(name, rows, nbytes, results):
    best = min(results, key=lambda r: r[1])
    for label, us in sorted(results, key=lambda r: r[1])[:4]:
        print(
            f"{name:12s} m={rows:2d} {label:28s} {us:8.1f} us "
            f"{nbytes / us / 1e3:7.1f} GB/s {100 * nbytes / us / 1e-6 / PEAK:5.1f}%"
        )
    return best


def bench_dense(iters):
    from vllm_sm70.models.inkling2_sm70.ops import small_m_gemv as g

    best = {}
    for name, (n, k, inputs) in DENSE.items():
        nbytes = n * k * 2
        copies = max(2, -(-COLD_BYTES // nbytes))
        weights = [
            torch.randn(n, k, device="cuda").half() * 0.02 for _ in range(copies)
        ]
        query = sum(QKVR[:4])
        bounds = (query, query + QKVR[4]) if inputs == 3 else ()
        for rows in (1, 2, 4, 8):
            xs = tuple(
                torch.randn(rows, k, device="cuda").half() for _ in range(inputs)
            )
            out = xs[0].new_empty(rows, n)
            results = []
            if inputs == 1:
                results.append(
                    (
                        "cublas",
                        timed(lambda i: F.linear(xs[0], weights[i % copies]), iters),
                    )
                )
            else:
                w_q, w_k, w_v = zip(*(w.split((query, *QKVR[4:])) for w in weights))

                def cublas(i):
                    c = i % copies
                    F.linear(xs[0], w_q[c])
                    F.linear(xs[1], w_k[c])
                    F.linear(xs[2], w_v[c])

                results.append(("cublas x3", timed(cublas, iters)))
            block_m = triton_block(rows)
            saved = g.CONFIGS[block_m]
            for cand in DENSE_CANDIDATES:
                if block_m * cand[0] * cand[1] > 8192:
                    continue
                g.CONFIGS[block_m] = cand
                us = timed(
                    lambda i: g.segment_gemv(xs, weights[i % copies], bounds, out),
                    iters,
                )
                results.append((f"gemv n{cand[0]} k{cand[1]} w{cand[2]}", us))
            g.CONFIGS[block_m] = saved
            best[(name, rows)] = report(name, rows, nbytes, results)
    return best


def triton_block(rows):
    return 1 << (rows - 1).bit_length()


def bench_nvfp4(iters):
    from vllm_sm70.models.inkling2_sm70.ops import nvfp4_gemv as nv
    from vllm_sm70.models.inkling2_sm70.ops.activation import softcapped_swiglu_out
    from vllm_sm70.ops import tm

    def bank(k, n):
        codes = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8)
        scales = (torch.rand(k // 16, n, device="cuda") * 0.01 + 1e-3).half()
        w, s, _ = tm.nvfp4_sm70_prepare(codes, scales)
        words = w.unsqueeze(0).repeat(EXPERTS, 1, 1).contiguous()
        prepared = s.unsqueeze(0).repeat(EXPERTS, 1, 1).contiguous()
        ptrs = tm.awq_moe_build_strided_ptrs(words, prepared, 32 * k, n, EXPERTS)
        return words, prepared, ptrs

    best = {}
    stages = {"w13": (HIDDEN, 2 * LOCAL), "w2": (LOCAL, HIDDEN)}
    for stage, (k, n) in stages.items():
        words, scales, ptrs = bank(k, n)
        per_expert = k * n // 2 + k // 16 * n * 2
        width = n // 2 if stage == "w13" else n
        caps = (4.0, 25.0) if stage == "w13" else None
        for slots in (8, 16, 32, 64):
            rows = torch.randn(slots, k, device="cuda").half()
            sets = [
                torch.randperm(EXPERTS, device="cuda")[:slots].sort().values.int()
                for _ in range(16)
            ]
            offsets = torch.arange(slots + 1, device="cuda", dtype=torch.int32)
            stage_out = rows.new_empty(slots, n)
            act = rows.new_empty(slots, width)

            def turbomind(i):
                tm.nvfp4_moe_dense_stage_sm70_out(
                    stage_out, rows, offsets, sets[i % 16], *ptrs, slots, k, n, 16
                )
                if caps:
                    softcapped_swiglu_out(stage_out, act, *caps)

            results = [("turbomind", timed(turbomind, iters))]
            saved = nv.CONFIGS[stage]
            for cand in NVFP4_CANDIDATES:
                nv.CONFIGS[stage] = cand
                us = timed(
                    lambda i: nv.nvfp4_slot_gemv(
                        rows, words, scales, sets[i % 16], act, caps
                    ),
                    iters,
                )
                results.append((f"gemv k8 {cand[0]} w{cand[1]}", us))
            nv.CONFIGS[stage] = saved
            best[(stage, slots)] = report(stage, slots, slots * per_expert, results)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("dense", "nvfp4", "all"), default="all")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0), "needs an SM70 GPU"
    best = {}
    with torch.inference_mode():
        if args.part in ("dense", "all"):
            best |= bench_dense(args.iters)
        if args.part in ("nvfp4", "all"):
            best |= bench_nvfp4(args.iters)
    print("\nbest per row:")
    for key, (label, us) in best.items():
        print(f"  {key}: {label} {us:.1f} us")


if __name__ == "__main__":
    main()
