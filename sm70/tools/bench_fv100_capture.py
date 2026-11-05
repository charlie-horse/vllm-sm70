#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cost of the captured-decode envelope (Quinn's #39 note), K07.

Under CUDA-graph capture the Flash-V100 decode launches enough partitions
for max_model_len (vllm_sm70/attention/flash_v100/dispatch.py), because a
graph replays for any context. This script measures, per model shape,
max_model_len and live context, the decode time per step for:

  eager       uncaptured, grid sized for the live context (what PIECEWISE
              graphs and eager mode run)
  graph-env   captured with the max_model_len envelope (what FULL graphs
              run today)
  graph-ctx   captured with a grid sized for the live context (what a
              context-bucketed capture would run)

and the decode workspace (tmp_out + max_logits + exp_sums) for each sizing.
One attention layer, one decode step; multiply by the layer count for a
model step. Needs an SM70 GPU and _sm70_C with the K05 ops.

  python sm70/tools/bench_fv100_capture.py [--batch 1] [--iters 200] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

SHAPES = {
    # name: (q heads, kv heads, head dim, window, sinks, max_model_lens)
    "flint-tp2": (16, 2, 128, (512, 0), True, (32768, 131072)),
    "qwen38-tp2-gqa6": (12, 2, 256, (-1, -1), False, (32768,)),
    "qwen36-tp2-gqa8": (8, 1, 256, (-1, -1), False, (32768,)),
}
CONTEXTS = (1024, 4096, 16384)


@dataclass
class Row:
    shape: str
    max_model_len: int
    context: int
    batch: int
    route: str
    eager_us: float
    graph_env_us: float
    graph_ctx_us: float
    workspace_env_mib: float
    workspace_ctx_mib: float


def _workspace_mib(fv100) -> float:
    total = 0
    for ws in fv100._decode_workspace_cache.values():
        for t in (ws.tmp_out, ws.max_logits, ws.exp_sums):
            total += t.numel() * t.element_size()
    return total / 2**20


def _time(fn, iters: int) -> float:
    import torch

    for _ in range(3):
        fn()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run(
    batch: int = 1, iters: int = 200, shapes=SHAPES, contexts=CONTEXTS
) -> list[Row]:
    import torch

    from vllm_sm70.attention.flash_v100 import dispatch
    from vllm_sm70.ops import fv100

    if not fv100.available():
        raise RuntimeError("_sm70_C has no K05 Flash-V100 ops")
    device = torch.device("cuda")
    rows: list[Row] = []
    for name, (hq, hkv, d, window, has_sinks, mmls) in shapes.items():
        for mml in mmls:
            block = 16
            width = -(-mml // block)
            pool = min(width * batch, 4096)
            k = torch.randn(pool, block, hkv, d, device=device, dtype=torch.float16)
            v = torch.randn_like(k)
            table = torch.randint(
                0, pool, (batch, width), device=device, dtype=torch.int32
            )
            q = torch.randn(batch, hq, d, device=device, dtype=torch.float16)
            out = torch.empty_like(q)
            sinks = torch.randn(hq, device=device) if has_sinks else None
            xqa = dispatch.use_xqa(q, k, window, sinks)
            route = "xqa" if xqa else ("sinks" if has_sinks else "scalar")
            for ctx in contexts:
                if ctx > mml:
                    continue
                lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)

                def call(hint, capacity):
                    kw = dict(
                        out=out,
                        max_seq_len_hint=hint,
                        workspace_seq_capacity_hint=capacity,
                    )
                    if xqa:
                        fv100.flash_attn_decode_paged_xqa(q, k, v, table, lens, **kw)
                    else:
                        fv100.flash_attn_decode_paged(
                            q, k, v, table, lens, window_size=window, sinks=sinks, **kw
                        )

                def graph_of(hint):
                    fv100.clear_workspaces()
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        call(hint, hint)
                    torch.cuda.current_stream().wait_stream(side)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        call(hint, hint)
                    return g, _workspace_mib(fv100)

                fv100.clear_workspaces()
                eager = _time(lambda: call(ctx, None), iters)
                g_env, ws_env = graph_of(mml)
                env = _time(g_env.replay, iters)
                g_ctx, ws_ctx = graph_of(ctx)
                tight = _time(g_ctx.replay, iters)
                rows.append(
                    Row(name, mml, ctx, batch, route, eager, env, tight, ws_env, ws_ctx)
                )
                del g_env, g_ctx
            fv100.clear_workspaces()
            torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    rows = run(args.batch, args.iters)
    hdr = (
        f"{'shape':18} {'mml':>7} {'ctx':>6} {'B':>2} {'route':6} {'eager':>8} "
        f"{'g-env':>8} {'g-ctx':>8} {'ws-env':>8} {'ws-ctx':>8}"
    )
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(
            f"{r.shape:18} {r.max_model_len:>7} {r.context:>6} {r.batch:>2} "
            f"{r.route:6} {r.eager_us:8.1f} {r.graph_env_us:8.1f} "
            f"{r.graph_ctx_us:8.1f} {r.workspace_env_mib:8.2f} "
            f"{r.workspace_ctx_mib:8.2f}"
        )
    print("times: us per layer-step; workspace: MiB per (device, stream, shape)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
