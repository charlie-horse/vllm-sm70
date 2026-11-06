#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TP2 all-reduce + add + RMSNorm (K17) latency per call inside a CUDA graph.

At decode batch 1 Flint runs 98 K17 calls per step on a 4 KB FP16 hidden
state, so the op is pure latency (launch, the PCIe P2P flag barriers, the
P2P read, the block reduce). This script spawns two ranks (one V100 each),
builds upstream's CustomAllreduce the way the K17 GPU test and the server
do, captures CALLS back-to-back calls in one CUDA graph (a decode step's
worth), replays it, and reports microseconds per call for hidden 2048 FP16
at each batch:

  noop       an empty kernel per call (torch.cuda._sleep(0)): the graph
             launch floor
  nccl       NCCL all_reduce of the same [B, 2048] tensor (torch.distributed
             NCCL group, captured)
  custom_ar  upstream's one-stage custom all-reduce alone (captured in the
             communicator's capture context)
  unfused    custom all-reduce + vLLM's _C.fused_add_rms_norm (what the model
             runs with the fusion off)
  fused      _sm70_C tp2_all_reduce_fused_add_rms_norm (K17), as the model
             calls it

Numbers are timed on each rank with CUDA events around REPLAYS graph
replays after a CPU barrier; the median of TRIALS is reported per rank and
the table shows both ranks. Values in the replayed buffers drift (in-place
NCCL and residual updates) and may reach inf; that does not change timing.

Needs two SM70 GPUs with P2P, vLLM's _C and _sm70_C built with csrc/custom_ar.
On c4130 stop the vllm service first (it holds both GPUs).

  python sm70/tools/bench_tp2_ar_latency.py [--batches 1 2 4] [--calls 98]
      [--replays 20] [--trials 7] [--routes noop nccl custom_ar unfused fused]
      [--timeout 600] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

HIDDEN = 2048
EPS = 1e-6
WORLD = 2
ROUTES = ("noop", "nccl", "custom_ar", "unfused", "fused")
DEFAULT_BATCHES = (1, 2, 4)
DEFAULT_CALLS = 98  # K17 calls per Flint decode step (2 per layer)
MAX_SIZE = 8 * 1024 * 1024  # CustomAllreduce default eager buffer


@dataclass
class Sample:
    """One rank's timing of one (route, batch) graph."""

    route: str
    batch: int
    rank: int
    calls: int
    replays: int
    us_per_call: list[float]  # one value per trial
    error: str | None = None


@dataclass
class Row:
    route: str
    batch: int
    rank0_us: float | None
    rank1_us: float | None
    min_us: float | None  # best trial over both ranks
    step_us: float | None  # max-rank median x calls: one decode step's worth
    error: str | None = None


def per_call_us(elapsed_ms: float, replays: int, calls: int) -> float:
    """Event time for `replays` replays of a `calls`-call graph -> us/call."""
    if replays <= 0 or calls <= 0:
        raise ValueError("replays and calls must be positive")
    return elapsed_ms * 1000.0 / (replays * calls)


def summarize(samples: list[Sample]) -> list[Row]:
    """Merge both ranks' samples into one row per (route, batch).

    Rows keep ROUTES order, then batch order. A route that failed on any
    rank reports the error and no numbers.
    """
    keyed: dict[tuple[str, int], dict[int, Sample]] = {}
    for s in samples:
        keyed.setdefault((s.route, s.batch), {})[s.rank] = s
    order = {r: i for i, r in enumerate(ROUTES)}
    rows = []
    for (route, batch), by_rank in sorted(
        keyed.items(), key=lambda kv: (order.get(kv[0][0], len(order)), kv[0][1])
    ):
        errors = [s.error for s in by_rank.values() if s.error]
        missing = [r for r in range(WORLD) if r not in by_rank]
        if errors or missing:
            err = errors[0] if errors else f"no result from rank(s) {missing}"
            rows.append(Row(route, batch, None, None, None, None, err))
            continue
        med = {r: statistics.median(by_rank[r].us_per_call) for r in range(WORLD)}
        best = min(min(by_rank[r].us_per_call) for r in range(WORLD))
        calls = by_rank[0].calls
        rows.append(Row(route, batch, med[0], med[1], best, max(med.values()) * calls))
    return rows


def format_table(rows: list[Row], calls: int) -> str:
    def f(v):
        return f"{v:9.2f}" if v is not None else f"{'-':>9}"

    hdr = (
        f"{'route':10} {'B':>2} {'rank0':>9} {'rank1':>9} {'best':>9} "
        f"{'x' + str(calls):>9}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        line = (
            f"{r.route:10} {r.batch:>2} {f(r.rank0_us)} {f(r.rank1_us)} "
            f"{f(r.min_us)} {f(r.step_us)}"
        )
        if r.error:
            line += f"  ERROR: {r.error}"
        lines.append(line)
    lines.append(
        f"us per call (median of trials per rank; best = fastest trial); "
        f"x{calls} = us per {calls}-call graph (one decode step's collectives)"
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    for b in args.batches:
        if not 1 <= b <= 16:
            ap.error(f"batch {b}: the fused op admits 1-16 rows")
    if args.calls < 1 or args.replays < 1 or args.trials < 1:
        ap.error("--calls, --replays and --trials must be positive")
    return args


# ---------------------------------------------------------------------------
# Per-rank worker (spawned; imports torch/vLLM only here)
# ---------------------------------------------------------------------------


def _worker(rank: int, rendezvous: str, outdir: str, cfg: dict) -> None:
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=WORLD,
        init_method=rendezvous,
        timeout=timedelta(seconds=cfg["timeout"]),
    )
    samples: list[Sample] = []
    ca = None
    try:
        import vllm._custom_ops  # noqa: F401  (loads vLLM's _C)
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm_sm70 import ar_rms_fusion

        routes = list(cfg["routes"])
        if "fused" in routes and not ar_rms_fusion.kernel_available():
            raise RuntimeError(
                "_sm70_C lacks tp2_all_reduce_fused_add_rms_norm (not loaded, "
                "or built without VLLM_SM70_VLLM_CSRC_DIR)"
            )
        alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "") + os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        )
        # As CudaCommunicator: a pool whose buffers cannot be IPC-exported
        # makes the stock collective copy under capture.
        expandable = "expandable_segments:True" in alloc_conf
        ca = CustomAllreduce(
            dist.group.WORLD,
            torch.device("cuda", rank),
            max_size=MAX_SIZE,
            register_graph_buffers=not expandable,
        )
        if ca.disabled:
            raise RuntimeError("CustomAllreduce is disabled (no P2P / no _C?)")
        nccl_group = dist.new_group(backend="nccl") if "nccl" in routes else None
        kernel = ar_rms_fusion._kernel() if "fused" in routes else None
        for batch in cfg["batches"]:
            for route in routes:
                samples.append(
                    _time_route(
                        torch, dist, ca, nccl_group, kernel, route, batch, rank, cfg
                    )
                )
                torch.accelerator.synchronize()
                dist.barrier()
    finally:
        with open(os.path.join(outdir, f"rank{rank}.json"), "w") as fh:
            json.dump([asdict(s) for s in samples], fh)
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def _time_route(torch, dist, ca, nccl_group, kernel, route, batch, rank, cfg):
    calls, replays = cfg["calls"], cfg["replays"]
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    x = torch.randn(batch, HIDDEN, generator=gen, device="cuda").half()
    gen = torch.Generator(device="cuda").manual_seed(99)
    residual = torch.randn(batch, HIDDEN, generator=gen, device="cuda").half()
    weight = (1 + 0.1 * torch.randn(HIDDEN, generator=gen, device="cuda")).half()
    out = torch.empty_like(x)
    res_out = torch.empty_like(x)
    nccl_buf = x.clone()
    res_inplace = residual.clone()

    def call():
        if route == "noop":
            torch.cuda._sleep(0)
        elif route == "nccl":
            dist.all_reduce(nccl_buf, group=nccl_group)
        elif route == "custom_ar":
            ca.custom_all_reduce(x)
        elif route == "unfused":
            y = ca.custom_all_reduce(x)
            torch.ops._C.fused_add_rms_norm(y, res_inplace, weight, EPS)
        else:  # fused, as vllm_sm70.ar_rms_fusion calls it
            kernel(
                ca._ptr,
                x,
                None,
                residual,
                weight,
                out,
                res_out,
                ca.buffer_ptrs[ca.rank],
                ca.max_size,
                EPS,
                1.0,
            )

    uses_ca = route in ("custom_ar", "unfused")
    error = None
    graph = None
    try:
        # Eager warmup (initializes NCCL communicators, loads modules).
        for _ in range(3):
            call()
        torch.accelerator.synchronize()
        dist.barrier()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            if uses_ca:
                with ca.capture():  # capture-time warmup runs eagerly
                    call()
            else:
                call()
        torch.cuda.current_stream().wait_stream(side)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        if uses_ca:
            # register_graph_buffers runs at context exit (collective on the
            # gloo group), so both ranks always leave it together.
            with ca.capture(), torch.cuda.graph(graph):
                for _ in range(calls):
                    call()
        else:
            with torch.cuda.graph(graph):
                for _ in range(calls):
                    call()
        torch.accelerator.synchronize()
    except Exception as exc:  # noqa: BLE001 - reported per route
        error = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
        graph = None

    # Both ranks must replay a cross-GPU graph, or one spins forever: agree
    # on success first.
    oks = [None] * WORLD
    dist.all_gather_object(oks, error)
    peer_error = next((e for e in oks if e), None)
    if peer_error is not None:
        return Sample(route, batch, rank, calls, replays, [], peer_error)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        graph.replay()
    torch.accelerator.synchronize()
    values = []
    for _ in range(cfg["trials"]):
        dist.barrier()
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(per_call_us(start.elapsed_time(end), replays, calls))
    del graph
    return Sample(route, batch, rank, calls, replays, values)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> list[Row]:
    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        raise SystemExit("needs two visible CUDA GPUs")
    caps = {torch.cuda.get_device_capability(i) for i in range(WORLD)}
    if caps != {(7, 0)}:
        print(f"warning: device capabilities {caps}, not SM70", file=sys.stderr)
    cfg = {
        "batches": list(args.batches),
        "calls": args.calls,
        "replays": args.replays,
        "trials": args.trials,
        "routes": list(args.routes),
        "timeout": max(60.0, args.timeout),
    }
    with tempfile.TemporaryDirectory(prefix="tp2-ar-bench-") as tmp:
        rendezvous = "file://" + os.path.join(tmp, "rendezvous")
        # spawn (never fork after CUDA init) and a bounded wait, so a stuck
        # rank (one side spinning on a P2P flag) ends the run.
        ctx = mp.spawn(_worker, args=(rendezvous, tmp, cfg), nprocs=WORLD, join=False)
        deadline = time.monotonic() + args.timeout
        while not ctx.join(timeout=5):
            if time.monotonic() > deadline:
                for proc in ctx.processes:
                    if proc.is_alive():
                        proc.kill()
                raise SystemExit(f"ranks did not finish within {args.timeout:.0f} s")
        samples = []
        for r in range(WORLD):
            path = os.path.join(tmp, f"rank{r}.json")
            if os.path.exists(path):
                with open(path) as fh:
                    samples += [Sample(**s) for s in json.load(fh)]
    return summarize(samples)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run(args)
    print(format_table(rows, args.calls))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in rows], fh, indent=1)
    return 1 if any(r.error for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
