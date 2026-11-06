#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Flash-V100 decode microbenchmark at Flint's decode shapes (K14 vs two-launch).

Flint per TP2 rank: 16 q heads, 2 KV heads, D128, fp16 KV, fp32 learned
sinks; sliding layers use window 513 (counts the current token, so the
kernels get window_size_left = 512) and full layers no window. KV block
size 16 (the Flint serve passes no --block-size), max_model_len 262144
(the serve default), which sets the captured workspace envelope.

Every mode is timed inside CUDA graphs with CUDA events:

  g1   one decode call per graph; us per replay (includes the graph launch)
  gN   --calls decode calls per graph (default 48, one Flint decode step per
       rank), each on its own KV pool (--pools rotate, so L2 stays cold as in
       a real step); us per call

Modes:

  dispatch     vllm_sm70.attention.flash_v100.dispatch.run_decode exactly as
               the captured serve calls it (dispatch.py:87-115): envelope
               hints, the K14 one-launch kernel, and the per-call
               active_num_partitions.fill_ the wrapper records
               (ops/fv100.py:485-487)
  dispatch-2l  same with VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0: grouped
               partition kernel over the whole envelope plus the reducer
  raw          the fused op alone, workspace preallocated, no fill_:
               dispatch - raw is the wrapper's captured overhead per call
  raw-zN       (--z-caps) the fused op with N partial slots, which caps the
               CTAs per (row, KV head) at N (gqa8_d128_fused_ctas_per_head)

Columns Z / act / tok are the launch model of flash_decode_paged.cu
(gqa8_d128_fused_ctas_per_head and the kernel's chunk split): CTAs per
(row, KV head), CTAs that do work, tokens per working CTA. `diff` is the
max |out - two-launch eager reference| of the 1-call graph.

  python sm70/tools/bench_fv100_decode.py
  python sm70/tools/bench_fv100_decode.py --z-caps 8,16,32 --layer-split 36,12
  python sm70/tools/bench_fv100_decode.py --config /path/to/checkpoint/config.json

--layer-split SWA,FULL (or --config, which reads layer_types and
sliding_window) adds a per-step estimate: SWA x swa_us + FULL x full_us.

The fork's kernel is byte-identical (sm70/csrc/flash_v100/PROVENANCE.md:
the fused kernel, ctas_per_head and launcher rows), so this measures it too.
To time the fork build anyway: in the fork checkout, with CUDA 12.8 and the
serve's torch, `cd flash-attention-v100 && TORCH_CUDA_ARCH_LIST=7.0 pip
install --no-build-isolation .` builds flash_attn_v100_cuda (setup.py, same
nvcc flags as sources.cmake); its flash_attn_interface.flash_attn_decode_paged
takes the same arguments plus active_num_partitions, which the fork's
metadata passes from a per-step buffer (fork metadata.py:359-404).

Needs an SM70 GPU and _sm70_C with the K05 ops (K14 for the fused modes).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field

# Flint per TP2 rank (fork vllm/models/flint/configs.py: 32 q / 4 kv heads,
# D128, sliding_window 513; divided by TP2).
HEADS_Q = 16
HEADS_KV = 2
HEAD_DIM = 128
SLIDING_WINDOW = 513  # counts the current token
BLOCK_SIZE = 16
MAX_MODEL_LEN = 262144
BATCHES = (1, 4)
CONTEXTS = (2048, 4096, 32768)
LAYERS = ("swa", "full")

# flash_decode_paged.cu:2083-2085, :40 and __launch_bounds__(kThreadsPerBlock, 2)
FUSED_MAX_CHUNK = 256
FUSED_MIN_CHUNK = 64
FUSED_MAX_CTAS_PER_HEAD = 256
FUSED_BLOCKS_PER_SM = 2
# ops/fv100.py DEFAULT_DECODE_PARTITION_SIZE and the >= 32K switch to 1024.
DEFAULT_PARTITION_SIZE = 256
LONG_PARTITION_SIZE = 1024
LONG_PARTITION_THRESHOLD = 32768

FUSED_DECODE_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE"


# ---------------------------------------------------------------------------
# Launch model (CPU, no torch)
# ---------------------------------------------------------------------------


def window_left(layer: str, sliding_window: int = SLIDING_WINDOW) -> int:
    """window_size_left the kernels get: W counts the current token."""
    return sliding_window - 1 if layer == "swa" else -1


def envelope_partition_slots(max_model_len: int) -> tuple[int, int]:
    """(partition size, tmp_out partition slots) of the captured workspace.

    ops/fv100.py _select_default_decode_partition_size and
    _round_decode_partition_capacity for workspace_seq_capacity_hint =
    max_model_len (dispatch.py:87-89).
    """
    seq = max(1, int(max_model_len))
    psize = (
        LONG_PARTITION_SIZE
        if seq >= LONG_PARTITION_THRESHOLD
        else DEFAULT_PARTITION_SIZE
    )
    parts = max(1, -(-seq // psize))
    slots = 1 if parts <= 1 else 1 << (parts - 1).bit_length()
    return psize, slots


def fused_ctas_per_head(
    batch: int,
    heads_kv: int,
    partial_slots: int,
    num_sms: int,
    blocks_per_sm: int = FUSED_BLOCKS_PER_SM,
) -> int:
    """Z, flash_decode_paged.cu gqa8_d128_fused_ctas_per_head."""
    slots = num_sms * max(blocks_per_sm, 1)
    rows = max(1, batch * heads_kv)
    ctas = -(-slots // rows)
    return max(1, min(ctas, partial_slots, FUSED_MAX_CTAS_PER_HEAD))


@dataclass(frozen=True)
class FusedWork:
    lo: int  # first key in the window
    length: int  # keys read per (row, KV head)
    num_chunks: int
    active: int  # CTAs that do work (and partials the last CTA combines)
    max_tokens_per_cta: int


def fused_work(seq_len: int, wleft: int, z: int) -> FusedWork:
    """The fused kernel's split of [lo, seq_len) (flash_decode_paged.cu:2131-2141)."""
    if seq_len <= 0:
        return FusedWork(0, 0, 0, 0, 0)
    lo = max(0, seq_len - 1 - wleft) if wleft >= 0 else 0
    length = seq_len - lo
    span = z * FUSED_MAX_CHUNK
    iters = -(-length // span)
    num_chunks = min(z * iters, -(-length // FUSED_MIN_CHUNK))
    active = min(z, num_chunks)
    chunks_per_cta = -(-num_chunks // active)
    per_chunk = -(-length // num_chunks)
    return FusedWork(lo, length, num_chunks, active, chunks_per_cta * per_chunk)


def live_partitions(seq_len: int, wleft: int, partition_size: int) -> int:
    """Partitions of the two-launch grid that hold in-window keys."""
    if seq_len <= 0:
        return 0
    lo = max(0, seq_len - 1 - wleft) if wleft >= 0 else 0
    return -(-seq_len // partition_size) - lo // partition_size


def layer_split_from_config(cfg: dict) -> tuple[int, int, int | None]:
    """(sliding layers, full layers, sliding_window) of an HF config dict."""
    cfg = cfg.get("text_config", cfg)
    n = int(cfg["num_hidden_layers"])
    types = cfg.get("layer_types") or ["full_attention"] * n
    if len(types) != n:
        raise ValueError(f"layer_types has {len(types)} entries, num_hidden_layers {n}")
    use_sw = cfg.get("use_sliding_window", True)
    window = cfg.get("sliding_window") if use_sw else None
    swa = sum(t == "sliding_attention" for t in types) if window else 0
    return swa, n - swa, window


def parse_layer_split(text: str) -> tuple[int, int]:
    swa, full = (int(x) for x in text.split(","))
    if swa < 0 or full < 0:
        raise ValueError(f"--layer-split needs non-negative counts, got {text!r}")
    return swa, full


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def step_estimate_us(rows: list[Row], swa: int, full: int, column: str = "gn_us"):
    """{(batch, ctx, mode): SWA x swa_us + FULL x full_us} where both exist."""
    by_key = {(r.batch, r.context, r.mode, r.layer): getattr(r, column) for r in rows}
    out = {}
    for (b, c, m, layer), us in by_key.items():
        if layer != "swa" or us is None:
            continue
        other = by_key.get((b, c, m, "full"))
        if other is not None:
            out[(b, c, m)] = swa * us + full * other
    return out


# ---------------------------------------------------------------------------
# GPU bench
# ---------------------------------------------------------------------------


@dataclass
class Row:
    batch: int
    context: int
    layer: str
    mode: str
    z: int
    active: int
    tokens_per_cta: int
    g1_us: float | None
    gn_us: float | None
    diff: float | None
    note: str = field(default="")


@contextlib.contextmanager
def _env(name: str, value: str | None):
    """Set `name` for the block; None leaves the environment as it is."""
    if value is None:
        yield
        return
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _time_replay(graph, iters: int, warmup: int = 5) -> float:
    import torch

    for _ in range(warmup):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _capture(fn):
    """Warm up on a side stream, then capture on it (as vLLM does)."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    return graph


class _Pool:
    """One layer's KV cache in upstream's LBNHC layout plus its block table."""

    def __init__(self, batch: int, ctx: int, max_model_len: int, device, gen):
        import torch
        from vllm_sm70.ops import fv100

        per_row = -(-ctx // BLOCK_SIZE)
        blocks = batch * per_row
        # Physical (blocks, block, H_kv, 2D); logical (blocks, H_kv, block, 2D).
        phys = torch.randn(
            blocks,
            BLOCK_SIZE,
            HEADS_KV,
            2 * HEAD_DIM,
            device=device,
            dtype=torch.float16,
            generator=gen,
        )
        self.k, self.v = fv100.split_lbnhc_kv_cache(phys.transpose(1, 2), HEAD_DIM)
        width = -(-max_model_len // BLOCK_SIZE)
        perm = torch.randperm(blocks, device=device, generator=gen).to(torch.int32)
        self.table = torch.zeros(batch, width, dtype=torch.int32, device=device)
        self.table[:, :per_row] = perm.view(batch, per_row)
        self._phys = phys


@dataclass
class _Case:
    """One (batch, ctx, layer) decode: the inputs every mode shares."""

    batch: int
    ctx: int
    max_model_len: int
    window: tuple[int, int]
    q: object
    sinks: object
    lens: object
    outs: list
    scale: float = HEAD_DIM**-0.5


def _dispatch_call(case: _Case):
    """dispatch.run_decode as the captured serve calls it (dispatch.py:62-115)."""
    import types

    from vllm_sm70.attention.flash_v100 import dispatch

    def call(i: int, pool: _Pool) -> None:
        meta = types.SimpleNamespace(
            decode_rows=case.batch,
            decode_seq_lens=case.lens,
            decode_block_table=pool.table,
            cudagraph_capture=True,
            max_model_len=case.max_model_len,
            max_seq_len=case.ctx,
        )
        dispatch.run_decode(
            case.q,
            pool.k,
            pool.v,
            case.outs[i],
            meta,
            scale=case.scale,
            window=case.window,
            sinks=case.sinks,
        )

    return call


def _raw_call(case: _Case, slots: int, partition_size: int):
    """The fused op alone on a preallocated workspace with `slots` partials."""
    import torch
    from vllm_sm70.ops import fv100

    dev = case.q.device
    b = case.batch
    tmp = torch.empty(b, HEADS_Q, slots, HEAD_DIM, device=dev, dtype=torch.float16)
    ml = torch.empty(b, HEADS_Q, slots, device=dev)
    es = torch.empty_like(ml)
    anp = torch.ones(1, dtype=torch.int32, device=dev)  # unread by the fused kernel
    counters = torch.zeros(b * HEADS_KV, dtype=torch.int32, device=dev)
    op = getattr(torch.ops._sm70_C, fv100.FUSED_DECODE_OP)
    wl, wr = case.window

    def call(i: int, pool: _Pool) -> None:
        op(
            case.outs[i],
            case.q,
            pool.k,
            pool.v,
            pool.table,
            case.lens,
            tmp,
            ml,
            es,
            anp,
            case.scale,
            partition_size,
            1,
            "auto",
            1.0,
            1.0,
            wl,
            wr,
            case.sinks,
            counters,
        )

    return call


def _graphs(call, kv: list[_Pool], calls: int, env_value: str | None):
    """(1-call graph on pool 0, `calls`-call graph rotating over the pools)."""
    with _env(FUSED_DECODE_ENV, env_value):
        g1 = _capture(lambda: call(0, kv[0]))
        gn = _capture(lambda: [call(i, kv[i % len(kv)]) for i in range(calls)])
    return g1, gn


def _reference(case: _Case, pool: _Pool):
    """Two-launch eager decode on `pool`, grid sized for the live context."""
    import torch
    from vllm_sm70.ops import fv100

    ref = torch.empty_like(case.q)
    with _env(FUSED_DECODE_ENV, "0"):
        fv100.flash_attn_decode_paged(
            case.q,
            pool.k,
            pool.v,
            pool.table,
            case.lens,
            softmax_scale=case.scale,
            out=ref,
            window_size=case.window,
            sinks=case.sinks,
            max_seq_len_hint=case.ctx,
        )
    torch.cuda.synchronize()
    return ref


def run(
    batches=BATCHES,
    contexts=CONTEXTS,
    layers=LAYERS,
    *,
    max_model_len: int = MAX_MODEL_LEN,
    calls: int = 48,
    pools: int = 8,
    iters: int = 100,
    z_caps: tuple[int, ...] = (),
    modes: tuple[str, ...] = ("dispatch", "dispatch-2l", "raw"),
    seed: int = 0,
) -> tuple[list[Row], float]:
    """Rows for every (batch, ctx, layer, mode) and the captured fill_ cost."""
    import torch
    from vllm_sm70.ops import fv100

    if not fv100.available():
        raise RuntimeError("_sm70_C has no K05 Flash-V100 ops")
    device = torch.device("cuda", torch.cuda.current_device())
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    fused_built = fv100.has_op(fv100.FUSED_DECODE_OP)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    psize, env_slots = envelope_partition_slots(max_model_len)
    all_modes = [*modes, *(f"raw-z{n}" for n in z_caps)]
    rows: list[Row] = []

    for batch in batches:
        for ctx in contexts:
            if ctx > max_model_len:
                continue
            fv100.clear_workspaces()
            kv = [_Pool(batch, ctx, max_model_len, device, gen) for _ in range(pools)]
            q = torch.randn(
                batch,
                HEADS_Q,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
                generator=gen,
            )
            sinks = torch.randn(HEADS_Q, device=device, generator=gen)
            lens = torch.full((batch,), ctx, dtype=torch.int32, device=device)
            outs = [torch.empty_like(q) for _ in range(calls)]
            for layer in layers:
                wl = window_left(layer)
                # TritonAttentionImpl's (W - 1, W - 1); decode reads only the left.
                window = (wl, wl) if wl >= 0 else (-1, -1)
                case = _Case(batch, ctx, max_model_len, window, q, sinks, lens, outs)
                ref = _reference(case, kv[0])
                for mode in all_modes:
                    two_launch = mode == "dispatch-2l"
                    if not two_launch and not fused_built:
                        rows.append(
                            Row(
                                batch,
                                ctx,
                                layer,
                                mode,
                                0,
                                0,
                                0,
                                None,
                                None,
                                None,
                                "fused op not built",
                            )
                        )
                        continue
                    slots = int(mode[5:]) if mode.startswith("raw-z") else env_slots
                    if two_launch:
                        z, act, tok = env_slots, live_partitions(ctx, wl, psize), psize
                    else:
                        z = fused_ctas_per_head(batch, HEADS_KV, slots, num_sms)
                        work = fused_work(ctx, wl, z)
                        act, tok = work.active, work.max_tokens_per_cta
                    if mode.startswith("raw"):
                        call, env_value = _raw_call(case, slots, psize), None
                    else:
                        call = _dispatch_call(case)
                        env_value = "0" if two_launch else None
                        if not two_launch:
                            fv100.flash_attn_decode_paged_sinks_prepare(device)
                    g1, gn = _graphs(call, kv, calls, env_value)
                    g1.replay()
                    torch.cuda.synchronize()
                    diff = (outs[0].float() - ref.float()).abs().max().item()
                    g1_us = _time_replay(g1, iters)
                    gn_us = _time_replay(gn, max(1, iters // 4)) / calls
                    rows.append(
                        Row(batch, ctx, layer, mode, z, act, tok, g1_us, gn_us, diff)
                    )
                    g1 = gn = None
            kv.clear()
            fv100.clear_workspaces()
            torch.cuda.empty_cache()

    # The wrapper's per-call captured fill_ (ops/fv100.py:485-487) on its own.
    anp = torch.empty(1, dtype=torch.int32, device=device)
    g = _capture(lambda: [anp.fill_(4) for _ in range(calls)])
    fill_us = _time_replay(g, max(1, iters // 4)) / calls
    return rows, fill_us


def _fmt(x, spec: str) -> str:
    if x is None:
        return "-".rjust(int(spec.split(".")[0]))
    return format(x, spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    ap.add_argument("--layers", default=",".join(LAYERS), help="swa,full")
    ap.add_argument("--modes", default="dispatch,dispatch-2l,raw")
    ap.add_argument("--z-caps", default="", help="e.g. 8,16,32: raw-zN modes")
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--calls", type=int, default=48, help="decode calls per gN graph")
    ap.add_argument("--pools", type=int, default=8, help="distinct KV pools per graph")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--layer-split", default=None, help="SWA,FULL layers per step")
    ap.add_argument("--config", default=None, help="HF config.json for the split")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    split = None
    if args.config:
        with open(args.config) as f:
            swa, full, _ = layer_split_from_config(json.load(f))
        split = (swa, full)
    if args.layer_split:
        split = parse_layer_split(args.layer_split)

    rows, fill_us = run(
        parse_int_list(args.batches),
        parse_int_list(args.contexts),
        tuple(x for x in args.layers.split(",") if x),
        max_model_len=args.max_model_len,
        calls=args.calls,
        pools=args.pools,
        iters=args.iters,
        z_caps=parse_int_list(args.z_caps),
        modes=tuple(x for x in args.modes.split(",") if x),
    )
    hdr = (
        f"{'B':>2} {'ctx':>6} {'layer':5} {'mode':12} {'Z':>4} {'act':>4} "
        f"{'tok':>5} {'g1 us':>8} {'gN us':>8} {'diff':>8}  note"
    )
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(
            f"{r.batch:>2} {r.context:>6} {r.layer:5} {r.mode:12} {r.z:>4} "
            f"{r.active:>4} {r.tokens_per_cta:>5} {_fmt(r.g1_us, '8.2f')} "
            f"{_fmt(r.gn_us, '8.2f')} {_fmt(r.diff, '8.1e')}  {r.note}"
        )
    print(
        f"captured active_num_partitions.fill_ alone: {fill_us:.2f} us per call "
        f"({args.calls} per graph)"
    )
    print(
        "g1: us per replay of a 1-call graph; gN: us per call in a "
        f"{args.calls}-call graph; Z/act/tok: two-launch rows show envelope "
        "partitions / live partitions / partition size"
    )
    if split:
        swa, full = split
        print(f"\nper-step estimate from gN, {swa} SWA + {full} full layers:")
        for (b, c, m), us in sorted(step_estimate_us(rows, swa, full).items()):
            print(f"  B={b} ctx={c:>6} {m:12} {us / 1000.0:7.3f} ms")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"rows": [asdict(r) for r in rows], "fill_us": fill_us}, f, indent=1
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
