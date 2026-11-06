# SPDX-License-Identifier: Apache-2.0
"""Subprocess body of the warmup GPU tests (TurboMind knobs, TM_GEMM_TRACE
included, are read once per process, so each case gets a fresh one).

    python _warmup_gpu_probe.py capture
        TP 1: warm one Flint NVFP4 dense shape (K=N=2048, capture sizes
        [4, 8]), then capture M=4 (warmed) and M=5 (not warmed) into one
        CUDA graph. stderr carries the TM_GEMM_TRACE lines between
        @@CAPTURE@@ and @@END@@.
    python _warmup_gpu_probe.py tp2 <rank> <store file>
        One of two gloo ranks on cuda:<rank>: same layers (same seed), the
        coordinated warmup, then prints a JSON result line.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import torch

K = N = 2048  # a Flint TP2 dense shape (o_proj)


def nvfp4_layer(device: torch.device, seed: int = 0) -> torch.nn.Module:
    from vllm_sm70.ops import tm
    from vllm_sm70.quant.nvfp4_linear import NVFP4_GROUP_SIZE, _store_state

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (K, N), generator=g, dtype=torch.uint8)
    scales = (torch.rand(K // 16, N, generator=g) * 0.01 + 0.001).half()
    w, s, meta = tm.nvfp4_sm70_prepare(codes.to(device), scales.to(device), 16, False)
    layer = torch.nn.Module()
    _store_state(layer, w, s, meta, NVFP4_GROUP_SIZE, N, N)
    return layer


def worker(device: torch.device, layers: list[torch.nn.Module], capture: list[int]):
    model = torch.nn.ModuleList(layers)
    mode = SimpleNamespace(
        name="FULL_AND_PIECEWISE", mixed_mode=lambda: SimpleNamespace(name="PIECEWISE")
    )
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=capture, cudagraph_mode=mode
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(enable_elastic_ep=False),
    )
    return SimpleNamespace(vllm_config=cfg, device=device, get_model=lambda: model)


def capture_case() -> None:
    from vllm_sm70 import warmup
    from vllm_sm70.quant.nvfp4_linear import apply_prepared_linear

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    layer = nvfp4_layer(device)
    res = warmup.run(worker(device, [layer], [4, 8]), group=None)
    assert res is not None and res.counts["nvfp4_dense"] > 0, res
    x4 = torch.zeros(4, K, dtype=torch.float16, device=device)
    x5 = torch.zeros(5, K, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    sys.stderr.write("@@CAPTURE@@\n")
    sys.stderr.flush()
    with torch.cuda.graph(graph):
        apply_prepared_linear(layer, x4, None)  # warmed: expect reuse
        apply_prepared_linear(layer, x5, None)  # not warmed: default
    torch.cuda.synchronize()
    sys.stderr.write("@@END@@\n")
    sys.stderr.flush()
    graph.replay()
    torch.cuda.synchronize()
    print(json.dumps({"counts": res.counts}))


def tp2_case(rank: int, store: str) -> None:
    import torch.distributed as dist
    from vllm_sm70 import warmup
    from vllm_sm70.warmup.coord import TorchGroup

    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    layers = [nvfp4_layer(device, seed=0)]
    group = TorchGroup(dist.group.WORLD, rank, [0, 1])
    res = warmup.run(worker(device, layers, [1, 2, 4, 8]), group=group)
    print(
        json.dumps(
            {
                "rank": rank,
                "mode": res.mode,
                "source": res.source,
                "imported": res.imported,
                "agree": res.agree,
                "digests": res.digests,
                "counts": res.counts,
            }
        )
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture_case()
    else:
        tp2_case(int(sys.argv[2]), sys.argv[3])
