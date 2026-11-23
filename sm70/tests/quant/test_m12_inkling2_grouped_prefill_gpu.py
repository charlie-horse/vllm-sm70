# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the M12 Inkling 2 grouped expert prefill (TurboMind).

One V100 with vllm_sm70._sm70_C built. The C++ knobs are read once per
process, so each setting runs in a child process: the default (grouped
stage) and VLLM_SM70_INKLING_TM_GROUPED_PREFILL=0 (per-expert loop), both
with VLLM_SM70_PROFILE_TRACE=1. At the real TP2 per-rank shapes with random
codes, for prefill batches of 512, 2048 and 3867 tokens:
- each stage call is one TurboMind GEMM launch on the grouped route and
  128 on the loop;
- the route log names the Inkling 2 grouped prefill, outside capture;
- both routes match the fp16 torch path of moe.py within 1e-2 (max-abs-rel)
  and each other within 1e-2;
- the grouped route is faster. Times per layer visit go to the JSON file
  named by INKLING2_SM70_TIMING_OUT when it is set.

    python -m pytest -q -p no:cacheprovider \
        sm70/tests/quant/test_m12_inkling2_grouped_prefill_gpu.py -m gpu
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

SWITCH = "VLLM_SM70_INKLING_TM_GROUPED_PREFILL"
TIMING_ENV = "INKLING2_SM70_TIMING_OUT"
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, TP = 128, 2304, 1152, 8, 2
LOCAL = INTERMEDIATE // TP
GATE_CAP, UP_CAP = 4.0, 25.0
TOL = 1e-2
TOKENS = (512, 2048, 3867)
CHUNK = 4096  # moe_tm.CHUNK_ROWS
ROUTE_LOG = "SM70 NVFP4 MoE Inkling 2 grouped TurboMind prefill path reached"


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


def _random_codes(name, n, k, seed):
    from vllm_sm70.models.inkling2_sm70 import expert_weights as ew

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (EXPERTS, n, k), generator=g, dtype=torch.uint8)
    block = (torch.rand(EXPERTS, n, k // 16, generator=g) * 139.8 + 1.2).to(
        torch.float8_e4m3fn
    )
    return ew.Nvfp4Codes(name, codes, block, torch.full((EXPERTS,), 1e-3))


def _dequant_fp16(codes):
    from vllm_sm70.models.inkling2_sm70 import expert_weights as ew

    lut = torch.tensor(ew.E2M1_VALUES, device="cuda")
    out = torch.empty(codes.codes.shape, dtype=torch.float16, device="cuda")
    for expert in range(codes.codes.shape[0]):
        block = codes.block_scale[expert].cuda().float().repeat_interleave(16, dim=-1)
        values = lut[codes.codes[expert].cuda().long()] * block
        out[expert] = (values * float(codes.global_scale[expert])).half()
    return out


def child(out_dir: str) -> None:
    """Run in a fresh process: outputs, launch counts and times per size."""
    from torch.profiler import ProfilerActivity, profile

    from vllm_sm70.models.inkling2_sm70 import moe, moe_tm

    torch.cuda.set_device(0)
    w13 = _random_codes("w13", 2 * LOCAL, HIDDEN, 7)
    w2 = _random_codes("w2", HIDDEN, LOCAL, 8)
    route = moe_tm.Nvfp4TmExperts(
        moe_tm.turbomind(), w13, w2,
        device=torch.device("cuda"), gate_cap=GATE_CAP, up_cap=UP_CAP,
        max_slots=CHUNK,
    )  # fmt: skip
    config = SimpleNamespace(
        n_routed_experts=EXPERTS, hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE, num_experts_per_tok=TOP_K,
        route_scale=8.0, mlp_gate_softcap=GATE_CAP, mlp_up_softcap=UP_CAP,
    )  # fmt: skip
    os.environ[moe_tm.SWITCH] = "0"
    oracle = moe.Inkling2Sm70Experts(config, tp_size=TP).cuda()
    oracle.w13.copy_(_dequant_fp16(w13))
    oracle.w2.copy_(_dequant_fp16(w2))
    report = {}
    for tokens in TOKENS:
        g = torch.Generator().manual_seed(tokens)
        x = (torch.randn(tokens, HIDDEN, generator=g) * 0.5).half().cuda()
        ids = torch.stack(
            [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)]
        ).cuda()
        weights = torch.rand(tokens, TOP_K, generator=g).cuda()
        out = torch.full_like(x, float("nan"))
        route.expert_sum(x, ids, weights, out)  # warm
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            route.expert_sum(x, ids, weights, out)
            torch.cuda.synchronize()
        launches = sum(1 for e in prof.events() if "gemm_kernel" in e.name)
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            route.expert_sum(x, ids, weights, out)
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop))
        ref = torch.empty_like(x)
        oracle.torch_expert_sum(x, ids, weights, ref)
        torch.save(out.cpu(), os.path.join(out_dir, f"out-{tokens}.pt"))
        report[str(tokens)] = {
            "launches": launches,
            "ms": sorted(times)[len(times) // 2],
            "vs_torch": max_abs_rel(out, ref),
            "finite": bool(torch.isfinite(out).all()),
        }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f)


def _run_child(tmp_path, name, env_extra):
    out_dir = tmp_path / name
    out_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    env.update(VLLM_SM70_PROFILE_TRACE="1", **env_extra)
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('gp', {__file__!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        f"m.child({str(out_dir)!r})"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], check=False, env=env, capture_output=True, text=True,
        timeout=1800,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads((out_dir / "report.json").read_text())
    return out_dir, report, done.stderr


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail("SM70 GPU present but vllm_sm70._sm70_C is not loaded")
    tmp = tmp_path_factory.mktemp("grouped_prefill")
    grouped = _run_child(tmp, "grouped", {})
    loop = _run_child(tmp, "loop", {SWITCH: "0"})
    if os.environ.get(TIMING_ENV):
        with open(os.environ[TIMING_ENV], "w") as f:
            json.dump({"grouped": grouped[1], "loop": loop[1]}, f, indent=1)
    return grouped, loop


@pytest.mark.parametrize("tokens", TOKENS)
def test_one_launch_per_stage(runs, tokens):
    (_, grouped, _), (_, loop, _) = runs
    chunks = -(-tokens * TOP_K // CHUNK)
    assert grouped[str(tokens)]["launches"] == 2 * chunks
    assert loop[str(tokens)]["launches"] == 2 * chunks * EXPERTS


def test_route_log(runs):
    (_, _, grouped_err), (_, _, loop_err) = runs
    lines = [line for line in grouped_err.splitlines() if ROUTE_LOG in line]
    assert lines and all("capture=none" in line for line in lines), grouped_err
    assert ROUTE_LOG not in loop_err


@pytest.mark.parametrize("tokens", TOKENS)
def test_matches_torch_path_and_loop(runs, tokens):
    (g_dir, grouped, _), (l_dir, loop, _) = runs
    for report in (grouped, loop):
        assert report[str(tokens)]["finite"]
        assert report[str(tokens)]["vs_torch"] < TOL
    a = torch.load(g_dir / f"out-{tokens}.pt")
    b = torch.load(l_dir / f"out-{tokens}.pt")
    assert max_abs_rel(a, b) < TOL


@pytest.mark.parametrize("tokens", TOKENS)
def test_grouped_is_faster(runs, tokens):
    (_, grouped, _), (_, loop, _) = runs
    assert grouped[str(tokens)]["ms"] < loop[str(tokens)]["ms"]
