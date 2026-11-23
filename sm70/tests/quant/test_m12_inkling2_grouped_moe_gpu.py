# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the M12 Inkling 2 grouped expert route
(inkling2_sm70/ops/grouped_gemm.py) and the soft-capped SwiGLU kernel
(ops/activation.py).

One V100. The route (Triton kernels on) is compared with the torch expert
sum of moe.py, max-abs-rel under 2e-3:
- at the real per-rank shape (128 experts, top-8, hidden 2304, local
  intermediate 576 at TP2) with random fp16 experts at M in {1, 2, 8, 64,
  512}, with padding rows and with skewed routing past the capacity;
- captured in a CUDA graph and replayed on new routing (equal to the eager
  route), and eagerly under ``torch.cuda.set_sync_debug_mode("error")``;
- on the tiny checkpoint's MXFP8 layer through the module;
- on the real MXFP8 layers 20-23 at both TP2 ranks when
  INKLING2_SM70_CHECKPOINT names the checkpoint directory (skipped
  otherwise).
The SwiGLU kernel is within 1 fp16 ulp of the torch form on random and
saturating inputs (tiny magnitudes included), bounded by 800, and allocates
nothing. ``test_time_per_visit`` prints the time per layer visit of the
route (eager and graph replay) and of the torch expert sum (row by row in a
graph for M <= 8, eager grouped otherwise); with INKLING2_SM70_TIMING_OUT
set it also writes them there as JSON.

    python -m pytest -q -s -p no:cacheprovider \
        sm70/tests/quant/test_m12_inkling2_grouped_moe_gpu.py -m gpu
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import moe, quant_config, sm70_configs
from vllm_sm70.models.inkling2_sm70.ops import activation, grouped_gemm, torch_ref

TINY = pathlib.Path(__file__).resolve().parents[1] / "models/fixtures/inkling2_tiny"
CHECKPOINT_ENV = "INKLING2_SM70_CHECKPOINT"
TIMING_ENV = "INKLING2_SM70_TIMING_OUT"
SWITCH = moe.SWITCH
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 128, 2304, 1152, 8
GATE_CAP, UP_CAP = 4.0, 25.0
TOL = 2e-3
SIZES = [1, 2, 8, 64, 512]
GRAPH_SIZES = [1, 2, 8, 64]
MXFP8_LAYERS = (20, 21, 22, 23)


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


@pytest.fixture(autouse=True)
def _gpu(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    monkeypatch.delenv(SWITCH, raising=False)
    torch.cuda.set_device(0)
    yield
    torch.cuda.empty_cache()


def real_config():
    return SimpleNamespace(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_per_tok=TOP_K,
        route_scale=8.0,
        mlp_gate_softcap=GATE_CAP,
        mlp_up_softcap=UP_CAP,
    )


@pytest.fixture(scope="module")
def real():
    """One TP2 rank's experts, random fp16 in a dequantized layer's range."""
    if not torch.cuda.is_available():
        pytest.skip("requires an SM70 GPU")
    patch = pytest.MonkeyPatch()
    patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
    module = moe.Inkling2Sm70Experts(real_config(), tp_size=2).cuda()
    patch.undo()
    g = torch.Generator(device="cuda").manual_seed(0)
    for name in ("w13", "w2"):
        buffer = getattr(module, name)
        buffer.copy_(torch.randn(buffer.shape, generator=g, device="cuda") * 0.02)
    yield module
    del module
    torch.cuda.empty_cache()


def routing(m, seed, *, skew=False, pad=()):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, HIDDEN, generator=g) * 0.5).half().cuda()
    if skew:  # a Zipf-like load: a few experts take most tokens
        probs = 1.0 / torch.arange(1, EXPERTS + 1).float()
        indices = torch.stack(
            [torch.multinomial(probs, TOP_K, generator=g) for _ in range(m)]
        )
    else:
        indices = torch.stack(
            [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(m)]
        )
    weights = torch.rand(m, TOP_K, generator=g) * 2
    indices[list(pad)] = EXPERTS
    weights[list(pad)] = 0.0
    return x, indices.cuda(), weights.cuda()


def route(module, x, indices, weights, out=None):
    out = torch.full_like(x, float("nan")) if out is None else out
    grouped_gemm.grouped_expert_sum(
        x, indices, weights, module.w13, module.w2, GATE_CAP, UP_CAP, out
    )
    return out


def oracle(module, x, indices, weights, rowwise=False):
    out = torch.full_like(x, float("nan"))
    module.torch_expert_sum(x, indices, weights, out, rowwise)
    return out


# --------------------------------------------------------------------------
# Real shape
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", SIZES)
def test_real_shape_matches_torch_path(real, m):
    args = routing(m, seed=m)
    out = route(real, *args)
    assert bool(torch.isfinite(out).all())
    assert max_abs_rel(out, oracle(real, *args)) < TOL


@pytest.mark.parametrize("m,pad", [(2, [1]), (64, [0, 31, 63]), (512, [5, 511])])
def test_real_shape_padding_rows(real, m, pad):
    args = routing(m, seed=m + 1, pad=pad)
    out = route(real, *args)
    assert bool((out[pad] == 0).all())
    assert max_abs_rel(out, oracle(real, *args)) < TOL


@pytest.mark.parametrize("m", [64, 512])
def test_real_shape_skewed_routing(real, m):
    args = routing(m, seed=m + 2, skew=True)
    counts = torch.bincount(args[1].flatten().cpu(), minlength=EXPERTS)
    assert int(counts.max()) > grouped_gemm.capacity(m * TOP_K, EXPERTS)
    assert max_abs_rel(route(real, *args), oracle(real, *args)) < TOL


@pytest.mark.parametrize("m", SIZES)
def test_route_has_no_host_sync(real, m):
    args = routing(m, seed=m + 3)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = route(real, *args)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert max_abs_rel(out, oracle(real, *args)) < TOL


def _captured(module, m):
    """A CUDA graph of the module's expert sum on static inputs."""
    x, indices, weights = routing(m, seed=0)
    out = torch.empty_like(x)
    assert moe.on_grouped(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module.expert_sum(x, indices, weights, out, rowwise=True)  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module.expert_sum(x, indices, weights, out, rowwise=True)
    return graph, (x, indices, weights), out


@pytest.mark.parametrize("m", GRAPH_SIZES)
def test_graph_capture_and_replay(real, m):
    graph, static, out = _captured(real, m)
    for seed in (11, 12):
        for slot, value in zip(static, routing(m, seed=seed, pad=[0])):
            slot.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        eager = route(real, *static)
        assert torch.equal(out, eager)
        assert bool((out[0] == 0).all())
        if m > 1:  # with m == 1 the only row is padding and the oracle is 0
            assert max_abs_rel(out, oracle(real, *static)) < TOL


def _ms(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _graph_ms(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return _ms(graph.replay)


def test_time_per_visit(real, record_property):
    rows = []
    for m in SIZES:
        x, indices, weights = routing(m, seed=m)
        args = (x, indices, weights, torch.empty_like(x))

        def grouped(args=args):
            route(real, *args)

        def rowwise(args=args):
            real.torch_expert_sum(*args, True)

        def torch_path(args=args):
            real.torch_expert_sum(*args)

        row = {
            "m": m,
            "capacity": grouped_gemm.capacity(m * TOP_K, EXPERTS),
            "grouped_eager_ms": _ms(grouped),
        }
        if m <= max(GRAPH_SIZES):
            row["grouped_graph_ms"] = _graph_ms(grouped)
        if m <= 8:
            row["torch_rowwise_graph_ms"] = _graph_ms(rowwise)
        row["torch_eager_ms"] = _ms(torch_path, iters=5)
        rows.append(row)
        print(row)
    record_property("time_per_visit", rows)
    target = os.environ.get(TIMING_ENV)
    if target:
        pathlib.Path(target).write_text(json.dumps(rows, indent=1))


# --------------------------------------------------------------------------
# Soft-capped SwiGLU kernel
# --------------------------------------------------------------------------
def _ulps(got, want):
    big = torch.maximum(got.float().abs(), want.float().abs())
    exponent = torch.floor(torch.log2(big.clamp(min=2.0**-14)))
    return (got.float() - want.float()).abs() / torch.exp2(exponent - 10)


SATURATING = [0.0, -0.0, 6e-8, -6e-8, 1e-7, -1e-7, 1e-3, -1e-3, 0.5, -3.0,
              16.0, -16.0, 30.0, -30.0, 100.0, -100.0, 1e3, -1e3, 3e4, -3e4,
              65504.0, -65504.0]  # fmt: skip


@pytest.mark.parametrize("inputs", ["random", "saturating"])
def test_softcap_kernel_within_one_ulp(inputs):
    g = torch.Generator().manual_seed(7)
    if inputs == "random":
        gate_up = (torch.randn(4096, 2 * 576, generator=g) * 6).half().cuda()
    else:
        values = torch.tensor(SATURATING)
        pairs = torch.cartesian_prod(values, values)
        gate_up = torch.cat([pairs[:, :1], pairs[:, 1:]], 1).half().cuda()
    out = torch.empty(
        gate_up.shape[0], gate_up.shape[1] // 2, dtype=torch.half, device="cuda"
    )
    activation.softcapped_swiglu_out(gate_up, out, GATE_CAP, UP_CAP)  # compile
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    got = activation.softcapped_swiglu_out(gate_up, out, GATE_CAP, UP_CAP)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    assert got is out
    gate, up = gate_up.chunk(2, dim=-1)
    want = torch_ref.softcapped_swiglu(gate, up, GATE_CAP, UP_CAP)
    assert bool(torch.isfinite(out).all())
    assert float(_ulps(out, want).max()) <= 1.0
    assert float(out.float().abs().max()) <= 800.0


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def test_tiny_checkpoint_mxfp8_layer_through_the_module(monkeypatch):
    from safetensors.torch import load_file

    native = load_file(TINY / "model.safetensors")
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    formats = quant_config.expert_formats(
        json.loads((TINY / "hf_quant_config.json").read_text())
    )
    module = moe.Inkling2Sm70Experts(config).cuda()
    module.load_expert_layer(ew.load_expert_layers(native, config, formats=formats)[23])
    prefix = "model.llm.layers.23.mlp.gate."
    module.gate.weight.data.copy_(native[prefix + "weight"])
    module.gate.bias.data.copy_(native[prefix + "bias"])
    recorded = load_file(TINY / "ops.safetensors")["experts_mxfp8.0.in.hidden_states"]
    for m in SIZES:
        x = recorded[torch.arange(m) % recorded.shape[0]].half().cuda()
        is_padding = torch.zeros(m, dtype=torch.bool, device="cuda")
        out, ref = torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))
        module(x, out, is_padding, m <= 8)
        monkeypatch.setenv(SWITCH, "0")
        module(x, ref, is_padding, m <= 8)
        monkeypatch.delenv(SWITCH)
        assert max_abs_rel(out, ref) < TOL, m


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_checkpoint_mxfp8_layers_match_torch_path(monkeypatch, tp_rank):
    directory = os.environ.get(CHECKPOINT_ENV)
    if not directory:
        pytest.skip(f"set {CHECKPOINT_ENV} to the checkpoint directory")
    from safetensors import safe_open

    directory = pathlib.Path(directory)
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((directory / "config.json").read_text())
    )
    wanted = re.compile(
        r"model\.llm\.layers\.(\d+)\.mlp\.(experts\.w(13|2)_weight.*|gate\.(weight|bias))"
    )
    loader = ew.Inkling2ExpertLoader(
        config, tp_size=2, tp_rank=tp_rank,
        formats=quant_config.load_expert_formats(str(directory)),
    )  # fmt: skip
    gates = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as shard:
            for name in shard.keys():  # noqa: SIM118 (safe_open, not a dict)
                match = wanted.fullmatch(name)
                if match is None or int(match.group(1)) not in MXFP8_LAYERS:
                    continue
                if ".gate." in name:
                    gates[name] = shard.get_tensor(name)
                else:
                    loader(name, shard.get_tensor(name))
    assert sorted(loader.layers) == list(MXFP8_LAYERS)
    for index in MXFP8_LAYERS:
        assert loader.layers[index].format == ew.MXFP8
        module = moe.Inkling2Sm70Experts(config, tp_size=2).cuda()
        module.load_expert_layer(loader.layers[index])
        prefix = f"model.llm.layers.{index}.mlp.gate."
        module.gate.weight.data.copy_(gates[prefix + "weight"])
        module.gate.bias.data.copy_(gates[prefix + "bias"])
        assert module.tm_route is None
        for m in SIZES:
            g = torch.Generator().manual_seed(1000 * index + m)
            x = torch.randn(m, HIDDEN, generator=g).half().cuda()
            is_padding = torch.zeros(m, dtype=torch.bool, device="cuda")
            out, ref = (
                torch.full_like(x, float("nan")),
                torch.full_like(x, float("nan")),
            )
            module(x, out, is_padding, m <= 8)
            monkeypatch.setenv(SWITCH, "0")
            module(x, ref, is_padding, m <= 8)
            monkeypatch.delenv(SWITCH)
            assert max_abs_rel(out, ref) < TOL, (index, m)
        del module
        torch.cuda.empty_cache()
