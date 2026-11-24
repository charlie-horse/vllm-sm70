# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 grouped expert route (inkling2_sm70/ops/grouped_gemm.py) and
the soft-capped SwiGLU op (ops/activation.py), on CPU.

- the kill switch VLLM_SM70_INKLING_GROUPED_MOE, the route choice of moe.py,
  the capacity rule; the kernel sources hold nothing sm_70 rejects;
- the torch form of the route against the torch expert sum of moe.py (fp32
  within 1e-5, fp16 within 2e-3) at M from 1 to 600, skewed routing, padding
  rows, out-of-range ids, chunks, and the tiny checkpoint's MXFP8 layer;
- no host sync and the same ops for two routings of one batch size;
- through Triton's interpreter: the SwiGLU kernel within 1 fp16 ulp of the
  torch form, the slot GEMV kernel, the whole route with its kernels.
The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). AOT compile check:
tests/ops/test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
from inkling2_sync_guard import no_host_sync

OPS = HERE.parents[1] / "vllm_sm70" / "models" / "inkling2_sm70" / "ops"
TINY = HERE.parent / "models" / "fixtures" / "inkling2_tiny"
SWITCH = "VLLM_SM70_INKLING_GROUPED_MOE"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 64, 32, 2
GATE_CAP, UP_CAP = 4.0, 25.0
DTYPES = [
    pytest.param(torch.float32, 1e-5, id="fp32"),
    pytest.param(torch.float16, 2e-3, id="fp16"),
]


def _moe():
    from vllm_sm70.models.inkling2_sm70 import moe

    return moe


def _gg():
    from vllm_sm70.models.inkling2_sm70.ops import grouped_gemm

    return grouped_gemm


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


def experts(dtype, seed=0):
    config = SimpleNamespace(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_per_tok=TOP_K,
        route_scale=2.0,
        mlp_gate_softcap=GATE_CAP,
        mlp_up_softcap=UP_CAP,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        module = _moe().Inkling2Sm70Experts(config, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    module.w13.copy_(torch.randn(module.w13.shape, generator=g) * 0.3)
    module.w2.copy_(torch.randn(module.w2.shape, generator=g) * 0.3)
    module.gate.weight.data.copy_(torch.randn(EXPERTS, HIDDEN, generator=g))
    return module


def batch(module, m, seed, *, skew=False, pad=()):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, HIDDEN, generator=g).to(module.dtype)
    is_padding = torch.zeros(m, dtype=torch.bool)
    is_padding[list(pad)] = True
    _, weights, indices = module.route(x, is_padding)
    if skew:  # every token on experts 0 and 1: past the capacity
        indices = torch.stack([torch.ones(m), torch.zeros(m)], 1).long()
        indices[list(pad)] = EXPERTS
    return x, indices, weights


def grouped(module, x, indices, weights, kernels):
    out = torch.full_like(x, float("nan"))
    _gg().grouped_expert_sum(
        x, indices, weights, module.w13, module.w2, GATE_CAP, UP_CAP, out,
        kernels=kernels,
    )  # fmt: skip
    return out


def oracle(module, x, indices, weights):
    out = torch.full_like(x, float("nan"))
    module.torch_expert_sum(x, indices, weights, out)
    return out


# --- source, switch, route and capacity -------------------------------------------


def test_kernel_sources_hold_nothing_sm70_rejects():
    for name in ("activation.py", "grouped_gemm.py"):
        text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / name).read_text())
        code = "\n".join(line.split("#")[0] for line in text.splitlines())
        for banned in ("tl.dot", "tanh.approx", "inline_asm", "gdc_", "bfloat16"):
            assert banned not in code, (name, banned)


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_route_off(monkeypatch, value, enabled):
    moe = _moe()
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert moe.SWITCH == SWITCH
    with mock.patch.object(moe._compat, "AVAILABLE", True):
        assert moe.on_grouped(SimpleNamespace(is_cuda=True)) is enabled
    if not INTERPRETED:
        assert moe.on_grouped(torch.zeros(1)) is False


@pytest.mark.parametrize("on", [True, False])
def test_expert_sum_takes_the_route(monkeypatch, on):
    moe = _moe()
    module = experts(torch.float32)
    x, indices, weights = batch(module, 3, 0)
    calls = []
    monkeypatch.setattr(moe, "on_grouped", lambda _: on)
    monkeypatch.setattr(
        moe.grouped_gemm, "grouped_expert_sum", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr(module, "torch_expert_sum", lambda *a: calls.append("torch"))
    module.expert_sum(x, indices, weights, torch.empty_like(x), rowwise=True)
    assert len(calls) == 1 and (calls[0] != "torch") is on


def test_capacity_rule():
    gg = _gg()
    real = {m: gg.capacity(8 * m, 128) for m in (1, 2, 8, 9, 64, 512)}
    assert real == {1: 0, 2: 0, 8: 0, 9: 8, 64: 8, 512: 64}
    for slots in range(gg.SLOT_MAX + 1, 4097, 37):
        cap = gg.capacity(slots, 128)
        assert cap % gg.CAP_ALIGN == 0 and cap * 128 >= gg.CAP_FACTOR * slots
    assert gg.capacity(gg.CHUNK_TOKENS * 8, 128) == 64


# --- torch form against the torch expert sum ----------------------------------------


@pytest.mark.parametrize("dtype, tol", DTYPES)
@pytest.mark.parametrize("m", [1, 2, 8, 40, 64, 512, 600])
def test_torch_form_matches_torch_expert_sum(dtype, tol, m):
    module = experts(dtype)
    args = batch(module, m, seed=m, pad=[m // 2] if m > 1 else ())
    got = grouped(module, *args, kernels=False)
    want = oracle(module, *args)
    assert got.dtype == dtype and max_abs_rel(got, want) <= tol
    if m > 1:
        assert bool((got[m // 2] == 0).all())


@pytest.mark.parametrize("dtype, tol", DTYPES)
@pytest.mark.parametrize("m", [8, 64])
def test_skewed_routing_past_the_capacity(dtype, tol, m):
    module = experts(dtype)
    args = batch(module, m, seed=1, skew=True, pad=[0])
    assert m * TOP_K <= _gg().SLOT_MAX or _gg().capacity(m * TOP_K, EXPERTS) < m
    got = grouped(module, *args, kernels=False)
    assert max_abs_rel(got, oracle(module, *args)) <= tol
    assert bool((got[0] == 0).all())


@pytest.mark.parametrize("m", [1, 40])
def test_ids_outside_the_experts_add_nothing(m):
    module = experts(torch.float32)
    x, indices, weights = batch(module, m, seed=3)
    indices = indices.clone()
    indices[0, 0] = -1
    indices[-1, 1] = EXPERTS + 3 if m > 1 else indices[-1, 1]
    got = grouped(module, x, indices, weights, kernels=False)
    assert max_abs_rel(got, oracle(module, x, indices, weights)) <= 1e-5
    x, indices, weights = batch(module, m, seed=3, pad=range(m))
    assert bool((grouped(module, x, indices, weights, kernels=False) == 0).all())


def test_chunks_match_one_pass(monkeypatch):
    module = experts(torch.float32)
    args = batch(module, 100, seed=4)
    whole = grouped(module, *args, kernels=False)
    monkeypatch.setattr(_gg(), "CHUNK_TOKENS", 16)
    assert max_abs_rel(grouped(module, *args, kernels=False), whole) <= 1e-6


@pytest.mark.parametrize("dtype, tol", DTYPES)
def test_mxfp8_layer_of_the_tiny_checkpoint(dtype, tol):
    """Layer 23 (MXFP8) against the reference's recorded expert sum, and
    against the torch expert sum in fp16."""
    from safetensors.torch import load_file

    from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
    from vllm_sm70.models.inkling2_sm70 import quant_config, sm70_configs

    native = load_file(TINY / "model.safetensors")
    ops = load_file(TINY / "ops.safetensors")
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    formats = quant_config.expert_formats(
        json.loads((TINY / "hf_quant_config.json").read_text())
    )
    module = _moe().Inkling2Sm70Experts(config, dtype=dtype)
    layer = ew.load_expert_layers(native, config, formats=formats)[23]
    assert layer.format == ew.MXFP8
    module.load_expert_layer(layer)
    rows = torch.arange(64) % 20
    x = ops["experts_mxfp8.0.in.hidden_states"][rows].to(dtype)
    indices = ops["experts_mxfp8.0.in.top_k_index"][rows]
    weights = ops["experts_mxfp8.0.in.top_k_weights"][rows]
    got = torch.full_like(x, float("nan"))
    _gg().grouped_expert_sum(
        x, indices, weights, *module.expert_weights(),
        module.gate_cap, module.up_cap, got, kernels=False,
    )  # fmt: skip
    if dtype == torch.float32:
        want = ops["experts_mxfp8.0.out.0"][rows]
        assert (got - want).abs().max() <= tol
    else:
        assert max_abs_rel(got, oracle(module, x, indices, weights)) <= tol


@pytest.mark.parametrize("m, maps", [(8, 1), (40, 2)])
def test_act_hook_maps_every_swiglu_output_before_w2(m, maps):
    module = experts(torch.float32)
    x, indices, weights = batch(module, m, seed=8)
    seen = []

    def hook(act):
        seen.append(act.shape[-1])
        return act * 0.5

    out = torch.full_like(x, float("nan"))
    _gg().grouped_expert_sum(
        x, indices, weights, module.w13, module.w2, GATE_CAP, UP_CAP, out,
        kernels=False, act_hook=hook,
    )  # fmt: skip
    assert seen == [INTERMEDIATE] * maps
    want = grouped(module, x, indices, weights, kernels=False) * 0.5
    assert max_abs_rel(out, want) <= 1e-5


# --- no host sync, static ops ---------------------------------------------------------


def _traced(module, m, seed, kernels):
    x, indices, weights = batch(module, m, seed=seed, pad=[0])
    trace: list = []
    out = torch.full_like(x, float("nan"))
    with no_host_sync(trace):
        _gg().grouped_expert_sum(
            x, indices, weights, module.w13, module.w2, GATE_CAP, UP_CAP, out,
            kernels=kernels,
        )  # fmt: skip
    return trace, out


@pytest.mark.parametrize("m", [1, 8, 40, 600])
def test_no_host_sync_and_the_same_ops_for_any_routing(m):
    module = experts(torch.float16)
    first, _ = _traced(module, m, 5, kernels=False)
    second, out = _traced(module, m, 6, kernels=False)
    assert first == second and len(first) > 0
    assert bool(torch.isfinite(out).all())


def test_softcap_torch_form_writes_into_out():
    from vllm_sm70.models.inkling2_sm70.ops import activation, torch_ref

    gate_up = torch.randn(5, 2 * INTERMEDIATE).half() * 8
    out = torch.empty(5, INTERMEDIATE, dtype=torch.float16)
    got = activation.softcapped_swiglu_out(gate_up, out, GATE_CAP, UP_CAP, kernel=False)
    gate, up = gate_up.chunk(2, dim=-1)
    assert got is out
    assert torch.equal(out, torch_ref.softcapped_swiglu(gate, up, GATE_CAP, UP_CAP))


# --- interpreted kernels ----------------------------------------------------------------


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    env.pop(SWITCH, None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", __file__, "-q", "-p",
            "no:cacheprovider", "-m", "not gpu", "-k",
            "interpreted and not test_interpreted_suite",
        ],
        env=env, capture_output=True, text=True, timeout=3600, check=False,
    )  # fmt: skip
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 20, summary
    assert "skipped" not in summary, summary


def _ulps(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """|got - want| in fp16 units in the last place of the larger value."""
    big = torch.maximum(got.float().abs(), want.float().abs())
    exponent = torch.floor(torch.log2(big.clamp(min=2.0**-14)))
    return (got.float() - want.float()).abs() / torch.exp2(exponent - 10)


# No tiny magnitudes: the interpreter's tanh form (ops/_compat.py) loses them;
# the GPU test covers them with libdevice tanh.
SATURATING = [0.0, -0.0, 0.5, -3.0, 16.0, -16.0, 30.0, -30.0,
              100.0, -100.0, 1e3, -1e3, 3e4, -3e4, 65504.0, -65504.0]  # fmt: skip


@interpreted
@pytest.mark.parametrize("inputs", ["random", "saturating"])
def test_interpreted_softcap_kernel_within_one_ulp(inputs):
    from vllm_sm70.models.inkling2_sm70.ops import activation, torch_ref

    g = torch.Generator().manual_seed(7)
    if inputs == "random":
        gate_up = (torch.randn(33, 2 * 48, generator=g) * 6).half()
    else:
        values = torch.tensor(SATURATING)
        pairs = torch.cartesian_prod(values, values)
        gate_up = torch.cat([pairs[:, :1], pairs[:, 1:]], 1).half()
    rows, units = gate_up.shape[0], gate_up.shape[1] // 2
    out = torch.full((rows, units), float("nan"), dtype=torch.float16)
    pointer = out.data_ptr()
    got = activation.softcapped_swiglu_out(gate_up, out, GATE_CAP, UP_CAP)
    assert got is out and out.data_ptr() == pointer
    gate, up = gate_up.chunk(2, dim=-1)
    want = torch_ref.softcapped_swiglu(gate, up, GATE_CAP, UP_CAP)
    assert bool(torch.isfinite(out).all())
    assert float(_ulps(out, want).max()) <= 1.0
    assert float(out.float().abs().max()) <= min(800.0, GATE_CAP * UP_CAP)


@interpreted
@pytest.mark.parametrize("row_div", [1, TOP_K])
def test_interpreted_slot_gemv(row_div):
    gg = _gg()
    g = torch.Generator().manual_seed(row_div)
    weight = torch.randn(EXPERTS, 40, 150, generator=g)
    slot_experts = torch.tensor([3, -1, 0, 7, 7, -1], dtype=torch.int32)
    inputs = torch.randn(6 // row_div, 150, generator=g)
    got = gg._slot_gemv(inputs, weight, slot_experts, row_div, kernels=True)
    want = gg._slot_gemv(inputs, weight, slot_experts, row_div, kernels=False)
    assert (got - want).abs().max() <= 1e-4
    assert bool((got[[1, 5]] == 0).all())


@interpreted
@pytest.mark.parametrize("dtype, tol", DTYPES)
@pytest.mark.parametrize(
    "m, skew",
    [(1, False), (2, False), (8, False), (40, False), (64, False), (40, True)],
)
def test_interpreted_route_matches_torch_paths(dtype, tol, m, skew):
    module = experts(dtype)
    args = batch(module, m, seed=m, skew=skew, pad=[m - 1] if m > 1 else ())
    got = grouped(module, *args, kernels=True)
    assert max_abs_rel(got, grouped(module, *args, kernels=False)) <= tol
    assert max_abs_rel(got, oracle(module, *args)) <= tol
    if m > 1:
        assert bool((got[m - 1] == 0).all())


@interpreted
@pytest.mark.parametrize("m", [1, 40])
def test_interpreted_module_forward_takes_the_kernels(monkeypatch, m):
    monkeypatch.delenv(SWITCH, raising=False)
    module = experts(torch.float16)
    x = batch(module, m, seed=9)[0]
    is_padding = torch.zeros(m, dtype=torch.bool)
    assert _moe().on_grouped(x)
    out, ref = torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))
    with no_host_sync():
        module(x, out, is_padding, True)
    monkeypatch.setenv(SWITCH, "0")
    assert not _moe().on_grouped(x)
    module(x, ref, is_padding, True)
    assert max_abs_rel(out, ref) <= 2e-3


@interpreted
@pytest.mark.parametrize("m", [2, 40])
def test_interpreted_kernels_have_no_host_sync(m):
    module = experts(torch.float16)
    first, _ = _traced(module, m, 5, kernels=True)
    second, out = _traced(module, m, 6, kernels=True)
    assert first == second and bool(torch.isfinite(out).all())
