# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 merged visit kernels (inkling2_sm70/ops/visit.py, the
RouterInput route of ops/routing.py) and the route of visit_fusion.py.

- the kill switch VLLM_SM70_INKLING_VISIT_FUSION and the route choice;
- through Triton's interpreter, against the launches each kernel merges:
  the router GEMV inside ``route``; ``gate_blend`` against ``gate_down``,
  ``gate_up`` and ``blend_rms_norm``; the pending conv + mix of
  ``norms.sublayer_inputs`` against ``short_conv`` and the unmerged mix on
  paged caches; ``conv_residual``; the merged prologue against
  ``query_rel`` and ``key_value_conv``.
The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). AOT compile check:
test_m12_inkling2_triton_sm70_compile_cpu.py; model-level check:
tests/models/test_m12_inkling2_visit_fusion_model_cpu.py.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _inkling2_sconv as sconv_cases

OPS = HERE.parents[1] / "vllm_sm70" / "models" / "inkling2_sm70" / "ops"
SWITCH = "VLLM_SM70_INKLING_VISIT_FUSION"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
# Inkling 2 Mini: hidden 2304, 128 experts, top-8, route scale 8.
MINI = (2304, 128, 8, 8.0)
SMALL = (64, 8, 2, 2.0)
GATE_RANK = 64


def _fusion():
    from vllm_sm70.models.inkling2_sm70 import visit_fusion

    return visit_fusion


def _ops(name: str):
    import importlib

    return importlib.import_module(f"vllm_sm70.models.inkling2_sm70.ops.{name}")


def _counters():
    from vllm_sm70.models.inkling2_sm70.ops import tickets

    return tickets.ticket_counter(torch.device("cpu"), 0)


# --- source, switch and route choice --------------------------------------------------


def test_kernel_source_holds_nothing_sm70_rejects():
    text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / "visit.py").read_text())
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    for banned in ("tl.dot", "tanh.approx", "inline_asm", "gdc_", "bfloat16"):
        assert banned not in code, banned


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_route_off(monkeypatch, value, enabled):
    fusion = _fusion()
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert fusion.SWITCH == SWITCH
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["models"]
    gpu = SimpleNamespace(is_cuda=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fusion._compat, "AVAILABLE", True)
        patch.setattr(fusion.sconv, "on_triton", lambda x: True)
        patch.setattr(fusion, "on_fused", lambda x, decode_only: decode_only)
        assert fusion.on_decode(gpu, True) is enabled
        assert fusion.on_visit(gpu, True, True) is enabled
        assert fusion.on_decode(gpu, False) is False
        assert fusion.on_visit(gpu, False, True) is False
        # The convs fuse only where norms.py runs its kernels.
        assert fusion.on_visit(gpu, True, False) is False
        patch.setattr(fusion.sconv, "on_triton", lambda x: False)
        assert fusion.on_visit(gpu, True, True) is False
    if not INTERPRETED:
        assert fusion.on_decode(torch.zeros(1, 4), True) is False


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    for name in (
        SWITCH,
        "VLLM_SM70_INKLING_TRITON_NORMS",
        "VLLM_SM70_INKLING_TRITON_SCONV",
        "VLLM_SM70_INKLING_FUSED_SMALL_OPS",
    ):
        env.pop(name, None)
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
    assert passed and int(passed.group(1)) >= 40, summary
    assert "skipped" not in summary, summary


# --- interpreted: router and route ------------------------------------------------------


def _router_inputs(tokens, size, seed):
    hidden, experts, _, _ = size
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g).half()
    weight = torch.randn(experts, hidden, generator=g) / hidden**0.5
    bias = torch.randn(experts, generator=g) * 0.05
    return x, weight, bias


def _dispatch(x, slots, experts):
    rows = torch.full((slots + 3, x.shape[1]), float("nan"), dtype=x.dtype)
    sorted_ids = torch.full((slots + 3,), -7, dtype=torch.int32)
    offsets = torch.full((experts + 1,), -7, dtype=torch.int32)
    return x, rows, sorted_ids, offsets


@interpreted
@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("size", [SMALL, MINI], ids=["small", "mini"])
@pytest.mark.parametrize("dispatch", [False, True], ids=["route", "dispatch"])
@pytest.mark.parametrize("pad_id", [-1, None], ids=["pad-1", "padE"])
def test_interpreted_router_in_the_route_launch(tokens, size, dispatch, pad_id):
    routing = _ops("routing")
    _, experts, top_k, scale = size
    pad_id = experts if pad_id is None else pad_id
    x, weight, bias = _router_inputs(tokens, size, tokens)
    flags = torch.zeros(tokens, dtype=torch.bool)
    if tokens > 1:
        flags[tokens // 2] = True
    # The rows come from the dispatch input (act-quant's output), the logits
    # from the router input.
    src = (x.float() * 0.5).half()
    slots = tokens * top_k
    want_buf = _dispatch(src, slots, experts) if dispatch else None
    got_buf = _dispatch(src, slots, experts) if dispatch else None
    logits = routing.router_logits(x, weight)
    want = routing.route(
        logits, bias, top_k, scale, flags, pad_id=pad_id, dispatch=want_buf
    )
    got = routing.route(
        routing.RouterInput(x, weight), bias, top_k, scale, flags, pad_id=pad_id,
        dispatch=got_buf,
    )  # fmt: skip
    assert torch.equal(got.ids, want.ids)
    assert torch.equal(got.weights, want.weights)
    if dispatch:
        assert torch.equal(got.pos, want.pos)
        for got_t, want_t in zip(got_buf[1:], want_buf[1:]):
            assert torch.equal(
                got_t[:slots].nan_to_num(9), want_t[:slots].nan_to_num(9)
            )
        assert torch.equal(got_buf[3], want_buf[3])
    else:
        assert got.pos is None
    assert int(_counters().abs().sum()) == 0


@interpreted
def test_interpreted_router_route_twice_reuses_the_counter():
    routing = _ops("routing")
    _, _, top_k, scale = SMALL
    for seed in (0, 1):
        x, weight, bias = _router_inputs(4, SMALL, seed)
        want = routing.route(routing.router_logits(x, weight), bias, top_k, scale,
                             None, pad_id=-1)  # fmt: skip
        got = routing.route(routing.RouterInput(x, weight), bias, top_k, scale,
                            None, pad_id=-1)  # fmt: skip
        assert torch.equal(got.ids, want.ids)
        assert int(_counters().abs().sum()) == 0


# --- interpreted: message gates ---------------------------------------------------------


def _gate_case(streams, tokens, d, dtype, seed, broadcast):
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(shape, generator=g) * scale

    dest = randn(1 if broadcast else streams, tokens, d)
    if broadcast:
        dest = dest.expand(streams, -1, -1)
    message = randn(streams, tokens, d, scale=2.0)
    down = [
        randn(GATE_RANK, 2 * d, scale=(2 * d) ** -0.5).to(dtype) for _ in range(streams)
    ]
    gain = [randn(d, GATE_RANK, scale=0.3).to(dtype) for _ in range(streams)]
    forget = [randn(d, GATE_RANK, scale=0.3).to(dtype) for _ in range(streams)]
    norm = 1 + randn(1 if broadcast else streams, d, scale=0.1)
    norm = norm.expand(streams, -1)
    return dest, message, down, gain, forget, norm


@interpreted
@pytest.mark.parametrize(
    "streams, tokens, d", [(1, 1, 2304), (3, 1, 2304), (1, 4, 96), (3, 5, 96)]
)
@pytest.mark.parametrize("weight", [torch.float16, torch.float32])
@pytest.mark.parametrize("out", [torch.float16, torch.float32])
def test_interpreted_gate_blend_matches_the_three_kernels(
    streams, tokens, d, weight, out
):
    fused, loop, visit = _ops("fused"), _ops("loop"), _ops("visit")
    for broadcast in (False, True):
        dest, message, down, gain, forget, norm = _gate_case(
            streams, tokens, d, weight, streams * 10 + tokens, broadcast
        )
        low = fused.gate_down(dest, message, down, 1e-6)
        gains = fused.gate_up(low, gain, forget)
        want = loop.blend_rms_norm(dest, message, gains, norm, 1e-5, out)
        got = visit.gate_blend(dest, message, down, gain, forget, 1e-6, norm, 1e-5, out)
        assert got.dtype == out and got.shape == want.shape
        diff = (got.float() - want.float()).abs()
        # Same values up to the order of the row's sum of squares.
        tol = 2e-6 if out == torch.float32 else 1e-3
        assert float(diff.max()) <= tol * float(want.float().abs().max()), float(
            diff.max()
        )
        assert int(_counters().abs().sum()) == 0


# --- interpreted: short convs with the mixes after them ---------------------------------

STEPS = {
    "decode": [[(0, 3), (1, 2), (2, 5)]] + [[(0, 1), (1, 1), (2, 1)]] * 3,
    "verify": [[(0, 30), (1, 5), (2, 2)]] + [[(0, 4), (1, 4), (2, 4)]] * 2,
}


def _bank(paged, total, states, seed):
    from vllm_sm70.models.inkling2_sm70 import norms

    g = torch.Generator().manual_seed(seed)
    like = torch.zeros(total, sconv_cases.HIDDEN, dtype=paged.conv_cache.dtype)
    bank = norms.Bank(like, states + 1, decode_only=True)
    assert bank.triton
    for _ in range(states):
        bank.append(torch.randn(total, sconv_cases.HIDDEN, generator=g).to(like.dtype))
    return bank


class _Gate:
    def __init__(self, g, d, dtype):
        self.eps = 1e-6
        self.down_weight = (
            torch.randn(GATE_RANK, 2 * d, generator=g) / (2 * d) ** 0.5
        ).to(dtype)
        self.gain_weight = (torch.randn(d, GATE_RANK, generator=g) * 0.3).to(dtype)
        self.forget_weight = (torch.randn(d, GATE_RANK, generator=g) * 0.3).to(dtype)


def _mix_case(mode, fused, dtype, steps, *, has_base, queries, gates, states):
    """Every output and the caches of `steps` on one route: per batch the
    conv of stream 2 into the mix (`mode` "mlp": the attention conv and the
    MLP input; "attn": the layer output and the next attention input)."""
    from vllm_sm70.models.inkling2_sm70 import norms, sconv, visit_fusion
    from vllm_sm70.models.inkling2_sm70.ops import fused as fused_ops

    paged = sconv_cases.Paged("cpu", dtype)
    weights = sconv_cases.Weights("cpu", dtype, False, seed=3)
    g = torch.Generator().manual_seed(7)
    d = sconv_cases.HIDDEN
    outputs = []
    for index, step in enumerate(steps):
        metadata, n = paged.batch(
            step, pad=1, hole=sconv_cases.n_hole(step),
            decode_only=len({count for _, count in step}) == 1,
        )  # fmt: skip
        total = n + 1
        padding = metadata["conv"].slot_mapping < 0
        x = weights.randn(total, d).masked_fill(padding[:, None], 0.0)
        base = weights.randn(total, d) if has_base else None
        update = weights.randn(total, d)
        query = torch.randn(queries, d, generator=g)
        norm_weight = (1 + 0.1 * torch.randn(max(queries, 1), d, generator=g)).to(dtype)
        if gates:
            norm_weight = norm_weight[:1].expand(gates, -1)
        gate_list = [_Gate(g, d, dtype) for _ in range(gates)]
        messages = torch.randn(max(gates, 1) + 3, total, d, generator=g)
        bank = _bank(paged, total, states, index)
        decode_only = metadata["conv"].decode_only
        conv = visit_fusion.Conv(x, weights.out_conv[False], paged.conv, 2)
        with paged.forward(metadata):
            if mode == "mlp":
                conv_out = torch.full_like(x, float("nan"))
                if fused and decode_only:
                    state = visit_fusion.Pending(conv, base, conv_out=conv_out)
                else:
                    sconv.short_conv(x, conv.weight, paged.conv, 2, conv_out,
                                     zero_padding=False)  # fmt: skip
                    state = norms.Residual(base, conv_out, bank.triton)
                side = conv_out
            else:
                hidden = torch.full_like(x, float("nan"))
                if fused and decode_only:
                    state = visit_fusion.Pending(
                        conv, base, update=update, state_out=hidden
                    )
                else:
                    mlp_out = torch.empty_like(x)
                    sconv.short_conv(x, conv.weight, paged.conv, 2, mlp_out,
                                     zero_padding=False)  # fmt: skip
                    hidden.copy_(fused_ops.residual_add(base, update, mlp_out, dtype))
                    state = hidden
                side = hidden
            streams = norms.sublayer_inputs(
                bank, state, query.float().contiguous(), norm_weight, 1e-5, dtype,
                gates=gate_list, messages=messages if gates else None,
            )  # fmt: skip
            if isinstance(state, visit_fusion.Pending):
                assert state.done
        outputs += [*streams, side]
    return outputs + [paged.conv_cache, paged.kv_cache]


def _same(got, want, tol=0.0):
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want)):
        assert a.shape == b.shape, i
        err = sconv_cases.rel_err(a, b)
        assert err <= tol, (i, err)


@interpreted
@pytest.mark.parametrize("mode", ["mlp", "attn"])
@pytest.mark.parametrize("has_base", [True, False], ids=["base", "block_start"])
@pytest.mark.parametrize("queries", [1, 3])
@pytest.mark.parametrize("scenario", sorted(STEPS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_interpreted_conv_mix_matches_the_conv_then_the_mix(
    mode, has_base, queries, scenario, dtype
):
    steps = STEPS[scenario]
    kwargs = {"has_base": has_base, "queries": queries, "gates": 0, "states": 2}
    want = _mix_case(mode, False, dtype, steps, **kwargs)
    got = _mix_case(mode, True, dtype, steps, **kwargs)
    _same(got, want)


@interpreted
@pytest.mark.parametrize("mode", ["mlp", "attn"])
@pytest.mark.parametrize("gates", [1, 3])
def test_interpreted_conv_mix_feeds_the_message_gates(mode, gates):
    kwargs = {"has_base": True, "queries": 1, "gates": gates, "states": 1}
    want = _mix_case(mode, False, torch.float32, STEPS["decode"], **kwargs)
    got = _mix_case(mode, True, torch.float32, STEPS["decode"], **kwargs)
    # The gate outputs differ by the order of a row's sum of squares.
    _same(got, want, tol=2e-6)


@interpreted
@pytest.mark.parametrize("has_base", [True, False], ids=["base", "block_start"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_interpreted_pending_output_alone_is_conv_and_add(has_base, dtype):
    from vllm_sm70.models.inkling2_sm70 import sconv, visit_fusion
    from vllm_sm70.models.inkling2_sm70.ops import fused as fused_ops

    outputs = {}
    for fused in (False, True):
        paged = sconv_cases.Paged("cpu", dtype)
        weights = sconv_cases.Weights("cpu", dtype, False, seed=5)
        got = []
        for step in STEPS["decode"]:
            metadata, n = paged.batch(step, pad=1, decode_only=True)
            total = n + 1
            x = weights.randn(total, sconv_cases.HIDDEN)
            x[n:] = 0
            base = weights.randn(total, sconv_cases.HIDDEN) if has_base else None
            update = weights.randn(total, sconv_cases.HIDDEN)
            with paged.forward(metadata):
                if fused:
                    conv = visit_fusion.Conv(x, weights.out_conv[False], paged.conv, 2)
                    out = torch.full_like(x, float("nan"))
                    pending = visit_fusion.Pending(
                        conv, base, update=update, state_out=out
                    )
                    got.append(visit_fusion.value(pending))
                else:
                    y = torch.empty_like(x)
                    sconv.short_conv(x, weights.out_conv[False], paged.conv, 2, y,
                                     zero_padding=False)  # fmt: skip
                    got.append(fused_ops.residual_add(base, update, y, dtype))
        outputs[fused] = got + [paged.conv_cache]
    _same(outputs[True], outputs[False])


# --- interpreted: prologue ------------------------------------------------------------


@interpreted
@pytest.mark.parametrize("dynamic", [False, True], ids=["sliding", "global"])
@pytest.mark.parametrize("scenario", ["decode", "verify", "short"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_interpreted_prologue_in_one_launch(monkeypatch, dynamic, scenario, dtype):
    visit = _ops("visit")
    calls = []
    real = visit.prologue
    monkeypatch.setattr(
        visit, "prologue", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    steps = sconv_cases.SCENARIOS[scenario]
    outputs = {}
    for value in ("0", "1"):
        monkeypatch.setenv(SWITCH, value)
        weights = sconv_cases.Weights("cpu", dtype, dynamic)
        outputs[value] = sconv_cases.run(weights, steps, pad=1, decode_only=True)
        assert bool(calls) == (value == "1")
    _same(outputs["1"], outputs["0"])
