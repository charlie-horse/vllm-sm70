# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 small decode kernels (inkling2_sm70/ops/small_ops.py) and
the route of fused_small.py, on CPU.

- the kill switch VLLM_SM70_INKLING_FUSED_SMALL_OPS and the route choice:
  decode-only batches only, tensors off the GPU, only "0" turns it off;
- through Triton's interpreter, each against the torch ops it replaces:
  - ``query_rel`` (q and r clips, tau, r @ rel_proj, padding rows) at the
    Inkling 2 Mini sizes and through ``prologue.attention_prologue`` with
    the paged caches of tests/ops/_inkling2_sconv.py;
  - ``gate`` and the gate of ``attention_torch.attention_with_metadata``
    with and without the output projection;
  - the TurboMind SwiGLU of moe_tm.py (pure-torch tm of
    tests/quant/test_m12_inkling2_nvfp4_tm_cpu.py) and the module forward
    of a decode-only batch;
  - the short conv without its padding mask on inputs that are zero there;
  - padding rows zero in every output.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). The AOT compile check is
test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import types
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
import _inkling2_decode_attn as attn_cases
import _inkling2_sconv as sconv_cases

OPS = HERE.parents[1] / "vllm_sm70" / "models" / "inkling2_sm70" / "ops"
TINY = HERE.parent / "models" / "fixtures" / "inkling2_tiny"
SWITCH = "VLLM_SM70_INKLING_FUSED_SMALL_OPS"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
# Inkling 2 Mini per rank at TP2: 12 query heads of 128, d_rel 16, extent 1024.
MINI = (12, 128, 16, 1024)
SMALL = (3, 24, 4, 40)
QK_CLIP, REL_CLIP = 22.627416997969522, 8.0
LOG_SCALING = (0.1, 8192.0)


def _route():
    from vllm_sm70.models.inkling2_sm70 import fused_small

    return fused_small


def _small_ops():
    from vllm_sm70.models.inkling2_sm70.ops import small_ops

    return small_ops


def max_abs_rel(got, want):
    scale = max(float(want.float().abs().max()), 1e-6)
    return float((got.float() - want.float()).abs().max() / scale)


# --- source, switch and route choice --------------------------------------------------


def test_kernel_source_holds_nothing_sm70_rejects():
    text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / "small_ops.py").read_text())
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    for banned in ("tl.dot", "tanh.approx", "inline_asm", "gdc_", "bfloat16"):
        assert banned not in code, banned


def test_switch_is_registered_in_models():
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["models"]
    assert _route().SWITCH == SWITCH


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_route_off(monkeypatch, value, enabled):
    route = _route()
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    gpu = SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(route._compat, "AVAILABLE", True)
    assert route.on_fused(gpu, True) is enabled
    assert route.on_fused(gpu, False) is False


def test_tensors_off_the_gpu_take_torch_outside_the_interpreter(monkeypatch):
    route = _route()
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(route._compat, "INTERPRET", False)
    assert route.on_fused(torch.zeros(1), True) is False


# --- query_rel ------------------------------------------------------------------------


def _query_inputs(size, tokens, seed, scale_dtype=torch.float16):
    heads, head_dim, d_rel, extent = size
    g = torch.Generator().manual_seed(seed)
    # q and r are column slices of one projection, as project_qkvr returns them.
    # Head norms around the q clip at every head size.
    spread = 3 * (128 / head_dim) ** 0.5
    wide = torch.randn(tokens, heads * (head_dim + d_rel) + 8, generator=g) * spread
    wide = wide.half()
    q = wide[:, : heads * head_dim]
    r = wide[:, heads * head_dim : heads * (head_dim + d_rel)]
    q_scale = (1 + 0.3 * torch.randn(heads, head_dim, generator=g)).to(scale_dtype)
    r_scale = (1 + 0.3 * torch.randn(heads, d_rel, generator=g)).to(scale_dtype)
    rel_proj = torch.randn(d_rel, extent, generator=g).half()
    return q, r, q_scale, r_scale, rel_proj


def _query_rel_both(size, tokens, slots, positions, n, log_scaling, scale_dtype):
    from vllm_sm70.models.inkling2_sm70 import prologue

    heads, head_dim, _, extent = size
    q, r, q_scale, r_scale, rel_proj = _query_inputs(size, tokens, tokens, scale_dtype)
    md = SimpleNamespace(
        positions=positions, slot_mapping=slots, decode_only=True, num_actual_tokens=n
    )
    outs = []
    for fused in (True, False):
        q_out = torch.full((tokens, heads, head_dim), float("nan")).half()
        rel_out = torch.full((tokens, heads, extent), float("nan")).half()
        if fused:
            _small_ops().query_rel(
                q, r, q_scale, r_scale, rel_proj, slots, positions, q_out, rel_out,
                n, q_clip=QK_CLIP, r_clip=REL_CLIP, log_scaling=log_scaling,
            )  # fmt: skip
        else:
            prologue._query_rel_torch(
                q, r, q_scale, r_scale, rel_proj, md, q_out, rel_out, n,
                QK_CLIP, REL_CLIP, log_scaling,
            )  # fmt: skip
        outs.append((q_out, rel_out))
    return outs


@interpreted
@pytest.mark.parametrize("size", [MINI, SMALL], ids=["mini", "small"])
@pytest.mark.parametrize("log_scaling", [None, LOG_SCALING], ids=["sliding", "global"])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float32])
def test_interpreted_query_rel_matches_torch(size, log_scaling, scale_dtype):
    tokens = 5
    # Row 1 has no slot, row 4 is after n; positions cross n_floor.
    slots = torch.tensor([40, -1, 7, 90, -1])
    positions = torch.tensor([3, 0, 8191, 30000, 0])
    (q_got, rel_got), (q_want, rel_want) = _query_rel_both(
        size, tokens, slots, positions, 4, log_scaling, scale_dtype
    )
    for got, want in ((q_got, q_want), (rel_got, rel_want)):
        assert not got.isnan().any()
        assert max_abs_rel(got, want) <= 1e-3
        for row in (1, 4):
            assert bool((got[row] == 0).all())
    # The clip is active on some heads, inactive on others.
    norms = q_want[[0, 2, 3]].float().norm(dim=-1)
    assert bool((norms > QK_CLIP * 0.999).any()) and bool((norms < QK_CLIP).any())


# --- gate -----------------------------------------------------------------------------


@interpreted
@pytest.mark.parametrize("channels", [12 * 128, 3 * 24])
def test_interpreted_gate_matches_torch(channels):
    from vllm_sm70.models.inkling2_sm70 import attention_torch

    g = torch.Generator().manual_seed(channels)
    attn = torch.randn(4, channels // 8, 8, generator=g).half()
    gate = (torch.randn(4, channels + 16, generator=g) * 4).half()[:, :channels]
    slots = torch.tensor([3, -1, 9])
    got = torch.full((4, channels), float("nan")).half()
    _small_ops().gate(attn, gate, slots, 3, got)
    want = attention_torch.gate_and_project(attn, gate, torch.empty_like(got))
    assert max_abs_rel(got[[0, 2]], want[[0, 2]]) <= 1e-3
    assert bool((got[[1, 3]] == 0).all())


@interpreted
@pytest.mark.parametrize("project", [False, True], ids=["gated", "projected"])
@pytest.mark.parametrize("pad", [0, 2])
def test_interpreted_attention_gate_route(monkeypatch, project, pad):
    attention = attn_cases.attention_module()
    case = attn_cases.Case(torch.float16, 64, 64, [70, 5, 33], pad=pad)
    rows = case.query.shape[0]
    channels = attn_cases.HEADS * attn_cases.HEAD_DIM
    g = torch.Generator().manual_seed(1)
    gate = torch.randn(rows, channels, generator=g).half()
    weight = (torch.randn(40, channels, generator=g) / channels**0.5).half()
    calls = []
    launch = _small_ops().gate
    monkeypatch.setattr(_small_ops(), "gate", lambda *a: calls.append(1) or launch(*a))

    def run():
        out = torch.full((rows, 40 if project else channels), float("nan")).half()
        return attention.attention_with_metadata(
            case.query, case.rel, case.sinks, gate, case.cache, case.md, out,
            weight if project else None,
        )  # fmt: skip

    monkeypatch.delenv(SWITCH, raising=False)
    got = run()
    assert calls == [1]
    monkeypatch.setenv(SWITCH, "0")
    want = run()
    assert calls == [1]
    assert not got.isnan().any()
    assert max_abs_rel(got, want) <= 2e-3
    if pad:
        assert bool((got[-pad:] == 0).all()) and bool((want[-pad:] == 0).all())


@interpreted
def test_interpreted_mixed_batch_keeps_the_torch_gate(monkeypatch):
    attention = attn_cases.attention_module()
    case = attn_cases.Case(
        torch.float16, 64, 64, [70, 5], decode_only=False, query_lens=[1, 3]
    )
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(_small_ops(), "gate", None)
    rows, channels = case.query.shape[0], attn_cases.HEADS * attn_cases.HEAD_DIM
    gate = torch.zeros(rows, channels).half()
    out = torch.empty(rows, channels).half()
    attention.attention_with_metadata(
        case.query, case.rel, case.sinks, gate, case.cache, case.md, out
    )
    assert not out.isnan().any()


# --- the prologue through the paged caches --------------------------------------------


@interpreted
@pytest.mark.parametrize("dynamic", [False, True], ids=["sliding", "global"])
@pytest.mark.parametrize("pad", [0, 2])
def test_interpreted_prologue_matches_the_torch_route(monkeypatch, dynamic, pad):
    batch = sconv_cases.Paged.batch

    def decode_only(self, *args, **kwargs):
        metadata, n = batch(self, *args, **kwargs)
        for md in metadata.values():
            md.decode_only = True
        return metadata, n

    monkeypatch.setattr(sconv_cases.Paged, "batch", decode_only)
    steps = sconv_cases.SCENARIOS["decode"]
    calls = []
    launch = _small_ops().query_rel
    monkeypatch.setattr(
        _small_ops(), "query_rel", lambda *a, **k: calls.append(1) or launch(*a, **k)
    )
    monkeypatch.delenv(SWITCH, raising=False)
    got = sconv_cases.run(
        sconv_cases.Weights("cpu", torch.float16, dynamic), steps, pad=pad
    )
    assert len(calls) == len(steps)
    monkeypatch.setenv(SWITCH, "0")
    want = sconv_cases.run(
        sconv_cases.Weights("cpu", torch.float16, dynamic), steps, pad=pad
    )
    assert len(calls) == len(steps)
    for index, (a, b) in enumerate(zip(got, want, strict=True)):
        assert sconv_cases.rel_err(a, b) <= 1e-3, index
    for q_out, rel_out in zip(got[0:-2:4], got[1:-2:4]):
        if pad:
            assert bool((q_out[-pad:] == 0).all()) and bool((rel_out[-pad:] == 0).all())


@interpreted
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_interpreted_short_conv_needs_no_mask_on_zero_padding(monkeypatch, dynamic):
    from vllm_sm70.models.inkling2_sm70 import sconv

    paged = sconv_cases.Paged("cpu", torch.float16)
    weights = sconv_cases.Weights("cpu", torch.float16, True)
    stream = 3 if dynamic else 2
    weight = weights.out_conv[dynamic]
    for step in sconv_cases.SCENARIOS["decode"]:
        metadata, n = paged.batch(step, 2)
        for md in metadata.values():
            md.decode_only = True
        x = weights.randn(n + 2, sconv_cases.HIDDEN)
        x[1] = 0
        x[n:] = 0
        metadata["conv"].slot_mapping[1] = -1
        dyn = (
            weights.randn(n + 2, sconv_cases.KV_HEADS, sconv_cases.TAPS)
            if dynamic
            else None
        )
        outs = []
        for zero_padding in (True, False):
            out = torch.full_like(x, float("nan"))
            with paged.forward(metadata):
                sconv.short_conv(
                    x, weight, paged.conv, stream, out, dynamic_taps=dyn,
                    zero_padding=zero_padding,
                )  # fmt: skip
            outs.append(out)
        assert torch.equal(outs[0], outs[1])
        assert bool((outs[1][1] == 0).all()) and bool((outs[1][n:] == 0).all())


# --- the TurboMind SwiGLU -------------------------------------------------------------


def _tm_test():
    sys.path.insert(0, str(HERE.parent / "quant"))
    import test_m12_inkling2_nvfp4_tm_cpu as tm_test

    return tm_test


@pytest.fixture
def tm(monkeypatch):
    """The pure-torch tm of the NVFP4 TurboMind test, installed."""
    tm_test = _tm_test()
    from vllm_sm70.models.inkling2_sm70 import moe_tm

    monkeypatch.setenv(tm_test.SWITCH, "1")
    mock = tm_test.make_mock_tm()
    ops = types.ModuleType("vllm_sm70.ops")
    ops.tm = mock
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", mock)
    monkeypatch.setattr(
        moe_tm, "INKLING_CONTRACTS", moe_tm.INKLING_CONTRACTS | {tm_test.TINY_CONTRACT}
    )
    monkeypatch.setattr(moe_tm, "INKLING_TP_SIZES", (1, 2))
    moe_tm.Workspace.clear()
    yield mock
    moe_tm.Workspace.clear()


@interpreted
@pytest.mark.parametrize("m", [1, 3, 8])
def test_interpreted_tm_swiglu_matches_torch(tm, m):
    tm_test = _tm_test()
    x, indices, weights = tm_test.random_batch(m)
    outs = []
    for fused_act in (True, False):
        out = torch.full_like(x, float("nan"))
        tm_test.random_route(tm, max_slots=2 * m).expert_sum(
            x, indices, weights, out, fused_act=fused_act
        )
        outs.append(out)
    assert not outs[0].isnan().any()
    assert max_abs_rel(outs[0], outs[1]) <= 2e-3


@interpreted
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize(
    "routed", [True, False], ids=["fused_routing", "torch_routing"]
)
def test_interpreted_tm_module_forward_decode_only(monkeypatch, tm, tokens, routed):
    tm_test = _tm_test()
    from vllm_sm70.models.inkling2_sm70 import moe, moe_tm, sm70_configs

    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    module = tm_test.build(config)
    assert module.tm_route is not None
    if not routed:
        monkeypatch.setenv("VLLM_SM70_INKLING_FUSED_ROUTING", "0")
    calls = []
    launch = moe_tm.softcapped_swiglu_out
    monkeypatch.setattr(
        moe_tm, "softcapped_swiglu_out", lambda *a: calls.append(1) or launch(*a)
    )
    x = tm_test.hidden(tokens)
    flags = torch.zeros(tokens, dtype=torch.bool)
    flags[-1] = tokens > 1

    def run(decode_only):
        out = torch.full_like(x, float("nan"))
        module(x, out, flags, decode_only)
        return out

    monkeypatch.delenv(SWITCH, raising=False)
    got = run(True)
    assert calls == [1]
    assert moe._fused_enabled() is routed
    want_mixed = run(False)
    assert calls == [1]
    monkeypatch.setenv(SWITCH, "0")
    want = run(True)
    assert calls == [1]
    assert max_abs_rel(got, want) <= 2e-3
    assert torch.equal(want_mixed, want) or max_abs_rel(want_mixed, want) <= 1e-3
    if tokens > 1:
        assert bool((got[-1] == 0).all())


# --- interpreted child run ------------------------------------------------------------


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    for name in (
        SWITCH,
        "VLLM_SM70_INKLING_FUSED_ROUTING",
        "VLLM_SM70_INKLING_TRITON_SCONV",
        "VLLM_SM70_INKLING_TRITON_DECODE_ATTN",
        "VLLM_SM70_INKLING_SPLIT_KV_ATTN",
    ):
        env.pop(name, None)
    # The small-op kernels on their own; test_m12_inkling2_visit_fusion_cpu.py
    # checks the prologue that merges query_rel into one launch.
    env["VLLM_SM70_INKLING_VISIT_FUSION"] = "0"
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
    assert passed and int(passed.group(1)) >= 28, summary
    assert "skipped" not in summary, summary
