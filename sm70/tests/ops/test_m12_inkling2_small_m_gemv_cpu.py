# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 small-batch dense GEMV (inkling2_sm70/ops/small_m_gemv.py)
and its route (inkling2_sm70/gemv.py), on CPU.

- the kill switch VLLM_SM70_INKLING_SMALL_M_GEMV and the route choice: 1 to
  8 fp16 rows with unit K stride, from 2 rows for the LM head, tensors off
  the GPU on cuBLAS's torch ops, only "0" turns it off;
- through Triton's interpreter, each against the torch op it replaces:
  - ``segment_gemv`` with one input and with three inputs over 3 and 5 row
    segments, strided inputs and output, at small sizes and the Inkling 2
    Mini TP2 shapes (gated qkvr with its k and v inputs, wo);
  - prologue.py ``project_qkvr`` (local and gated) and ``project_packed``
    (with and without the conv taps) against their F.linear forms;
  - the output projection of attention_torch.py and the LM head of
    model.py.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). The AOT compile check is
test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import importlib.util
import itertools
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

OPS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "vllm_sm70"
    / "models"
    / "inkling2_sm70"
    / "ops"
)
SWITCH = "VLLM_SM70_INKLING_SMALL_M_GEMV"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
HIDDEN = 2304
# One TP2 rank of an Inkling 2 Mini gated layer: q, r, gate, sink, k, v rows.
QKVR_SIZES = (1536, 192, 1536, 12, 512, 512)
TOL = 2e-3


def _route():
    from vllm_sm70.models.inkling2_sm70 import gemv

    return gemv


def _kernels():
    from vllm_sm70.models.inkling2_sm70.ops import small_m_gemv

    return small_m_gemv


def max_abs_rel(got, want):
    scale = max(float(want.float().abs().max()), 1e-6)
    return float((got.float() - want.float()).abs().max() / scale)


def _rand(*shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).half()


# --- source, switch and route choice --------------------------------------------------


def test_kernel_source_holds_nothing_sm70_rejects():
    text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / "small_m_gemv.py").read_text())
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    for banned in ("tl.dot", "tanh.approx", "inline_asm", "gdc_", "bfloat16"):
        assert banned not in code, banned


def test_switch_is_registered_in_models():
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["models"]
    assert _route().SMALL_M_SWITCH == SWITCH


def _gpu_like(rows, k=64, dtype=torch.float16, stride=1):
    return SimpleNamespace(
        ndim=2,
        shape=(rows, k),
        dtype=dtype,
        is_cuda=True,
        stride=lambda dim: stride if dim == 1 else k,
    )


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_route_off(monkeypatch, value, enabled):
    route = _route()
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    monkeypatch.setattr(route._compat, "AVAILABLE", True)
    monkeypatch.setattr(route._compat, "INTERPRET", False)
    assert route.on_small_m(_gpu_like(4), _gpu_like(32)) is enabled


def test_route_takes_one_to_eight_fp16_rows(monkeypatch):
    route = _route()
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(route._compat, "AVAILABLE", True)
    monkeypatch.setattr(route._compat, "INTERPRET", False)
    weight = _gpu_like(32)
    assert [route.on_small_m(_gpu_like(m), weight) for m in range(10)] == [
        False, *[True] * 8, False,
    ]  # fmt: skip
    assert not route.on_small_m(_gpu_like(1), weight, route.HEAD_MIN_ROWS)
    assert route.on_small_m(_gpu_like(2), weight, route.HEAD_MIN_ROWS)
    assert not route.on_small_m(_gpu_like(4, dtype=torch.float32), weight)
    assert not route.on_small_m(_gpu_like(4, stride=2), weight)
    assert not route.on_small_m(_gpu_like(4, k=48), weight)


def test_tensors_off_the_gpu_take_cublas_outside_the_interpreter(monkeypatch):
    route = _route()
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(route._compat, "INTERPRET", False)
    x, w = _rand(3, 64, seed=1), _rand(40, 64, seed=2)
    assert route.on_small_m(x, w) is False
    assert route.segments((x, x, x), w, (10, 20)) is None
    assert torch.equal(route.linear(x, w), torch.nn.functional.linear(x, w))
    out = torch.empty(3, 40).half()
    route.linear(x, w, out=out)
    assert torch.equal(out, torch.nn.functional.linear(x, w))


def test_configs_cover_every_row_count():
    kernels = _kernels()
    for rows in range(1, kernels.MAX_ROWS + 1):
        block_m, block_n, block_k, warps = kernels.config(rows)
        assert block_m >= rows and block_n * block_k * block_m <= 8192
        assert warps in (1, 2, 4, 8)


# --- segment_gemv ------------------------------------------------------------------------


def _reference(inputs, weight, bounds):
    edges = (0, *bounds, weight.shape[0])
    order = (0, 1, 2, 1, 2)
    return torch.cat(
        [
            inputs[order[i]].float() @ weight[a:b].float().t()
            for i, (a, b) in enumerate(itertools.pairwise(edges))
        ],
        dim=1,
    )


@interpreted
@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8])
@pytest.mark.parametrize(
    "bounds", [(), (20, 45), (5, 20, 46, 61)], ids=["one", "three", "five"]
)
def test_interpreted_segments_match_torch(rows, bounds):
    kernels = _kernels()
    k = 200  # not a multiple of any BLOCK_K
    weight = _rand(77, k, seed=rows, scale=0.1)
    wide = _rand(3, rows, k + 8, seed=10 + rows)
    inputs = tuple(wide[i, :, 3 : 3 + k] for i in range(3 if bounds else 1))
    out = torch.full((rows, 90), float("nan")).half()
    kernels.segment_gemv(inputs, weight, bounds, out[:, :77])
    want = _reference(inputs if bounds else inputs * 3, weight, bounds)
    assert max_abs_rel(out[:, :77], want) <= TOL
    assert bool(out[:, 77:].isnan().all())


@interpreted
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_interpreted_mini_gated_qkvr(rows):
    from vllm_sm70.models.inkling2_sm70 import prologue

    weight = _rand(sum(QKVR_SIZES), HIDDEN, seed=3, scale=0.02)
    x_q, x_k, x_v = (_rand(rows, HIDDEN, seed=4 + i) for i in range(3))
    got = prologue.project_qkvr(x_q, x_k, x_v, weight, QKVR_SIZES)
    query_rows = sum(QKVR_SIZES[:4])
    w_q, w_k, w_v = weight.split((query_rows, *QKVR_SIZES[4:]), dim=0)
    want = (
        *torch.nn.functional.linear(x_q, w_q).split(QKVR_SIZES[:4], dim=-1),
        torch.nn.functional.linear(x_k, w_k),
        torch.nn.functional.linear(x_v, w_v),
    )
    assert len(got) == len(want) == 6
    for a, b in zip(got, want):
        assert a.shape == b.shape and max_abs_rel(a, b) <= TOL


@interpreted
@pytest.mark.parametrize("rows", [1, 3])
def test_interpreted_local_qkvr_and_wo(rows):
    from vllm_sm70.models.inkling2_sm70 import attention_torch, prologue

    sizes = (64, 16, 64, 4, 32, 32)
    weight = _rand(sum(sizes), 96, seed=5, scale=0.1)
    x = _rand(rows, 96, seed=6)
    got = prologue.project_qkvr(x, None, None, weight, sizes)
    want = torch.nn.functional.linear(x, weight).split(sizes, dim=-1)
    for a, b in zip(got, want):
        assert max_abs_rel(a, b) <= TOL
    # The Inkling 2 Mini TP2 output projection, into a row-strided output.
    w_o = _rand(HIDDEN, 1536, seed=7, scale=0.02)
    gated = _rand(rows, 1536, seed=8)
    out = torch.full((rows, HIDDEN + 16), float("nan")).half()
    attention_torch._project(gated, out[:, :HIDDEN], w_o)
    assert max_abs_rel(out[:, :HIDDEN], gated.float() @ w_o.float().t()) <= TOL
    assert bool(out[:, HIDDEN:].isnan().all())


@interpreted
@pytest.mark.parametrize("taps", [0, 12], ids=["no_taps", "taps"])
@pytest.mark.parametrize("rows", [1, 6])
def test_interpreted_packed_projections(taps, rows):
    from vllm_sm70.models.inkling2_sm70 import prologue

    sizes = (48, 12, 48, 3, 16, 16)
    weight = _rand(sum(sizes) + 2 * taps, 80, seed=9, scale=0.1)
    x_q, x_k, x_v = (_rand(rows, 80, seed=20 + i) for i in range(3))
    got = prologue.project_packed(x_q, x_k, x_v, weight, sizes, taps)
    edges = [0, *torch.tensor((*sizes, taps, taps)).cumsum(0).tolist()]
    rows_of = [weight[a:b] for a, b in itertools.pairwise(edges)]
    inputs = (x_q, x_q, x_q, x_q, x_k, x_v, x_k, x_v)
    for i, part in enumerate(got):
        if i >= 6 and not taps:
            assert part is None
            continue
        want = torch.nn.functional.linear(inputs[i], rows_of[i])
        assert max_abs_rel(part, want) <= TOL


@interpreted
@pytest.mark.parametrize("rows", [1, 2, 8])
def test_interpreted_lm_head_from_two_rows(rows, monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import model

    calls = []
    kernels = _kernels()
    real = kernels.segment_gemv

    def counted(*args):
        calls.append(args[0][0].shape[0])
        return real(*args)

    monkeypatch.setattr(_route(), "segment_gemv", counted)
    head = SimpleNamespace(weight=_rand(300, 64, seed=11, scale=0.1))
    hidden = _rand(rows, 64, seed=12).float()
    got = model._head_logits(head, hidden, None)
    want = torch.nn.functional.linear(hidden.half(), head.weight)
    assert got.dtype == torch.float16 and max_abs_rel(got, want) <= TOL
    assert calls == ([] if rows < 2 else [rows])


# --- interpreted child run ------------------------------------------------------------


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
    assert passed and int(passed.group(1)) >= 27, summary
    assert "skipped" not in summary, summary
