# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 Triton norm kernels (the vendored inkling2_sm70/ops/norm.py,
attn_res.py, loop.py and the gate and residual kernels of ops/fused.py) and
their route (inkling2_sm70/norms.py), on CPU.

- provenance: each vendored file, with its stored patch reverse-applied, is
  the upstream file of the recorded sha256;
- the sources hold nothing sm_70 rejects or runs slowly: PDL only behind
  USE_GDC, no tanh.approx, no tl.dot, no bfloat16, no SM100/SM120 import;
- the switch VLLM_SM70_INKLING_TRITON_NORMS and the route choice: the
  kernels on decode-only batches by default, none with "0", all with "1";
- the kernels through Triton's interpreter against the torch reference ops,
  in fp32 within 1e-5 and in fp16 within 2e-3, and the route built on them
  against the torch route.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent they show as skipped. The GPU
version is test_m12_inkling2_triton_norms_gpu.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parents[1] / "vllm_sm70" / "models" / "inkling2_sm70"
OPS = PKG / "ops"
VENDORED = ("norm.py", "attn_res.py", "loop.py")
COMMIT = "c32251b5d192f0f72f9ca8c5595d1d43f1dd37b8"
SWITCH = "VLLM_SM70_INKLING_TRITON_NORMS"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
FP32_TOL = 1e-5
FP16_TOL = 2e-3
DTYPES = [
    pytest.param(torch.float32, FP32_TOL, id="fp32"),
    pytest.param(torch.float16, FP16_TOL, id="fp16"),
]


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    env.pop(SWITCH, None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu",
            "-k",
            "not test_interpreted_suite",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 60, summary
    assert "skipped" not in summary, summary


# --- provenance ---------------------------------------------------------------


def _reverse_apply(vendored: str, patch: str) -> str:
    """The upstream text: `patch` (unified diff upstream -> vendored) taken
    back out of `vendored`. Raises when a hunk does not match."""
    new = vendored.splitlines(keepends=True)
    old: list[str] = []
    pos = 0
    lines = patch.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("@@"))
    for line in lines[start:]:
        if line.startswith("@@"):
            head = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            # An empty new side names the line before it.
            first = int(head[1]) - (head[2] != "0")
            assert first >= pos, line
            old += new[pos:first]
            pos = first
        elif line[0] == "-":
            old.append(line[1:])
        else:
            assert line[0] in " +", line
            assert pos < len(new) and new[pos] == line[1:], (pos, line)
            if line[0] == " ":
                old.append(line[1:])
            pos += 1
    return "".join(old + new[pos:])


def _patch(name: str) -> tuple[dict[str, str], str]:
    text = (OPS / "patches" / f"{name}.patch").read_text(encoding="utf-8")
    head = text[: text.index("--- ")]
    return dict(re.findall(r"^(\w+): (\S+)$", head, re.MULTILINE)), text


@pytest.mark.parametrize("name", VENDORED)
def test_vendored_file_is_upstream_plus_its_patch(name):
    fields, patch = _patch(name)
    assert fields["Path"] == f"vllm/models/inkling2/nvidia/ops/{name}"
    assert fields["Commit"] == COMMIT
    vendored = (OPS / name).read_text(encoding="utf-8")
    upstream = _reverse_apply(vendored, patch)
    assert hashlib.sha256(upstream.encode()).hexdigest() == fields["sha256"]
    provenance = (PKG / "PROVENANCE.md").read_text(encoding="utf-8")
    assert f"`ops/{name}`" in provenance and fields["Path"] in provenance


@pytest.mark.parametrize("name", VENDORED)
def test_an_unrecorded_edit_fails_the_provenance_check(name):
    fields, patch = _patch(name)
    vendored = (OPS / name).read_text(encoding="utf-8")
    lines = vendored.splitlines(keepends=True)
    for edited in (
        vendored + "\n# edit\n",
        "".join(lines[:-1]),
        vendored.replace("tl.float32", "tl.float16", 1),
    ):
        assert edited != vendored
        try:
            upstream = _reverse_apply(edited, patch)
        except AssertionError:
            continue
        assert hashlib.sha256(upstream.encode()).hexdigest() != fields["sha256"]


# --- sources ------------------------------------------------------------------


def _code(name: str) -> list[str]:
    """Source lines without comments and docstrings."""
    text = (OPS / name).read_text(encoding="utf-8")
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return [line.split("#")[0].rstrip() for line in text.splitlines()]


@pytest.mark.parametrize("name", (*VENDORED, "fused.py", "_compat.py"))
def test_sources_hold_nothing_sm70_rejects(name):
    lines = _code(name)
    code = "\n".join(lines)
    for banned in ("tanh.approx", "tl.dot", "bfloat16", "launch_pdl=True"):
        assert banned not in code, banned
    assert not re.search(r"sm100|sm120|sm90|vllm\.models\b", code, re.IGNORECASE)
    for i, line in enumerate(lines):
        if "gdc_" in line:
            assert lines[i - 1].strip() == "if USE_GDC:", (name, i + 1)
    if name != "_compat.py":
        assert "tldevice" not in code and "vllm.triton_utils" not in code
        assert "inline_asm" not in code or name == "norm.py"


def test_pdl_follows_the_compute_capability(monkeypatch):
    from vllm_sm70.models.inkling2_sm70.ops import _compat

    source = (OPS / "attn_res.py").read_text(encoding="utf-8")
    assert source.count("USE_GDC=use_gdc()") == 1
    assert source.count("launch_pdl=use_gdc()") == 1
    assert source.count("gdc_wait()") == 2 == source.count("    if USE_GDC:\n")
    if INTERPRETED:
        assert _compat.use_gdc() is False
        return
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for capability, expected in (((7, 0), False), ((8, 9), False), ((9, 0), True)):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *a, c=capability: c
        )
        _compat.use_gdc.cache_clear()
        assert _compat.use_gdc() is expected
    _compat.use_gdc.cache_clear()
    assert _compat.MODEL_DTYPE is torch.float16


# --- switch and route ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "mode"),
    [(None, None), ("", None), ("true", None), (" 1", None), ("0", False), ("1", True)],
)
def test_switch_sets_the_route(monkeypatch, value, mode):
    from vllm_sm70 import envs
    from vllm_sm70.models.inkling2_sm70 import norms
    from vllm_sm70.models.inkling2_sm70.ops import _compat

    assert norms.SWITCH == SWITCH and SWITCH in envs._SECTIONS["models"]
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert getattr(envs, SWITCH) is mode
    x = torch.zeros(2, 8)
    gpu = type("Gpu", (), {"is_cuda": True})()
    for decode_only in (False, True):
        want = decode_only if mode is None else mode
        assert norms.on_triton(x, decode_only) is (INTERPRETED and want)
        bank = norms.Bank(x, 1, decode_only=decode_only)
        assert bank.triton is (INTERPRETED and want)
        # A GPU tensor follows the route when Triton is there.
        monkeypatch.setattr(_compat, "AVAILABLE", True)
        assert norms.on_triton(gpu, decode_only) is want
        monkeypatch.setattr(_compat, "AVAILABLE", False)
        assert norms.on_triton(gpu, decode_only) is (INTERPRETED and want)
    assert norms.on_triton(x) is (INTERPRETED and bool(mode))


FUSED = ("gate_down", "gate_up", "residual_add")


def _forbid_kernels(monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import norms

    def forbidden(*args, **kwargs):
        raise AssertionError("a Triton kernel wrapper was called")

    monkeypatch.setattr(norms.norm_ops, "rms_norm", forbidden)
    monkeypatch.setattr(norms.attn_res_ops, "attn_res", forbidden)
    monkeypatch.setattr(norms.loop_ops, "pair_rms_norm", forbidden)
    monkeypatch.setattr(norms.loop_ops, "blend_rms_norm", forbidden)
    for name in FUSED:
        monkeypatch.setattr(norms.fused_ops, name, forbidden)


def _counted(calls: dict[str, int], name: str, real):
    def wrapper(*args, **kwargs):
        calls[name] = calls.get(name, 0) + 1
        return real(*args, **kwargs)

    return wrapper


def _count_kernels(monkeypatch) -> dict[str, int]:
    from vllm_sm70.models.inkling2_sm70 import norms

    calls: dict[str, int] = {}
    for module, name in (
        (norms.norm_ops, "rms_norm"),
        (norms.attn_res_ops, "attn_res"),
        (norms.loop_ops, "pair_rms_norm"),
        (norms.loop_ops, "blend_rms_norm"),
        *((norms.fused_ops, name) for name in FUSED),
    ):
        monkeypatch.setattr(module, name, _counted(calls, name, getattr(module, name)))
    return calls


def test_torch_route_matches_the_reference_ops_and_calls_no_kernel(monkeypatch):
    """The route off the GPU, and with the switch at 0 in the interpreter."""
    import _inkling2_norms as cases

    from vllm_sm70.models.inkling2_sm70.ops import torch_ref

    monkeypatch.setenv(SWITCH, "0")
    _forbid_kernels(monkeypatch)
    for case in cases.ROUTE_CASES:
        out = cases.route_outputs(cases.Inputs("cpu", torch.float32), case, 5, 48, 2)
        assert len(out) - 2 == (case[3] or max(case[1], case[2])), case[0]
        assert out[-1].shape == (5, 2, 48) and out[-1].dtype == torch.float32
    inputs = cases.Inputs("cpu", torch.float32, seed=3)
    state, bank_state = inputs.randn(5, 48), inputs.randn(5, 48)
    query, weight = inputs.randn(1, 48), inputs.weight(1, 48)
    from vllm_sm70.models.inkling2_sm70 import norms

    bank = norms.Bank(state, 1)
    bank.append(bank_state)
    (got,) = norms.sublayer_inputs(bank, state, query, weight, cases.EPS, state.dtype)
    mixed = torch_ref.attn_res(bank_state[None, :, None], state[None], query)[0, :, 0]
    assert torch.equal(got, torch_ref.rms_norm(mixed, weight[0], cases.EPS))
    for start in (False, True):
        for gated in (False, True):
            cases.residual_outputs(inputs, 5, 48, 1, start=start, gated=gated)


def test_default_route_takes_the_kernels_on_decode_only_batches(monkeypatch):
    """Switch unset: a forward that is not decode-only calls no kernel; in
    the interpreter a decode-only one does."""
    import _inkling2_norms as cases

    from vllm_sm70.models.inkling2_sm70 import norms

    monkeypatch.delenv(SWITCH, raising=False)
    calls = _count_kernels(monkeypatch)
    inputs = cases.Inputs("cpu", torch.float32, seed=4)
    state, query, weight = (
        inputs.randn(3, 48),
        inputs.randn(1, 48),
        inputs.weight(1, 48),
    )
    for decode_only in (False, True):
        calls.clear()
        bank = norms.Bank(state, 1, decode_only=decode_only)
        bank.append(inputs.randn(3, 48))
        residual = norms.Residual(state, inputs.randn(3, 48), bank.triton)
        norms.sublayer_inputs(bank, residual, query, weight, cases.EPS, state.dtype)
        residual.add(inputs.randn(3, 48), state.dtype)
        if decode_only and INTERPRETED:
            assert calls == {"rms_norm": 1, "attn_res": 1, "residual_add": 1}
        else:
            assert calls == {}


# --- interpreted kernels --------------------------------------------------------


@pytest.fixture
def cases():
    if not INTERPRETED:
        pytest.skip("runs in the TRITON_INTERPRET=1 child (test_interpreted_suite)")
    import _inkling2_norms

    return _inkling2_norms


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize(("tokens", "hidden"), [(1, 48), (5, 64), (3, 600)])
def test_rms_norm_kernel(cases, dtype, tol, tokens, hidden):
    assert cases.check_rms_norm(cases.Inputs("cpu", dtype), tokens, hidden) <= tol


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("stack_normed", [False, True], ids=["raw", "normed"])
@pytest.mark.parametrize("norm", [False, True], ids=["mix", "mix_norm"])
@pytest.mark.parametrize("case", range(6))
def test_attn_res_kernels(cases, dtype, tol, stack_normed, norm, case):
    tokens, blocks, queries = cases.ATTN_RES_CASES[case]
    err = cases.check_attn_res(
        cases.Inputs("cpu", dtype, seed=case),
        tokens,
        48,
        blocks,
        queries,
        norm=norm,
        addend=case % 2 == 1,
        stack_normed=stack_normed,
    )
    assert err <= tol


@pytest.mark.parametrize("norm", [False, True], ids=["mix", "mix_norm"])
def test_attn_res_blocked_kernel_walks_a_wide_row(cases, norm):
    """H above the 1024-wide block of the large-batch kernel."""
    err = cases.check_attn_res(
        cases.Inputs("cpu", torch.float32),
        257,
        1100,
        2,
        1,
        norm=norm,
        addend=True,
        stack_normed=True,
    )
    assert err <= FP32_TOL


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize(("streams", "tokens"), [(1, 1), (3, 5), (2, 17)])
def test_loop_kernels(cases, dtype, tol, streams, tokens):
    """Up to fused.MAX_TOKENS tokens the gate kernels, above it the pair
    norm and torch.bmm."""
    inputs = cases.Inputs("cpu", dtype, seed=streams)
    assert cases.check_pair_rms_norm(inputs, streams, tokens, 48) <= tol
    assert cases.check_blend(inputs, streams, tokens, 48) <= tol


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize(("streams", "tokens", "hidden"), [(1, 1, 48), (3, 4, 600)])
def test_gate_kernels(cases, dtype, tol, streams, tokens, hidden):
    inputs = cases.Inputs("cpu", dtype, seed=tokens)
    assert cases.check_gate_kernels(inputs, streams, tokens, hidden) <= tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_residual_add_kernel_is_exact(cases, dtype):
    assert cases.check_residual_add(cases.Inputs("cpu", dtype), 3, 1100) == 0.0


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("gated", [False, True], ids=["first_pass", "gated"])
@pytest.mark.parametrize("start", [False, True], ids=["inside", "block_start"])
def test_residual_route_matches_the_torch_route(
    cases, monkeypatch, dtype, tol, gated, start
):
    calls = _count_kernels(monkeypatch)
    monkeypatch.setenv(SWITCH, "1")
    inputs = cases.Inputs("cpu", dtype, seed=7)
    got = cases.residual_outputs(inputs, 3, 48, 2, start=start, gated=gated)
    assert calls["residual_add"] == calls["attn_res"] - gated == 1
    assert calls.get("gate_down", 0) == calls.get("gate_up", 0) == int(gated)
    used = dict(calls)
    monkeypatch.setenv(SWITCH, "0")
    inputs = cases.Inputs("cpu", dtype, seed=7)
    want = cases.residual_outputs(inputs, 3, 48, 2, start=start, gated=gated)
    assert calls == used, "the torch route launched a kernel"
    assert cases.rel_err(got[0], want[0]) <= tol
    assert torch.equal(got[1], want[1])


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("case", range(8))
@pytest.mark.parametrize("blocks", [0, 2])
def test_triton_route_matches_the_torch_route(
    cases, monkeypatch, dtype, tol, case, blocks
):
    case = cases.ROUTE_CASES[case]
    if case[3] and not blocks:
        blocks = 1
    calls = _count_kernels(monkeypatch)
    monkeypatch.setenv(SWITCH, "1")
    got = cases.route_outputs(cases.Inputs("cpu", dtype), case, 5, 48, blocks)
    assert calls.get("rms_norm", 0) >= 1 + blocks
    if case[1]:
        assert calls["attn_res"] == 1 + bool(case[3])
    if case[3]:
        # Five tokens: the gate kernels, not the pair norm and torch.bmm.
        assert calls["gate_down"] == calls["gate_up"] == calls["blend_rms_norm"] == 1
        assert "pair_rms_norm" not in calls
    if case[0] in ("local_attention", "global_attention"):
        # One launch for the mix and its norms.
        assert calls["rms_norm"] == 1 + blocks
    used = dict(calls)
    monkeypatch.setenv(SWITCH, "0")
    want = cases.route_outputs(cases.Inputs("cpu", dtype), case, 5, 48, blocks)
    assert calls == used, "the torch route launched a kernel"
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert cases.rel_err(a, b) <= tol, case[0]
