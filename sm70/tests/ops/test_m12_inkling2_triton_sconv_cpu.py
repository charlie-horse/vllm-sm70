# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 paged short-conv kernels (inkling2_sm70/ops/paged_conv.py)
and their route (sconv.py ``short_conv``, prologue.py
``attention_prologue``), on CPU.

- the kill switch VLLM_SM70_INKLING_TRITON_SCONV and the route choice: on by
  default for GPU tensors, torch off the GPU;
- the kernel source holds nothing sm_70 rejects;
- through Triton's interpreter against the torch route, in fp32 within 1e-5
  and in fp16 within 2e-3: pure prefill, pure decode, mixed batches,
  positions across the 32-token block boundary, requests shorter than the
  conv width, padded and unpadded metadata, a scheduled row without a slot,
  the LBNHC cache view, static and dynamic convs; the conv-state and K/V
  caches after every scenario;
- one kernel launch per short conv and per prologue K/V pair, and no host
  sync or data-dependent op on the route;
- the tiny model through the interpreted kernels against the torch route.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent they show as skipped. The GPU
version is test_m12_inkling2_triton_sconv_gpu.py.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import types

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
OPS = HERE.parents[1] / "vllm_sm70" / "models" / "inkling2_sm70" / "ops"
SWITCH = "VLLM_SM70_INKLING_TRITON_SCONV"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
FP32_TOL = 1e-5
FP16_TOL = 2e-3
DTYPES = [
    pytest.param(torch.float32, FP32_TOL, id="fp32"),
    pytest.param(torch.float16, FP16_TOL, id="fp16"),
]
CONVS = pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
# Ops that wait for the device or whose result shape depends on data.
SYNC_OPS = {
    "nonzero",
    "_local_scalar_dense",
    "is_nonzero",
    "masked_select",
    "unique",
    "_unique2",
    "bincount",
}


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


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
        timeout=3600,
        check=False,
    )
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 40, summary
    assert "skipped" not in summary, summary


# --- source, switch and route ---------------------------------------------------


def _code(path: pathlib.Path) -> str:
    text = re.sub(r'"""(?:.|\n)*?"""', "", path.read_text(encoding="utf-8"))
    return "\n".join(line.split("#")[0].rstrip() for line in text.splitlines())


def test_kernel_source_holds_nothing_sm70_rejects():
    code = _code(OPS / "paged_conv.py")
    for banned in (
        "tanh",
        "tl.dot",
        "bfloat16",
        "launch_pdl",
        "gdc_",
        "inline_asm",
        "tldevice",
        "vllm.triton_utils",
    ):
        assert banned not in code, banned
    assert not re.search(r"sm100|sm120|sm90|vllm\.models\b", code, re.IGNORECASE)
    # Nothing on the host reads device data: no .item(), .tolist(), nonzero.
    for banned in (".item(", ".tolist(", "nonzero", ".cpu(", "int(metadata"):
        assert banned not in code, banned


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, True), ("1", True), ("", True), ("0", False), ("off", True)],
)
def test_kill_switch_only_zero_turns_the_kernels_off(monkeypatch, value, enabled):
    from vllm_sm70 import envs
    from vllm_sm70.models.inkling2_sm70 import sconv
    from vllm_sm70.models.inkling2_sm70.ops import _compat

    assert sconv.SWITCH == SWITCH and SWITCH in envs._SECTIONS["models"]
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert getattr(envs, SWITCH) is enabled
    assert sconv.on_triton(torch.zeros(2, 8)) is (INTERPRETED and enabled)
    gpu = type("Gpu", (), {"is_cuda": True})()
    monkeypatch.setattr(_compat, "AVAILABLE", True)
    assert sconv.on_triton(gpu) is enabled
    monkeypatch.setattr(_compat, "AVAILABLE", False)
    assert sconv.on_triton(gpu) is (INTERPRETED and enabled)


class _Ops:
    """Records the aten ops that run, without views."""

    VIEWS = frozenset(
        {
            "view",
            "_unsafe_view",
            "reshape",
            "slice",
            "select",
            "as_strided",
            "t",
            "transpose",
            "unsqueeze",
            "squeeze",
            "expand",
            "detach",
            "alias",
            "split",
            "split_with_sizes",
            "unflatten",
            "permute",
        }
    )

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        names = self.names = []

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = func.overloadpacket.__name__
                if name not in _Ops.VIEWS:
                    names.append(name)
                return func(*args, **(kwargs or {}))

        self.mode = Mode()


def _counted_kernels(monkeypatch, stub: bool):
    """Count the launches of the two kernels; `stub` skips running them."""
    from vllm_sm70.models.inkling2_sm70.ops import paged_conv

    launches: dict[str, int] = {}

    class Counted:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches[self.name] = launches.get(self.name, 0) + 1
                if not stub:
                    self.kernel[grid](*args, **kwargs)

            return launch

    for name in ("_short_conv_kernel", "_key_value_conv_kernel"):
        kernel = getattr(paged_conv, name)
        monkeypatch.setattr(paged_conv, name, Counted(name, kernel))
    return launches


def _one_step(cases, dynamic: bool, step, pad: int = 0):
    """Ops of each call of one batch: (prologue, static conv, dynamic conv)."""
    from vllm_sm70.models.inkling2_sm70 import prologue, sconv

    weights = cases.Weights("cpu", torch.float32, dynamic)
    paged = cases.Paged("cpu", torch.float32)
    metadata, n = paged.batch(step, pad)
    total = n + pad
    t = weights.randn
    q, r = (
        t(total, cases.Q_HEADS * cases.HEAD_DIM),
        t(total, cases.Q_HEADS * cases.D_REL),
    )
    k, v = (
        t(total, cases.KV_HEADS * cases.HEAD_DIM),
        t(total, cases.KV_HEADS * cases.HEAD_DIM),
    )
    taps = (
        t(total, cases.KV_HEADS * cases.TAPS),
        t(total, cases.KV_HEADS * cases.TAPS),
    )
    if not dynamic:
        taps = (None, None)
    q_out = t(total, cases.Q_HEADS, cases.HEAD_DIM)
    rel_out = t(total, cases.Q_HEADS, cases.EXTENT)
    x, out = t(total, cases.HIDDEN), t(total, cases.HIDDEN)
    dyn = t(total, cases.KV_HEADS, cases.TAPS)
    calls = []
    with paged.forward(metadata):
        for call in (
            lambda: prologue.attention_prologue(
                q, r, k, v, weights.q_scale, weights.rel_scale, weights.rel_proj,
                weights.k_conv, weights.v_conv, *taps, paged.attn, paged.conv,
                q_out, rel_out, qk_clip=cases.QK_CLIP, rel_clip=cases.REL_CLIP,
                eps=cases.EPS, log_scaling=(0.1, 16.0) if dynamic else None,
            ),
            lambda: sconv.short_conv(x, weights.out_conv[False], paged.conv, 2, out),
            lambda: sconv.short_conv(
                x, weights.out_conv[True], paged.conv, 3, out, dynamic_taps=dyn
            ),
        ):  # fmt: skip
            ops = _Ops()
            with ops.mode:
                call()
            calls.append(ops.names)
    return calls


def test_torch_route_off_the_gpu_launches_no_kernel(monkeypatch):
    """The route on CPU tensors, and with the switch at 0 in the
    interpreter: the torch ops, with their host syncs."""
    import _inkling2_sconv as cases

    if INTERPRETED:
        monkeypatch.setenv(SWITCH, "0")
    launches = _counted_kernels(monkeypatch, stub=True)
    prologue_ops, static_ops, dynamic_ops = _one_step(cases, True, [(0, 3), (1, 2)])
    assert launches == {}
    for ops in (prologue_ops, static_ops, dynamic_ops):
        assert "nonzero" in ops


# --- interpreted kernels ----------------------------------------------------------


@pytest.fixture
def cases():
    if not INTERPRETED:
        pytest.skip("runs in the TRITON_INTERPRET=1 child (test_interpreted_suite)")
    import _inkling2_sconv

    return _inkling2_sconv


def _routes(cases, monkeypatch, dtype, dynamic, steps, **kwargs):
    """(Triton route, torch route) outputs of `steps`; checks the launches."""
    launches = _counted_kernels(monkeypatch, stub=False)
    monkeypatch.delenv(SWITCH, raising=False)
    got = cases.run(cases.Weights("cpu", dtype, dynamic), steps, **kwargs)
    assert launches == {
        "_short_conv_kernel": 2 * len(steps),
        "_key_value_conv_kernel": len(steps),
    }
    monkeypatch.setenv(SWITCH, "0")
    want = cases.run(cases.Weights("cpu", dtype, dynamic), steps, **kwargs)
    assert launches["_key_value_conv_kernel"] == len(steps), "torch route launched"
    return got, want


def _check(cases, got, want, tol):
    names = ["q_out", "rel_out", "attn_out", "mlp_out"]
    for i, (a, b) in enumerate(zip(got, want)):
        name = names[i % 4] if i < len(got) - 2 else ("conv cache", "kv cache")[i % 2]
        err = cases.rel_err(a, b)
        assert err <= tol, (i // 4, name, err)


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@CONVS
@pytest.mark.parametrize(
    "scenario", ["prefill", "decode", "boundary", "short", "mixed"]
)
def test_triton_route_matches_the_torch_route(
    cases, monkeypatch, dtype, tol, dynamic, scenario
):
    steps = cases.SCENARIOS[scenario]
    got, want = _routes(cases, monkeypatch, dtype, dynamic, steps)
    _check(cases, got, want, tol)


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@CONVS
@pytest.mark.parametrize(
    ("pad", "pad_metadata", "hole"),
    [(3, True, False), (3, False, False), (0, True, True), (2, True, True)],
    ids=["padded", "unpadded", "hole", "padded_hole"],
)
@pytest.mark.parametrize("scenario", ["decode", "mixed"])
def test_padding_rows_match_the_torch_route(
    cases, monkeypatch, dtype, tol, dynamic, pad, pad_metadata, hole, scenario
):
    steps = cases.SCENARIOS[scenario]
    got, want = _routes(
        cases, monkeypatch, dtype, dynamic, steps,
        pad=pad, pad_metadata=pad_metadata, hole=hole,
    )  # fmt: skip
    _check(cases, got, want, tol)
    if pad:
        for out in got[2:-2:4] + got[3:-2:4]:
            assert not out[-pad:].any()


@CONVS
def test_lbnhc_cache_view_matches_the_torch_route(cases, monkeypatch, dynamic):
    steps = cases.SCENARIOS["mixed"]
    got, want = _routes(
        cases, monkeypatch, torch.float32, dynamic, steps, lbnhc=True, pad=1
    )
    assert got[-2].stride(1) == cases.Paged("cpu", torch.float32, lbnhc=True).head_size
    _check(cases, got, want, FP32_TOL)


def test_non_finite_row_of_another_request_stays_out_of_the_taps(cases, monkeypatch):
    """Request 0 has NaN inputs; request 1's outputs stay finite, in its
    prefill and in its decode that reads its taps from the cache."""
    from vllm_sm70.models.inkling2_sm70 import sconv

    monkeypatch.delenv(SWITCH, raising=False)
    paged = cases.Paged("cpu", torch.float32)
    weights = cases.Weights("cpu", torch.float32, False)
    for step in ([(0, 3), (1, 4)], [(0, 1), (1, 1)]):
        metadata, n = paged.batch(step)
        x = weights.randn(n, cases.HIDDEN)
        first = step[0][1]
        x[:first] = float("nan")
        out = torch.empty_like(x)
        with paged.forward(metadata):
            sconv.short_conv(x, weights.out_conv[False], paged.conv, 2, out)
        assert out[:first].isnan().all() and torch.isfinite(out[first:]).all()


@CONVS
@pytest.mark.parametrize(
    ("step", "pad"),
    [([(0, 1), (1, 1), (2, 1)], 0), ([(0, 1), (1, 1), (2, 5)], 3)],
    ids=["decode", "mixed_padded"],
)
def test_one_launch_and_no_host_sync_per_call(cases, monkeypatch, dynamic, step, pad):
    """With the kernels stubbed, a short conv runs no aten op at all and the
    prologue only the torch q/r ops: no sync, no data-dependent op."""
    monkeypatch.delenv(SWITCH, raising=False)
    launches = _counted_kernels(monkeypatch, stub=True)
    prologue_ops, static_ops, dynamic_ops = _one_step(cases, dynamic, step, pad)
    assert launches == {"_short_conv_kernel": 2, "_key_value_conv_kernel": 1}
    assert static_ops == [] and dynamic_ops == []
    assert not SYNC_OPS & set(prologue_ops), prologue_ops
    assert not {"index", "index_put", "index_put_"} & set(prologue_ops)


# --- the model --------------------------------------------------------------------


def test_model_on_interpreted_kernels_matches_the_torch_route(cases, monkeypatch):
    """The tiny model's prefill and two decode steps, every visit's state
    and the logits, on the kernels and on the torch convs."""
    sys.path.insert(0, str(HERE.parent / "models"))
    pytest.importorskip("safetensors")
    import inkling2_harness as hx
    import test_m12_inkling2_model_cpu as body

    length, prompt = 7, 5

    def run():
        generator = torch.Generator().manual_seed(5)
        vocab = body._raw_config()["text_config"]["unpadded_vocab_size"]
        ids = torch.randint(0, vocab, (length,), generator=generator)
        return engine.run(hx.decode_plan("a", length, prompt), {"a": ids})["a"]

    engine = body.Engine()
    launches = _counted_kernels(monkeypatch, stub=False)
    monkeypatch.delenv(SWITCH, raising=False)
    got = run()
    steps = 1 + length - prompt
    visits = len(engine.model.model.visits)
    assert launches == {
        "_short_conv_kernel": 2 * visits * steps,
        "_key_value_conv_kernel": visits * steps,
    }
    monkeypatch.setenv(SWITCH, "0")
    want = run()
    assert launches["_key_value_conv_kernel"] == visits * steps
    assert got.shape == want.shape
    assert float((got - want).abs().max()) <= 1e-4
