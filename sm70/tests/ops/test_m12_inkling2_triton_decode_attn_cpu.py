# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 paged decode-attention kernel
(inkling2_sm70/ops/paged_attention.py) and its route (attention_torch.py
``gather_attention``), on CPU.

- the kill switch VLLM_SM70_INKLING_TRITON_DECODE_ATTN and the route
  choice: on by default for GPU tensors, torch off the GPU and with
  ``hf_rounding``;
- the kernel source holds nothing sm_70 rejects;
- through Triton's interpreter against the torch route, in fp32 within 1e-5
  and in fp16 within 2e-3: decode-only batches with padding rows, eager
  batches with one-query rows before and after a prefill, global and
  sliding layers, contexts across many 32-token blocks, a bias extent
  longer than a block, GQA, a head dim that is not a power of 2. Every slot
  a row may not read (null block, unwritten slots, keys out of the window)
  holds NaN.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent they show as skipped. The AOT
compile check is test_m12_inkling2_triton_sm70_compile_cpu.py.
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
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import _inkling2_decode_attn as cases


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
SWITCH = cases.SWITCH
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
DTYPES = [
    pytest.param(torch.float32, 1e-5, id="fp32"),
    pytest.param(torch.float16, 2e-3, id="fp16"),
]
LAYERS = pytest.mark.parametrize(
    "window, extent",
    [(None, 8), (None, 48), (40, 8), (64, 48)],
    ids=["global", "global-long-bias", "sliding", "sliding-long-bias"],
)


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
    assert passed and int(passed.group(1)) >= 30, summary
    assert "skipped" not in summary, summary


# --- source, switch and route ---------------------------------------------------


def test_kernel_source_holds_nothing_sm70_rejects():
    text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / "paged_attention.py").read_text())
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    for banned in ("tl.dot", "tanh", "inline_asm", "gdc_", "launch_pdl", "bfloat16"):
        assert banned not in code, banned


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_kernel_off(monkeypatch, value, enabled):
    attention = cases.attention_module()
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    monkeypatch.setattr(attention._compat, "INTERPRET", False)
    monkeypatch.setattr(attention._compat, "AVAILABLE", True)
    assert attention.on_triton(SimpleNamespace(is_cuda=True)) is enabled
    assert attention.on_triton(SimpleNamespace(is_cuda=False)) is False


def test_torch_route_off_the_gpu_launches_no_kernel(monkeypatch):
    attention = cases.attention_module()
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(attention._compat, "INTERPRET", False)

    def launched(*args, **kwargs):
        raise AssertionError("kernel launched off the GPU")

    monkeypatch.setattr(attention, "decode_attention", launched)
    for decode_only in (True, False):
        case = cases.Case(
            torch.float32, None, 8, [70, 5], pad=2, decode_only=decode_only
        )
        case.run(attention)


# --- the cases (_inkling2_decode_attn.py) ---------------------------------------


@interpreted
@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@LAYERS
@pytest.mark.parametrize(
    "seq_lens, pad",
    [([1], 0), ([33], 3), ([7, 70, 32, 160], 0), ([300, 2, 64], 5)],
    ids=["first-token", "one-padded", "four", "long-padded"],
)
def test_decode_only_batch_matches_the_torch_route(
    monkeypatch, dtype, tol, window, extent, seq_lens, pad
):
    case = cases.Case(dtype, window, extent, seq_lens, pad=pad)
    got, want = cases.both_routes(case, monkeypatch)
    assert cases.rel_err(got, want) <= tol
    assert bool((got[len(seq_lens) :] == 0).all())


@interpreted
@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@LAYERS
@pytest.mark.parametrize(
    "seq_lens, query_lens",
    [([90, 40, 50], [1, 1, 20]), ([90, 50, 40], [1, 20, 1]), ([129, 5], [1, 1])],
    ids=["decode-first", "decode-after-prefill", "all-decode"],
)
def test_eager_one_query_rows_match_the_torch_route(
    monkeypatch, dtype, tol, window, extent, seq_lens, query_lens
):
    case = cases.Case(
        dtype, window, extent, seq_lens, pad=2, decode_only=False,
        query_lens=query_lens, seed=1,
    )  # fmt: skip
    got, want = cases.both_routes(case, monkeypatch)
    assert cases.rel_err(got, want) <= tol
    assert bool((got[sum(query_lens) :] == 0).all())


@interpreted
def test_hf_rounding_stays_on_torch(monkeypatch):
    attention = cases.attention_module()

    def launched(*args, **kwargs):
        raise AssertionError("kernel launched with hf_rounding")

    monkeypatch.setattr(attention, "decode_attention", launched)
    cases.Case(torch.float32, None, 8, [40, 9]).run(attention, hf_rounding=True)
