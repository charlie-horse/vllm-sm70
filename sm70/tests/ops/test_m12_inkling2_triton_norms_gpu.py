# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 vendored Triton norm kernels on an SM70 GPU.

Each kernel of inkling2_sm70/ops/norm.py, attn_res.py and loop.py, compiled
for sm_70, on fp16 inputs at the checkpoint's hidden size against the torch
reference ops (ops/torch_ref.py) within 2e-3 of the largest reference value;
the route of inkling2_sm70/norms.py against its torch route; PDL off. The
cases are those of test_m12_inkling2_triton_norms_cpu.py
(_inkling2_norms.py), which runs them through Triton's interpreter.

    pytest -m gpu sm70/tests/ops/test_m12_inkling2_triton_norms_gpu.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

HIDDEN = 2304
TOL = 2e-3
SWITCH = "VLLM_SM70_INKLING_TRITON_NORMS"
TOKENS = [1, 64, 512]


@pytest.fixture
def cases(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    import _inkling2_norms

    from vllm_sm70.models.inkling2_sm70 import norms

    monkeypatch.setenv(SWITCH, "1")
    assert norms.on_triton(torch.zeros(1, device="cuda"))
    return _inkling2_norms


def _inputs(cases, seed: int = 0):
    return cases.Inputs("cuda", torch.float16, seed)


def test_pdl_is_off_on_sm70(cases):
    from vllm_sm70.models.inkling2_sm70.ops import _compat

    assert _compat.use_gdc() is False and not _compat.INTERPRET


@pytest.mark.parametrize("tokens", TOKENS)
def test_rms_norm_kernel(cases, tokens):
    assert cases.check_rms_norm(_inputs(cases), tokens, HIDDEN) <= TOL


@pytest.mark.parametrize("stack_normed", [False, True], ids=["raw", "normed"])
@pytest.mark.parametrize("norm", [False, True], ids=["mix", "mix_norm"])
@pytest.mark.parametrize("addend", [False, True], ids=["last", "last_addend"])
@pytest.mark.parametrize(
    ("tokens", "blocks", "queries"),
    [(1, 0, 1), (1, 4, 3), (64, 2, 1), (64, 4, 4), (512, 0, 3), (512, 4, 1)],
)
def test_attn_res_kernels(cases, stack_normed, norm, addend, tokens, blocks, queries):
    err = cases.check_attn_res(
        _inputs(cases, seed=tokens + blocks),
        tokens,
        HIDDEN,
        blocks,
        queries,
        norm=norm,
        addend=addend,
        stack_normed=stack_normed,
    )
    assert err <= TOL


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("streams", [1, 3])
def test_loop_kernels(cases, tokens, streams):
    inputs = _inputs(cases, seed=streams)
    assert cases.check_pair_rms_norm(inputs, streams, tokens, HIDDEN) <= TOL
    assert cases.check_blend(inputs, streams, tokens, HIDDEN) <= TOL


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("case", range(8))
def test_triton_route_matches_the_torch_route(cases, monkeypatch, tokens, case):
    case = cases.ROUTE_CASES[case]
    got = cases.route_outputs(_inputs(cases), case, tokens, HIDDEN, 3)
    monkeypatch.setenv(SWITCH, "0")
    want = cases.route_outputs(_inputs(cases), case, tokens, HIDDEN, 3)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.is_cuda and cases.rel_err(a, b) <= TOL, case[0]
