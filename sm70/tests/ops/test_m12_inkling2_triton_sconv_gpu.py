# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 paged short-conv kernels (inkling2_sm70/ops/paged_conv.py)
on an SM70 GPU.

The scenarios of test_m12_inkling2_triton_sconv_cpu.py (_inkling2_sconv.py),
compiled for sm_70, against the torch route on the same GPU: fp16 within
2e-3 and fp32 within 1e-5 of the largest reference value, outputs and
caches. The kernels run with CUDA sync debugging at "error", and a short
conv plus a prologue K/V pair captured in a CUDA graph replay a later
decode step with the metadata updated in place as the eager kernels do.

    pytest -m gpu sm70/tests/ops/test_m12_inkling2_triton_sconv_gpu.py
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

SWITCH = "VLLM_SM70_INKLING_TRITON_SCONV"
DTYPES = [
    pytest.param(torch.float32, 1e-5, id="fp32"),
    pytest.param(torch.float16, 2e-3, id="fp16"),
]


@pytest.fixture
def cases(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    import _inkling2_sconv

    from vllm_sm70.models.inkling2_sm70 import sconv

    monkeypatch.delenv(SWITCH, raising=False)
    assert sconv.on_triton(torch.zeros(1, device="cuda"))
    return _inkling2_sconv


def _routes(cases, monkeypatch, dtype, dynamic, steps, **kwargs):
    monkeypatch.delenv(SWITCH, raising=False)
    got = cases.run(cases.Weights("cuda", dtype, dynamic), steps, **kwargs)
    monkeypatch.setenv(SWITCH, "0")
    want = cases.run(cases.Weights("cuda", dtype, dynamic), steps, **kwargs)
    monkeypatch.delenv(SWITCH, raising=False)
    return got, want


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    "scenario", ["prefill", "decode", "boundary", "short", "mixed"]
)
@pytest.mark.parametrize("pad", [0, 3])
def test_triton_route_matches_the_torch_route(
    cases, monkeypatch, dtype, tol, dynamic, scenario, pad
):
    got, want = _routes(
        cases, monkeypatch, dtype, dynamic, cases.SCENARIOS[scenario], pad=pad
    )
    for i, (a, b) in enumerate(zip(got, want)):
        assert cases.rel_err(a, b) <= tol, i


def _decode_inputs(cases, weights, n):
    t = weights.randn
    kv = cases.KV_HEADS * cases.HEAD_DIM
    return {
        "x": t(n, cases.HIDDEN),
        "k": t(n, kv),
        "v": t(n, kv),
        "k_taps": t(n, cases.KV_HEADS * cases.TAPS),
        "v_taps": t(n, cases.KV_HEADS * cases.TAPS),
    }


def _launch(cases, weights, paged, metadata, inputs, out):
    """The kernels of one decode step: a static short conv and a dynamic
    K/V pair."""
    from vllm_sm70.models.inkling2_sm70.ops import paged_conv

    conv, attn = metadata["conv"], metadata["attn"]
    paged_conv.short_conv(
        inputs["x"], weights.out_conv[False], None, paged.conv_cache, conv,
        paged.stream_ranges[2], conv.num_actual_tokens, out,
    )  # fmt: skip
    paged_conv.key_value_conv(
        inputs["k"], inputs["v"], weights.k_conv, weights.v_conv,
        inputs["k_taps"], inputs["v_taps"], paged.conv_cache, conv,
        paged.stream_ranges[0], paged.stream_ranges[1], paged.kv_cache,
        attn.slot_mapping, conv.num_actual_tokens, qk_clip=cases.QK_CLIP,
        eps=cases.EPS,
    )  # fmt: skip


def test_kernels_never_sync_the_host(cases):
    weights = cases.Weights("cuda", torch.float16, True)
    paged = cases.Paged("cuda", torch.float16)
    paged.batch([(0, 40), (1, 3)])
    metadata, n = paged.batch([(0, 1), (1, 1), (2, 5)], pad=3)
    inputs = _decode_inputs(cases, weights, n + 3)
    out = torch.empty_like(inputs["x"])
    _launch(cases, weights, paged, metadata, inputs, out)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _launch(cases, weights, paged, metadata, inputs, out)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_graph_replay_matches_eager_launches(cases):
    weights = cases.Weights("cuda", torch.float16, True)
    paged = cases.Paged("cuda", torch.float16)
    # Finite history in place of a prefill run.
    paged.conv_cache.normal_()
    paged.kv_cache.zero_()
    paged.batch([(0, 40), (1, 33)])
    decode = [(0, 1), (1, 1)]
    steps = [paged.batch(decode)[0] for _ in range(3)]
    eager = cases.Paged("cuda", torch.float16)
    eager.conv_cache.copy_(paged.conv_cache)
    eager.kv_cache.copy_(paged.kv_cache)
    eager.conv.kv_cache, eager.attn.kv_cache = eager.conv_cache, eager.kv_cache
    inputs = [_decode_inputs(cases, weights, 2) for _ in steps]
    static = {name: t.clone() for name, t in inputs[0].items()}
    meta = {
        kind: type(steps[0][kind])(**{
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in vars(steps[0][kind]).items()
        })
        for kind in ("conv", "attn")
    }  # fmt: skip
    out = torch.empty_like(static["x"])
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Warm-up compiles the kernels; it writes slots that step 0 rewrites.
        _launch(cases, weights, paged, meta, static, out)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        _launch(cases, weights, paged, meta, static, out)
    for step, step_inputs in zip(steps, inputs):
        for kind in ("conv", "attn"):
            for key, value in vars(step[kind]).items():
                if torch.is_tensor(value):
                    getattr(meta[kind], key).copy_(value)
        for name, t in step_inputs.items():
            static[name].copy_(t)
        graph.replay()
        want = torch.empty_like(out)
        _launch(cases, weights, eager, step, step_inputs, want)
        torch.cuda.synchronize()
        assert torch.equal(out, want)
    assert torch.equal(paged.conv_cache, eager.conv_cache)
    assert torch.equal(paged.kv_cache, eager.kv_cache)
