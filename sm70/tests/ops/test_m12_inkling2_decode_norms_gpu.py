# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 decode route of inkling2_sm70/norms.py on an SM70 GPU.

- the gate kernels of ops/fused.py at the checkpoint's sizes (hidden 2304,
  gate rank 64) against the torch GEMMs within 2e-3, the fp16 blend within
  5e-3, and the residual add bit-identical to the torch adds;
- a layer's MLP input and output (``Residual``, with and without a message
  gate) on the kernels against the torch route;
- the route: unset switch, kernels on a decode-only forward only;
- launches: a gated sublayer of one token is four kernels (mix, gate_down,
  gate_up, blend_rms_norm), a residual add one;
- a gated sublayer captured in a CUDA graph replays as it ran eagerly.
CPU interpreter version: test_m12_inkling2_triton_norms_cpu.py.

    pytest -m gpu sm70/tests/ops/test_m12_inkling2_decode_norms_gpu.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

HIDDEN = 2304
GATE_RANK = 64
TOL = 2e-3
# The fp16 blend against a reference rounded once: about 5 fp16 ulps of the max.
BLEND_TOL = 5e-3
SWITCH = "VLLM_SM70_INKLING_TRITON_NORMS"


@pytest.fixture
def cases(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    import _inkling2_norms

    monkeypatch.setenv(SWITCH, "1")
    return _inkling2_norms


def _inputs(cases, seed: int = 0):
    return cases.Inputs("cuda", torch.float16, seed, rank=GATE_RANK)


@pytest.mark.parametrize("tokens", [1, 4, 16])
@pytest.mark.parametrize("streams", [1, 3])
def test_gate_kernels(cases, tokens, streams):
    inputs = _inputs(cases, seed=tokens)
    assert cases.check_gate_kernels(inputs, streams, tokens, HIDDEN) <= TOL
    assert cases.check_blend(inputs, streams, tokens, HIDDEN) <= BLEND_TOL


@pytest.mark.parametrize("tokens", [1, 4, 512])
def test_residual_add_is_exact(cases, tokens):
    assert cases.check_residual_add(_inputs(cases), tokens, HIDDEN) == 0.0


@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("gated", [False, True], ids=["first_pass", "gated"])
@pytest.mark.parametrize("start", [False, True], ids=["inside", "block_start"])
def test_residual_route_matches_the_torch_route(
    cases, monkeypatch, tokens, gated, start
):
    got = cases.residual_outputs(
        _inputs(cases, 3), tokens, HIDDEN, 3, start=start, gated=gated
    )
    monkeypatch.setenv(SWITCH, "0")
    want = cases.residual_outputs(
        _inputs(cases, 3), tokens, HIDDEN, 3, start=start, gated=gated
    )
    assert got[0].is_cuda and cases.rel_err(got[0], want[0]) <= TOL
    assert torch.equal(got[1], want[1])


def test_default_route_is_decode_only(cases, monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import norms

    monkeypatch.delenv(SWITCH)
    x = torch.zeros(4, HIDDEN, device="cuda", dtype=torch.float16)
    assert norms.Bank(x, 1, decode_only=True).triton
    assert not norms.Bank(x, 1).triton
    monkeypatch.setenv(SWITCH, "0")
    assert not norms.Bank(x, 1, decode_only=True).triton


def _gated_sublayer(cases, tokens: int):
    """A forward of one token on the kernels: a bank of three states and
    the MLP sublayer of a gated loop visit."""
    from vllm_sm70.models.inkling2_sm70 import norms

    inputs = _inputs(cases, 5)
    layer_input = inputs.randn(tokens, HIDDEN, scale=2.0)
    attn_out = inputs.randn(tokens, HIDDEN)
    bank = norms.Bank(layer_input, 4, decode_only=True)
    for _ in range(3):
        bank.append(inputs.randn(tokens, HIDDEN))
    query = inputs.randn(1, HIDDEN, dtype=torch.float32)
    weight = inputs.weight(1, HIDDEN)
    (gate,) = inputs.gates(1, HIDDEN)
    gate.gain_weight = gate.gain_weight.half()
    gate.forget_weight = gate.forget_weight.half()
    messages = inputs.randn(4, tokens, HIDDEN, dtype=torch.float32)

    def run():
        residual = norms.Residual(layer_input, attn_out, bank.triton)
        (x,) = norms.sublayer_inputs(
            bank,
            residual,
            query,
            weight,
            cases.EPS,
            torch.float16,
            gates=[gate],
            messages=messages,
            first=3,
        )
        return x

    return run


@pytest.mark.parametrize("fusion", ["1", "0"], ids=["fused", "unfused"])
def test_launches_of_a_gated_sublayer(cases, monkeypatch, fusion):
    monkeypatch.setenv("VLLM_SM70_INKLING_VISIT_FUSION", fusion)
    run = _gated_sublayer(cases, 1)
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        run()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    if fusion == "1":
        assert kernels == ["_attn_res_row_kernel", "_gate_blend_kernel"], kernels
        return
    assert len(kernels) == 4, kernels
    assert {"_gate_down_kernel", "_gate_up_kernel"} <= set(kernels), kernels


def test_gated_sublayer_replays_in_a_cuda_graph(cases):
    run = _gated_sublayer(cases, 4)
    want = run().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
