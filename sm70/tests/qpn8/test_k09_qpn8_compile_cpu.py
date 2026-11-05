# SPDX-License-Identifier: Apache-2.0
"""K09: the shared dense scratch survives torch.compile without a clone.

`qpn8_native_linear` declares `dense_scratch` as `Tensor(a!)?` (mutated in
place). Functionalization turns that into auto_functionalized_v2 plus a
copy_ back to the input; Inductor's reinplacing must then hand the original
storage to the op, so no per-call clone or copy of a K x N fp16 tensor
appears. Checked on CPU with Python stand-ins for the CUDA kernels, for the
scratch as a module buffer and as a graph input: the op must see the
caller's data_ptr, and the generated code must not clone or copy it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
run_and_get_code = pytest.importorskip("torch._inductor.utils").run_and_get_code

K, N, M = 256, 128, 40  # M > QPN8_MAX_M: the dequantize-into-scratch branch


@pytest.fixture
def seen(monkeypatch, qpn8_stub):
    ptrs = []

    def dequant(out, codes, scales):
        ptrs.append(out.data_ptr())
        out.fill_(0.5)

    monkeypatch.setattr(qpn8_stub, "fp8_qpn8_dequantize_sm70_out", dequant)
    monkeypatch.setattr(
        qpn8_stub, "fp8_qpn8_gemm_sm70_out", lambda out, *a: out.fill_(1)
    )
    assert qpn8_stub.QPN8_MAX_M < M
    torch._dynamo.reset()
    yield ptrs
    torch._dynamo.reset()


def _no_clone(code: list[str]) -> None:
    src = "\n".join(code)
    assert "qpn8_native_linear" in src
    for line in src.splitlines():
        if "clone" in line or "copy_" in line:
            pytest.fail(f"generated code copies a tensor: {line.strip()}")


def _operands():
    return (
        torch.zeros(K, N, dtype=torch.uint8),
        torch.zeros(K // 128, N // 32, dtype=torch.float16),
    )


def test_module_buffer_scratch_is_not_cloned(seen):
    codes, scales = _operands()

    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("codes", codes)
            self.register_buffer("scales", scales)
            self.register_buffer("scratch", torch.zeros(K * N, dtype=torch.float16))

        def forward(self, x):
            y = torch.ops.vllm_sm70.qpn8_native_linear(
                x, self.codes, self.scales, 16, 2, False, None, None, 0, 0, self.scratch
            )
            return y * 2

    mod = Lin()
    compiled = torch.compile(mod, fullgraph=True, dynamic=True)
    y, code = run_and_get_code(compiled, torch.ones(M, K, dtype=torch.float16))
    assert seen and seen[-1] == mod.scratch.data_ptr()
    assert bool((mod.scratch == 0.5).all())
    assert y.shape == (M, N)
    _no_clone(code)


def test_graph_input_scratch_is_not_cloned(seen):
    codes, scales = _operands()
    scratch = torch.zeros(K * N, dtype=torch.float16)

    def fn(x, scratch):
        y = torch.ops.vllm_sm70.qpn8_native_linear(
            x, codes, scales, 16, 2, False, None, None, 0, 0, scratch
        )
        return y + 1

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    _, code = run_and_get_code(compiled, torch.ones(M, K, dtype=torch.float16), scratch)
    assert seen and seen[-1] == scratch.data_ptr()
    _no_clone(code)
