# SPDX-License-Identifier: Apache-2.0
"""Pre-scaled NVFP4 batch GEMM must equal the ordinary one bit for bit.

Ported from fork cf9492a tests/kernels/quantization/
test_sm70_nvfp4_prescaled_batch.py (a42e887cf). Changes:
- `_prescale_nvfp4_batch_scales` (fork vllm/model_executor/layers/
  quantization/sm70_turbomind.py:485-490, Mara's quant layer) is copied here.
- The fork's env knob VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M=64 is now the default
  (tm_config.h kNvfp4DenseTuneMaxMDefault).
- The QPN2 dispatch test belongs with the QPN2 ops (Rosa) and is not ported.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")


def _prescale_nvfp4_batch_scales(scales: torch.Tensor) -> bool:
    """Fold the exact FP4 conversion factor into the existing scale allocation."""
    if not float(scales.abs().amax()) <= 65504.0 / 16384.0:
        return False
    scales.mul_(16384.0)
    return True


@pytest.mark.parametrize("rows", [33, 40, 48, 49, 56, 64, 65, 128, 1024, 8192])
@pytest.mark.parametrize("gated", [False, True])
def test_nvfp4_prescaled_same_partition_graph(tm_gpu, rows, gated):
    torch.manual_seed(2079)
    k, n = 4352, 4096
    codes = torch.randint(16, (k, n), device="cuda", dtype=torch.uint8)
    values = torch.tensor(
        [0, 2**-24, 2**-14, 0.005, 0.1, 1, 3.998046875],
        device="cuda",
        dtype=torch.float16,
    )
    scales = values[torch.randint(values.numel(), (k // 16, n), device="cuda")]
    weight, packed, meta = torch.ops._sm70_C.nvfp4_sm70_prepare(
        codes, scales, 16, gated
    )
    shifted = packed.clone()
    assert _prescale_nvfp4_batch_scales(shifted)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.01
    reference = torch.empty(rows, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    actual = torch.empty_like(reference)

    def call(out, scale, prescaled):
        op = (
            torch.ops._sm70_C.nvfp4_gemm_sm70_prescaled_out
            if prescaled
            else torch.ops._sm70_C.nvfp4_gemm_sm70_out
        )
        op(out, x, weight, scale, 16, int(meta[0]), int(meta[1]), gated)

    call(reference, packed, False)
    call(actual, shifted, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(actual, shifted, True)
    for amplitude in (0.01, 0.03):
        x.normal_(0, amplitude)
        call(reference, packed, False)
        actual.fill_(float("nan"))
        graph.replay()
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
