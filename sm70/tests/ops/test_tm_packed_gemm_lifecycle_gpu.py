# SPDX-License-Identifier: Apache-2.0
"""Packed dense/grouped views keep their strides and live replay inputs.

Ported from fork cf9492a tests/kernels/quantization/
test_sm70_packed_gemm_lifecycle.py (156ec4deb). Changes: the native-policy
bindings (NativeBindings, Sm70NativeConfig, KernelConfig, runtime release)
are gone, so the ops are called through vllm_sm70.ops.tm. AWQ and MXFP4 are
not in K03 (AWQ moves to K04; MXFP4 is dropped).
"""

from __future__ import annotations

import hashlib

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")

FORMATS = ("fp8", "nvfp4")
K, N = 512, 256


def prepare_weight(tm, fmt, gated=False):
    group = {"fp8": 128, "nvfp4": 16}[fmt]
    if fmt == "fp8":
        weight = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.rand(N // 128, K // 128, device="cuda") * 0.01
        packed, scales, meta = tm.fp8_sm70_prepare(weight, scale, group, gated)
    else:
        scale = torch.rand(K // group, N, device="cuda", dtype=torch.float16) * 0.01
        weight = torch.randint(16, (K, N), device="cuda", dtype=torch.uint8)
        packed, scales, meta = tm.nvfp4_sm70_prepare(weight, scale, group, gated)
    return packed, scales, group, int(meta[0]), int(meta[1])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("gated", [False, True])
def test_dense_dynamic_rows_output_stride_and_replay(
    tm_gpu, fmt, gated, record_property
):
    torch.manual_seed(101)
    digest = hashlib.sha256()
    weight, scales, group, k_ld, q_ld = prepare_weight(tm_gpu, fmt, gated)
    operation = getattr(tm_gpu, f"{fmt}_gemm_sm70_out")
    width = N // 2 if gated else N
    for rows in (1, 8, 32, 33, 64):
        # Inputs are row views with LD = K + 128.
        storage = torch.randn(rows, K + 128, device="cuda", dtype=torch.float16)
        x = storage[:, :K]
        buffers = [
            torch.full((rows, width + 16), 42, device="cuda", dtype=x.dtype)
            for _ in range(2)
        ]
        expected, actual = [buffer[:, :width] for buffer in buffers]

        def run(out, x=x):
            operation(out, x, weight, scales, group, k_ld, q_ld, gated)

        run(actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(actual)
        for amplitude in (0.01, 0.2):
            x.normal_(0, amplitude)
            run(expected)
            graph.replay()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert all(torch.all(buffer[:, width:] == 42) for buffer in buffers)
            digest.update(actual.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        graph.reset()
    with pytest.raises(RuntimeError, match="input must be float16"):
        operation(actual, x.float(), weight, scales, group, k_ld, q_ld, gated)
    record_property("output_sha256", digest.hexdigest())


@pytest.mark.parametrize("fmt", FORMATS)
def test_grouped_replay_updates_experts_and_empty_segments(
    tm_gpu, fmt, record_property
):
    torch.manual_seed(107)
    digest = hashlib.sha256()
    prepared = [prepare_weight(tm_gpu, fmt) for _ in range(3)]
    weights = torch.stack([row[0] for row in prepared])
    scales = torch.stack([row[1] for row in prepared])
    _, _, group, k_ld, q_ld = prepared[0]
    ptrs = tm_gpu.awq_moe_build_strided_ptrs(weights, scales, k_ld, q_ld, 3)
    x = torch.randn(33, K + 128, device="cuda", dtype=torch.float16)[:, :K]
    offsets = torch.tensor([0, 19, 19, 33], device="cuda", dtype=torch.int32)
    experts = torch.tensor([2, 0, 2], device="cuda", dtype=torch.int32)
    expected, actual = [
        torch.empty(33, N, device="cuda", dtype=x.dtype) for _ in range(2)
    ]
    operation = getattr(tm_gpu, f"{fmt}_moe_dense_stage_sm70_out")

    def run(out):
        operation(out, x, offsets, experts, *ptrs, 3, K, N, group)

    run(actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(actual)
    for ids, lengths in (([0, 2, 1], [0, 11, 11, 33]), ([1, 2, 0], [0, 0, 20, 33])):
        experts.copy_(torch.tensor(ids, device="cuda", dtype=experts.dtype))
        offsets.copy_(torch.tensor(lengths, device="cuda", dtype=offsets.dtype))
        x.normal_(0, 0.05)
        run(expected)
        graph.replay()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        digest.update(actual.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    graph.reset()
    record_property("output_sha256", digest.hexdigest())
