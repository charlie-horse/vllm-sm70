// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared includes and fp16 SiLU helpers for the TurboMind op files (K03).
// Moved verbatim from fork cf9492a csrc/sm70_turbomind/ops/awq_sm70_gemm.cu;
// see ../PROVENANCE.md for the fork line range of every function.
#pragma once

// fork awq_sm70_gemm.cu:1-37, adapted: tm_config.h replaces the fork's
// sm70_policy.h; custom_all_reduce.cuh (only for the dropped
// awq_gemm_sm70_out_tile_reduce) and <cutlass/half.h> (unused) are gone.
#include "tm_config.h"
#include "gemm_runtime.h"
#include "packed_gemm.h"
/*
 * SM70 AWQ GEMM integration using TurboMind s884h kernels.
 * Adapted from LMDeploy TurboMind (Apache-2.0).
 */

#include <torch/all.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <atomic>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "qwen38_prefill_cutlass.cuh"

namespace {

__global__ void sm70_silu_and_mul_fp16_kernel(__half* out, const __half* input,
                                              int rows, int d, int input_stride,
                                              int out_stride) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __half* row_in = input + row * input_stride;
  const __half* gate = row_in;
  const __half* up = row_in + d;
  __half* row_out = out + row * out_stride;
  for (int col = threadIdx.x; col < d; col += blockDim.x) {
    const float gate_f = __half2float(gate[col]);
    const __half silu = __float2half(gate_f / (1.0f + expf(-gate_f)));
    row_out[col] = __hmul(silu, up[col]);
  }
}

void sm70_silu_and_mul_fp16_out(torch::Tensor out, torch::Tensor input) {
  const int rows = static_cast<int>(out.size(0));
  const int d = static_cast<int>(out.size(1));
  const int input_stride = static_cast<int>(input.stride(0));
  const int out_stride = static_cast<int>(out.stride(0));
  constexpr int kThreads = 256;
  sm70_silu_and_mul_fp16_kernel<<<rows, kThreads, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()), rows, d,
      input_stride, out_stride);
}

__global__ void sm70_silu_and_mul_interleaved_fp16_kernel(__half* out,
                                                          const __half* input,
                                                          int rows, int d,
                                                          int input_stride,
                                                          int out_stride) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __half* row_in = input + row * input_stride;
  __half* row_out = out + row * out_stride;
  for (int col = threadIdx.x; col < d; col += blockDim.x) {
    const int gate_col = col * 2;
    const float gate_f = __half2float(row_in[gate_col]);
    const __half silu = __float2half(gate_f / (1.0f + expf(-gate_f)));
    row_out[col] = __hmul(silu, row_in[gate_col + 1]);
  }
}

void sm70_silu_and_mul_interleaved_fp16_out(torch::Tensor out,
                                            torch::Tensor input) {
  const int rows = static_cast<int>(out.size(0));
  const int d = static_cast<int>(out.size(1));
  const int input_stride = static_cast<int>(input.stride(0));
  const int out_stride = static_cast<int>(out.stride(0));
  constexpr int kThreads = 256;
  sm70_silu_and_mul_interleaved_fp16_kernel<<<
      rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()), rows, d,
      input_stride, out_stride);
}

}  // namespace

namespace vllm::awq_sm70 {
// Defined in tm_f16_sm70.cu; used by the fp8 and nvfp4 prepare ops.
torch::Tensor interleave_gated_silu_cols(torch::Tensor tensor);
}  // namespace vllm::awq_sm70
