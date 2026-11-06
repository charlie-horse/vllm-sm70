// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The M8 execution layout follows the in-tree QPN8 kernels, derived from
// dnv2003/v100-skinny (MIT). See LICENSE.v100-skinny for the retained notice.
// Weights stay FP16; the independent K partitions accumulate in FP32.
//
// K18: ported from fork cf9492a csrc/sm70_turbomind/ops/dflash2_fp16_m8_sm70.cu
// (see PROVENANCE.md). Ops register in the _sm70_C namespace.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/mm.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cstdint>

#define MMA(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

namespace {

template <int Warps, int Tile>
__global__ void dflash2_fp16_m8_kernel(half* __restrict__ output,
                                       const half* __restrict__ input,
                                       const half* __restrict__ weight, int n,
                                       int k) {
  constexpr int Split = Warps * (Tile == 16 ? 2 : 1);
  __shared__ float partials[Split][8 * Tile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = (lane >> 2) & 3;
  const int split = warp * (Tile == 16 ? 2 : 1) + (Tile == 16 ? quad / 2 : 0);
  const int slot = Tile == 16 ? (lane & 7) | ((lane & 16) >> 1) : lane;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups = k / 16;
  const int count = groups / Split;
  const int begin = split * count;
  float accum[2][8] = {};
  uint4 w0, w1;
  auto load_weight = [&](int group, uint4& first, uint4& second) {
    const half* ptr =
        weight +
        ((static_cast<size_t>(blockIdx.x) * groups + group) * Tile + slot) * 16;
    first = __ldcs(reinterpret_cast<const uint4*>(ptr));
    second = __ldcs(reinterpret_cast<const uint4*>(ptr + 8));
  };
#pragma unroll 2
  for (int group = begin; group < begin + count; ++group) {
    load_weight(group, w0, w1);
    const half* x = input + static_cast<size_t>(row) * k + group * 16;
    const uint4 a0 = *reinterpret_cast<const uint4*>(x);
    const uint4 a1 = *reinterpret_cast<const uint4*>(x + 8);
    MMA(accum[0], a0.x, a0.y, w0.x, w0.y);
    MMA(accum[1], a0.z, a0.w, w0.z, w0.w);
    MMA(accum[0], a1.x, a1.y, w1.x, w1.y);
    MMA(accum[1], a1.z, a1.w, w1.z, w1.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int out_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int out_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    const int tile_quad = Tile == 16 ? quad & 1 : quad;
    partials[split][out_row * Tile + tile_quad * 8 + out_col] =
        accum[0][i] + accum[1][i];
  }
  __syncthreads();
  for (int element = threadIdx.x; element < 8 * Tile; element += blockDim.x) {
    float value = 0;
#pragma unroll
    for (int part = 0; part < Split; ++part) value += partials[part][element];
    const int out_row = element / Tile;
    const int out_col = element % Tile;
    output[static_cast<size_t>(out_row) * n + blockIdx.x * Tile + out_col] =
        __float2half_rn(value);
  }
}

// K18: the kernel loads input rows and packed weights as uint4.
bool aligned16(const torch::Tensor& t) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
}

}  // namespace

void sm70_dflash2_fp16_m8_out(torch::Tensor output, torch::Tensor input,
                              torch::Tensor packed, int64_t tile,
                              int64_t warps) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda() && packed.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf &&
              packed.scalar_type() == at::kHalf);
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() &&
              packed.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 8 && output.dim() == 2 &&
              output.size(0) == 8);
  TORCH_CHECK(input.device() == output.device() &&
              input.device() == packed.device());
  const int n = output.size(1), k = input.size(1);
  TORCH_CHECK((tile == 16 || tile == 32) && (warps == 8 || warps == 16));
  TORCH_CHECK(n % tile == 0 && k % (16 * warps * (tile == 16 ? 2 : 1)) == 0);
  TORCH_CHECK(packed.numel() == static_cast<int64_t>(n) * k);
  // K18: the fork assumed 16-byte aligned base pointers; check it.
  TORCH_CHECK(aligned16(input) && aligned16(packed),
              "DFlash2 FP16 M8: input and packed must be 16-byte aligned");
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
#define LAUNCH(W, T)                                          \
  dflash2_fp16_m8_kernel<W, T><<<n / T, W * 32, 0, stream>>>( \
      reinterpret_cast<half*>(output.data_ptr()),             \
      reinterpret_cast<const half*>(input.data_ptr()),        \
      reinterpret_cast<const half*>(packed.data_ptr()), n, k)
  if (tile == 16 && warps == 16) {
    LAUNCH(16, 16);
  } else if (tile == 32 && warps == 8) {
    LAUNCH(8, 32);
  } else if (tile == 32 && warps == 16) {
    LAUNCH(16, 32);
  } else {
    TORCH_CHECK(false, "Unsupported DFlash2 FP16 M8 geometry");
  }
#undef LAUNCH
  AT_CUDA_CHECK(cudaGetLastError());
}

void sm70_dflash2_fp16_dispatch_out(torch::Tensor output, torch::Tensor input,
                                    torch::Tensor packed, torch::Tensor weight,
                                    int64_t tile, int64_t warps) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && output.dim() == 2);
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
              weight.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
              output.is_contiguous());
  TORCH_CHECK(input.device() == weight.device() &&
              input.device() == output.device());
  TORCH_CHECK(input.size(1) == weight.size(1) &&
              output.size(0) == input.size(0) &&
              output.size(1) == weight.size(0));
  const int64_t m = input.size(0);
  // K18: misaligned views (e.g. a row slice) keep the matrix product.
  const bool aligned = aligned16(input) && aligned16(packed);
  if (m == 8 && aligned) {
    sm70_dflash2_fp16_m8_out(output, input, packed, tile, warps);
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  if (m >= 1 && m < 8 && aligned) {
    // K18 (new): M 1..7 runs the M8 kernel on zero-padded rows. Each output
    // row depends only on its own input row and the reduction order is fixed,
    // so rows [0, m) are bitwise those of an M8 call on the padded input.
    auto padded_in = at::zeros({8, input.size(1)}, input.options());
    padded_in.narrow(0, 0, m).copy_(input);
    auto padded_out = at::empty({8, output.size(1)}, output.options());
    sm70_dflash2_fp16_m8_out(padded_out, padded_in, packed, tile, warps);
    output.copy_(padded_out.narrow(0, 0, m));
    return;
  }
  // F.linear with a 2D input and no bias uses this same matrix product.
  // Keep runtime M dispatch inside the op so a dynamic AOT prefill trace
  // does not erase the M8 decode route.
  at::mm_out(output, input, weight.t());
}

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def(
      "sm70_dflash2_fp16_m8_out(Tensor(a!) output, Tensor input, Tensor "
      "packed, int tile, int warps) -> ()");
  m.def(
      "sm70_dflash2_fp16_dispatch_out(Tensor(a!) output, Tensor input, Tensor "
      "packed, Tensor weight, int tile, int warps) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("sm70_dflash2_fp16_m8_out", &sm70_dflash2_fp16_m8_out);
  m.impl("sm70_dflash2_fp16_dispatch_out", &sm70_dflash2_fp16_dispatch_out);
}
