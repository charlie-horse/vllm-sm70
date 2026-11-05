// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The QPN8 execution layout is derived from dnv2003/v100-skinny (MIT) and
// its block-scale adaptation in haohervchb/sglang-V100. Automatic operator
// dispatch is restricted to accepted SM70 tensor/layout contracts and can be
// disabled with VLLM_SM70_FP8_QPN8=0.
// See LICENSE.v100-skinny in this directory for the retained MIT notice.
//
// vllm-sm70 (K09): vendored from fork cf9492a, trimmed to prepare,
// dequantize and the M<=8 GEMM, registered in the _sm70_C namespace.
// PROVENANCE.md lists every change.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <mutex>


namespace {

__device__ __forceinline__ void fp8x8_to_half2x4(uint2 q, half2 out[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned b0 = (q.x >> (8 * i)) & 0xffu;
    const unsigned b1 = (q.y >> (8 * i)) & 0xffu;
    const unsigned h0 = ((b0 & 0x80u) << 8) | ((b0 & 0x7fu) << 7);
    const unsigned h1 = ((b1 & 0x80u) << 8) | ((b1 & 0x7fu) << 7);
    const unsigned packed = h0 | (h1 << 16);
    out[i] = *reinterpret_cast<const half2*>(&packed);
  }
}

__device__ __forceinline__ void fp8x8_to_half2x4_fast(uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  unsigned permuted[4];
  permuted[0] = __byte_perm(q.x, q.y, 0x0400);
  permuted[1] = __byte_perm(q.x, q.y, 0x0501);
  permuted[2] = __byte_perm(q.x, q.y, 0x0602);
  permuted[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned value =
        ((permuted[i] << 8) & kSign) | ((permuted[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

constexpr int kQpn8PrepareThreads = 256;

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn8_lane_from_col(int col) {
  return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}

__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
  const int local = logical_k & 7;
  return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

__global__ void fp8_qpn8_prepack_sm70_kernel(
    uint8_t* __restrict__ codes, const uint8_t* __restrict__ qweight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k;
  if (index >= numel) {
    return;
  }

  const int row = static_cast<int>(index / k);
  const int logical_k = static_cast<int>(index % k);
  const int tile = row >> 5;
  const int lane = qpn8_lane_from_col(row & 31);
  const int group = logical_k >> 4;
  const int physical_k = qpn8_physical_k(logical_k & 15);
  const int groups_k16 = k >> 4;
  const size_t packed_index =
      (((static_cast<size_t>(tile) * groups_k16 + group) * 32 + lane) * 16 +
       physical_k);
  codes[packed_index] = qweight[index];
}

__global__ void fp8_qpn8_scale_sm70_kernel(half* __restrict__ group_scales,
                                           const float* __restrict__ scales,
                                           int n_blocks, int k_blocks) {
  const int n_tiles = n_blocks * 4;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int numel = k_blocks * n_tiles;
  if (index >= numel) {
    return;
  }

  const int k_block = index / n_tiles;
  const int n_tile = index - k_block * n_tiles;
  group_scales[index] =
      __float2half(scales[(n_tile >> 2) * k_blocks + k_block] * 256.0f);
}

__global__ void fp8_qpn8_channel_scale_sm70_kernel(
    half* __restrict__ channel_scales, const float* __restrict__ scales,
    int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n) {
    channel_scales[col] = __float2half(scales[col] * 256.0f);
  }
}

__global__ void fp8_qpn8_dequantize_sm70_kernel(
    half* __restrict__ output, const uint8_t* __restrict__ codes,
    const half* __restrict__ group_scales, int n, int k, bool channel_scales) {
  const size_t word_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t word_count = static_cast<size_t>(n) * k / 16;
  if (word_index >= word_count) {
    return;
  }

  const int groups_k16 = k >> 4;
  const int lane = static_cast<int>(word_index & 31);
  const size_t tile_word = word_index >> 5;
  const int group = static_cast<int>(tile_word % groups_k16);
  const int tile = static_cast<int>(tile_word / groups_k16);
  const int col = tile * 32 + qpn8_col_from_lane(lane);
  const int tiles_n32 = n >> 5;
  const half scale = channel_scales
                         ? group_scales[col]
                         : group_scales[(group >> 3) * tiles_n32 + tile];
  const half2 scale2 = __halves2half2(scale, scale);
  const uint4 packed = reinterpret_cast<const uint4*>(codes)[word_index];
  half2 weights[8];
  fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
  fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const half2 value = __hmul2(weights[pair], scale2);
    const int k_base = group * 16 + pair * 2;
    output[static_cast<size_t>(k_base) * n + col] = __low2half(value);
    output[static_cast<size_t>(k_base + 1) * n + col] = __high2half(value);
  }
}

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, bool FusedBA = false, bool SplitOutputs = false,
          int RowTiles = 1>
__global__ void fp8_qpn8_sm70_kernel(
    const uint8_t* __restrict__ codes, const half* __restrict__ group_scales,
    const half* __restrict__ input, half* __restrict__ output,
    half* __restrict__ z_output, const half* __restrict__ ba_weight,
    half* __restrict__ ba_output, half* __restrict__ b_output,
    half* __restrict__ a_output, int ba_n, int qkv_n, int n, int k, int m,
    bool channel_scales) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "QPN8 supports one or two 8-row tiles");
  static_assert(!M1Only || RowTiles == 1,
                "QPN8 M=1 specialization uses one row tile");
  __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  if constexpr (FusedBA) {
    const int qpn_tiles = n >> 5;
    if (tile >= qpn_tiles) {
      constexpr int kBARowsPerBlock = 2;
      constexpr int kBAThreadsPerRow = 256;
      const int ba_blocks_per_token = (ba_n + 1) / kBARowsPerBlock;
      const int token = M1Only ? 0 : (tile - qpn_tiles) / ba_blocks_per_token;
      const int ba_block = (tile - qpn_tiles) % ba_blocks_per_token;
      const int ba_group = threadIdx.x / kBAThreadsPerRow;
      const int ba_thread = threadIdx.x % kBAThreadsPerRow;
      const int ba_warp = ba_thread >> 5;
      const int ba_row = ba_block * kBARowsPerBlock + ba_group;
      float value = 0.0f;
      const half2* input2 = reinterpret_cast<const half2*>(
          input + static_cast<size_t>(token) * k);
      const half2* weight2 = reinterpret_cast<const half2*>(
          ba_weight + static_cast<size_t>(ba_row) * k);
      for (int pair = ba_thread; pair < k / 2; pair += kBAThreadsPerRow) {
        const float2 x = __half22float2(__ldg(input2 + pair));
        const float2 weight = __half22float2(__ldg(weight2 + pair));
        value = fmaf(x.x, weight.x, value);
        value = fmaf(x.y, weight.y, value);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
      }
      if (lane == 0) {
        partials[ba_group][ba_warp] = value;
      }
      __syncthreads();
      if (ba_warp == 0) {
        value = lane < 8 ? partials[ba_group][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0 && ba_row < ba_n) {
          if constexpr (SplitOutputs) {
            if (ba_row < ba_n / 2) {
              b_output[static_cast<size_t>(token) * (ba_n / 2) + ba_row] =
                  __float2half(value);
            } else {
              a_output[static_cast<size_t>(token) * (ba_n / 2) + ba_row -
                       ba_n / 2] = __float2half(value);
            }
          } else {
            ba_output[static_cast<size_t>(token) * ba_n + ba_row] =
                __float2half(value);
          }
        }
      }
      return;
    }
  }
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const int group_begin = warp * groups_per_warp;
  const int tiles_n32 = n >> 5;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const half* scale_ptr = group_scales + tile;

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][chain][i] = 0.0f;
      }
    }
  }
  int loaded_scale_group = -1;
  half loaded_scale =
      channel_scales
          ? __ldg(group_scales + tile * 32 + qpn8_col_from_lane(lane))
          : __float2half(0.0f);
  uint4 prefetched = make_uint4(0, 0, 0, 0);
  if constexpr (PrefetchCodes) {
    prefetched = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const int scale_group = group >> 3;
    if (!channel_scales && scale_group != loaded_scale_group) {
      loaded_scale =
          __ldg(scale_ptr + static_cast<size_t>(scale_group) * tiles_n32);
      loaded_scale_group = scale_group;
    }

    const uint4 packed =
        PrefetchCodes ? prefetched
                      : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    uint4 next = make_uint4(0, 0, 0, 0);
    if constexpr (PrefetchCodes) {
      if (group + 1 < group_begin + groups_per_warp) {
        next = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
      }
    }
    half2 weights[8];
    if constexpr (FastDecoder) {
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
    } else {
      fp8x8_to_half2x4(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4(make_uint2(packed.z, packed.w), weights + 4);
    }

    const half2 scale2 = __halves2half2(loaded_scale, loaded_scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      weights[i] = __hmul2(weights[i], scale2);
    }

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int input_row_idx = row_tile * 8 + row;
      if (input_row_idx < m) {
        const half* input_row = input + static_cast<size_t>(input_row_idx) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }

      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
    if constexpr (PrefetchCodes) {
      prefetched = next;
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][0][i] += accum[row_tile][chain][i];
      }
    }
  }

  if constexpr (M1Only) {
    if ((lane & 17) == 0) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
        for (int offset = 0; offset < 2; ++offset) {
          const int i = pair * 4 + offset;
          const int output_col =
              offset | (((lane >> 1) & 1) << 1) | (pair << 2);
          partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
        }
      }
    }
  } else {
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int output_row =
            row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partials[warp][output_row * 32 + quadpair * 8 + output_col] =
            accum[row_tile][0][i];
      }
    }
  }
  __syncthreads();

  constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
  for (int element = threadIdx.x; element < kOutputElements;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    if constexpr (M1Only) {
      const int output_col = tile * 32 + element;
      if constexpr (SplitOutputs) {
        if (output_col < qkv_n) {
          output[output_col] = __float2half(value);
        } else {
          z_output[output_col - qkv_n] = __float2half(value);
        }
      } else {
        output[output_col] = __float2half(value);
      }
    } else {
      const int output_row = element >> 5;
      const int output_col = element & 31;
      if (output_row < m) {
        const int col = tile * 32 + output_col;
        if constexpr (SplitOutputs) {
          if (col < qkv_n) {
            output[static_cast<size_t>(output_row) * qkv_n + col] =
                __float2half(value);
          } else {
            z_output[static_cast<size_t>(output_row) * (n - qkv_n) + col -
                     qkv_n] = __float2half(value);
          }
        } else {
          output[static_cast<size_t>(output_row) * n + col] =
              __float2half(value);
        }
      }
    }
  }
}

template <int SplitK, int NAcc, bool FastDecoder, bool PrefetchCodes,
          bool M1Only = false, int RowTiles = 1>
void launch_fp8_qpn8_sm70(const uint8_t* codes, const half* group_scales,
                          const half* input, half* output, int n, int k, int m,
                          bool channel_scales, cudaStream_t stream) {
  fp8_qpn8_sm70_kernel<SplitK, NAcc, FastDecoder, PrefetchCodes, M1Only, false,
                       false, RowTiles><<<(n / 32), (32 * SplitK), 0, stream>>>(
      codes, group_scales, input, output, nullptr, nullptr, nullptr, nullptr,
      nullptr, 0, n, n, k, m, channel_scales);
}

}  // namespace

std::vector<torch::Tensor> fp8_qpn8_prepare_sm70(torch::Tensor qweight,
                                                 torch::Tensor scales) {
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(),
              "fp8_qpn8_prepare_sm70: tensors must be CUDA tensors");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_qpn8_prepare_sm70: weight must be float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "fp8_qpn8_prepare_sm70: scales must be float32");
  TORCH_CHECK(qweight.dim() == 2 && scales.dim() == 2,
              "fp8_qpn8_prepare_sm70: tensors must be 2D");
  TORCH_CHECK(qweight.is_contiguous() && scales.is_contiguous(),
              "fp8_qpn8_prepare_sm70: tensors must be contiguous");
  TORCH_CHECK(qweight.get_device() == scales.get_device(),
              "fp8_qpn8_prepare_sm70: tensors must share one device");

  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1);
  const bool channel_scales = scales.size(0) == n && scales.size(1) == 1;
  const bool block_scales =
      scales.size(0) == n / 128 && scales.size(1) == k / 128;
  TORCH_CHECK(channel_scales ? (n > 0 && n % 32 == 0 && k > 0 && k % 16 == 0)
                             : (n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0),
              "fp8_qpn8_prepare_sm70: channel scales require N%32=K%16=0; "
              "block scales require N%128=K%128=0");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_prepare_sm70: expected channel scales [N, 1] or "
              "block scales [N/128, K/128]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto codes = torch::empty(
      {k, n},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  auto group_scales = torch::empty(
      {channel_scales ? 1 : k / 128, channel_scales ? n : n / 32},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kFloat16));

  const int64_t weight_numel = n * k;
  const int weight_blocks = static_cast<int>(
      (weight_numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads);
  fp8_qpn8_prepack_sm70_kernel<<<weight_blocks, kQpn8PrepareThreads, 0,
                                 stream>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(qweight.data_ptr()), static_cast<int>(n),
      static_cast<int>(k));

  const int64_t scale_numel = group_scales.numel();
  const int scale_blocks = static_cast<int>(
      (scale_numel + kQpn8PrepareThreads - 1) / kQpn8PrepareThreads);
  if (channel_scales) {
    fp8_qpn8_channel_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                         stream>>>(
        reinterpret_cast<half*>(group_scales.data_ptr<at::Half>()),
        scales.data_ptr<float>(), static_cast<int>(n));
  } else {
    fp8_qpn8_scale_sm70_kernel<<<scale_blocks, kQpn8PrepareThreads, 0,
                                 stream>>>(
        reinterpret_cast<half*>(group_scales.data_ptr<at::Half>()),
        scales.data_ptr<float>(), static_cast<int>(n / 128),
        static_cast<int>(k / 128));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, group_scales};
}

void fp8_qpn8_dequantize_sm70_out(torch::Tensor out, torch::Tensor codes,
                                  torch::Tensor group_scales) {
  TORCH_CHECK(out.is_cuda() && codes.is_cuda() && group_scales.is_cuda(),
              "fp8_qpn8_dequantize_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 &&
                  codes.scalar_type() == torch::kUInt8 &&
                  group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_dequantize_sm70_out: dtype mismatch");
  TORCH_CHECK(out.dim() == 2 && codes.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_dequantize_sm70_out: tensors must be 2D");
  TORCH_CHECK(out.is_contiguous() && codes.is_contiguous() &&
                  group_scales.is_contiguous(),
              "fp8_qpn8_dequantize_sm70_out: tensors must be contiguous");
  TORCH_CHECK(out.get_device() == codes.get_device() &&
                  out.get_device() == group_scales.get_device(),
              "fp8_qpn8_dequantize_sm70_out: tensors must share one device");

  const int64_t k = out.size(0);
  const int64_t n = out.size(1);
  const bool channel_scales =
      group_scales.size(0) == 1 && group_scales.size(1) == n;
  const bool block_scales =
      group_scales.size(0) == k / 128 && group_scales.size(1) == n / 32;
  TORCH_CHECK(channel_scales ? (n > 0 && n % 32 == 0 && k > 0 && k % 16 == 0)
                             : (n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0),
              "fp8_qpn8_dequantize_sm70_out: channel scales require "
              "N%32=K%16=0; block scales require N%128=K%128=0");
  TORCH_CHECK(codes.numel() == k * n,
              "fp8_qpn8_dequantize_sm70_out: packed code size mismatch");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_dequantize_sm70_out: scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const int64_t word_count = k * n / 16;
  const int blocks = static_cast<int>((word_count + kQpn8PrepareThreads - 1) /
                                      kQpn8PrepareThreads);
  fp8_qpn8_dequantize_sm70_kernel<<<blocks, kQpn8PrepareThreads, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>()),
      static_cast<int>(n), static_cast<int>(k), channel_scales);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_qpn8_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                            torch::Tensor codes, torch::Tensor group_scales,
                            int64_t split_k, int64_t accumulator_chains,
                            bool fast_decoder, bool prefetch_codes) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda(),
              "fp8_qpn8_gemm_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "fp8_qpn8_gemm_sm70_out: input and output must be float16");
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_gemm_sm70_out: codes must be uint8");
  TORCH_CHECK(group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_gemm_sm70_out: group scales must be float16");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_gemm_sm70_out: expected 2D tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous() &&
                  codes.is_contiguous() && group_scales.is_contiguous(),
              "fp8_qpn8_gemm_sm70_out: tensors must be contiguous");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == codes.get_device() &&
                  input.get_device() == group_scales.get_device(),
              "fp8_qpn8_gemm_sm70_out: tensors must share one device");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  // K09: M<=8 only (one 8-row tile). Larger M goes to TurboMind FP8 or a
  // dequantize + mm (vllm_sm70/ops/qpn8.py qpn8_linear).
  TORCH_CHECK(m >= 1 && m <= 8,
              "fp8_qpn8_gemm_sm70_out: M must be in [1, 8]");
  TORCH_CHECK(out.size(0) == m, "fp8_qpn8_gemm_sm70_out: output M mismatch");
  TORCH_CHECK(n > 0 && n % 32 == 0,
              "fp8_qpn8_gemm_sm70_out: N must be a positive multiple of 32");
  const bool channel_scales =
      group_scales.size(0) == 1 && group_scales.size(1) == n;
  const bool block_scales =
      group_scales.size(0) == k / 128 && group_scales.size(1) == n / 32;
  TORCH_CHECK(channel_scales ? (k > 0 && k % 16 == 0) : (k > 0 && k % 128 == 0),
              "fp8_qpn8_gemm_sm70_out: channel scales require K%16=0; "
              "block scales require K%128=0");
  TORCH_CHECK(codes.numel() == n * k,
              "fp8_qpn8_gemm_sm70_out: packed code size mismatch");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_gemm_sm70_out: scale shape mismatch");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 12 || split_k == 16 ||
                  split_k == 32,
              "fp8_qpn8_gemm_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "fp8_qpn8_gemm_sm70_out: K/16 must be divisible by split_k");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "fp8_qpn8_gemm_sm70_out: accumulator_chains must be 1 or 2");
  TORCH_CHECK(!prefetch_codes || fast_decoder,
              "fp8_qpn8_gemm_sm70_out: prefetch experiment requires the "
              "fast decoder");
  const bool exact_shape_split = split_k == 12;
  TORCH_CHECK(!exact_shape_split ||
                  (accumulator_chains == 2 && fast_decoder && !prefetch_codes),
              "fp8_qpn8_gemm_sm70_out: split_k 12 requires the "
              "fast decoder, two accumulator chains, and no prefetch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

  // Keep the M=1 reduction order restricted to the two tuned output/down
  // projections. Extending it to the other split variants changed the frozen
  // random-sampling token stream without an end-to-end decode win.
  if (m == 1 && accumulator_chains == 2 && fast_decoder && !prefetch_codes &&
      (split_k == 12 || split_k == 16)) {
    if (split_k == 12) {
      launch_fp8_qpn8_sm70<12, 2, true, false, true>(
          code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),
          static_cast<int>(k), 1, channel_scales, stream);
    } else {
      launch_fp8_qpn8_sm70<16, 2, true, false, true>(
          code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),
          static_cast<int>(k), 1, channel_scales, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

#define VLLM_LAUNCH_QPN8(SPLIT, NACC, FAST, PREFETCH)                        \
  do {                                                                       \
    launch_fp8_qpn8_sm70<SPLIT, NACC, FAST, PREFETCH, false, 1>(             \
        code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),     \
        static_cast<int>(k), static_cast<int>(m), channel_scales, stream);   \
  } while (0)

  if (prefetch_codes) {
    if (split_k == 4 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(4, 1, true, true);
    } else if (split_k == 4 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(4, 2, true, true);
    } else if (split_k == 8 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(8, 1, true, true);
    } else if (split_k == 8 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(8, 2, true, true);
    } else if (split_k == 16 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(16, 1, true, true);
    } else if (split_k == 16 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(16, 2, true, true);
    } else if (split_k == 32 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(32, 1, true, true);
    } else {
      VLLM_LAUNCH_QPN8(32, 2, true, true);
    }
  } else if (fast_decoder) {
    if (split_k == 4 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(4, 1, true, false);
    } else if (split_k == 4 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(4, 2, true, false);
    } else if (split_k == 8 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(8, 1, true, false);
    } else if (split_k == 8 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(8, 2, true, false);
    } else if (split_k == 12) {
      VLLM_LAUNCH_QPN8(12, 2, true, false);
    } else if (split_k == 16 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(16, 1, true, false);
    } else if (split_k == 16 && accumulator_chains == 2) {
      VLLM_LAUNCH_QPN8(16, 2, true, false);
    } else if (split_k == 32 && accumulator_chains == 1) {
      VLLM_LAUNCH_QPN8(32, 1, true, false);
    } else {
      VLLM_LAUNCH_QPN8(32, 2, true, false);
    }
  } else if (split_k == 4 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(4, 1, false, false);
  } else if (split_k == 4 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8(4, 2, false, false);
  } else if (split_k == 8 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(8, 1, false, false);
  } else if (split_k == 8 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8(8, 2, false, false);
  } else if (split_k == 16 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(16, 1, false, false);
  } else if (split_k == 16 && accumulator_chains == 2) {
    VLLM_LAUNCH_QPN8(16, 2, false, false);
  } else if (split_k == 32 && accumulator_chains == 1) {
    VLLM_LAUNCH_QPN8(32, 1, false, false);
  } else {
    VLLM_LAUNCH_QPN8(32, 2, false, false);
  }
#undef VLLM_LAUNCH_QPN8

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// K09: registration in the _sm70_C namespace (the fork registered these in
// csrc/torch_bindings.cpp:723-758 under _C with a native_policy argument).
// Fakes live in vllm_sm70/ops/qpn8.py.
TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def("fp8_qpn8_prepare_sm70(Tensor qweight, Tensor scales) -> Tensor[]");
  m.def(
      "fp8_qpn8_dequantize_sm70_out(Tensor(a!) out, Tensor codes, "
      "Tensor group_scales) -> ()");
  m.def(
      "fp8_qpn8_gemm_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor group_scales, int split_k, int accumulator_chains, "
      "bool fast_decoder, bool prefetch_codes) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("fp8_qpn8_prepare_sm70", &fp8_qpn8_prepare_sm70);
  m.impl("fp8_qpn8_dequantize_sm70_out", &fp8_qpn8_dequantize_sm70_out);
  m.impl("fp8_qpn8_gemm_sm70_out", &fp8_qpn8_gemm_sm70_out);
}
