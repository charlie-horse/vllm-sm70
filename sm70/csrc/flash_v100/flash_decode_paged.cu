#include "flash_v100_policy.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>

#include "fp8_kv_utils.cuh"
#include "fused_mma.h"

namespace {

int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto" || kv_cache_dtype == "float16" ||
      kv_cache_dtype == "bfloat16") {
    return flash_v100::KV_CACHE_DTYPE_FP16;
  }
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  }
  if (kv_cache_dtype == "fp8_e5m2") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E5M2;
  }
  return -1;
}

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kXQATCBlockN = 128;
constexpr int kXQATCStride = 128;
// QSA exposes selected four-token microblocks as virtual paged KV. Keep enough
// page slots for one full XQA tile at that minimum supported granularity.
constexpr int kXQATCPageIdsCapacity = kXQATCBlockN / 4;
constexpr int kXQATC256WideWarpCount = 8;
constexpr int kXQATC256WideThreads = kXQATC256WideWarpCount * kWarpSize;
constexpr int kXQATC256WideBlockM = 8;
constexpr int kXQATC256WideThreadsPerRow = kWarpSize;
constexpr int kXQARouteAllSeqLens = 0;
constexpr int kXQARouteShortSeqLens = -1;
constexpr int kXQARouteLongSeqLens = 1;
constexpr int kXQARouteP1024Sawtooth = 4;
constexpr int kXQARouteP256Sawtooth = 5;
constexpr int kXQARouteP1024SawtoothMid = 6;
constexpr int kXQARouteP1024SawtoothFinal = 7;
constexpr int kXQARouteRangeSeqLens = 8;
constexpr int kXQARouteWaveLongSeqLens = 9;
// The dense 256/128-half strides alias Volta shared-memory banks in the
// WMMA A/B fragment loads. Both padded strides remain 16-byte aligned.
constexpr int kXQATC256WidePaddedQStride = 264;
constexpr int kXQATC256WidePaddedKVStride = 136;
constexpr int kXQATC256WideAlignedPaddedQStride = 272;
constexpr int kXQATC256WideAlignedPaddedKVStride = 144;
constexpr int kXQATCQKPipelinePanelDim = 64;
constexpr int kXQATCQKPipelineKVStride = 72;
constexpr float kXQANegInf = -1.0e30f;

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kKVStride;
  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    alignas(16) __half k[kXQATCBlockN * kKVStride];
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int) { return reuse_kv.k; }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCQKPipelineSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kXQATCQKPipelineKVStride;
  struct QKBuffers {
    alignas(16) __half panel[2][kXQATCBlockN * kQKStride];
  };

  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    QKBuffers qk;
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int index) {
    return reuse_kv.qk.panel[index];
  }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

bool xqa_padded_smem_enabled() {
  return flash_v100::policy::value(
      flash_v100::policy::Field::xqa_padded_smem_enabled);
}

template <int SEQ_LEN_ROUTE>
__device__ __forceinline__ bool xqa_seq_len_route_active(
    const int seq_len, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  if constexpr (SEQ_LEN_ROUTE == kXQARouteShortSeqLens) {
    return seq_len < route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024Sawtooth) {
    return (seq_len >= route_seq_len_begin && seq_len < route_seq_len_end) ||
           seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP256Sawtooth) {
    return seq_len < route_seq_len_begin ||
           (seq_len >= route_seq_len_end && seq_len < route_seq_len_final);
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothMid) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothFinal) {
    return seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteRangeSeqLens) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  }
  return true;
}

__device__ __forceinline__ int xqa_sawtooth_partition_size(
    const int seq_len, const int p1024_mid_seq_len, const int p256_long_seq_len,
    const int p1024_final_seq_len) {
  const bool use_p1024 =
      (seq_len >= p1024_mid_seq_len && seq_len < p256_long_seq_len) ||
      seq_len >= p1024_final_seq_len;
  return use_p1024 ? 1024 : 256;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : 0.f;
  if (warp == 0) {
    val = warp_reduce_sum(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_max(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : -1.0e20f;
  if (warp == 0) {
    val = warp_reduce_max(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

__device__ __forceinline__ uint32_t
fp8_e5m2_pair_to_half2_bits(const uint16_t raw_pair) {
  return (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
         (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
}

__device__ __forceinline__ uint4 fp8_e5m2_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint16_t fp8_e4m3fn_to_half_bits(const uint8_t raw) {
  const uint16_t sign = static_cast<uint16_t>(raw & 0x80u) << 8;
  const uint8_t magnitude = raw & 0x7fu;
  const uint8_t exponent = magnitude >> 3;
  const uint8_t mantissa = magnitude & 0x07u;
  if (magnitude == 0) {
    return sign;
  }
  if (exponent == 0) {
    // E4M3 subnormals are exact fp16 normals: mantissa * 2^-9.
    const uint16_t magnitude_bits =
        mantissa < 2
            ? 0x1800u
            : (mantissa < 4
                   ? static_cast<uint16_t>(0x1c00u | ((mantissa - 2) << 9))
                   : static_cast<uint16_t>(0x2000u | ((mantissa - 4) << 8)));
    return sign | magnitude_bits;
  }
  if (magnitude == 0x7fu) {
    return sign | 0x7e00u;
  }
  return sign | static_cast<uint16_t>((exponent + 8) << 10) |
         static_cast<uint16_t>(mantissa << 7);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits(const uint16_t raw_pair) {
  return static_cast<uint32_t>(
             fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair))) |
         (static_cast<uint32_t>(
              fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair >> 8)))
          << 16);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits_fast(const uint16_t raw_pair) {
  const uint8_t raw0 = static_cast<uint8_t>(raw_pair);
  const uint8_t raw1 = static_cast<uint8_t>(raw_pair >> 8);
  if ((raw0 & 0x7fu) == 0x7fu || (raw1 & 0x7fu) == 0x7fu) {
    return fp8_e4m3fn_pair_to_half2_bits(raw_pair);
  }

  // Moving a finite E4M3 encoding into the corresponding fp16 sign,
  // exponent, and mantissa fields represents exactly value / 256. A packed
  // half2 multiply restores both values without per-byte exponent branches.
  const uint32_t expanded = (static_cast<uint32_t>(raw_pair & 0x0080u) << 8) |
                            (static_cast<uint32_t>(raw_pair & 0x007fu) << 7) |
                            (static_cast<uint32_t>(raw_pair & 0x8000u) << 16) |
                            (static_cast<uint32_t>(raw_pair & 0x7f00u) << 15);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = expanded;
  converter.h2 = __hmul2(converter.h2, __float2half2_rn(256.0f));
  return converter.u;
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8_fast(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4 fp8_e4m3fn_vector_to_half8_lut(
    const uint64_t raw, const uint16_t* __restrict__ lut) {
  return make_uint4(
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 8)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 16)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 24)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 32)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 40)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 48)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 56)]) << 16));
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ uint4 load_xqa_tc_kv_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const uint16_t* __restrict__ e4m3_lut = nullptr) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  static_assert(BLOCK_SIZE == 0 || BLOCK_SIZE == 4 || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 784 || BLOCK_SIZE == 800 ||
                    BLOCK_SIZE == 1568 || BLOCK_SIZE == 1648 ||
                    BLOCK_SIZE == 3296,
                "Unsupported paged-KV block-size specialization");
  static_assert(!CONTIGUOUS_HKV1_LAYOUT || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 800 || BLOCK_SIZE == 1568 ||
                    BLOCK_SIZE == 1648 || BLOCK_SIZE == 3296,
                "The fixed-stride Hkv=1 layout requires a specialized page");
  int logical_block;
  int block_offset;
  if constexpr (BLOCK_SIZE == 4) {
    logical_block = token_offset >> 2;
    block_offset = token_offset & 3;
  } else if constexpr (BLOCK_SIZE == 16) {
    logical_block = token_offset >> 4;
    block_offset = token_offset & 15;
  } else if constexpr (BLOCK_SIZE == 784) {
    logical_block = token_offset / 784;
    block_offset = token_offset - logical_block * 784;
  } else if constexpr (BLOCK_SIZE == 800) {
    // E4M3 long-wave partitions can span four 800-token pages. Keep the
    // constant-page specialization division-free without truncating at page 1.
    logical_block =
        (token_offset >= 800) + (token_offset >= 1600) + (token_offset >= 2400);
    block_offset = token_offset - logical_block * 800;
  } else if constexpr (BLOCK_SIZE == 1568) {
    logical_block = token_offset / 1568;
    block_offset = token_offset - logical_block * 1568;
  } else if constexpr (BLOCK_SIZE == 1648) {
    logical_block = token_offset / 1648;
    block_offset = token_offset - logical_block * 1648;
  } else if constexpr (BLOCK_SIZE == 3296) {
    logical_block = token_offset / 3296;
    block_offset = token_offset - logical_block * 3296;
  } else {
    logical_block = token_offset / block_size;
    block_offset = token_offset % block_size;
  }
  const int physical_block = page_ids[logical_block];
  int64_t physical_offset;
  if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kBlockStride =
        BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
    physical_offset = static_cast<int64_t>(physical_block) * kBlockStride +
                      static_cast<int64_t>(block_offset) * kHeadDim +
                      panel_offset;
  } else {
    physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                      static_cast<int64_t>(block_offset) * token_stride +
                      static_cast<int64_t>(kv_head_idx) * head_stride +
                      panel_offset;
  }
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const uint4* cache_vec = reinterpret_cast<const uint4*>(kv_cache);
    return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
  } else {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
                  "XQA only supports fp16, FP8 E4M3, and FP8 E5M2 KV");
    const uint64_t* cache_vec = reinterpret_cast<const uint64_t*>(kv_cache);
    const uint64_t raw = __ldg(&cache_vec[physical_offset / 8 + vec_col]);
    if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3) {
      if constexpr (E4M3_SHARED_LUT) {
        return fp8_e4m3fn_vector_to_half8_lut(raw, e4m3_lut);
      } else {
        return fp8_e4m3fn_vector_to_half8(raw);
      }
    } else {
      static_assert(!E4M3_SHARED_LUT,
                    "The E4M3 conversion LUT requires E4M3 KV");
      return fp8_e5m2_vector_to_half8(raw);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool FP8_PAIR_LOAD = false, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx = threadIdx.x,
    const uint16_t* __restrict__ e4m3_lut = nullptr) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  if constexpr (FP8_PAIR_LOAD) {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
                  "Paired XQA loads require FP8 KV");
    static_assert(!E4M3_SHARED_LUT,
                  "Paired E4M3 conversion does not use the shared LUT");
    const int pair_stride = panel_d_stride_uint4 / 2;
    const int pair_count = valid_kv_tile_rows * pair_stride;
    for (int pair_idx = copy_thread_idx; pair_idx < pair_count;
         pair_idx += NUM_THREADS) {
      const int row = pair_idx / pair_stride;
      const int vec_pair = pair_idx % pair_stride;
      const int token_offset = tile_page_offset + kv_tile_start + row;
      int logical_block;
      int block_offset;
      if constexpr (BLOCK_SIZE == 4) {
        logical_block = token_offset >> 2;
        block_offset = token_offset & 3;
      } else if constexpr (BLOCK_SIZE == 16) {
        logical_block = token_offset >> 4;
        block_offset = token_offset & 15;
      } else if constexpr (BLOCK_SIZE == 784) {
        logical_block = token_offset / 784;
        block_offset = token_offset - logical_block * 784;
      } else if constexpr (BLOCK_SIZE == 800) {
        logical_block = (token_offset >= 800) + (token_offset >= 1600) +
                        (token_offset >= 2400);
        block_offset = token_offset - logical_block * 800;
      } else if constexpr (BLOCK_SIZE == 1568) {
        logical_block = token_offset / 1568;
        block_offset = token_offset - logical_block * 1568;
      } else if constexpr (BLOCK_SIZE == 1648) {
        logical_block = token_offset / 1648;
        block_offset = token_offset - logical_block * 1648;
      } else if constexpr (BLOCK_SIZE == 3296) {
        logical_block = token_offset / 3296;
        block_offset = token_offset - logical_block * 3296;
      } else {
        logical_block = token_offset / block_size;
        block_offset = token_offset % block_size;
      }
      const int physical_block = page_ids[logical_block];
      int64_t physical_offset;
      if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
        constexpr int64_t kHeadDim = 256;
        constexpr int64_t kPhysicalBlockStride =
            BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
        physical_offset =
            static_cast<int64_t>(physical_block) * kPhysicalBlockStride +
            static_cast<int64_t>(block_offset) * kHeadDim + panel_offset;
      } else {
        physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                          static_cast<int64_t>(block_offset) * token_stride +
                          static_cast<int64_t>(kv_head_idx) * head_stride +
                          panel_offset;
      }
      const uint4 raw = __ldg(reinterpret_cast<const uint4*>(kv_cache) +
                              physical_offset / 16 + vec_pair);
      const int shared_offset = row * kv_smem_stride_uint4 + vec_pair * 2;
      const uint64_t raw_lo =
          static_cast<uint64_t>(raw.x) | (static_cast<uint64_t>(raw.y) << 32);
      shared_vec[shared_offset] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_lo)
              : fp8_e5m2_vector_to_half8(raw_lo);
      const uint64_t raw_hi =
          static_cast<uint64_t>(raw.z) | (static_cast<uint64_t>(raw.w) << 32);
      shared_vec[shared_offset + 1] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_hi)
              : fp8_e5m2_vector_to_half8(raw_hi);
    }
  } else {
    const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
    for (int copy_idx = copy_thread_idx; copy_idx < copy_count;
         copy_idx += NUM_THREADS) {
      const int row = copy_idx / panel_d_stride_uint4;
      const int vec_col = copy_idx % panel_d_stride_uint4;
      shared_vec[row * kv_smem_stride_uint4 + vec_col] =
          load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                E4M3_SHARED_LUT>(
              kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
              tile_page_offset, kv_tile_start, block_size, kv_head_idx,
              block_stride, token_stride, head_stride, panel_offset, e4m3_lut);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel_and_zero(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx,
    const uint16_t* __restrict__ e4m3_lut = nullptr) {
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  constexpr int kLoadStages = 4;
  for (int copy_base = copy_thread_idx; copy_base < copy_count;
       copy_base += NUM_THREADS * kLoadStages) {
    uint4 staged[kLoadStages];
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        staged[stage] =
            load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                  E4M3_SHARED_LUT>(
                kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
                tile_page_offset, kv_tile_start, block_size, kv_head_idx,
                block_stride, token_stride, head_stride, panel_offset,
                e4m3_lut);
      }
    }
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        const int row = copy_idx / panel_d_stride_uint4;
        const int vec_col = copy_idx % panel_d_stride_uint4;
        shared_vec[row * kv_smem_stride_uint4 + vec_col] = staged[stage];
      }
    }
  }
  for (int copy_idx = copy_thread_idx + copy_count;
       copy_idx < kXQATCBlockN * panel_d_stride_uint4;
       copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
}

template <int D>
__device__ __forceinline__ float dot_qk_half2(const __half* __restrict__ q_ptr,
                                              const __half* __restrict__ k_ptr,
                                              const int lane) {
  static_assert(D % 2 == 0, "Head dim must be even for half2 dot");
  const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
  const __half2* k_ptr2 = reinterpret_cast<const __half2*>(k_ptr);

  float acc = 0.f;
#pragma unroll
  for (int i = lane; i < D / 2; i += kWarpSize) {
    const float2 qv = __half22float2(q_ptr2[i]);
    const float2 kv = __half22float2(k_ptr2[i]);
    acc = fmaf(qv.x, kv.x, acc);
    acc = fmaf(qv.y, kv.y, acc);
  }
  return warp_reduce_sum(acc);
}

template <int D, int KV_DTYPE, bool E4M3_BITS = false>
__device__ __forceinline__ float dot_qk_cache(const __half* __restrict__ q_ptr,
                                              const void* __restrict__ k_cache,
                                              const int64_t k_index_base,
                                              const int lane) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const __half* k_ptr =
        reinterpret_cast<const __half*>(k_cache) + k_index_base;
    return dot_qk_half2<D>(q_ptr, k_ptr, lane);
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2) {
    static_assert(D % 2 == 0, "Head dim must be even for e5m2 half2 dot");
    const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
    float acc = 0.f;
#pragma unroll
    for (int i = lane; i < D / 2; i += kWarpSize) {
      const float2 qv = __half22float2(q_ptr2[i]);
      const __half2 k_h2 = flash_v100::load_fp8_e5m2_half2_unscaled(
          k_cache, k_index_base + static_cast<int64_t>(i) * 2);
      const float2 kv = __half22float2(k_h2);
      acc = fmaf(qv.x, kv.x, acc);
      acc = fmaf(qv.y, kv.y, acc);
    }
    return warp_reduce_sum(acc);
  } else {
    float acc = 0.f;
#pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float qv = __half2float(q_ptr[d]);
      const float kv =
          flash_v100::load_kv_cache_float_unscaled<KV_DTYPE, E4M3_BITS>(
              k_cache, k_index_base + d);
      acc = fmaf(qv, kv, acc);
    }
    return warp_reduce_sum(acc);
  }
}

template <int D, int PARTITION_SIZE, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool ANCHORED_SWA = false,
          typename PARTIAL_T = __half, bool TP2_E4M3_FAST = false>
__global__ void flash_attention_decode_partition_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, const int batch_size,
    const int max_num_blocks, const int max_num_partitions,
    const int num_heads_q, const int num_heads_kv, const int block_size,
    const int64_t q_stride0, const int64_t q_stride1,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const int64_t v_block_stride, const int64_t v_token_stride,
    const int64_t v_head_stride, const float softmax_scale, const float k_scale,
    const float v_scale, const int window_size_left,
    const int window_size_right, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final,
    const int* __restrict__ anchor_lens, const int anchored_window) {
  // The anchored decode-window mask variant is only generated for the fp16-KV
  // configuration; every other instantiation keeps its original code path.
  static_assert(!ANCHORED_SWA || KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16,
                "ANCHORED_SWA requires an fp16 KV cache");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int query_pos = seq_len - 1;
  const int min_token_idx =
      window_size_left >= 0 ? max(0, query_pos - window_size_left) : 0;
  const int max_token_idx =
      window_size_right >= 0 ? min(seq_len - 1, query_pos + window_size_right)
                             : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);

  // Anchored decode window: keys below the per-request prompt length stay
  // globally visible; later (generated) keys must fall inside the sliding
  // window over generated positions. Keys inside [gap_lo, gap_hi) are masked.
  int gap_lo = seq_len;
  int gap_hi = seq_len;
  if constexpr (ANCHORED_SWA) {
    const int anchor_len = anchor_lens[batch_idx];
    gap_lo = anchor_len;
    gap_hi = max(anchor_len, query_pos - anchored_window + 1);
  }
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? softmax_scale
                                : softmax_scale * k_scale;

  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride2;
  if (part_start >= part_end) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      tmp_out[tmp_out_base + d] = __float2half(0.f);
    }
    if (threadIdx.x == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = -1.0e20f;
      exp_sums[stats_index] = 0.f;
    }
    return;
  }

  const int part_tokens = part_end - part_start;

  __shared__ __half q_shared[D];
  __shared__ float scores_shared[PARTITION_SIZE];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  const __half* q_ptr = q + static_cast<int64_t>(batch_idx) * q_stride0 +
                        static_cast<int64_t>(head_idx) * q_stride1;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_shared[d] = q_ptr[d];
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = part_start + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] =
        block_table[batch_idx * max_num_blocks + logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }
  __syncthreads();

  float local_max = -1.0e20f;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int physical_block = block_idx_shared[token_local];
    const int block_offset = block_offset_shared[token_local];
    const int64_t k_index =
        static_cast<int64_t>(physical_block) * k_block_stride +
        static_cast<int64_t>(block_offset) * k_token_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;

    float score = dot_qk_cache<D, KV_DTYPE, TP2_E4M3_FAST>(q_shared, k_cache,
                                                           k_index, lane);
    if (lane == 0) {
      if constexpr (ANCHORED_SWA) {
        const int token_idx = part_start + token_local;
        if (token_idx >= gap_lo && token_idx < gap_hi) {
          // Masked gap key: never contributes (nulled gap blocks included).
          scores_shared[token_local] = -1.0e30f;
        } else {
          score *= score_scale;
          scores_shared[token_local] = score;
          local_max = fmaxf(local_max, score);
        }
      } else {
        score *= score_scale;
        scores_shared[token_local] = score;
        local_max = fmaxf(local_max, score);
      }
    }
  }

  const float part_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    if constexpr (ANCHORED_SWA) {
      const int token_idx = part_start + i;
      if (token_idx >= gap_lo && token_idx < gap_hi) {
        // Exact zero weight for masked keys (no reliance on expf underflow).
        scores_shared[i] = 0.f;
        continue;
      }
    }
    const float p = __expf(scores_shared[i] - part_max);
    scores_shared[i] = p;
    local_sum += p;
  }
  const float part_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_part_sum = part_sum > 0.f ? 1.f / part_sum : 0.f;
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    if constexpr (TP2_E4M3_FAST) {
      // Expose independent loads while retaining the ascending-token FMA chain.
#pragma unroll 8
      for (int i = 0; i < part_tokens; ++i) {
        const int physical_block = block_idx_shared[i];
        const int block_offset = block_offset_shared[i];
        const int64_t v_index =
            static_cast<int64_t>(physical_block) * v_block_stride +
            static_cast<int64_t>(block_offset) * v_token_stride +
            static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
        const float vv =
            flash_v100::load_kv_cache_float_unscaled<KV_DTYPE, true>(v_cache,
                                                                     v_index);
        acc = fmaf(scores_shared[i], vv, acc);
      }
    } else {
      for (int i = 0; i < part_tokens; ++i) {
        const int physical_block = block_idx_shared[i];
        const int block_offset = block_offset_shared[i];
        const int64_t v_index =
            static_cast<int64_t>(physical_block) * v_block_stride +
            static_cast<int64_t>(block_offset) * v_token_stride +
            static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
        const float vv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
            v_cache, v_index);
        acc = fmaf(scores_shared[i], vv, acc);
      }
    }
    const float out_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? inv_part_sum
                                : inv_part_sum * v_scale;
    if constexpr (std::is_same_v<PARTIAL_T, float>) {
      tmp_out[tmp_out_base + d] = acc * out_scale;
    } else {
      tmp_out[tmp_out_base + d] = __float2half(acc * out_scale);
    }
  }

  if (threadIdx.x == 0) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
    max_logits[stats_index] = part_max;
    exp_sums[stats_index] = part_sum;
  }
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM, int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          bool ALIGNED_PADDED_SMEM, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          bool E4M3_SHARED_LUT = false, typename PARTIAL_T = __half>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_partition_kernel_256_wide(
        const __half* __restrict__ q, const void* __restrict__ k_cache,
        const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
        float* __restrict__ max_logits, float* __restrict__ exp_sums,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale,
        const float k_scale, const float v_scale, const int route_seq_len_begin,
        const int route_seq_len_end, const int route_seq_len_final) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPVPanelDim = kXQATCStride;
  constexpr int kNumPVPanels = D / kPVPanelDim;
  constexpr int kQKPanelDim =
      QK_SW_PIPELINE ? kXQATCQKPipelinePanelDim : kXQATCStride;
  constexpr int kNumQKPanels = D / kQKPanelDim;
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  // Keep softmax P in fp32 through PV for FP16 and E4M3 KV, avoiding a half
  // round-trip before the existing fp32 FMA accumulation. Preserve E5M2's
  // established half-P numerical contract. This branch is compile-time only.
  constexpr bool kKeepPfp32 = KV_DTYPE != flash_v100::KV_CACHE_DTYPE_FP8_E5M2;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int pv_panel_d_stride_uint4 = kPVPanelDim / 8;
  constexpr int qk_smem_stride_uint4 =
      QK_SW_PIPELINE ? SmemLayout::kQKStride / 8 : kv_smem_stride_uint4;
  constexpr int qk_panel_d_stride_uint4 = kQKPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  constexpr bool kPVHalf2 = CONTIGUOUS_HKV1_LAYOUT &&
                            (BLOCK_SIZE == 800 || BLOCK_SIZE == 1568) &&
                            KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one full softmax/PV warp");
  static_assert(
      !QK_SW_PIPELINE || (GROUP_SIZE == 6 && (NUM_THREADS == 6 * kWarpSize ||
                                              NUM_THREADS == 8 * kWarpSize)),
      "The QK software pipeline requires a six- or eight-warp G6 "
      "kernel");
  static_assert(
      !E4M3_SHARED_LUT || KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
      "The shared conversion LUT is only valid for E4M3 KV");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    const int partition_size_begin =
        route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin;
    partition_size = seq_len < partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int start_token_idx = partition_idx * partition_size;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + partition_size - 1) / partition_size;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(partition_size, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  uint16_t* e4m3_lut = nullptr;
  if constexpr (E4M3_SHARED_LUT) {
    e4m3_lut = reinterpret_cast<uint16_t*>(smem_raw + sizeof(SmemLayout));
    for (int raw = tid; raw < 256; raw += NUM_THREADS) {
      e4m3_lut[raw] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw));
    }
  }
  __half* sQ = smem.q;
  __half* sK = smem.k_buffer(0);
  __half* sV = smem.v_buffer();
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  int partition_start_page = 0;
  int partition_page_offset = 0;
  if constexpr (PARTITION_PAGE_IDS) {
    int partition_page_count;
    if constexpr (BLOCK_SIZE == 800) {
      partition_start_page = start_token_idx / 800;
      partition_page_offset = start_token_idx - partition_start_page * 800;
      partition_page_count = (partition_page_offset + part_tokens + 799) / 800;
    } else {
      partition_start_page = start_token_idx / block_size;
      partition_page_offset =
          start_token_idx - partition_start_page * block_size;
      partition_page_count =
          (partition_page_offset + part_tokens + block_size - 1) / block_size;
    }
    for (int idx = tid; idx < partition_page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[partition_start_page + idx]);
    }
    __syncthreads();
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    int start_page = partition_start_page;
    int tile_page_offset;
    int page_count = 0;
    if constexpr (PARTITION_PAGE_IDS) {
      tile_page_offset = partition_page_offset + block_n * kXQATCBlockN;
    } else if constexpr (BLOCK_SIZE == 16) {
      start_page = tile_token_start >> 4;
      tile_page_offset = tile_token_start & 15;
      page_count = (tile_page_offset + valid_k_rows + 15) >> 4;
    } else if constexpr (BLOCK_SIZE == 784) {
      start_page = tile_token_start / 784;
      tile_page_offset = tile_token_start - start_page * 784;
      page_count = (tile_page_offset + valid_k_rows + 783) / 784;
    } else if constexpr (BLOCK_SIZE == 800) {
      start_page = tile_token_start / 800;
      tile_page_offset = tile_token_start - start_page * 800;
      page_count = 1 + (tile_page_offset + valid_k_rows > 800);
    } else if constexpr (BLOCK_SIZE == 1568) {
      start_page = tile_token_start / 1568;
      tile_page_offset = tile_token_start - start_page * 1568;
      page_count = (tile_page_offset + valid_k_rows + 1567) / 1568;
    } else {
      start_page = tile_token_start / block_size;
      tile_page_offset = tile_token_start - start_page * block_size;
      page_count =
          (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    }

    if constexpr (!PARTITION_PAGE_IDS) {
      for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
        smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
      }
      __syncthreads();
    }

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kXQATCBlockN) {
      const int valid_kv_tile_rows =
          min(kXQATCBlockN, valid_k_rows - kv_tile_start);
      // The QK fragments die before softmax/PV. This permits the compiler to
      // reuse their registers for the memory-latency-sensitive PV phase.
      volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::row_major>
          qk_a_frag;
      volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::col_major>
          qk_b_frag;
      volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
          qk_acc_frag;
      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        volta::fill_fragment(qk_acc_frag, 0.0f);
      }

      if constexpr (QK_SW_PIPELINE) {
        constexpr int kConsumerWarps = kXQATCBlockN / WMMA_N;
        constexpr int kProducerThreads =
            NUM_THREADS - kConsumerWarps * kWarpSize;
        const int producer_tid = tid - kConsumerWarps * kWarpSize;
        if (producer_tid >= 0) {
          load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                        kProducerThreads, KV_DTYPE,
                                        E4M3_SHARED_LUT>(
              smem.k_buffer(0), k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, 0, producer_tid, e4m3_lut);
        }
        __syncthreads();

#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          if (producer_tid >= 0 && panel_idx + 1 < kNumQKPanels) {
            load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                          kProducerThreads, KV_DTYPE,
                                          E4M3_SHARED_LUT>(
                smem.k_buffer((panel_idx + 1) & 1), k_cache, smem.page_ids,
                valid_kv_tile_rows, qk_panel_d_stride_uint4,
                qk_smem_stride_uint4, tile_page_offset, kv_tile_start,
                block_size, kv_head_idx, k_block_stride, k_token_stride,
                k_head_stride, panel_offset + kQKPanelDim, producer_tid,
                e4m3_lut);
          }
          if (warp_id < kConsumerWarps) {
            const int tile_n = warp_id * WMMA_N;
            const __half* sK_panel = smem.k_buffer(panel_idx & 1);
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag,
                  sK_panel + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      } else {
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                               KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
              sK, k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, panel_offset, threadIdx.x,
              e4m3_lut);
          for (int idx = tid + valid_kv_tile_rows * qk_panel_d_stride_uint4;
               idx < kXQATCBlockN * qk_panel_d_stride_uint4;
               idx += NUM_THREADS) {
            const int row = idx / qk_panel_d_stride_uint4;
            const int vec_col = idx % qk_panel_d_stride_uint4;
            reinterpret_cast<uint4*>(sK)[row * qk_smem_stride_uint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();

          if (warp_id < (kXQATCBlockN / WMMA_N)) {
            const int tile_n = warp_id * WMMA_N;
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag, sK + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      }

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
        for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
          qk_acc_frag.x[i] *= KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                  ? softmax_scale
                                  : softmax_scale * k_scale;
        }
        volta::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                 qk_acc_frag, kXQATCBlockN,
                                 volta::mem_row_major);
      }
      __syncthreads();
    }

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      float* sP_row_f = sS + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        if constexpr (kKeepPfp32) {
          reinterpret_cast<float4*>(sP_row_f)[vec_col] =
              make_float4(e0, e1, e2, e3);
        } else {
          packed_exp0 = __float22half2_rn(make_float2(e0, e1));
          packed_exp1 = __float22half2_rn(make_float2(e2, e3));
        }
      }

#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        if constexpr (kKeepPfp32) {
          sP_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
        } else {
          sP_row_h[c] =
              (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
        }
      }

#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if constexpr (!kKeepPfp32) {
        __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
        if (vec_col < vec_cols) {
          const int base_offset = vec_col * 2;
          sP_half2[base_offset] = packed_exp0;
          sP_half2[base_offset + 1] = packed_exp1;
        }
      }

      if (block_n > 0) {
#pragma unroll
        for (int i = 0; i < kAccumsPerThread; ++i) {
          out_acc[i] *= exp_diff;
        }
      }
    }
    __syncthreads();

    for (int panel_idx = 0; panel_idx < kNumPVPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPVPanelDim;
      for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
           kv_tile_start += kXQATCBlockN) {
        const int valid_kv_tile_rows =
            min(kXQATCBlockN, valid_k_rows - kv_tile_start);
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
            sV, v_cache, smem.page_ids, valid_kv_tile_rows,
            pv_panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, v_block_stride,
            v_token_stride, v_head_stride, panel_offset, threadIdx.x, e4m3_lut);
        for (int idx = tid + valid_kv_tile_rows * pv_panel_d_stride_uint4;
             idx < kXQATCBlockN * pv_panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / pv_panel_d_stride_uint4;
          const int vec_col = idx % pv_panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const __half* sP_row = sP + row * kXQATCBlockN + kv_tile_start;
          const float* sP_row_f = sS + row * kXQATCBlockN + kv_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCBlockN; ++token) {
            if (token >= valid_kv_tile_rows) {
              break;
            }
            float prob;
            if constexpr (kKeepPfp32) {
              prob = sP_row_f[token];
            } else {
              prob = __half2float(sP_row[token]);
            }
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
            if constexpr (kPVHalf2) {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / (2 * kWarpSize));
                   ++d_iter) {
                const int local_d2 = lane_id + d_iter * kWarpSize;
                const float2 value = __half22float2(
                    reinterpret_cast<const __half2*>(sV_row)[local_d2]);
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter * 2;
                out_acc[acc_idx] = fmaf(prob, value.x, out_acc[acc_idx]);
                out_acc[acc_idx + 1] =
                    fmaf(prob, value.y, out_acc[acc_idx + 1]);
              }
            } else {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / kWarpSize);
                   ++d_iter) {
                const int local_d = lane_id + d_iter * kWarpSize;
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter;
                out_acc[acc_idx] =
                    fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    PARTIAL_T* tmp_out_ptr =
        tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
        static_cast<int64_t>(head_idx) * tmp_out_stride1 +
        static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    const float output_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                   ? inv_row_sum
                                   : inv_row_sum * v_scale;
    if constexpr (kPVHalf2) {
#pragma unroll
      for (int acc_idx = 0; acc_idx < kAccumsPerThread; ++acc_idx) {
        constexpr int kAccumsPerPanel = kPVPanelDim / kWarpSize;
        const int panel_idx = acc_idx / kAccumsPerPanel;
        const int panel_acc_idx = acc_idx % kAccumsPerPanel;
        const int d_iter = panel_acc_idx / 2;
        const int component = panel_acc_idx & 1;
        const int d = panel_idx * kPVPanelDim + d_iter * (2 * kWarpSize) +
                      thread_in_row * 2 + component;
        const float value = out_acc[acc_idx] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    } else {
      for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
        const float value = out_acc[d / kWarpSize] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}
template <int D, int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          typename PARTIAL_T = __half>
__global__ void flash_attention_decode_reduce_kernel(
    const PARTIAL_T* __restrict__ tmp_out, const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, __half* __restrict__ out,
    const int batch_size, const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_partition_size_begin,
    const int route_seq_len_begin, const int route_seq_len_end,
    const int route_seq_len_final, const float* __restrict__ sinks) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    partition_size = seq_len < route_partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  (void)active_num_partitions;

  if (seq_len <= 0 || num_partitions <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[static_cast<int64_t>(batch_idx) * out_stride0 +
          static_cast<int64_t>(head_idx) * out_stride1 + d] = __float2half(0.f);
    }
    return;
  }

  extern __shared__ float shared_mem[];
  float* max_shared = shared_mem;
  float* weight_shared = shared_mem + max_num_partitions;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float m = max_logits[stats_index];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float block_max = block_reduce_max<kWarpsPerBlock>(local_max);
  // Learned attention sink (natural-log units, the scaled-score domain of
  // max_logits): one extra softmax denominator term exp(sink) with no value
  // row. Folding it into the reference max keeps exp() in range. A null
  // pointer leaves the sink-free reduction unchanged.
  const float sink = sinks != nullptr ? sinks[head_idx] : 0.f;
  const float global_max =
      sinks != nullptr ? fmaxf(block_max, sink) : block_max;

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float weight =
        exp_sums[stats_index] * __expf(max_shared[i] - global_max);
    weight_shared[i] = weight;
    local_sum += weight;
  }
  float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (sinks != nullptr) {
    global_sum += __expf(sink - global_max);
  }
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncthreads();

  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                           static_cast<int64_t>(head_idx) * out_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < num_partitions; ++i) {
      acc = fmaf(weight_shared[i],
                 static_cast<float>(
                     tmp_out[tmp_out_base +
                             static_cast<int64_t>(i) * tmp_out_stride2 + d]),
                 acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
__global__ void flash_attention_decode_xqa_reduce_stats_kernel(
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t stats_stride0, const int64_t stats_stride1,
    const int route_seq_len_begin, const int route_seq_len_end,
    const int route_seq_len_final) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    return;
  }

  extern __shared__ float max_shared[];
  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float m = max_logits[stats_base + i];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float weight =
        exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
    max_logits[stats_base + i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (threadIdx.x == 0) {
    // The partition sum is dead after weights have been materialized.
    exp_sums[stats_base] = global_sum;
  }
}

template <int D, int PARTITION_SIZE, int D_TILE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, typename PARTIAL_T = __half>
__global__ void flash_attention_decode_xqa_reduce_output_kernel(
    const PARTIAL_T* __restrict__ tmp_out, const float* __restrict__ weights,
    const float* __restrict__ global_sums, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  static_assert(D_TILE > 0 && D_TILE <= kWarpSize,
                "Split-reduce dimension tile must fit in one warp");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = blockIdx.z * D_TILE + threadIdx.x;
  if (batch_idx >= batch_size || head_idx >= num_heads_q || d >= D) {
    return;
  }

  const int64_t out_index = static_cast<int64_t>(batch_idx) * out_stride0 +
                            static_cast<int64_t>(head_idx) * out_stride1 + d;
  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    out[out_index] = __float2half(0.f);
    return;
  }

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const float global_sum = global_sums[stats_base];
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;

  float acc = 0.f;
  for (int i = 0; i < num_partitions; ++i) {
    acc = fmaf(weights[stats_base + i],
               static_cast<float>(
                   tmp_out[tmp_out_base +
                           static_cast<int64_t>(i) * tmp_out_stride2 + d]),
               acc);
  }
  out[out_index] = __float2half(acc * inv_global_sum);
}

// Query heads per KV head that the grouped D128 decode kernel covers.
constexpr int kGqaDecodeGroup = 8;
constexpr int kGqaDecodeHeadDim = 128;

__device__ __forceinline__ float warp_allreduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_allreduce_max(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ void load_half4_as_float(const __half* ptr,
                                                    float (&out)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
  const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
  const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
  out[0] = lo.x;
  out[1] = lo.y;
  out[2] = hi.x;
  out[3] = hi.y;
}

__device__ __forceinline__ void load_half8_as_float(const __half* ptr,
                                                    float (&out)[8]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f =
        __half22float2(*reinterpret_cast<const __half2*>(&words[j]));
    out[2 * j] = f.x;
    out[2 * j + 1] = f.y;
  }
}

// Grouped-query paged decode partition for head_dim 128, fp16 KV and eight
// query heads per KV head. One CTA covers the whole query group of one KV
// head, so each K/V row is read once per group instead of once per query
// head, and no thread walks the partition alone:
//   QK      warp w scores tokens w, w + 8, ...; each lane holds four dims of
//           the eight queries and of the key row (coalesced 256-byte rows),
//           and a reduce-scatter butterfly leaves head (lane >> 2) & 7's
//           score on four lanes (9 shuffles per token for 8 heads).
//   softmax warp w owns query head w of the group.
//   PV      warp w accumulates head w over the partition, four dims a lane.
// Its partials and (max, sum) statistics have exactly the layout and meaning
// of flash_attention_decode_partition_kernel's, so the generic reducer,
// which applies learned attention sinks, combines them unchanged.
template <int PARTITION_SIZE>
__global__ void __launch_bounds__(kThreadsPerBlock)
    flash_attention_decode_gqa8_d128_partition_kernel(
        const __half* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, __half* __restrict__ tmp_out,
        float* __restrict__ max_logits, float* __restrict__ exp_sums,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_kv, const int block_size, const int64_t q_stride0,
        const int64_t q_stride1, const int64_t tmp_out_stride0,
        const int64_t tmp_out_stride1, const int64_t tmp_out_stride2,
        const int64_t stats_stride0, const int64_t stats_stride1,
        const int64_t k_block_stride, const int64_t k_token_stride,
        const int64_t k_head_stride, const int64_t v_block_stride,
        const int64_t v_token_stride, const int64_t v_head_stride,
        const float softmax_scale, const int window_size_left,
        const int window_size_right) {
  constexpr int G = kGqaDecodeGroup;
  static_assert(kWarpsPerBlock == G, "one warp per query head of the group");
  static_assert(kGqaDecodeHeadDim == 4 * kWarpSize, "four dims per lane");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }
  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int query_pos = seq_len - 1;
  const int min_token_idx =
      window_size_left >= 0 ? max(0, query_pos - window_size_left) : 0;
  const int max_token_idx =
      window_size_right >= 0 ? min(seq_len - 1, query_pos + window_size_right)
                             : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int dim = 4 * lane;
  // Softmax and PV: this warp's query head.
  const int head_idx = kv_head_idx * G + warp;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride2;
  const int64_t stats_index = static_cast<int64_t>(batch_idx) * stats_stride0 +
                              static_cast<int64_t>(head_idx) * stats_stride1 +
                              partition_idx;
  if (part_start >= part_end) {
    *reinterpret_cast<uint2*>(tmp_out + tmp_out_base + dim) = make_uint2(0, 0);
    if (lane == 0) {
      max_logits[stats_index] = -1.0e20f;
      exp_sums[stats_index] = 0.f;
    }
    return;
  }
  const int part_tokens = part_end - part_start;

  // The group's scores during QK and softmax, then the warps' PV partials.
  constexpr int kScoreFloats = G * PARTITION_SIZE;
  constexpr int kPartialFloats = kWarpsPerBlock * G * kGqaDecodeHeadDim;
  __shared__ __align__(16) float
      pv_shared[kScoreFloats > kPartialFloats ? kScoreFloats : kPartialFloats];
  float (*scores_shared)[PARTITION_SIZE] =
      reinterpret_cast<float (*)[PARTITION_SIZE]>(pv_shared);
  __shared__ float inv_sum_shared[G];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = part_start + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] =
        block_table[static_cast<int64_t>(batch_idx) * max_num_blocks +
                    logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }

  // Each half-warp owns one token at a time; its 16 lanes hold eight dims
  // each of the key row and of the group's eight queries (fp32 registers).
  const int pair_half = lane >> 4;
  const int half_lane = lane & 15;
  const int dim8 = 8 * half_lane;
  float q_reg[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    load_half8_as_float(
        q + static_cast<int64_t>(batch_idx) * q_stride0 +
            static_cast<int64_t>(kv_head_idx * G + g) * q_stride1 + dim8,
        q_reg[g]);
  }
  __syncthreads();

  const bool upper8 = (half_lane & 8) != 0;
  const bool upper4 = (half_lane & 4) != 0;
  const bool upper2 = (half_lane & 2) != 0;
  const int score_head = (half_lane >> 1) & (G - 1);
  const int64_t k_head_base = static_cast<int64_t>(kv_head_idx) * k_head_stride;
  // Token pairs keep both halves in the loop together for the shuffles.
#pragma unroll 2
  for (int pair = warp; 2 * pair < part_tokens; pair += kWarpsPerBlock) {
    const int t = 2 * pair + pair_half;
    const bool valid = t < part_tokens;
    float k[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (valid) {
      load_half8_as_float(
          k_cache + static_cast<int64_t>(block_idx_shared[t]) * k_block_stride +
              static_cast<int64_t>(block_offset_shared[t]) * k_token_stride +
              k_head_base + dim8,
          k);
    }
    float partial[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float acc = q_reg[g][0] * k[0];
#pragma unroll
      for (int j = 1; j < 8; ++j) {
        acc = fmaf(q_reg[g][j], k[j], acc);
      }
      partial[g] = acc;
    }
    // Reduce-scatter over half-lane bits 3, 2 and 1: each round keeps half
    // of the heads and adds the partner's partial sums for them; bit 0
    // completes the sum.
    float half_sums[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float send = upper8 ? partial[j] : partial[j + 4];
      const float keep = upper8 ? partial[j + 4] : partial[j];
      half_sums[j] = keep + __shfl_xor_sync(0xffffffff, send, 8);
    }
    float quarter_sums[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float send = upper4 ? half_sums[j] : half_sums[j + 2];
      const float keep = upper4 ? half_sums[j + 2] : half_sums[j];
      quarter_sums[j] = keep + __shfl_xor_sync(0xffffffff, send, 4);
    }
    float score =
        (upper2 ? quarter_sums[1] : quarter_sums[0]) +
        __shfl_xor_sync(0xffffffff, upper2 ? quarter_sums[0] : quarter_sums[1],
                        2);
    score += __shfl_xor_sync(0xffffffff, score, 1);
    if (valid && (half_lane & 1) == 0) {
      scores_shared[score_head][t] = score * softmax_scale;
    }
  }
  __syncthreads();

  float* head_scores = scores_shared[warp];
  float part_max = -1.0e20f;
  for (int i = lane; i < part_tokens; i += kWarpSize) {
    part_max = fmaxf(part_max, head_scores[i]);
  }
  part_max = warp_allreduce_max(part_max);
  float part_sum = 0.f;
  for (int i = lane; i < part_tokens; i += kWarpSize) {
    const float p = __expf(head_scores[i] - part_max);
    head_scores[i] = p;
    part_sum += p;
  }
  part_sum = warp_allreduce_sum(part_sum);
  if (lane == 0) {
    inv_sum_shared[warp] = part_sum > 0.f ? 1.f / part_sum : 0.f;
  }
  __syncthreads();

  // PV: warp w accumulates all eight heads over its eighth of the
  // partition, four dims a lane, so each V row is read once per CTA.
  constexpr int kTokensPerWarp = PARTITION_SIZE / kWarpsPerBlock;
  const int t_begin = warp * kTokensPerWarp;
  const int t_end = min(t_begin + kTokensPerWarp, part_tokens);
  float acc[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[g][j] = 0.f;
    }
  }
  const int64_t v_head_base = static_cast<int64_t>(kv_head_idx) * v_head_stride;
#pragma unroll 4
  for (int t = t_begin; t < t_end; ++t) {
    float v[4];
    load_half4_as_float(
        v_cache + static_cast<int64_t>(block_idx_shared[t]) * v_block_stride +
            static_cast<int64_t>(block_offset_shared[t]) * v_token_stride +
            v_head_base + dim,
        v);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float p = scores_shared[g][t];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[g][j] = fmaf(p, v[j], acc[g][j]);
      }
    }
  }
  __syncthreads();             // every warp has read the probabilities
  float* partial = pv_shared;  // [warp][head][dim]
#pragma unroll
  for (int g = 0; g < G; ++g) {
    *reinterpret_cast<float4*>(partial + (warp * G + g) * kGqaDecodeHeadDim +
                               dim) =
        make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
  }
  __syncthreads();
  // Thread (warp = head, lane) sums its four dims over the warps' slices.
  float total[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int w = 0; w < kWarpsPerBlock; ++w) {
    const float4 slice = *reinterpret_cast<const float4*>(
        partial + (w * G + warp) * kGqaDecodeHeadDim + dim);
    total[0] += slice.x;
    total[1] += slice.y;
    total[2] += slice.z;
    total[3] += slice.w;
  }
  const float inv_part_sum = inv_sum_shared[warp];
  const __half2 lo =
      __floats2half2_rn(total[0] * inv_part_sum, total[1] * inv_part_sum);
  const __half2 hi =
      __floats2half2_rn(total[2] * inv_part_sum, total[3] * inv_part_sum);
  uint2 packed;
  packed.x = *reinterpret_cast<const uint32_t*>(&lo);
  packed.y = *reinterpret_cast<const uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(tmp_out + tmp_out_base + dim) = packed;
  if (lane == 0) {
    max_logits[stats_index] = part_max;
    exp_sums[stats_index] = part_sum;
  }
}

// Cross-partition reducer for the grouped D128 decode. Same math as
// flash_attention_decode_reduce_kernel (sink folded into the reference max,
// exp(sink) added to the denominator), but it starts at the first partition
// that overlaps the sliding window, since earlier ones carry no weight, and
// the eight warps split the partitions instead of one thread per dim walking
// all of them. One CTA per (sequence, query head).
template <int PARTITION_SIZE>
__global__ void __launch_bounds__(kThreadsPerBlock)
    flash_attention_decode_gqa8_d128_reduce_kernel(
        const __half* __restrict__ tmp_out,
        const float* __restrict__ max_logits,
        const float* __restrict__ exp_sums, const int* __restrict__ seq_lens,
        __half* __restrict__ out, const int batch_size,
        const int max_num_partitions, const int num_heads_q,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t out_stride0,
        const int64_t out_stride1, const int window_size_left,
        const float* __restrict__ sinks) {
  constexpr int D = kGqaDecodeHeadDim;
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int dim = 4 * lane;
  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                           static_cast<int64_t>(head_idx) * out_stride1;
  const int seq_len = seq_lens[batch_idx];
  const int num_partitions =
      min(max_num_partitions, (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  if (seq_len <= 0 || num_partitions <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[out_base + d] = __float2half(0.f);
    }
    return;
  }
  const int first_partition =
      window_size_left >= 0
          ? min(num_partitions - 1,
                max(0, seq_len - 1 - window_size_left) / PARTITION_SIZE)
          : 0;
  const int count = num_partitions - first_partition;

  extern __shared__ float shared_mem[];
  float* max_shared = shared_mem;
  float* weight_shared = shared_mem + max_num_partitions;
  __shared__ float partial_out[kWarpsPerBlock][D];

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1 +
                             first_partition;
  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float m = max_logits[stats_base + i];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float block_max = block_reduce_max<kWarpsPerBlock>(local_max);
  const float sink = sinks != nullptr ? sinks[head_idx] : 0.f;
  const float global_max =
      sinks != nullptr ? fmaxf(block_max, sink) : block_max;

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float weight =
        exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
    weight_shared[i] = weight;
    local_sum += weight;
  }
  float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (sinks != nullptr) {
    global_sum += __expf(sink - global_max);
  }
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncthreads();

  const __half* tmp_base =
      tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(first_partition) * tmp_out_stride2 + dim;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int i = warp; i < count; i += kWarpsPerBlock) {
    float v[4];
    load_half4_as_float(tmp_base + static_cast<int64_t>(i) * tmp_out_stride2,
                        v);
    const float w = weight_shared[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[j] = fmaf(w, v[j], acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    partial_out[warp][dim + j] = acc[j];
  }
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += partial_out[w][d];
    }
    out[out_base + d] = __float2half(total * inv_global_sum);
  }
}

// Single-kernel grouped D128 decode with sinks: the partition kernel and the
// cross-partition reducer above fused into one launch per layer.
//
// The two-kernel route launches one CTA per (sequence, KV head, partition)
// over the whole captured envelope (1024 partitions at a 262K workspace), so
// at short context almost every CTA exits at once, and the reducer is a
// second launch whose latency chain sits behind the partition kernel. Here
// the grid is (sequence, KV head, Z) with Z sized on the host to fill the
// device once (independent of sequence lengths, so it is CUDA-graph safe),
// and the in-window token range [lo, seq_len) is split on the device into
// balanced chunks of at most MAX_CHUNK tokens:
//   chunks  N = min(Z * ceil(len / (Z * MAX_CHUNK)), ceil(len / kMinChunk))
//   CTAs    W = min(Z, N) work; CTA z takes chunks z, z + W, ...
// A CTA combines its chunks online in fp32 registers and writes one partial
// (fp16 normalised output, fp32 max and sum: the reducer's inputs). The last
// CTA of the (sequence, KV head) to arrive, found with a device-scope atomic
// counter, combines the W partials in index order (so the result does not
// depend on which CTA arrives last), applies the sink exactly as
// flash_attention_decode_gqa8_d128_reduce_kernel does, writes the output and
// resets the counter for the next launch. With W == 1 (short windows on a
// small grid) the CTA writes the output directly from registers.
constexpr int kGqaFusedMaxChunk = 256;
constexpr int kGqaFusedMinChunk = 64;
constexpr int kGqaFusedMaxCtasPerHead = 256;

template <int MAX_CHUNK>
__global__ void __launch_bounds__(kThreadsPerBlock, 2)
    flash_attention_decode_gqa8_d128_fused_kernel(
        const __half* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, __half* __restrict__ out,
        __half* __restrict__ tmp_out, float* __restrict__ max_logits,
        float* __restrict__ exp_sums, unsigned int* __restrict__ counters,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const float* __restrict__ sinks, const int max_num_blocks,
        const int num_heads_kv, const int block_size, const int64_t q_stride0,
        const int64_t q_stride1, const int64_t out_stride0,
        const int64_t out_stride1, const int64_t tmp_out_stride0,
        const int64_t tmp_out_stride1, const int64_t tmp_out_stride2,
        const int64_t stats_stride0, const int64_t stats_stride1,
        const int64_t k_block_stride, const int64_t k_token_stride,
        const int64_t k_head_stride, const int64_t v_block_stride,
        const int64_t v_token_stride, const int64_t v_head_stride,
        const float softmax_scale, const int window_size_left) {
  constexpr int G = kGqaDecodeGroup;
  constexpr int D = kGqaDecodeHeadDim;
  static_assert(kWarpsPerBlock == G, "one warp per query head of the group");
  static_assert(D == 4 * kWarpSize, "four dims per lane");
  static_assert(MAX_CHUNK % kWarpsPerBlock == 0, "whole PV slices");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int cta = blockIdx.z;
  const int num_ctas = gridDim.z;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int dim = 4 * lane;
  // Softmax, PV output and the final reduction: this warp's query head.
  const int head_idx = kv_head_idx * G + warp;
  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                           static_cast<int64_t>(head_idx) * out_stride1;

  const int seq_len = seq_lens[batch_idx];
  if (seq_len <= 0) {
    // As the reducer: an empty row decodes to zeros.
    if (cta == 0) {
      *reinterpret_cast<uint2*>(out + out_base + dim) = make_uint2(0, 0);
    }
    return;
  }
  // Decode queries sit at seq_len - 1, so the right window never clips.
  const int lo =
      window_size_left >= 0 ? max(0, seq_len - 1 - window_size_left) : 0;
  const int len = seq_len - lo;
  const int span = num_ctas * MAX_CHUNK;
  const int iters = (len + span - 1) / span;
  const int num_chunks =
      min(num_ctas * iters, (len + kGqaFusedMinChunk - 1) / kGqaFusedMinChunk);
  const int active = min(num_ctas, num_chunks);
  if (cta >= active) {
    return;
  }

  // The group's scores during QK and softmax, then the warps' PV partials,
  // then the last CTA's per-head (max, weight) staging.
  constexpr int kScoreFloats = G * MAX_CHUNK;
  constexpr int kPartialFloats = kWarpsPerBlock * G * D;
  static_assert(kPartialFloats >= G * 2 * kGqaFusedMaxCtasPerHead,
                "reduction staging fits the PV buffer");
  __shared__ __align__(16) float
      pv_shared[kScoreFloats > kPartialFloats ? kScoreFloats : kPartialFloats];
  float (*scores_shared)[MAX_CHUNK] =
      reinterpret_cast<float (*)[MAX_CHUNK]>(pv_shared);
  __shared__ int block_idx_shared[MAX_CHUNK];
  __shared__ int block_offset_shared[MAX_CHUNK];
  __shared__ int is_last_shared;
  // The group's queries, staged once: QK reloads them from shared memory for
  // each chunk instead of holding 64 registers across PV (two CTAs per SM).
  __shared__ __align__(16) __half q_shared[G * D];
  for (int i = threadIdx.x; i < G * D / 8; i += blockDim.x) {
    const int g = i / (D / 8);
    const int d8 = 8 * (i % (D / 8));
    *reinterpret_cast<uint4*>(q_shared + g * D + d8) =
        *reinterpret_cast<const uint4*>(
            q + static_cast<int64_t>(batch_idx) * q_stride0 +
            static_cast<int64_t>(kv_head_idx * G + g) * q_stride1 + d8);
  }

  // Each half-warp owns one token at a time in QK; its 16 lanes hold eight
  // dims each of the key row and of the group's eight queries.
  const int pair_half = lane >> 4;
  const int half_lane = lane & 15;
  const int dim8 = 8 * half_lane;
  const bool upper8 = (half_lane & 8) != 0;
  const bool upper4 = (half_lane & 4) != 0;
  const bool upper2 = (half_lane & 2) != 0;
  const int score_head = (half_lane >> 1) & (G - 1);
  const int64_t k_head_base = static_cast<int64_t>(kv_head_idx) * k_head_stride;
  const int64_t v_head_base = static_cast<int64_t>(kv_head_idx) * v_head_stride;
  const int64_t table_base = static_cast<int64_t>(batch_idx) * max_num_blocks;

  // Running (max, sum, unnormalised output) of this warp's head over the
  // CTA's chunks, four dims a lane.
  float run_max = -1.0e20f;
  float run_sum = 0.f;
  float run_acc[4] = {0.f, 0.f, 0.f, 0.f};

  for (int chunk = cta; chunk < num_chunks; chunk += active) {
    const int chunk_start =
        lo + static_cast<int>((static_cast<int64_t>(chunk) * len) / num_chunks);
    const int chunk_end =
        lo +
        static_cast<int>((static_cast<int64_t>(chunk + 1) * len) / num_chunks);
    const int n = chunk_end - chunk_start;  // 1..MAX_CHUNK
    __syncthreads();  // the previous chunk's shared reads are done
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int token_idx = chunk_start + i;
      const int logical_block = token_idx / block_size;
      block_idx_shared[i] = block_table[table_base + logical_block];
      block_offset_shared[i] = token_idx - logical_block * block_size;
    }
    __syncthreads();

    float q_reg[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      load_half8_as_float(q_shared + g * D + dim8, q_reg[g]);
    }
    // QK: token pairs keep both halves in the loop together for the shuffles.
#pragma unroll 2
    for (int pair = warp; 2 * pair < n; pair += kWarpsPerBlock) {
      const int t = 2 * pair + pair_half;
      const bool valid = t < n;
      float k[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (valid) {
        load_half8_as_float(
            k_cache +
                static_cast<int64_t>(block_idx_shared[t]) * k_block_stride +
                static_cast<int64_t>(block_offset_shared[t]) * k_token_stride +
                k_head_base + dim8,
            k);
      }
      float partial[G];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        float acc = q_reg[g][0] * k[0];
#pragma unroll
        for (int j = 1; j < 8; ++j) {
          acc = fmaf(q_reg[g][j], k[j], acc);
        }
        partial[g] = acc;
      }
      float half_sums[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float send = upper8 ? partial[j] : partial[j + 4];
        const float keep = upper8 ? partial[j + 4] : partial[j];
        half_sums[j] = keep + __shfl_xor_sync(0xffffffff, send, 8);
      }
      float quarter_sums[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float send = upper4 ? half_sums[j] : half_sums[j + 2];
        const float keep = upper4 ? half_sums[j + 2] : half_sums[j];
        quarter_sums[j] = keep + __shfl_xor_sync(0xffffffff, send, 4);
      }
      float score =
          (upper2 ? quarter_sums[1] : quarter_sums[0]) +
          __shfl_xor_sync(0xffffffff,
                          upper2 ? quarter_sums[0] : quarter_sums[1], 2);
      score += __shfl_xor_sync(0xffffffff, score, 1);
      if (valid && (half_lane & 1) == 0) {
        scores_shared[score_head][t] = score * softmax_scale;
      }
    }
    __syncthreads();

    float* head_scores = scores_shared[warp];
    float part_max = -1.0e20f;
    for (int i = lane; i < n; i += kWarpSize) {
      part_max = fmaxf(part_max, head_scores[i]);
    }
    part_max = warp_allreduce_max(part_max);
    float part_sum = 0.f;
    for (int i = lane; i < n; i += kWarpSize) {
      const float p = __expf(head_scores[i] - part_max);
      head_scores[i] = p;
      part_sum += p;
    }
    part_sum = warp_allreduce_sum(part_sum);
    __syncthreads();

    // PV: warp w accumulates all eight heads over its eighth of the chunk.
    const int tokens_per_warp = (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int t_begin = warp * tokens_per_warp;
    const int t_end = min(t_begin + tokens_per_warp, n);
    float acc[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[g][j] = 0.f;
      }
    }
#pragma unroll 4
    for (int t = t_begin; t < t_end; ++t) {
      float v[4];
      load_half4_as_float(
          v_cache + static_cast<int64_t>(block_idx_shared[t]) * v_block_stride +
              static_cast<int64_t>(block_offset_shared[t]) * v_token_stride +
              v_head_base + dim,
          v);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float p = scores_shared[g][t];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[g][j] = fmaf(p, v[j], acc[g][j]);
        }
      }
    }
    __syncthreads();             // every warp has read the probabilities
    float* partial = pv_shared;  // [warp][head][dim]
#pragma unroll
    for (int g = 0; g < G; ++g) {
      *reinterpret_cast<float4*>(partial + (warp * G + g) * D + dim) =
          make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
    }
    __syncthreads();
    float total[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      const float4 slice =
          *reinterpret_cast<const float4*>(partial + (w * G + warp) * D + dim);
      total[0] += slice.x;
      total[1] += slice.y;
      total[2] += slice.z;
      total[3] += slice.w;
    }
    // Online combine (part_max and part_sum are uniform across the warp).
    const float new_max = fmaxf(run_max, part_max);
    const float run_scale = __expf(run_max - new_max);
    const float part_scale = __expf(part_max - new_max);
    run_sum = run_sum * run_scale + part_sum * part_scale;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      run_acc[j] = run_acc[j] * run_scale + total[j] * part_scale;
    }
    run_max = new_max;
  }

  const float sink = sinks[head_idx];
  if (active == 1) {
    // The whole window in one CTA: apply the sink and finish here.
    const float global_max = fmaxf(run_max, sink);
    const float weight = run_sum * __expf(run_max - global_max);
    const float global_sum = weight + __expf(sink - global_max);
    const float scale = global_sum > 0.f && run_sum > 0.f
                            ? weight / (run_sum * global_sum)
                            : 0.f;
    const __half2 lo2 =
        __floats2half2_rn(run_acc[0] * scale, run_acc[1] * scale);
    const __half2 hi2 =
        __floats2half2_rn(run_acc[2] * scale, run_acc[3] * scale);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t*>(&lo2);
    packed.y = *reinterpret_cast<const uint32_t*>(&hi2);
    *reinterpret_cast<uint2*>(out + out_base + dim) = packed;
    return;
  }

  // Publish this CTA's partial exactly as the partition kernel would.
  const int64_t tmp_head_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const int64_t stats_head_base =
      static_cast<int64_t>(batch_idx) * stats_stride0 +
      static_cast<int64_t>(head_idx) * stats_stride1;
  {
    const float inv_run_sum = run_sum > 0.f ? 1.f / run_sum : 0.f;
    const __half2 lo2 =
        __floats2half2_rn(run_acc[0] * inv_run_sum, run_acc[1] * inv_run_sum);
    const __half2 hi2 =
        __floats2half2_rn(run_acc[2] * inv_run_sum, run_acc[3] * inv_run_sum);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t*>(&lo2);
    packed.y = *reinterpret_cast<const uint32_t*>(&hi2);
    *reinterpret_cast<uint2*>(tmp_out + tmp_head_base +
                              static_cast<int64_t>(cta) * tmp_out_stride2 +
                              dim) = packed;
    if (lane == 0) {
      max_logits[stats_head_base + cta] = run_max;
      exp_sums[stats_head_base + cta] = run_sum;
    }
  }
  // Release: every thread's partial is visible device-wide before thread 0
  // counts this CTA in.
  __threadfence();
  __syncthreads();
  unsigned int* counter =
      counters + static_cast<int64_t>(batch_idx) * num_heads_kv + kv_head_idx;
  if (threadIdx.x == 0) {
    const unsigned int arrived = atomicAdd(counter, 1u);
    is_last_shared = arrived == static_cast<unsigned int>(active - 1);
  }
  __syncthreads();
  if (!is_last_shared) {
    return;
  }
  __threadfence();

  // Last CTA: warp w combines head w's W partials in index order. Loads use
  // the L2-coherent path (ld.global.cg): the partials were written by other
  // SMs during this kernel.
  float* max_staged = pv_shared + warp * 2 * kGqaFusedMaxCtasPerHead;
  float* weight_staged = max_staged + kGqaFusedMaxCtasPerHead;
  float local_max = -1.0e20f;
  for (int i = lane; i < active; i += kWarpSize) {
    const float m = __ldcg(max_logits + stats_head_base + i);
    max_staged[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = fmaxf(warp_allreduce_max(local_max), sink);
  float local_sum = 0.f;
  for (int i = lane; i < active; i += kWarpSize) {
    const float weight = __ldcg(exp_sums + stats_head_base + i) *
                         __expf(max_staged[i] - global_max);
    weight_staged[i] = weight;
    local_sum += weight;
  }
  const float global_sum =
      warp_allreduce_sum(local_sum) + __expf(sink - global_max);
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncwarp();

  const __half* tmp_base = tmp_out + tmp_head_base + dim;
  float out_acc[4] = {0.f, 0.f, 0.f, 0.f};
  // Sixteen independent L2 loads in flight per lane: the tail is latency
  // bound (64 partials at 4K on a 160-slot grid).
#pragma unroll 16
  for (int i = 0; i < active; ++i) {
    const uint2 raw = __ldcg(reinterpret_cast<const uint2*>(
        tmp_base + static_cast<int64_t>(i) * tmp_out_stride2));
    const float2 v01 =
        __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
    const float2 v23 =
        __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
    const float w = weight_staged[i];
    out_acc[0] = fmaf(w, v01.x, out_acc[0]);
    out_acc[1] = fmaf(w, v01.y, out_acc[1]);
    out_acc[2] = fmaf(w, v23.x, out_acc[2]);
    out_acc[3] = fmaf(w, v23.y, out_acc[3]);
  }
  const __half2 lo2 = __floats2half2_rn(out_acc[0] * inv_global_sum,
                                        out_acc[1] * inv_global_sum);
  const __half2 hi2 = __floats2half2_rn(out_acc[2] * inv_global_sum,
                                        out_acc[3] * inv_global_sum);
  uint2 packed;
  packed.x = *reinterpret_cast<const uint32_t*>(&lo2);
  packed.y = *reinterpret_cast<const uint32_t*>(&hi2);
  *reinterpret_cast<uint2*>(out + out_base + dim) = packed;
  if (threadIdx.x == 0) {
    // Ready for the next launch; kernel boundaries order this store.
    *counter = 0u;
  }
}

// CTAs per (sequence, KV head) for the fused kernel: enough to fill every SM
// once at the kernel's occupancy, within the workspace's partial slots. A
// function of the launch shape and the device only, never of sequence
// lengths, so a captured graph replays correctly for any length.
inline int gqa8_d128_fused_ctas_per_head(const int batch_size,
                                         const int num_heads_kv,
                                         const int partial_slots) {
  static const int blocks_per_sm = [] {
    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            flash_attention_decode_gqa8_d128_fused_kernel<kGqaFusedMaxChunk>,
            kThreadsPerBlock, 0) != cudaSuccess) {
      (void)cudaGetLastError();
      blocks = 1;
    }
    return std::max(blocks, 1);
  }();
  const int slots =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount *
      blocks_per_sm;
  const int rows = std::max(1, batch_size * num_heads_kv);
  const int ctas = (slots + rows - 1) / rows;
  return std::max(1, std::min({ctas, partial_slots, kGqaFusedMaxCtasPerHead}));
}

// The grouped kernel loads q, K, V and the partials as aligned half8 vectors.
inline bool gqa8_d128_decode_aligned(const at::Tensor& t) {
  if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0) {
    return false;
  }
  for (int64_t i = 0; i + 1 < t.dim(); ++i) {
    if (t.stride(i) % 8 != 0) {
      return false;
    }
  }
  return t.stride(-1) == 1;
}

template <int D, int PARTITION_SIZE, int SEQ_LEN_ROUTE, typename PARTIAL_T>
bool gqa8_d128_decode_eligible(const at::Tensor& q, const at::Tensor& k_cache,
                               const at::Tensor& v_cache,
                               const at::Tensor& tmp_out, bool use_anchored,
                               const float* sinks) {
  if constexpr (D != kGqaDecodeHeadDim || !std::is_same_v<PARTIAL_T, __half> ||
                SEQ_LEN_ROUTE != kXQARouteAllSeqLens) {
    return false;
  } else {
    // Contained to layers with learned sinks (the Flint route); sink-free
    // layers keep the per-head kernel and its validated numerics.
    return sinks != nullptr && !use_anchored &&
           q.size(1) == kGqaDecodeGroup * k_cache.size(2) &&
           gqa8_d128_decode_aligned(q) && gqa8_d128_decode_aligned(k_cache) &&
           gqa8_d128_decode_aligned(v_cache) &&
           gqa8_d128_decode_aligned(tmp_out);
  }
}

template <int D, int PARTITION_SIZE, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, typename PARTIAL_T = __half>
void launch_flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const int launch_num_partitions, const float k_scale, const float v_scale,
    const int window_size_left, const int window_size_right,
    cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true, const int* anchor_lens = nullptr,
    const int anchored_window = 0, const float* sinks = nullptr,
    unsigned int* fused_counters = nullptr) {
  // Sinks enter only in the cross-partition reducer below, so a caller that
  // reduces the partials itself cannot drop them silently.
  TORCH_CHECK(launch_reduce || sinks == nullptr,
              "attention sinks require the generic decode reducer");
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = launch_num_partitions;

  const dim3 partition_grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 reduce_grid(batch_size, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);

  const bool use_anchored = anchor_lens != nullptr && anchored_window > 0;
  // Second kernel version: the anchored decode-window mask is a separate
  // template instantiation, generated only for the fp16-KV configuration;
  // the non-anchored instantiations stay untouched.
  const auto launch_partition = [&](auto anchored_tag, auto fast_tag) {
    constexpr bool kAnchored = decltype(anchored_tag)::value;
    constexpr bool kFast = decltype(fast_tag)::value;
    flash_attention_decode_partition_kernel<
        D, PARTITION_SIZE, KV_DTYPE, SEQ_LEN_ROUTE, kAnchored, PARTIAL_T, kFast>
        <<<partition_grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            k_cache.data_ptr(), v_cache.data_ptr(),
            reinterpret_cast<PARTIAL_T*>(tmp_out.data_ptr()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks,
            max_num_partitions, num_heads_q, num_heads_kv, block_size,
            q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            softmax_scale, k_scale, v_scale, window_size_left,
            window_size_right, route_seq_len_begin, route_seq_len_end,
            route_seq_len_final, anchor_lens, anchored_window);
  };
  bool grouped = false;
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    if (gqa8_d128_decode_eligible<D, PARTITION_SIZE, SEQ_LEN_ROUTE, PARTIAL_T>(
            q, k_cache, v_cache, tmp_out, use_anchored, sinks) &&
        fused_counters != nullptr && launch_reduce &&
        gqa8_d128_decode_aligned(out)) {
      // One launch per layer: partitions and the sink-aware reduction.
      const int ctas_per_head = gqa8_d128_fused_ctas_per_head(
          batch_size, num_heads_kv, static_cast<int>(tmp_out.size(2)));
      const dim3 fused_grid(batch_size, num_heads_kv, ctas_per_head);
      flash_attention_decode_gqa8_d128_fused_kernel<kGqaFusedMaxChunk>
          <<<fused_grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(tmp_out.data_ptr()),
              max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
              fused_counters, block_table.data_ptr<int>(),
              seq_lens.data_ptr<int>(), sinks, max_num_blocks, num_heads_kv,
              block_size, q.stride(0), q.stride(1), out.stride(0),
              out.stride(1), tmp_out.stride(0), tmp_out.stride(1),
              tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
              k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
              v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
              softmax_scale, window_size_left);
      return;
    } else if (gqa8_d128_decode_eligible<D, PARTITION_SIZE, SEQ_LEN_ROUTE,
                                         PARTIAL_T>(
                   q, k_cache, v_cache, tmp_out, use_anchored, sinks)) {
      grouped = true;
      const dim3 group_grid(batch_size, num_heads_kv, max_num_partitions);
      flash_attention_decode_gqa8_d128_partition_kernel<PARTITION_SIZE>
          <<<group_grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(tmp_out.data_ptr()),
              max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
              block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
              active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks,
              max_num_partitions, num_heads_kv, block_size, q.stride(0),
              q.stride(1), tmp_out.stride(0), tmp_out.stride(1),
              tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
              k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
              v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
              softmax_scale, window_size_left, window_size_right);
    } else {
      launch_partition(std::false_type{}, std::false_type{});
    }
  }

  if (!launch_reduce) {
    return;
  }

  if (grouped) {
    flash_attention_decode_gqa8_d128_reduce_kernel<PARTITION_SIZE>
        <<<reduce_grid, block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const __half*>(tmp_out.data_ptr()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            max_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1),
            window_size_left, sinks);
    return;
  }

  flash_attention_decode_reduce_kernel<D, PARTITION_SIZE, kXQARouteAllSeqLens,
                                       PARTIAL_T>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
          reinterpret_cast<const PARTIAL_T*>(tmp_out.data_ptr()),
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
          max_num_partitions, num_heads_q, tmp_out.stride(0), tmp_out.stride(1),
          tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
          out.stride(0), out.stride(1), 0, 0, 0, 0, sinks);
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          typename PARTIAL_T = __half>
void launch_flash_attention_decode_xqa_split_reduce(
    at::Tensor& out, const at::Tensor& seq_lens, const at::Tensor& tmp_out,
    at::Tensor& max_logits, at::Tensor& exp_sums,
    const int launch_num_partitions, const int dim_tile, cudaStream_t stream,
    const int route_seq_len_begin = 0, const int route_seq_len_end = 0,
    const int route_seq_len_final = 0) {
  const int batch_size = out.size(0);
  const int num_heads_q = out.size(1);
  const dim3 stats_grid(batch_size, num_heads_q, 1);
  const dim3 stats_block(kThreadsPerBlock);
  const size_t stats_shared_mem =
      static_cast<size_t>(launch_num_partitions) * sizeof(float);
  flash_attention_decode_xqa_reduce_stats_kernel<PARTITION_SIZE, SEQ_LEN_ROUTE>
      <<<stats_grid, stats_block, stats_shared_mem, stream>>>(
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), batch_size, launch_num_partitions,
          num_heads_q, max_logits.stride(0), max_logits.stride(1),
          route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_SPLIT_REDUCE_OUTPUT(D_TILE)                                   \
  do {                                                                       \
    const dim3 output_grid(batch_size, num_heads_q,                          \
                           (256 + D_TILE - 1) / D_TILE);                     \
    const dim3 output_block(D_TILE);                                         \
    flash_attention_decode_xqa_reduce_output_kernel<                         \
        256, PARTITION_SIZE, D_TILE, SEQ_LEN_ROUTE, PARTIAL_T>               \
        <<<output_grid, output_block, 0, stream>>>(                          \
            reinterpret_cast<const PARTIAL_T*>(tmp_out.data_ptr()),          \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),        \
            seq_lens.data_ptr<int>(),                                        \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size, \
            launch_num_partitions, num_heads_q, tmp_out.stride(0),           \
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),      \
            max_logits.stride(1), out.stride(0), out.stride(1),              \
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);    \
  } while (0)

  switch (dim_tile) {
    case 16:
      LAUNCH_SPLIT_REDUCE_OUTPUT(16);
      break;
    case 32:
      LAUNCH_SPLIT_REDUCE_OUTPUT(32);
      break;
    default:
      LAUNCH_SPLIT_REDUCE_OUTPUT(8);
      break;
  }
#undef LAUNCH_SPLIT_REDUCE_OUTPUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM,
          int NUM_THREADS = kXQATC256WideThreads, int MIN_BLOCKS_PER_SM = 1,
          int BLOCK_SIZE = 0, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool ALIGNED_PADDED_SMEM = false,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          int KV_DTYPE_OVERRIDE = -1, bool E4M3_SHARED_LUT = false,
          typename PARTIAL_T = std::conditional_t<
              KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3, float,
              __half>>
void launch_flash_attention_decode_paged_xqa_tc_256_wide(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const float k_scale, const float v_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true) {
  static_assert(!E4M3_SHARED_LUT ||
                    KV_DTYPE_OVERRIDE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
                "The shared conversion LUT requires an E4M3 specialization");
  static_assert(KV_DTYPE_OVERRIDE != flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                    std::is_same_v<PARTIAL_T, float>,
                "E4M3 XQA must preserve partition outputs in FP32");
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  const size_t shared_mem =
      sizeof(SmemLayout) + (E4M3_SHARED_LUT ? 256 * sizeof(uint16_t) : 0);
#define LAUNCH_XQA_PARTITION(KV_DTYPE)                                         \
  do {                                                                         \
    auto partition_kernel =                                                    \
        (void*)flash_attention_decode_xqa_tc_partition_kernel_256_wide<        \
            PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,              \
            MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,             \
            ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,      \
            PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT, PARTIAL_T>;    \
    cudaFuncSetAttribute(partition_kernel,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         shared_mem);                                          \
    if constexpr (MIN_BLOCKS_PER_SM > 1) {                                     \
      cudaFuncSetAttribute(partition_kernel,                                   \
                           cudaFuncAttributePreferredSharedMemoryCarveout,     \
                           100);                                               \
    }                                                                          \
    flash_attention_decode_xqa_tc_partition_kernel_256_wide<                   \
        PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,                  \
        MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,                 \
        ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,          \
        PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT, PARTIAL_T>         \
        <<<partition_grid, NUM_THREADS, shared_mem, stream>>>(                 \
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),           \
            k_cache.data_ptr(), v_cache.data_ptr(),                            \
            reinterpret_cast<PARTIAL_T*>(tmp_out.data_ptr()),                  \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),          \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),             \
            active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks, \
            launch_num_partitions, num_heads_q, num_heads_kv, k_cache.size(1), \
            q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),    \
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),     \
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),           \
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),           \
            softmax_scale, k_scale, v_scale, route_seq_len_begin,              \
            route_seq_len_end, route_seq_len_final);                           \
  } while (0)

    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP16);
#undef LAUNCH_XQA_PARTITION
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (!launch_reduce) {
    return;
  }

  if constexpr (PARTITION_SIZE == -1) {
    TORCH_CHECK(!use_split_reduce,
                "Runtime wave partitions do not support split reduction");
  }
  if (use_split_reduce) {
    if constexpr (PARTITION_SIZE != -1) {
      launch_flash_attention_decode_xqa_split_reduce<PARTITION_SIZE,
                                                     SEQ_LEN_ROUTE, PARTIAL_T>(
          out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
          split_reduce_dim_tile, stream);
    }
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, PARTITION_SIZE, SEQ_LEN_ROUTE,
                                         PARTIAL_T>
        <<<reduce_grid, block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const PARTIAL_T*>(tmp_out.data_ptr()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            launch_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1),
            route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin,
            route_seq_len_begin, route_seq_len_end, route_seq_len_final,
            nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

static at::Tensor decode_paged_sinks_impl(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window, const std::optional<at::Tensor>& sinks,
    const std::optional<at::Tensor>& fused_counters) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k/v cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0,
              "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
  } else {
    TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                "fp8 k_cache must be stored as uint8");
    TORCH_CHECK(v_cache.dtype() == torch::kUInt8,
                "fp8 v_cache must be stored as uint8");
    TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                "fp8 k/v scales must be positive");
  }
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16 ||
                  (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                   tmp_out.dtype() == torch::kFloat32),
              "tmp_out must be fp16, or fp32 for E4M3 scalar decode");
  TORCH_CHECK(max_logits.dtype() == torch::kFloat32, "max_logits must be fp32");
  TORCH_CHECK(exp_sums.dtype() == torch::kFloat32, "exp_sums must be fp32");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4,
              "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(v_cache.dim() == 4,
              "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  TORCH_CHECK(max_logits.dim() == 3,
              "max_logits must have shape [B_cap, H, P]");
  TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");
  TORCH_CHECK(v_cache.stride(-1) == 1, "v_cache last dim must be contiguous");
  TORCH_CHECK(tmp_out.stride(-1) == 1, "tmp_out last dim must be contiguous");

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);

  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= tmp_out.size(0),
              "tmp_out batch size must cover q batch size");
  TORCH_CHECK(num_heads_q == tmp_out.size(1),
              "tmp_out head dimension mismatch");
  TORCH_CHECK(head_dim == tmp_out.size(3), "tmp_out head_dim mismatch");
  TORCH_CHECK(max_logits.size(0) == tmp_out.size(0) &&
                  max_logits.size(1) == tmp_out.size(1) &&
                  max_logits.size(2) == tmp_out.size(2),
              "max_logits shape mismatch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(),
              "exp_sums shape mismatch");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache head_dim mismatch");
  TORCH_CHECK(
      partition_size == 256 || partition_size == 512 || partition_size == 1024,
      "Unsupported decode partition_size: ", partition_size);
  TORCH_CHECK(
      launch_num_partitions > 0 && launch_num_partitions <= tmp_out.size(2),
      "launch_num_partitions must be in (0, tmp_out.size(2)]");
  TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
              "window sizes must be >= -1");

  TORCH_CHECK(anchored_window >= 0 && anchored_window <= INT_MAX,
              "anchored_window must fit in a non-negative int32");
  TORCH_CHECK((anchored_window > 0) == anchor_lens.has_value(),
              "anchor_lens and a positive anchored_window must be provided "
              "together");
  const bool use_anchored = anchor_lens.has_value();
  const int* anchor_lens_ptr = nullptr;
  if (use_anchored) {
    const at::Tensor& anchors = anchor_lens.value();
    TORCH_CHECK(anchors.is_cuda(), "anchor_lens must be on CUDA");
    TORCH_CHECK(anchors.device() == q.device(),
                "anchor_lens must be on the q device");
    TORCH_CHECK(anchors.dtype() == torch::kInt32, "anchor_lens must be int32");
    TORCH_CHECK(anchors.dim() == 1 && anchors.size(0) == batch_size,
                "anchor_lens must have shape [B]");
    TORCH_CHECK(anchors.is_contiguous(), "anchor_lens must be contiguous");
    TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16,
                "anchored decode window requires an fp16 KV cache");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "anchored decode window cannot be combined with "
                "sliding-window attention");
    anchor_lens_ptr = anchors.data_ptr<int>();
  }

  // Optional learned attention sinks, one fp32 natural-log value per query
  // head. They are consumed by the cross-partition reducer only.
  const float* sinks_ptr = nullptr;
  if (sinks.has_value()) {
    const at::Tensor& sink_values = sinks.value();
    TORCH_CHECK(sink_values.is_cuda(), "sinks must be on CUDA");
    TORCH_CHECK(sink_values.device() == q.device(),
                "sinks must be on the q device");
    TORCH_CHECK(sink_values.dtype() == torch::kFloat32, "sinks must be fp32");
    TORCH_CHECK(sink_values.dim() == 1 && sink_values.size(0) == num_heads_q,
                "sinks must have shape [H_q]");
    TORCH_CHECK(sink_values.is_contiguous(), "sinks must be contiguous");
    sinks_ptr = sink_values.data_ptr<float>();
  }

  // Optional arrival counters for the single-kernel grouped sink decode, one
  // per (sequence, KV head), zero between launches. Without them, or for a
  // shape the grouped kernel does not take, decode keeps two launches.
  unsigned int* fused_counters_ptr = nullptr;
  if (fused_counters.has_value() && sinks_ptr != nullptr) {
    const at::Tensor& counters = fused_counters.value();
    TORCH_CHECK(counters.is_cuda() && counters.device() == q.device(),
                "fused decode counters must be on the q device");
    TORCH_CHECK(counters.dtype() == torch::kInt32,
                "fused decode counters must be int32");
    TORCH_CHECK(counters.is_contiguous(),
                "fused decode counters must be contiguous");
    TORCH_CHECK(
        counters.numel() >= static_cast<int64_t>(batch_size) * num_heads_kv,
        "fused decode counters must cover [B, H_kv]");
    fused_counters_ptr = reinterpret_cast<unsigned int*>(counters.data_ptr());
  }

  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard device_guard(q.device());

#define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                          \
  launch_flash_attention_decode_paged<HDIM, PARTITION, KV_DTYPE_CODE>(        \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,   \
      exp_sums, active_num_partitions, softmax_scale, launch_num_partitions,  \
      k_scale, v_scale, window_size_left, window_size_right, stream, 0, 0, 0, \
      true, anchor_lens_ptr, static_cast<int>(anchored_window), sinks_ptr,    \
      fused_counters_ptr)

#define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                   \
  do {                                                                        \
    switch (kv_dtype_code) {                                                  \
      case flash_v100::KV_CACHE_DTYPE_FP16:                                   \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16);       \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype);   \
    }                                                                         \
  } while (0)

#define LAUNCH_BY_PARTITION(HDIM)                                           \
  do {                                                                      \
    switch (partition_size) {                                               \
      case 256:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 256);                                      \
        break;                                                              \
      case 512:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 512);                                      \
        break;                                                              \
      case 1024:                                                            \
        LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                     \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false,                                                  \
                    "Unsupported decode partition_size: ", partition_size); \
    }                                                                       \
  } while (0)

  switch (head_dim) {
    case 128:
      LAUNCH_BY_PARTITION(128);
      break;
    case 256:
      LAUNCH_BY_PARTITION(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim for paged decode: ", head_dim);
  }

#undef LAUNCH_BY_PARTITION
#undef LAUNCH_BY_KV_DTYPE
#undef LAUNCH_TYPED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_paged_sinks(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window, const std::optional<at::Tensor>& sinks) {
  return decode_paged_sinks_impl(
      q, k_cache, v_cache, out_, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, active_num_partitions, softmax_scale, partition_size,
      launch_num_partitions, kv_cache_dtype, k_scale, v_scale, window_size_left,
      window_size_right, anchor_lens, anchored_window, sinks, std::nullopt);
}

at::Tensor flash_attention_decode_paged_sinks_fused(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window, const std::optional<at::Tensor>& sinks,
    const at::Tensor& fused_counters) {
  return decode_paged_sinks_impl(
      q, k_cache, v_cache, out_, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, active_num_partitions, softmax_scale, partition_size,
      launch_num_partitions, kv_cache_dtype, k_scale, v_scale, window_size_left,
      window_size_right, anchor_lens, anchored_window, sinks, fused_counters);
}

// The sink-free ABI keeps its original arity for existing direct callers.
at::Tensor flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window) {
  return flash_attention_decode_paged_sinks(
      q, k_cache, v_cache, out_, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, active_num_partitions, softmax_scale, partition_size,
      launch_num_partitions, kv_cache_dtype, k_scale, v_scale, window_size_left,
      window_size_right, anchor_lens, anchored_window, std::nullopt);
}

at::Tensor flash_attention_decode_paged_xqa(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const int batch_context_max_seq_len) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k_cache and v_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "decode workspaces must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16 ||
                  kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                  kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
              "XQA decode supports fp16, fp8_e4m3, and fp8_e5m2 KV cache "
              "only");
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16 &&
                    v_cache.dtype() == torch::kFloat16,
                "fp16 XQA requires fp16 K/V tensors");
  } else {
    TORCH_CHECK(
        k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8,
        "FP8 XQA requires uint8 K/V tensors");
  }
  TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
              "XQA K/V scales must be positive");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
              "XQA decode does not support sliding-window attention");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "KV cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
              "KV cache last dim must be contiguous");
  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "K/V cache shape mismatch");
  TORCH_CHECK(k_cache.size(3) == q.size(2), "KV head_dim mismatch");
  TORCH_CHECK(q.size(2) == 256, "XQA decode supports head_dim=256 only");
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0 && num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  const int q_per_kv = num_heads_q / num_heads_kv;
  TORCH_CHECK(q_per_kv == 4 || q_per_kv == 6 || q_per_kv == 8,
              "XQA decode supports q_per_kv in {4, 6, 8}, got ", q_per_kv);
  TORCH_CHECK(partition_size == 64 || partition_size == 128 ||
                  partition_size == 256 || partition_size == 512 ||
                  partition_size == 1024 ||
                  ((partition_size == 896 || partition_size == 1664) &&
                   kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3),
              "Unsupported XQA decode partition_size: ", partition_size);
  TORCH_CHECK(launch_num_partitions > 0,
              "launch_num_partitions must be positive");
  const auto expected_partial_dtype =
      kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ? torch::kFloat32
                                                           : torch::kFloat16;
  TORCH_CHECK(tmp_out.dtype() == expected_partial_dtype,
              "XQA decode tmp_out must be fp32 for E4M3 KV and fp16 otherwise");
  TORCH_CHECK(tmp_out.size(0) >= q.size(0) && tmp_out.size(1) >= q.size(1) &&
                  tmp_out.size(2) >= launch_num_partitions &&
                  tmp_out.size(3) == q.size(2),
              "tmp_out shape does not cover XQA launch");
  TORCH_CHECK(max_logits.size(0) >= q.size(0) &&
                  max_logits.size(1) >= q.size(1) &&
                  max_logits.size(2) >= launch_num_partitions,
              "max_logits shape does not cover XQA launch");
  TORCH_CHECK(exp_sums.size(0) >= q.size(0) && exp_sums.size(1) >= q.size(1) &&
                  exp_sums.size(2) >= launch_num_partitions,
              "exp_sums shape does not cover XQA launch");

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Default only the shape with exact direct and model-level evidence.
  const bool padded_smem_enabled = xqa_padded_smem_enabled();
  const bool use_padded_smem =
      padded_smem_enabled && q_per_kv == 6 && partition_size == 256;
#define LAUNCH_XQA_WIDE(GROUP_SIZE, PARTITION)                                 \
  do {                                                                         \
    if (use_padded_smem) {                                                     \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION,           \
                                                          GROUP_SIZE, true>(   \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    } else {                                                                   \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION,           \
                                                          GROUP_SIZE, false>(  \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    }                                                                          \
  } while (0)

#define DISPATCH_PARTITION(GROUP_SIZE)                                   \
  do {                                                                   \
    switch (partition_size) {                                            \
      case 256:                                                          \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 256);                                \
        break;                                                           \
      case 512:                                                          \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 512);                                \
        break;                                                           \
      case 1024:                                                         \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 1024);                               \
        break;                                                           \
      default:                                                           \
        TORCH_CHECK(false,                                               \
                    "Unsupported XQA partition_size: ", partition_size); \
    }                                                                    \
  } while (0)

  if (q_per_kv == 4) {
    DISPATCH_PARTITION(4);
  } else if (q_per_kv == 6) {
    DISPATCH_PARTITION(6);
  } else {
    DISPATCH_PARTITION(8);
  }

#undef DISPATCH_PARTITION
#undef LAUNCH_XQA_WIDE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
