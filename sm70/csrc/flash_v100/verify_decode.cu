// SPDX-License-Identifier: Apache-2.0
//
// B2: Flash-V100 DFlash2 verify attention, KV read once per verify step.
//
// No fork counterpart (new for vllm-sm70). K07 verifies a DFlash2 step by
// expanding its 8 query tokens into 8 decode rows that the XQA kernel runs
// one by one, so every token streams the request's whole KV. Bea's
// microbench (one Qwen3.8 TP2 layer, 2 kv heads, D256) measured the 8 rows of
// one request at 4K context at 239-258 us, against 70 us for a single row.
//
// This kernel takes the same expanded rows (K07's decode_seq_lens and
// decode_block_table, query rows token-major: row = request * 8 + token) and
// runs one CTA per (request, kv head, 128-key partition). The CTA holds all
// 8 tokens x G query heads of that kv head (R = 8 * G = 48 or 64 rows) and
// reads each K and V tile once:
//
//   1. Q rows -> shared memory (R x 256 fp16).
//   2. For each 32-key tile: gather K through the block table, then
//      S[R, 32] = Q K^T with WMMA m16n16k16 (fp32 accumulate).
//   3. Softmax over the partition per row, with the causal limit of the row's
//      token (key position < seq_lens[row]); P = exp(S - max) in fp16.
//   4. For each 32-key tile: gather V, O[R, 256] += P V with WMMA.
//   5. Write O / sum, the row max and the row sum in the XQA partial layout
//      (tmp_out [rows, Hq, partitions, 256], max_logits / exp_sums
//      [rows, Hq, partitions]); a row with no visible key in the partition
//      writes nothing (the reducer never reads that partition).
//
// A second kernel reduces the partitions of each (row, head) exactly like
// the XQA reducer (flash_decode_paged.cu flash_attention_decode_reduce_kernel
// with a fixed partition size and no sinks).
//
// Contract (checked on the host): fp16 Q, KV and output; D = 256; 8 query
// tokens per request (rows % 8 == 0, all 8 rows of a request contiguous);
// G = H_q / H_kv in {6, 8}; KV views [num_blocks, block_size, H_kv, 256]
// with a contiguous last dim. Rows of a padding request have seq_len 0 and
// produce zeros. Shared memory: 66 KB (G = 6) or 85 KB (G = 8).

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>

namespace vllm_sm70::fv100 {
namespace {

using namespace nvcuda;

constexpr int kD = 256;
constexpr int kQLen = 8;
constexpr int kPart = 128;  // keys per partition (one CTA)
constexpr int kTileN = 32;  // keys per shared-memory K/V tile
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kLdQ = kD + 8;      // halves; +8 keeps WMMA loads bank-spread
constexpr int kLdKV = kD + 8;     // halves
constexpr int kLdS = kPart + 4;   // floats
constexpr int kLdP = kPart + 8;   // halves (aliases the S buffer)
constexpr int kVecPerRow = kD / 8;  // uint4 per 256-half row
// Finite "minus infinity": --use_fast_math is on for these sources, so
// masked scores never rely on IEEE infinities.
constexpr float kNegInf = -1.0e30f;
// Reducer smem holds one float per partition (48 KB default limit).
constexpr int kMaxPartitions = 12288;

template <int G>
struct VerifyShape {
  static constexpr int kRows = G * kQLen;
  static constexpr int kMTiles = kRows / 16;
  static constexpr int kQKTiles = kMTiles * (kTileN / 16);
  static constexpr int kOutTiles = kMTiles * (kD / 16);
  static constexpr int kOutTilesPerWarp = kOutTiles / kWarps;
  static constexpr int kRowsPerWarp = kRows / kWarps;
  static constexpr size_t kQBytes = size_t(kRows) * kLdQ * sizeof(__half);
  static constexpr size_t kKVBytes = size_t(kTileN) * kLdKV * sizeof(__half);
  static constexpr size_t kSBytes = size_t(kRows) * kLdS * sizeof(float);
  static constexpr size_t kStatBytes =
      size_t(kRows) * 2 * sizeof(float) + kQLen * sizeof(int);
  static constexpr size_t kSmemBytes = kQBytes + kKVBytes + kSBytes + kStatBytes;
  static_assert(kRows % 16 == 0, "rows must fill m16 tiles");
  static_assert(kQKTiles <= kWarps, "one QK tile per warp");
  static_assert(kOutTiles % kWarps == 0, "output tiles split evenly");
  static_assert(kRows % kWarps == 0, "softmax rows split evenly");
  static_assert(size_t(kRows) * kLdP * sizeof(__half) <= kSBytes,
                "P aliases S");
  static_assert(size_t(kWarps) * 256 * sizeof(float) <= kKVBytes,
                "epilogue staging aliases the K/V tile");
  static_assert(kQBytes % 32 == 0 && kKVBytes % 32 == 0 && kSBytes % 32 == 0,
                "WMMA needs 32-byte aligned buffers");
  static_assert(kSmemBytes <= 96 * 1024, "SM70 shared memory limit");
};

// Gather keys [key0, key0 + kTileN) of a partition (local index j, global
// position start + j) into a [kTileN][kLdKV] tile; keys >= n_keys are zero
// (P is zero there, but zero K/V keeps NaN/Inf out of the MMAs).
__device__ __forceinline__ void load_kv_tile(
    __half* tile, const __half* __restrict__ cache, const int* __restrict__ bt,
    const int start, const int key0, const int n_keys, const int block_size,
    const int kv_head, const int64_t block_stride, const int64_t token_stride,
    const int64_t head_stride, const int tid) {
#pragma unroll
  for (int i = 0; i < (kTileN * kVecPerRow) / kThreads; ++i) {
    const int idx = tid + i * kThreads;
    const int tok = idx / kVecPerRow;
    const int vec = idx % kVecPerRow;
    uint4 value = make_uint4(0, 0, 0, 0);
    const int j = key0 + tok;
    if (j < n_keys) {
      const int pos = start + j;
      const int page = __ldg(&bt[pos / block_size]);
      const int offset = pos - (pos / block_size) * block_size;
      const __half* src = cache + static_cast<int64_t>(page) * block_stride +
                          static_cast<int64_t>(offset) * token_stride +
                          static_cast<int64_t>(kv_head) * head_stride + vec * 8;
      value = __ldg(reinterpret_cast<const uint4*>(src));
    }
    reinterpret_cast<uint4*>(tile + tok * kLdKV)[vec] = value;
  }
}

template <int G>
__global__ void __launch_bounds__(kThreads, 1) verify_partition_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, __half* __restrict__ tmp_out,
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const int num_reqs, const int64_t bt_stride, const int num_partitions,
    const int block_size, const int64_t q_stride0, const int64_t q_stride1,
    const int64_t k_block_stride, const int64_t k_token_stride,
    const int64_t k_head_stride, const int64_t v_block_stride,
    const int64_t v_token_stride, const int64_t v_head_stride,
    const int64_t tmp_stride0, const int64_t tmp_stride1,
    const int64_t tmp_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const float softmax_scale) {
  using Shape = VerifyShape<G>;
  constexpr int kRows = Shape::kRows;
  const int req = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int part = blockIdx.z;
  if (req >= num_reqs || part >= num_partitions) {
    return;
  }
  const int row0 = req * kQLen;
  // The request's last token sees the most keys.
  const int seq_last = seq_lens[row0 + kQLen - 1];
  const int start = part * kPart;
  if (seq_last <= 0 || start >= seq_last) {
    return;
  }
  const int n_keys = min(kPart, seq_last - start);
  const int n_tiles = (n_keys + kTileN - 1) / kTileN;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  extern __shared__ __align__(128) char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);
  __half* sKV = reinterpret_cast<__half*>(smem_raw + Shape::kQBytes);
  float* sS = reinterpret_cast<float*>(smem_raw + Shape::kQBytes +
                                       Shape::kKVBytes);
  __half* sP = reinterpret_cast<__half*>(sS);  // after softmax
  float* sMax = reinterpret_cast<float*>(smem_raw + Shape::kQBytes +
                                         Shape::kKVBytes + Shape::kSBytes);
  float* sSum = sMax + kRows;
  int* sLimit = reinterpret_cast<int*>(sSum + kRows);

  // All 8 rows of a request share one block-table row (K07 copies it).
  const int* bt = block_table + static_cast<int64_t>(row0) * bt_stride;

  // 1. Q: row r = token * G + g <- q[row0 + token, kv_head * G + g, :].
  for (int idx = tid; idx < kRows * kVecPerRow; idx += kThreads) {
    const int r = idx / kVecPerRow;
    const int vec = idx % kVecPerRow;
    const int token = r / G;
    const int head = kv_head * G + r % G;
    const __half* src = q + static_cast<int64_t>(row0 + token) * q_stride0 +
                        static_cast<int64_t>(head) * q_stride1 + vec * 8;
    reinterpret_cast<uint4*>(sQ + r * kLdQ)[vec] =
        __ldg(reinterpret_cast<const uint4*>(src));
  }
  if (tid < kQLen) {
    sLimit[tid] = seq_lens[row0 + tid];
  }

  // 2. S = Q K^T, one 32-key tile at a time.
  for (int kt = 0; kt < n_tiles; ++kt) {
    __syncthreads();  // Q stored / previous tile consumed
    load_kv_tile(sKV, k_cache, bt, start, kt * kTileN, n_keys, block_size,
                 kv_head, k_block_stride, k_token_stride, k_head_stride, tid);
    __syncthreads();
    if (warp < Shape::kQKTiles) {
      const int mt = warp / (kTileN / 16);
      const int nt = warp % (kTileN / 16);
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
#pragma unroll
      for (int k = 0; k < kD / 16; ++k) {
        wmma::load_matrix_sync(a, sQ + mt * 16 * kLdQ + k * 16, kLdQ);
        wmma::load_matrix_sync(b, sKV + nt * 16 * kLdKV + k * 16, kLdKV);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(sS + mt * 16 * kLdS + kt * kTileN + nt * 16, c,
                              kLdS, wmma::mem_row_major);
    }
  }
  __syncthreads();

  // 3. Softmax per row over the partition (causal limit per token).
  constexpr int kColsPerLane = kPart / 32;
  float prob[Shape::kRowsPerWarp][kColsPerLane];
#pragma unroll
  for (int rr = 0; rr < Shape::kRowsPerWarp; ++rr) {
    const int r = warp * Shape::kRowsPerWarp + rr;
    const int limit = sLimit[r / G];
    float row_max = kNegInf;
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      const int j = lane + 32 * i;
      const bool valid = j < n_keys && start + j < limit;
      const float s = valid ? sS[r * kLdS + j] * softmax_scale : kNegInf;
      prob[rr][i] = s;
      row_max = fmaxf(row_max, s);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, o));
    }
    float row_sum = 0.f;
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      const float e = (row_max <= kNegInf || prob[rr][i] <= kNegInf)
                          ? 0.f
                          : __expf(prob[rr][i] - row_max);
      prob[rr][i] = e;
      row_sum += e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, o);
    }
    if (lane == 0) {
      sMax[r] = row_max;
      sSum[r] = row_sum;
    }
  }
  __syncthreads();  // every S read is done; P may overwrite it
#pragma unroll
  for (int rr = 0; rr < Shape::kRowsPerWarp; ++rr) {
    const int r = warp * Shape::kRowsPerWarp + rr;
#pragma unroll
    for (int i = 0; i < kColsPerLane; ++i) {
      sP[r * kLdP + lane + 32 * i] = __float2half_rn(prob[rr][i]);
    }
  }

  // 4. O = P V, one 32-key tile at a time.
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      acc[Shape::kOutTilesPerWarp];
#pragma unroll
  for (int t = 0; t < Shape::kOutTilesPerWarp; ++t) {
    wmma::fill_fragment(acc[t], 0.0f);
  }
  for (int kt = 0; kt < n_tiles; ++kt) {
    __syncthreads();  // P stored / previous tile consumed
    load_kv_tile(sKV, v_cache, bt, start, kt * kTileN, n_keys, block_size,
                 kv_head, v_block_stride, v_token_stride, v_head_stride, tid);
    __syncthreads();
#pragma unroll
    for (int t = 0; t < Shape::kOutTilesPerWarp; ++t) {
      const int tile = warp + t * kWarps;
      const int mt = tile / (kD / 16);
      const int nt = tile % (kD / 16);
#pragma unroll
      for (int kk = 0; kk < kTileN / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
        wmma::load_matrix_sync(
            a, sP + mt * 16 * kLdP + kt * kTileN + kk * 16, kLdP);
        wmma::load_matrix_sync(b, sKV + kk * 16 * kLdKV + nt * 16, kLdKV);
        wmma::mma_sync(acc[t], a, b, acc[t]);
      }
    }
  }
  __syncthreads();  // the K/V tile becomes epilogue staging

  // 5. Partials in the XQA layout (normalized by the row sum).
  float* stage = reinterpret_cast<float*>(sKV) + warp * 256;
#pragma unroll
  for (int t = 0; t < Shape::kOutTilesPerWarp; ++t) {
    const int tile = warp + t * kWarps;
    const int mt = tile / (kD / 16);
    const int nt = tile % (kD / 16);
    wmma::store_matrix_sync(stage, acc[t], 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
      const int r = mt * 16 + e / 16;
      const int d = nt * 16 + e % 16;
      const float sum = sSum[r];
      if (sum > 0.f) {
        const int row = row0 + r / G;
        const int head = kv_head * G + r % G;
        tmp_out[static_cast<int64_t>(row) * tmp_stride0 +
                static_cast<int64_t>(head) * tmp_stride1 +
                static_cast<int64_t>(part) * tmp_stride2 + d] =
            __float2half(stage[e] / sum);
      }
    }
    __syncwarp();
  }
  for (int r = tid; r < kRows; r += kThreads) {
    if (sSum[r] > 0.f) {
      const int row = row0 + r / G;
      const int head = kv_head * G + r % G;
      const int64_t idx = static_cast<int64_t>(row) * stats_stride0 +
                          static_cast<int64_t>(head) * stats_stride1 + part;
      max_logits[idx] = sMax[r];
      exp_sums[idx] = sSum[r];
    }
  }
}

__device__ __forceinline__ float block_reduce(float value, float* scratch,
                                              const bool is_max) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, value, o);
    value = is_max ? fmaxf(value, other) : value + other;
  }
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? scratch[lane] : (is_max ? kNegInf : 0.f);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, value, o);
      value = is_max ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) {
      scratch[kWarps] = value;
    }
  }
  __syncthreads();
  const float result = scratch[kWarps];
  __syncthreads();
  return result;
}

// One CTA per (row, head): combine the partitions of the row. Same math as
// flash_attention_decode_reduce_kernel (flash_decode_paged.cu) for a fixed
// partition size and no sinks.
__global__ void __launch_bounds__(kThreads) verify_reduce_kernel(
    const __half* __restrict__ tmp_out, const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int num_rows, const int num_heads,
    const int max_partitions, const int64_t tmp_stride0,
    const int64_t tmp_stride1, const int64_t tmp_stride2,
    const int64_t stats_stride0, const int64_t stats_stride1,
    const int64_t out_stride0, const int64_t out_stride1) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= num_rows || head >= num_heads) {
    return;
  }
  __half* out_row = out + static_cast<int64_t>(row) * out_stride0 +
                    static_cast<int64_t>(head) * out_stride1;
  const int seq_len = seq_lens[row];
  const int n = seq_len > 0
                    ? min(max_partitions, (seq_len + kPart - 1) / kPart)
                    : 0;
  if (n <= 0) {
    for (int d = threadIdx.x; d < kD; d += kThreads) {
      out_row[d] = __float2half(0.f);
    }
    return;
  }
  extern __shared__ float weights[];  // n floats
  __shared__ float scratch[kWarps + 1];
  const int64_t stats = static_cast<int64_t>(row) * stats_stride0 +
                        static_cast<int64_t>(head) * stats_stride1;
  float local_max = kNegInf;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    local_max = fmaxf(local_max, max_logits[stats + i]);
  }
  const float global_max = block_reduce(local_max, scratch, true);
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    const float w = exp_sums[stats + i] * __expf(max_logits[stats + i] - global_max);
    weights[i] = w;
    local_sum += w;
  }
  const float global_sum = block_reduce(local_sum, scratch, false);
  const float inv_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  const __half* tmp_row = tmp_out + static_cast<int64_t>(row) * tmp_stride0 +
                          static_cast<int64_t>(head) * tmp_stride1;
  for (int d = threadIdx.x; d < kD; d += kThreads) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i) {
      acc = fmaf(weights[i],
                 __half2float(tmp_row[static_cast<int64_t>(i) * tmp_stride2 + d]),
                 acc);
    }
    out_row[d] = __float2half(acc * inv_sum);
  }
}

template <int G>
void launch_verify(const at::Tensor& q, const at::Tensor& k_cache,
                   const at::Tensor& v_cache, at::Tensor& out,
                   const at::Tensor& block_table, const at::Tensor& seq_lens,
                   at::Tensor& tmp_out, at::Tensor& max_logits,
                   at::Tensor& exp_sums, const float softmax_scale,
                   const int num_partitions, cudaStream_t stream) {
  using Shape = VerifyShape<G>;
  const int num_rows = static_cast<int>(q.size(0));
  const int num_reqs = num_rows / kQLen;
  const int num_heads_q = static_cast<int>(q.size(1));
  const int num_heads_kv = static_cast<int>(k_cache.size(2));
  auto kernel = verify_partition_kernel<G>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(Shape::kSmemBytes)));
  const dim3 grid(num_reqs, num_heads_kv, num_partitions);
  kernel<<<grid, kThreads, Shape::kSmemBytes, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), num_reqs,
      block_table.stride(0), num_partitions,
      static_cast<int>(k_cache.size(1)), q.stride(0), q.stride(1),
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
      tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2),
      max_logits.stride(0), max_logits.stride(1), softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const dim3 reduce_grid(num_rows, num_heads_q);
  const size_t reduce_smem = static_cast<size_t>(num_partitions) * sizeof(float);
  verify_reduce_kernel<<<reduce_grid, kThreads, reduce_smem, stream>>>(
      reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), num_rows,
      num_heads_q, num_partitions, tmp_out.stride(0), tmp_out.stride(1),
      tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
      out.stride(0), out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The op body (fv100_ops.cpp registers it as _sm70_C::fv100_decode_verify).
void decode_verify(at::Tensor out, const at::Tensor& q,
                   const at::Tensor& k_cache, const at::Tensor& v_cache,
                   const at::Tensor& block_table, const at::Tensor& seq_lens,
                   at::Tensor tmp_out, at::Tensor max_logits,
                   at::Tensor exp_sums, double softmax_scale,
                   int64_t query_len, int64_t launch_num_partitions) {
  constexpr const char* op = "fv100_decode_verify";
  TORCH_CHECK(query_len == kQLen, op, ": query_len must be ", kQLen);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3,
              op, ": q must be a 3-D fp16 CUDA tensor");
  TORCH_CHECK(q.size(2) == kD && q.stride(2) == 1, op,
              ": q head dim must be 256 and contiguous");
  TORCH_CHECK(q.size(0) % kQLen == 0, op, ": q rows must be whole requests");
  TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == at::kHalf &&
                  out.stride(2) == 1 && out.device() == q.device(),
              op, ": out must be fp16 with q's shape");
  for (const at::Tensor* cache : {&k_cache, &v_cache}) {
    TORCH_CHECK(cache->scalar_type() == at::kHalf && cache->dim() == 4 &&
                    cache->size(3) == kD && cache->stride(3) == 1 &&
                    cache->device() == q.device(),
                op, ": KV views must be fp16 [blocks, block_size, H_kv, 256]");
  }
  const int64_t num_heads_kv = k_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0 && q.size(1) % num_heads_kv == 0, op,
              ": H_q must be a multiple of H_kv");
  const int64_t group = q.size(1) / num_heads_kv;
  TORCH_CHECK(group == 6 || group == 8, op, ": H_q / H_kv must be 6 or 8");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 &&
                  block_table.size(0) >= q.size(0) && block_table.stride(1) == 1,
              op, ": block_table must be int32 [rows, blocks]");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 &&
                  seq_lens.size(0) >= q.size(0),
              op, ": seq_lens must be int32 [rows]");
  TORCH_CHECK(launch_num_partitions > 0 &&
                  launch_num_partitions <= kMaxPartitions,
              op, ": launch_num_partitions must be in [1, ", kMaxPartitions, "]");
  // 16-byte vector loads of Q, K and V rows.
  auto aligned16 = [](const at::Tensor& t) {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  TORCH_CHECK(aligned16(q) && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0, op,
              ": q rows must be 16-byte aligned");
  for (const at::Tensor* cache : {&k_cache, &v_cache}) {
    TORCH_CHECK(aligned16(*cache) && cache->stride(0) % 8 == 0 &&
                    cache->stride(1) % 8 == 0 && cache->stride(2) % 8 == 0,
                op, ": KV rows must be 16-byte aligned");
  }
  TORCH_CHECK(tmp_out.scalar_type() == at::kHalf && tmp_out.dim() == 4 &&
                  tmp_out.size(0) >= q.size(0) && tmp_out.size(1) >= q.size(1) &&
                  tmp_out.size(2) >= launch_num_partitions &&
                  tmp_out.size(3) == kD && tmp_out.stride(3) == 1,
              op, ": tmp_out must be fp16 [rows, H_q, partitions, 256]");
  for (const at::Tensor* stats : {&max_logits, &exp_sums}) {
    TORCH_CHECK(stats->scalar_type() == at::kFloat && stats->dim() == 3 &&
                    stats->size(0) >= q.size(0) && stats->size(1) >= q.size(1) &&
                    stats->size(2) >= launch_num_partitions &&
                    stats->stride(2) == 1,
                op, ": max_logits/exp_sums must be fp32 [rows, H_q, partitions]");
  }
  if (q.size(0) == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard device_guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto scale = static_cast<float>(softmax_scale);
  const auto parts = static_cast<int>(launch_num_partitions);
  if (group == 6) {
    launch_verify<6>(q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
                     max_logits, exp_sums, scale, parts, stream);
  } else {
    launch_verify<8>(q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
                     max_logits, exp_sums, scale, parts, stream);
  }
}

}  // namespace vllm_sm70::fv100
