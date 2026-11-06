// SPDX-License-Identifier: Apache-2.0
//
// K17 (C6): TP2 custom all-reduce with fused_add_rms_norm as its epilogue,
// built into _sm70_C against upstream vLLM's own custom all-reduce headers.
//
// Ported from fork cf9492a csrc/custom_all_reduce.cu: kernel L505-651
// (sm70_tp2_cross_device_reduce_fused_add_rms_norm), host L1011-1095
// (tp2_all_reduce_fused_add_rms_norm), fork commits 45896b591 and 9f175fc6e.
// See PROVENANCE.md for the function map and what changed.
//
// Upstream's CustomAllreduce lives in vLLM's _C_custom_ar .so. This file gets
// its handle (`fa`, CustomAllreduce::_ptr in Python) and reads only rank_,
// world_size_, sg_, self_sg_ and buffers_ through the header it was built
// from (VLLM_SM70_VLLM_CSRC_DIR, the same patched tree as vLLM's _C). The
// static_asserts below pin that layout for v0.33.0; a CPU test pins the
// headers' hashes (sm70/tests/compile/test_k17_custom_ar_headers.py).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/headeronly/util/Half.h>
#include <torch/library.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// From VLLM_SM70_VLLM_CSRC_DIR (upstream csrc/). custom_all_reduce.cuh
// includes custom_collective_common.cuh (Signal, RankData, RankSignals,
// packed_t, upcast/downcast, flag loads/stores).
#include "custom_all_reduce.cuh"
#include "type_convert.cuh"

namespace vllm_sm70::custom_ar {

// ---------------------------------------------------------------------------
// ABI pins: upstream v0.33.0 (patches/BASE) + patches/series, x86_64,
// libstdc++. If one fails, upstream changed the structs this file reads
// from another .so; re-check every field use below before updating them.
// ---------------------------------------------------------------------------
static_assert(vllm::kMaxBlocks == 36);
static_assert(vllm::kMaxCustomCollectiveRanks == 16);
static_assert(std::is_same_v<vllm::FlagType, uint32_t>);
static_assert(sizeof(vllm::Signal) == 4864);
static_assert(offsetof(vllm::Signal, start) == 0);
static_assert(offsetof(vllm::Signal, end) == 2304);
static_assert(offsetof(vllm::Signal, _flag) == 4608);
static_assert(sizeof(vllm::RankData) == 128);
static_assert(sizeof(vllm::RankSignals) == 128);

#if defined(__GNUC__)
  #pragma GCC diagnostic push
  // CustomAllreduce holds std containers, so it is not standard-layout;
  // offsetof is conditionally supported there and is what we want to pin.
  #pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
static_assert(sizeof(vllm::CustomAllreduce) == 304);
static_assert(offsetof(vllm::CustomAllreduce, rank_) == 0);
static_assert(offsetof(vllm::CustomAllreduce, world_size_) == 4);
static_assert(offsetof(vllm::CustomAllreduce, sg_) == 16);
static_assert(offsetof(vllm::CustomAllreduce, buffers_) == 144);
static_assert(offsetof(vllm::CustomAllreduce, self_sg_) == 200);
static_assert(offsetof(vllm::CustomAllreduce, d_rank_data_base_) == 208);
static_assert(offsetof(vllm::CustomAllreduce, graph_unreg_buffers_) == 224);
static_assert(offsetof(vllm::CustomAllreduce, ipc_handles_) == 248);
#if defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

// One CTA covers up to four rows, 256 threads (8 halves each) per row.
constexpr int kRowThreads = 256;
constexpr int kHiddenSize = kRowThreads * 8;  // 2048
constexpr int kRowsPerCta = 4;
constexpr int kMaxRows = 16;
static_assert((kMaxRows + kRowsPerCta - 1) / kRowsPerCta <= vllm::kMaxBlocks);

// Upstream's barrier_at_start publishes its flag with a plain volatile store:
// the stock kernels only read inputs written by earlier kernels. Here every
// rank writes its staging buffer in this kernel, so (as the fork's
// barrier_at_start does, fork custom_all_reduce.cuh:617-631) a system fence
// orders those stores before the flag, and one more after the poll orders
// the peer reads after it.
__device__ __forceinline__ void membar_sys() {
  asm volatile("membar.sys;" ::: "memory");
}

template <int ngpus>
__device__ __forceinline__ void staged_barrier_at_start(
    const vllm::RankSignals& sg, vllm::Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    membar_sys();
    vllm::st_flag_volatile(peer_counter_ptr, flag);
    while (vllm::ld_flag_volatile(self_counter_ptr) != flag);
    membar_sys();
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// The one-stage all-reduce of cross_device_reduce_1stage<half, 2> with
// fused_add_rms_norm_kernel<Half, 8> (1024-thread launch) as its epilogue.
// Every rounding point of that pair is kept, so both outputs are bitwise
// equal to all-reduce + fused_add_rms_norm and identical on both ranks:
//   - peer sum upcast(rank 0) + upcast(rank 1) in FP32, one FP16 rounding
//     (packed_reduce order, the same on both ranks);
//   - optional scale: fp16(fp32(x) * scale) (an elementwise `x * scale`);
//   - residual = __hadd2(x, residual) (_f16Vec::operator+=), stored as FP16;
//   - _f16Vec::sum_squares per thread, then CUB's BlockReduce tree: a
//     shuffle-down tree per warp and the warp sums added in warp order
//     (the norm launch's threads past the row add exact zeros);
//   - rsqrtf(variance / hidden_size + epsilon), out = fp16(x * inv * w).
// With Sum2 each rank first adds its two inputs in FP16 (__hadd), as the
// elementwise add that would otherwise feed the collective does.
//
// R5 fix (fork 9f175fc6e): the peer never reads the caller's tensors. Each
// rank stores its (summed) input into its own staging buffer, the
// communicator's eager buffer IPC-registered at init (buffers_), and reads
// the peer's staging copy after the start barrier. Nothing goes into
// graph_unreg_buffers_, so operands may live anywhere (expandable segments,
// cuMem pools, static buffers) and the op is the same eagerly, in capture
// warmup and under FULL or PIECEWISE capture. The end flag is published once
// the peer data is consumed and awaited after the norm, so a rank cannot
// rewrite its staging buffer while the peer still reads it.
template <bool Sum2>
__global__ void __launch_bounds__(kRowThreads* kRowsPerCta, 1)
    tp2_cross_device_reduce_fused_add_rms_norm(
        vllm::RankData* _staging, vllm::RankSignals sg, vllm::Signal* self_sg,
        const half* __restrict__ input, const half* __restrict__ input_b,
        const half* __restrict__ residual, const half* __restrict__ weight,
        half* __restrict__ normalized_out, half* __restrict__ residual_out,
        int rank, int rows, int hidden_size, float epsilon, float scale) {
  constexpr int ngpus = 2;
  constexpr int kRowWarps = kRowThreads / 32;
  using P = typename vllm::packed_t<half>::P;
  using A = typename vllm::packed_t<half>::A;
  using V = vllm::_f16Vec<torch::headeronly::Half, 8>;
  using Converter = vllm::_typeConvert<torch::headeronly::Half>;
  static_assert(sizeof(P) == sizeof(V) && P::size == 8);

  __shared__ float warp_sums[kRowsPerCta * kRowWarps];
  __shared__ float s_variance[kRowsPerCta];

  const int tid = threadIdx.x;
  const int local_row = tid / kRowThreads;
  const int column = tid % kRowThreads;
  const int row = blockIdx.x * kRowsPerCta + local_row;
  const bool valid = row < rows;
  const int pack = row * kRowThreads + column;
  const int peer = rank ^ 1;
  // Index the rank table in global memory: a by-value copy indexed by the
  // runtime rank would live on the stack.
  P* own_staging =
      reinterpret_cast<P*>(const_cast<void*>(_staging->ptrs[rank]));
  const P* peer_staging = reinterpret_cast<const P*>(_staging->ptrs[peer]);

  P own;
  V residual_in, w;
  if (valid) {
    own = reinterpret_cast<const P*>(input)[pack];
    if constexpr (Sum2) {
      vllm::packed_assign_add(own, reinterpret_cast<const P*>(input_b)[pack]);
    }
    own_staging[pack] = own;
    residual_in = reinterpret_cast<const V*>(residual)[pack];
    w = reinterpret_cast<const V*>(weight)[column];
  }
  // Every thread's staging store precedes the fenced flag publish.
  __syncthreads();
  staged_barrier_at_start<ngpus>(sg, self_sg, rank);

  V temp;
  float variance = 0.0f;
  if (valid) {
    const P other = peer_staging[pack];
    A acc = vllm::upcast(rank == 0 ? own : other);
    vllm::packed_assign_add(acc, vllm::upcast(rank == 0 ? other : own));
    P reduced = vllm::downcast<P>(acc);
    if (scale != 1.0f) {
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        reduced.data[i] =
            __float2half_rn(__half2float(reduced.data[i]) * scale);
      }
    }
    temp = *reinterpret_cast<const V*>(&reduced);
    temp += residual_in;
    variance = temp.sum_squares();
    reinterpret_cast<V*>(residual_out)[pack] = temp;
  }

  // CUB WarpReduceShfl: lane 0 accumulates x + shfl_down(x, 1, 2, 4, 8, 16).
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    variance += __shfl_down_sync(0xffffffffu, variance, offset);
  }
  if (tid % 32 == 0) warp_sums[tid / 32] = variance;

  // Peer data is consumed (the warp sums depend on it): publish the final
  // flag now and wait for the peer's only after the norm.
  __syncthreads();
  vllm::FlagType end_flag = 0;
  if (tid < ngpus) {
    end_flag = self_sg->_flag[blockIdx.x] + 1;
    vllm::st_flag_volatile(&sg.signals[tid]->end[blockIdx.x][rank], end_flag);
  }
  if (column == 0 && valid) {
    // CUB ApplyWarpAggregates: warp 0's sum, then the others in order.
    const float* sums = warp_sums + local_row * kRowWarps;
    float total = sums[0];
#pragma unroll
    for (int warp = 1; warp < kRowWarps; ++warp) total += sums[warp];
    s_variance[local_row] = rsqrtf(total / hidden_size + epsilon);
  }
  __syncthreads();

  if (valid) {
    const float inverse_rms = s_variance[local_row];
    V out;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float x = Converter::convert(temp.data[j]);
      const float wf = Converter::convert(w.data[j]);
      out.data[j] = Converter::convert(x * inverse_rms * wf);
    }
    reinterpret_cast<V*>(normalized_out)[pack] = out;
  }

  if (tid < ngpus) {
    while (vllm::ld_flag_volatile(&self_sg->end[blockIdx.x][tid]) != end_flag);
  }
  if (tid == 0) self_sg->_flag[blockIdx.x] = end_flag;
}

/**
 * x = all_reduce(inp [+ inp_b]) [* scale]
 * residual_out = x + residual
 * normalized_out = rms_norm(residual_out) * weight
 * TP2, FP16, [1-16, 2048]. Bitwise equal to upstream's custom all-reduce
 * followed by fused_add_rms_norm. `fa` is upstream's CustomAllreduce handle;
 * `reg_buffer` its registered eager buffer (buffer_ptrs[rank]), used as the
 * staging buffer, so no operand needs IPC registration in or out of capture.
 */
void tp2_all_reduce_fused_add_rms_norm(
    int64_t fa_handle, const torch::Tensor& inp,
    const std::optional<torch::Tensor>& inp_b, const torch::Tensor& residual,
    const torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, int64_t reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon, double scale) {
  TORCH_CHECK(fa_handle != 0, "TP2 fused all-reduce RMSNorm: null handle");
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(fa_handle);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(fa->world_size_ == 2,
              "TP2 fused all-reduce RMSNorm requires two ranks");
  TORCH_CHECK(fa->rank_ == 0 || fa->rank_ == 1, "bad custom all-reduce rank");
  TORCH_CHECK(inp.dim() == 2 && inp.size(1) == kHiddenSize,
              "TP2 fused all-reduce RMSNorm requires [M, ", kHiddenSize,
              "] input");
  const int64_t rows = inp.size(0);
  TORCH_CHECK(rows >= 1 && rows <= kMaxRows,
              "TP2 fused all-reduce RMSNorm supports 1-", kMaxRows,
              " rows, got ", rows);
  std::vector<const torch::Tensor*> tensors = {&inp, &residual,
                                               &normalized_out, &residual_out};
  if (inp_b.has_value()) tensors.push_back(&inp_b.value());
  for (const torch::Tensor* t : tensors) {
    TORCH_CHECK(t->is_cuda() && t->device() == inp.device() &&
                    t->scalar_type() == at::ScalarType::Half &&
                    t->sizes() == inp.sizes() && t->is_contiguous() &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "TP2 fused all-reduce RMSNorm requires aligned, "
                "contiguous FP16 tensors of the input shape");
  }
  TORCH_CHECK(weight.is_cuda() && weight.device() == inp.device() &&
                  weight.scalar_type() == at::ScalarType::Half &&
                  weight.dim() == 1 && weight.numel() == inp.size(1) &&
                  weight.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "TP2 fused all-reduce RMSNorm requires an aligned FP16 "
              "weight of the hidden size");

  // The staging buffer was registered at initialization (register_buffer),
  // so its peer table is valid in and out of capture; it is never added to
  // the graph-buffer list.
  void* staging = reinterpret_cast<void*>(reg_buffer);
  TORCH_CHECK(inp.numel() * inp.element_size() <= reg_buffer_sz_bytes,
              "TP2 fused all-reduce RMSNorm: staging buffer too small");
  auto it = fa->buffers_.find(staging);
  TORCH_CHECK(it != fa->buffers_.end(),
              "TP2 fused all-reduce RMSNorm needs the registered staging "
              "buffer");
  vllm::RankData* staging_ptrs = it->second;

  const auto* input_ptr = reinterpret_cast<const half*>(inp.data_ptr());
  const auto* input_b_ptr =
      inp_b.has_value() ? reinterpret_cast<const half*>(inp_b->data_ptr())
                        : nullptr;
  const auto* residual_ptr = reinterpret_cast<const half*>(residual.data_ptr());
  const auto* weight_ptr = reinterpret_cast<const half*>(weight.data_ptr());
  auto* normalized_ptr = reinterpret_cast<half*>(normalized_out.data_ptr());
  auto* residual_out_ptr = reinterpret_cast<half*>(residual_out.data_ptr());
  const int rows_i = static_cast<int>(rows);
  const int rows_per_cta = std::min(rows_i, kRowsPerCta);
  const int blocks = (rows_i + kRowsPerCta - 1) / kRowsPerCta;
  const int threads = rows_per_cta * kRowThreads;
  const int hidden_size = static_cast<int>(inp.size(1));
  const float epsilon_f = static_cast<float>(epsilon);
  const float scale_f = static_cast<float>(scale);
  if (input_b_ptr != nullptr) {
    tp2_cross_device_reduce_fused_add_rms_norm<true>
        <<<blocks, threads, 0, stream>>>(
            staging_ptrs, fa->sg_, fa->self_sg_, input_ptr, input_b_ptr,
            residual_ptr, weight_ptr, normalized_ptr, residual_out_ptr,
            fa->rank_, rows_i, hidden_size, epsilon_f, scale_f);
  } else {
    tp2_cross_device_reduce_fused_add_rms_norm<false>
        <<<blocks, threads, 0, stream>>>(
            staging_ptrs, fa->sg_, fa->self_sg_, input_ptr, nullptr,
            residual_ptr, weight_ptr, normalized_ptr, residual_out_ptr,
            fa->rank_, rows_i, hidden_size, epsilon_f, scale_f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm_sm70::custom_ar

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork csrc/torch_bindings.cpp (45896b591) custom_ar.def, same schema with
  // the outputs marked as distinct aliases.
  m.def(
      "tp2_all_reduce_fused_add_rms_norm(int fa, Tensor inp, Tensor? inp_b, "
      "Tensor residual, Tensor weight, Tensor(a!) normalized_out, "
      "Tensor(b!) residual_out, int reg_buffer, int reg_buffer_sz_bytes, "
      "float epsilon, float scale) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("tp2_all_reduce_fused_add_rms_norm",
         &vllm_sm70::custom_ar::tp2_all_reduce_fused_add_rms_norm);
}
