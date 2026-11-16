// SPDX-License-Identifier: Apache-2.0
//
// F12 custom-AR exchange: the greedy top-1 (value, token id) pair reduction
// over upstream's custom all-reduce buffers instead of an NCCL all-gather.
//
// Ported from fork cf9492a: kernel csrc/custom_all_reduce.cuh:1720-1750
// (cross_device_top1_argmax<ngpus>), host :3072-3106
// (CustomAllreduce::top1_argmax) and csrc/custom_all_reduce.cu:1449-1475
// (top1_argmax), schema csrc/torch_bindings.cpp:1762-1765. See PROVENANCE.md
// for what changed (staged exchange as K17's R5 fix, no graph buffers).
//
// Every rank writes its [rows, 2] FP32 pairs into its own staging buffer
// (upstream's eager buffer, IPC-registered at init: buffers_), meets the
// peers at a fenced start barrier, reads every rank's pairs in rank order and
// writes the same int64 token per row on every rank. The end flag is
// published once the peer pairs are consumed and awaited before exit, so no
// rank rewrites its staging buffer while a peer still reads it.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

// From VLLM_SM70_VLLM_CSRC_DIR (upstream csrc/): custom_all_reduce.cuh comes
// in through custom_ar_common.cuh.
#include "custom_ar_common.cuh"

namespace vllm_sm70::custom_ar {

// One CTA, one thread per row (fork: <<<1, 128>>>, numel <= 256).
constexpr int kTop1Threads = 128;
constexpr int kTop1MaxRows = kTop1Threads;

// Tie rule (fork cross_device_top1_argmax): match a full-vocabulary argmax.
// NaNs precede finite values, and the first original vocabulary id wins among
// NaNs or equal values. Ranks own increasing vocabulary ranges, so this is
// also the lowest rank, as torch.argmax over the all-gathered pairs.
__device__ __forceinline__ bool top1_better(float value, int64_t index,
                                            float best_value,
                                            int64_t best_index) {
  const bool value_nan = isnan(value), best_nan = isnan(best_value);
  return (value_nan && (!best_nan || index < best_index)) ||
         (!value_nan && !best_nan &&
          (value > best_value || (value == best_value && index < best_index)));
}

template <int ngpus>
__global__ void __launch_bounds__(kTop1Threads, 1)
    staged_cross_device_top1_argmax(vllm::RankData* _staging,
                                    vllm::RankSignals sg, vllm::Signal* self_sg,
                                    const float* __restrict__ pair,
                                    int64_t* __restrict__ output, int rank,
                                    int rows) {
  const int row = threadIdx.x;
  // Index the rank table in global memory: a by-value copy indexed by the
  // runtime rank would live on the stack.
  float* own = reinterpret_cast<float*>(const_cast<void*>(_staging->ptrs[rank]));
  if (row < rows) {
    own[2 * row] = pair[2 * row];
    own[2 * row + 1] = pair[2 * row + 1];
  }
  // Every thread's staging store precedes the fenced flag publish.
  __syncthreads();
  staged_barrier_at_start<ngpus>(sg, self_sg, rank);

  if (row < rows) {
    float best_value = -std::numeric_limits<float>::infinity();
    int64_t best_index = std::numeric_limits<int64_t>::max();
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const float* peer = reinterpret_cast<const float*>(_staging->ptrs[i]);
      const float value = peer[2 * row];
      const int64_t index = static_cast<int64_t>(llrintf(peer[2 * row + 1]));
      if (top1_better(value, index, best_value, best_index)) {
        best_value = value;
        best_index = index;
      }
    }
    output[row] = best_index;
  }

  // All peer reads are done: publish the end flag, then wait for the
  // peers' before any rank may rewrite its staging buffer. The first
  // __syncthreads also makes thread 0's _flag update (end of the start
  // barrier) visible to every thread before it is read here.
  __syncthreads();
  const vllm::FlagType end_flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    vllm::st_flag_volatile(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                           end_flag);
    while (vllm::ld_flag_volatile(&self_sg->end[blockIdx.x][threadIdx.x]) !=
           end_flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = end_flag;
}

/**
 * output[r] = the token id of the best (value, id) pair of row r over all
 * ranks (the full-vocabulary argmax, first id on ties, NaN first).
 * `pair` is this rank's [rows, 2] FP32 (value, global token id), rows
 * 1-128; `output` int64 [rows]. `fa` is upstream's CustomAllreduce handle;
 * `reg_buffer` its registered eager buffer (buffer_ptrs[rank]), used as the
 * staging buffer, so the pair needs no IPC registration.
 */
void custom_ar_top1_argmax(int64_t fa_handle, const torch::Tensor& pair,
                           torch::Tensor& output, int64_t reg_buffer,
                           int64_t reg_buffer_sz_bytes) {
  TORCH_CHECK(fa_handle != 0, "custom-AR top-1: null handle");
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(fa_handle);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(pair));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(fa->rank_ >= 0 && fa->rank_ < fa->world_size_,
              "custom-AR top-1: bad custom all-reduce rank");
  TORCH_CHECK(pair.is_cuda() && pair.scalar_type() == at::ScalarType::Float &&
                  pair.is_contiguous(),
              "custom-AR top-1 requires a contiguous FP32 CUDA pair tensor");
  const int64_t numel = pair.numel();
  TORCH_CHECK(numel >= 2 && numel <= 2 * kTop1MaxRows && numel % 2 == 0,
              "custom-AR top-1 supports 1-", kTop1MaxRows, " rows, got ",
              numel, " pair values");
  TORCH_CHECK(output.is_cuda() && output.device() == pair.device() &&
                  output.scalar_type() == at::ScalarType::Long &&
                  output.is_contiguous() && output.numel() == numel / 2,
              "custom-AR top-1 requires a contiguous int64 output of one id "
              "per row on the pair's device");

  // Registered at initialization (register_buffer), so its peer table is
  // valid in and out of capture; it is never added to the graph-buffer list.
  void* staging = reinterpret_cast<void*>(reg_buffer);
  TORCH_CHECK(numel * static_cast<int64_t>(sizeof(float)) <=
                  reg_buffer_sz_bytes,
              "custom-AR top-1: staging buffer too small");
  auto it = fa->buffers_.find(staging);
  TORCH_CHECK(it != fa->buffers_.end(),
              "custom-AR top-1 needs the registered staging buffer");
  vllm::RankData* staging_ptrs = it->second;

  const auto* pair_ptr = reinterpret_cast<const float*>(pair.data_ptr());
  auto* out_ptr = reinterpret_cast<int64_t*>(output.data_ptr());
  const int rows = static_cast<int>(numel / 2);

#define VLLM_SM70_TOP1_CASE(ngpus)                                        \
  case ngpus:                                                             \
    staged_cross_device_top1_argmax<ngpus>                                \
        <<<1, kTop1Threads, 0, stream>>>(staging_ptrs, fa->sg_,           \
                                         fa->self_sg_, pair_ptr, out_ptr, \
                                         fa->rank_, rows);                \
    break;

  switch (fa->world_size_) {
    VLLM_SM70_TOP1_CASE(2)
    VLLM_SM70_TOP1_CASE(4)
    VLLM_SM70_TOP1_CASE(6)
    VLLM_SM70_TOP1_CASE(8)
    default:
      TORCH_CHECK(false,
                  "custom-AR top-1 supports 2, 4, 6 or 8 ranks, got ",
                  fa->world_size_);
  }
#undef VLLM_SM70_TOP1_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm_sm70::custom_ar

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork csrc/torch_bindings.cpp:1762-1765 custom_ar.def("top1_argmax"),
  // same arguments with the output marked as an alias.
  m.def(
      "custom_ar_top1_argmax(int fa, Tensor pair, Tensor(a!) output, "
      "int reg_buffer, int reg_buffer_sz_bytes) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("custom_ar_top1_argmax",
         &vllm_sm70::custom_ar::custom_ar_top1_argmax);
}
