// SPDX-License-Identifier: Apache-2.0
//
// Shared by the csrc/custom_ar kernels (TP2 all-reduce + RMSNorm, custom-AR
// greedy top-1): the ABI pins of upstream's CustomAllreduce and the staged
// start barrier.
//
// Every kernel here gets upstream's CustomAllreduce handle (`fa`,
// CustomAllreduce::_ptr in Python), which lives in vLLM's _C_custom_ar .so,
// and reads only rank_, world_size_, sg_, self_sg_ and buffers_ through the
// header it was built from (VLLM_SM70_VLLM_CSRC_DIR, the same patched tree as
// vLLM's _C). The static_asserts below pin that layout for v0.33.0; a CPU
// test pins the headers' hashes (sm70/tests/compile/test_k17_custom_ar_headers.py).

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// From VLLM_SM70_VLLM_CSRC_DIR (upstream csrc/). custom_all_reduce.cuh
// includes custom_collective_common.cuh (Signal, RankData, RankSignals,
// packed_t, upcast/downcast, flag loads/stores).
#include "custom_all_reduce.cuh"

namespace vllm_sm70::custom_ar {

// ---------------------------------------------------------------------------
// ABI pins: upstream v0.33.0 (patches/BASE) + patches/series, x86_64,
// libstdc++. If one fails, upstream changed the structs these kernels read
// from another .so; re-check every field use before updating them.
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

// Upstream's barrier_at_start publishes its flag with a plain volatile store:
// the stock kernels only read inputs written by earlier kernels. The staged
// kernels here have every rank write its staging buffer in the same kernel,
// so a system fence orders those stores before the flag, and one more after
// the poll orders the peer reads after it.
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

}  // namespace vllm_sm70::custom_ar
