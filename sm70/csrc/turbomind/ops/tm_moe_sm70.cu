// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboMind MoE: strided pointer tables, grouped GEMM core, FP8 and NVFP4 MoE stages,
// single-token decode. AWQ-only MoE ops are compiled with VLLM_SM70_TM_AWQ (K04).
// Moved verbatim from fork cf9492a csrc/sm70_turbomind/ops/awq_sm70_gemm.cu;
// see ../PROVENANCE.md for the fork line range of every function.
#include "tm_ops_common_sm70.cuh"

#include <torch/library.h>

namespace {

bool sm70_profile_trace_enabled() {
  return vllm::sm70::policy_exact_one(vllm::sm70::PolicyField::profile_trace);
}

const char* sm70_scalar_type_name(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return "float32";
    case at::ScalarType::Half:
      return "float16";
    case at::ScalarType::BFloat16:
      return "bfloat16";
    default:
      return "other";
  }
}

const char* sm70_capture_status_name(cudaStreamCaptureStatus status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return "none";
    case cudaStreamCaptureStatusActive:
      return "active";
    case cudaStreamCaptureStatusInvalidated:
      return "invalidated";
    default:
      return "unknown";
  }
}

unsigned sm70_capture_status_bit(cudaStreamCaptureStatus status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return 1u;
    case cudaStreamCaptureStatusActive:
      return 2u;
    case cudaStreamCaptureStatusInvalidated:
      return 4u;
    default:
      return 8u;
  }
}

void maybe_log_sm70_moe_route_once(const char* counter_name, const char* route,
                                   const torch::Tensor& input, int64_t tokens,
                                   int64_t experts_or_top_k) {
  if (!sm70_profile_trace_enabled()) {
    return;
  }
  auto& logged_mask = vllm::sm70::diagnostic_counter(counter_name);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  const unsigned bit = sm70_capture_status_bit(capture_status);
  const unsigned previous =
      logged_mask.fetch_or(bit, std::memory_order_relaxed);
  if ((previous & bit) != 0u) {
    return;
  }
  std::cerr << route << " tokens=" << tokens
            << " experts_or_top_k=" << experts_or_top_k
            << " dtype=" << sm70_scalar_type_name(input.scalar_type())
            << " capture=" << sm70_capture_status_name(capture_status)
            << std::endl;
}

}  // namespace

namespace vllm::awq_sm70 {

turbomind::gemm::DispatchPolicy select_awq_moe_dispatch_policy(
    int device, int total_tokens, int n, int k, int num_experts, int group_size,
    cudaStream_t stream) {
  const auto override_policy = awq_moe_dispatch_policy_override();
  if (override_policy.has_value()) {
    return override_policy.value();
  }
  return select_moe_dispatch_policy_impl(
      device, total_tokens, n, k, num_experts, group_size, stream,
      TuneKeyKind::kAwqMoe, awq_tune_small_shapes_enabled());
}

}  // namespace vllm::awq_sm70

bool sm70_fp8_moe_prepare_vec_enabled() {
  return vllm::sm70::policy_atoi(vllm::sm70::PolicyField::fp8_moe_prepare_vec,
                                 0) != 0;
}

namespace {

__global__ void awq_moe_fill_strided_ptrs_kernel(
    turbomind::gemm::StridedPtr* ptrs, char* base, int64_t expert_stride,
    int stride, int64_t num_experts, bool compact_metadata) {
  const int64_t expert_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (expert_id < num_experts) {
    const uintptr_t data_ptr =
        reinterpret_cast<uintptr_t>(base + expert_id * expert_stride);
    ptrs[expert_id] = {
        reinterpret_cast<void*>(
            data_ptr | (compact_metadata ? uintptr_t{1} : uintptr_t{0})),
        stride,
    };
  }
}

}  // namespace

std::vector<torch::Tensor> awq_moe_build_strided_ptrs(
    torch::Tensor tm_weights,  // [E, ...]  stacked TM weights
    torch::Tensor tm_scales,   // [E, ...]  stacked TM scales
    int64_t k_ld, int64_t q_ld, int64_t num_experts) {
  TORCH_CHECK(tm_weights.is_cuda(),
              "awq_moe_build_strided_ptrs: weights must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(),
              "awq_moe_build_strided_ptrs: scales must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "awq_moe_build_strided_ptrs: num_experts must be > 0.");
  TORCH_CHECK(tm_weights.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: weights dim0 != num_experts.");
  TORCH_CHECK(tm_scales.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: scales dim0 != num_experts.");
  if (q_ld < 0) {
    TORCH_CHECK(tm_scales.scalar_type() == torch::kUInt8 &&
                    tm_scales.dim() == 4 && tm_scales.size(3) == 3 &&
                    tm_scales.is_contiguous(),
                "awq_moe_build_strided_ptrs: compact scales must be "
                "contiguous uint8 [E,G,N,3].");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tm_weights));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t w_expert_stride =
      tm_weights.stride(0) * tm_weights.element_size();
  const int64_t s_expert_stride =
      tm_scales.stride(0) * tm_scales.element_size();
  char* w_base = static_cast<char*>(tm_weights.data_ptr());
  char* s_base = static_cast<char*>(tm_scales.data_ptr());
  const bool compact_metadata = q_ld < 0;

  // StridedPtr is 16 bytes (__align__(16): void* ptr + int stride + padding).
  const int64_t buf_bytes = num_experts * 16;
  auto opts =
      torch::TensorOptions().device(tm_weights.device()).dtype(torch::kUInt8);
  auto w_tensor = torch::empty({buf_bytes}, opts);
  auto s_tensor = torch::empty({buf_bytes}, opts);
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((num_experts + kThreads - 1) / kThreads);
  awq_moe_fill_strided_ptrs_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<turbomind::gemm::StridedPtr*>(w_tensor.data_ptr()),
      w_base, w_expert_stride, static_cast<int>(k_ld), num_experts, false);
  awq_moe_fill_strided_ptrs_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<turbomind::gemm::StridedPtr*>(s_tensor.data_ptr()),
      s_base, s_expert_stride,
      static_cast<int>(compact_metadata ? -q_ld : q_ld), num_experts,
      compact_metadata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {w_tensor, s_tensor};
}

template <typename index_t>
__global__ void awq_moe_single_token_exact_layout_prepare_kernel(
    const index_t* topk_ids, const __half* x, __half* compact_input,
    int* expert_offsets, int64_t* expert_offsets64, int* inv_permuted_idx,
    int* sorted_expert_ids, int top_k, int hidden_size, int num_experts) {
  __shared__ int sorted_ids[32];
  __shared__ int sorted_src[32];

  if (threadIdx.x == 0) {
    for (int expert = 0; expert <= num_experts; ++expert) {
      expert_offsets[expert] = 0;
    }
    for (int i = 0; i < top_k; ++i) {
      sorted_ids[i] = static_cast<int>(topk_ids[i]);
      sorted_src[i] = i;
    }

    // Keep the same stable expert ordering as moe_permute for a single token.
    for (int i = 1; i < top_k; ++i) {
      const int expert_id = sorted_ids[i];
      const int src_idx = sorted_src[i];
      int j = i - 1;
      while (j >= 0 && sorted_ids[j] > expert_id) {
        sorted_ids[j + 1] = sorted_ids[j];
        sorted_src[j + 1] = sorted_src[j];
        --j;
      }
      sorted_ids[j + 1] = expert_id;
      sorted_src[j + 1] = src_idx;
    }

    for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
      const int expert_id = sorted_ids[sorted_pos];
      const int src_idx = sorted_src[sorted_pos];
      inv_permuted_idx[src_idx] = sorted_pos;
      if (sorted_expert_ids != nullptr) {
        sorted_expert_ids[sorted_pos] = expert_id;
      }
      expert_offsets[expert_id + 1] += 1;
    }
    for (int expert = 0; expert < num_experts; ++expert) {
      expert_offsets[expert + 1] += expert_offsets[expert];
    }
    if (expert_offsets64 != nullptr) {
      for (int expert = 0; expert <= num_experts; ++expert) {
        expert_offsets64[expert] = static_cast<int64_t>(expert_offsets[expert]);
      }
    }
  }
  __syncthreads();

  if ((hidden_size & 1) == 0) {
    const int hidden_pairs = hidden_size >> 1;
    const int total_pairs = top_k * hidden_pairs;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x);
    __half2* compact_vec = reinterpret_cast<__half2*>(compact_input);
    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
      const int col_pair = idx % hidden_pairs;
      compact_vec[idx] = x_vec[col_pair];
    }
  } else {
    const int total = top_k * hidden_size;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
      const int col = idx % hidden_size;
      compact_input[idx] = x[col];
    }
  }
}

void awq_moe_single_token_exact_layout_prepare(
    torch::Tensor topk_ids, torch::Tensor x, torch::Tensor compact_input,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, int64_t num_experts) {
  TORCH_CHECK(
      topk_ids.is_cuda(),
      "awq_moe_single_token_exact_layout_prepare: topk_ids must be CUDA.");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 ||
                  topk_ids.scalar_type() == torch::kInt64,
              "awq_moe_single_token_exact_layout_prepare: topk_ids must be "
              "int32/int64.");
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_exact_layout_prepare: x must be CUDA float16.");
  TORCH_CHECK(
      compact_input.is_cuda() && compact_input.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_exact_layout_prepare: compact_input must be CUDA "
      "float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_exact_layout_prepare: index buffers must "
              "be CUDA int32/int64.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_exact_layout_prepare: x must have shape "
              "[1, hidden].");

  topk_ids = topk_ids.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_exact_layout_prepare: top_k must be in [1, 32].");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "awq_moe_single_token_exact_layout_prepare: compact_input shape "
              "mismatch.");
  TORCH_CHECK(expert_offsets.numel() == num_experts + 1 &&
                  expert_offsets64.numel() == num_experts + 1,
              "awq_moe_single_token_exact_layout_prepare: expert_offsets size "
              "mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "awq_moe_single_token_exact_layout_prepare: inv_permuted_idx "
              "size mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_exact_layout_prepare_kernel<int32_t>
        <<<1, kThreads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
            expert_offsets.data_ptr<int32_t>(),
            expert_offsets64.data_ptr<int64_t>(),
            inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
            static_cast<int>(x.size(1)), static_cast<int>(num_experts));
  } else {
    awq_moe_single_token_exact_layout_prepare_kernel<int64_t>
        <<<1, kThreads, 0, stream>>>(
            topk_ids.data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
            expert_offsets.data_ptr<int32_t>(),
            expert_offsets64.data_ptr<int64_t>(),
            inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
            static_cast<int>(x.size(1)), static_cast<int>(num_experts));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef VLLM_SM70_TM_AWQ  // AWQ/int4 MoE ops (K04)

void awq_moe_gemm_sm70_out(torch::Tensor out, torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s, int64_t num_experts,
                           int64_t k, int64_t n, int64_t group_size,
                           bool gated_silu);

void awq_moe_gemm_sm70_per_expert_dispatch_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu);

#endif  // VLLM_SM70_TM_AWQ

void awq_moe_gemm_sm70_out_impl(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu, torch::Tensor b_group_indices, bool per_expert_dispatch,
    torch::Tensor reduce_out, torch::Tensor sorted_weights,
    bool weighted_reduce, int64_t logical_total_tokens,
    torch::Tensor a_indices);

template <typename index_t>
__global__ void awq_moe_single_token_prepare_kernel(
    const index_t* topk_ids, const float* topk_weights, const __half* x,
    const uint8_t* src_w13_ptrs_w_rows, const uint8_t* src_w13_ptrs_s_rows,
    const uint8_t* src_w2_ptrs_w_rows, const uint8_t* src_w2_ptrs_s_rows,
    uint8_t* dst_w13_ptrs_w_rows, uint8_t* dst_w13_ptrs_s_rows,
    uint8_t* dst_w2_ptrs_w_rows, uint8_t* dst_w2_ptrs_s_rows,
    __half* compact_input, float* sorted_weights, int* expert_offsets,
    int64_t* expert_offsets64, int* inv_permuted_idx, int* sorted_expert_ids,
    int top_k, int hidden_size, int src_row_stride, int dst_row_stride,
    int row_bytes, bool copy_expert_ptr_rows, bool vectorized_prepare) {
  __shared__ int sorted_ids[32];
  __shared__ int sorted_src[32];

  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
    if (expert_offsets64 != nullptr) {
      expert_offsets64[0] = 0;
    }
    for (int i = 0; i < top_k; ++i) {
      sorted_ids[i] = static_cast<int>(topk_ids[i]);
      sorted_src[i] = i;
      expert_offsets[i + 1] = i + 1;
      if (expert_offsets64 != nullptr) {
        expert_offsets64[i + 1] = static_cast<int64_t>(i + 1);
      }
    }

    // Stable insertion sort to match native single-token MoE permutation.
    for (int i = 1; i < top_k; ++i) {
      const int expert_id = sorted_ids[i];
      const int src_idx = sorted_src[i];
      int j = i - 1;
      while (j >= 0 && sorted_ids[j] > expert_id) {
        sorted_ids[j + 1] = sorted_ids[j];
        sorted_src[j + 1] = sorted_src[j];
        --j;
      }
      sorted_ids[j + 1] = expert_id;
      sorted_src[j + 1] = src_idx;
    }

    for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
      const int src_idx = sorted_src[sorted_pos];
      inv_permuted_idx[src_idx] = sorted_pos;
      if (sorted_expert_ids != nullptr) {
        sorted_expert_ids[sorted_pos] = sorted_ids[sorted_pos];
      }
      if (sorted_weights != nullptr && topk_weights != nullptr) {
        sorted_weights[sorted_pos] = topk_weights[src_idx];
      }
    }
  }
  __syncthreads();

  if (copy_expert_ptr_rows) {
    if (vectorized_prepare &&
        (row_bytes % static_cast<int>(sizeof(uint4)) == 0) &&
        (src_row_stride % static_cast<int>(sizeof(uint4)) == 0) &&
        (dst_row_stride % static_cast<int>(sizeof(uint4)) == 0)) {
      const int row_vecs = row_bytes / static_cast<int>(sizeof(uint4));
      const int src_row_vec_stride =
          src_row_stride / static_cast<int>(sizeof(uint4));
      const int dst_row_vec_stride =
          dst_row_stride / static_cast<int>(sizeof(uint4));
      const int total_row_vecs = top_k * row_vecs;
      const uint4* src_w13_w_vec =
          reinterpret_cast<const uint4*>(src_w13_ptrs_w_rows);
      const uint4* src_w13_s_vec =
          reinterpret_cast<const uint4*>(src_w13_ptrs_s_rows);
      const uint4* src_w2_w_vec =
          reinterpret_cast<const uint4*>(src_w2_ptrs_w_rows);
      const uint4* src_w2_s_vec =
          reinterpret_cast<const uint4*>(src_w2_ptrs_s_rows);
      uint4* dst_w13_w_vec = reinterpret_cast<uint4*>(dst_w13_ptrs_w_rows);
      uint4* dst_w13_s_vec = reinterpret_cast<uint4*>(dst_w13_ptrs_s_rows);
      uint4* dst_w2_w_vec = reinterpret_cast<uint4*>(dst_w2_ptrs_w_rows);
      uint4* dst_w2_s_vec = reinterpret_cast<uint4*>(dst_w2_ptrs_s_rows);

      for (int idx = threadIdx.x; idx < total_row_vecs; idx += blockDim.x) {
        const int sorted_pos = idx / row_vecs;
        const int vec_idx = idx - sorted_pos * row_vecs;
        const int expert_id = sorted_ids[sorted_pos];
        const int src_offset = expert_id * src_row_vec_stride + vec_idx;
        const int dst_offset = sorted_pos * dst_row_vec_stride + vec_idx;

        dst_w13_w_vec[dst_offset] = src_w13_w_vec[src_offset];
        dst_w13_s_vec[dst_offset] = src_w13_s_vec[src_offset];
        dst_w2_w_vec[dst_offset] = src_w2_w_vec[src_offset];
        dst_w2_s_vec[dst_offset] = src_w2_s_vec[src_offset];
      }
    } else {
      const int total_row_bytes = top_k * row_bytes;
      for (int idx = threadIdx.x; idx < total_row_bytes; idx += blockDim.x) {
        const int sorted_pos = idx / row_bytes;
        const int byte_idx = idx - sorted_pos * row_bytes;
        const int expert_id = sorted_ids[sorted_pos];

        const uint8_t* src_w13_w =
            src_w13_ptrs_w_rows + expert_id * src_row_stride;
        const uint8_t* src_w13_s =
            src_w13_ptrs_s_rows + expert_id * src_row_stride;
        const uint8_t* src_w2_w =
            src_w2_ptrs_w_rows + expert_id * src_row_stride;
        const uint8_t* src_w2_s =
            src_w2_ptrs_s_rows + expert_id * src_row_stride;

        uint8_t* dst_w13_w = dst_w13_ptrs_w_rows + sorted_pos * dst_row_stride;
        uint8_t* dst_w13_s = dst_w13_ptrs_s_rows + sorted_pos * dst_row_stride;
        uint8_t* dst_w2_w = dst_w2_ptrs_w_rows + sorted_pos * dst_row_stride;
        uint8_t* dst_w2_s = dst_w2_ptrs_s_rows + sorted_pos * dst_row_stride;

        dst_w13_w[byte_idx] = src_w13_w[byte_idx];
        dst_w13_s[byte_idx] = src_w13_s[byte_idx];
        dst_w2_w[byte_idx] = src_w2_w[byte_idx];
        dst_w2_s[byte_idx] = src_w2_s[byte_idx];
      }
    }
  }

  if (compact_input == nullptr) {
    return;
  }

  if (vectorized_prepare && (hidden_size & 1) == 0) {
    const int hidden_pairs = hidden_size >> 1;
    const int total_pairs = top_k * hidden_pairs;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x);
    __half2* compact_vec = reinterpret_cast<__half2*>(compact_input);
    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
      const int col_pair = idx % hidden_pairs;
      compact_vec[idx] = x_vec[col_pair];
    }
  } else {
    const int total_elems = top_k * hidden_size;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
      const int col = idx % hidden_size;
      compact_input[idx] = x[col];
    }
  }
}

__global__ void awq_moe_single_token_weighted_reduce_kernel(
    const __half* sorted_output, const float* topk_weights,
    const int* inv_permuted_idx, __half* out, int top_k,
    int hidden_logical_size, int sorted_output_row_stride) {
  for (int col = blockIdx.x * blockDim.x + threadIdx.x;
       col < hidden_logical_size; col += blockDim.x * gridDim.x) {
    float acc = 0.f;
    for (int route_idx = 0; route_idx < top_k; ++route_idx) {
      const int sorted_pos = inv_permuted_idx[route_idx];
      const __half value =
          sorted_output[sorted_pos * sorted_output_row_stride + col];
      acc = fmaf(topk_weights[route_idx], __half2float(value), acc);
    }
    out[col] = __float2half(acc);
  }
}

template <int TOPK>
__global__ void awq_moe_single_token_weighted_reduce_half2_kernel(
    const __half* sorted_output, const float* topk_weights,
    const int* inv_permuted_idx, __half* out, int hidden_logical_size,
    int sorted_output_row_stride) {
  __shared__ int sorted_pos_sh[TOPK];
  __shared__ float weight_sh[TOPK];

  if (threadIdx.x < TOPK) {
    const int route_idx = threadIdx.x;
    sorted_pos_sh[route_idx] = inv_permuted_idx[route_idx];
    weight_sh[route_idx] = topk_weights[route_idx];
  }
  __syncthreads();

  const int hidden_pairs = hidden_logical_size >> 1;
  for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pair_idx < hidden_pairs; pair_idx += blockDim.x * gridDim.x) {
    float2 acc = {0.f, 0.f};
  #pragma unroll
    for (int route_idx = 0; route_idx < TOPK; ++route_idx) {
      const __half2 value = reinterpret_cast<const __half2*>(
          sorted_output +
          sorted_pos_sh[route_idx] * sorted_output_row_stride)[pair_idx];
      const float2 value_f = __half22float2(value);
      const float weight = weight_sh[route_idx];
      acc.x = fmaf(weight, value_f.x, acc.x);
      acc.y = fmaf(weight, value_f.y, acc.y);
    }
    reinterpret_cast<__half2*>(out)[pair_idx] = __floats2half2_rn(acc.x, acc.y);
  }
}

void awq_moe_single_token_weighted_reduce_out(torch::Tensor sorted_output,
                                              torch::Tensor topk_weights,
                                              torch::Tensor inv_permuted_idx,
                                              torch::Tensor out, int64_t top_k,
                                              int64_t hidden_logical_size) {
  TORCH_CHECK(
      sorted_output.is_cuda() && sorted_output.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_weighted_reduce_out: sorted_output must be CUDA "
      "float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "awq_moe_single_token_weighted_reduce_out: topk_weights must be CUDA "
      "float32.");
  TORCH_CHECK(inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_weighted_reduce_out: inv_permuted_idx must "
              "be CUDA int32.");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_weighted_reduce_out: out must be CUDA float16.");
  TORCH_CHECK(sorted_output.dim() == 2 && out.dim() == 2 && out.size(0) == 1,
              "awq_moe_single_token_weighted_reduce_out: invalid tensor rank.");
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_weighted_reduce_out: top_k must be in [1, 32].");
  TORCH_CHECK(sorted_output.size(0) >= top_k && topk_weights.numel() >= top_k &&
                  inv_permuted_idx.numel() >= top_k,
              "awq_moe_single_token_weighted_reduce_out: top_k size mismatch.");
  TORCH_CHECK(
      out.size(1) >= hidden_logical_size &&
          sorted_output.size(1) >= hidden_logical_size,
      "awq_moe_single_token_weighted_reduce_out: hidden size mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if ((top_k == 6 || top_k == 8) && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    if (top_k == 6) {
      awq_moe_single_token_weighted_reduce_half2_kernel<6>
          <<<blocks, kThreads, 0, stream>>>(
              reinterpret_cast<const __half*>(
                  sorted_output.data_ptr<at::Half>()),
              topk_weights.data_ptr<float>(),
              inv_permuted_idx.data_ptr<int32_t>(),
              reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
              hidden_logical_size_i, sorted_output_row_stride);
    } else {
      awq_moe_single_token_weighted_reduce_half2_kernel<8>
          <<<blocks, kThreads, 0, stream>>>(
              reinterpret_cast<const __half*>(
                  sorted_output.data_ptr<at::Half>()),
              topk_weights.data_ptr<float>(),
              inv_permuted_idx.data_ptr<int32_t>(),
              reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
              hidden_logical_size_i, sorted_output_row_stride);
    }
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(top_k), hidden_logical_size_i,
        sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
  __nv_fp8_e4m3 v;
  v.__x = static_cast<__nv_fp8_storage_t>(x);
  return static_cast<float>(v);
}

template <typename TopKIdT>
__global__ void fp8_moe_w2_direct_reduce_kernel(
    const __half* __restrict__ intermediate,
    const float* __restrict__ topk_weights,
    const TopKIdT* __restrict__ topk_ids,
    const int* __restrict__ inv_permuted_idx,
    const uint8_t* __restrict__ w2_weight, const float* __restrict__ w2_scales,
    __half* __restrict__ out, int top_k, int k, int n, int scale_n_blocks,
    int scale_k_blocks) {
  constexpr int kWarpsPerBlock = 4;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int col = blockIdx.x * kWarpsPerBlock + warp_id;
  if (col >= n) {
    return;
  }

  float total = 0.f;
  for (int route_idx = 0; route_idx < top_k; ++route_idx) {
    const int sorted_pos = __ldg(inv_permuted_idx + route_idx);
    const int expert_id = static_cast<int>(__ldg(topk_ids + route_idx));
    const float route_weight = __ldg(topk_weights + route_idx);
    const uint8_t* weight_row =
        w2_weight + (static_cast<int64_t>(expert_id) * n + col) * k;
    const float* scale_base =
        w2_scales +
        (static_cast<int64_t>(expert_id) * scale_n_blocks + col / 128) *
            scale_k_blocks;
    const __half* input_row =
        intermediate + static_cast<int64_t>(sorted_pos) * k;

    float acc = 0.f;
    for (int kk = lane_id; kk < k; kk += WARP_SIZE) {
      const float a = __half2float(__ldg(input_row + kk));
      const float scale = __ldg(scale_base + kk / 128);
      const float b = fp8_e4m3_to_float(__ldg(weight_row + kk)) * scale;
      acc = fmaf(a, b, acc);
    }
  #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane_id == 0) {
      total = fmaf(route_weight, acc, total);
    }
  }
  if (lane_id == 0) {
    out[col] = __float2half_rn(total);
  }
}

void fp8_moe_w2_direct_reduce_sm70_out(
    torch::Tensor out, torch::Tensor intermediate, torch::Tensor topk_weights,
    torch::Tensor topk_ids, torch::Tensor inv_permuted_idx,
    torch::Tensor w2_weight, torch::Tensor w2_scales, int64_t top_k, int64_t k,
    int64_t n, int64_t group_size, cudaStream_t stream) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_w2_direct_reduce_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(
      intermediate.is_cuda() && intermediate.scalar_type() == torch::kFloat16,
      "fp8_moe_w2_direct_reduce_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_w2_direct_reduce_sm70_out: weights must be CUDA float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && (topk_ids.scalar_type() == torch::kInt32 ||
                             topk_ids.scalar_type() == torch::kInt64),
      "fp8_moe_w2_direct_reduce_sm70_out: topk ids must be int32/int64.");
  TORCH_CHECK(
      inv_permuted_idx.is_cuda() &&
          inv_permuted_idx.scalar_type() == torch::kInt32,
      "fp8_moe_w2_direct_reduce_sm70_out: inverse indices must be int32.");
  TORCH_CHECK(w2_weight.is_cuda() &&
                  w2_weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 must be float8_e4m3fn.");
  TORCH_CHECK(w2_scales.is_cuda() && w2_scales.scalar_type() == torch::kFloat32,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 scales must be float32.");
  TORCH_CHECK(
      group_size == 128,
      "fp8_moe_w2_direct_reduce_sm70_out: only group_size=128 is supported.");
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_w2_direct_reduce_sm70_out: invalid top_k.");
  TORCH_CHECK(k > 0 && n > 0,
              "fp8_moe_w2_direct_reduce_sm70_out: invalid dimensions.");
  TORCH_CHECK(intermediate.dim() == 2 && intermediate.size(0) >= top_k &&
                  intermediate.size(1) == k,
              "fp8_moe_w2_direct_reduce_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1 && out.size(1) == n &&
                  out.stride(1) == 1,
              "fp8_moe_w2_direct_reduce_sm70_out: output shape mismatch.");
  TORCH_CHECK(topk_weights.numel() >= top_k && topk_ids.numel() >= top_k &&
                  inv_permuted_idx.numel() >= top_k,
              "fp8_moe_w2_direct_reduce_sm70_out: top-k buffers too small.");
  TORCH_CHECK(w2_weight.dim() == 3 && w2_weight.size(1) == n &&
                  w2_weight.size(2) == k && w2_weight.stride(2) == 1,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 shape mismatch.");
  const int64_t scale_n_blocks = (n + group_size - 1) / group_size;
  const int64_t scale_k_blocks = (k + group_size - 1) / group_size;
  TORCH_CHECK(w2_scales.dim() == 3 && w2_scales.size(0) == w2_weight.size(0) &&
                  w2_scales.size(1) == scale_n_blocks &&
                  w2_scales.size(2) == scale_k_blocks &&
                  w2_scales.stride(2) == 1,
              "fp8_moe_w2_direct_reduce_sm70_out: W2 scale shape mismatch.");

  constexpr int kThreads = 128;
  constexpr int kWarpsPerBlock = kThreads / WARP_SIZE;
  const int blocks =
      static_cast<int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (topk_ids.scalar_type() == torch::kInt32) {
    fp8_moe_w2_direct_reduce_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(intermediate.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int32_t>(),
        inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(w2_weight.data_ptr()),
        w2_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(top_k), static_cast<int>(k), static_cast<int>(n),
        static_cast<int>(scale_n_blocks), static_cast<int>(scale_k_blocks));
  } else {
    fp8_moe_w2_direct_reduce_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(intermediate.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int64_t>(),
        inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(w2_weight.data_ptr()),
        w2_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<int>(top_k), static_cast<int>(k), static_cast<int>(n),
        static_cast<int>(scale_n_blocks), static_cast<int>(scale_k_blocks));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef VLLM_SM70_TM_AWQ  // AWQ/int4 MoE ops (K04)

void awq_moe_single_token_sm70_out(
    torch::Tensor out, torch::Tensor x, torch::Tensor topk_weights,
    torch::Tensor topk_ids, torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows, torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows, torch::Tensor compact_input,
    torch::Tensor intermediate, torch::Tensor sorted_output,
    torch::Tensor sorted_weights, torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows, torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows, torch::Tensor expert_offsets,
    torch::Tensor inv_permuted_idx, int64_t w13_k, int64_t w13_n, int64_t w2_k,
    int64_t w2_n, int64_t group_size, int64_t hidden_logical_size) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "awq_moe_single_token_sm70_out: topk_weights must be CUDA float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && (topk_ids.scalar_type() == torch::kInt32 ||
                             topk_ids.scalar_type() == torch::kInt64),
      "awq_moe_single_token_sm70_out: topk_ids must be CUDA int32/int64.");
  TORCH_CHECK(
      compact_input.is_cuda() &&
          compact_input.scalar_type() == torch::kFloat16 &&
          intermediate.is_cuda() &&
          intermediate.scalar_type() == torch::kFloat16 &&
          sorted_output.is_cuda() &&
          sorted_output.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_sm70_out: scratch buffers must be CUDA float16.");
  TORCH_CHECK(
      sorted_weights.is_cuda() &&
          sorted_weights.scalar_type() == torch::kFloat32,
      "awq_moe_single_token_sm70_out: sorted_weights must be CUDA float32.");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.scalar_type() == torch::kInt32 &&
          inv_permuted_idx.is_cuda() &&
          inv_permuted_idx.scalar_type() == torch::kInt32,
      "awq_moe_single_token_sm70_out: index buffers must be CUDA int32.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
          src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda() &&
          dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
          dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
      "awq_moe_single_token_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_sm70_out: x must have shape [1, hidden].");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == 1,
      "awq_moe_single_token_sm70_out: out must have shape [1, hidden].");
  TORCH_CHECK(out.size(1) == hidden_logical_size,
              "awq_moe_single_token_sm70_out: out cols must match "
              "hidden_logical_size.");

  topk_ids = topk_ids.contiguous().view({-1});
  topk_weights = topk_weights.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});

  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(static_cast<int>(topk_weights.numel()) == top_k,
              "awq_moe_single_token_sm70_out: topk_weights size mismatch.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "awq_moe_single_token_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(intermediate.dim() == 2 && intermediate.size(0) == top_k &&
                  intermediate.size(1) == w13_n / 2,
              "awq_moe_single_token_sm70_out: intermediate shape mismatch.");
  TORCH_CHECK(sorted_output.dim() == 2 && sorted_output.size(0) == top_k &&
                  sorted_output.size(1) == w2_n,
              "awq_moe_single_token_sm70_out: sorted_output shape mismatch.");
  TORCH_CHECK(sorted_weights.numel() >= top_k,
              "awq_moe_single_token_sm70_out: sorted_weights size mismatch.");
  TORCH_CHECK(expert_offsets.numel() == top_k + 1,
              "awq_moe_single_token_sm70_out: expert_offsets size mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "awq_moe_single_token_sm70_out: inv_permuted_idx size mismatch.");

  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(
      src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
          src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
          dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
          dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
      "awq_moe_single_token_sm70_out: ptr row tensors must be 2D.");
  TORCH_CHECK(
      dst_w13_ptrs_w_rows.size(0) == top_k &&
          dst_w13_ptrs_s_rows.size(0) == top_k &&
          dst_w2_ptrs_w_rows.size(0) == top_k &&
          dst_w2_ptrs_s_rows.size(0) == top_k &&
          dst_w13_ptrs_w_rows.size(1) == row_bytes &&
          dst_w13_ptrs_s_rows.size(1) == row_bytes &&
          dst_w2_ptrs_w_rows.size(1) == row_bytes &&
          dst_w2_ptrs_s_rows.size(1) == row_bytes,
      "awq_moe_single_token_sm70_out: destination ptr row shapes mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));
  const bool vectorized_prepare = false;
  constexpr int kThreads = 256;

  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        sorted_weights.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(),
        nullptr, inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
        static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), true, vectorized_prepare);
  } else {
    awq_moe_single_token_prepare_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        sorted_weights.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(),
        nullptr, inv_permuted_idx.data_ptr<int32_t>(), nullptr, top_k,
        static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), true, vectorized_prepare);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_gemm_sm70_per_expert_dispatch_out(
      intermediate, compact_input, expert_offsets, dst_w13_ptrs_w_rows,
      dst_w13_ptrs_s_rows, top_k, w13_k, w13_n, group_size, true);
  // The weighted-reduce GEMM epilogue accumulates expert CTAs into the same
  // FP16 output with atomicAdd. CTA arrival order is not deterministic, so
  // repeated greedy requests can drift even with identical inputs. Materialize
  // the eight expert rows and reduce them below in a fixed route order with
  // FP32 FMA accumulation.
  awq_moe_gemm_sm70_per_expert_dispatch_out(
      sorted_output, intermediate, expert_offsets, dst_w2_ptrs_w_rows,
      dst_w2_ptrs_s_rows, top_k, w2_k, w2_n, group_size, false);

  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if (top_k == 8 && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i, sorted_output_row_stride);
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), top_k,
        hidden_logical_size_i, sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif  // VLLM_SM70_TM_AWQ

void awq_moe_gemm_sm70_out_impl(
    torch::Tensor out,
    torch::Tensor sorted_input,    // [total_tokens, K] float16
    torch::Tensor expert_offsets,  // [num_experts + 1] int32
    torch::Tensor
        strided_ptrs_w,  // [num_experts * 16] uint8 (StridedPtr array)
    torch::Tensor
        strided_ptrs_s,  // [num_experts * 16] uint8 (StridedPtr array)
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu, torch::Tensor b_group_indices = torch::Tensor(),
    bool per_expert_dispatch = false,
    torch::Tensor reduce_out = torch::Tensor(),
    torch::Tensor sorted_weights = torch::Tensor(),
    bool weighted_reduce = false, int64_t logical_total_tokens = -1,
    torch::Tensor a_indices = torch::Tensor()) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "awq_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32,
      "awq_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "awq_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_gemm_sm70: output must be CUDA float16.");
  if (weighted_reduce) {
    TORCH_CHECK(
        reduce_out.is_cuda() && reduce_out.scalar_type() == torch::kFloat16,
        "awq_moe_gemm_sm70: reduce_out must be CUDA float16.");
    TORCH_CHECK(sorted_weights.is_cuda() &&
                    sorted_weights.scalar_type() == torch::kFloat32,
                "awq_moe_gemm_sm70: sorted_weights must be CUDA float32.");
  }
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "awq_moe_gemm_sm70: invalid dimensions.");
  TORCH_CHECK(k % group_size == 0,
              "awq_moe_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(sorted_input.dim() == 2 && sorted_input.size(1) == k,
              "awq_moe_gemm_sm70: input shape mismatch.");
  torch::Tensor b_group_indices_flat = b_group_indices;
  const bool use_b_group_indices =
      b_group_indices.defined() && b_group_indices.numel() > 0;
  if (use_b_group_indices) {
    TORCH_CHECK(b_group_indices.is_cuda() &&
                    b_group_indices.scalar_type() == torch::kInt32,
                "awq_moe_gemm_sm70: B group indices must be CUDA int32.");
    b_group_indices_flat = b_group_indices.contiguous().view({-1});
    TORCH_CHECK(b_group_indices_flat.numel() >= num_experts,
                "awq_moe_gemm_sm70: B group indices size mismatch.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t input_tokens = sorted_input.size(0);
  const int64_t total_tokens =
      logical_total_tokens > 0 ? logical_total_tokens : input_tokens;
  const bool use_a_indices = a_indices.defined() && a_indices.numel() > 0;
  torch::Tensor a_indices_flat = a_indices;
  if (use_a_indices) {
    TORCH_CHECK(a_indices.is_cuda() &&
                    a_indices.scalar_type() == torch::kInt32 &&
                    a_indices.is_contiguous(),
                "awq_moe_gemm_sm70: A row indices must be contiguous CUDA "
                "int32.");
    a_indices_flat = a_indices.view({-1});
    TORCH_CHECK(a_indices_flat.numel() >= total_tokens,
                "awq_moe_gemm_sm70: A row indices size mismatch.");
  } else {
    TORCH_CHECK(input_tokens == total_tokens,
                "awq_moe_gemm_sm70: non-indexed input rows must match "
                "logical rows.");
  }
  TORCH_CHECK(out.size(0) == total_tokens,
              "awq_moe_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "awq_moe_gemm_sm70: output must be row-major contiguous.");
  if (weighted_reduce) {
    TORCH_CHECK(
        !gated_silu,
        "awq_moe_gemm_sm70: weighted reduce cannot combine gated_silu.");
    TORCH_CHECK(out.size(1) == n,
                "awq_moe_gemm_sm70: output cols must match n.");
    TORCH_CHECK(reduce_out.dim() == 2 && reduce_out.size(0) == 1 &&
                    reduce_out.size(1) == n && reduce_out.stride(1) == 1,
                "awq_moe_gemm_sm70: reduce_out shape mismatch.");
    TORCH_CHECK(sorted_weights.numel() >= total_tokens,
                "awq_moe_gemm_sm70: sorted_weights size mismatch.");
  } else if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "awq_moe_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "awq_moe_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "awq_moe_gemm_sm70: output cols must match n.");
  }

  if (total_tokens == 0) return;

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_moe_gemm_sm70: no compatible TurboMind converters.");

  // desc_A: input activations with offsets (kBlocked mode)
  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(sorted_input.stride(0)),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();
  desc_A.idxs = use_a_indices ? a_indices_flat.data_ptr<int>() : nullptr;

  turbomind::gemm::MatrixLayout desc_U{};

  // desc_B: weights via StridedPtr (ld=0 triggers StridedPtr resolution)
  auto desc_B =
      vllm::sm70::packed_weight_layout(*conv_w, turbomind::kUint4, n, k, 0);
  desc_B.num = static_cast<int>(num_experts);
  desc_B.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  // desc_V: scales via StridedPtr
  auto desc_V = vllm::sm70::packed_scale_layout(*conv_s, turbomind::kUint32, n,
                                                k / group_size, 0);
  desc_V.num = static_cast<int>(num_experts);
  desc_V.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  // desc_D: output with offsets (same as A)
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::select_awq_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  turbomind::gemm::MoeWeightedReduceParam moe_reduce_param{};
  if (weighted_reduce) {
    op.epilogue = static_cast<turbomind::gemm::Epilogue>(
        static_cast<int>(op.epilogue) |
        static_cast<int>(turbomind::gemm::Epilogue::kMoeWeightedReduce));
    moe_reduce_param.out = reduce_out.data_ptr();
    moe_reduce_param.sorted_weights = sorted_weights.data_ptr<float>();
    moe_reduce_param.offsets = expert_offsets.data_ptr<int>();
    op.reserved = &moe_reduce_param;
  } else {
    op.reserved = nullptr;
  }
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;
  op.dispatch_num_override = per_expert_dispatch ? 1 : 0;

  auto& workspace_holder = vllm::awq_sm70::workspace_for(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);

  const int ec = gemm.Run(
      op, 1.f, sorted_input.data_ptr(), desc_A, nullptr, desc_U,
      strided_ptrs_w.data_ptr(), desc_B, strided_ptrs_s.data_ptr(), desc_V, 0.f,
      out.data_ptr(), desc_D, out.data_ptr(), desc_D, workspace_holder, stream);

  TORCH_CHECK(ec == 0,
              "awq_moe_gemm_sm70: TurboMind batched GEMM failed (ec=", ec,
              ").");
}

#ifdef VLLM_SM70_TM_AWQ  // AWQ/int4 MoE ops (K04)

void awq_moe_gemm_sm70_out(torch::Tensor out, torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s, int64_t num_experts,
                           int64_t k, int64_t n, int64_t group_size,
                           bool gated_silu) {
  awq_moe_gemm_sm70_out_impl(out, sorted_input, expert_offsets, strided_ptrs_w,
                             strided_ptrs_s, num_experts, k, n, group_size,
                             gated_silu);
}

void awq_moe_gemm_sm70_per_expert_dispatch_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu) {
  awq_moe_gemm_sm70_out_impl(out, sorted_input, expert_offsets, strided_ptrs_w,
                             strided_ptrs_s, num_experts, k, n, group_size,
                             gated_silu, torch::Tensor(), true);
}

void awq_moe_dense_stage_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor expert_offsets,
                                  torch::Tensor dense_expert_ids,
                                  torch::Tensor ptrs_w, torch::Tensor ptrs_s,
                                  int64_t num_experts, int64_t k, int64_t n,
                                  int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "awq_moe_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous(),
              "awq_moe_dense_stage_sm70_out: expert_offsets must be contiguous "
              "CUDA int32.");
  TORCH_CHECK(dense_expert_ids.is_cuda() &&
                  dense_expert_ids.scalar_type() == torch::kInt32 &&
                  dense_expert_ids.is_contiguous(),
              "awq_moe_dense_stage_sm70_out: dense_expert_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "awq_moe_dense_stage_sm70_out: num_experts must be positive.");
  TORCH_CHECK(
      group_size == 32 || group_size == 64 || group_size == 128,
      "awq_moe_dense_stage_sm70_out: group_size must be 32, 64, or 128.");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == k,
              "awq_moe_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == input.size(0) && out.size(1) == n,
      "awq_moe_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1,
              "awq_moe_dense_stage_sm70_out: expert_offsets too small.");
  TORCH_CHECK(dense_expert_ids.numel() >= num_experts,
              "awq_moe_dense_stage_sm70_out: dense_expert_ids too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  maybe_log_sm70_moe_route_once(
      "logged_awq_dense_stage",
      "SM70 AWQ MoE CUDA-graph-safe dense-stage path enabled C++ op reached",
      input, input.size(0), num_experts);
  for (int expert = 0; expert < static_cast<int>(num_experts); ++expert) {
    torch::Tensor offsets = expert_offsets.narrow(0, expert, 2);
    torch::Tensor expert_idx = dense_expert_ids.narrow(0, expert, 1);
    awq_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, expert_idx);
  }
}

namespace {

__global__ void awq_moe_build_active_expert_segments_kernel(
    const int* __restrict__ permuted_experts_id,
    int* __restrict__ active_expert_offsets,
    int* __restrict__ active_expert_ids, int total_slots) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (total_slots <= 0) {
    active_expert_offsets[0] = 0;
    return;
  }

  int active_count = 0;
  int previous_expert = -1;
  for (int row = 0; row < total_slots; ++row) {
    const int expert = permuted_experts_id[row];
    if (row == 0 || expert != previous_expert) {
      active_expert_offsets[active_count] = row;
      active_expert_ids[active_count] = expert;
      ++active_count;
    }
    previous_expert = expert;
  }
  active_expert_offsets[active_count] = total_slots;
  for (int segment = active_count + 1; segment <= total_slots; ++segment) {
    active_expert_offsets[segment] = total_slots;
  }
  for (int segment = active_count; segment < total_slots; ++segment) {
    active_expert_ids[segment] = 0;
  }
}

}  // namespace

void awq_moe_active_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor permuted_experts_id,
    torch::Tensor active_expert_offsets, torch::Tensor active_expert_ids,
    torch::Tensor ptrs_w, torch::Tensor ptrs_s, int64_t total_slots, int64_t k,
    int64_t n, int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16,
      "awq_moe_active_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_active_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(permuted_experts_id.is_cuda() &&
                  permuted_experts_id.scalar_type() == torch::kInt32 &&
                  permuted_experts_id.is_contiguous() &&
                  active_expert_offsets.is_cuda() &&
                  active_expert_offsets.scalar_type() == torch::kInt32 &&
                  active_expert_offsets.is_contiguous() &&
                  active_expert_ids.is_cuda() &&
                  active_expert_ids.scalar_type() == torch::kInt32 &&
                  active_expert_ids.is_contiguous(),
              "awq_moe_active_dense_stage_sm70_out: index buffers must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_active_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      total_slots > 0,
      "awq_moe_active_dense_stage_sm70_out: total_slots must be positive.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_active_dense_stage_sm70_out: group_size must be 32, 64, "
              "or 128.");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= total_slots && input.size(1) == k,
      "awq_moe_active_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= total_slots && out.size(1) == n,
              "awq_moe_active_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(permuted_experts_id.numel() >= total_slots &&
                  active_expert_offsets.numel() >= total_slots + 1 &&
                  active_expert_ids.numel() >= total_slots,
              "awq_moe_active_dense_stage_sm70_out: index buffer too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  awq_moe_build_active_expert_segments_kernel<<<1, 1, 0, stream>>>(
      permuted_experts_id.data_ptr<int>(),
      active_expert_offsets.data_ptr<int>(), active_expert_ids.data_ptr<int>(),
      static_cast<int>(total_slots));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  maybe_log_sm70_moe_route_once(
      "logged_awq_active_dense_stage",
      "SM70 AWQ MoE active dense-stage path enabled C++ op reached", input,
      total_slots, total_slots);

  // Reuse the active-segment ABI for the narrow Qwen3.8 TP4 decode shapes.
  // Equal trailing offsets describe empty groups; nonempty groups may contain
  // multiple rows. Keep offsets-based scheduling (not one-row slot dispatch).

  if ((vllm::sm70::policy_atoi(
           vllm::sm70::PolicyField::awq_qwen38_moe_compact_grouped_decode, 1) !=
       0) &&
      (vllm::sm70::policy_atoi(
           vllm::sm70::PolicyField::awq_moe_batched_active_exact_w2, 0) == 0) &&
      group_size == 32 && total_slots >= 20 && total_slots <= 80 &&
      total_slots % 10 == 0 &&
      ((k == 2560 && n == 320) || (k == 160 && n == 2560))) {
    awq_moe_gemm_sm70_out_impl(out, input, active_expert_offsets, ptrs_w,
                               ptrs_s, total_slots, k, n, group_size, false,
                               active_expert_ids, true);
    return;
  }

  for (int segment = 0; segment < static_cast<int>(total_slots); ++segment) {
    torch::Tensor offsets = active_expert_offsets.narrow(0, segment, 2);
    torch::Tensor expert_idx = active_expert_ids.narrow(0, segment, 1);
    awq_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, expert_idx);
  }
}

#endif  // VLLM_SM70_TM_AWQ

namespace {

__device__ __forceinline__ int awq_moe_lower_bound_source_row(
    const int* __restrict__ permuted_idx, int begin, int end, int target) {
  while (begin < end) {
    const int middle = begin + (end - begin) / 2;
    if (permuted_idx[middle] < target) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

__global__ void awq_moe_chunk_ranges_kernel(
    const int* __restrict__ expert_offsets,
    const int* __restrict__ permuted_idx,
    int* __restrict__ chunk_expert_offsets, int* __restrict__ chunk_range_begin,
    int* __restrict__ chunk_range_end, int num_experts, int source_begin,
    int source_end) {
  const int expert = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (expert >= num_experts) {
    return;
  }

  const int expert_begin = expert_offsets[expert];
  const int expert_end = expert_offsets[expert + 1];
  const int range_begin = awq_moe_lower_bound_source_row(
      permuted_idx, expert_begin, expert_end, source_begin);
  const int range_end = awq_moe_lower_bound_source_row(
      permuted_idx, range_begin, expert_end, source_end);
  chunk_range_begin[expert] = range_begin;
  chunk_range_end[expert] = range_end;
  chunk_expert_offsets[expert + 1] = range_end - range_begin;
}

__global__ void awq_moe_prefix_chunk_offsets_kernel(
    int* __restrict__ chunk_expert_offsets, int num_experts) {
  int prefix = 0;
  chunk_expert_offsets[0] = 0;
  for (int expert = 0; expert < num_experts; ++expert) {
    prefix += chunk_expert_offsets[expert + 1];
    chunk_expert_offsets[expert + 1] = prefix;
  }
}

__global__ void awq_moe_fill_chunk_indices_kernel(
    const int* __restrict__ permuted_idx,
    const int* __restrict__ chunk_expert_offsets,
    const int* __restrict__ chunk_range_begin,
    const int* __restrict__ chunk_range_end, int* __restrict__ chunk_a_indices,
    int* __restrict__ chunk_inv_permuted_idx, int source_begin) {
  const int expert = static_cast<int>(blockIdx.x);
  const int range_begin = chunk_range_begin[expert];
  const int range_end = chunk_range_end[expert];
  const int chunk_begin = chunk_expert_offsets[expert];
  for (int global_row = range_begin + threadIdx.x; global_row < range_end;
       global_row += blockDim.x) {
    const int chunk_row = chunk_begin + global_row - range_begin;
    const int local_source_row = permuted_idx[global_row] - source_begin;
    chunk_a_indices[chunk_row] = global_row;
    chunk_inv_permuted_idx[local_source_row] = chunk_row;
  }
}

__global__ void awq_moe_chunk_weighted_reduce_kernel(
    __half* __restrict__ out, const __half* __restrict__ chunk_output,
    const float* __restrict__ topk_weights,
    const int* __restrict__ chunk_inv_permuted_idx, int output_token_begin,
    int chunk_tokens, int top_k, int hidden_logical_size, int out_stride,
    int chunk_output_stride) {
  const int local_token = static_cast<int>(blockIdx.x);
  if (local_token >= chunk_tokens) {
    return;
  }
  const int output_token = output_token_begin + local_token;
  for (int col = threadIdx.x; col < hidden_logical_size; col += blockDim.x) {
    float reduced = 0.0f;
    for (int route = 0; route < top_k; ++route) {
      const int source_row = local_token * top_k + route;
      const int chunk_row = chunk_inv_permuted_idx[source_row];
      const float value = __half2float(
          chunk_output[static_cast<int64_t>(chunk_row) * chunk_output_stride +
                       col]);
      const float weight = topk_weights[output_token * top_k + route];
      reduced = fmaf(weight, value, reduced);
    }
    out[static_cast<int64_t>(output_token) * out_stride + col] =
        __float2half_rn(reduced);
  }
}

}  // namespace

void awq_moe_chunked_w2_sm70_out(
    torch::Tensor out, torch::Tensor chunk_output, torch::Tensor input,
    torch::Tensor expert_offsets, torch::Tensor permuted_idx,
    torch::Tensor topk_weights, torch::Tensor chunk_expert_offsets,
    torch::Tensor chunk_range_begin, torch::Tensor chunk_range_end,
    torch::Tensor chunk_a_indices, torch::Tensor chunk_inv_permuted_idx,
    torch::Tensor ptrs_w, torch::Tensor ptrs_s, int64_t num_tokens,
    int64_t top_k, int64_t num_experts, int64_t k, int64_t n,
    int64_t hidden_logical_size, int64_t group_size, int64_t chunk_tokens) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16 &&
                  out.is_contiguous(),
              "awq_moe_chunked_w2_sm70_out: out must be contiguous CUDA "
              "float16.");
  TORCH_CHECK(chunk_output.is_cuda() &&
                  chunk_output.scalar_type() == torch::kFloat16 &&
                  chunk_output.is_contiguous(),
              "awq_moe_chunked_w2_sm70_out: chunk output must be "
              "contiguous CUDA float16.");
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
                  input.is_contiguous(),
              "awq_moe_chunked_w2_sm70_out: input must be contiguous "
              "CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() && permuted_idx.is_cuda() &&
                  permuted_idx.scalar_type() == torch::kInt32 &&
                  permuted_idx.is_contiguous() && topk_weights.is_cuda() &&
                  topk_weights.scalar_type() == torch::kFloat32 &&
                  topk_weights.is_contiguous(),
              "awq_moe_chunked_w2_sm70_out: route metadata dtype/device "
              "mismatch.");
  TORCH_CHECK(
      chunk_expert_offsets.is_cuda() &&
          chunk_expert_offsets.scalar_type() == torch::kInt32 &&
          chunk_expert_offsets.is_contiguous() && chunk_range_begin.is_cuda() &&
          chunk_range_begin.scalar_type() == torch::kInt32 &&
          chunk_range_begin.is_contiguous() && chunk_range_end.is_cuda() &&
          chunk_range_end.scalar_type() == torch::kInt32 &&
          chunk_range_end.is_contiguous() && chunk_a_indices.is_cuda() &&
          chunk_a_indices.scalar_type() == torch::kInt32 &&
          chunk_a_indices.is_contiguous() && chunk_inv_permuted_idx.is_cuda() &&
          chunk_inv_permuted_idx.scalar_type() == torch::kInt32 &&
          chunk_inv_permuted_idx.is_contiguous(),
      "awq_moe_chunked_w2_sm70_out: chunk scratch must be "
      "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_chunked_w2_sm70_out: weight metadata must be CUDA.");
  TORCH_CHECK(num_tokens > 0 && top_k > 0 && num_experts > 0 && k > 0 &&
                  n > 0 && group_size > 0 &&
                  (chunk_tokens == 4096 || chunk_tokens == 6144) &&
                  chunk_tokens < num_tokens && hidden_logical_size > 0 &&
                  hidden_logical_size <= n,
              "awq_moe_chunked_w2_sm70_out: invalid dimensions.");
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(num_tokens <= kIntMax / top_k && num_experts <= kIntMax &&
                  k <= kIntMax && n <= kIntMax &&
                  hidden_logical_size <= kIntMax && group_size <= kIntMax &&
                  chunk_tokens <= kIntMax,
              "awq_moe_chunked_w2_sm70_out: dimensions exceed int32.");
  const int64_t total_slots = num_tokens * top_k;
  const int64_t max_chunk_slots = std::min(num_tokens, chunk_tokens) * top_k;
  constexpr int64_t kMinIndexedChunkTokens = 2048;
  const int64_t num_chunks = (num_tokens + chunk_tokens - 1) / chunk_tokens;
  const int64_t base_chunk_tokens = num_tokens / num_chunks;
  const int64_t extra_tokens = num_tokens % num_chunks;
  const int64_t tail_tokens = num_tokens % chunk_tokens;
  const bool balance_chunks =
      tail_tokens > 0 && tail_tokens < kMinIndexedChunkTokens;
  TORCH_CHECK(base_chunk_tokens >= kMinIndexedChunkTokens,
              "awq_moe_chunked_w2_sm70_out: a balanced chunk must contain at "
              "least 2048 tokens.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == num_tokens &&
          out.size(1) == hidden_logical_size && input.dim() == 2 &&
          input.size(0) == total_slots && input.size(1) == k &&
          expert_offsets.dim() == 1 &&
          expert_offsets.numel() == num_experts + 1 &&
          permuted_idx.dim() == 1 && permuted_idx.numel() == total_slots &&
          topk_weights.dim() == 2 && topk_weights.size(0) == num_tokens &&
          topk_weights.size(1) == top_k,
      "awq_moe_chunked_w2_sm70_out: input/metadata shape "
      "mismatch.");
  TORCH_CHECK(
      chunk_output.dim() == 2 && chunk_output.size(0) >= max_chunk_slots &&
          chunk_output.size(1) == n && chunk_expert_offsets.dim() == 1 &&
          chunk_expert_offsets.numel() >= num_experts + 1 &&
          chunk_range_begin.dim() == 1 &&
          chunk_range_begin.numel() >= num_experts &&
          chunk_range_end.dim() == 1 &&
          chunk_range_end.numel() >= num_experts &&
          chunk_a_indices.dim() == 1 &&
          chunk_a_indices.numel() >= max_chunk_slots &&
          chunk_inv_permuted_idx.dim() == 1 &&
          chunk_inv_permuted_idx.numel() >= max_chunk_slots,
      "awq_moe_chunked_w2_sm70_out: scratch/output too small.");

  const int device = input.get_device();
  TORCH_CHECK(
      out.get_device() == device && chunk_output.get_device() == device &&
          expert_offsets.get_device() == device &&
          permuted_idx.get_device() == device &&
          topk_weights.get_device() == device &&
          chunk_expert_offsets.get_device() == device &&
          chunk_range_begin.get_device() == device &&
          chunk_range_end.get_device() == device &&
          chunk_a_indices.get_device() == device &&
          chunk_inv_permuted_idx.get_device() == device &&
          ptrs_w.get_device() == device && ptrs_s.get_device() == device,
      "awq_moe_chunked_w2_sm70_out: tensors must share a device.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  const int range_blocks =
      static_cast<int>((num_experts + kThreads - 1) / kThreads);

  // The production permutation is a stable expert-major sort, so source route
  // indices remain ascending inside each expert segment. Intersecting every
  // segment with a contiguous source-route interval therefore keeps the GEMM
  // expert-major while allowing the final reduction to replay route 0..top_k-1
  // in exactly the same FP32 order as moe_unpermute.
  // Preserve the existing schedule when its tail is safe. Only rebalance tiny
  // tails, keeping every partition within [2048, chunk_tokens].
  int64_t token_begin = 0;
  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    const int64_t current_tokens =
        balance_chunks ? base_chunk_tokens + (chunk < extra_tokens)
                       : std::min(chunk_tokens, num_tokens - token_begin);
    const int64_t current_slots = current_tokens * top_k;
    const int64_t source_begin = token_begin * top_k;
    const int64_t source_end = source_begin + current_slots;

    awq_moe_chunk_ranges_kernel<<<range_blocks, kThreads, 0, stream>>>(
        expert_offsets.data_ptr<int>(), permuted_idx.data_ptr<int>(),
        chunk_expert_offsets.data_ptr<int>(), chunk_range_begin.data_ptr<int>(),
        chunk_range_end.data_ptr<int>(), static_cast<int>(num_experts),
        static_cast<int>(source_begin), static_cast<int>(source_end));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    awq_moe_prefix_chunk_offsets_kernel<<<1, 1, 0, stream>>>(
        chunk_expert_offsets.data_ptr<int>(), static_cast<int>(num_experts));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    awq_moe_fill_chunk_indices_kernel<<<num_experts, kThreads, 0, stream>>>(
        permuted_idx.data_ptr<int>(), chunk_expert_offsets.data_ptr<int>(),
        chunk_range_begin.data_ptr<int>(), chunk_range_end.data_ptr<int>(),
        chunk_a_indices.data_ptr<int>(), chunk_inv_permuted_idx.data_ptr<int>(),
        static_cast<int>(source_begin));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    awq_moe_gemm_sm70_out_impl(
        chunk_output.narrow(0, 0, current_slots), input, chunk_expert_offsets,
        ptrs_w, ptrs_s, num_experts, k, n, group_size, false, torch::Tensor(),
        false, torch::Tensor(), torch::Tensor(), false, current_slots,
        chunk_a_indices);

    awq_moe_chunk_weighted_reduce_kernel<<<current_tokens, kThreads, 0,
                                           stream>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(chunk_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), chunk_inv_permuted_idx.data_ptr<int>(),
        static_cast<int>(token_begin), static_cast<int>(current_tokens),
        static_cast<int>(top_k), static_cast<int>(hidden_logical_size),
        static_cast<int>(out.stride(0)),
        static_cast<int>(chunk_output.stride(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    token_begin += current_tokens;
  }
}

#ifdef VLLM_SM70_TM_AWQ  // AWQ/int4 MoE ops (K04)

void awq_moe_single_token_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "awq_moe_single_token_dense_stage_sm70_out: index buffers must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(
      ptrs_w.is_cuda() && ptrs_s.is_cuda(),
      "awq_moe_single_token_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_dense_stage_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
      "awq_moe_single_token_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
              "awq_moe_single_token_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "awq_moe_single_token_dense_stage_sm70_out: index buffer too small.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_dense_stage_sm70_out: group_size must be "
              "32, 64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  for (int route = 0; route < static_cast<int>(top_k); ++route) {
    torch::Tensor offsets = expert_offsets.narrow(0, route, 2);
    torch::Tensor expert_idx = sorted_expert_ids.narrow(0, route, 1);
    awq_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, expert_idx);
  }
}

void awq_moe_single_token_indexed_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: input must "
              "be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: out must be "
              "CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "awq_moe_single_token_indexed_dense_stage_sm70_out: index "
              "buffers must be contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "awq_moe_single_token_indexed_dense_stage_sm70_out: ptr rows "
              "must be CUDA.");
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: top_k must "
              "be in [1, 32].");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: input shape "
              "mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
      "awq_moe_single_token_indexed_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "awq_moe_single_token_indexed_dense_stage_sm70_out: index buffer too "
      "small.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_indexed_dense_stage_sm70_out: group_size "
              "must be 32, 64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  awq_moe_gemm_sm70_out_impl(out, input, expert_offsets, ptrs_w, ptrs_s, top_k,
                             k, n, group_size, false, sorted_expert_ids, true);
}

void awq_moe_single_token_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16,
      "awq_moe_single_token_dense_w13_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_dense_w13_sm70_out: scratch buffers must "
              "be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "awq_moe_single_token_dense_w13_sm70_out: topk_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "awq_moe_single_token_dense_w13_sm70_out: index buffers must be "
              "CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "awq_moe_single_token_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(
      w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
      "awq_moe_single_token_dense_w13_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_dense_w13_sm70_out: x must have shape [1, "
              "hidden].");
  TORCH_CHECK(x.size(1) == hidden_logical_size,
              "awq_moe_single_token_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "awq_moe_single_token_dense_w13_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 &&
          expert_offsets64.numel() >= top_k + 1 &&
          inv_permuted_idx.numel() >= top_k &&
          sorted_expert_ids.numel() >= top_k,
      "awq_moe_single_token_dense_w13_sm70_out: index buffer too small.");
  TORCH_CHECK(
      compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
          compact_input.size(1) == hidden_logical_size,
      "awq_moe_single_token_dense_w13_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(
      gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
          gate_up.size(1) == w13_n,
      "awq_moe_single_token_dense_w13_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_dense_w13_sm70_out: group_size must be 32, "
              "64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  maybe_log_sm70_moe_route_once("logged_awq_single_token",
                                "SM70 AWQ MoE single-token active-expert dense "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_single_token_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void awq_moe_single_token_indexed_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "awq_moe_single_token_indexed_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "awq_moe_single_token_indexed_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
              "awq_moe_single_token_indexed_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "awq_moe_single_token_indexed_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_indexed_dense_w13_sm70_out: group_size "
              "must be 32, 64, or 128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  maybe_log_sm70_moe_route_once(
      "logged_awq_single_token_indexed",
      "SM70 AWQ MoE single-token indexed dense path enabled C++ op reached", x,
      x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_single_token_indexed_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void awq_moe_single_token_compact_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor compact_w13_ptrs_w, torch::Tensor compact_w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_compact_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_compact_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "awq_moe_single_token_compact_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "awq_moe_single_token_compact_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "awq_moe_single_token_compact_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda() &&
                  compact_w13_ptrs_w.is_cuda() && compact_w13_ptrs_s.is_cuda(),
              "awq_moe_single_token_compact_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(compact_w13_ptrs_w.scalar_type() == torch::kUInt8 &&
                  compact_w13_ptrs_s.scalar_type() == torch::kUInt8,
              "awq_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "rows must be uint8.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_compact_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "awq_moe_single_token_compact_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_compact_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "awq_moe_single_token_compact_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "awq_moe_single_token_compact_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "awq_moe_single_token_compact_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_moe_single_token_compact_dense_w13_sm70_out: group_size "
              "must be 32, 64, or 128.");

  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  TORCH_CHECK(compact_w13_ptrs_w.numel() >= top_k * kPtrRowBytes &&
                  compact_w13_ptrs_s.numel() >= top_k * kPtrRowBytes,
              "awq_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "buffer too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  maybe_log_sm70_moe_route_once("logged_awq_single_token_compact",
                                "SM70 AWQ MoE single-token compact grouped W13 "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, true, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_gemm_sm70_per_expert_dispatch_out(
      gate_up, compact_input, expert_offsets, compact_w13_ptrs_w,
      compact_w13_ptrs_s, top_k, w13_k, w13_n, group_size, false);
}

#endif  // VLLM_SM70_TM_AWQ

void fp8_moe_gemm_sm70_out_impl(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu, torch::Tensor reduce_out, torch::Tensor sorted_weights,
    bool weighted_reduce, int64_t logical_total_tokens = -1,
    int64_t a_ld_override = -1, torch::Tensor a_indices = torch::Tensor(),
    torch::Tensor b_group_indices = torch::Tensor(),
    bool per_expert_dispatch = false, int64_t dispatch_num_experts = -1,
    torch::Tensor active_group_indices = torch::Tensor(),
    int64_t active_group_count = -1) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "fp8_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32,
      "fp8_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "fp8_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_gemm_sm70: output must be CUDA float16.");
  if (weighted_reduce) {
    TORCH_CHECK(
        reduce_out.is_cuda() && reduce_out.scalar_type() == torch::kFloat16,
        "fp8_moe_gemm_sm70: reduce_out must be CUDA float16.");
    TORCH_CHECK(sorted_weights.is_cuda() &&
                    sorted_weights.scalar_type() == torch::kFloat32,
                "fp8_moe_gemm_sm70: sorted_weights must be CUDA float32.");
  }
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "fp8_moe_gemm_sm70: invalid dimensions.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_gemm_sm70: only group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(sorted_input.dim() == 2 && sorted_input.size(1) == k,
              "fp8_moe_gemm_sm70: input shape mismatch.");
  TORCH_CHECK(a_ld_override >= -1,
              "fp8_moe_gemm_sm70: invalid A leading dimension override.");
  const int64_t input_tokens = sorted_input.size(0);
  const int64_t total_tokens =
      logical_total_tokens > 0 ? logical_total_tokens : input_tokens;
  const bool broadcast_a = input_tokens != total_tokens;
  TORCH_CHECK(a_ld_override != 0,
              "fp8_moe_gemm_sm70: A ld=0 broadcast is unsupported.");
  TORCH_CHECK(total_tokens >= 0,
              "fp8_moe_gemm_sm70: invalid logical token count.");
  if (broadcast_a) {
    TORCH_CHECK(input_tokens == 1,
                "fp8_moe_gemm_sm70: broadcast input must have one row.");
  } else {
    TORCH_CHECK(input_tokens == total_tokens,
                "fp8_moe_gemm_sm70: input rows must match logical rows.");
  }
  torch::Tensor a_indices_flat = a_indices;
  const bool use_a_indices = a_indices.defined() && a_indices.numel() > 0;
  if (use_a_indices) {
    TORCH_CHECK(a_indices.is_cuda() && a_indices.scalar_type() == torch::kInt32,
                "fp8_moe_gemm_sm70: A row indices must be CUDA int32.");
    a_indices_flat = a_indices.contiguous().view({-1});
    TORCH_CHECK(a_indices_flat.numel() >= total_tokens,
                "fp8_moe_gemm_sm70: A row indices size mismatch.");
  }
  torch::Tensor b_group_indices_flat = b_group_indices;
  const bool use_b_group_indices =
      b_group_indices.defined() && b_group_indices.numel() > 0;
  if (use_b_group_indices) {
    TORCH_CHECK(b_group_indices.is_cuda() &&
                    b_group_indices.scalar_type() == torch::kInt32,
                "fp8_moe_gemm_sm70: B group indices must be CUDA int32.");
    b_group_indices_flat = b_group_indices.contiguous().view({-1});
    TORCH_CHECK(b_group_indices_flat.numel() >= num_experts,
                "fp8_moe_gemm_sm70: B group indices size mismatch.");
  }
  torch::Tensor active_group_indices_flat = active_group_indices;
  const bool use_active_group_indices = active_group_indices.defined() &&
                                        active_group_indices.numel() > 0 &&
                                        active_group_count > 0;
  if (use_active_group_indices) {
    TORCH_CHECK(active_group_indices.is_cuda() &&
                    active_group_indices.scalar_type() == torch::kInt32,
                "fp8_moe_gemm_sm70: active group indices must be CUDA int32.");
    active_group_indices_flat = active_group_indices.contiguous().view({-1});
    TORCH_CHECK(active_group_indices_flat.numel() >= active_group_count,
                "fp8_moe_gemm_sm70: active group indices size mismatch.");
    TORCH_CHECK(
        active_group_count <= num_experts,
        "fp8_moe_gemm_sm70: active group count exceeds source experts.");
  }
  const int64_t selected_dispatch_num_experts =
      dispatch_num_experts > 0 ? dispatch_num_experts : num_experts;
  TORCH_CHECK(out.size(0) == total_tokens,
              "fp8_moe_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "fp8_moe_gemm_sm70: output must be row-major contiguous.");
  if (weighted_reduce) {
    TORCH_CHECK(
        !gated_silu,
        "fp8_moe_gemm_sm70: weighted reduce cannot combine gated_silu.");
    TORCH_CHECK(out.size(1) == n,
                "fp8_moe_gemm_sm70: output cols must match n.");
    TORCH_CHECK(reduce_out.dim() == 2 && reduce_out.size(0) == 1 &&
                    reduce_out.size(1) == n && reduce_out.stride(1) == 1,
                "fp8_moe_gemm_sm70: reduce_out shape mismatch.");
    TORCH_CHECK(sorted_weights.numel() == total_tokens,
                "fp8_moe_gemm_sm70: sorted_weights size mismatch.");
  } else if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "fp8_moe_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "fp8_moe_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "fp8_moe_gemm_sm70: output cols must match n.");
  }
  if (total_tokens == 0) return;

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_moe_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(a_ld_override >= 0 ? a_ld_override : k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = (broadcast_a && !use_a_indices)
                       ? nullptr
                       : expert_offsets.data_ptr<int>();
  desc_A.idxs = use_a_indices ? a_indices_flat.data_ptr<int>() : nullptr;
  turbomind::gemm::MatrixLayout desc_U{};

  auto desc_B = vllm::sm70::packed_weight_layout(
      *conv_w, turbomind::kFloat8_e4m3, n, k, 0);
  desc_B.num = static_cast<int>(num_experts);
  desc_B.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  auto desc_V = vllm::sm70::packed_scale_layout(
      *conv_s, turbomind::kUint16, n, (k + group_size - 1) / group_size, 0);
  desc_V.num = static_cast<int>(num_experts);
  desc_V.group_idxs =
      use_b_group_indices ? b_group_indices_flat.data_ptr<int>() : nullptr;

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();
  desc_D.group_idxs = use_active_group_indices
                          ? active_group_indices_flat.data_ptr<int>()
                          : nullptr;

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::select_fp8_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(selected_dispatch_num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  turbomind::gemm::MoeWeightedReduceParam moe_reduce_param{};
  if (weighted_reduce) {
    op.epilogue = static_cast<turbomind::gemm::Epilogue>(
        static_cast<int>(op.epilogue) |
        static_cast<int>(turbomind::gemm::Epilogue::kMoeWeightedReduce));
    moe_reduce_param.out = reduce_out.data_ptr();
    moe_reduce_param.sorted_weights = sorted_weights.data_ptr<float>();
    moe_reduce_param.offsets = expert_offsets.data_ptr<int>();
    op.reserved = &moe_reduce_param;
  } else {
    op.reserved = nullptr;
  }
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;
  op.dispatch_num_override = per_expert_dispatch ? 1 : 0;
  op.active_group_count =
      use_active_group_indices ? static_cast<int>(active_group_count) : 0;

  auto& workspace_holder = vllm::awq_sm70::workspace_for(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int ec = gemm.Run(
      op, 1.f, sorted_input.data_ptr(), desc_A, nullptr, desc_U,
      strided_ptrs_w.data_ptr(), desc_B, strided_ptrs_s.data_ptr(), desc_V, 0.f,
      out.data_ptr(), desc_D, out.data_ptr(), desc_D, workspace_holder, stream);
  TORCH_CHECK(ec == 0,
              "fp8_moe_gemm_sm70: TurboMind batched GEMM failed (ec=", ec,
              ").");
}

void fp8_moe_gemm_sm70_out(torch::Tensor out, torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s, int64_t num_experts,
                           int64_t k, int64_t n, int64_t group_size,
                           bool gated_silu) {
  fp8_moe_gemm_sm70_out_impl(out, sorted_input, expert_offsets, strided_ptrs_w,
                             strided_ptrs_s, num_experts, k, n, group_size,
                             gated_silu, torch::Tensor(), torch::Tensor(),
                             false, -1, -1);
}

void fp8_moe_gemm_sm70_per_expert_dispatch_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    bool gated_silu) {
  fp8_moe_gemm_sm70_out_impl(
      out, sorted_input, expert_offsets, strided_ptrs_w, strided_ptrs_s,
      num_experts, k, n, group_size, gated_silu, torch::Tensor(),
      torch::Tensor(), false, -1, -1, torch::Tensor(), torch::Tensor(), true);
}

void fp8_moe_dense_stage_sm70_out(torch::Tensor out, torch::Tensor input,
                                  torch::Tensor expert_offsets,
                                  torch::Tensor dense_expert_ids,
                                  torch::Tensor ptrs_w, torch::Tensor ptrs_s,
                                  int64_t num_experts, int64_t k, int64_t n,
                                  int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "fp8_moe_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous(),
              "fp8_moe_dense_stage_sm70_out: expert_offsets must be contiguous "
              "CUDA int32.");
  TORCH_CHECK(dense_expert_ids.is_cuda() &&
                  dense_expert_ids.scalar_type() == torch::kInt32 &&
                  dense_expert_ids.is_contiguous(),
              "fp8_moe_dense_stage_sm70_out: dense_expert_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "fp8_moe_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "fp8_moe_dense_stage_sm70_out: num_experts must be positive.");
  TORCH_CHECK(
      group_size == 128,
      "fp8_moe_dense_stage_sm70_out: only group_size=128 is supported.");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == k,
              "fp8_moe_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == input.size(0) && out.size(1) == n,
      "fp8_moe_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1,
              "fp8_moe_dense_stage_sm70_out: expert_offsets too small.");
  TORCH_CHECK(dense_expert_ids.numel() >= num_experts,
              "fp8_moe_dense_stage_sm70_out: dense_expert_ids too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  maybe_log_sm70_moe_route_once(
      "logged_fp8_dense_stage",
      "SM70 FP8 MoE CUDA-graph-safe dense-stage path enabled C++ op reached",
      input, input.size(0), num_experts);
  for (int expert = 0; expert < static_cast<int>(num_experts); ++expert) {
    torch::Tensor offsets = expert_offsets.narrow(0, expert, 2);
    torch::Tensor expert_idx = dense_expert_ids.narrow(0, expert, 1);
    fp8_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, torch::Tensor(),
                               torch::Tensor(), false, -1, -1, torch::Tensor(),
                               expert_idx);
  }
}

void nvfp4_moe_gemm_sm70_out_impl(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w, torch::Tensor strided_ptrs_s,
    int64_t num_experts, int64_t k, int64_t n, int64_t group_size,
    torch::Tensor b_group_indices, bool compact_grouped_rows = false,
    int64_t logical_total_tokens = -1,
    torch::Tensor a_indices = torch::Tensor(), bool gated_silu = false) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "nvfp4_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.scalar_type() == torch::kInt32 &&
          expert_offsets.is_contiguous(),
      "nvfp4_moe_gemm_sm70: expert_offsets must be contiguous CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "nvfp4_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "nvfp4_moe_gemm_sm70: output must be CUDA float16.");
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "nvfp4_moe_gemm_sm70: invalid dimensions.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_moe_gemm_sm70: only group_size=16 is supported.");
  TORCH_CHECK(k % group_size == 0,
              "nvfp4_moe_gemm_sm70: k must be divisible by group_size.");
  TORCH_CHECK(sorted_input.dim() == 2 && sorted_input.size(1) == k,
              "nvfp4_moe_gemm_sm70: input shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1,
              "nvfp4_moe_gemm_sm70: expert_offsets too small.");
  TORCH_CHECK(b_group_indices.is_cuda() &&
                  b_group_indices.scalar_type() == torch::kInt32 &&
                  b_group_indices.is_contiguous() &&
                  b_group_indices.numel() >= num_experts,
              "nvfp4_moe_gemm_sm70: B group indices must be contiguous CUDA "
              "int32.");

  const int64_t input_tokens = sorted_input.size(0);
  const int64_t total_tokens =
      logical_total_tokens > 0 ? logical_total_tokens : input_tokens;
  const bool use_a_indices = a_indices.defined() && a_indices.numel() > 0;
  torch::Tensor a_indices_flat = a_indices;
  if (use_a_indices) {
    TORCH_CHECK(a_indices.is_cuda() &&
                    a_indices.scalar_type() == torch::kInt32 &&
                    a_indices.is_contiguous(),
                "nvfp4_moe_gemm_sm70: A row indices must be contiguous CUDA "
                "int32.");
    a_indices_flat = a_indices.view({-1});
    TORCH_CHECK(a_indices_flat.numel() >= total_tokens,
                "nvfp4_moe_gemm_sm70: A row indices size mismatch.");
  } else {
    TORCH_CHECK(input_tokens == total_tokens,
                "nvfp4_moe_gemm_sm70: non-indexed input rows must match "
                "logical rows.");
  }
  TORCH_CHECK(!gated_silu || n % 2 == 0,
              "nvfp4_moe_gemm_sm70: gated SiLU requires even n.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == total_tokens &&
                  out.size(1) == (gated_silu ? n / 2 : n) && out.stride(1) == 1,
              "nvfp4_moe_gemm_sm70: output must be contiguous [tokens, n].");
  if (total_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto fp4_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto fp8_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = fp4_converters[0];
  const auto* conv_s = fp8_converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "nvfp4_moe_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(sorted_input.stride(0)),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();
  desc_A.idxs = use_a_indices ? a_indices_flat.data_ptr<int>() : nullptr;
  turbomind::gemm::MatrixLayout desc_U{};

  auto desc_B = vllm::sm70::packed_weight_layout(
      *conv_w, turbomind::kFloat4_e2m1, n, k, 0);
  desc_B.num = static_cast<int>(num_experts);
  desc_B.group_idxs = b_group_indices.data_ptr<int>();

  auto desc_V = vllm::sm70::packed_scale_layout(*conv_s, turbomind::kUint16, n,
                                                k / group_size, 0);
  desc_V.num = static_cast<int>(num_experts);
  desc_V.group_idxs = b_group_indices.data_ptr<int>();

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::select_nvfp4_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;
  op.dispatch_num_override = compact_grouped_rows ? 1 : 0;
  const bool dynamic_glm53_expert_rows =
      compact_grouped_rows && num_experts >= 16 && num_experts <= 80 &&
      num_experts % 8 == 0 && group_size == 16 &&
      ((k == 4096 && (n == 512 || n == 1024)) ||
       (n == 4096 && (k == 256 || k == 512))) &&
      vllm::awq_sm70::nvfp4_moe_grouped_expert_rows_enabled();
  op.active_group_count = compact_grouped_rows && !dynamic_glm53_expert_rows
                              ? -static_cast<int>(num_experts)
                              : 0;

  auto& workspace_holder = vllm::awq_sm70::workspace_for(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int ec = gemm.Run(
      op, 1.f, sorted_input.data_ptr(), desc_A, nullptr, desc_U,
      strided_ptrs_w.data_ptr(), desc_B, strided_ptrs_s.data_ptr(), desc_V, 0.f,
      out.data_ptr(), desc_D, out.data_ptr(), desc_D, workspace_holder, stream);
  TORCH_CHECK(ec == 0,
              "nvfp4_moe_gemm_sm70: TurboMind batched GEMM failed (ec=", ec,
              ").");
}

void nvfp4_moe_dense_stage_sm70_out(torch::Tensor out, torch::Tensor input,
                                    torch::Tensor expert_offsets,
                                    torch::Tensor dense_expert_ids,
                                    torch::Tensor ptrs_w, torch::Tensor ptrs_s,
                                    int64_t num_experts, int64_t k, int64_t n,
                                    int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "nvfp4_moe_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "nvfp4_moe_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous(),
              "nvfp4_moe_dense_stage_sm70_out: expert_offsets must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(dense_expert_ids.is_cuda() &&
                  dense_expert_ids.scalar_type() == torch::kInt32 &&
                  dense_expert_ids.is_contiguous(),
              "nvfp4_moe_dense_stage_sm70_out: dense_expert_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "nvfp4_moe_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "nvfp4_moe_dense_stage_sm70_out: num_experts must be positive.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_moe_dense_stage_sm70_out: only group_size=16 is "
              "supported.");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == k,
              "nvfp4_moe_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == input.size(0) && out.size(1) == n,
      "nvfp4_moe_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1,
              "nvfp4_moe_dense_stage_sm70_out: expert_offsets too small.");
  TORCH_CHECK(dense_expert_ids.numel() >= num_experts,
              "nvfp4_moe_dense_stage_sm70_out: dense_expert_ids too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  maybe_log_sm70_moe_route_once(
      "logged_nvfp4_dense_stage",
      "SM70 NVFP4 MoE CUDA-graph-safe TurboMind path enabled C++ op reached",
      input, input.size(0), num_experts);
  constexpr int kNvfp4LegacyCompactGroups = 8 * 8;
  // Cover the validated top-k8 B10 verifier bound. Duplicate expert selections
  // remain separate one-row groups, so this is a routed-slot bound rather
  // than an active-expert bound. Keep the dense-grouped cutoff independent:
  // a disabled compact route above 64 rows must not fall back per expert.
  constexpr int kNvfp4MaxCompactGroups = 10 * 8;
  const bool compact_decode_shape =
      input.size(0) == num_experts && num_experts <= kNvfp4MaxCompactGroups;
  if (compact_decode_shape) {
    nvfp4_moe_gemm_sm70_out_impl(out, input, expert_offsets, ptrs_w, ptrs_s,
                                 num_experts, k, n, group_size,
                                 dense_expert_ids, true);
    return;
  }
  const bool exact_qwen36_prefill_shape =
      input.size(0) > kNvfp4LegacyCompactGroups && num_experts == 256 &&
      ((k == 2048 && (n == 1024 || n == 512 || n == 256)) ||
       (n == 2048 && (k == 512 || k == 256 || k == 128)));
  const bool exact_qwen4_exp_prefill_shape =
      input.size(0) > kNvfp4LegacyCompactGroups && num_experts == 512 &&
      ((k == 2560 && n == 320) || (k == 160 && n == 2560));
  if (vllm::awq_sm70::nvfp4_moe_grouped_prefill_enabled() &&
      (exact_qwen36_prefill_shape || exact_qwen4_exp_prefill_shape)) {
    maybe_log_sm70_moe_route_once(
        "logged_nvfp4_grouped_prefill",
        "SM70 NVFP4 MoE grouped TurboMind prefill path enabled C++ op reached",
        input, input.size(0), num_experts);
    nvfp4_moe_gemm_sm70_out_impl(out, input, expert_offsets, ptrs_w, ptrs_s,
                                 num_experts, k, n, group_size,
                                 dense_expert_ids);
    return;
  }
  for (int expert = 0; expert < static_cast<int>(num_experts); ++expert) {
    torch::Tensor offsets = expert_offsets.narrow(0, expert, 2);
    torch::Tensor expert_idx = dense_expert_ids.narrow(0, expert, 1);
    nvfp4_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                                 group_size, expert_idx);
  }
}

void fp8_moe_single_token_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_dense_stage_sm70_out: input must be CUDA float16.");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_dense_stage_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "fp8_moe_single_token_dense_stage_sm70_out: index buffers must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(
      ptrs_w.is_cuda() && ptrs_s.is_cuda(),
      "fp8_moe_single_token_dense_stage_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "fp8_moe_single_token_dense_stage_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
      "fp8_moe_single_token_dense_stage_sm70_out: input shape mismatch.");
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
              "fp8_moe_single_token_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "fp8_moe_single_token_dense_stage_sm70_out: index buffer too small.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_dense_stage_sm70_out: only group_size=128 "
              "is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  for (int route = 0; route < static_cast<int>(top_k); ++route) {
    torch::Tensor offsets = expert_offsets.narrow(0, route, 2);
    torch::Tensor expert_idx = sorted_expert_ids.narrow(0, route, 1);
    fp8_moe_gemm_sm70_out_impl(out, input, offsets, ptrs_w, ptrs_s, 1, k, n,
                               group_size, false, torch::Tensor(),
                               torch::Tensor(), false, -1, -1, torch::Tensor(),
                               expert_idx);
  }
}

void fp8_moe_single_token_indexed_dense_stage_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor expert_offsets,
    torch::Tensor sorted_expert_ids, torch::Tensor ptrs_w, torch::Tensor ptrs_s,
    int64_t top_k, int64_t k, int64_t n, int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: input must "
              "be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: out must be "
              "CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets.is_contiguous() &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_contiguous(),
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: index "
              "buffers must be contiguous CUDA int32.");
  TORCH_CHECK(ptrs_w.is_cuda() && ptrs_s.is_cuda(),
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: ptr rows "
              "must be CUDA.");
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: top_k must "
              "be in [1, 32].");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= top_k && input.size(1) == k,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: input shape "
              "mismatch.");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) >= top_k && out.size(1) == n,
      "fp8_moe_single_token_indexed_dense_stage_sm70_out: out shape mismatch.");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 && sorted_expert_ids.numel() >= top_k,
      "fp8_moe_single_token_indexed_dense_stage_sm70_out: index buffer too "
      "small.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_indexed_dense_stage_sm70_out: only "
              "group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  fp8_moe_gemm_sm70_out_impl(out, input, expert_offsets, ptrs_w, ptrs_s, top_k,
                             k, n, group_size, false, torch::Tensor(),
                             torch::Tensor(), false, -1, -1, torch::Tensor(),
                             sorted_expert_ids);
}

void fp8_moe_single_token_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_dense_w13_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_dense_w13_sm70_out: scratch buffers must "
              "be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "fp8_moe_single_token_dense_w13_sm70_out: topk_ids must be "
              "contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_dense_w13_sm70_out: index buffers must be "
              "CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(
      w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
      "fp8_moe_single_token_dense_w13_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_dense_w13_sm70_out: x must have shape [1, "
              "hidden].");
  TORCH_CHECK(x.size(1) == hidden_logical_size,
              "fp8_moe_single_token_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(
      top_k > 0 && top_k <= 32,
      "fp8_moe_single_token_dense_w13_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(
      expert_offsets.numel() >= top_k + 1 &&
          expert_offsets64.numel() >= top_k + 1 &&
          inv_permuted_idx.numel() >= top_k &&
          sorted_expert_ids.numel() >= top_k,
      "fp8_moe_single_token_dense_w13_sm70_out: index buffer too small.");
  TORCH_CHECK(
      compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
          compact_input.size(1) == hidden_logical_size,
      "fp8_moe_single_token_dense_w13_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(
      gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
          gate_up.size(1) == w13_n,
      "fp8_moe_single_token_dense_w13_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_dense_w13_sm70_out: only group_size=128 is "
              "supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  maybe_log_sm70_moe_route_once("logged_fp8_single_token",
                                "SM70 FP8 MoE single-token active-expert dense "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_moe_single_token_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void fp8_moe_single_token_indexed_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_indexed_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda(),
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "fp8_moe_single_token_indexed_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_indexed_dense_w13_sm70_out: only "
              "group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  maybe_log_sm70_moe_route_once(
      "logged_fp8_single_token_indexed",
      "SM70 FP8 MoE single-token indexed dense path enabled C++ op reached", x,
      x.size(0), top_k);
  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, false, vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_moe_single_token_indexed_dense_stage_sm70_out(
      gate_up, compact_input, expert_offsets, sorted_expert_ids, w13_ptrs_w,
      w13_ptrs_s, top_k, w13_k, w13_n, group_size);
}

void fp8_moe_single_token_compact_dense_w13_sm70_out(
    torch::Tensor gate_up, torch::Tensor compact_input, torch::Tensor x,
    torch::Tensor topk_ids, torch::Tensor w13_ptrs_w, torch::Tensor w13_ptrs_s,
    torch::Tensor compact_w13_ptrs_w, torch::Tensor compact_w13_ptrs_s,
    torch::Tensor expert_offsets, torch::Tensor expert_offsets64,
    torch::Tensor inv_permuted_idx, torch::Tensor sorted_expert_ids,
    int64_t w13_k, int64_t w13_n, int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: x must be CUDA "
              "float16.");
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kFloat16 &&
                  compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: scratch "
              "buffers must be CUDA float16.");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kInt32 &&
                  topk_ids.is_contiguous(),
              "fp8_moe_single_token_compact_dense_w13_sm70_out: topk_ids must "
              "be contiguous CUDA int32.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  expert_offsets64.is_cuda() &&
                  expert_offsets64.scalar_type() == torch::kInt64 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32 &&
                  sorted_expert_ids.is_cuda() &&
                  sorted_expert_ids.scalar_type() == torch::kInt32,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: index buffers "
              "must be CUDA int32/int64.");
  TORCH_CHECK(
      expert_offsets.is_contiguous() && expert_offsets64.is_contiguous() &&
          inv_permuted_idx.is_contiguous() && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_compact_dense_w13_sm70_out: index buffers must be "
      "contiguous.");
  TORCH_CHECK(w13_ptrs_w.is_cuda() && w13_ptrs_s.is_cuda() &&
                  compact_w13_ptrs_w.is_cuda() && compact_w13_ptrs_s.is_cuda(),
              "fp8_moe_single_token_compact_dense_w13_sm70_out: ptr rows must "
              "be CUDA.");
  TORCH_CHECK(compact_w13_ptrs_w.scalar_type() == torch::kUInt8 &&
                  compact_w13_ptrs_s.scalar_type() == torch::kUInt8,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "rows must be uint8.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: x must have "
              "shape [1, hidden].");
  TORCH_CHECK(
      x.size(1) == hidden_logical_size,
      "fp8_moe_single_token_compact_dense_w13_sm70_out: hidden size mismatch.");

  topk_ids = topk_ids.view({-1});
  inv_permuted_idx = inv_permuted_idx.view({-1});
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: top_k must be "
              "in [1, 32].");
  TORCH_CHECK(expert_offsets.numel() >= top_k + 1 &&
                  expert_offsets64.numel() >= top_k + 1 &&
                  inv_permuted_idx.numel() >= top_k &&
                  sorted_expert_ids.numel() >= top_k,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: index buffer "
              "too small.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) >= top_k &&
                  compact_input.size(1) == hidden_logical_size,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: compact_input "
              "shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: gate_up shape "
              "mismatch.");
  TORCH_CHECK(group_size == 128,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: only "
              "group_size=128 is supported.");

  constexpr int kThreads = 256;
  constexpr int kPtrRowBytes = 16;
  TORCH_CHECK(compact_w13_ptrs_w.numel() >= top_k * kPtrRowBytes &&
                  compact_w13_ptrs_s.numel() >= top_k * kPtrRowBytes,
              "fp8_moe_single_token_compact_dense_w13_sm70_out: compact ptr "
              "buffer too small.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  maybe_log_sm70_moe_route_once("logged_fp8_single_token_compact",
                                "SM70 FP8 MoE single-token compact grouped W13 "
                                "path enabled C++ op reached",
                                x, x.size(0), top_k);
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();

  awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(), nullptr,
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      w13_ptrs_w.data_ptr<uint8_t>(), w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      compact_w13_ptrs_w.data_ptr<uint8_t>(),
      compact_w13_ptrs_s.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()), nullptr,
      expert_offsets.data_ptr<int32_t>(), expert_offsets64.data_ptr<int64_t>(),
      inv_permuted_idx.data_ptr<int32_t>(),
      sorted_expert_ids.data_ptr<int32_t>(), top_k,
      static_cast<int>(hidden_logical_size), kPtrRowBytes, kPtrRowBytes,
      kPtrRowBytes, true, vectorized_prepare);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_moe_gemm_sm70_out_impl(gate_up, compact_input, expert_offsets,
                             compact_w13_ptrs_w, compact_w13_ptrs_s, top_k,
                             w13_k, w13_n, group_size, false, torch::Tensor(),
                             torch::Tensor(), false, -1, -1);
}

void fp8_moe_single_token_sm70_out(
    torch::Tensor out, torch::Tensor x, torch::Tensor topk_weights,
    torch::Tensor topk_ids, torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows, torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows, torch::Tensor compact_input,
    torch::Tensor gate_up, torch::Tensor intermediate,
    torch::Tensor sorted_output, torch::Tensor sorted_weights,
    torch::Tensor dst_w13_ptrs_w_rows, torch::Tensor dst_w13_ptrs_s_rows,
    torch::Tensor dst_w2_ptrs_w_rows, torch::Tensor dst_w2_ptrs_s_rows,
    torch::Tensor expert_offsets, torch::Tensor inv_permuted_idx,
    torch::Tensor sorted_expert_ids, torch::Tensor broadcast_input_indices,
    torch::Tensor w2_raw_weight, torch::Tensor w2_raw_scale_inv, int64_t w13_k,
    int64_t w13_n, int64_t w2_k, int64_t w2_n, int64_t group_size,
    int64_t hidden_logical_size, bool fused_gated_silu,
    bool fused_weighted_reduce, bool broadcast_input, bool w2_direct_reduce,
    bool indexed_expert_ptrs, bool exact_per_route) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "fp8_moe_single_token_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_single_token_sm70_out: topk_weights must be CUDA float32.");
  TORCH_CHECK(
      topk_ids.is_cuda() && (topk_ids.scalar_type() == torch::kInt32 ||
                             topk_ids.scalar_type() == torch::kInt64),
      "fp8_moe_single_token_sm70_out: topk_ids must be CUDA int32/int64.");
  TORCH_CHECK(
      compact_input.is_cuda() &&
          compact_input.scalar_type() == torch::kFloat16 && gate_up.is_cuda() &&
          gate_up.scalar_type() == torch::kFloat16 && intermediate.is_cuda() &&
          intermediate.scalar_type() == torch::kFloat16 &&
          sorted_output.is_cuda() &&
          sorted_output.scalar_type() == torch::kFloat16,
      "fp8_moe_single_token_sm70_out: scratch buffers must be CUDA float16.");
  TORCH_CHECK(
      sorted_weights.is_cuda() &&
          sorted_weights.scalar_type() == torch::kFloat32,
      "fp8_moe_single_token_sm70_out: sorted_weights must be CUDA float32.");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.scalar_type() == torch::kInt32 &&
          inv_permuted_idx.is_cuda() &&
          inv_permuted_idx.scalar_type() == torch::kInt32 &&
          sorted_expert_ids.is_cuda() &&
          sorted_expert_ids.scalar_type() == torch::kInt32 &&
          broadcast_input_indices.is_cuda() &&
          broadcast_input_indices.scalar_type() == torch::kInt32,
      "fp8_moe_single_token_sm70_out: index buffers must be CUDA int32.");
  TORCH_CHECK(
      src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
          src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda() &&
          dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
          dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
      "fp8_moe_single_token_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "fp8_moe_single_token_sm70_out: x must have shape [1, hidden].");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == 1,
      "fp8_moe_single_token_sm70_out: out must have shape [1, hidden].");
  TORCH_CHECK(out.size(1) == hidden_logical_size,
              "fp8_moe_single_token_sm70_out: out cols must match "
              "hidden_logical_size.");

  topk_ids = topk_ids.contiguous().view({-1});
  topk_weights = topk_weights.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});
  broadcast_input_indices = broadcast_input_indices.contiguous().view({-1});

  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "fp8_moe_single_token_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(static_cast<int>(topk_weights.numel()) == top_k,
              "fp8_moe_single_token_sm70_out: topk_weights size mismatch.");
  TORCH_CHECK(compact_input.dim() == 2 && compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "fp8_moe_single_token_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) == top_k &&
                  gate_up.size(1) == w13_n,
              "fp8_moe_single_token_sm70_out: gate_up shape mismatch.");
  TORCH_CHECK(intermediate.dim() == 2 && intermediate.size(0) == top_k &&
                  intermediate.size(1) == w13_n / 2,
              "fp8_moe_single_token_sm70_out: intermediate shape mismatch.");
  TORCH_CHECK(sorted_output.dim() == 2 && sorted_output.size(0) == top_k &&
                  sorted_output.size(1) == w2_n,
              "fp8_moe_single_token_sm70_out: sorted_output shape mismatch.");
  TORCH_CHECK(sorted_weights.numel() == top_k,
              "fp8_moe_single_token_sm70_out: sorted_weights size mismatch.");
  const int64_t source_num_experts = src_w13_ptrs_w_rows.size(0);
  TORCH_CHECK(
      source_num_experts > 0,
      "fp8_moe_single_token_sm70_out: source expert count must be positive.");
  const int64_t expected_offsets =
      exact_per_route ? source_num_experts + 1 : top_k + 1;
  TORCH_CHECK(expert_offsets.numel() == expected_offsets,
              "fp8_moe_single_token_sm70_out: expert_offsets size mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "fp8_moe_single_token_sm70_out: inv_permuted_idx size mismatch.");
  TORCH_CHECK(
      sorted_expert_ids.numel() >= top_k && sorted_expert_ids.is_contiguous(),
      "fp8_moe_single_token_sm70_out: sorted expert ids must be contiguous and "
      "large enough.");
  if (broadcast_input && broadcast_input_indices.numel() > 0) {
    TORCH_CHECK(broadcast_input_indices.numel() >= top_k,
                "fp8_moe_single_token_sm70_out: broadcast input indices size "
                "mismatch.");
  }
  if (w2_direct_reduce) {
    TORCH_CHECK(!fused_weighted_reduce,
                "fp8_moe_single_token_sm70_out: W2 direct reduce conflicts "
                "with weighted-reduce epilogue.");
    TORCH_CHECK(w2_raw_weight.defined() && w2_raw_scale_inv.defined(),
                "fp8_moe_single_token_sm70_out: W2 direct reduce requires raw "
                "W2 tensors.");
  }
  if (indexed_expert_ptrs) {
    TORCH_CHECK(src_w13_ptrs_w_rows.is_contiguous() &&
                    src_w13_ptrs_s_rows.is_contiguous() &&
                    src_w2_ptrs_w_rows.is_contiguous() &&
                    src_w2_ptrs_s_rows.is_contiguous(),
                "fp8_moe_single_token_sm70_out: indexed expert ptrs require "
                "contiguous source ptr rows.");
  }
  if (exact_per_route) {
    TORCH_CHECK(!fused_gated_silu && !fused_weighted_reduce &&
                    !broadcast_input && !w2_direct_reduce,
                "fp8_moe_single_token_sm70_out: exact per-route mode only "
                "supports the unfused compact path.");
    TORCH_CHECK(src_w13_ptrs_w_rows.is_contiguous() &&
                    src_w13_ptrs_s_rows.is_contiguous() &&
                    src_w2_ptrs_w_rows.is_contiguous() &&
                    src_w2_ptrs_s_rows.is_contiguous(),
                "fp8_moe_single_token_sm70_out: exact per-route mode requires "
                "contiguous source ptr rows.");
  }
  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(
      src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
          src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
          dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
          dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
      "fp8_moe_single_token_sm70_out: ptr row tensors must be 2D.");
  TORCH_CHECK(
      dst_w13_ptrs_w_rows.size(0) == top_k &&
          dst_w13_ptrs_s_rows.size(0) == top_k &&
          dst_w2_ptrs_w_rows.size(0) == top_k &&
          dst_w2_ptrs_s_rows.size(0) == top_k &&
          dst_w13_ptrs_w_rows.size(1) == row_bytes &&
          dst_w13_ptrs_s_rows.size(1) == row_bytes &&
          dst_w2_ptrs_w_rows.size(1) == row_bytes &&
          dst_w2_ptrs_s_rows.size(1) == row_bytes,
      "fp8_moe_single_token_sm70_out: destination ptr row shapes mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));
  const bool vectorized_prepare = sm70_fp8_moe_prepare_vec_enabled();
  const bool prepare_sorted_weight_copy =
      fused_weighted_reduce || !vectorized_prepare;
  const float* prepare_topk_weights =
      prepare_sorted_weight_copy ? topk_weights.data_ptr<float>() : nullptr;
  float* prepare_sorted_weights =
      prepare_sorted_weight_copy ? sorted_weights.data_ptr<float>() : nullptr;
  constexpr int kThreads = 256;
  const bool copy_expert_ptr_rows = !indexed_expert_ptrs && !exact_per_route;

  if (exact_per_route) {
    if (topk_ids.scalar_type() == torch::kInt32) {
      awq_moe_single_token_exact_layout_prepare_kernel<int32_t>
          <<<1, kThreads, 0, stream>>>(
              topk_ids.data_ptr<int32_t>(),
              reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
              expert_offsets.data_ptr<int32_t>(), nullptr,
              inv_permuted_idx.data_ptr<int32_t>(),
              sorted_expert_ids.data_ptr<int32_t>(), top_k,
              static_cast<int>(x.size(1)),
              static_cast<int>(source_num_experts));
    } else {
      awq_moe_single_token_exact_layout_prepare_kernel<int64_t>
          <<<1, kThreads, 0, stream>>>(
              topk_ids.data_ptr<int64_t>(),
              reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
              reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
              expert_offsets.data_ptr<int32_t>(), nullptr,
              inv_permuted_idx.data_ptr<int32_t>(),
              sorted_expert_ids.data_ptr<int32_t>(), top_k,
              static_cast<int>(x.size(1)),
              static_cast<int>(source_num_experts));
    }
  } else if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_prepare_kernel<int32_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), prepare_topk_weights,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        broadcast_input
            ? nullptr
            : reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        prepare_sorted_weights, expert_offsets.data_ptr<int32_t>(), nullptr,
        inv_permuted_idx.data_ptr<int32_t>(),
        indexed_expert_ptrs ? sorted_expert_ids.data_ptr<int32_t>() : nullptr,
        top_k, static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), copy_expert_ptr_rows, vectorized_prepare);
  } else {
    awq_moe_single_token_prepare_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), prepare_topk_weights,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
        dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
        broadcast_input
            ? nullptr
            : reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
        prepare_sorted_weights, expert_offsets.data_ptr<int32_t>(), nullptr,
        inv_permuted_idx.data_ptr<int32_t>(),
        indexed_expert_ptrs ? sorted_expert_ids.data_ptr<int32_t>() : nullptr,
        top_k, static_cast<int>(x.size(1)), src_row_stride, dst_row_stride,
        static_cast<int>(row_bytes), copy_expert_ptr_rows, vectorized_prepare);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor w13_input = broadcast_input ? x : compact_input;
  const int64_t w13_logical_total_tokens = broadcast_input ? top_k : -1;
  const int64_t w13_a_ld_override = -1;
  torch::Tensor w13_input_indices =
      (broadcast_input && broadcast_input_indices.numel() > 0)
          ? broadcast_input_indices
          : torch::Tensor();
  torch::Tensor w13_ptrs_w =
      indexed_expert_ptrs ? src_w13_ptrs_w_rows : dst_w13_ptrs_w_rows;
  torch::Tensor w13_ptrs_s =
      indexed_expert_ptrs ? src_w13_ptrs_s_rows : dst_w13_ptrs_s_rows;
  torch::Tensor w2_ptrs_w =
      indexed_expert_ptrs ? src_w2_ptrs_w_rows : dst_w2_ptrs_w_rows;
  torch::Tensor w2_ptrs_s =
      indexed_expert_ptrs ? src_w2_ptrs_s_rows : dst_w2_ptrs_s_rows;
  torch::Tensor expert_ptr_indices =
      indexed_expert_ptrs ? sorted_expert_ids : torch::Tensor();
  const bool single_token_per_expert_dispatch =
      vllm::awq_sm70::fp8_moe_single_token_per_expert_dispatch_enabled();
  if (exact_per_route) {
    fp8_moe_gemm_sm70_out_impl(
        gate_up, compact_input, expert_offsets, src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows, source_num_experts, w13_k, w13_n, group_size,
        false, torch::Tensor(), torch::Tensor(), false, -1, -1, torch::Tensor(),
        torch::Tensor(), false, -1, sorted_expert_ids, top_k);
    sm70_silu_and_mul_fp16_out(intermediate, gate_up);
    fp8_moe_gemm_sm70_out_impl(
        sorted_output, intermediate, expert_offsets, src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows, source_num_experts, w2_k, w2_n, group_size, false,
        torch::Tensor(), torch::Tensor(), false, -1, -1, torch::Tensor(),
        torch::Tensor(), false, -1, sorted_expert_ids, top_k);
  } else {
    if (fused_gated_silu) {
      fp8_moe_gemm_sm70_out_impl(
          intermediate, w13_input, expert_offsets, w13_ptrs_w, w13_ptrs_s,
          top_k, w13_k, w13_n, group_size, true, torch::Tensor(),
          torch::Tensor(), false, w13_logical_total_tokens, w13_a_ld_override,
          w13_input_indices, expert_ptr_indices,
          single_token_per_expert_dispatch, -1);
    } else {
      fp8_moe_gemm_sm70_out_impl(
          gate_up, w13_input, expert_offsets, w13_ptrs_w, w13_ptrs_s, top_k,
          w13_k, w13_n, group_size, false, torch::Tensor(), torch::Tensor(),
          false, w13_logical_total_tokens, w13_a_ld_override, w13_input_indices,
          expert_ptr_indices, single_token_per_expert_dispatch, -1);
      sm70_silu_and_mul_fp16_out(intermediate, gate_up);
    }
    if (fused_weighted_reduce) {
      fp8_moe_gemm_sm70_out_impl(sorted_output, intermediate, expert_offsets,
                                 w2_ptrs_w, w2_ptrs_s, top_k, w2_k, w2_n,
                                 group_size, false, out, sorted_weights, true,
                                 -1, -1, torch::Tensor(), expert_ptr_indices,
                                 single_token_per_expert_dispatch, -1);
      return;
    }
    if (w2_direct_reduce) {
      fp8_moe_w2_direct_reduce_sm70_out(out, intermediate, topk_weights,
                                        topk_ids, inv_permuted_idx,
                                        w2_raw_weight, w2_raw_scale_inv, top_k,
                                        w2_k, w2_n, group_size, stream);
      return;
    }

    fp8_moe_gemm_sm70_out_impl(
        sorted_output, intermediate, expert_offsets, w2_ptrs_w, w2_ptrs_s,
        top_k, w2_k, w2_n, group_size, false, torch::Tensor(), torch::Tensor(),
        false, -1, -1, torch::Tensor(), expert_ptr_indices,
        single_token_per_expert_dispatch, -1);
  }

  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride =
      static_cast<int>(sorted_output.stride(0));
  if (top_k == 8 && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i, sorted_output_row_stride);
  } else {
    const int blocks =
        std::max<int>(1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), top_k,
        hidden_logical_size_i, sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Op registration (K03). Schemas are the fork's minus `str? native_policy`.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork csrc/torch_bindings.cpp:1167
  m.def(
      "awq_moe_build_strided_ptrs(Tensor tm_weights, Tensor tm_scales, "
      "int k_ld, int q_ld, int num_experts) -> Tensor[]");
  // fork csrc/torch_bindings.cpp:1221
  m.def(
      "awq_moe_chunked_w2_sm70_out(Tensor(a!) out, Tensor(b!) "
      "chunk_output, Tensor input, Tensor expert_offsets, Tensor "
      "permuted_idx, Tensor topk_weights, Tensor(c!) "
      "chunk_expert_offsets, Tensor(d!) chunk_range_begin, Tensor(e!) "
      "chunk_range_end, Tensor(f!) chunk_a_indices, Tensor(g!) "
      "chunk_inv_permuted_idx, Tensor ptrs_w, Tensor ptrs_s, int "
      "num_tokens, int top_k, int num_experts, int k, int n, int "
      "hidden_logical_size, int group_size, int chunk_tokens) -> ()");
  // fork csrc/torch_bindings.cpp:1287
  m.def(
      "awq_moe_single_token_exact_layout_prepare(Tensor topk_ids, Tensor "
      "x, Tensor(a!) compact_input, Tensor(b!) expert_offsets, Tensor(c!)"
      " expert_offsets64, Tensor(d!) inv_permuted_idx, int num_experts) "
      "-> ()");
  // fork csrc/torch_bindings.cpp:1297
  m.def(
      "awq_moe_single_token_weighted_reduce_out(Tensor sorted_output, "
      "Tensor topk_weights, Tensor inv_permuted_idx, Tensor(a!) out, int "
      "top_k, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1327
  m.def(
      "fp8_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, Tensor "
      "expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, int "
      "num_experts, int k, int n, int group_size, bool gated_silu) -> ()");
  // fork csrc/torch_bindings.cpp:1335
  m.def(
      "fp8_moe_gemm_sm70_per_expert_dispatch_out(Tensor(a!) out, Tensor "
      "sorted_input, Tensor expert_offsets, Tensor strided_ptrs_w, Tensor"
      " strided_ptrs_s, int num_experts, int k, int n, int group_size, "
      "bool gated_silu) -> ()");
  // fork csrc/torch_bindings.cpp:1345
  m.def(
      "fp8_moe_dense_stage_sm70_out(Tensor(a!) out, Tensor input, Tensor "
      "expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, Tensor "
      "ptrs_s, int num_experts, int k, int n, int group_size) -> ()");
  // fork csrc/torch_bindings.cpp:1531
  m.def(
      "fp8_moe_single_token_dense_stage_sm70_out(Tensor(a!) out, Tensor "
      "input, Tensor expert_offsets, Tensor sorted_expert_ids, Tensor "
      "ptrs_w, Tensor ptrs_s, int top_k, int k, int n, int group_size) ->"
      " ()");
  // fork csrc/torch_bindings.cpp:1539
  m.def(
      "fp8_moe_single_token_indexed_dense_stage_sm70_out(Tensor(a!) out, "
      "Tensor input, Tensor expert_offsets, Tensor sorted_expert_ids, "
      "Tensor ptrs_w, Tensor ptrs_s, int top_k, int k, int n, int "
      "group_size) -> ()");
  // fork csrc/torch_bindings.cpp:1548
  m.def(
      "fp8_moe_single_token_dense_w13_sm70_out(Tensor(a!) gate_up, "
      "Tensor(b!) compact_input, Tensor x, Tensor topk_ids, Tensor "
      "w13_ptrs_w, Tensor w13_ptrs_s, Tensor(c!) expert_offsets, "
      "Tensor(d!) expert_offsets64, Tensor(e!) inv_permuted_idx, "
      "Tensor(f!) sorted_expert_ids, int w13_k, int w13_n, int "
      "group_size, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1559
  m.def(
      "fp8_moe_single_token_indexed_dense_w13_sm70_out(Tensor(a!) "
      "gate_up, Tensor(b!) compact_input, Tensor x, Tensor topk_ids, "
      "Tensor w13_ptrs_w, Tensor w13_ptrs_s, Tensor(c!) expert_offsets, "
      "Tensor(d!) expert_offsets64, Tensor(e!) inv_permuted_idx, "
      "Tensor(f!) sorted_expert_ids, int w13_k, int w13_n, int "
      "group_size, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1571
  m.def(
      "fp8_moe_single_token_compact_dense_w13_sm70_out(Tensor(a!) "
      "gate_up, Tensor(b!) compact_input, Tensor x, Tensor topk_ids, "
      "Tensor w13_ptrs_w, Tensor w13_ptrs_s, Tensor(c!) "
      "compact_w13_ptrs_w, Tensor(d!) compact_w13_ptrs_s, Tensor(e!) "
      "expert_offsets, Tensor(f!) expert_offsets64, Tensor(g!) "
      "inv_permuted_idx, Tensor(h!) sorted_expert_ids, int w13_k, int "
      "w13_n, int group_size, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1584
  m.def(
      "fp8_moe_single_token_sm70_out(Tensor(a!) out, Tensor x, Tensor "
      "topk_weights, Tensor topk_ids, Tensor src_w13_ptrs_w_rows, Tensor "
      "src_w13_ptrs_s_rows, Tensor src_w2_ptrs_w_rows, Tensor "
      "src_w2_ptrs_s_rows, Tensor(b!) compact_input, Tensor(c!) gate_up, "
      "Tensor(d!) intermediate, Tensor(e!) sorted_output, Tensor(f!) "
      "sorted_weights, Tensor(g!) dst_w13_ptrs_w_rows, Tensor(h!) "
      "dst_w13_ptrs_s_rows, Tensor(i!) dst_w2_ptrs_w_rows, Tensor(j!) "
      "dst_w2_ptrs_s_rows, Tensor(k!) expert_offsets, Tensor(l!) "
      "inv_permuted_idx, Tensor(m!) sorted_expert_ids, Tensor "
      "broadcast_input_indices, Tensor w2_raw_weight, Tensor "
      "w2_raw_scale_inv, int w13_k, int w13_n, int w2_k, int w2_n, int "
      "group_size, int hidden_logical_size, bool fused_gated_silu, bool "
      "fused_weighted_reduce, bool broadcast_input, bool "
      "w2_direct_reduce, bool indexed_expert_ptrs, bool exact_per_route) "
      "-> ()");
  // fork csrc/torch_bindings.cpp:1491
  m.def(
      "nvfp4_moe_dense_stage_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, "
      "Tensor ptrs_s, int num_experts, int k, int n, int group_size) -> "
      "()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("awq_moe_build_strided_ptrs", &awq_moe_build_strided_ptrs);
  m.impl("awq_moe_chunked_w2_sm70_out", &awq_moe_chunked_w2_sm70_out);
  m.impl("awq_moe_single_token_exact_layout_prepare", &awq_moe_single_token_exact_layout_prepare);
  m.impl("awq_moe_single_token_weighted_reduce_out", &awq_moe_single_token_weighted_reduce_out);
  m.impl("fp8_moe_gemm_sm70_out", &fp8_moe_gemm_sm70_out);
  m.impl("fp8_moe_gemm_sm70_per_expert_dispatch_out", &fp8_moe_gemm_sm70_per_expert_dispatch_out);
  m.impl("fp8_moe_dense_stage_sm70_out", &fp8_moe_dense_stage_sm70_out);
  m.impl("fp8_moe_single_token_dense_stage_sm70_out", &fp8_moe_single_token_dense_stage_sm70_out);
  m.impl("fp8_moe_single_token_indexed_dense_stage_sm70_out", &fp8_moe_single_token_indexed_dense_stage_sm70_out);
  m.impl("fp8_moe_single_token_dense_w13_sm70_out", &fp8_moe_single_token_dense_w13_sm70_out);
  m.impl("fp8_moe_single_token_indexed_dense_w13_sm70_out", &fp8_moe_single_token_indexed_dense_w13_sm70_out);
  m.impl("fp8_moe_single_token_compact_dense_w13_sm70_out", &fp8_moe_single_token_compact_dense_w13_sm70_out);
  m.impl("fp8_moe_single_token_sm70_out", &fp8_moe_single_token_sm70_out);
  m.impl("nvfp4_moe_dense_stage_sm70_out", &nvfp4_moe_dense_stage_sm70_out);
}

#ifdef VLLM_SM70_TM_AWQ
TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork csrc/torch_bindings.cpp:1174
  m.def(
      "awq_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, Tensor "
      "expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, int "
      "num_experts, int k, int n, int group_size, bool gated_silu) -> ()");
  // fork csrc/torch_bindings.cpp:1182
  m.def(
      "awq_moe_gemm_sm70_per_expert_dispatch_out(Tensor(a!) out, Tensor "
      "sorted_input, Tensor expert_offsets, Tensor strided_ptrs_w, Tensor"
      " strided_ptrs_s, int num_experts, int k, int n, int group_size, "
      "bool gated_silu) -> ()");
  // fork csrc/torch_bindings.cpp:1192
  m.def(
      "awq_moe_dense_stage_sm70_out(Tensor(a!) out, Tensor input, Tensor "
      "expert_offsets, Tensor dense_expert_ids, Tensor ptrs_w, Tensor "
      "ptrs_s, int num_experts, int k, int n, int group_size) -> ()");
  // fork csrc/torch_bindings.cpp:1211
  m.def(
      "awq_moe_active_dense_stage_sm70_out(Tensor(a!) out, Tensor input, "
      "Tensor permuted_experts_id, Tensor(b!) active_expert_offsets, "
      "Tensor(c!) active_expert_ids, Tensor ptrs_w, Tensor ptrs_s, int "
      "total_slots, int k, int n, int group_size) -> ()");
  // fork csrc/torch_bindings.cpp:1234
  m.def(
      "awq_moe_single_token_dense_stage_sm70_out(Tensor(a!) out, Tensor "
      "input, Tensor expert_offsets, Tensor sorted_expert_ids, Tensor "
      "ptrs_w, Tensor ptrs_s, int top_k, int k, int n, int group_size) ->"
      " ()");
  // fork csrc/torch_bindings.cpp:1242
  m.def(
      "awq_moe_single_token_indexed_dense_stage_sm70_out(Tensor(a!) out, "
      "Tensor input, Tensor expert_offsets, Tensor sorted_expert_ids, "
      "Tensor ptrs_w, Tensor ptrs_s, int top_k, int k, int n, int "
      "group_size) -> ()");
  // fork csrc/torch_bindings.cpp:1251
  m.def(
      "awq_moe_single_token_dense_w13_sm70_out(Tensor(a!) gate_up, "
      "Tensor(b!) compact_input, Tensor x, Tensor topk_ids, Tensor "
      "w13_ptrs_w, Tensor w13_ptrs_s, Tensor(c!) expert_offsets, "
      "Tensor(d!) expert_offsets64, Tensor(e!) inv_permuted_idx, "
      "Tensor(f!) sorted_expert_ids, int w13_k, int w13_n, int "
      "group_size, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1262
  m.def(
      "awq_moe_single_token_indexed_dense_w13_sm70_out(Tensor(a!) "
      "gate_up, Tensor(b!) compact_input, Tensor x, Tensor topk_ids, "
      "Tensor w13_ptrs_w, Tensor w13_ptrs_s, Tensor(c!) expert_offsets, "
      "Tensor(d!) expert_offsets64, Tensor(e!) inv_permuted_idx, "
      "Tensor(f!) sorted_expert_ids, int w13_k, int w13_n, int "
      "group_size, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1274
  m.def(
      "awq_moe_single_token_compact_dense_w13_sm70_out(Tensor(a!) "
      "gate_up, Tensor(b!) compact_input, Tensor x, Tensor topk_ids, "
      "Tensor w13_ptrs_w, Tensor w13_ptrs_s, Tensor(c!) "
      "compact_w13_ptrs_w, Tensor(d!) compact_w13_ptrs_s, Tensor(e!) "
      "expert_offsets, Tensor(f!) expert_offsets64, Tensor(g!) "
      "inv_permuted_idx, Tensor(h!) sorted_expert_ids, int w13_k, int "
      "w13_n, int group_size, int hidden_logical_size) -> ()");
  // fork csrc/torch_bindings.cpp:1305
  m.def(
      "awq_moe_single_token_sm70_out(Tensor(a!) out, Tensor x, Tensor "
      "topk_weights, Tensor topk_ids, Tensor src_w13_ptrs_w_rows, Tensor "
      "src_w13_ptrs_s_rows, Tensor src_w2_ptrs_w_rows, Tensor "
      "src_w2_ptrs_s_rows, Tensor(b!) compact_input, Tensor(c!) "
      "intermediate, Tensor(d!) sorted_output, Tensor(e!) sorted_weights,"
      " Tensor(f!) dst_w13_ptrs_w_rows, Tensor(g!) dst_w13_ptrs_s_rows, "
      "Tensor(h!) dst_w2_ptrs_w_rows, Tensor(i!) dst_w2_ptrs_s_rows, "
      "Tensor(j!) expert_offsets, Tensor(k!) inv_permuted_idx, int w13_k,"
      " int w13_n, int w2_k, int w2_n, int group_size, int "
      "hidden_logical_size) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("awq_moe_gemm_sm70_out", &awq_moe_gemm_sm70_out);
  m.impl("awq_moe_gemm_sm70_per_expert_dispatch_out", &awq_moe_gemm_sm70_per_expert_dispatch_out);
  m.impl("awq_moe_dense_stage_sm70_out", &awq_moe_dense_stage_sm70_out);
  m.impl("awq_moe_active_dense_stage_sm70_out", &awq_moe_active_dense_stage_sm70_out);
  m.impl("awq_moe_single_token_dense_stage_sm70_out", &awq_moe_single_token_dense_stage_sm70_out);
  m.impl("awq_moe_single_token_indexed_dense_stage_sm70_out", &awq_moe_single_token_indexed_dense_stage_sm70_out);
  m.impl("awq_moe_single_token_dense_w13_sm70_out", &awq_moe_single_token_dense_w13_sm70_out);
  m.impl("awq_moe_single_token_indexed_dense_w13_sm70_out", &awq_moe_single_token_indexed_dense_w13_sm70_out);
  m.impl("awq_moe_single_token_compact_dense_w13_sm70_out", &awq_moe_single_token_compact_dense_w13_sm70_out);
  m.impl("awq_moe_single_token_sm70_out", &awq_moe_single_token_sm70_out);
}
#endif  // VLLM_SM70_TM_AWQ
