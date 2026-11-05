// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboMind fp16 dense GEMM, fp16 gate-mul and interleaved SiLU (K03).
// Moved verbatim from fork cf9492a csrc/sm70_turbomind/ops/awq_sm70_gemm.cu;
// see ../PROVENANCE.md for the fork line range of every function.
#include "tm_ops_common_sm70.cuh"

#include <torch/library.h>

// The fork defined this in the global namespace, next to a global
// silu_and_mul() with the same name as upstream vLLM's op. Keep it in
// vllm_sm70 so neither can collide with symbols from _C_stable_libtorch.
namespace vllm_sm70 {

void silu_and_mul_interleaved(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(),
              "silu_and_mul_interleaved SM70 op expects CUDA tensors.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half,
              "silu_and_mul_interleaved only supports float16.");
  TORCH_CHECK(input.dim() >= 1,
              "silu_and_mul_interleaved expects input rank >= 1.");
  TORCH_CHECK(input.size(-1) == out.size(-1) * 2,
              "silu_and_mul_interleaved input last dim must be twice output.");
  TORCH_CHECK(input.numel() == out.numel() * 2,
              "silu_and_mul_interleaved batch shape mismatch.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  sm70_silu_and_mul_interleaved_fp16_out(out, input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm_sm70

namespace vllm::awq_sm70 {
namespace {

void validate_f16_weight(const torch::Tensor& weight, const char* op_name) {
  TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be CUDA.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16, op_name,
              ": weight must be float16.");
  TORCH_CHECK(weight.dim() == 2, op_name, ": weight must be 2D.");
}

void validate_f16_input(const torch::Tensor& in_feats,
                        const torch::Tensor& tm_weight,
                        const torch::Tensor& out, bool gated_silu,
                        const char* op_name) {
  TORCH_CHECK(in_feats.is_cuda(), op_name, ": input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), op_name, ": weight must be CUDA.");
  TORCH_CHECK(out.is_cuda(), op_name, ": output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16, op_name,
              ": input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kFloat16, op_name,
              ": weight must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16, op_name,
              ": output must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, op_name, ": input must be 2D.");
  TORCH_CHECK(tm_weight.dim() == 2, op_name, ": weight must be 2D.");
  TORCH_CHECK(out.dim() == 2, op_name, ": output must be 2D.");
}

void validate_f16_gate_mul_input(const torch::Tensor& out,
                                 const torch::Tensor& in_feats,
                                 const torch::Tensor& gate_weight) {
  TORCH_CHECK(out.is_cuda(), "sm70_f16_gate_mul_out: out must be CUDA.");
  TORCH_CHECK(in_feats.is_cuda(), "sm70_f16_gate_mul_out: input must be CUDA.");
  TORCH_CHECK(gate_weight.is_cuda(),
              "sm70_f16_gate_mul_out: gate weight must be CUDA.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: out must be float16.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: input must be float16.");
  TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: gate weight must be float16.");
  TORCH_CHECK(out.dim() == 2, "sm70_f16_gate_mul_out: out must be 2D.");
  TORCH_CHECK(in_feats.dim() == 2, "sm70_f16_gate_mul_out: input must be 2D.");
  TORCH_CHECK(gate_weight.dim() == 2,
              "sm70_f16_gate_mul_out: gate weight must be 2D.");
  TORCH_CHECK(gate_weight.size(0) == 1,
              "sm70_f16_gate_mul_out: gate weight must have one output row.");
  TORCH_CHECK(out.size(0) == in_feats.size(0),
              "sm70_f16_gate_mul_out: out/input batch mismatch.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_gate_mul_out: out must be row-major contiguous.");
  TORCH_CHECK(in_feats.stride(1) == 1,
              "sm70_f16_gate_mul_out: input must be row-major contiguous.");
  TORCH_CHECK(gate_weight.stride(1) == 1,
              "sm70_f16_gate_mul_out: gate weight must be contiguous.");
  TORCH_CHECK(in_feats.size(1) == gate_weight.size(1),
              "sm70_f16_gate_mul_out: input/gate K mismatch.");
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <int THREADS>
__global__ void sm70_f16_gate_mul_kernel(half* out, const half* in_feats,
                                         const half* gate_weight, int out_rows,
                                         int out_cols, int k,
                                         int64_t out_row_stride,
                                         int64_t in_row_stride) {
  const int row = blockIdx.x;
  if (row >= out_rows) {
    return;
  }

  const int tid = threadIdx.x;
  const half* x_row = in_feats + row * in_row_stride;

  float dot = 0.f;
  const int k2 = k >> 1;
  const half2* x2 = reinterpret_cast<const half2*>(x_row);
  const half2* w2 = reinterpret_cast<const half2*>(gate_weight);

  for (int idx = tid; idx < k2; idx += THREADS) {
    const half2 a = x2[idx];
    const half2 b = w2[idx];
    const half2 prod = __hmul2(a, b);
    dot += __half2float(__low2half(prod));
    dot += __half2float(__high2half(prod));
  }
  if ((k & 1) && tid == 0) {
    dot += __half2float(x_row[k - 1]) * __half2float(gate_weight[k - 1]);
  }

  dot = warp_reduce_sum(dot);

  __shared__ float block_sum[THREADS / 32];
  if ((tid & 31) == 0) {
    block_sum[tid >> 5] = dot;
  }
  __syncthreads();

  if (tid < 32) {
    float sum = tid < (THREADS / 32) ? block_sum[tid] : 0.f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      block_sum[0] = 1.f / (1.f + __expf(-sum));
    }
  }
  __syncthreads();

  const float gate = block_sum[0];
  const half2 gate2 = __float2half2_rn(gate);
  half* out_row = out + row * out_row_stride;
  half2* out_row2 = reinterpret_cast<half2*>(out_row);
  const int out_cols2 = out_cols >> 1;
  for (int idx = tid; idx < out_cols2; idx += THREADS) {
    out_row2[idx] = __hmul2(out_row2[idx], gate2);
  }
  if ((out_cols & 1) && tid == 0) {
    out_row[out_cols - 1] =
        __float2half(__half2float(out_row[out_cols - 1]) * gate);
  }
}

}  // namespace
}  // namespace vllm::awq_sm70

namespace vllm::awq_sm70 {

torch::Tensor interleave_gated_silu_cols(torch::Tensor tensor) {
  const int64_t n = tensor.size(-1);
  TORCH_CHECK((n % 2) == 0,
              "awq_sm70_prepare: gated_silu interleave requires even columns.");
  const int64_t half = n / 2;
  auto first = tensor.slice(-1, 0, half);
  auto second = tensor.slice(-1, half, n);
  return torch::stack({first, second}, -1).reshape(tensor.sizes());
}

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor weight) {
  validate_f16_weight(weight, "sm70_f16_prepare");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto entry = get_sm70_f16_cached_weight(weight, stream);

  auto meta = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, entry.k_ld);
  return {entry.tm_weight, meta};
}

void sm70_f16_gemm_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, int64_t k_ld, bool gated_silu) {
  validate_f16_input(in_feats, tm_weight, out, gated_silu, "sm70_f16_gemm");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(0);

  TORCH_CHECK(tm_weight.size(1) == k, "sm70_f16_gemm: weight shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "sm70_f16_gemm: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_gemm: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "sm70_f16_gemm: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "sm70_f16_gemm: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n, "sm70_f16_gemm: output cols must match n.");
  }

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_gemm: no compatible TurboMind converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);
  turbomind::gemm::MatrixLayout desc_U{};
  turbomind::gemm::MatrixLayout desc_V{};
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,    turbomind::gemm::kRowMajor,      static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_dense_dispatch_policy(device, static_cast<int>(m),
                                             static_cast<int>(n),
                                             static_cast<int>(k), 0, stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kNone, 0};
  op.batch_dim = 0;

  auto& workspace_holder = workspace_for(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f, in_feats.data_ptr(), desc_A, nullptr, desc_U,
                          tm_weight.data_ptr(), desc_B, nullptr, desc_V, 0.f,
                          out.data_ptr(), desc_D, out.data_ptr(), desc_D,
                          workspace_holder, stream);
  TORCH_CHECK(ec == 0, "sm70_f16_gemm: TurboMind GEMM failed.");
}

void sm70_f16_gate_mul_out(torch::Tensor out, torch::Tensor in_feats,
                           torch::Tensor gate_weight) {
  validate_f16_gate_mul_input(out, in_feats, gate_weight);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = out.size(1);
  if (m == 0 || n == 0) {
    return;
  }

  constexpr int kThreads = 256;
  sm70_f16_gate_mul_kernel<kThreads>
      <<<static_cast<unsigned int>(m), kThreads, 0, stream>>>(
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(in_feats.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(gate_weight.data_ptr<at::Half>()),
          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
          out.stride(0), in_feats.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor sm70_f16_gemm(torch::Tensor in_feats, torch::Tensor weight) {
  validate_f16_weight(weight, "sm70_f16_gemm");
  TORCH_CHECK(in_feats.is_cuda(), "sm70_f16_gemm: input must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_gemm: input must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, "sm70_f16_gemm: input must be 2D.");
  TORCH_CHECK(in_feats.size(1) == weight.size(1),
              "sm70_f16_gemm: input/weight K mismatch.");
  if (in_feats.size(0) > sm70_f16_dense_max_m()) {
    return torch::mm(in_feats, weight.transpose(0, 1));
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto entry = get_sm70_f16_cached_weight(weight, stream);
  const int64_t n = weight.size(0);
  auto out = torch::empty(
      {in_feats.size(0), n},
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device()));
  sm70_f16_gemm_out(out, in_feats, entry.tm_weight, entry.k_ld, false);
  return out;
}

}  // namespace vllm::awq_sm70

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor _kernel) {
  return vllm::awq_sm70::sm70_f16_prepare(_kernel);
}

torch::Tensor sm70_f16_gemm(torch::Tensor _in_feats, torch::Tensor _kernel) {
  return vllm::awq_sm70::sm70_f16_gemm(_in_feats, _kernel);
}

void sm70_f16_gemm_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, int64_t k_ld, bool gated_silu) {
  vllm::awq_sm70::sm70_f16_gemm_out(out, _in_feats, _kernel, k_ld, gated_silu);
}

void sm70_f16_gate_mul_out(torch::Tensor out, torch::Tensor _in_feats,
                           torch::Tensor _gate_weight) {
  vllm::awq_sm70::sm70_f16_gate_mul_out(out, _in_feats, _gate_weight);
}

// ---------------------------------------------------------------------------
// Op registration (K03). Schemas are the fork's minus `str? native_policy`.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork csrc/torch_bindings.cpp:661
  m.def(
      "sm70_f16_prepare(Tensor _kernel) -> Tensor[]");
  // fork csrc/torch_bindings.cpp:682
  m.def(
      "sm70_f16_gemm(Tensor _in_feats, Tensor _kernel) -> Tensor");
  // fork csrc/torch_bindings.cpp:1005
  m.def(
      "sm70_f16_gemm_out(Tensor(a!) out, Tensor _in_feats, Tensor "
      "_kernel, int k_ld, bool gated_silu) -> ()");
  // fork csrc/torch_bindings.cpp:1139
  m.def(
      "sm70_f16_gate_mul_out(Tensor(a!) out, Tensor _in_feats, Tensor "
      "_gate_weight) -> ()");
  // fork csrc/torch_bindings.cpp:596
  m.def(
      "silu_and_mul_interleaved(Tensor! result, Tensor input) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("sm70_f16_prepare", &sm70_f16_prepare);
  m.impl("sm70_f16_gemm", &sm70_f16_gemm);
  m.impl("sm70_f16_gemm_out", &sm70_f16_gemm_out);
  m.impl("sm70_f16_gate_mul_out", &sm70_f16_gate_mul_out);
  m.impl("silu_and_mul_interleaved", &vllm_sm70::silu_and_mul_interleaved);
}
