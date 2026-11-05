// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboMind NVFP4 (W4A16, group 16) dense GEMM: prepare and GEMM (K03).
// Moved verbatim from fork cf9492a csrc/sm70_turbomind/ops/awq_sm70_gemm.cu;
// see ../PROVENANCE.md for the fork line range of every function.
#include "tm_ops_common_sm70.cuh"

#include <torch/library.h>

namespace vllm::awq_sm70 {
namespace {

void validate_nvfp4_inputs(const torch::Tensor& qweight,
                           const torch::Tensor& scales, int64_t group_size,
                           const char* op_name) {
  TORCH_CHECK(qweight.is_cuda(), op_name, ": qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), op_name, ": scales must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, op_name,
              ": qweight must be unpacked uint8.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, op_name,
              ": scales must be float16.");
  TORCH_CHECK(qweight.dim() == 2, op_name, ": qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, op_name, ": scales must be 2D.");
  TORCH_CHECK(group_size == 16, op_name, ": only group_size=16 is supported.");
}

}  // namespace
}  // namespace vllm::awq_sm70

namespace vllm::awq_sm70 {

std::vector<torch::Tensor> nvfp4_sm70_prepare(torch::Tensor qweight,
                                              torch::Tensor scales,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  constexpr const char* op_name = "nvfp4_sm70_prepare";
  validate_nvfp4_inputs(qweight, scales, group_size, op_name);

  qweight = qweight.contiguous();
  scales = scales.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  const int64_t num_groups = scales.size(0);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, op_name,
              ": K and N must be multiples of 8.");
  TORCH_CHECK(k % group_size == 0, op_name,
              ": K must be divisible by group size.");
  TORCH_CHECK(k / group_size == num_groups, op_name,
              ": scales group mismatch.");
  TORCH_CHECK(scales.size(1) == n, op_name, ": scales shape mismatch.");

  const auto fp4_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto fp8_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = fp4_converters[0];
  const auto* conv_s = fp8_converters[1];
  TORCH_CHECK(conv_w && conv_s, op_name,
              ": no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto u16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight.data_ptr()), tmp_u16.numel(),
      stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

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

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kFloat4_e2m1;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n / 8},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              op_name, ": weight conversion failed.");

  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
  }

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty_like(scales);
  TORCH_CHECK(conv_s->Convert(scales.data_ptr(), s_desc, tm_scales.data_ptr(),
                              q_desc, stream) == 0,
              op_name, ": scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                         torch::Tensor tm_weight, torch::Tensor tm_scales,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu, bool prescaled = false) {
  TORCH_CHECK(in_feats.is_cuda(), "nvfp4_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "nvfp4_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "nvfp4_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "nvfp4_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "nvfp4_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "nvfp4_gemm_sm70: weight must be int32.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kFloat16,
              "nvfp4_gemm_sm70: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "nvfp4_gemm_sm70: output must be float16.");
  TORCH_CHECK(group_size == 16,
              "nvfp4_gemm_sm70: only group_size=16 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k,
              "nvfp4_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "nvfp4_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "nvfp4_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "nvfp4_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "nvfp4_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "nvfp4_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "nvfp4_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "nvfp4_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "nvfp4_gemm_sm70: output cols must match weight output dim.");
  }

  const auto fp4_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat4_e2m1, turbomind::kHalf, true, 70);
  const auto fp8_converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = fp4_converters[0];
  const auto* conv_s = fp8_converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "nvfp4_gemm_sm70: no compatible TurboMind converters.");

  auto desc_B = vllm::sm70::packed_weight_layout(
      *conv_w, turbomind::kFloat4_e2m1, n, k, k_ld);

  auto desc_V = vllm::sm70::packed_scale_layout(*conv_s, turbomind::kUint16, n,
                                                k / group_size, q_ld);

  turbomind::gemm::Operation op{};
  op.dispatch = select_nvfp4_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  if (prescaled) {
    TORCH_CHECK(m > 32, "nvfp4_gemm_sm70: prescaled input requires M>32.");
    op.dispatch =
        op.dispatch | turbomind::gemm::DispatchPolicy::kSm70Nvfp4Prescaled;
  }
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = workspace_for(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = vllm::sm70::run_dense_packed_gemm(
      gemm, workspace_holder, op, stream, in_feats, out, tm_weight.data_ptr(),
      desc_B, tm_scales.data_ptr(), desc_V, n, in_feats.stride(0));
  TORCH_CHECK(ec == 0, "nvfp4_gemm_sm70: TurboMind GEMM failed.");
}

}  // namespace vllm::awq_sm70

std::vector<torch::Tensor> nvfp4_sm70_prepare(torch::Tensor _kernel,
                                              torch::Tensor _scaling_factors,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  return vllm::awq_sm70::nvfp4_sm70_prepare(_kernel, _scaling_factors,
                                            group_size, interleave_gated_silu);
}

void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                         torch::Tensor _kernel, torch::Tensor _scaling_factors,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu) {
  vllm::awq_sm70::nvfp4_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                      group_size, k_ld, q_ld, gated_silu);
}

void nvfp4_gemm_sm70_prescaled_out(torch::Tensor out, torch::Tensor input,
                                   torch::Tensor weight, torch::Tensor scales,
                                   int64_t group_size, int64_t k_ld,
                                   int64_t q_ld, bool gated_silu) {
  vllm::awq_sm70::nvfp4_gemm_sm70_out(out, input, weight, scales, group_size,
                                      k_ld, q_ld, gated_silu, true);
}

// ---------------------------------------------------------------------------
// Op registration (K03). Schemas are the fork's minus `str? native_policy`.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork csrc/torch_bindings.cpp:653
  m.def(
      "nvfp4_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, int "
      "group_size=16, bool interleave_gated_silu=False) -> Tensor[]");
  // fork csrc/torch_bindings.cpp:947
  m.def(
      "nvfp4_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor "
      "_kernel, Tensor _scaling_factors, int group_size, int k_ld, int "
      "q_ld, bool gated_silu=False) -> ()");
  // fork csrc/torch_bindings.cpp:963
  m.def(
      "nvfp4_gemm_sm70_prescaled_out(Tensor(a!) out, Tensor input, Tensor"
      " weight, Tensor scales, int group_size, int k_ld, int q_ld, bool "
      "gated_silu) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("nvfp4_sm70_prepare", &nvfp4_sm70_prepare);
  m.impl("nvfp4_gemm_sm70_out", &nvfp4_gemm_sm70_out);
  m.impl("nvfp4_gemm_sm70_prescaled_out", &nvfp4_gemm_sm70_prescaled_out);
}
