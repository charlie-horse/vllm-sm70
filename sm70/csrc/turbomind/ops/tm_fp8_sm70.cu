// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboMind FP8 (E4M3, block-128) dense GEMM: prepare, dequant, GEMM, prefill dispatch (K03).
// Moved verbatim from fork cf9492a csrc/sm70_turbomind/ops/awq_sm70_gemm.cu;
// see ../PROVENANCE.md for the fork line range of every function.
#include "tm_ops_common_sm70.cuh"

namespace vllm::awq_sm70 {
namespace {

void validate_fp8_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales, int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda(), "fp8_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "fp8_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_sm70_prepare: qweight must be float8_e4m3fn.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "fp8_sm70_prepare: scales must be float32.");
  TORCH_CHECK(qweight.dim() == 2, "fp8_sm70_prepare: qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, "fp8_sm70_prepare: scales must be 2D.");
  TORCH_CHECK(group_size == 128,
              "fp8_sm70_prepare: only group_size=128 is supported.");
}

}  // namespace
}  // namespace vllm::awq_sm70

namespace vllm::awq_sm70 {

constexpr int kFp8PrefillDequantThreads = 256;

constexpr int kFp8QuantValuesPerWord = 8;

constexpr int kFp8OutputColumnsPerTile = 32;

__global__ void fp8_sm70_dequantize_kernel(
    __half* __restrict__ output, const uint8_t* __restrict__ packed_weight,
    const __half* __restrict__ packed_scales, int k, int n, int group_size) {
  const int64_t word_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t word_count =
      static_cast<int64_t>(k) * n / kFp8QuantValuesPerWord;
  if (word_index >= word_count) {
    return;
  }

  const int words_per_n_tile = k / kFp8QuantValuesPerWord;
  const int n_in_tile = static_cast<int>(word_index % kFp8OutputColumnsPerTile);
  const int64_t tile_word = word_index / kFp8OutputColumnsPerTile;
  const int k_word = static_cast<int>(tile_word % words_per_n_tile);
  const int n_tile = static_cast<int>(tile_word / words_per_n_tile);
  const int col = n_tile * kFp8OutputColumnsPerTile + n_in_tile;
  const int k_base = k_word * kFp8QuantValuesPerWord;
  const int group = k_base / group_size;

  const uint2 quant = reinterpret_cast<const uint2*>(packed_weight)[word_index];
  const auto& quant_lo =
      reinterpret_cast<const turbomind::Array<turbomind::fp8_e4m3_t, 4>&>(
          quant.x);
  const auto& quant_hi =
      reinterpret_cast<const turbomind::Array<turbomind::fp8_e4m3_t, 4>&>(
          quant.y);
  const auto half_lo = turbomind::cvt_f16x4_e4m3(quant_lo);
  const auto half_hi = turbomind::cvt_f16x4_e4m3(quant_hi);
  const __half scale = packed_scales[static_cast<int64_t>(group) * n + col];

  // cvt_f16x4_e4m3 reverses the converter's physical [0, 2, 1, 3]
  // byte order and returns each logical K4 in-order.
  output[static_cast<int64_t>(k_base + 0) * n + col] =
      __hmul(half_lo[0], scale);
  output[static_cast<int64_t>(k_base + 1) * n + col] =
      __hmul(half_lo[1], scale);
  output[static_cast<int64_t>(k_base + 2) * n + col] =
      __hmul(half_lo[2], scale);
  output[static_cast<int64_t>(k_base + 3) * n + col] =
      __hmul(half_lo[3], scale);
  output[static_cast<int64_t>(k_base + 4) * n + col] =
      __hmul(half_hi[0], scale);
  output[static_cast<int64_t>(k_base + 5) * n + col] =
      __hmul(half_hi[1], scale);
  output[static_cast<int64_t>(k_base + 6) * n + col] =
      __hmul(half_hi[2], scale);
  output[static_cast<int64_t>(k_base + 7) * n + col] =
      __hmul(half_hi[3], scale);
}

void fp8_sm70_dequantize_out(torch::Tensor output, torch::Tensor packed_weight,
                             torch::Tensor packed_scales, int64_t group_size) {
  TORCH_CHECK(
      output.is_cuda() && packed_weight.is_cuda() && packed_scales.is_cuda(),
      "SM70 FP8 prefill dequant expects CUDA tensors.");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 &&
                  packed_weight.scalar_type() == torch::kUInt8 &&
                  packed_scales.scalar_type() == torch::kFloat16,
              "SM70 FP8 prefill dequant dtype mismatch.");
  TORCH_CHECK(
      output.dim() == 2 && packed_weight.dim() == 2 && packed_scales.dim() == 2,
      "SM70 FP8 prefill dequant expects rank-2 tensors.");
  TORCH_CHECK(output.is_contiguous() && packed_weight.is_contiguous() &&
                  packed_scales.is_contiguous(),
              "SM70 FP8 prefill dequant expects contiguous tensors.");
  TORCH_CHECK(group_size == 128,
              "SM70 FP8 prefill dequant requires group_size=128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const int64_t k = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(k % group_size == 0 && k % kFp8QuantValuesPerWord == 0 &&
                  n % kFp8OutputColumnsPerTile == 0,
              "SM70 FP8 prefill dequant shape alignment mismatch.");
  TORCH_CHECK(packed_weight.numel() == k * n,
              "SM70 FP8 prefill packed-weight shape mismatch.");
  TORCH_CHECK(
      packed_scales.size(0) == k / group_size && packed_scales.size(1) == n,
      "SM70 FP8 prefill packed-scale shape mismatch.");
  TORCH_CHECK(output.device() == packed_weight.device() &&
                  output.device() == packed_scales.device(),
              "SM70 FP8 prefill tensors must share a device.");

  const int64_t words = packed_weight.numel() / kFp8QuantValuesPerWord;
  const int blocks = static_cast<int>((words + kFp8PrefillDequantThreads - 1) /
                                      kFp8PrefillDequantThreads);
  fp8_sm70_dequantize_kernel<<<blocks, kFp8PrefillDequantThreads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      packed_weight.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(packed_scales.data_ptr<at::Half>()),
      static_cast<int>(k), static_cast<int>(n), static_cast<int>(group_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  validate_fp8_inputs(qweight, scales, group_size);

  qweight = qweight.contiguous();
  scales = scales.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "fp8_sm70_prepare: K and N must be multiples of 8.");
  const bool channelwise_scales = scales.size(0) == n && scales.size(1) == 1;
  const bool blockwise_scales =
      scales.size(0) == (n + group_size - 1) / group_size &&
      scales.size(1) == (k + group_size - 1) / group_size;
  TORCH_CHECK(channelwise_scales || blockwise_scales,
              "fp8_sm70_prepare: expected [N, 1] channel scales or "
              "[ceil(N/128), ceil(K/128)] block scales.");

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto qweight_kn = qweight.transpose(0, 1).contiguous();
  auto i16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, i16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight_kn.data_ptr()), tmp_u16.numel(),
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
  k_desc.type = turbomind::data_type_v<uint8_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              "fp8_sm70_prepare: weight conversion failed.");

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;

  torch::Tensor group_scales;
  if (channelwise_scales) {
    group_scales = scales.transpose(0, 1)
                       .contiguous()
                       .to(torch::kFloat16)
                       .repeat({num_groups, 1})
                       .contiguous();
  } else {
    group_scales = scales.transpose(0, 1)
                       .contiguous()
                       .to(torch::kFloat16)
                       .repeat_interleave(group_size, 1)
                       .slice(1, 0, n)
                       .contiguous();
  }
  if (interleave_gated_silu) {
    group_scales = interleave_gated_silu_cols(group_scales);
  }

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

  auto tm_scales = torch::empty_like(group_scales);
  TORCH_CHECK(conv_s->Convert(group_scales.data_ptr(), s_desc,
                              tm_scales.data_ptr(), q_desc, stream) == 0,
              "fp8_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

void fp8_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, torch::Tensor tm_scales,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu, bool exact_8k_prefill_prescaled = false,
                       bool preserve_default_partition = false) {
  TORCH_CHECK(in_feats.is_cuda(), "fp8_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "fp8_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "fp8_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "fp8_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kUInt8,
              "fp8_gemm_sm70: weight must be uint8.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: output must be float16.");
  TORCH_CHECK(group_size == 128,
              "fp8_gemm_sm70: only group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1);
  TORCH_CHECK(tm_weight.size(0) == k, "fp8_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(tm_scales.size(0) == (k + group_size - 1) / group_size,
              "fp8_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "fp8_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "fp8_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "fp8_gemm_sm70: output must be row-major contiguous.");
  if (exact_8k_prefill_prescaled) {
    const bool qwen38_prefill =
        m == 8000 && k == 5120 && (n == 4096 || n == 3584);
    const bool prescaled_m1 =
        m == 1 && ((n == 1536 && k == 4096) || (n == 8192 && k == 1024) ||
                   (n == 4096 && k == 2048) || (n == 1024 && k == 4096) ||
                   (n == 4096 && k == 512));
    const bool prescaled_batch =
        m > 32 && m <= 64 &&
        ((n == 5120 && k == 1536) || ((n == 4096 || n == 3584) && k == 5120));
    TORCH_CHECK(qwen38_prefill || prescaled_m1 || prescaled_batch,
                "fp8_gemm_sm70: pre-scaled block-FP8 requires an accepted "
                "8K prefill, M=1, or M=33..64 tensor shape.");
    TORCH_CHECK(!gated_silu,
                "fp8_gemm_sm70: pre-scaled path does not fuse gated SILU.");
  }
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "fp8_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "fp8_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "fp8_gemm_sm70: output cols must match weight output dim.");
  }

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_gemm_sm70: no compatible TurboMind converters.");

  auto desc_B = vllm::sm70::packed_weight_layout(
      *conv_w, turbomind::kFloat8_e4m3, n, k, k_ld);

  auto desc_V = vllm::sm70::packed_scale_layout(
      *conv_s, turbomind::kUint16, n, (k + group_size - 1) / group_size, q_ld);

  turbomind::gemm::Operation op{};
  op.dispatch = select_fp8_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  if (exact_8k_prefill_prescaled) {
    op.dispatch =
        op.dispatch | turbomind::gemm::DispatchPolicy::kSm70Fp8PrefillPrescaled;
  }
  if (preserve_default_partition) {
    op.dispatch = op.dispatch |
                  turbomind::gemm::DispatchPolicy::kPreserveDefaultPartition;
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
  TORCH_CHECK(ec == 0, "fp8_gemm_sm70: TurboMind GEMM failed.");
}

bool sm70_fp8_prefill_cutlass_gated_silu_enabled(
    const torch::Tensor& in_feats, const torch::Tensor& dense_weight) {
  return (vllm::sm70::policy_atoi(vllm::sm70::PolicyField::fp8_prefill_cutlass,
                                  1) != 0) &&
         in_feats.size(0) >= 8000 && in_feats.size(0) <= 8192 &&
         in_feats.size(1) > 0 && in_feats.size(1) % 128 == 0 &&
         dense_weight.size(1) > 0 && dense_weight.size(1) % 128 == 0;
}

void fp8_gemm_sm70_prefill_dispatch_out(
    torch::Tensor out, int64_t dense_weight_ptr, torch::Tensor in_feats,
    torch::Tensor tm_weight, torch::Tensor tm_scales, int64_t group_size,
    int64_t k_ld, int64_t q_ld, bool gated_silu, int64_t min_prefill_m) {
  if (in_feats.size(0) < min_prefill_m) {
    fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, k_ld,
                      q_ld, gated_silu);
    return;
  }

  TORCH_CHECK(in_feats.dim() == 2 && dense_weight_ptr != 0,
              "SM70 FP8 prefill dispatch expects rank-2 input and workspace.");
  auto dense_weight = torch::from_blob(
      reinterpret_cast<void*>(dense_weight_ptr),
      {in_feats.size(1), tm_weight.size(1)}, in_feats.options());
  fp8_sm70_dequantize_out(dense_weight, tm_weight, tm_scales, group_size);
  if (gated_silu &&
      sm70_fp8_prefill_cutlass_gated_silu_enabled(in_feats, dense_weight)) {
    auto gate_up =
        at::empty({in_feats.size(0), dense_weight.size(1)}, in_feats.options());
    if (sm70_fp8_prefill_cutlass_out(gate_up, in_feats, dense_weight, false)) {
      sm70_silu_and_mul_interleaved_fp16_out(out, gate_up);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  if (sm70_fp8_prefill_cutlass_out(out, in_feats, dense_weight, gated_silu)) {
    return;
  }
  if (gated_silu) {
    auto gate_up = at::mm(in_feats, dense_weight);
    sm70_silu_and_mul_interleaved_fp16_out(out, gate_up);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  at::mm_out(out, in_feats, dense_weight);
}

void fp8_gemm_sm70_out_auto(torch::Tensor out, torch::Tensor in_feats,
                            torch::Tensor tm_weight, torch::Tensor tm_scales) {
  constexpr int64_t group_size = 128;
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1);

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_gemm_sm70: no compatible TurboMind converters.");

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
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,  order_s,
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }

  fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, desc_B.ld,
                    desc_V.ld, false);
}

void fp8_gemm_sm70_out_meta(torch::Tensor out, torch::Tensor in_feats,
                            torch::Tensor tm_weight, torch::Tensor tm_scales,
                            torch::Tensor meta, bool gated_silu) {
  TORCH_CHECK(meta.scalar_type() == torch::kInt64,
              "fp8_gemm_sm70: meta must be int64.");
  TORCH_CHECK(meta.numel() >= 2, "fp8_gemm_sm70: meta must have two values.");
  auto meta_cpu = meta.device().is_cpu() ? meta.contiguous()
                                         : meta.to(torch::kCPU).contiguous();
  const int64_t* meta_ptr = meta_cpu.data_ptr<int64_t>();
  fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, 128, meta_ptr[0],
                    meta_ptr[1], gated_silu);
}

}  // namespace vllm::awq_sm70

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  return vllm::awq_sm70::fp8_sm70_prepare(_kernel, _scaling_factors, group_size,
                                          interleave_gated_silu);
}

void fp8_sm70_dequantize_out(torch::Tensor out, torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             int64_t group_size) {
  vllm::awq_sm70::fp8_sm70_dequantize_out(out, _kernel, _scaling_factors,
                                          group_size);
}

void fp8_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, torch::Tensor _scaling_factors,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu, bool preserve_default_partition) {
  vllm::awq_sm70::fp8_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                    group_size, k_ld, q_ld, gated_silu, false,
                                    preserve_default_partition);
}

void fp8_gemm_sm70_prefill_prescaled_out(torch::Tensor out,
                                         torch::Tensor _in_feats,
                                         torch::Tensor _kernel,
                                         torch::Tensor _prescaled_factors,
                                         int64_t group_size, int64_t k_ld,
                                         int64_t q_ld) {
  vllm::awq_sm70::fp8_gemm_sm70_out(out, _in_feats, _kernel, _prescaled_factors,
                                    group_size, k_ld, q_ld, false, true);
}

void fp8_gemm_sm70_prescaled_m1_out(torch::Tensor out, torch::Tensor _in_feats,
                                    torch::Tensor _kernel,
                                    torch::Tensor _prescaled_factors,
                                    int64_t group_size, int64_t k_ld,
                                    int64_t q_ld) {
  TORCH_CHECK(_in_feats.dim() == 2 && _in_feats.size(0) == 1,
              "fp8_gemm_sm70_prescaled_m1_out requires a rank-2 M=1 input.");
  vllm::awq_sm70::fp8_gemm_sm70_out(out, _in_feats, _kernel, _prescaled_factors,
                                    group_size, k_ld, q_ld, false, true);
}

void fp8_gemm_sm70_prefill_dispatch_out(
    torch::Tensor out, int64_t dense_weight_ptr, torch::Tensor _in_feats,
    torch::Tensor _kernel, torch::Tensor _scaling_factors, int64_t group_size,
    int64_t k_ld, int64_t q_ld, bool gated_silu, int64_t min_prefill_m) {
  vllm::awq_sm70::fp8_gemm_sm70_prefill_dispatch_out(
      out, dense_weight_ptr, _in_feats, _kernel, _scaling_factors, group_size,
      k_ld, q_ld, gated_silu, min_prefill_m);
}

void fp8_gemm_sm70_out_auto(torch::Tensor out, torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors) {
  vllm::awq_sm70::fp8_gemm_sm70_out_auto(out, _in_feats, _kernel,
                                         _scaling_factors);
}

void fp8_gemm_sm70_out_meta(torch::Tensor out, torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors, torch::Tensor _meta,
                            bool gated_silu) {
  vllm::awq_sm70::fp8_gemm_sm70_out_meta(out, _in_feats, _kernel,
                                         _scaling_factors, _meta, gated_silu);
}
