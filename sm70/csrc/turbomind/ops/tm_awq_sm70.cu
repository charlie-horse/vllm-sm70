// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboMind AWQ/int4 dense GEMM (K04). Compiled only with VLLM_SM70_TM_AWQ.
// Moved verbatim from fork cf9492a csrc/sm70_turbomind/ops/awq_sm70_gemm.cu;
// see ../PROVENANCE.md for the fork line range of every function.
#include "tm_ops_common_sm70.cuh"

#ifdef VLLM_SM70_TM_AWQ

namespace turbomind {

void unpack_awq_gemm(uint4_t* dst, const uint4_t* src, int rows, int cols,
                     cudaStream_t st);

}  // namespace turbomind

namespace vllm::awq_sm70 {
namespace {

void validate_awq_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales,
                         const torch::Tensor& qzeros) {
  TORCH_CHECK(qweight.is_cuda(), "awq_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "awq_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qzeros.is_cuda(), "awq_sm70_prepare: qzeros must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qweight must be int32.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qzeros must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "awq_sm70_prepare: scales must be float16.");
}

void validate_uint4_inputs(const torch::Tensor& qweight,
                           const torch::Tensor& scales,
                           const torch::Tensor& qzeros, const char* op_name) {
  TORCH_CHECK(qweight.is_cuda(), op_name, ": qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), op_name, ": scales must be CUDA.");
  TORCH_CHECK(qzeros.is_cuda(), op_name, ": qzeros must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, op_name,
              ": qweight must be unpacked uint8.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kFloat16, op_name,
              ": qzeros must be float16.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, op_name,
              ": scales must be float16.");
  TORCH_CHECK(qweight.dim() == 2, op_name, ": qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, op_name, ": scales must be 2D.");
  TORCH_CHECK(qzeros.dim() == 2, op_name, ": qzeros must be 2D.");
}

}  // namespace
}  // namespace vllm::awq_sm70

namespace vllm::awq_sm70 {

constexpr int kAwqPrefillDequantThreads = 256;

constexpr int kAwqQuantValuesPerWord = 8;

constexpr int kAwqOutputColumnsPerTile = 32;

__device__ __forceinline__ int awq_prefill_physical_nibble(int logical_k) {
  // TurboMind's SM70 converter stores each logical K8 fragment as
  // [0, 2, 4, 6, 1, 3, 5, 7].
  return ((logical_k & 1) << 2) + (logical_k >> 1);
}

__global__ void awq_sm70_dequantize_kernel(
    __half* __restrict__ output, const int32_t* __restrict__ packed_weight,
    const int32_t* __restrict__ packed_scales, int k, int n, int group_size) {
  const int64_t word_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t word_count = static_cast<int64_t>(k) * n / 8;
  if (word_index >= word_count) {
    return;
  }

  const int words_per_n_tile = k / kAwqQuantValuesPerWord;
  const int n_in_tile = static_cast<int>(word_index % kAwqOutputColumnsPerTile);
  const int64_t tile_word = word_index / kAwqOutputColumnsPerTile;
  const int k_word = static_cast<int>(tile_word % words_per_n_tile);
  const int n_tile = static_cast<int>(tile_word / words_per_n_tile);
  const int col = n_tile * kAwqOutputColumnsPerTile + n_in_tile;
  const int k_base = k_word * kAwqQuantValuesPerWord;
  const int group = k_base / group_size;

  const uint32_t quant = static_cast<uint32_t>(packed_weight[word_index]);
  const uint32_t stats_bits =
      static_cast<uint32_t>(packed_scales[group * n + col]);
  const __half2 stats = *reinterpret_cast<const __half2*>(&stats_bits);
  const __half scale = __low2half(stats);
  const __half bias = __high2half(stats);

#pragma unroll
  for (int logical_k = 0; logical_k < kAwqQuantValuesPerWord; ++logical_k) {
    const int shift = awq_prefill_physical_nibble(logical_k) * 4;
    const int value = static_cast<int>((quant >> shift) & 0xfu);
    output[static_cast<int64_t>(k_base + logical_k) * n + col] =
        __hfma(__int2half_rn(value), scale, bias);
  }
}

void awq_sm70_dequantize_out(torch::Tensor output, torch::Tensor packed_weight,
                             torch::Tensor packed_scales, int64_t group_size) {
  TORCH_CHECK(
      output.is_cuda() && packed_weight.is_cuda() && packed_scales.is_cuda(),
      "SM70 AWQ prefill dequant expects CUDA tensors.");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 &&
                  packed_weight.scalar_type() == torch::kInt32 &&
                  packed_scales.scalar_type() == torch::kInt32,
              "SM70 AWQ prefill dequant dtype mismatch.");
  TORCH_CHECK(
      output.dim() == 2 && packed_weight.dim() == 2 && packed_scales.dim() == 2,
      "SM70 AWQ prefill dequant expects rank-2 tensors.");
  TORCH_CHECK(output.is_contiguous() && packed_weight.is_contiguous() &&
                  packed_scales.is_contiguous(),
              "SM70 AWQ prefill dequant expects contiguous tensors.");
  TORCH_CHECK(group_size == 128,
              "SM70 AWQ prefill dequant requires group_size=128.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const int64_t k = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(k % group_size == 0 && k % kAwqQuantValuesPerWord == 0 &&
                  n % kAwqOutputColumnsPerTile == 0,
              "SM70 AWQ prefill dequant shape alignment mismatch.");
  TORCH_CHECK(packed_weight.numel() == k * n / 8,
              "SM70 AWQ prefill packed-weight shape mismatch.");
  TORCH_CHECK(
      packed_scales.size(0) == k / group_size && packed_scales.size(1) == n,
      "SM70 AWQ prefill packed-scale shape mismatch.");
  TORCH_CHECK(output.device() == packed_weight.device() &&
                  output.device() == packed_scales.device(),
              "SM70 AWQ prefill tensors must share a device.");

  const int64_t words = packed_weight.numel();
  const int blocks = static_cast<int>((words + kAwqPrefillDequantThreads - 1) /
                                      kAwqPrefillDequantThreads);
  awq_sm70_dequantize_kernel<<<blocks, kAwqPrefillDequantThreads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      packed_weight.data_ptr<int32_t>(), packed_scales.data_ptr<int32_t>(),
      static_cast<int>(k), static_cast<int>(n), static_cast<int>(group_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void pack_awq_compact_source_kernel(uint32_t* packed,
                                               const half* scales,
                                               const uint8_t* zeros,
                                               int64_t numel) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    const uint16_t scale_bits = __half_as_ushort(scales[idx]);
    packed[idx] = static_cast<uint32_t>(scale_bits) |
                  (static_cast<uint32_t>(zeros[idx]) << 16);
  }
}

__global__ void compact_awq_stats_kernel(uint8_t* compact,
                                         const uint32_t* packed,
                                         int64_t numel) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    const uint32_t value = packed[idx];
    compact[idx * 3] = static_cast<uint8_t>(value);
    compact[idx * 3 + 1] = static_cast<uint8_t>(value >> 8);
    compact[idx * 3 + 2] = static_cast<uint8_t>(value >> 16);
  }
}

std::vector<torch::Tensor> awq_sm70_prepare_impl(
    torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros,
    int64_t group_size, bool interleave_gated_silu, bool compact_metadata) {
  validate_awq_inputs(qweight, scales, qzeros);

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  qzeros = qzeros.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1) * 8;
  const int64_t num_groups = scales.size(0);

  TORCH_CHECK(scales.size(1) == n, "awq_sm70_prepare: scales shape mismatch.");
  TORCH_CHECK(qzeros.size(0) == num_groups,
              "awq_sm70_prepare: qzeros group mismatch.");
  TORCH_CHECK(qzeros.size(1) * 8 == n,
              "awq_sm70_prepare: qzeros shape mismatch.");
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "awq_sm70_prepare: K and N must be multiples of 8.");
  TORCH_CHECK(k % num_groups == 0,
              "awq_sm70_prepare: input dim must be divisible by groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }
  TORCH_CHECK(k / num_groups == group_size,
              "awq_sm70_prepare: group_size mismatch with scales.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_sm70_prepare: SM70 AWQ supports group_size=32/64/128, got ",
              group_size, ".");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w = turbomind::gemm::get_operand_tag(conv_w->pack) ==
                      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto packed_weight = torch::empty_like(qweight);
  turbomind::unpack_awq_gemm(
      reinterpret_cast<turbomind::uint4_t*>(packed_weight.data_ptr<int>()),
      reinterpret_cast<const turbomind::uint4_t*>(qweight.data_ptr<int>()),
      static_cast<int>(k), static_cast<int>(n), stream);

  auto u16_opts =
      torch::TensorOptions().device(qweight.device()).dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const turbomind::uint4_t*>(
          packed_weight.data_ptr<int>()),
      tmp_u16.numel(), stream);
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
  k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(qweight);
  TORCH_CHECK(conv_w->Convert(tmp_u16_conv.data_ptr(), w_desc,
                              tm_weight.data_ptr(), k_desc, stream) == 0,
              "awq_sm70_prepare: weight conversion failed.");

  // Unpack AWQ zeros using PyTorch tensor ops (matches lmdeploy's Python
  // approach).  The C++ unpack_awq_gemm() requires rows%8==0 which fails
  // when num_groups < 8 (e.g. Qwen3-30B-A3B w2: K=768, num_groups=6).
  const int awq_order[] = {0, 4, 1, 5, 2, 6, 3, 7};
  std::vector<torch::Tensor> zslices;
  auto zz = qzeros;
  for (int i = 0; i < 8; ++i) {
    zslices.push_back((zz & 0xF).to(torch::kUInt8));
    zz = zz.__rshift__(4);
  }
  std::vector<torch::Tensor> zordered;
  for (int i = 0; i < 8; ++i) {
    zordered.push_back(zslices[awq_order[i]]);
  }
  auto zeros_u8 = torch::stack(zordered, -1).reshape({num_groups, n});
  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
    zeros_u8 = interleave_gated_silu_cols(zeros_u8);
  }

  torch::Tensor source_stats;
  if (compact_metadata) {
    source_stats = torch::empty(
        {num_groups, n},
        torch::TensorOptions().device(scales.device()).dtype(torch::kInt32));
    constexpr int kThreads = 256;
    const int blocks =
        static_cast<int>((scales.numel() + kThreads - 1) / kThreads);
    pack_awq_compact_source_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<uint32_t*>(source_stats.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        zeros_u8.data_ptr<uint8_t>(), scales.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto zeros_half = zeros_u8.to(torch::kFloat16);
    source_stats = torch::empty(
        {num_groups, n * 2},
        torch::TensorOptions().device(scales.device()).dtype(torch::kFloat16));
    turbomind::fuse_scales_and_zeros(
        reinterpret_cast<half*>(source_stats.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(zeros_half.data_ptr<at::Half>()),
        scales.numel(), stream);
  }

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,  order_s,
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

  auto tm_stats = torch::empty(
      {num_groups, n},
      torch::TensorOptions().device(scales.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_s->Convert(source_stats.data_ptr(), s_desc,
                              tm_stats.data_ptr(), q_desc, stream) == 0,
              "awq_sm70_prepare: scale conversion failed.");

  torch::Tensor tm_scales = tm_stats;
  if (compact_metadata) {
    tm_scales = torch::empty(
        {num_groups, n, 3},
        torch::TensorOptions().device(scales.device()).dtype(torch::kUInt8));
    constexpr int kThreads = 256;
    const int blocks =
        static_cast<int>((tm_stats.numel() + kThreads - 1) / kThreads);
    compact_awq_stats_kernel<<<blocks, kThreads, 0, stream>>>(
        tm_scales.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(tm_stats.data_ptr<int32_t>()),
        tm_stats.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  // q_ld is an internal contract between prepare, pointer construction and
  // the SM70 iterator.  Its sign distinguishes compact 3-byte statistics;
  // the magnitude remains the logical TurboMind leading dimension.
  meta.index_put_({1}, compact_metadata ? -q_desc.ld : q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  return awq_sm70_prepare_impl(qweight, scales, qzeros, group_size,
                               interleave_gated_silu, false);
}

std::vector<torch::Tensor> awq_sm70_prepare_compact(
    torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros,
    int64_t group_size, bool interleave_gated_silu) {
  return awq_sm70_prepare_impl(qweight, scales, qzeros, group_size,
                               interleave_gated_silu, true);
}

std::vector<torch::Tensor> uint4_sm70_prepare(torch::Tensor qweight,
                                              torch::Tensor scales,
                                              torch::Tensor zeros,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  constexpr const char* op_name = "uint4_sm70_prepare";
  validate_uint4_inputs(qweight, scales, zeros, op_name);

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  zeros = zeros.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1);
  const int64_t num_groups = scales.size(0);

  TORCH_CHECK(scales.size(1) == n, op_name, ": scales shape mismatch.");
  TORCH_CHECK(zeros.size(0) == num_groups, op_name, ": zeros group mismatch.");
  TORCH_CHECK(zeros.size(1) == n, op_name, ": zeros shape mismatch.");
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, op_name,
              ": K and N must be multiples of 8.");
  TORCH_CHECK(num_groups > 0 && k % num_groups == 0, op_name,
              ": input dim must be divisible by groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }
  TORCH_CHECK(k / num_groups == group_size, op_name,
              ": group_size mismatch with scales.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              op_name, ": SM70 uint4 supports group_size=32/64/128, got ",
              group_size, ".");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
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
  k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
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
    zeros = interleave_gated_silu_cols(zeros);
  }

  auto fused = torch::empty(
      {num_groups, n * 2},
      torch::TensorOptions().device(scales.device()).dtype(torch::kFloat16));
  turbomind::fuse_scales_and_zeros(
      reinterpret_cast<half*>(fused.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(zeros.data_ptr<at::Half>()), scales.numel(),
      stream);

  const auto order_s = conv_s->order;
  const bool is_A_s = turbomind::gemm::get_operand_tag(conv_s->pack) ==
                      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,  order_s,
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

  auto tm_scales = torch::empty(
      {num_groups, n},
      torch::TensorOptions().device(scales.device()).dtype(torch::kInt32));
  TORCH_CHECK(conv_s->Convert(fused.data_ptr(), s_desc, tm_scales.data_ptr(),
                              q_desc, stream) == 0,
              op_name, ": scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, torch::Tensor tm_scales,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu,
                       const turbomind::gemm::TileAllReduceParam* tile_reduce) {
  const bool compact_metadata = q_ld < 0;
  TORCH_CHECK(in_feats.is_cuda(), "awq_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "awq_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "awq_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "awq_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: weight must be int32.");
  TORCH_CHECK(
      (!compact_metadata && tm_scales.scalar_type() == torch::kInt32) ||
          (compact_metadata && tm_scales.scalar_type() == torch::kUInt8),
      "awq_gemm_sm70: scales must be int32, or uint8 with compact q_ld.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: output must be float16.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k, "awq_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "awq_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "awq_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n, "awq_gemm_sm70: scale shape mismatch.");
  if (compact_metadata) {
    TORCH_CHECK(tm_scales.dim() == 3 && tm_scales.size(2) == 3 &&
                    tm_scales.is_contiguous(),
                "awq_gemm_sm70: compact scales must be contiguous [G,N,3].");
  } else {
    TORCH_CHECK(tm_scales.dim() == 2,
                "awq_gemm_sm70: regular scales must be rank 2.");
  }
  TORCH_CHECK(out.size(0) == m,
              "awq_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "awq_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "awq_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "awq_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n, "awq_gemm_sm70: output cols must match n.");
  }

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_gemm_sm70: no compatible TurboMind converters.");

  auto desc_B =
      vllm::sm70::packed_weight_layout(*conv_w, turbomind::kUint4, n, k, k_ld);

  auto desc_V = vllm::sm70::packed_scale_layout(
      *conv_s, turbomind::kUint32, n, k / group_size,
      compact_metadata ? -q_ld : q_ld);

  turbomind::gemm::Operation op{};
  op.dispatch = select_awq_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  if (tile_reduce != nullptr) {
    TORCH_CHECK(!gated_silu,
                "awq_gemm_sm70: tile all-reduce cannot combine gated_silu.");
    op.epilogue = static_cast<turbomind::gemm::Epilogue>(
        static_cast<int>(op.epilogue) |
        static_cast<int>(turbomind::gemm::Epilogue::kTileAllReduce));
    op.tile_allreduce =
        const_cast<turbomind::gemm::TileAllReduceParam*>(tile_reduce);
  }
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = workspace_for(device, stream);
  auto& gemm = get_gemm(device);

  void* stats_ptr = tm_scales.data_ptr();
  if (compact_metadata) {
    stats_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(stats_ptr) |
                                        uintptr_t{1});
  }
  const int ec = vllm::sm70::run_dense_packed_gemm(
      gemm, workspace_holder, op, stream, in_feats, out, tm_weight.data_ptr(),
      desc_B, stats_ptr, desc_V, n, k);
  TORCH_CHECK(ec == 0, "awq_gemm_sm70: TurboMind GEMM failed.");
}

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor in_feats,
                       torch::Tensor tm_weight, torch::Tensor tm_scales,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu) {
  awq_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, k_ld, q_ld,
                    gated_silu, nullptr);
}

torch::Tensor awq_gemm_sm70(torch::Tensor in_feats, torch::Tensor tm_weight,
                            torch::Tensor tm_scales, int64_t group_size,
                            int64_t k_ld, int64_t q_ld) {
  const int64_t n = tm_weight.size(1) * 8;
  auto out = torch::empty(
      {in_feats.size(0), n},
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device()));
  awq_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, k_ld, q_ld,
                    false);
  return out;
}

}  // namespace vllm::awq_sm70

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            torch::Tensor _zeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  return vllm::awq_sm70::awq_sm70_prepare(_kernel, _scaling_factors, _zeros,
                                          group_size, interleave_gated_silu);
}

std::vector<torch::Tensor> awq_sm70_prepare_compact(
    torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros,
    int64_t group_size, bool interleave_gated_silu) {
  return vllm::awq_sm70::awq_sm70_prepare_compact(
      _kernel, _scaling_factors, _zeros, group_size, interleave_gated_silu);
}

void awq_sm70_dequantize_out(torch::Tensor out, torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             int64_t group_size) {
  vllm::awq_sm70::awq_sm70_dequantize_out(out, _kernel, _scaling_factors,
                                          group_size);
}

std::vector<torch::Tensor> uint4_sm70_prepare(torch::Tensor _kernel,
                                              torch::Tensor _scaling_factors,
                                              torch::Tensor _zeros,
                                              int64_t group_size,
                                              bool interleave_gated_silu) {
  return vllm::awq_sm70::uint4_sm70_prepare(_kernel, _scaling_factors, _zeros,
                                            group_size, interleave_gated_silu);
}

torch::Tensor awq_gemm_sm70(torch::Tensor _in_feats, torch::Tensor _kernel,
                            torch::Tensor _scaling_factors, int64_t group_size,
                            int64_t k_ld, int64_t q_ld) {
  return vllm::awq_sm70::awq_gemm_sm70(_in_feats, _kernel, _scaling_factors,
                                       group_size, k_ld, q_ld);
}

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, torch::Tensor _scaling_factors,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu) {
  vllm::awq_sm70::awq_gemm_sm70_out(out, _in_feats, _kernel, _scaling_factors,
                                    group_size, k_ld, q_ld, gated_silu);
}

#endif  // VLLM_SM70_TM_AWQ
