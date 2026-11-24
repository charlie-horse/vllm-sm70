// SPDX-License-Identifier: Apache-2.0
//
// Flash-V100 op registration in the `_sm70_C` namespace, prefixed `fv100_`.
// `out` is a required first argument (`Tensor(a!)`) and the ops return ();
// the workspaces the kernels write (tmp_out, max_logits, exp_sums) are
// mutable too. The anchored-SWA kernels are not compiled, so the kernel
// entry points get std::nullopt / 0 for them.
//
// `fv100_decode_paged_sinks_fused` (under VLLM_SM70_FV100_FUSED_DECODE) is
// fv100_decode_paged_sinks with required sinks plus int32 arrival counters
// (>= B * H_kv entries, zero between launches; the kernel's last CTA
// re-zeroes its counter). Shapes the fused kernel does not take ignore the
// counters and decode in two launches. `fv100_decode_verify` is in
// verify_decode.cu.

// KV cache views: k_cache and v_cache are [num_blocks, block_size, H_kv, D]
// views with arbitrary strides in the first three dims and a contiguous last
// dim. Upstream v0.33.0 (LBNHC) interleaves K and V per head, so a view has
// stride(2) == 2 * D; the kernels take explicit strides and accept that.

#include <torch/library.h>

#include <optional>
#include <string>

#include "fused_mha.h"

namespace vllm_sm70::fv100 {
namespace {

void check_out(const at::Tensor& out, const at::Tensor& q, const char* op) {
  TORCH_CHECK(out.is_cuda() && out.device() == q.device(), op,
              ": out must be on the q device");
  TORCH_CHECK(out.scalar_type() == at::kHalf, op, ": out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), op, ": out must have q's shape");
}

}  // namespace

// Calls flash_attention_decode_paged (flash_decode_paged.cu)
void decode_paged(at::Tensor out, const at::Tensor& q,
                  const at::Tensor& k_cache, const at::Tensor& v_cache,
                  const at::Tensor& block_table, const at::Tensor& seq_lens,
                  at::Tensor tmp_out, at::Tensor max_logits,
                  at::Tensor exp_sums,
                  const at::Tensor& active_num_partitions,
                  double softmax_scale, int64_t partition_size,
                  int64_t launch_num_partitions,
                  const std::string& kv_cache_dtype, double k_scale,
                  double v_scale, int64_t window_size_left,
                  int64_t window_size_right) {
  check_out(out, q, "fv100_decode_paged");
  std::optional<at::Tensor> out_opt(out);
  flash_attention_decode_paged(
      q, k_cache, v_cache, out_opt, block_table, seq_lens, tmp_out,
      max_logits, exp_sums, active_num_partitions,
      static_cast<float>(softmax_scale), static_cast<int>(partition_size),
      static_cast<int>(launch_num_partitions), kv_cache_dtype,
      static_cast<float>(k_scale), static_cast<float>(v_scale),
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      std::nullopt, 0);
}

// Calls flash_attention_decode_paged_sinks (flash_decode_paged.cu)
void decode_paged_sinks(at::Tensor out, const at::Tensor& q,
                        const at::Tensor& k_cache, const at::Tensor& v_cache,
                        const at::Tensor& block_table,
                        const at::Tensor& seq_lens, at::Tensor tmp_out,
                        at::Tensor max_logits, at::Tensor exp_sums,
                        const at::Tensor& active_num_partitions,
                        double softmax_scale, int64_t partition_size,
                        int64_t launch_num_partitions,
                        const std::string& kv_cache_dtype, double k_scale,
                        double v_scale, int64_t window_size_left,
                        int64_t window_size_right,
                        const std::optional<at::Tensor>& sinks) {
  check_out(out, q, "fv100_decode_paged_sinks");
  std::optional<at::Tensor> out_opt(out);
  flash_attention_decode_paged_sinks(
      q, k_cache, v_cache, out_opt, block_table, seq_lens, tmp_out,
      max_logits, exp_sums, active_num_partitions,
      static_cast<float>(softmax_scale), static_cast<int>(partition_size),
      static_cast<int>(launch_num_partitions), kv_cache_dtype,
      static_cast<float>(k_scale), static_cast<float>(v_scale),
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      std::nullopt, 0, sinks);
}

#ifdef VLLM_SM70_FV100_FUSED_DECODE
// Calls flash_attention_decode_paged_sinks_fused (flash_decode_paged.cu)
void decode_paged_sinks_fused(
    at::Tensor out, const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& v_cache, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor tmp_out, at::Tensor max_logits,
    at::Tensor exp_sums, const at::Tensor& active_num_partitions,
    double softmax_scale, int64_t partition_size,
    int64_t launch_num_partitions, const std::string& kv_cache_dtype,
    double k_scale, double v_scale, int64_t window_size_left,
    int64_t window_size_right, const at::Tensor& sinks, at::Tensor counters) {
  check_out(out, q, "fv100_decode_paged_sinks_fused");
  std::optional<at::Tensor> out_opt(out);
  const std::optional<at::Tensor> sinks_opt(sinks);
  flash_attention_decode_paged_sinks_fused(
      q, k_cache, v_cache, out_opt, block_table, seq_lens, tmp_out,
      max_logits, exp_sums, active_num_partitions,
      static_cast<float>(softmax_scale), static_cast<int>(partition_size),
      static_cast<int>(launch_num_partitions), kv_cache_dtype,
      static_cast<float>(k_scale), static_cast<float>(v_scale),
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      std::nullopt, 0, sinks_opt, counters);
}
#endif  // VLLM_SM70_FV100_FUSED_DECODE

// Calls flash_attention_decode_paged_xqa (flash_decode_paged.cu)
void decode_paged_xqa(at::Tensor out, const at::Tensor& q,
                      const at::Tensor& k_cache, const at::Tensor& v_cache,
                      const at::Tensor& block_table, const at::Tensor& seq_lens,
                      at::Tensor tmp_out, at::Tensor max_logits,
                      at::Tensor exp_sums,
                      const at::Tensor& active_num_partitions,
                      double softmax_scale, int64_t partition_size,
                      int64_t launch_num_partitions,
                      const std::string& kv_cache_dtype, double k_scale,
                      double v_scale, int64_t window_size_left,
                      int64_t window_size_right,
                      int64_t batch_context_max_seq_len) {
  check_out(out, q, "fv100_decode_paged_xqa");
  std::optional<at::Tensor> out_opt(out);
  flash_attention_decode_paged_xqa(
      q, k_cache, v_cache, out_opt, block_table, seq_lens, tmp_out,
      max_logits, exp_sums, active_num_partitions,
      static_cast<float>(softmax_scale), static_cast<int>(partition_size),
      static_cast<int>(launch_num_partitions), kv_cache_dtype,
      static_cast<float>(k_scale), static_cast<float>(v_scale),
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      static_cast<int>(batch_context_max_seq_len));
}

// Calls flash_attention_prefill_paged (fused_mha_forward_paged.cu).
// q and out are [B, H, M, D] and contiguous (the kernel indexes them densely).
void prefill_paged(at::Tensor out, const at::Tensor& q,
                   const at::Tensor& k_cache, const at::Tensor& v_cache,
                   const at::Tensor& block_table, const at::Tensor& seq_lens,
                   double softmax_scale, const std::string& kv_cache_dtype,
                   double k_scale, double v_scale, bool is_causal,
                   int64_t window_size_left, int64_t window_size_right) {
  check_out(out, q, "fv100_prefill_paged");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(),
              "fv100_prefill_paged: q and out must be contiguous [B, H, M, D]");
  std::optional<at::Tensor> out_opt(out);
  flash_attention_prefill_paged(
      q, k_cache, v_cache, out_opt, block_table, seq_lens,
      static_cast<float>(softmax_scale), kv_cache_dtype,
      static_cast<float>(k_scale), static_cast<float>(v_scale), is_causal,
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      std::nullopt, 0);
}

// Calls flash_attention_prefill_paged_sinks (fused_mha_forward_paged.cu)
void prefill_paged_sinks(at::Tensor out, const at::Tensor& q,
                         const at::Tensor& k_cache, const at::Tensor& v_cache,
                         const at::Tensor& block_table,
                         const at::Tensor& seq_lens, double softmax_scale,
                         const std::string& kv_cache_dtype, double k_scale,
                         double v_scale, bool is_causal,
                         int64_t window_size_left, int64_t window_size_right,
                         const at::Tensor& sinks) {
  check_out(out, q, "fv100_prefill_paged_sinks");
  TORCH_CHECK(
      q.is_contiguous() && out.is_contiguous(),
      "fv100_prefill_paged_sinks: q and out must be contiguous [B, H, M, D]");
  std::optional<at::Tensor> out_opt(out);
  flash_attention_prefill_paged_sinks(
      q, k_cache, v_cache, out_opt, block_table, seq_lens,
      static_cast<float>(softmax_scale), kv_cache_dtype,
      static_cast<float>(k_scale), static_cast<float>(v_scale), is_causal,
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      std::nullopt, 0, sinks);
}

// Defined in verify_decode.cu.
void decode_verify(at::Tensor out, const at::Tensor& q,
                   const at::Tensor& k_cache, const at::Tensor& v_cache,
                   const at::Tensor& block_table, const at::Tensor& seq_lens,
                   at::Tensor tmp_out, at::Tensor max_logits,
                   at::Tensor exp_sums, double softmax_scale,
                   int64_t query_len, int64_t launch_num_partitions);

}  // namespace vllm_sm70::fv100

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def(
      "fv100_decode_paged(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor "
      "v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) tmp_out, "
      "Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right) -> ()");
  m.def(
      "fv100_decode_paged_sinks(Tensor(a!) out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) "
      "tmp_out, Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right, Tensor? sinks) "
      "-> ()");
  m.def(
      "fv100_decode_paged_xqa(Tensor(a!) out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) "
      "tmp_out, Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right, int "
      "batch_context_max_seq_len) -> ()");
  m.def(
      "fv100_prefill_paged(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor "
      "v_cache, Tensor block_table, Tensor seq_lens, float softmax_scale, str "
      "kv_cache_dtype, float k_scale, float v_scale, bool is_causal, int "
      "window_size_left, int window_size_right) -> ()");
  m.def(
      "fv100_prefill_paged_sinks(Tensor(a!) out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor block_table, Tensor seq_lens, float "
      "softmax_scale, str kv_cache_dtype, float k_scale, float v_scale, bool "
      "is_causal, int window_size_left, int window_size_right, Tensor sinks) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("fv100_decode_paged", &vllm_sm70::fv100::decode_paged);
  m.impl("fv100_decode_paged_sinks", &vllm_sm70::fv100::decode_paged_sinks);
  m.impl("fv100_decode_paged_xqa", &vllm_sm70::fv100::decode_paged_xqa);
  m.impl("fv100_prefill_paged", &vllm_sm70::fv100::prefill_paged);
  m.impl("fv100_prefill_paged_sinks", &vllm_sm70::fv100::prefill_paged_sinks);
}

// DFlash2 verify attention, KV read once per verify step.
TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def(
      "fv100_decode_verify(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor "
      "v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) tmp_out, "
      "Tensor(c!) max_logits, Tensor(d!) exp_sums, float softmax_scale, int "
      "query_len, int launch_num_partitions) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("fv100_decode_verify", &vllm_sm70::fv100::decode_verify);
}

#ifdef VLLM_SM70_FV100_FUSED_DECODE
// One-launch grouped GQA-8 D128 sink decode.
TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def(
      "fv100_decode_paged_sinks_fused(Tensor(a!) out, Tensor q, Tensor "
      "k_cache, Tensor v_cache, Tensor block_table, Tensor seq_lens, "
      "Tensor(b!) tmp_out, Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right, Tensor sinks, "
      "Tensor(e!) counters) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("fv100_decode_paged_sinks_fused",
         &vllm_sm70::fv100::decode_paged_sinks_fused);
}
#endif  // VLLM_SM70_FV100_FUSED_DECODE
