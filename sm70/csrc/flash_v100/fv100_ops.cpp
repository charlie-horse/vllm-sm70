// SPDX-License-Identifier: Apache-2.0
//
// Flash-V100 op registration (patch K05), replacing the fork's pybind module
// flash-attention-v100/kernel/fused_mha_api.cpp (cf9492a).
//
// The fork bound ~30 entry points with pybind `m.def` plus `*_configured`
// twins that carried an env-derived PreparedPolicy. The lean subset keeps five
// entry points and registers them in the `_sm70_C` namespace, prefixed
// `fv100_`. Schema differences from the fork functions:
//   * `out` is a required first argument, `Tensor(a!)`; the ops return ().
//     The workspaces the kernels write (tmp_out, max_logits, exp_sums) are
//     marked mutable too.
//   * No anchored-SWA arguments (anchor_lens, anchored_window): the anchored
//     kernels are not compiled; the fork functions are called with
//     std::nullopt / 0.
//   * No policy object: knobs are compile-time (fv100_config.h).
//
// K14 adds a sixth op under `#ifdef VLLM_SM70_FV100_FUSED_DECODE` (on by
// default, see sources.cmake): `fv100_decode_paged_sinks_fused`, the fork's
// decode_paged_sinks_fused_fwd. It is fv100_decode_paged_sinks with required
// sinks plus the int32 arrival counters, `Tensor(e!) counters`, of the
// one-launch grouped GQA-8 D128 sink decode (>= B * H_kv entries, zero
// between launches; the kernel's last CTA re-zeroes its counter). Shapes the
// fused kernel does not take ignore the counters and decode in two launches.
//
// B2 adds `fv100_decode_verify` (verify_decode.cu, new code): DFlash2 verify
// rows (8 query tokens per request, K07-expanded seq_lens and block table)
// in one CTA per (request, kv head, 128-key partition), so each verify step
// reads the KV once, plus its reducer. D256, GQA 6/8, fp16 KV only.
//
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

// fork flash_attention_decode_paged (flash_decode_paged.cu)
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

// fork flash_attention_decode_paged_sinks (flash_decode_paged.cu)
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
// fork flash_attention_decode_paged_sinks_fused (flash_decode_paged.cu)
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

// fork flash_attention_decode_paged_xqa (flash_decode_paged.cu)
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

// fork flash_attention_prefill_paged (fused_mha_forward_paged.cu).
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

// fork flash_attention_prefill_paged_sinks (fused_mha_forward_paged.cu)
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

// B2: defined in verify_decode.cu.
void decode_verify(at::Tensor out, const at::Tensor& q,
                   const at::Tensor& k_cache, const at::Tensor& v_cache,
                   const at::Tensor& block_table, const at::Tensor& seq_lens,
                   at::Tensor tmp_out, at::Tensor max_logits,
                   at::Tensor exp_sums, double softmax_scale,
                   int64_t query_len, int64_t launch_num_partitions);

}  // namespace vllm_sm70::fv100

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  // fork fused_mha_api.cpp:24 decode_paged_fwd
  m.def(
      "fv100_decode_paged(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor "
      "v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) tmp_out, "
      "Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right) -> ()");
  // fork fused_mha_api.cpp:26 decode_paged_sinks_fwd
  m.def(
      "fv100_decode_paged_sinks(Tensor(a!) out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) "
      "tmp_out, Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right, Tensor? sinks) "
      "-> ()");
  // fork fused_mha_api.cpp:33 decode_paged_xqa_fwd
  m.def(
      "fv100_decode_paged_xqa(Tensor(a!) out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor block_table, Tensor seq_lens, Tensor(b!) "
      "tmp_out, Tensor(c!) max_logits, Tensor(d!) exp_sums, Tensor "
      "active_num_partitions, float softmax_scale, int partition_size, int "
      "launch_num_partitions, str kv_cache_dtype, float k_scale, float "
      "v_scale, int window_size_left, int window_size_right, int "
      "batch_context_max_seq_len) -> ()");
  // fork fused_mha_api.cpp:75 prefill_paged_fwd
  m.def(
      "fv100_prefill_paged(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor "
      "v_cache, Tensor block_table, Tensor seq_lens, float softmax_scale, str "
      "kv_cache_dtype, float k_scale, float v_scale, bool is_causal, int "
      "window_size_left, int window_size_right) -> ()");
  // fork fused_mha_api.cpp:77 prefill_paged_sinks_fwd
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

// B2: DFlash2 verify attention, KV read once per verify step.
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
// K14 (C4): fork fused_mha_api.cpp:29 decode_paged_sinks_fused_fwd.
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
