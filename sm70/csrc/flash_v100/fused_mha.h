#ifndef FUSED_MHA_H
#define FUSED_MHA_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window);

// flash_attention_decode_paged plus optional learned attention sinks: fp32
// [H_q], natural-log units, an extra softmax denominator term with no value
// row. std::nullopt is the sink-free path above, bit for bit.
at::Tensor flash_attention_decode_paged_sinks(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window, const std::optional<at::Tensor>& sinks);

// flash_attention_decode_paged_sinks with int32 arrival counters (>= B * H_kv
// entries, zero between launches): sink layers on the grouped GQA-8 D128
// kernel then decode in one launch per layer instead of partition + reduce.
// Other shapes ignore the counters and take the two-launch path.
at::Tensor flash_attention_decode_paged_sinks_fused(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window, const std::optional<at::Tensor>& sinks,
    const at::Tensor& fused_counters);

at::Tensor flash_attention_decode_paged_xqa(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, const int batch_context_max_seq_len);

at::Tensor flash_attention_prefill_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, const float softmax_scale,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const bool is_causal, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window);

at::Tensor flash_attention_prefill_paged_sinks(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, const float softmax_scale,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const bool is_causal, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window, const at::Tensor& sinks);

#endif
