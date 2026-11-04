// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TurboMind SM70 tuning knobs for the vllm-sm70 plugin.
//
// Replaces the fork's per-engine "native policy" ABI (fork cf9492a
// csrc/sm70_policy.h + sm70_policy_fields.inc + sm70_runtime_state.h). Every
// knob is a compile-time default that one cached getenv() pass can override,
// read once per process. There are no policy tokens, no per-call scopes and
// no engine-owned runtime objects. The names below keep the fork's
// vllm::sm70::{PolicyField, policy_atoi, policy_value, ...} spelling, so the
// vendored lmdeploy files (gemm.cu, dispatch_cache.cu, sm70_dflash_context.h)
// stay verbatim. They include this through the sm70_policy.h shim.
//
// Environment names are the fork's, so existing tuning notes still apply.
// Only knobs read by the K02 file set are listed; K03+ add their own.
#pragma once

// Same std headers the fork's sm70_policy.h chain provided, so vendored
// includers that relied on them transitively keep compiling.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm::sm70 {

// X(field, env name). Defaults live at each call site (the fork's fallback
// argument), so this table only maps fields to environment variables.
#define VLLM_SM70_TM_KNOBS(X)                                                \
  X(tm_gemm_trace, "TM_GEMM_TRACE")                                          \
  X(tm_gemm_trace_filter, "TM_GEMM_TRACE_FILTER")                            \
  X(tm_gemm_trace_limit, "TM_GEMM_TRACE_LIMIT")                              \
  X(tm_gemm_tune, "TM_GEMM_TUNE")                                            \
  X(tm_gemm_warn_cache_miss, "TM_GEMM_WARN_CACHE_MISS")                      \
  X(tm_gemm_cache_summary, "TM_GEMM_CACHE_SUMMARY")                          \
  X(awq_tp2_fast_selector, "VLLM_SM70_AWQ_TP2_FAST_SELECTOR")                \
  X(awq_tp2_fast_targets, "VLLM_SM70_AWQ_TP2_FAST_TARGETS")                  \
  X(awq_tp4_qkv_cta64, "VLLM_SM70_AWQ_TP4_QKV_CTA64")                        \
  X(awq_mtp_m5_fast_selector, "VLLM_SM70_AWQ_MTP_M5_FAST_SELECTOR")          \
  X(mxfp4_moe_grouped_m8, "VLLM_SM70_MXFP4_MOE_GROUPED_M8")                  \
  X(mxfp4_moe_grouped_m8_fast_selector,                                      \
    "VLLM_SM70_MXFP4_MOE_GROUPED_M8_FAST_SELECTOR")                          \
  X(nvfp4_qwen38_tp4_m1_fast_selector,                                       \
    "VLLM_SM70_NVFP4_QWEN38_TP4_M1_FAST_SELECTOR")                           \
  X(nvfp4_qwen38_moe_fast_prefill, "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL") \
  X(fp8_prefill_fast_selector, "VLLM_SM70_FP8_PREFILL_FAST_SELECTOR")        \
  X(fp8_grouped_bmm_decode, "VLLM_SM70_FP8_GROUPED_BMM_DECODE")              \
  X(dflash2_qpn8_rerank, "VLLM_SM70_DFLASH2_QPN8_RERANK")                    \
  X(dflash2_qpn8_rerank_shadow, "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW")      \
  X(dflash_sharded_context_fc, "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC")       \
  X(awq_tune_small_shapes, "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES")                \
  X(fp8_tune_small_shapes, "VLLM_SM70_FP8_TUNE_SMALL_SHAPES")                \
  X(nvfp4_tune_small_shapes, "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES")            \
  X(nvfp4_moe_grouped_prefill, "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL")        \
  X(nvfp4_moe_grouped_expert_rows, "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS") \
  X(fp8_moe_single_token_per_expert_dispatch,                                \
    "VLLM_SM70_FP8_MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH")                    \
  X(fp8_0dot3_dense_selector, "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR")          \
  X(fp8_safe_fast_selector, "VLLM_SM70_FP8_SAFE_FAST_SELECTOR")              \
  X(awq_reuse_imported_cache, "VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE")          \
  X(fp8_reuse_imported_cache, "VLLM_SM70_FP8_REUSE_IMPORTED_CACHE")          \
  X(awq_preserve_default_splits, "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS")    \
  X(awq_preserve_default_splits_only,                                        \
    "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY")                            \
  X(fp8_preserve_default_splits, "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS")    \
  X(fp8_preserve_default_splits_only,                                        \
    "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY")                            \
  X(awq_moe_dispatch_policy, "VLLM_SM70_AWQ_MOE_DISPATCH_POLICY")            \
  X(awq_dense_tune_max_m, "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M")                  \
  X(f16_dense_tune_max_m, "VLLM_SM70_F16_DENSE_TUNE_MAX_M")                  \
  X(fp8_dense_tune_max_m, "VLLM_SM70_FP8_DENSE_TUNE_MAX_M")                  \
  X(nvfp4_dense_tune_max_m, "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M")              \
  X(awq_moe_tune_max_tokens, "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS")            \
  X(nvfp4_moe_tune_max_tokens, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS")        \
  X(f16_dense_max_m, "VLLM_SM70_F16_DENSE_MAX_M")

enum class PolicyField {
#define X(field, env) field,
  VLLM_SM70_TM_KNOBS(X)
#undef X
      count
};

inline constexpr size_t policy_size = static_cast<size_t>(PolicyField::count);
inline constexpr const char* policy_names[] = {
#define X(field, env) env,
    VLLM_SM70_TM_KNOBS(X)
#undef X
};

inline const char* policy_name(PolicyField field) {
  return policy_names[static_cast<size_t>(field)];
}

// The one getenv pass. It runs on first use and is thread-safe (magic static).
// An unset variable stays nullptr, and the call site's default applies.
// The fork re-read a few knobs on every call ("dynamic_legacy"). Here every
// knob is fixed at first use: set the environment before the first GEMM.
struct Knobs {
  std::array<bool, policy_size> present{};
  std::array<std::string, policy_size> raw{};
};
inline const Knobs& knobs() {
  static const Knobs k = [] {
    Knobs r;
    for (size_t i = 0; i < policy_size; ++i) {
      if (const char* v = std::getenv(policy_names[i])) {
        r.present[i] = true;
        r.raw[i] = v;
      }
    }
    return r;
  }();
  return k;
}

inline const char* policy_value(PolicyField field) {
  const auto i = static_cast<size_t>(field);
  return knobs().present[i] ? knobs().raw[i].c_str() : nullptr;
}

// `dynamic_legacy` is accepted for source compatibility and ignored.
inline int policy_atoi(PolicyField field, int fallback = 0,
                       bool /*dynamic_legacy*/ = false) {
  const char* raw = policy_value(field);
  return raw ? std::atoi(raw) : fallback;
}

inline bool policy_exact_one(PolicyField field, bool fallback = false,
                             bool /*dynamic_legacy*/ = false) {
  const char* raw = policy_value(field);
  return raw ? raw[0] == '1' && raw[1] == '\0' : fallback;
}

enum class DispatchOverride { Unset, Default, Reuse, Measure, Invalid };

inline DispatchOverride policy_dispatch_override(PolicyField field) {
  const char* raw = policy_value(field);
  if (!raw || !*raw) return DispatchOverride::Unset;
  const std::string s(raw);
  if (s == "default") return DispatchOverride::Default;
  if (s == "reuse") return DispatchOverride::Reuse;
  if (s == "measure") return DispatchOverride::Measure;
  return DispatchOverride::Invalid;
}

// VLLM_SM70_AWQ_TP2_FAST_TARGETS="<desc>|MxNxK:splits:swizzle:mgroup[@name];..."
// The parser is copied from fork csrc/sm70_policy.h (parse_gemm_targets).
struct GemmTargetPolicy {
  std::string descriptor;
  std::string entry;
  int cta_m{}, cta_n{}, cta_k{}, splits{}, swizzle{}, require_mgroup{};
  std::string name_contains;
  bool valid = false;
};

inline std::vector<GemmTargetPolicy> parse_gemm_targets(
    const std::string& targets) {
  std::vector<GemmTargetPolicy> result;
  size_t begin = 0;
  while (begin <= targets.size()) {
    const size_t end = targets.find(';', begin);
    const auto entry = targets.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    const size_t sep = entry.find('|');
    if (sep != std::string::npos) {
      GemmTargetPolicy target;
      target.descriptor = entry.substr(0, sep);
      target.entry = entry;
      std::string spec = entry.substr(sep + 1);
      if (const size_t name_sep = spec.find('@');
          name_sep != std::string::npos) {
        target.name_contains = spec.substr(name_sep + 1);
        spec.resize(name_sep);
      }
      std::replace(spec.begin(), spec.end(), 'x', ' ');
      std::replace(spec.begin(), spec.end(), ':', ' ');
      std::replace(spec.begin(), spec.end(), ',', ' ');
      std::istringstream input(spec);
      target.valid = static_cast<bool>(input >> target.cta_m >> target.cta_n >>
                                       target.cta_k >> target.splits >>
                                       target.swizzle >> target.require_mgroup);
      if (target.valid && target.name_contains.empty())
        input >> target.name_contains;
      result.emplace_back(std::move(target));
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

inline const std::vector<GemmTargetPolicy>& policy_gemm_targets() {
  static const auto targets = [] {
    const char* raw = policy_value(PolicyField::awq_tp2_fast_targets);
    return parse_gemm_targets(raw ? raw : "");
  }();
  return targets;
}

// Process-wide named counters (the fork kept them per engine runtime). Used
// only to rate-limit TM_GEMM_TRACE output.
inline std::atomic<unsigned>& diagnostic_counter(const std::string& name) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::atomic<unsigned>> counters;
  std::lock_guard<std::mutex> lock(mutex);
  return counters.try_emplace(name, 0u).first->second;
}

}  // namespace vllm::sm70
