// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Top-k candidate sampler op registration (patch K16).
//
// Adapted from the fork's sidecar fragment
// csrc/sm70_turbomind/ops/sm70_top20_philox_fragment.cpp (cf9492a): the
// declarations at :34-41 and the defs at :82-92. The fork registered them in
// `_C` from a separate `_sm70_sampler_C` module; here they are two more ops of
// `_sm70_C`. The schemas are the fork's, unchanged. The other defs of that
// fragment (sorted/chunked top20 Philox samplers, sampler graph branches) are
// not carried (see PROVENANCE.md).
//
// The kernels allocate only their outputs and one per-call scratch tensor
// (through the caching allocator, so graph capture takes them from the graph
// pool). The optional overflow counter is created by the caller once, before
// any capture (vllm_sm70/sample.py:prepare, R4); the ops never create state.

#include <torch/all.h>
#include <torch/library.h>

#include <optional>

torch::Tensor sm70_local_top32_half(torch::Tensor logits, int64_t vocab_start,
                                    int64_t num_valid);

torch::Tensor sm70_sample_topk_candidates(
    torch::Tensor wire, torch::Tensor temperature, torch::Tensor top_k,
    const std::optional<torch::Tensor>& top_p,
    const std::optional<at::Generator>& generator, int64_t vocab_size,
    int64_t tp_size, const std::optional<torch::Tensor>& overflow);

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def(
      "sm70_local_top32_half(Tensor logits, int vocab_start, int num_valid) "
      "-> Tensor");
  m.def(
      "sm70_sample_topk_candidates(Tensor wire, Tensor temperature, "
      "Tensor top_k, Tensor? top_p, Generator? generator, int vocab_size, "
      "int tp_size, Tensor(a!)? overflow) -> Tensor");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("sm70_local_top32_half", &sm70_local_top32_half);
  m.impl("sm70_sample_topk_candidates", &sm70_sample_topk_candidates);
}
