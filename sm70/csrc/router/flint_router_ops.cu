// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// _sm70_C::flint_route (patch K13): the fused SM70 decode router for the
// E256/K6 sigmoid MoE with an FP32 gate [256, 2048] and FP16 activations,
// one or two tokens. Kernel: flint_router.cuh (verbatim from the fork).
//
// Ported from fork cf9492a csrc/sm70_turbomind/ops/sm70_router.cu:
//   kept:    flint_route (:72-123), its schema (:131-133), check_sm70 (:27-31)
//   dropped: top10, select_quantize and their helpers (GGUF E512 router,
//            not used by Flint or Qwen), the separate TORCH_LIBRARY
//            (vllm_sm70_router) and the pybind module.
// The op is registered as a fragment of _sm70_C. The workspace (logits) and
// the arrival counter are caller-owned and allocated once per layer before
// CUDA graph capture (vllm_sm70/router.py); this op allocates nothing.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "flint_router.cuh"

namespace vllm_sm70::router {
namespace {

void check_sm70() {
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 router requires V100");
}

}  // namespace

void flint_route(torch::Tensor weights, torch::Tensor ids, torch::Tensor x,
                 torch::Tensor gate, torch::Tensor logits,
                 torch::Tensor counter) {
  namespace fr = vllm::sm70_flint_router;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(0) >= 1 && x.size(0) <= fr::kMaxTokens &&
                  x.size(1) == fr::kHidden && x.stride(1) == 1 &&
                  x.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
              "Expected FP16 router input [M1..2,2048] with 8-byte aligned "
              "unit-stride rows");
  const c10::cuda::CUDAGuard guard(x.device());
  check_sm70();
  const int64_t m = x.size(0);
  TORCH_CHECK(
      gate.device() == x.device() && gate.scalar_type() == at::kFloat &&
          gate.is_contiguous() &&
          gate.sizes() == at::IntArrayRef({fr::kExperts, fr::kHidden}) &&
          reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0,
      "Expected contiguous 16-byte aligned FP32 gate [256,2048]");
  TORCH_CHECK(weights.device() == x.device() &&
                  weights.scalar_type() == at::kFloat &&
                  weights.is_contiguous() &&
                  weights.sizes() == at::IntArrayRef({m, fr::kTopK}),
              "Expected contiguous FP32 top-k weights [M,6]");
  TORCH_CHECK(ids.device() == x.device() && ids.scalar_type() == at::kInt &&
                  ids.is_contiguous() &&
                  ids.sizes() == at::IntArrayRef({m, fr::kTopK}),
              "Expected contiguous INT32 top-k ids [M,6]");
  TORCH_CHECK(logits.device() == x.device() &&
                  logits.scalar_type() == at::kFloat &&
                  logits.is_contiguous() &&
                  logits.numel() >= fr::kMaxTokens * fr::kExperts,
              "Expected an FP32 logits workspace of at least 512 elements");
  TORCH_CHECK(counter.device() == x.device() &&
                  counter.scalar_type() == at::kInt &&
                  counter.is_contiguous() && counter.numel() >= 1,
              "Expected an INT32 arrival counter");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto launch = [&](auto kernel) {
    kernel<<<fr::kExperts, fr::kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()), x.stride(0),
        gate.data_ptr<float>(), logits.data_ptr<float>(),
        counter.data_ptr<int>(), weights.data_ptr<float>(),
        ids.data_ptr<int>());
  };
  if (m == 1)
    launch(fr::route_kernel<1>);
  else
    launch(fr::route_kernel<2>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm_sm70::router

TORCH_LIBRARY_FRAGMENT(_sm70_C, m) {
  m.def(
      "flint_route(Tensor(a!) weights, Tensor(b!) ids, Tensor x, Tensor gate, "
      "Tensor(c!) logits, Tensor(d!) counter) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m) {
  m.impl("flint_route", &vllm_sm70::router::flint_route);
}
