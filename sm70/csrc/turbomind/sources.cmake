# TurboMind SM70 GEMM core (patch K02): source list for the _sm70_C extension.
#
# sm70/CMakeLists.txt (K01) does `include(csrc/turbomind/sources.cmake)` when
# this file exists, then adds ${VLLM_SM70_TM_SRCS} to the _sm70_C target with
# ${VLLM_SM70_TM_INCLUDE_DIRS} on its include path. It compiles them for
# sm_70 only (-gencode arch=compute_70,code=sm_70).
#
# Mirrors the Tier A subset of fork cf9492a CMakeLists.txt:381-405
# (VLLM_SM70_TURBOMIND_SRCS). The fork's non-GEMM ops (:406-448) belong to
# later patches (K03+, Gus, Rosa) or are dropped.
#
# Optional: AWQ/int4 kernels (patch K04) are compiled only when the consumer
# adds -DVLLM_SM70_TM_AWQ, e.g. by setting VLLM_SM70_TM_AWQ=ON before include.
#
# Requires C++20 / CUDA C++20 (the fork set CMAKE_{CXX,CUDA}_STANDARD 20 at
# CMakeLists.txt:16-19; gemm.cu uses string_view::starts_with). Torch's
# default nvcc flags apply; the fork set no per-file flags for these TUs.
#
# Not needed: CUTLASS. Link: CUDA::cudart and CUDA::cublas/cublasLt (headers
# only via cuda_utils.h; no cuBLAS calls in this file set), plus torch.

set(_VLLM_SM70_TM_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(_VLLM_SM70_TM_LM "${_VLLM_SM70_TM_ROOT}/lmdeploy/src/turbomind")

set(VLLM_SM70_TM_INCLUDE_DIRS
  # "src/turbomind/..." includes.
  "${_VLLM_SM70_TM_ROOT}/lmdeploy"
  # sm70_policy.h shim and tm_config.h.
  "${_VLLM_SM70_TM_ROOT}/ops")

set(VLLM_SM70_TM_SRCS
  "${_VLLM_SM70_TM_LM}/core/check.cc"
  "${_VLLM_SM70_TM_LM}/core/layout.cc"
  "${_VLLM_SM70_TM_LM}/core/context.cc"
  "${_VLLM_SM70_TM_LM}/core/allocator.cc"
  "${_VLLM_SM70_TM_LM}/core/buffer.cc"
  "${_VLLM_SM70_TM_LM}/core/stream.cc"
  "${_VLLM_SM70_TM_LM}/utils/logger.cc"
  "${_VLLM_SM70_TM_LM}/utils/parser.cc"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/gemm.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/kernel.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/dispatch_cache.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/context.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/convert_v3.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/cast.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/unpack.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/tuner/cache_utils.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/tuner/measurer.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/tuner/sampler.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/tuner/stopping_criterion.cc"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/tuner/params.cc"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/kernel/sm70_884_4.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/kernel/sm70_884_8.cu"
  "${_VLLM_SM70_TM_LM}/kernels/gemm/kernel/sm70_884_16.cu"
  "${_VLLM_SM70_TM_ROOT}/ops/tm_registry_sm70.cu"
  "${_VLLM_SM70_TM_ROOT}/ops/gemm_runtime.cpp")

set(VLLM_SM70_TM_DEFINITIONS "")
if(VLLM_SM70_TM_AWQ)
  list(APPEND VLLM_SM70_TM_DEFINITIONS "VLLM_SM70_TM_AWQ=1")
endif()

unset(_VLLM_SM70_TM_LM)
unset(_VLLM_SM70_TM_ROOT)
