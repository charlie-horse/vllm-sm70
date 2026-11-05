# TurboMind SM70 GEMM core (patch K02): source list for the _sm70_C extension.
#
# sm70/CMakeLists.txt (K01) does `include(csrc/turbomind/sources.cmake)` when
# this file exists. This file appends to K01's lists (SM70_CXX_SRCS,
# SM70_CUDA_SRCS, SM70_INCLUDE_DIRS); K01 builds the .cu files for sm_70 only.
# VLLM_SM70_TM_SRCS / VLLM_SM70_TM_INCLUDE_DIRS stay set for reference.
#
# K03 fix: the K02 version only set VLLM_SM70_TM_*, which K01 never reads,
# so the TurboMind core was not compiled into _sm70_C. Everything is now
# appended only when K01 found CUDA and sm_70 (a CPU-only build has none).
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
# K03 needs CUTLASS (headers only) for qwen38_prefill_cutlass.cu, the
# Qwen3.8 FP8 prefill route. Pass -DVLLM_SM70_CUTLASS_DIR=<cutlass tree>
# (K01 adds its include dirs). The K02 core alone does not need it. Link: CUDA::cudart and CUDA::cublas/cublasLt (headers
# only via cuda_utils.h; no cuBLAS calls in this file set), plus torch.

set(_VLLM_SM70_TM_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(_VLLM_SM70_TM_LM "${_VLLM_SM70_TM_ROOT}/lmdeploy/src/turbomind")

set(VLLM_SM70_TM_INCLUDE_DIRS
  # "src/turbomind/..." includes.
  "${_VLLM_SM70_TM_ROOT}/lmdeploy"
  # sm70_policy.h shim, tm_config.h and the op headers.
  "${_VLLM_SM70_TM_ROOT}/ops")

set(VLLM_SM70_TM_SRCS
  # K02: TurboMind GEMM core.
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
  "${_VLLM_SM70_TM_ROOT}/ops/gemm_runtime.cpp"
  # K03: dense + MoE ops (split from fork awq_sm70_gemm.cu).
  "${_VLLM_SM70_TM_ROOT}/ops/tm_f16_sm70.cu"
  "${_VLLM_SM70_TM_ROOT}/ops/tm_fp8_sm70.cu"
  "${_VLLM_SM70_TM_ROOT}/ops/tm_nvfp4_sm70.cu"
  "${_VLLM_SM70_TM_ROOT}/ops/tm_moe_sm70.cu"
  "${_VLLM_SM70_TM_ROOT}/ops/qwen38_prefill_cutlass.cu"
  # K04: AWQ/int4 dense; an empty TU unless VLLM_SM70_TM_AWQ is on.
  "${_VLLM_SM70_TM_ROOT}/ops/tm_awq_sm70.cu")

set(VLLM_SM70_TM_DEFINITIONS "")
if(VLLM_SM70_TM_AWQ)
  list(APPEND VLLM_SM70_TM_DEFINITIONS "VLLM_SM70_TM_AWQ=1")
endif()

if(SM70_HAS_CUDA AND SM70_ARCHS)
  if(NOT VLLM_SM70_CUTLASS_DIR)
    message(FATAL_ERROR
      "vllm_sm70: TurboMind ops (K03) need CUTLASS headers for "
      "qwen38_prefill_cutlass.cu; set -DVLLM_SM70_CUTLASS_DIR=<cutlass tree>.")
  endif()
  foreach(_src IN LISTS VLLM_SM70_TM_SRCS)
    if(_src MATCHES "\\.cu$")
      list(APPEND SM70_CUDA_SRCS "${_src}")
    else()
      list(APPEND SM70_CXX_SRCS "${_src}")
    endif()
    if(VLLM_SM70_TM_DEFINITIONS)
      # COMPILE_OPTIONS is owned by K01 (gencode); definitions are separate.
      set_property(SOURCE "${_src}" APPEND PROPERTY
        COMPILE_DEFINITIONS ${VLLM_SM70_TM_DEFINITIONS})
    endif()
  endforeach()
  list(APPEND SM70_INCLUDE_DIRS ${VLLM_SM70_TM_INCLUDE_DIRS})
  message(STATUS "vllm_sm70: TurboMind core + ops (K02/K03)"
                 " AWQ=${VLLM_SM70_TM_AWQ}")
endif()

unset(_VLLM_SM70_TM_LM)
unset(_VLLM_SM70_TM_ROOT)
