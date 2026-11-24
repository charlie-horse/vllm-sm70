# Flint top-k candidate sampler: source list for _sm70_C.
#
# sm70/CMakeLists.txt does `include(csrc/sampler/sources.cmake)` when this
# file exists. Like the other areas it appends to SM70_CXX_SRCS /
# SM70_CUDA_SRCS only when CUDA and sm_70 were found (the CPU-only binding has
# no sampler ops). Torch and the CUDA runtime only (cub and curand are
# header-only parts of the toolkit). No CUTLASS.
#
# nvcc flags: none beyond the sm_70 gencode. No --use_fast_math: the kernel's
# expf() and the Philox/exponential transform must round as the reference
# sampler's torch kernels do. torch's __CUDA_NO_HALF*__ defines stay.

set(VLLM_SM70_SAMPLER_CUDA_SRCS
  "${CMAKE_CURRENT_LIST_DIR}/topk_candidates_sampler.cu")
set(VLLM_SM70_SAMPLER_CXX_SRCS
  "${CMAKE_CURRENT_LIST_DIR}/sampler_ops.cpp")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_SAMPLER_CUDA_SRCS})
  list(APPEND SM70_CXX_SRCS ${VLLM_SM70_SAMPLER_CXX_SRCS})
  message(STATUS "vllm_sm70: Flint top-k candidate sampler (K16)")
endif()
