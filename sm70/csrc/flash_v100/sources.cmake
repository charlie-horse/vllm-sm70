# Flash-V100 attention, lean subset (patch K05): source list for _sm70_C.
#
# sm70/CMakeLists.txt does `include(csrc/flash_v100/sources.cmake)` when this
# file exists, after the TurboMind include. Like csrc/turbomind/sources.cmake
# it appends to SM70_CXX_SRCS / SM70_CUDA_SRCS / SM70_INCLUDE_DIRS, and only
# when CUDA and sm_70 were found (the CPU-only binding has no CUDA sources).
#
# Mirrors the fork's flash-attention-v100/setup.py CUDAExtension
# "flash_attn_v100_cuda" (cf9492a), reduced to the two lean kernel files; the
# pybind module (fused_mha_api.cpp) is replaced by fv100_ops.cpp, which
# registers TORCH_LIBRARY_FRAGMENT(_sm70_C) ops. No CUTLASS, no cuBLAS.
#
# nvcc flags: the fork's setup.py flags minus -std (CMake sets C++20) and the
# gencode (K01 appends -gencode=arch=compute_70,code=sm_70 per source):
#   -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
#   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
#   -U__CUDA_NO_HALF2_OPERATORS__
# (the -U flags undo torch's -D__CUDA_NO_HALF*__; upstream vLLM does the same
# in cmake/utils.cmake).
#
# The one-launch fused sink decode (C4, patch K14) is compiled only with
# -DVLLM_SM70_FV100_FUSED_DECODE (set VLLM_SM70_FV100_FUSED_DECODE=ON before
# this include). K05 leaves it off.

set(_VLLM_SM70_FV100_ROOT "${CMAKE_CURRENT_LIST_DIR}")

set(VLLM_SM70_FV100_CUDA_SRCS
  "${_VLLM_SM70_FV100_ROOT}/flash_decode_paged.cu"
  "${_VLLM_SM70_FV100_ROOT}/fused_mha_forward_paged.cu")
set(VLLM_SM70_FV100_CXX_SRCS
  "${_VLLM_SM70_FV100_ROOT}/fv100_ops.cpp")

set(VLLM_SM70_FV100_CUDA_FLAGS
  -O3
  --use_fast_math
  --expt-relaxed-constexpr
  --expt-extended-lambda
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__)

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_FV100_CUDA_SRCS})
  list(APPEND SM70_CXX_SRCS ${VLLM_SM70_FV100_CXX_SRCS})
  list(APPEND SM70_INCLUDE_DIRS "${_VLLM_SM70_FV100_ROOT}")
  # K01 appends the gencode to COMPILE_OPTIONS after this include.
  set_property(SOURCE ${VLLM_SM70_FV100_CUDA_SRCS} APPEND PROPERTY
    COMPILE_OPTIONS ${VLLM_SM70_FV100_CUDA_FLAGS})
  if(VLLM_SM70_FV100_FUSED_DECODE)
    set_property(SOURCE ${VLLM_SM70_FV100_CUDA_SRCS} ${VLLM_SM70_FV100_CXX_SRCS}
      APPEND PROPERTY COMPILE_DEFINITIONS VLLM_SM70_FV100_FUSED_DECODE=1)
  endif()
  message(STATUS "vllm_sm70: Flash-V100 lean attention (K05)"
                 " fused_decode=${VLLM_SM70_FV100_FUSED_DECODE}")
endif()

unset(_VLLM_SM70_FV100_ROOT)
