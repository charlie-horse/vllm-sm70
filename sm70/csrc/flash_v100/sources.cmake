# Flash-V100 attention: source list for _sm70_C.
#
# sm70/CMakeLists.txt does `include(csrc/flash_v100/sources.cmake)` when this
# file exists, after the TurboMind include. It appends to SM70_CXX_SRCS /
# SM70_CUDA_SRCS / SM70_INCLUDE_DIRS only when CUDA and sm_70 were found.
# fv100_ops.cpp registers the ops. No CUTLASS, no cuBLAS. nvcc flags below;
# the -U flags undo torch's -D__CUDA_NO_HALF*__, as upstream's
# cmake/utils.cmake does. The sm_70 gencode is appended after this include.

# VLLM_SM70_FV100_FUSED_DECODE (default ON) compiles the one-launch fused
# sink decode: the kernel, its launch branch and the
# `fv100_decode_paged_sinks_fused` op. With it OFF the Python side finds no
# fused op and keeps two launches. At runtime
# VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0 selects the two-launch route instead.
option(VLLM_SM70_FV100_FUSED_DECODE
       "Flash-V100 one-launch sink decode (K14)" ON)

set(_VLLM_SM70_FV100_ROOT "${CMAKE_CURRENT_LIST_DIR}")

set(VLLM_SM70_FV100_CUDA_SRCS
  "${_VLLM_SM70_FV100_ROOT}/flash_decode_paged.cu"
  "${_VLLM_SM70_FV100_ROOT}/fused_mha_forward_paged.cu"
  "${_VLLM_SM70_FV100_ROOT}/verify_decode.cu")
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
  # The sm_70 gencode is appended to COMPILE_OPTIONS after this include.
  set_property(SOURCE ${VLLM_SM70_FV100_CUDA_SRCS} APPEND PROPERTY
    COMPILE_OPTIONS ${VLLM_SM70_FV100_CUDA_FLAGS})
  if(VLLM_SM70_FV100_FUSED_DECODE)
    set_property(SOURCE ${VLLM_SM70_FV100_CUDA_SRCS} ${VLLM_SM70_FV100_CXX_SRCS}
      APPEND PROPERTY COMPILE_DEFINITIONS VLLM_SM70_FV100_FUSED_DECODE=1)
  endif()
  message(STATUS "vllm_sm70: Flash-V100 lean attention (K05/K14)"
                 " fused_decode=${VLLM_SM70_FV100_FUSED_DECODE}")
endif()

unset(_VLLM_SM70_FV100_ROOT)
