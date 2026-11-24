# Flint NVFP4 MoE QPN decode: source list for the _sm70_C extension.
#
# sm70/CMakeLists.txt does `include(csrc/flint_moe/sources.cmake)` when this
# file exists. It appends to SM70_CUDA_SRCS only when CMake found CUDA and
# sm_70 (a CPU-only build has no decode ops). No TurboMind headers: torch and
# the CUDA runtime only.
#
# nvcc flags: the -U flags drop torch's -D__CUDA_NO_HALF*__ defines, as
# upstream's torch nvcc flags do for CUDA >= 12. No --use_fast_math: the
# SwiGLU epilogue's expf must match the eager FP16 SiLU arithmetic. The
# sm_70 gencode is appended after this include.

set(VLLM_SM70_FLINT_MOE_SRCS
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_moe_decode_qpn_sm70.cu")

set(VLLM_SM70_FLINT_MOE_CUDA_FLAGS
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__)

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_FLINT_MOE_SRCS})
  set_property(SOURCE ${VLLM_SM70_FLINT_MOE_SRCS} APPEND PROPERTY
    COMPILE_OPTIONS ${VLLM_SM70_FLINT_MOE_CUDA_FLAGS})
  message(STATUS "vllm_sm70: Flint NVFP4 MoE QPN decode (K12)")
endif()
