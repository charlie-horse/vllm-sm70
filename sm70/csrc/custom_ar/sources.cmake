# Kernels on upstream's CustomAllreduce: source list for _sm70_C.
#   tp2_ar_fused_add_rms_norm.cu  TP2 all-reduce + add + RMSNorm (C6, K17)
#   top1_custom_ar.cu             greedy top-1 pair exchange (F12,
#                                 VLLM_SM70_TOP1_CUSTOM_AR)
# Both include custom_ar_common.cuh (ABI pins, staged start barrier).
#
# sm70/CMakeLists.txt does `include(csrc/custom_ar/sources.cmake)` when this
# file exists. Same contract as the other areas: append to SM70_CUDA_SRCS
# only when CUDA and sm_70 were found.
#
# The kernel reads upstream vLLM's CustomAllreduce (a handle from vLLM's
# _C_custom_ar), so it must be built against the exact csrc headers vLLM was
# built from: $ENV{VLLM_SM70_VLLM_CSRC_DIR}, the patched tree's csrc/ (Nix
# sets it in nix/sm70-ext.nix). That directory goes on the include path of
# these sources only, so upstream headers can never shadow ours elsewhere.
# Without it the area is skipped with a message; the Python side then finds
# no _sm70_C::tp2_all_reduce_fused_add_rms_norm (the fusion stays off) and
# no _sm70_C::custom_ar_top1_argmax (greedy top-1 keeps the all-gather).
#
# nvcc flags: -O3 and the -U__CUDA_NO_HALF*__ flags (undo torch's defines;
# _f16Vec and packed_assign_add use the half operators), as upstream vLLM
# does in cmake/utils.cmake. No --use_fast_math: rsqrtf and the FP32 sums
# must round exactly as upstream's fused_add_rms_norm kernel does.

set(_VLLM_SM70_CUSTOM_AR_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(VLLM_SM70_VLLM_CSRC_DIR "$ENV{VLLM_SM70_VLLM_CSRC_DIR}")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  if(VLLM_SM70_VLLM_CSRC_DIR STREQUAL "")
    message(STATUS "vllm_sm70: VLLM_SM70_VLLM_CSRC_DIR is unset; skipping "
                   "custom_ar (K17 TP2 all-reduce + RMSNorm and the F12 custom-AR "
                   "top-1 exchange stay off)")
  elseif(NOT EXISTS "${VLLM_SM70_VLLM_CSRC_DIR}/custom_all_reduce.cuh"
         OR NOT EXISTS "${VLLM_SM70_VLLM_CSRC_DIR}/custom_collective_common.cuh"
         OR NOT EXISTS "${VLLM_SM70_VLLM_CSRC_DIR}/type_convert.cuh")
    message(FATAL_ERROR "VLLM_SM70_VLLM_CSRC_DIR=${VLLM_SM70_VLLM_CSRC_DIR} "
                        "lacks custom_all_reduce.cuh, "
                        "custom_collective_common.cuh or type_convert.cuh")
  else()
    # The kernel reads CustomAllreduce from another .so: refuse headers other
    # than the pinned ones (UPSTREAM_HEADERS.sha256).
    file(STRINGS "${_VLLM_SM70_CUSTOM_AR_ROOT}/UPSTREAM_HEADERS.sha256" _pins
         REGEX "^[0-9a-f]+  ")
    foreach(_pin IN LISTS _pins)
      string(REGEX MATCH "^([0-9a-f]+)  (.+)$" _m "${_pin}")
      set(_want "${CMAKE_MATCH_1}")
      set(_name "${CMAKE_MATCH_2}")
      file(SHA256 "${VLLM_SM70_VLLM_CSRC_DIR}/${_name}" _got)
      if(NOT _got STREQUAL _want)
        message(FATAL_ERROR "custom_ar: ${VLLM_SM70_VLLM_CSRC_DIR}/${_name} "
                            "is not the pinned v0.33.0 header (sha256 ${_got}, "
                            "want ${_want}); see csrc/custom_ar/PROVENANCE.md")
      endif()
    endforeach()
    set(VLLM_SM70_CUSTOM_AR_CUDA_SRCS
      "${_VLLM_SM70_CUSTOM_AR_ROOT}/tp2_ar_fused_add_rms_norm.cu"
      "${_VLLM_SM70_CUSTOM_AR_ROOT}/top1_custom_ar.cu")
    list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_CUSTOM_AR_CUDA_SRCS})
    set_property(SOURCE ${VLLM_SM70_CUSTOM_AR_CUDA_SRCS} APPEND PROPERTY
      INCLUDE_DIRECTORIES "${VLLM_SM70_VLLM_CSRC_DIR}")
    set_property(SOURCE ${VLLM_SM70_CUSTOM_AR_CUDA_SRCS} APPEND PROPERTY
      COMPILE_OPTIONS
        -O3
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__)
    message(STATUS "vllm_sm70: custom_ar (K17, F12 top-1) against ${VLLM_SM70_VLLM_CSRC_DIR}")
  endif()
endif()

unset(_VLLM_SM70_CUSTOM_AR_ROOT)
