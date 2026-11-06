# DFlash2 draft FP16 M<=8 GEMM + compact FP16 top-64 (patch K18): source list
# for the _sm70_C extension.
#
# sm70/CMakeLists.txt does `include(csrc/dflash2/sources.cmake)` when this
# file exists. Same contract as csrc/gdn/sources.cmake: it appends to
# SM70_CUDA_SRCS only when CMake found CUDA and sm_70 (a CPU-only build has no
# DFlash2 ops). In the fork both files were part of VLLM_SM70_TURBOMIND_SRCS
# (fork CMakeLists.txt:407-408) and built with upstream's VLLM_GPU_FLAGS;
# neither needs those flags (no half operators, only intrinsics and inline
# PTX). Each file registers its own TORCH_LIBRARY_FRAGMENT(_sm70_C, ...).
# No extra include dirs, defines or libraries: torch, CUB (from the CUDA
# toolkit) and the CUDA runtime only.

set(VLLM_SM70_DFLASH2_SRCS
  "${CMAKE_CURRENT_LIST_DIR}/dflash2_fp16_m8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/compact_half_topk.cu")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_DFLASH2_SRCS})
  message(STATUS "vllm_sm70: DFlash2 FP16 M8 GEMM + compact top-k (K18)")
endif()
