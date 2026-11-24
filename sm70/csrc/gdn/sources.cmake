# FlashQLA SM70 GDN kernels: source list for the _sm70_C extension.
#
# sm70/CMakeLists.txt does `include(csrc/gdn/sources.cmake)` when this file
# exists. It appends to SM70_CUDA_SRCS only when CMake found CUDA and sm_70
# (a CPU-only build has no GDN ops). gdn_forward.cu registers with
# TORCH_LIBRARY_FRAGMENT(_sm70_C, ...). No extra include dirs, defines or
# libraries: torch and the CUDA runtime only. Not CUTLASS.

set(VLLM_SM70_GDN_SRCS "${CMAKE_CURRENT_LIST_DIR}/gdn_forward.cu")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_GDN_SRCS})
  message(STATUS "vllm_sm70: FlashQLA GDN kernels (K08)")
endif()
