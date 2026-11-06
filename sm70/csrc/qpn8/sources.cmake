# QPN8 block-FP8 decode GEMV (patch K09): source list for the _sm70_C extension.
#
# sm70/CMakeLists.txt does `include(csrc/qpn8/sources.cmake)` when this file
# exists. It appends to SM70_CUDA_SRCS only when CMake found CUDA and sm_70
# (a CPU-only build has no QPN8 ops). The fork compiled fp8_qpn8_sm70.cu in
# its SM70 TurboMind source list (fork CMakeLists.txt:406) and registered the
# ops in csrc/torch_bindings.cpp:723-758; here it is one more TU of _sm70_C,
# registered with TORCH_LIBRARY_FRAGMENT(_sm70_C, ...) in the file itself.
# No extra include dirs, defines or libraries: torch and the CUDA runtime
# only. Not CUTLASS.

set(VLLM_SM70_QPN8_SRCS "${CMAKE_CURRENT_LIST_DIR}/fp8_qpn8_sm70.cu")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_QPN8_SRCS})
  message(STATUS "vllm_sm70: QPN8 FP8 decode GEMV (K09)")
endif()
