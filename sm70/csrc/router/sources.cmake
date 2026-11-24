# Flint fused SM70 decode router: source list for _sm70_C.
#
# sm70/CMakeLists.txt does `include(csrc/router/sources.cmake)` when this file
# exists. It appends to SM70_CUDA_SRCS / SM70_INCLUDE_DIRS only when CMake
# found CUDA and sm_70 (a CPU-only build has no router op). The op registers
# with TORCH_LIBRARY_FRAGMENT(_sm70_C). No CUTLASS, no cuBLAS, no extra
# defines. No --use_fast_math: the kernel's __expf matches topk_sigmoid
# without fast-math.

set(VLLM_SM70_ROUTER_SRCS "${CMAKE_CURRENT_LIST_DIR}/flint_router_ops.cu")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_ROUTER_SRCS})
  list(APPEND SM70_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")
  message(STATUS "vllm_sm70: Flint fused router (K13)")
endif()
