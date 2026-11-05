# Flint fused SM70 decode router (patch K13): source list for _sm70_C.
#
# sm70/CMakeLists.txt does `include(csrc/router/sources.cmake)` when this file
# exists. It appends to SM70_CUDA_SRCS / SM70_INCLUDE_DIRS only when CMake
# found CUDA and sm_70 (a CPU-only build has no router op). The fork built
# sm70_router.cu as its own extension `_sm70_router_C` (fork CMakeLists.txt
# :841-846); here it is one more TU of _sm70_C (TORCH_LIBRARY_FRAGMENT).
# No CUTLASS, no cuBLAS, no extra defines. The fork compiled it with the
# _C flags only (no --use_fast_math); keep it that way: the kernel's
# __expf matches topk_sigmoid without fast-math.

set(VLLM_SM70_ROUTER_SRCS "${CMAKE_CURRENT_LIST_DIR}/flint_router_ops.cu")

if(SM70_HAS_CUDA AND SM70_ARCHS)
  list(APPEND SM70_CUDA_SRCS ${VLLM_SM70_ROUTER_SRCS})
  list(APPEND SM70_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")
  message(STATUS "vllm_sm70: Flint fused router (K13)")
endif()
