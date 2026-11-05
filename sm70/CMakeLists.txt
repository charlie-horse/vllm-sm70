# SPDX-License-Identifier: Apache-2.0
#
# vllm_sm70._sm70_C: Volta (sm_70) kernels for vLLM, built out of tree.
# Uses torch's CMake config (torch 2.14). Ops register with the non-stable
# TORCH_LIBRARY API in their own namespace, `_sm70_C`.
#
# Driven by setup.py. Offline: nothing here uses FetchContent.

cmake_minimum_required(VERSION 3.26)
project(vllm_sm70 LANGUAGES CXX)

# torch 2.14 headers need C++20 (as UP CMakeLists.txt:16-19).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Bump when an op schema or kernel ABI changes. Exposed as
# torch.ops._sm70_C.abi_version() and folded into the compile-cache key.
set(VLLM_SM70_ABI_VERSION 1)

option(VLLM_SM70_REQUIRE_CUDA "Fail unless CUDA with sm_70 is available" ON)
set(VLLM_SM70_CUTLASS_DIR "" CACHE PATH "Optional CUTLASS source tree (read-only)")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#
# Arch selection: the same idea as upstream's cuda_archs_loose_intersection,
# but the only supported arch is 7.0. "+PTX" and a/f suffixes are dropped.
#
set(SM70_SUPPORTED_ARCHS "7.0")

function(sm70_archs_loose_intersection OUT SUPPORTED REQUESTED)
  set(_result)
  foreach(_arch IN LISTS REQUESTED)
    string(REGEX REPLACE "\\+PTX$" "" _arch "${_arch}")
    string(REGEX REPLACE "[af]$" "" _arch "${_arch}")
    if(_arch IN_LIST SUPPORTED)
      list(APPEND _result "${_arch}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _result)
  set(${OUT} "${_result}" PARENT_SCOPE)
endfunction()

if(DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
  string(REPLACE " " ";" _requested_archs "$ENV{TORCH_CUDA_ARCH_LIST}")
else()
  set(_requested_archs "7.0")
endif()
sm70_archs_loose_intersection(SM70_ARCHS "${SM70_SUPPORTED_ARCHS}" "${_requested_archs}")

set(SM70_HAS_CUDA OFF)
if(CMAKE_CUDA_COMPILER OR TORCH_CUDA_LIBRARIES OR CUDA_FOUND)
  set(SM70_HAS_CUDA ON)
endif()

if(SM70_HAS_CUDA AND SM70_ARCHS)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot target sm_70; use 12.x")
  endif()
  # torch's config appends -gencode flags for TORCH_CUDA_ARCH_LIST; drop
  # them and set gencode per source below, as upstream vLLM does.
  string(REGEX REPLACE "-gencode[= ]+arch=compute_[0-9a-z]+,code=[a-z_0-9,\\[\\]]+" ""
         CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  set(SM70_CUDA_GENCODE "-gencode=arch=compute_70,code=sm_70")
  message(STATUS "vllm_sm70: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, archs: ${SM70_ARCHS}")
elseif(VLLM_SM70_REQUIRE_CUDA)
  message(FATAL_ERROR
    "vllm_sm70 needs CUDA 12.x and sm_70 in TORCH_CUDA_ARCH_LIST "
    "(requested: ${_requested_archs}; CUDA found: ${SM70_HAS_CUDA}). "
    "Set VLLM_SM70_SKIP_EXT=1 to build the Python package only.")
else()
  message(WARNING "vllm_sm70: building the CPU-only binding (no CUDA sources)")
endif()

#
# Sources. Area files append to these lists:
#   SM70_CXX_SRCS      host C++ sources
#   SM70_CUDA_SRCS     .cu sources (built for sm_70 only)
#   SM70_INCLUDE_DIRS  extra include directories
#
set(SM70_CXX_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch_bindings_sm70.cpp")
set(SM70_CUDA_SRCS)
set(SM70_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

if(VLLM_SM70_CUTLASS_DIR)
  # Headers only, so the tree may be read-only (a Nix store path).
  if(NOT EXISTS "${VLLM_SM70_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "VLLM_SM70_CUTLASS_DIR=${VLLM_SM70_CUTLASS_DIR} has no include/cutlass/cutlass.h")
  endif()
  list(APPEND SM70_INCLUDE_DIRS
    "${VLLM_SM70_CUTLASS_DIR}/include"
    "${VLLM_SM70_CUTLASS_DIR}/tools/util/include")
endif()

# Hook for K02 (TurboMind GEMM core). The file appends to the lists above.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/turbomind/sources.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/csrc/turbomind/sources.cmake")
endif()

# Hook for K08 (FlashQLA GDN prefill/decode). Same contract as above.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn/sources.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/csrc/gdn/sources.cmake")
endif()

# Hook for K05 (Flash-V100 attention, lean subset). Same contract.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_v100/sources.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_v100/sources.cmake")
endif()

if(SM70_CUDA_SRCS AND NOT (SM70_HAS_CUDA AND SM70_ARCHS))
  message(FATAL_ERROR "vllm_sm70: CUDA sources listed but no CUDA/sm_70 toolchain")
endif()
# APPEND (K05): area files may set their own per-file nvcc options first.
foreach(_src IN LISTS SM70_CUDA_SRCS)
  set_property(SOURCE "${_src}" APPEND PROPERTY COMPILE_OPTIONS "${SM70_CUDA_GENCODE}")
endforeach()

Python_add_library(_sm70_C MODULE WITH_SOABI ${SM70_CXX_SRCS} ${SM70_CUDA_SRCS})
target_include_directories(_sm70_C PRIVATE ${SM70_INCLUDE_DIRS})
target_compile_definitions(_sm70_C PRIVATE
  TORCH_EXTENSION_NAME=_sm70_C
  VLLM_SM70_ABI_VERSION=${VLLM_SM70_ABI_VERSION})
target_link_libraries(_sm70_C PRIVATE ${TORCH_LIBRARIES})
if(SM70_CUDA_SRCS)
  set_target_properties(_sm70_C PROPERTIES CUDA_ARCHITECTURES OFF)
endif()
