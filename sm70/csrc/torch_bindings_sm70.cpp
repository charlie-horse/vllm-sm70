// SPDX-License-Identifier: Apache-2.0
//
// Op registry for vllm_sm70._sm70_C. Every sm70 op lives in the `_sm70_C`
// namespace so it can never collide with upstream's stable `_C` ops.
// Later patches (K02+) add their defs here and their impls in csrc/.

#include <Python.h>
#include <torch/library.h>

#include <cstdint>

#ifndef VLLM_SM70_ABI_VERSION
  #error "VLLM_SM70_ABI_VERSION must be defined by CMakeLists.txt"
#endif

namespace vllm_sm70 {

// Bumped in CMakeLists.txt whenever an op schema or kernel ABI changes.
int64_t abi_version() { return VLLM_SM70_ABI_VERSION; }

}  // namespace vllm_sm70

TORCH_LIBRARY(_sm70_C, m) { m.def("abi_version() -> int"); }

TORCH_LIBRARY_IMPL(_sm70_C, CPU, m) {
  m.impl("abi_version", &vllm_sm70::abi_version);
}

// An op with no tensor arguments has an empty dispatch key set, so the CPU
// kernel alone is never selected; this is its fallback (as upstream does for
// its no-tensor ops, UP csrc/torch_bindings.cpp:963).
TORCH_LIBRARY_IMPL(_sm70_C, CompositeExplicitAutograd, m) {
  m.impl("abi_version", &vllm_sm70::abi_version);
}

// Makes `import vllm_sm70._sm70_C` work; the ops register on load.
static struct PyModuleDef sm70_module = {PyModuleDef_HEAD_INIT, "_sm70_C",
                                         nullptr, 0, nullptr};

PyMODINIT_FUNC PyInit__sm70_C(void) { return PyModule_Create(&sm70_module); }
