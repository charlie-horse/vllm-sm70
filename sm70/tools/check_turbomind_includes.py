#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks on sm70/csrc/turbomind that need no nvcc.

1. Every `#include "..."` in a kept file resolves inside the tree. Lookup
   order: the includer's directory, then VLLM_SM70_TM_INCLUDE_DIRS
   (lmdeploy/, ops/), the same order nvcc uses for quoted includes.
2. Every `#include <...>` is a C++ standard, CUDA toolkit, torch or CUTLASS
   header.
3. No kept file names a dropped symbol (gguf, Lattice, MXF4/MXFP4,
   U8_Affine, U2_Affine, Lut4, BitPlane) outside an allowlist of files known to keep such references
   verbatim. Every hit is printed with the reason it is allowed.
4. Every source listed in sources.cmake exists.
Preprocessor conditionals are ignored, so (1)/(2) over-approximate.

Exit status: 0 = clean, 1 = problems found.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TREE = REPO / "sm70/csrc/turbomind"
INCLUDE_DIRS = [TREE / "lmdeploy", TREE / "ops"]
EXTS = {".h", ".cuh", ".cu", ".cc", ".cpp", ".inc"}

STD = set(
    "algorithm array atomic cassert cfloat climits cmath cstddef cstdint cstdio "
    "cstdlib cstring filesystem fstream functional initializer_list iomanip "
    "iostream iterator limits map memory mutex numeric optional ostream random "
    "regex set sstream stack stdexcept string string_view thread tuple "
    "type_traits unordered_map unordered_set utility variant vector".split()
)
CUDA = set(
    "cuda.h cuda_runtime.h cuda_runtime_api.h cuda_fp16.h cuda_bf16.h "
    "cuda_fp8.h cuda_pipeline_primitives.h cublas_v2.h cublasLt.h "
    "cusparseLt.h".split()
)
# cutlass/: headers from VLLM_SM70_CUTLASS_DIR (K03 qwen38_prefill_cutlass.cu).
PREFIXES = ("cub/", "torch/", "ATen/", "c10/", "cutlass/")

# Dropped-format names (Kai's list plus U2_Affine/Lut4/BitPlane) that remain (R7: headers kept verbatim;
# no kernels for these formats are registered any more).
DROPPED = re.compile(
    r"gguf|Gguf|GGUF|Lattice|lattice|MXF4|Mxfp4|mxfp4|MXFP4|U8_Affine|U2_Affine"
    r"|Lut4|BitPlane"
)
ALLOW = {
    "lmdeploy/src/turbomind/kernels/gemm/lattice_codebooks.h":
        "R7: lattice header kept verbatim",
    "lmdeploy/src/turbomind/kernels/gemm/lattice_transform.h":
        "R7: lattice header kept verbatim",
    "lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h":
        "R7: GGUF/MXF4/U8 Config_* templates defined (includes "
        "lattice_transform.h); no registry instantiates them now",
    "lmdeploy/src/turbomind/kernels/gemm/transform.h":
        "R7: GGUF transforms defined, never instantiated",
    "lmdeploy/src/turbomind/kernels/gemm/types.h":
        "verbatim: QuantType::kLattice* names, DispatchPolicy::"
        "kMxfp4MoeGroupedM8Fast flag",
    "lmdeploy/src/turbomind/kernels/gemm/convert.h":
        "verbatim: declares GetGguf{BitPlane,Lut4,Lattice}Converters",
    "lmdeploy/src/turbomind/kernels/gemm/convert_v3.cu":
        "verbatim: GetGguf*Converters build generic uint2/uint4/uint16 layout "
        "converters; nothing in K02/K03 calls them",
    "lmdeploy/src/turbomind/kernels/gemm/gemm.cu":
        "verbatim: MXFP4 grouped-m8 fast selector, gated by "
        "VLLM_SM70_MXFP4_MOE_GROUPED_M8 (default 0)",
    "lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h":
        "verbatim: QuantType::kBitPlane in the generic quant-type deduction",
    "ops/tm_config.h": "knob table keeps the two MXFP4 knobs gemm.cu reads",
}
MUST_BE_CLEAN = {
    "lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu",
    "lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu",
    "lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu",
    "ops/gemm_runtime.cpp",
    "ops/gemm_runtime.h",
    "ops/tm_registry_sm70.cu",
}
INC = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)


def main() -> int:
    problems: list[str] = []
    files = sorted(p for p in TREE.rglob("*") if p.suffix in EXTS)
    n_quoted = n_angle = 0
    for f in files:
        text = f.read_text(errors="replace")
        rel = f.relative_to(TREE).as_posix()
        for kind, name in INC.findall(text):
            if kind == '"':
                n_quoted += 1
                cands = [f.parent / name] + [d / name for d in INCLUDE_DIRS]
                hit = next((c for c in cands if c.is_file()), None)
                if hit is None:
                    problems.append(f"unresolved {rel}: \"{name}\"")
                elif TREE not in hit.resolve().parents and hit.resolve() != (
                    REPO / "sm70/csrc/sm70_tile_runtime_signal.cuh"
                ):
                    problems.append(f"outside tree {rel}: \"{name}\" -> {hit}")
            else:
                n_angle += 1
                if not (name in STD or name in CUDA or name.startswith(PREFIXES)):
                    problems.append(f"non-system angle include {rel}: <{name}>")
        hits = sorted(set(DROPPED.findall(text)))
        if hits:
            if rel in MUST_BE_CLEAN:
                problems.append(f"dropped symbol in {rel}: {hits}")
            elif rel in ALLOW:
                print(f"allowed  {rel}: {hits} ({ALLOW[rel]})")
            else:
                problems.append(f"dropped symbol in {rel}: {hits} (not allowlisted)")

    cmake = (TREE / "sources.cmake").read_text()
    lm = TREE / "lmdeploy/src/turbomind"
    for m in re.finditer(r'"\$\{(_VLLM_SM70_TM_LM|_VLLM_SM70_TM_ROOT)\}/([^"]+)"', cmake):
        base = lm if m.group(1) == "_VLLM_SM70_TM_LM" else TREE
        if not (base / m.group(2)).exists():
            problems.append(f"sources.cmake: missing {m.group(2)}")

    for p in problems:
        print("PROBLEM", p)
    print(
        f"{len(files)} files, {n_quoted} quoted + {n_angle} angle includes: "
        f"{'OK' if not problems else f'{len(problems)} problem(s)'}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
