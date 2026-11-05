# What vLLM's CMake would FetchContent at configure time (the sandbox has no
# network), passed through the env var named here. With patch 0001 and
# VLLM_SM70_MINIMAL_EXTERNALS (nix/package.nix), only these three are used:
# CUTLASS is compiled against; triton_kernels and vllm-flash-attn are Python
# files copied into the tree (FA's kernels are not built).
# Revisions are the ones pinned in v0.33.0's CMakeLists.txt / cmake/external_projects.
{ pkgs }:
let
  gh =
    owner: repo: rev: hash: extra:
    pkgs.fetchFromGitHub ({ inherit owner repo rev hash; } // extra);
in
{
  # CMakeLists.txt:512 CUTLASS_REVISION
  VLLM_CUTLASS_SRC_DIR = gh "NVIDIA" "cutlass" "v4.7.1" "sha256-z/jASQGRj9fYwgyJKvzWgowG7j2zNKb3xeFUzYwwLQA=" { };
  # triton_kernels.cmake:19 (path is the triton_kernels package dir itself)
  TRITON_KERNELS_SRC_DIR = {
    src = gh "triton-lang" "triton" "v3.5.1" "sha256-dyNRtS1qtU8C/iAf0Udt/1VgtKGSvng1+r2BtvT9RB4=" { };
    subdir = "python/triton_kernels/triton_kernels";
  };
  # vllm_flash_attn.cmake:42. Python files only, so no submodules (its
  # csrc/cutlass is for the FA2/FA3 kernels).
  VLLM_FLASH_ATTN_SRC_DIR = gh "vllm-project" "flash-attention"
    "b273a02621d3c94bc747df67204f591b67f38fae" "sha256-8qGtJl18nIo8Kd/jPYlDK8K83uuu4W1mZ0x0oTvXU8w=" { };
}
