# What vLLM's CMake would FetchContent at configure time (the sandbox has no
# network), passed through the env var named here. With patch 0001 and
# VLLM_SM70_TRIM_EXTERNALS=1 (nix/package.nix), only these three are used:
# CUTLASS is compiled against; triton_kernels is Python files copied into the
# tree; vllm-flash-attn builds FA2 (vllm.vllm_flash_attn needs it to import)
# and installs its Python files. FA3 is configured but not built.
# Revisions are the ones pinned in v0.33.0's CMakeLists.txt / cmake/external_projects.
{ pkgs }:
let
  gh =
    owner: repo: rev: hash: extra:
    pkgs.fetchFromGitHub ({ inherit owner repo rev hash; } // extra);

  # vllm_flash_attn.cmake:42. Fetched without submodules (composable_kernel
  # and aiter are ROCm-only, 513 MB with them); FA2 needs only csrc/cutlass,
  # added at the commit the submodule pins.
  flashAttn = gh "vllm-project" "flash-attention" "b273a02621d3c94bc747df67204f591b67f38fae"
    "sha256-8qGtJl18nIo8Kd/jPYlDK8K83uuu4W1mZ0x0oTvXU8w=" { };
  flashAttnCutlass = gh "NVIDIA" "cutlass" "62750a2b75c802660e4894434dc55e839f322277"
    "sha256-eL64JYiFAqrn6OJDzijpMR4DbEB4El/8yT7++iCZBlE=" { };
in
{
  # CMakeLists.txt:512 CUTLASS_REVISION v4.7.1
  VLLM_CUTLASS_SRC_DIR = gh "NVIDIA" "cutlass" "cb4247394dd82148787aed73e5dc7cef33cbf862" "sha256-z/jASQGRj9fYwgyJKvzWgowG7j2zNKb3xeFUzYwwLQA=" { };
  # triton_kernels.cmake:19 TRITON_KERNELS_TAG v3.5.1 (path is the
  # triton_kernels package dir itself)
  TRITON_KERNELS_SRC_DIR = {
    src = gh "triton-lang" "triton" "0add68262ab0a2e33b84524346cb27cbb2787356" "sha256-dyNRtS1qtU8C/iAf0Udt/1VgtKGSvng1+r2BtvT9RB4=" { };
    subdir = "python/triton_kernels/triton_kernels";
  };
  VLLM_FLASH_ATTN_SRC_DIR = pkgs.runCommand "flash-attention-b273a02-src" { } ''
    cp -r --no-preserve=mode ${flashAttn} $out
    rmdir $out/csrc/cutlass 2>/dev/null || rm -rf $out/csrc/cutlass
    cp -r --no-preserve=mode ${flashAttnCutlass} $out/csrc/cutlass
  '';
}
