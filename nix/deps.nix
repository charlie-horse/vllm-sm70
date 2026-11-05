# Everything vLLM's CMake would FetchContent at configure time (the sandbox
# has no network). v0.33.0 includes all of these whenever the target is CUDA,
# whatever the arch; on sm_70 + nvcc 12.6 only CUTLASS is compiled against,
# the rest are copied (Python files) or produce empty targets. Each is passed
# through the env var named here, so the upstream CMake stays unpatched.
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
  # vllm_flash_attn.cmake:42
  VLLM_FLASH_ATTN_SRC_DIR = gh "vllm-project" "flash-attention"
    "b273a02621d3c94bc747df67204f591b67f38fae" "sha256-0akJwKOdunEZeU289NbdvYH/3QXqeiMCRLdmId4ONRY=" { fetchSubmodules = true; };
  # deepgemm.cmake:40 (submodules third-party/cutlass, third-party/deep_jit)
  DEEPGEMM_SRC_DIR = gh "vllm-project" "DeepGEMM"
    "0820030756945b2daec208e8e8255268360f337d" "sha256-Qbup5RiPYdbREooAt8m2Gq5C/B8EB4ywkTJjFJIxRbU=" { fetchSubmodules = true; };
  # deepselect.cmake:22
  DEEPSELECT_SRC_DIR = gh "vllm-project" "DeepSelect"
    "d96d33afe1fab0d6066da49cdc91e64c2bee65ea" "sha256-7+lmKQH08v9dhQ9KaZtcs25haMdkwKOEk5noS9/GiB8=" { };
  # fmha_sm100.cmake:20 (python/fmha_sm100/cutlass is a submodule)
  FMHA_SM100_SRC_DIR = gh "vllm-project" "MSA"
    "968061ab16cb14991a5c7d4c0e8eabf531be0dcf" "sha256-KDDjfMXys47XeYTMS3eQC+qa257XU88PikoEsMaCO9g=" { fetchSubmodules = true; };
  # flashmla.cmake:22
  FLASH_MLA_SRC_DIR = gh "vllm-project" "FlashMLA"
    "e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b" "sha256-ffqgYT/L2cYs/PY1QZPsSskPz56Rw3F5XAvaUHbytK8=" { fetchSubmodules = true; };
  # flashkda.cmake:16 (submodule cutlass)
  FLASH_KDA_SRC_DIR = gh "vllm-project" "FlashKDA"
    "17a037d98da546deb4591e967cf961a43c034d8b" "sha256-1IYJcMN70jY9gT/OWNhyEZswc1h1+b+HsSbZ8uLPQSQ=" { fetchSubmodules = true; };
  # qutlass.cmake:25
  QUTLASS_SRC_DIR = gh "IST-DASLab" "qutlass"
    "e74319e3405ce6d71965732880f5dc1f52371f64" "sha256-Gzl3KuYXXLXMrVciEYrBPu1FH2cplGUPTFpWzFfUmMo=" { };
  # tml_fa4.cmake:17
  TML_FA4_SRC_DIR = gh "vllm-project" "tml-fa4"
    "75765e76a9c2c012c1f6ecd64577eb646eb4d303" "sha256-zw19GAgsA0+5XzJNGLInIMN4GMRXqJsLkiVViRkv6fI=" { };
}
