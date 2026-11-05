# Pins

Locked 2026-11-04. Change a pin only through a PR that updates this file.

**Base: vLLM v0.33.0 @ 4ba787ec9959e566d3b587a796dff07673324da5.** It was the only v0.33.* release when pinned, and it pins torch==2.14.0 and Triton 3.8.0. The vLLM and cutlass hashes were re-confirmed with `nix flake prefetch` on a separate host.

## Pins
| component | pin | source | hash | evidence |
|---|---|---|---|---|
| vLLM | v0.33.0 = 4ba787ec9959e566d3b587a796dff07673324da5 (== v0.33.0rc3) | github:vllm-project/vllm | fetchFromGitHub `sha256-I72HXJjBOCmNSwbuK/D3KB5/YGGJ0f1OY5BhWDevI/w=` | `git ls-remote` shows v0.33.0^{} = 4ba787e, commit date 2026-10-31. Release published 11-02 and marked Latest. Hash = `nix flake prefetch github:vllm-project/vllm/v0.33.0` (the authoritative value; it hashes GitHub's tarball). It equals `nix hash path` of a `git archive` export only because the tag has no `.gitattributes` with `export-ignore`/`export-subst`. Re-check that if a future tag adds one |
| torch | 2.14.0+cu126, cp313 manylinux_2_28_x86_64 | https://download-r2.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp313-cp313-manylinux_2_28_x86_64.whl | sha256 `ac9d1ce1093606f5cb1deb4f612d2fcb273ed10ff4a33703abba09cc16c5dcc8` = `sha256-rJ0c4Qk2BvXLHetPYS0vyyc+0Q/0ozcDq7oJzBbF3Mg=` (869,176,671 B) | Download sha256sum matches the index `#sha256`. **sm_70 SASS:** a .nv_fatbin parse of torch/lib/libtorch_cuda.so finds ELF cubins sm_50/60/**70**/75/80/86 ×418 each, sm_90 ×422, no PTX. The embedded build flags string is `-gencode;arch=compute_70,code=sm_70;…compute_90,code=sm_90`. cp310–cp315 all exist |
| triton | 3.8.0, cp313 manylinux_2_27/2_28 x86_64 | https://files.pythonhosted.org/packages/b0/b6/3394d5548404c1cabd1dadadd28d0b3f9478db1dff8180da53bb3f0a1e19/triton-3.8.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl | sha256 `1f0497218e26b7d79773ad9c2a3fa3b539ee69f587a13fac2e552b1d322a8015` = `sha256-HwSXIY4mt9eXc62cKj+jtTnuafWHoT+sLlUrHTIqgBU=` (247,975,122 B) | torch METADATA `triton~=3.8.0`. Bundled `backends/nvidia/bin/ptxas` reports "release 12.9, V12.9.86", and its strings include `.target sm_70` (sm_70 OK). ptxas-blackwell is 13.3, and cuobjdump/nvdisasm are 13.1 |
| python | CPython 3.13.17 (python3 → python313) | nixos-26.05 all-packages.nix:4831 | – | nix eval |
| nixpkgs | nixos-26.05 @ 7dcdd87373d72e7a2173e4db02e20d3d7f495cfd (2026-11-02, nixos-26.05.13446) | github:NixOS/nixpkgs | narHash `sha256-tItrb9M8eYA6WhFgmT9tpGKueri3+wRuVA952o/2xSU=` | ls-remote plus channels.nixos.org git-revision; hash from git archive |
| CUDA toolkit | cudaPackages_12_6 (manifest 12.6.3): nvcc 12.6.85, cudart 12.6.77, cublas 12.6.4.1, cusparse 12.5.4.2, cupti 12.6.80, cufile 1.11.1.6, nvjitlink 12.6.85, cuda_nvtx 12.6.77, libcusparse_lt 0.6.3.2, libcutensor 2.3.1.0, nccl 2.28.7-1 | nixpkgs | – | nix eval with `cudaCapabilities=["7.0"]` |
| host cc | gcc 13.4.0 (`cudaPackages_12_6.backendStdenv.cc`) | nixpkgs | – | nvcc 12.6 supports gcc ≤13. **vLLM requires GCC ≥13** (CMakeLists.txt:1690-1700, FATAL_ERROR, for C++20 `<format>` in DeepGEMM/DeepSelect), so gcc13 is the only choice |
| cudnn | **use the pip wheel nvidia-cudnn-cu12 9.10.2.21** that torch needs, not nixpkgs | see wheels below | | nixpkgs cudnn for 12_6 is 9.22.0.52, and cudnn.nix:78-100 refuses caps <7.5 for cudnn ≥9.12 (eval error "not available on the requested hostPlatform"). 9.10 predates that cutoff |

### nvidia-* wheels required by torch 2.14.0+cu126 (all py3-none manylinux x86_64; URLs under files.pythonhosted.org/packages/)
Torch METADATA requires `cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvrtc,nvtx]==12.6.3`, plus `cuda-bindings>=12.9.4,<13` (latest is 12.9.10), cudnn, cusparselt, nccl, nvshmem and nvjitlink. cuda-toolkit 12.6.3 itself is a metapackage with no wheel; it expands to the rows below.
| wheel | ver | sha256 |
|---|---|---|
| nvidia-cublas-cu12 | 12.6.4.1 | 08ed2686e9875d01b58e3cb379c6896df8e76c75e0d4a7f7dace3d7b6d9ef8eb |
| nvidia-cuda-runtime-cu12 | 12.6.77 | ba3b56a4f896141e25e19ab287cd71e52a6a0f4b29d0d31609f60e3b4d5219b7 |
| nvidia-cufft-cu12 | 11.3.0.4 | ccba62eb9cef5559abd5e0d54ceed2d9934030f51163df018532142a8ec533e5 |
| nvidia-cufile-cu12 | 1.11.1.6 | cc23469d1c7e52ce6c1d55253273d32c565dd22068647f3aa59b3c6b005bf159 |
| nvidia-cuda-cupti-cu12 | 12.6.80 | 6768bad6cab4f19e8292125e5f1ac8aa7d1718704012a0e3272a6f61c4bce132 |
| nvidia-curand-cu12 | 10.3.7.77 | a42cd1344297f70b9e39a1e4f467a4e1c10f1da54ff7a85c12197f6c652c8bdf |
| nvidia-cusolver-cu12 | 11.7.1.2 | e9e49843a7707e42022babb9bcfa33c29857a93b88020c4e4434656a655b698c |
| nvidia-cusparse-cu12 | 12.5.4.2 | 7556d9eca156e18184b94947ade0fba5bb47d69cec46bf8660fd2c71a4b48b73 |
| nvidia-cuda-nvrtc-cu12 | 12.6.85 | 800927308ccc5dd6246d3f61f7fcef2ed7ec4e59e199090d360d3293f78bd5a2 |
| nvidia-nvtx-cu12 | 12.6.77 | b90bed3df379fa79afbd21be8e04a0314336b8ae16768b58f2d34cb1d04cd7d2 |
| nvidia-cudnn-cu12 | 9.10.2.21 | 949452be657fa16687d0930933f032835951ef0892b37d2d53824d1a84dc97a8 |
| nvidia-cusparselt-cu12 | 0.7.1 | f1bb701d6b930d5a7cea44c19ceb973311500847f81b634d802b7b539dc55623 |
| nvidia-nccl-cu12 | 2.29.3 | 35ad42e7d5d722a83c36a3a478e281c20a5646383deaf1b9ed1a9ab7d61bed53 |
| nvidia-nvshmem-cu12 | 3.4.5 | 042f2500f24c021db8a06c5eec2539027d57460e1c1a762055a6554f72c369bd |
| nvidia-nvjitlink-cu12 | 12.6.85 | eedc36df9e88b682efe4309aa16b5b4e78c2407eac59e8c10a6a47535164369a |
Full URLs are in the PyPI JSON (`/pypi/<name>/<ver>/json`).

### vLLM v0.33.0 Python/GPU deps (requirements/cuda.txt, build/cuda.txt, pyproject)
- **torch:** `torch==2.14.0` in pyproject build-system, requirements/build/cuda.txt and requirements/cuda.txt, so **2.14.0+cu126 is accepted** (a local version satisfies ==2.14.0).
- **torch companions:** torchaudio==2.11.0 and torchvision==0.29.0.
- **triton:** 3.8.0, via torch (vLLM itself doesn't pin it).
- **flashinfer-python==0.7.2.post1:** its backend floor is raised to SM80 (flashinfer.py:696), so it is unused on sm_70.
- **xformers:** no longer a dependency.
- **CUDA 13–only extras** (`nvidia-cutlass-dsl[cu13]==4.8.0`, `humming-kernels[cu13]==0.1.19`, `quack-kernels`, `tilelang==0.1.12`, `tokenspeed-mla`, `PyNvVideoCodec`): drop or stub them, since they are unusable on sm_70/cu12.
- **Others:** numba 0.65.0, apache-tvm-ffi 0.1.11, nvidia-cudnn-frontend>=1.31, nvtx 0.2.15, fastsafetensors, instanttensor.
- **Build tools:** cmake>=3.26.1, setuptools>=78.1.1,<81, setuptools-scm>=8, setuptools-rust>=1.9 (there is a Rust component: requirements/build/rust.txt), protobuf constraints.
- **Python:** requires-python is >=3.11,<3.15.
- **kv_connectors_cu12.txt:** lmcache==0.5.6 (LMCache cu129 release), cuda-python>=12,<13, cupy-cuda12x, nixl==1.5.0.
- **Attention on sm_70:** only TRITON_ATTN (`supports_compute_capability → True`, triton_attn.py:405). FLASH_ATTN needs ≥8.0 (flash_attn.py:458).
- **bf16:** platforms/cuda.py:262 already drops bf16 for Pascal/Volta/Turing.


### LMCache
- **lmcache 0.5.6, built from the sdist.** The PyPI wheels are a CUDA 13 build (torch 2.13, archs 7.5+) with no sm_70 code.
  - sdist: https://files.pythonhosted.org/packages/84/25/3ccd46844c94e08897aa8d72716a538c07e4583410d84dbc6cd12cb60dc1/lmcache-0.5.6.tar.gz
  - sha256: `d6951e60ce8818a76aa42d0866a4ba85f1c95cd919355fff30ff24dcc3243f1e`
- **Build settings:** `LMCACHE_CUDA_MAJOR=12`, `TORCH_CUDA_ARCH_LIST=7.0`, `--no-build-isolation`, against torch 2.14.0+cu126.
- **Build:** `nix/lmcache-build.nix` (`packages.lmcache`): `BUILD_WITH_CUDA=1`, `setup.py build_ext --inplace`, patches from `nix/patches/lmcache`. Not built: the gRPC stubs (`build_py` needs grpcio-tools 1.78, which needs protobuf<7; only `--transport grpc` uses them), `rust/raw_block`, NIXL/Mooncake/Aerospike/HF3FS.
- **Runtime deps:** vLLM's lock plus the `lmcache` group in `nix/pyenv`: sortedcontainers 2.4.0, cupy-cuda12x 14.2.0, opentelemetry-exporter-prometheus 0.66b2. Not added (unused on our paths): aiofile(s), awscrt, redis, google-*, cufile-python, grpcio-observability, cuda-python (cuda-bindings 12.9 is already locked).
- **Constraints:** requires-python <3.14. Five locked versions are above LMCache's declared caps; none breaks an import on our paths, and no runtime-deps check runs (LMCache isn't a resolved lock entry; vLLM reads only its version):

  | package | LMCache's cap | locked |
  |---|---|---|
  | numpy | `<=2.2.6` | 2.3.5 |
  | protobuf | `<7` | 7.36.2 |
  | opentelemetry-api | `<=1.40.0` | 1.45.2 |
  | prometheus_client | `<=0.24.1` | 0.26.0 |
  | opentelemetry-exporter-prometheus | `<=0.61b0` | 0.66b2 |

## Build-time fetches (all FetchContent, all fetched unconditionally for CUDA builds; the arch gate only skips compiling. The Nix sandbox has no network, so feed every one via its env var or patch out the include)
Two hashes per dependency:
- **without submodules:** `nix hash path` of a `git archive` export;
- **with submodules:** `fetchSubmodules = true`, reproduced with `git submodule update --init --recursive`, `.git` stripped, then `nix hash path`. This matches `nix/deps.nix`.

FetchContent's default recursively inits submodules. DeepSelect and qutlass have a cutlass submodule that we don't fetch: DeepSelect only uses it for 10.0f (nvcc ≥ 12.9), and qutlass prefers vLLM's cutlass. Once patch 0001 (`build`, PR #15) lands, only cutlass will be fetched; the other rows will then document what `VLLM_SM70_TRIM_EXTERNALS=0` brings back. Until then, all ten are fetched.
| dep | rev | env override | subm. | Nix hash (no submodules) | Nix hash (with submodules) | sm_70? |
|---|---|---|---|---|---|---|
| cutlass | v4.7.1 = cb4247394dd82148787aed73e5dc7cef33cbf862 (CMakeLists.txt:512) | VLLM_CUTLASS_SRC_DIR | 0 | sha256-z/jASQGRj9fYwgyJKvzWgowG7j2zNKb3xeFUzYwwLQA= | same (no submodules) | Header-only; needed by core _C (scaled_mm/machete sources are arch-gated, but the include path is always used). Keep |
| vllm-flash-attn | b273a02621d3c94bc747df67204f591b67f38fae | VLLM_FLASH_ATTN_SRC_DIR | 3 (cutlass, composable_kernel, aiter) | sha256-8qGtJl18nIo8Kd/jPYlDK8K83uuu4W1mZ0x0oTvXU8w= | `sha256-0akJwKOdunEZeU289NbdvYH/3QXqeiMCRLdmId4ONRY=` | No: FA's own CUDA_SUPPORTED_ARCHS is "8.0;8.6;8.9;9.0(+10/12)" (its CMakeLists.txt:25), and FA2 is "8.0+PTX". Its arch list ends up empty but doesn't FATAL. Patch out CMakeLists.txt:1713 or feed the source; it builds nothing for 7.0 |
| triton_kernels | triton-lang/triton v3.5.1 = 0add68262ab0a2e33b84524346cb27cbb2787356 (SOURCE_SUBDIR python/triton_kernels) | TRITON_KERNELS_SRC_DIR (point it at the python/triton_kernels/triton_kernels dir) | 0 | sha256-dyNRtS1qtU8C/iAf0Udt/1VgtKGSvng1+r2BtvT9RB4= | same (no submodules) | Python-only (MoE matmul_ogs); mostly Hopper paths. Can feed it or patch out CMakeLists.txt:1686-1688. **Note the v3.5.1 vs runtime triton 3.8 mismatch** |
| DeepGEMM | vllm-project/DeepGEMM 0820030756945b2daec208e8e8255268360f337d (deepgemm.cmake:40) | DEEPGEMM_SRC_DIR | 2 | sha256-xBqbHoQw0P0N3zPF1H4dYEpuB6wMfeC3N78ts/Sgd1U= | `sha256-Qbup5RiPYdbREooAt8m2Gq5C/B8EB4ywkTJjFJIxRbU=` | No (9.0a/10.x/12.x only) |
| DeepSelect | d96d33afe1fab0d6066da49cdc91e64c2bee65ea | DEEPSELECT_SRC_DIR | 1 | sha256-7+lmKQH08v9dhQ9KaZtcs25haMdkwKOEk5noS9/GiB8= | not fetched; see above | No (10.0f, nvcc≥12.9) |
| FMHA_SM100 (MSA) | 968061ab16cb14991a5c7d4c0e8eabf531be0dcf | FMHA_SM100_SRC_DIR | 1 | sha256-gEkRT++7MDBokuqzDHp4F/e3yCXSt8xuUz0vmJ5CN60= | `sha256-KDDjfMXys47XeYTMS3eQC+qa257XU88PikoEsMaCO9g=` | No (SM100 python/JIT; always populated and installed) |
| FlashMLA | e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b | FLASH_MLA_SRC_DIR | 1 | sha256-HN0iqHMYSXYb2XZ5/5QWYdZ9/Hi6DU9XdboPwFao9LU= | `sha256-ffqgYT/L2cYs/PY1QZPsSskPz56Rw3F5XAvaUHbytK8=` | No (9.0a/10.x) |
| FlashKDA | 17a037d98da546deb4591e967cf961a43c034d8b | FLASH_KDA_SRC_DIR | 1 | sha256-3+y7I5QNs+Vr/vDeBhqyy2WwZvQEo8JiB8Jskc3zuvs= | `sha256-1IYJcMN70jY9gT/OWNhyEZswc1h1+b+HsSbZ8uLPQSQ=` | No (9.0a/10.x/12.x) |
| qutlass | IST-DASLab/qutlass e74319e3405ce6d71965732880f5dc1f52371f64 (qutlass.cmake:25) | QUTLASS_SRC_DIR | 1 | sha256-Gzl3KuYXXLXMrVciEYrBPu1FH2cplGUPTFpWzFfUmMo= | not fetched; see above | No (10.x/12.x) |
| tml-fa4 | 75765e76a9c2c012c1f6ecd64577eb646eb4d303 | TML_FA4_SRC_DIR | 0 | sha256-zw19GAgsA0+5XzJNGLInIMN4GMRXqJsLkiVViRkv6fI= | same (no submodules) | No (CuteDSL FA4, Blackwell) |
| oneDNN | 9c5be1cc… / v3.13 | – | – | – | – | CPU build only, not pulled for CUDA |

How to disable them:
- Simplest: patch CMakeLists.txt:1686-1714 to include only cutlass (plus triton_kernels if wanted).
- Otherwise point each *_SRC_DIR at a fetchFromGitHub output; all ten respect the env var.
- Everything except cutlass is dead weight on sm_70.

**Lines that drop arch 7.0 for nvcc ≥ 12.8:** CMakeLists.txt:119-135.
- 13.4: `7.5;8.0;…` (:123)
- 13.0: `7.5;…` (:129)
- **≥12.8: `7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1` (:130-132)**
- else (<12.8): `7.0;7.5;8.0;8.6;8.7;8.9;9.0` (:134)

They are applied by `cuda_archs_loose_intersection(CUDA_ARCHS "${CUDA_SUPPORTED_ARCHS}" …)` at :228-237, which hits FATAL_ERROR if the result is empty. **With nvcc 12.6, 7.0 survives unpatched.** The vllm-flash-attn CMakeLists.txt:25 list never includes 7.0.

## Risks
1. **cuDNN:** nixpkgs cudnn (9.22) refuses sm_70. Use torch's wheel cudnn 9.10.2.21. Any nixpkgs package pulling `cudaPackages.cudnn` with caps 7.0 fails eval (that includes nixpkgs `python3Packages.torch`, which is also 2.11 on CUDA 12.9 by default, so don't use it).
2. **GCC window is exactly 13:** vLLM needs ≥13 and nvcc 12.6 needs ≤13. backendStdenv already gives 13.4.0. Don't let gcc15 (the default stdenv) reach nvcc.
3. **Upstream is CUDA-13-first:** v0.33.0 release wheels are CUDA 13.0 only (12.9 dropped), and several deps are `[cu13]`. Expect stale cu12 code paths. Also bundled triton_kernels v3.5.1 vs runtime triton 3.8.0, and triton's ptxas 12.9 vs system nvcc 12.6. The 12.9 ptxas still targets sm_70; set TRITON_PTXAS_PATH if the driver-580 JIT needs a match.
4. **Attention:** the only sm_70-eligible backend is TRITON_ATTN, and Triton has no tensor-core MMA lowering on Volta. Speed parity with today's fork (1Cat custom sm70 kernels) depends on the patch series. FA2/FlashInfer are ≥SM80, and PR #55380 (Turing flashinfer) is open.
