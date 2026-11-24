# Pins

A **pin** fixes the exact version of something the build uses, so every
build gets the same result. Every pin is declared in code; read the versions
and hashes there.

| pin | where |
|---|---|
| upstream vLLM tag and commit | `patches/BASE`, `nix/source.nix`, `BASE_COMMIT` in `.github/workflows/ci.yml` |
| FetchContent sources | `nix/deps.nix` |
| torch, Triton and the Python lock | `nix/pyenv/pyproject.toml`, `nix/pyenv/uv.lock`; `TORCH_VERSION` in `ci.yml` |
| CUDA toolkit and host compiler | `nix/package.nix` |
| nixpkgs and flake inputs | `flake.lock` |
| LMCache sdist | `nix/lmcache-build.nix` |

## Rules

These keep the build working on Volta ([volta.md](volta.md)).

- CUDA stays on 12.x: CUDA 13 cannot target sm_70, and nvcc ≥ 12.8 drops
  arch 7.0 from vLLM's default list.
- torch stays on a `+cu126` wheel that ships sm_70 SASS. Check a new wheel
  for `sm_70` cubins in `torch/lib/libtorch_cuda.so` before moving.
- The host compiler is gcc 13: vLLM needs ≥ 13 and nvcc 12.6 needs ≤ 13.
  Never let the default stdenv's gcc reach nvcc.
- cuDNN comes from torch's `nvidia-cudnn-cu12` wheel. Nothing may pull
  nixpkgs `cudaPackages.cudnn` (or nixpkgs `python3Packages.torch`): it
  refuses capability 7.0 at eval.
- Run models with `--dtype half`; Volta has no bf16.
- Every FetchContent source is fed through its `*_SRC_DIR` variable in
  `nix/deps.nix`; the sandbox has no network.
- Hosts set `inputs.nixpkgs.follows = "nixpkgs"` on this flake.

## Moving to a new vLLM release

1. Rebase the `sm70` working branch onto the new tag
   ([patch-header.md](patch-header.md#working-on-patches)).
2. Set the tag in `patches/BASE`, and its commit in `nix/source.nix` and in
   `BASE_COMMIT` (`ci.yml`), in one PR. Take the hash from
   `nix flake prefetch github:vllm-project/vllm/<tag>`.
3. Match `nix/deps.nix` to the revisions in the new tag's `CMakeLists.txt`
   and `cmake/external_projects/`; get each hash with `nix flake prefetch`.
4. Update the torch and Triton pins in `nix/pyenv/pyproject.toml` to what
   the tag requires, then re-lock: `uv lock --python 3.13` in `nix/pyenv/`.
   Update `TORCH_VERSION` in `ci.yml`.
5. `nix build .#vllm`, then run the [GPU gate](checks.md#gpu-gate).

## LMCache

- LMCache is built from the PyPI sdist (`packages.lmcache`): its wheels are
  a CUDA 13 build with no sm_70 code.
- Build with `LMCACHE_CUDA_MAJOR=12` and `TORCH_CUDA_ARCH_LIST=7.0` against
  the same torch as vLLM.
- Its runtime deps that vLLM's lock lacks go in the `lmcache` group of
  `nix/pyenv/pyproject.toml`.
- After a bump, the `lmcache-patches-apply` flake check must pass and the
  [LMCache build check](checks.md#lmcache-build-check) must pass on the
  serving host.
