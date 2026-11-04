# vllm-sm70

A Nix flake that builds upstream [vLLM](https://github.com/vllm-project/vllm) for NVIDIA Volta GPUs (SM70, e.g. Tesla V100). It does this by applying a small, ordered patch series on top of a pinned release tag.

Status: work in progress. Nothing here builds yet.

## Pins

| component | pin |
|---|---|
| vLLM | `v0.33.0` (`4ba787ec9959e566d3b587a796dff07673324da5`) |
| torch | 2.14.0+cu126 wheel (the last release with sm_70 SASS) |
| Triton | 3.8.0 |
| CUDA | 12.6 (CUDA 13 cannot target sm_70) |
| nixpkgs | `nixos-26.05` (`7dcdd87373d72e7a2173e4db02e20d3d7f495cfd`) |
| host compiler | gcc 13 (vLLM needs ≥13; nvcc 12.6 needs ≤13) |

Full pins, hashes and evidence are in [docs/pins.md](docs/pins.md). Volta has no bf16, so run models with `--dtype half`.

## Layout

| path | contents |
|---|---|
| `flake.nix` | flake entry point |
| `nix/` | derivations and NixOS modules |
| `patches/` | the patch series applied on top of the vLLM tag |
| `tests/` | sm70 checks |
| `docs/` | `pins.md` (locked pins and evidence) |

## License

Apache-2.0, the same as vLLM. Each patch is a derivative of vLLM and keeps its license.
