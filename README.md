# vllm-sm70

A Nix flake that builds upstream [vLLM](https://github.com/vllm-project/vllm) for NVIDIA Volta GPUs (SM70, e.g. Tesla V100). It does this with two pieces on top of a pinned release tag: an out-of-tree plugin (`sm70/`: CUDA kernels plus the `vllm_sm70` package) and a small, ordered series of upstream-file patches (`patches/`) that add the hooks the plugin uses.

Status: Phase 1 spike (branch `phase1-spike`): unpatched v0.33.0 for sm_70, TRITON_ATTN only. Not yet built.

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
| `nix/` | `source.nix` (tag + `patches/series`), `deps.nix` (CMake FetchContent sources), `package.nix` (uv2nix wheel env, sm_70 build, wrapper), `pyenv/` (`pyproject.toml` + `uv.lock`, re-lock with `uv lock --python 3.13`) |
| `sm70/` | the plugin: `csrc/` and CMake for the `vllm_sm70._sm70_C` extension, and the `vllm_sm70` Python package |
| `patches/` | upstream-file edits only: `BASE` (upstream tag), `series` (apply order), `NNNN-*.patch` |
| `scripts/regen-patches.sh` | regenerates `patches/` from a vLLM working branch, or checks that the series applies |
| `tests/` | sm70 checks |
| `docs/` | `pins.md` (locked pins), `patch-header.md` (change format), `roadmap.md` (ordered plan), `phase1-gpu-runbook.md` (c4130 test) |
| `.github/` | PR template and CI |

## Contributing

New code goes in `sm70/`, as one PR per feature, with the header fields from `docs/patch-header.md` in the PR description. Only edit an upstream vLLM file when a hook is unavoidable. Those edits go through the patch workflow below. All work goes on a branch and a PR to `main`; nobody merges their own PR.

## Working on patches

1. Clone upstream vLLM and create the working branch on the base tag: `git switch -c sm70 v0.33.0`.
2. Commit each change with the header in `docs/patch-header.md`.
3. Regenerate the series: `scripts/regen-patches.sh ~/src/vllm`.
4. Check that it applies to a clean tag: `scripts/regen-patches.sh --check ~/src/vllm`.

To move to a new upstream release, rebase `sm70` onto the new tag, then repeat steps 3–4.

## License

Apache-2.0, the same as vLLM. Each patch is a derivative of vLLM and keeps its license.
