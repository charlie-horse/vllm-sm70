# vllm-sm70

[vLLM](https://github.com/vllm-project/vllm) is a program that runs large
language models and answers requests to them. Its current releases no
longer support NVIDIA's **Volta** GPUs (the chip in the Tesla V100, called
`sm_70` by NVIDIA's tools). This repository builds an unchanged vLLM release
so that it runs well on Volta again.

It has three parts:

- **`sm70/`, the plugin.** GPU code (kernels) written for Volta, plus a Python
  package, `vllm_sm70`, that vLLM loads at start-up. It adds fast paths for
  Volta and the models we run.
- **`patches/`, small edits to vLLM itself.** Only where vLLM has no place
  for a plugin to hook in. Each edit just hands control to the plugin.
- **`nix/` and `flake.nix`, the build.** A [Nix](https://nixos.org) flake
  that fetches the pinned vLLM release, applies the patches, compiles
  everything for Volta and packages it. It also ships NixOS modules that run
  vLLM and its cache (LMCache) as services.

Start here: [docs/README.md](docs/README.md). The chip itself, and what it
can and can't do: [docs/volta.md](docs/volta.md).

## License

Apache-2.0, the same as vLLM. Each patch is a derivative of vLLM and keeps
its license.
