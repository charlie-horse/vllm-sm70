# vllm-sm70

A Nix flake that builds upstream [vLLM](https://github.com/vllm-project/vllm)
for NVIDIA Volta GPUs (SM70, e.g. Tesla V100): a pinned release tag, an
out-of-tree plugin (`sm70/`: CUDA kernels plus the `vllm_sm70` package) and a
small, ordered series of upstream-file patches (`patches/`) that add the hooks
the plugin uses. It also ships NixOS modules that serve vLLM and LMCache.

- Docs: [docs/README.md](docs/README.md).
- Rules for contributors and agents: [CLAUDE.md](CLAUDE.md).

## License

Apache-2.0, the same as vLLM. Each patch is a derivative of vLLM and keeps
its license.
