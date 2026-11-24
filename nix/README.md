# nix/

The flake's packages, NixOS modules (`services.vllm-sm70`,
`services.lmcache-sm70`), the Flint checkpoint and the flake checks. Outputs
are in `../flake.nix`; each file's header says what it holds.

How to use the flake from a host and operate the modules (model overrides,
kill switches, Inkling 2 serve rules, the TurboMind GEMM LUT, LMCache, the
Flint checkpoint store path): [docs/nix.md](../docs/nix.md).
