# Docs

These docs explain the code in this repository, with only the hardware
detail needed to use it.
Anything the code states (versions, options, defaults) is read from the
code itself; the docs point to where. Rules for writing them:
[CLAUDE.md](../CLAUDE.md#documentation-policy).

| Doc | What it explains |
| --- | --- |
| [volta.md](volta.md) | The Volta chip: what it can and can't do, and what the code does about it |
| [patch-header.md](patch-header.md) | How a change is made: plugin code vs. vLLM patches, kill switches, the patch workflow |
| [pins.md](pins.md) | Which versions are fixed, where, and how to move to a new vLLM release |
| [nix.md](nix.md) | Using the flake and its NixOS modules, model settings, kill switches, Inkling 2 rules, the GEMM tuning table, LMCache, the Flint weights |
| [checks.md](checks.md) | Testing the code on a Volta GPU, including the GPU gate and the Inkling 2 gate |
| [release.md](release.md) | The `latest` branch and when it moves |
