# Docs

Procedures and rules for building, changing and serving vllm-sm70. What is
built and how it is configured: read the Nix and Python source. Rules for
these docs: [CLAUDE.md](../CLAUDE.md#documentation-policy).

| Doc | Procedures and rules for |
| --- | --- |
| [patch-header.md](patch-header.md) | Change format, kill switches, the patch workflow and series order |
| [pins.md](pins.md) | Where pins live, pin rules, moving to a new vLLM release, LMCache |
| [nix.md](nix.md) | Using the flake from a host, serving models, the TurboMind GEMM LUT, LMCache, the Flint checkpoint |
| [checks.md](checks.md) | GPU tests on c4130, the GPU gate, LMCache checks |
| [release.md](release.md) | Moving the `latest` branch |
