<!--
Title: <area>(<scope>): <imperative summary>, e.g. "kernels: TurboMind fp16/FP8/NVFP4 dense + MoE ops".
Fill in every header field. See docs/patch-header.md for the rules.
-->

## Header

```
What:
Why:
Upstream-files:
Kill-switch:
Tests:
Upstream-status:
```

Depends on (PRs):

## Evidence

<!-- Commands and output excerpts. GPU results: host, GPU, model, numbers before/after. -->

## Checklist

- [ ] `scripts/regen-patches.sh --check` passes (if `patches/` changed)
- [ ] Upstream edits are text-only, minimal, and no-ops without the plugin
- [ ] The kill switch restores the upstream path, and is tested
- [ ] Docs follow the CLAUDE.md documentation policy and changed with the code
- [ ] No secrets, credentials, IPs or tailnet names in code, logs or this description
