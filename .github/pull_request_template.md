<!--
Title: <area>(<node>): <imperative summary>, e.g. "kernels(K03): TurboMind fp16/FP8/NVFP4 dense + MoE ops".
Fill in every header field. See docs/patch-header.md for the rules and docs/roadmap.md for node IDs.
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

Roadmap node(s):
Depends on (PRs / nodes):

## Evidence

<!-- Commands and output excerpts. GPU results: host, GPU, model, numbers before/after. -->

## Checklist

- [ ] `scripts/regen-patches.sh --check` passes (if `patches/` changed)
- [ ] Upstream edits are text-only, minimal, and no-ops without the plugin
- [ ] The kill switch restores the upstream path, and is tested
- [ ] No secrets, credentials, IPs or tailnet names in code, logs or this description (host names like r440 or c4130 are fine)
