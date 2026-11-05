# Change format: patches and sm70/ features

The repo carries two kinds of change. Both use the same six header fields.

| kind | where it lives | where the header goes | example |
|---|---|---|---|
| **Upstream edit**: a change to a file that exists in vLLM | `patches/`, one file per commit on the vLLM `sm70` working branch | the commit message (checked by `scripts/regen-patches.sh`) | U1 platform GEMM hook in `vllm/model_executor/layers/utils.py` |
| **sm70 feature**: new code that vLLM doesn't have | `sm70/`: CUDA sources and CMake for the `vllm_sm70._sm70_C` extension, plus the `vllm_sm70` plugin package | the PR description (`.github/pull_request_template.md`) | K03 TurboMind dense ops |

Keep upstream edits tiny. They are hooks that call into `vllm_sm70`, so a rebase onto a new vLLM release only has to carry them. Anything that can live in `sm70/` must live there.

## Header fields

```
<area>: <imperative summary, under 72 chars>

What: <one or two sentences on what the change does>
Why: <why Volta / CUDA 12.6 / V100 TP2 needs it; link the upstream line or issue>
Upstream-files: <comma-separated upstream paths touched, or "none (sm70/ only)">
Kill-switch: <VLLM_SM70_<THING>=0, or "none (build-time)", or "hook returns None -> upstream">
Tests: <exit test that proves it, e.g. tests/test_plugin_load.py or "build only">
Upstream-status: <not-upstreamable | candidate | submitted: <PR url> | backport: <commit>>
```

For a feature PR, the PR title is the summary line and the fields go in the template's header block. A PR may add a roadmap node ID (e.g. `K03`, `M5`, `U1`) to the summary: `kernels(K03): TurboMind fp16/FP8/NVFP4 dense + MoE ops`.

## Patch areas, in series order

`patches/` holds only upstream edits:

| range | area | owner | roadmap nodes |
|---|---|---|---|
| 0001–0099 | `build`: CMake arch list, FetchContent trimming, CUDA-13-only deps | Penny | – |
| 0100–0199 | `platform`: hooks in upstream files that sm70/ plugs into | Kai (U1 co-owned with Mara) | U1–U5 |
| 0500–0699 | `models`: empty. Models and quant register through the plugin | Mara | – |
| 0700–0799 | `kv`: empty. LMCache loads through `kv_connector_module_path` with no vLLM-side patch | Theo | – |
| 0900–0999 | `misc`: logging, docs | anyone | – |

`git format-patch` numbers the files sequentially. The ranges set the commit order on the branch, not the file numbers. See `docs/roadmap.md` for the full dependency graph.

## Rules

- Each patch or PR does one thing and builds on its own. Split it rather than writing "and" in the summary.
- Use a runtime kill switch wherever behaviour changes, named `VLLM_SM70_<THING>=0`. Each switch only turns its feature off, back to the upstream path. Defaults live in typed config (`Sm70Config`), not in env vars.
- Upstream hooks must be no-ops when the plugin isn't loaded: no provider means upstream behaviour.
- Touch as few upstream lines as possible. Never reformat upstream code.
- Patches are text only: no binary hunks, renames, or mode changes. The Nix build applies the series with plain `patch -p1 --fuzz=0`, and `--check` uses `git am` without `--3way`, so a patch with stale context fails in both places.

## Example (upstream edit)

```
platform: add unquantized-GEMM and warmup hooks for out-of-tree kernels

What: dispatch_unquantized_gemm() asks current_platform for a GEMM first and
  falls back to upstream when it returns None; same for kernel warmup.
Why: lets vllm_sm70 route fp16 GEMMs to TurboMind on Volta without editing
  each layer (layers/utils.py:682).
Upstream-files: vllm/model_executor/layers/utils.py, vllm/platforms/interface.py,
  vllm/model_executor/warmup/kernel_warmup.py
Kill-switch: hook returns None -> upstream (no plugin, or VLLM_SM70_TURBOMIND=0)
Tests: tests/test_platform_hooks.py (CPU, fake platform); GPU: K03 exit test
Upstream-status: candidate
```
