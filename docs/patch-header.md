# Changes: format, kill switches, patches

The repo carries two kinds of change. Both use the same six header fields.

| kind | where it lives | where the header goes |
|---|---|---|
| **Upstream edit**: a change to a file that exists in vLLM | `patches/`, one file per commit on the vLLM `sm70` working branch | the commit message (checked by `scripts/regen-patches.sh`) |
| **sm70 feature**: new code that vLLM doesn't have | `sm70/`: CUDA sources and CMake for the `vllm_sm70._sm70_C` extension, plus the `vllm_sm70` plugin package | the PR description (`.github/pull_request_template.md`) |

Anything that can live in `sm70/` lives there. Upstream edits are hooks that
call into `vllm_sm70`, so a rebase onto a new vLLM release only carries them.

## Header fields

```
<area>(<scope>): <imperative summary, under 72 chars>

What: <one or two sentences on what the change does>
Why: <what Volta / CUDA 12.6 needs; link the upstream line or issue>
Upstream-files: <comma-separated upstream paths touched, or "none (sm70/ only)">
Kill-switch: <VLLM_SM70_<THING>=0, or "none (build-time)", or "hook returns None -> upstream">
Tests: <test that proves it, e.g. sm70/tests/plugin/test_u1_hooks.py or "build only">
Upstream-status: <not-upstreamable | candidate | submitted: <PR url> | backport: <commit>>
```

`(<scope>)` is optional. For a feature PR, the PR title is the summary line
and the fields go in the template's header block.

## Rules

- Each patch or PR does one thing and builds on its own. Split it rather
  than writing "and" in the summary.
- Models, quant methods and KV connectors register through vLLM's plugin
  APIs (`additional_config`, `register_quantization_config`,
  `register_model`, `LINEAR_METHOD_BUILDERS`, `FusedMoEMethodCls`,
  `kv_connector_module_path`) and touch no upstream file.
- Upstream hooks are no-ops when the plugin isn't loaded: no provider means
  upstream behaviour.
- Touch as few upstream lines as possible. Never reformat upstream code.
- Patches are text only: no binary hunks, renames or mode changes. The Nix
  build applies the series with `patch -p1 --fuzz=0`, and `--check` uses
  `git am` without `--3way`, so a patch with stale context fails in both.

## Kill switches

- Every behaviour change gets a runtime kill switch `VLLM_SM70_<THING>`,
  declared in its area's section of `sm70/vllm_sm70/envs.py` with a comment
  saying what `0` does.
- Only `0` acts, and only turns the feature off, back to the upstream path.
  The environment never turns a feature on, except an `opt_in` switch (only
  `1` acts) or a `route_switch` (`0` and `1` force a route).
- A switch is read where the path is chosen, normally in the plugin's model
  wiring. Kernel ops in `_sm70_C` carry no switch; callers feature-detect an
  optional op and take the plain path when it is missing.
- Defaults and tuning live in typed config (`Sm70Config`, set with
  `--additional-config '{"sm70": {...}}'`), not in env variables.
- A new kill switch that changes kernels joins the switch envelope of the
  [GPU gate](checks.md#gpu-gate).

## Working on patches

`patches/BASE` names the upstream tag, `patches/series` the apply order.

1. Clone upstream vLLM and create the working branch on the base tag:
   `git switch -c sm70 $(cat patches/BASE)`.
2. Commit each change with the header above.
3. Regenerate the series: `scripts/regen-patches.sh ~/src/vllm`.
4. Check that it applies to a clean tag:
   `scripts/regen-patches.sh --check ~/src/vllm`.

To move to a new upstream release, rebase `sm70` onto the new tag, then
repeat steps 3–4 and follow [pins.md](pins.md).

### Series order

The ranges set the commit order on the branch; `git format-patch` numbers
the files sequentially.

| range | area |
|---|---|
| 0001–0099 | `build`: CMake arch list, FetchContent trimming, CUDA-13-only deps |
| 0100–0199 | `platform`: hooks in upstream files that `sm70/` plugs into |
| 0500–0699 | `models`: none; models register through the plugin |
| 0700–0799 | `kv`: none; LMCache loads through `kv_connector_module_path` |
| 0900–0999 | `misc`: logging, docs |

## Example (upstream edit)

```
platform: add unquantized-GEMM and warmup hooks for out-of-tree kernels

What: dispatch_unquantized_gemm() asks current_platform for a GEMM first and
  falls back to upstream when it returns None; same for kernel warmup.
Why: lets vllm_sm70 route fp16 GEMMs to TurboMind on Volta without editing
  each layer (layers/utils.py:682).
Upstream-files: vllm/model_executor/layers/utils.py, vllm/platforms/interface.py,
  vllm/model_executor/warmup/kernel_warmup.py
Kill-switch: hook returns None -> upstream (no plugin)
Tests: sm70/tests/plugin/test_u1_hooks.py (CPU)
Upstream-status: candidate
```
