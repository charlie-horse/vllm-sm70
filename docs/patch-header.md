# Patch header template

Every commit on the `sm70` working branch becomes one file in `patches/`. `scripts/regen-patches.sh` refuses to regenerate the series if a commit is missing any of the fields below.

```
<area>: <imperative summary, under 72 chars>

What: <one or two sentences on what the patch changes>
Why: <why Volta / CUDA 12.6 / V100 TP2 needs it; link the upstream line or issue>
Upstream-files: <comma-separated paths the patch touches>
Kill-switch: <env var or CLI flag that turns it off at runtime, or "none (build-time)">
Tests: <test or check that covers it, e.g. tests/sm70/test_xyz.py or "build only">
Upstream-status: <not-upstreamable | candidate | submitted: <PR url> | backport: <commit>>
```

## Areas, in series order

| range | area | owner |
|---|---|---|
| 0001–0099 | `build`: CMake arch list, FetchContent trimming, CUDA-13-only deps | Penny |
| 0100–0199 | `platform`: capability checks, dtype (fp16-only), backend selection | Penny / Kai |
| 0200–0499 | `kernels`: sm70 attention, GEMM, MoE, quant kernels | Kai |
| 0500–0699 | `models`: Flint, Qwen3.6/3.8 model and config fixes | Mara |
| 0700–0799 | `kv`: LMCache / KV-connector glue | Theo |
| 0900–0999 | `misc`: logging, docs, tests | anyone |

`git format-patch` numbers the files sequentially. The ranges set the commit order on the branch, not the file numbers.

## Rules

- Each patch should do one thing and build on its own. Split a patch rather than writing "and" in its summary.
- Use a runtime kill switch wherever behaviour changes. Name it `VLLM_SM70_<THING>=0`.
- Touch as few upstream files as possible. Put new code in new files where you can, since rebases are cheaper that way.
- Never reformat upstream code.

## Example

```
build: keep sm_70 in CUDA_SUPPORTED_ARCHS for nvcc 12.x

What: Adds 7.0 to the nvcc>=12.8 branch of CUDA_SUPPORTED_ARCHS.
Why: CMakeLists.txt:130-132 drops 7.0 for nvcc>=12.8; we build with 12.6
  today, but this guards against a toolkit bump silently dropping Volta.
Upstream-files: CMakeLists.txt
Kill-switch: none (build-time)
Tests: build only; tests/sm70/test_arch.py checks the cubin list
Upstream-status: not-upstreamable
```
