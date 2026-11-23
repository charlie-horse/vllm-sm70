# Nix: outputs, modules, checkpoint

## Outputs (the drop-in contract for hosts that used charlie-horse/vLLM)

| Output | What |
|---|---|
| `nixosModules.vllm` (= `default`) | `services.vllm-sm70`: `vllm` + `vllm-model-fetch` units, the model catalog. Imports `nixosModules.lmcache`. |
| `nixosModules.lmcache` | `services.lmcache-sm70`: `vllm-lmcache`, `vllm-lmcache-ui`, `vllm-lmcache-l2-prune`. |
| `packages.x86_64-linux.vllm` (= `default`) | the wrapped vLLM env (`bin/vllm`, `bin/hf`, `bin/lmcache`). |
| `packages.x86_64-linux.lmcache` | LMCache 0.5.6 built from the sdist for sm_70 (`lmcache-build.nix`); the `vllm` wrapper has it on PYTHONPATH. |
| `packages.x86_64-linux.flint-checkpoint` | Flint's converted weights (fixed-output, 23 GB). |
| `checks.x86_64-linux.*` | checkpoint outPath guard, one NixOS eval per built-in model, `lmcache-patches-apply` (the LMCache patches apply to the sdist the build uses). |

The option surface of both modules is unchanged from the old fork (cf9492a),
including the `c4130.vllm.*` rename aliases. Packages are built with the
host's `pkgs` (`packages-module.nix`), once for both modules.

## Files

| File | Owner | What |
|---|---|---|
| `module.nix` | integration | `services.vllm-sm70` and the built-in models |
| `lmcache.nix` | integration | `services.lmcache-sm70` |
| `packages-module.nix` | integration | `_module.args.vllmSm70Packages` |
| `packages.nix` | integration | `{ vllm, lmcacheUi, flintCheckpoint }` |
| `vllm.nix` | build | `{ vllm, lmcacheUi }` (contract in the file) |
| `lmcache-build.nix` | build | LMCache 0.5.6 from the sdist for sm_70, on the wrapper's PYTHONPATH (`packages.lmcache`) |
| `patches/lmcache/` | integration | LMCache sdist patches (list in `default.nix`), applied by `lmcache-build.nix` |
| `checkpoint.nix` | integration | `flintCheckpoint` (`requireFile`, fixed output; no build recipe) |
| `checks.nix` | integration | flake checks |

## Flint checkpoint

`flintCheckpoint` is fixed-output (`name = "flint-nvfp4-modelopt"`, recursive
sha256 `sha256-HyBUnISsTQXSKmmOft+/fNzrhFLwBJ6imMCa7YRAwoQ=`), so its path is
always `/nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt`,
the same as the old flake's. It is a `requireFile`: Nix substitutes it from a
store or cache, and the build fails if it is in none. This flake has no
recipe for it and no longer depends on the fork (charlie-horse/vLLM).

If the path is missing from every store:

1. Copy it from a host that still has it (c4130, r440):
   `nix copy --from ssh://<host> /nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt`.
2. Otherwise regenerate it with the in-repo copy of the fork's converter,
   `sm70/tools/flint/convert_gguf.py` (+ `gguf_nvfp4.py`), copied unchanged
   from the fork at cf9492a (`cf9492a01eeca0cac731c0fa62015e2c85aaf272`; the
   two files are identical at the fork's converter pin f66728d).
   `sm70/tools/flint/PROVENANCE.md` lists the pinned GGUF and `--hf-config`
   files (Hugging Face revs and sha256s) and what the converter needs: the
   `.#vllm` Python plus `gguf==0.19.0`, which that env does not ship. Put the
   GGUF at `$gguf`, the config files in a directory `$hfConfig`, and set
   `$python` to a Python that has those packages; then, from the repo root,
   convert (~15 min, ~25 GB RAM, ~23 GB output):

   ```sh
   export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
   PYTHONPATH=sm70/tools $python -m flint.convert_gguf \
     $gguf ./flint-nvfp4-modelopt --hf-config $hfConfig --device cpu
   rm -f ./flint-nvfp4-modelopt/conversion_report.json  # run metadata, not reproducible
   nix-store --add-fixed --recursive sha256 ./flint-nvfp4-modelopt
   ```

   The directory must be named `flint-nvfp4-modelopt`. `nix-store` must print
   the path above; a different path means the output differs and the module
   will not use it. A full conversion with the in-repo copy has not been run
   yet (only the fork's converter unit tests, on CPU).

## Fork-only settings left out for now

Marked `TODO(sm70-attn)` / `TODO(sm70-moe)` in `module.nix`; upstream v0.33.0
rejects them at startup until the patch series adds them:

- `--kernel-config.sm70_skinny_moe false` (Flint).

Attention needs no flag: the sm70 platform picks Flash-V100 (K06,
`AttentionBackendEnum.CUSTOM`) for fp16 head-128/256 layers, target
and DFlash draft alike, unless `VLLM_SM70_FLASH_V100=0`.

## Per-model serve values

A host overrides single fields of a built-in model, for example
`services.vllm-sm70.models.flint.maxNumSeqs = 20`. `maxModelLen`,
`maxNumSeqs`, `maxNumBatchedTokens` and `gpuMemoryUtilization` set
`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens` and
`--gpu-memory-utilization`; unset, the module passes 262144, 4, 8192 and
0.88. The comment above the `models` option in `module.nix` lists every field.

## Qwen serve flags

The `qwen3.6-35b-a3b` and `qwen3.8-27b` entries use the flags measured on
c4130 (Phase 3 Block F): async scheduling, the V2 model runner
(`v2ModelRunner = true` sets `VLLM_USE_V2_MODEL_RUNNER=1`, which DFlash drafts
need), Flash-V100 attention, and LMCache. A host `env` replaces the built-in
`env` as a whole (Flint's and Qwen's `CUDA_DEVICE_ORDER`/`VLLM_SKIP_P2P_CHECK`
included, so repeat them when adding a kill switch), but not `v2ModelRunner`. The fork/1Cat serve flags are an opt-in for A/B
runs (they measured 12-27% slower per step): set
`models."qwen3.8-27b".asyncScheduling = false` (plus `maxNumSeqs = 4`)
in the host, and the same for `qwen3.6-35b-a3b`. The comment above `builtinModels` in
`module.nix` has both entries.

## Inkling 2 serve flags and client rules

- Pass `--reasoning-parser inkling2_sm70` only together with
  `--tool-call-parser inkling2_sm70 --enable-auto-tool-choice`. Any other
  pairing fails at server start.
- With `--tokenizer-mode inkling2_sm70`, a client that wants one tool call per
  response sends `chat_template_kwargs: {"parallel_tool_calls": false}`. The
  top-level `parallel_tool_calls: false` only trims the response to the first
  call; the prompt does not carry the instruction.
- Without `--tokenizer-mode inkling2_sm70` (the checkpoint's Jinja template),
  request text must not contain `<|unused_200054|>` when the request has
  images; such a request is rejected.

## TurboMind small-shape tuning and the GEMM LUT

The fork timed TurboMind NVFP4/FP8 kernels for small shapes (M <= 64) at
startup (`*_TUNE_SMALL_SHAPES=1`). Here that is off by default (#77): timing
per instance made restarts, TP ranks, and eager vs graph decode pick different
kernels. The way back to the fork's speed with fixed kernels is tuning on plus
a LUT exported once, which the `services.vllm-sm70.turbomind` options set:

| option | env | default |
|---|---|---|
| `tuneSmallShapes` | `VLLM_SM70_{NVFP4,FP8}_TUNE_SMALL_SHAPES=1` | `false` (C++ default 0 applies) |
| `gemmLutPath` | `VLLM_SM70_GEMM_LUT_PATH` | `null` |
| `gemmLutExportPath` | `VLLM_SM70_GEMM_LUT_EXPORT_PATH` | `null` |
| `gemmLutStrict` | `VLLM_SM70_GEMM_LUT_STRICT=1` | `false` |

A model's `env` still wins over these. The default stays off until the A/B
below passes: no tuned+LUT run of T4c (two fresh instances bitwise equal) or
of tok/s is recorded yet.

**Export the LUT once on c4130** (as admin; outside the GPU window; the vllm
user owns `/var/lib/vllm`). A runtime drop-in needs no rebuild and is gone
after a reboot or `revert`:

```sh
sudo systemctl edit --runtime vllm
#   [Service]
#   Environment=VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=1 VLLM_SM70_FP8_TUNE_SMALL_SHAPES=1
#   Environment=VLLM_SM70_GEMM_LUT_EXPORT_PATH=/var/lib/vllm/tm-lut-flint-{arch}.bin
sudo systemctl restart vllm
# wait for "Application startup complete", then:
sudo journalctl -u vllm -b | grep -E "TurboMind pre-capture warmup|LUT exported"
ls -l /var/lib/vllm/tm-lut-flint-sm70.bin /var/lib/vllm/tm-lut-flint-sm70.bin.plan.json
sudo systemctl revert vllm && sudo systemctl restart vllm
```

`{arch}` becomes `sm70`. Then (A/B and T4c, `docs/m5-calibration-nov7.md`
"Tuning on with a LUT") start twice with tuning on, `VLLM_SM70_GEMM_LUT_PATH`
set to the same template and `VLLM_SM70_GEMM_LUT_STRICT=1`: each start must
log "tuned (measured) 0 calls on this rank" and "N/N calls covered", and the
outputs must be bitwise equal. Measure decode tok/s at n = 1/4/8/20 against the
default (tuning off). If both pass, the host sets:

```nix
services.vllm-sm70.turbomind = {
  tuneSmallShapes = true;
  gemmLutPath = "/var/lib/vllm/tm-lut-flint-{arch}.bin";
};
```

Re-export after any change to the vLLM/`_sm70_C` build, the model, TP size,
`maxNumSeqs` or the CUDA graph capture sizes (the warmup plans one call per
capture size). A LUT from another build, GPU or knob setting is refused with a
WARNING and the warmup tunes as without it (not reproducible until
re-exported); `gemmLutStrict = true` makes that a startup failure instead.

## LMCache patches

`patches/lmcache/default.nix` lists the patches for the LMCache 0.5.6 sdist;
`lmcache-build.nix` applies them (`-p1 --fuzz=0`) and
`checks.x86_64-linux.lmcache-patches-apply` checks that they apply to the same
sdist. After a build, `scripts/check-lmcache-cpu.sh` checks the result on a
host without touching the GPUs (`docs/checks.md`).

- `0001-mp-worker-heartbeat-from-registration.patch`: the MP worker starts
  pinging at KV-cache registration. Without it, a worker that sees no
  full-chunk request within `--worker-registration-grace-seconds` (1 h) of
  registering is reaped while alive and never re-registers, so every STORE is
  rejected until vLLM restarts. See `docs/upstream/lmcache-heartbeat-issue.md`.
  Verified on hardware by check 6 (`scripts/check6-lmcache-reaper.py`, steps in
  `docs/checks.md`).

`services.lmcache-sm70.extraArgs` appends extra `lmcache server` flags, e.g.
`[ "--worker-registration-grace-seconds" "604800" ]`.

Metrics: `lmcache server` doesn't start LMCache's standalone Prometheus
exporter (`run_cache_server(..., start_prometheus_http_server=False)`,
`lmcache/v1/multiprocess/http_server.py:89`), so nothing listens on :9090.
Metrics are `GET /metrics` on the admin API, `127.0.0.1:${httpPort}` (8080).

## Kill switches (VLLM_SM70_*)

Agreed by the patch owners (Mara/Kai). Only `=0` does anything: it turns that
sm_70 feature off. Environment variables can't turn features on, and any other
`VLLM_SM70_*` variable from the old fork is ignored on this stack. Defaults and
tuning go through `--additional-config '{"sm70":{...}}'`, and only to change a
default. Set a switch per model with `models.<name>.env`.

| Variable | Kept from the fork |
|---|---|
| `VLLM_SM70_MODELOPT_NVFP4_GEMV` | yes (Flint) |
| `VLLM_SM70_FLINT_MOE_QPN_DECODE` | yes (Flint) |
| `VLLM_SM70_FLINT_QK_ROPE_KV` | yes (Flint) |
| `VLLM_SM70_FLINT_FUSED_ROUTER` | yes (Flint) |
| `VLLM_SM70_FLINT_SAMPLER` | yes (Flint) |
| `VLLM_SM70_FLINT_ATTN_FUSED_DECODE` | yes (Flint) |
| `VLLM_SM70_FLINT_AR_RMSNORM` | yes (Flint) |
| `VLLM_SM70_FLINT_MOE_COMPACT` | new, not in fork (the fork route has no switch); approved as the eighth Flint switch |
| `VLLM_SM70_INKLING_NVFP4_TM` | new (Inkling 2; kill switch: `0` keeps all experts in fp16 on the torch path, default on runs the admitted NVFP4 layers on the TurboMind expert route) |
| `VLLM_SM70_INKLING_TRITON_SCONV` | new (Inkling 2; kill switch: `0` sends the paged short convs and the prologue K/V convs back to the torch ops, default on runs one Triton kernel per call on GPU tensors) |
| `VLLM_SM70_INKLING_GROUPED_MOE` | new (Inkling 2; kill switch: `0` sends the experts held in fp16 back to the torch expert sum, default on runs the sync-free grouped route on GPU tensors) |
| `VLLM_SM70_INKLING_FUSED_ROUTING` | new (Inkling 2; kill switch: `0` sends batches of up to 64 routed slots back to the torch router and slot sort, default on routes them with the Triton routing kernels on GPU tensors) |
| `VLLM_SM70_INKLING_TM_TUNE` | new (Inkling 2; kill switch: `0` leaves the TurboMind NVFP4 tuning knobs unset, default on has the engine-config hook set them for the Inkling 2 expert stages up to the decode graph cap when none is set) |
| `VLLM_SM70_INKLING_ASYNC_SCHED` | new (Inkling 2; kill switch: `0` has the engine-config hook turn an unset async scheduling off and refuse `--async-scheduling`, default on leaves it to upstream, which turns it on when unset) |
| `VLLM_SM70_INKLING_SHARDED_HEAD` | new (Inkling 2; kill switch: `0` keeps the whole LM head on every TP rank, default on splits its rows over the ranks when the vocabulary splits evenly and all-gathers the logits) |
| `VLLM_SM70_INKLING_DSPARK` | new (Inkling 2; kill switch: `0` refuses every `--speculative-config` and the U6 hook returns no speculator; default on accepts `method dspark` with a staged `inkling2_dspark` draft on the V2 runner) |
| `VLLM_SM70_INKLING_DSPARK_AUTO` | new (Inkling 2; kill switch: `0` sets no speculative-decoding schedule, so every decode step verifies `num_speculative_tokens` drafts; default on speculates only in steps of at most `sm70.inkling2.dspark_spec_max_reqs` requests) |
| `VLLM_SM70_INKLING_DSPARK_KERNELS` | new (Inkling 2; kill switch: `0` runs the DSpark draft's norm, RoPE, block attention, SwiGLU, Markov gate and soft-cap ops on torch, default on runs the Triton kernels of `ops/dspark.py`) |
| `VLLM_SM70_INKLING_FUSED_SMALL_OPS` | new (Inkling 2; kill switch: `0` sends the small chains of decode-only batches (q/r clips and relative projection, attention output gate, TurboMind SwiGLU, short-conv padding masks) back to the torch ops, default on runs one Triton kernel each on GPU tensors) |
| `VLLM_SM70_INKLING_GREEDY_TOP1` | new (Inkling 2; kill switch: `0` keeps every batch on compute_logits, the fp16 logit all-gather and the upstream Sampler, default on gives an all-greedy batch the F12 fast path admits a per-shard soft-capped top-1 and a pair all-gather) |
| `VLLM_SM70_INKLING_VISIT_FUSION` | new (Inkling 2; kill switch: `0` puts decode-only batches back on one kernel per op of a layer visit; default on merges the short convs with the residual add and the AttnRes mix after them, the three message-gate kernels, the prologue's q/r and K/V kernels, and the router GEMV with the route, on GPU tensors) |
| `VLLM_SM70_INKLING_MERGED_PROJ` | new (Inkling 2; opt-in: default off keeps each projection in its own buffer and GEMM, `1` packs the projections that share an input and runs a small visit's projections as one GEMM) |
| `VLLM_SM70_FP8_MOE_DECODE` | new |
| `VLLM_SM70_FP8_QPN8` | new name meaning (not the fork's TP4 one) |
| `VLLM_SM70_QUANT` | new |
| `VLLM_SM70_FP8_TURBOMIND` | new |
| `VLLM_SM70_DFLASH2_EMULATION` | new |
| `VLLM_SM70_DFLASH2_VERIFIER` | new |
| `VLLM_SM70_SHARED_MOE_OVERLAP` | new |
| `VLLM_SM70_GRAPH_CAPTURE_SIZES` | new switch for the fork's SM70 CUDA graph capture sizes (T8; the fork had no switch) |
| `VLLM_SM70_GRAPH_MEMORY_POLICY` | new switch for the fork's `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` on V1 (T9) |

To turn the whole plugin off, set `VLLM_PLUGINS` to a comma-separated list
that leaves out `sm70`. vLLM then loads neither entry point
(`vllm.general_plugins` and `vllm.platform_plugins`, both named `sm70`), and
none of the switches above apply. Unset, vLLM loads every installed plugin.

## Host use

```nix
# flake.nix
inputs.vllm-fork.url = "github:charlie-horse/vllm-sm70";
inputs.vllm-fork.inputs.nixpkgs.follows = "nixpkgs";

# a host module
{ inputs, ... }: {
  imports = [ inputs.vllm-fork.nixosModules.default ];
  services.vllm-sm70 = {
    enable = true;
    activeModel = "flint";
    firewallInterfaces = [ "eno1" "tailscale0" ];
  };
  services.lmcache-sm70.ui.firewallInterfaces = [ "tailscale0" ];
}
```
