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
| `checkpoint.nix` | integration | `flintCheckpoint` FOD, recipe from input `vllm-legacy` |
| `checks.nix` | integration | flake checks |

## Flint checkpoint

`flintCheckpoint` is fixed-output (`name = "flint-nvfp4-modelopt"`, recursive
sha256 `sha256-HyBUnISsTQXSKmmOft+/fNzrhFLwBJ6imMCa7YRAwoQ=`), so its path is
always `/nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt`,
the same as the old flake's. Upstream vLLM has no Flint converter, so the
recipe (converter env, GGUF, HF config) comes from the old flake (input
`vllm-legacy`, cf9492a) with its own locked nixpkgs/uv2nix; it only builds if
the path is in no store or cache.

## Fork-only settings left out for now

Marked `TODO(sm70-attn)` / `TODO(sm70-moe)` in `module.nix`; upstream v0.33.0
rejects them at startup until the patch series adds them:

- `--kernel-config.sm70_skinny_moe false` (Flint).

Attention needs no flag: the sm70 platform picks Flash-V100 (K06,
`AttentionBackendEnum.CUSTOM`) for fp16 head-128/256 layers, target
and DFlash draft alike, unless `VLLM_SM70_FLASH_V100=0`.

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
| `VLLM_SM70_FP8_MOE_DECODE` | new |
| `VLLM_SM70_FP8_QPN8` | new name meaning (not the fork's TP4 one) |
| `VLLM_SM70_QUANT` | new |
| `VLLM_SM70_FP8_TURBOMIND` | new |
| `VLLM_SM70_DFLASH2_EMULATION` | new |
| `VLLM_SM70_DFLASH2_VERIFIER` | new |
| `VLLM_SM70_SHARED_MOE_OVERLAP` | new |

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
