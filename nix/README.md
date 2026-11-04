# Nix: outputs, modules, checkpoint

## Outputs (the drop-in contract for hosts that used charlie-horse/vLLM)

| Output | What |
|---|---|
| `nixosModules.vllm` (= `default`) | `services.vllm-sm70`: `vllm` + `vllm-model-fetch` units, the model catalog. Imports `nixosModules.lmcache`. |
| `nixosModules.lmcache` | `services.lmcache-sm70`: `vllm-lmcache`, `vllm-lmcache-ui`, `vllm-lmcache-l2-prune`. |
| `packages.x86_64-linux.vllm` (= `default`) | the wrapped vLLM env (`bin/vllm`, `bin/hf`, `bin/lmcache`). |
| `packages.x86_64-linux.flint-checkpoint` | Flint's converted weights (fixed-output, 23 GB). |
| `checks.x86_64-linux.*` | eval-only: checkpoint outPath guard, one NixOS eval per built-in model. |

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
| `vllm.nix` | build | `{ vllm, lmcacheUi }` (stub until the sm_70 build lands; contract in the file) |
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
- `--attention-backend FLASH_ATTN_V100` (qwen3.8-27b) and the DFlash draft's
  `attention_backend = "FLASH_ATTN_V100"`. Without them vLLM picks TRITON_ATTN.

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
