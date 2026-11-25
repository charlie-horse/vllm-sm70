# Nix: using the flake and its modules

The build is a [Nix flake](https://nixos.org): a file (`flake.nix`) that says
exactly what to build and from which pinned sources. Its outputs are the
vLLM package and two **NixOS modules**, settings blocks a NixOS machine
imports to run vLLM (`services.vllm-sm70`, in `nix/module.nix`) and its
**LMCache** server (a cache that keeps already-computed prompt data so
repeated text is not recomputed; `services.lmcache-sm70`, in
`nix/lmcache.nix`). Those files list every option and built-in model.

## Use the flake in a NixOS configuration

```nix
# flake.nix
inputs.vllm-sm70.url = "github:charlie-horse/vllm-sm70";
inputs.vllm-sm70.inputs.nixpkgs.follows = "nixpkgs";

# a host module
{ inputs, ... }: {
  imports = [ inputs.vllm-sm70.nixosModules.default ];
  services.vllm-sm70.enable = true;
}
```

- Always set `inputs.nixpkgs.follows`: the packages build with the host's
  `pkgs`.
- Import `nixosModules.lmcache` instead to run only the cache server.
- The header of `nix/module.nix` has a full host example. Switch models by
  setting `activeModel` and running `nixos-rebuild switch`.

## Override a built-in model

- Set single fields, e.g. `services.vllm-sm70.models.flint.maxNumSeqs = 20;`.
  Every built-in field is a `mkDefault`. The fields are listed in the
  comment above the `models` option in `nix/module.nix`; the entries are in
  `builtinModels` there.
- A host `env` replaces the built-in `env` as a whole. Repeat
  `CUDA_DEVICE_ORDER` and `VLLM_SKIP_P2P_CHECK` from the built-in entry when
  you set one (e.g. a kill switch). `v2ModelRunner` is a separate field and
  survives an `env` override.

## Kill switches

Every `VLLM_SM70_*` runtime switch and its kind is in
`sm70/vllm_sm70/envs.py`. Set one per model with `models.<name>.env`.

- A kill switch acts only on `=0`, which turns its feature off. Any other
  value, `1` included, keeps the default; the environment cannot turn a
  feature on.
- An opt-in switch (`opt_in`) acts only on `=1`.
- A route switch (`route_switch`) acts on `=0` (route off everywhere) and
  `=1` (route on everywhere); any other value keeps the default route.
- Change a plugin default only with `--additional-config '{"sm70":{...}}'`
  (fields: `sm70/vllm_sm70/config/__init__.py`); unknown keys are refused.
- To turn the whole plugin off, set `VLLM_PLUGINS` to a comma-separated list
  that leaves out `sm70`. Then no `VLLM_SM70_*` switch applies. Unset, vLLM
  loads every installed plugin.

## Inkling 2

Serve rules:

- Pass `--reasoning-parser inkling2_sm70` only together with
  `--tool-call-parser inkling2_sm70 --enable-auto-tool-choice`.
- Block size: leave it unset or use 32. KV cache dtype: `auto` or
  `float16`. No pipeline parallelism and no `--attention-backend`.
- CUDA graphs: leave `cudagraph_mode` unset or use `full_decode_only`,
  `piecewise` or `none`; `full` is refused. A mode with piecewise graphs
  captures mixed batches only with `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. With
  `VLLM_SM70_INKLING_DECODE_GRAPH=0` or a `visit_dump_dir`, no mode with FULL
  graphs is accepted. Capture sizes you set above q × `DECODE_GRAPH_MAX_BATCH`
  tokens are refused (q = 1, or 1 + `num_speculative_tokens`). With
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and the mode unset, mixed batches (prefill
  chunks) also replay piecewise graphs (CUDA graphs broken around the
  attention, conv and expert ops) up to `--max-num-batched-tokens`;
  `VLLM_SM70_INKLING_MIXED_GRAPH=0` runs them eagerly.
  The checks are in `sm70/vllm_sm70/models/inkling2_sm70/config.py`.
- With `VLLM_SM70_INKLING_ASYNC_SCHED=0`, don't pass `--async-scheduling`.
- `sm70.inkling2` fields are set only through `--additional-config`
  (`'{"sm70":{"inkling2":{...}}}'`); they have no env variable.

Client rules:

- With `--tokenizer-mode inkling2_sm70`, a client that wants one tool call
  per response sends `chat_template_kwargs: {"parallel_tool_calls": false}`.
  The top-level `parallel_tool_calls: false` only trims the response to the
  first call.
- Without `--tokenizer-mode inkling2_sm70`, request text must not contain
  `<|unused_200054|>` when the request has images.
- An image larger than `max_image_encoder_tokens` allows is scaled down,
  keeping its shape, until it fits. With `VLLM_SM70_INKLING_IMAGE_RESIZE=0`
  it is rejected instead.
- A photo whose EXIF Orientation tag says to turn it is turned upright
  before the model sees it. With `VLLM_SM70_INKLING_IMAGE_EXIF=0` the
  pixels are used as stored.

### DSpark draft

The built-in `inkling-2-mini` and `inkling-2-mini-uncensored` entries each
stage their draft in `preStart` at every start and serve it with DSpark. To
serve one without DSpark (or with
`VLLM_SM70_INKLING_DSPARK=0`), override its `args` without
`--speculative-config`. For a hand-run server:

1. Stage the draft from the target checkpoint:

   ```sh
   python -m vllm_sm70.models.inkling2_sm70.draft <target dir> <draft dir>
   ```

2. Serve on the V2 runner (`v2ModelRunner = true`, i.e.
   `VLLM_USE_V2_MODEL_RUNNER=1`) with
   `--speculative-config '{"method":"dspark","model":"<draft dir>","num_speculative_tokens":<draft block - 1>}'`.
   Any other speculative config is refused.
3. Restage whenever the target checkpoint changes.

## TurboMind GEMM LUT

The `services.vllm-sm70.turbomind` options (`nix/module.nix`) set the
small-shape tuning and LUT variables; a model's `env` overrides them.

Export a LUT with a runtime drop-in (no rebuild; gone after reboot or
`revert`):

```sh
sudo systemctl edit --runtime vllm
#   [Service]
#   Environment=VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=1 VLLM_SM70_FP8_TUNE_SMALL_SHAPES=1
#   Environment=VLLM_SM70_GEMM_LUT_EXPORT_PATH=/var/lib/vllm/tm-lut-<model>-{arch}.bin
sudo systemctl restart vllm
# after "Application startup complete":
sudo journalctl -u vllm -b | grep -E "TurboMind pre-capture warmup|LUT exported"
ls -l /var/lib/vllm/tm-lut-<model>-*.bin /var/lib/vllm/tm-lut-<model>-*.bin.plan.json
sudo systemctl revert vllm && sudo systemctl restart vllm
```

Use it by setting `turbomind.tuneSmallShapes = true` and
`turbomind.gemmLutPath` to the same template. A start that uses the LUT
logs `tuned (measured) 0 calls on this rank` and `N/N calls covered`.

- Write the LUT under `/var/lib/vllm`; the vllm user must own it.
- Re-export after any change to the vLLM/`_sm70_C` build, the model, TP
  size, `maxNumSeqs` or the CUDA graph capture sizes. A stale or foreign LUT
  is refused with a WARNING and the warmup re-tunes;
  `turbomind.gemmLutStrict = true` makes that a startup failure.

## LMCache

- Patches to the LMCache sdist are listed in
  `nix/patches/lmcache/default.nix`, each with its reason in its header.
  Add a patch there; `lmcache-build.nix` applies them with `--fuzz=0` and
  the `lmcache-patches-apply` flake check must pass.
- Check a built LMCache on a host without the GPUs:
  `scripts/check-lmcache-cpu.sh` ([checks.md](checks.md)).
- Extra `lmcache server` flags go in `services.lmcache-sm70.extraArgs`
  (appended last, so they override the module's).
- Metrics: scrape `GET /metrics` on the admin API,
  `127.0.0.1:<services.lmcache-sm70.httpPort>`. `lmcache server` starts no
  standalone Prometheus exporter.

## Flint checkpoint store path

The checkpoint is a fixed-output `requireFile` (`nix/checkpoint.nix`); Nix
only substitutes it. If its path is in no store:

1. Copy it from a host that has it:
   `nix copy --from ssh://<host> <expectedOutPath>`
   (`expectedOutPath` in `nix/checkpoint.nix`).
2. Otherwise regenerate it with `sm70/tools/flint/convert_gguf.py`. The
   pinned GGUF and `--hf-config` files and the converter's dependencies are
   in `sm70/tools/flint/PROVENANCE.md`; the `.#vllm` Python lacks `gguf`,
   so use a Python that has it (`$python`). From the repo root:

   ```sh
   export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
   PYTHONPATH=sm70/tools $python -m flint.convert_gguf \
     $gguf ./flint-nvfp4-modelopt --hf-config $hfConfig --device cpu
   rm -f ./flint-nvfp4-modelopt/conversion_report.json
   nix-store --add-fixed --recursive sha256 ./flint-nvfp4-modelopt
   ```

   The directory must be named `flint-nvfp4-modelopt`, and `nix-store` must
   print `expectedOutPath`. Any other path is a different output; the module
   does not use it.
