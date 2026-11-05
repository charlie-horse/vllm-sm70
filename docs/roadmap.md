# Roadmap

This is the ordered plan from upstream vLLM v0.33.0 to cutover on c4130 (2× V100, TP2). It merges Kai's kernel plan (U1–U5, K01–K18; detail in his kernel-carry-map) and Mara's model plan (M1–M11). Infra nodes are added for the flake, CI and cutover.

- **Kinds:**
  - **U** is an upstream edit, the only kind of change in `patches/`.
  - **K** is a kernel or op in `sm70/`: CUDA sources built into `vllm_sm70._sm70_C`, plus its Python wrapper.
  - **M** is a model or quant feature in the `vllm_sm70` plugin. It registers through vLLM's plugin APIs (`additional_config`, `register_quantization_config`, `register_model`, `LINEAR_METHOD_BUILDERS`, `FusedMoEMethodCls`) and touches **no upstream files**.
  - **N** is Nix or CI. **C** is cutover.
- **Level** is the longest dependency chain below a node, counting from K01 or a U node at level 0. Everything on one level can be built in parallel.
- **Kill switches** are `VLLM_SM70_<THING>=0`. A switch lives where the path is *chosen*, normally on the M (wiring) side, and only ever turns a feature off. K nodes ship ops without switches. M nodes feature-detect optional ops, so a missing op means the plain path.
- **Phases:**
  - 1: unpatched build;
  - 2: tier A, run-correct;
  - 3: Flash-V100, GDN and Qwen speed;
  - 4: Flint fast paths;
  - 5: cutover.
- **Opt** marks an optional node. All GPU tests run on c4130 in fp16.

## Upstream-line budget

The whole carry against upstream is these patches:

| patch | lines (approx.) | files |
|---|---|---|
| U1 | +12 | `layers/utils.py:682`, `platforms/interface.py`, `warmup/kernel_warmup.py:164` |
| U2 | ~8 | `fla/ops/fused_sigmoid_gating.py:208-212`, `fla/ops/fused_recurrent.py:196-200` |
| U3 | ~30 | `fused_moe/runner/moe_runner.py` ~:630, :907 |
| U4 | ~60 | `v1/worker/gpu_model_runner.py` :3403, :4208 |
| U5 | ~330 | `csrc/custom_all_reduce.cu`, `csrc/torch_bindings.cpp`, `custom_all_reduce.py`, `cuda_communicator.py` |
| **total** | **~110 without U5; ~440 with it** | |

## Phase 1: unpatched build

| node | what | owner | depends | exit test |
|---|---|---|---|---|
| N0 | Flake builds stock v0.33.0 for sm_70 (FetchContent fed via `*_SRC_DIR`; `--fuzz=0` patching), PR #2 | Felix | – | `nix build .#vllm` on r440; `import vllm` on c4130 reports capability (7,0) |
| N1 | CI: series lint, `--check`, CPU unit tests, nix eval and `nix build .#source` (PR #6, merged) | Penny | – | CI green on main |

## Phase 2: tier A, run-correct

The first runnable cut is **K01 + U1 + K02 + K03**. Attention is upstream TRITON_ATTN until K06.

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| K01 | 0 | sm70 extension (`torch.ops._sm70_C`) and plugin skeleton (general + platform plugins); `VLLM_SM70_SKIP_EXT=1` skips the CUDA build | Kai | N0 | plugin not loaded | CPU `sm70/tests/plugin`: entry points resolve; `register()` is idempotent without `_sm70_C`; env merge works; `activate()` returns None with no GPU; composite-pass uuid is stable. GPU: `_sm70_C` and `_C_stable_libtorch` co-import with no symbol clash |
| N2 | 1 | Flake builds `sm70/` (extension + package) into the venv | Felix | K01, N0 | – | `nix build .#vllm`; `import vllm_sm70` and `torch.ops._sm70_C` load on c4130 |
| U1 | 0 | Platform hooks: unquantized GEMM + warmup | Kai + Mara | – | hook returns None → upstream | CPU: with no hook, upstream outputs are bit-identical; a fake hook is called |
| M1 | 0 | `Sm70Config` (8 fields) + `sm70_config()` reading `additional_config["sm70"]` (PR #8). `shared_moe_overlap` defaults to False (the effective value at cf9492a). `fp8_qpn8` and `dflash2_verifier` resolve to on for Qwen3.8-27B and off for Qwen3.6-35B-A3B | Mara | – | n/a (defaults = current c4130 behaviour) | CPU: defaults equal the frozen effective c4130 values; unknown key rejected; each `VLLM_SM70_<THING>=0` kills only its field, and `1` never enables; compile hash changes only for graph fields |
| U2 | 0 | Volta launch schedule for FLA recurrent GDN | Kai | – | `VLLM_SM70_FLA_RECURRENT_SCHED=0` | GPU: upstream GDN tests pass on sm70 |
| K02 | 1 | Vendored TurboMind GEMM core (lean lmdeploy, ~20K) | Kai | K01 | required | builds for sm_70 only; PROVENANCE sha256 matches the fork (cf9492a) |
| M2 | 1 | `Sm70Fp8Config` / `Sm70ModelOptNvFp4Config` registered over `"fp8"` / `"modelopt_fp4"` (PR #9) | Mara | K01 | `VLLM_SM70_QUANT=0` (upstream classes, which refuse on sm70) | CPU, mocked cc: gate is 70 at cc 7.0 with `_sm70_C`, 75 without; identical to upstream at cc 8.0; registry identity asserted in `register()`; real Flint (nested ModelOpt W4A16_NVFP4) and Qwen fp8 block-128 configs pass the upstream capability check (`config/vllm.py:926-936`) |
| M7 | 1 | DFlash2 draft: BF16-range emulation in fp16, as a `DFlash2DraftModel` subclass re-registered by the plugin, using upstream's `model_cls` / `decoder_layer_cls` factory hooks (`qwen3_dflash.py:737`, `:388`); no upstream edit | Mara | M1 | `VLLM_SM70_DFLASH2_EMULATION=0` | GPU: draft forward in fp16 has no inf/NaN (plain cuBLAS fp16). Acceptance-rate e2e runs after M6 |
| K03 | 2 | TurboMind fp16/FP8/NVFP4 dense + MoE ops (~5.8K), `vllm_sm70/ops.py` | Kai | K02, U1 | `VLLM_SM70_TURBOMIND=0` (fp16 path only) | GPU: TM fp16/FP8-block/NVFP4 GEMM vs `torch.matmul` on dequantized weights, max-abs-rel < 1e-2 at M ∈ {1,2,8,64,512}, N/K from the Flint and Qwen shapes; MoE vs a reference loop; the fork tests in the carry map pass |
| M3 | 3 | NVFP4 dense linear → TurboMind GEMM, via the plugin's `LINEAR_METHOD_BUILDERS` entry; the M≤2 GEMV is wired only if K11's op is present. Must handle or refuse online `--quantization fp8`, which M2's lowered gate admits on sm70 | Mara | M2, K03 | `VLLM_SM70_MODELOPT_NVFP4_GEMV=0` (GEMM only) | GPU: layer output vs dequantized fp16 reference |
| M4 | 4 | NVFP4 MoE → TurboMind MoE via `FusedMoEMethodCls`; the decode fast path is wired only if K12's op is present | Mara | M3, K03 | `VLLM_SM70_FLINT_MOE_QPN_DECODE=0` | GPU: MoE block vs reference |
| M5 | 5 | `FlintSm70ForCausalLM` (FP16 rescale), registered with `register_model` and selected with `--hf-overrides architectures`; fused qk-norm/RoPE/KV is used only if K15 is present | Mara | M3, M4, U1 | `VLLM_SM70_FLINT_QK_ROPE_KV=0` (unfused) | GPU e2e quality suite: math 10/10, code 5/5, needle@32K 3/3, no gibberish at concurrency 20, glm47 tool/reasoning parse, LMCache prefix reuse |
| M6 | 5 | Qwen FP8 block-128 dense + MoE → TurboMind FP8 weight-only | Mara | M2, M4, K03 | `VLLM_SM70_FP8_TURBOMIND=0` → refuse with a clear error; `VLLM_SM70_FP8_MOE_DECODE=0` turns off the FP8 MoE single-token decode route | GPU e2e: Qwen3.6-35B-A3B and Qwen3.8-27B FP8 at TP2 within 32 GB/GPU; quality suite |

**Phase 2 exit:** Flint and Qwen3.6/3.8 serve on c4130 TP2 with `--dtype half` on TRITON_ATTN, and the M5 and M6 quality suites pass. Speed isn't gated yet.

## Phase 3: Flash-V100, GDN and Qwen speed

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| K05 | 1 | Flash-V100 kernels, lean (~8.7K) | Kai | K01 | – | GPU: `test_sm70_flash_v100_sinks_prefill` and `test_sm70_flash_v100_splitkv_n` pass |
| K06 | 2 | Flash-V100 attention backend (CUSTOM slot) | Kai | K05 | `VLLM_SM70_FLASH_V100=0` → TRITON_ATTN | GPU: Flint TP2 4K gives a greedy-identical 256-token continuation to TRITON_ATTN on 5 prompts; decode tok/s ≥ the fork's grouped two-launch number; LBNHC stride test passes |
| K07 | 3 | Flash-V100 spec-decode verify for Qwen DFlash2 | Kai | K06 | `VLLM_SM70_FLASH_V100_SPEC=0` | GPU: smallq exactness test; Qwen DFlash2 acceptance within 2% of the fork |
| K08 | 1 | GDN FlashQLA native prefill + decode | Kai | K01 | `VLLM_SM70_GDN_DECODE_FLASHQLA=0`, `VLLM_SM70_GDN_FLASHQLA_PREFILL=0` | GPU: prefill/decode state vs upstream Triton, rel-L2 < 1e-3 |
| M11 | 2 | Qwen GDN layer subclass (`vllm_sm70/gdn/layer.py`, ~60 lines). Overrides `_forward_core_decode_non_spec` (upstream `qwen_gdn_linear_attn.py:1981`) and `_warmup_prefill_kernels` (:1308-1415) to call K08 | Mara | K08 | K08's switches | GPU: GDN layer output vs the upstream path within tolerance |
| K10 | 2 | opt: GDN TileLang prefill (needs a Nix tilelang) | Kai | K08 | `VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0` (default 0) | GPU: TileLang vs native state rel-L2 < 4e-4 |
| K09 | 3 | QPN8 block-FP8 decode GEMV (op only) | Kai | K03 | – (switch is M8's) | GPU: vs TM FP8 max-abs-rel < 1e-2 |
| K18 | 3 | DFlash2 draft fp16 M≤8 GEMM + compact half top-k (`dflash2_fp16_m8_sm70.cu`, `compact_half_topk_sm70.cu`, `vllm_sm70/ops/dflash2.py`) | Kai | K03 | `VLLM_SM70_DFLASH2_FP16=0` | GPU: vs `torch.matmul` max-abs-rel < 1e-2; draft acceptance unchanged |
| M8 | 6 | Wire QPN8 FP8 decode into Qwen (TP2 only: the fork's TP4 meaning of the switch is not carried) | Mara | M6, K09 | `VLLM_SM70_FP8_QPN8=0` | GPU: Qwen decode tok/s not worse than M6 alone; outputs unchanged |
| M9 | 6 | DFlash2 M8 verifier GEMVs. **Not in a PR until its hook point is named** | Mara | M6, M7 | `VLLM_SM70_DFLASH2_VERIFIER=0` | GPU: DFlash2 spec decode ≥ fork (fork reference: 111 tok/s at 4K on the 3.6-35B) |

**Phase 3 exit:** Qwen3.6/3.8 decode and prefill on c4130 are at or above the fork's numbers in `bench_vllm.py`, with Flash-V100, GDN FlashQLA and DFlash2 on.

## Phase 4: Flint fast paths

**D2:** the V1-vs-V2 sampler decision is deferred until V1 and V2 are measured, before K16. **D3:** U5 ships, and it lands last in this phase.

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| U3 | 0 | MoE router/gate provider hook | Kai | – | no provider → upstream | CPU: with no provider, upstream behaviour is bit-identical |
| U4 | 0 | V1 sampler hook (subject to D2) | Kai | – | no hook → upstream | CPU: with no hook, upstream behaviour is bit-identical |
| K11 | 3 | Flint NVFP4 M≤2 GEMV (op only; M3 feature-detects it) | Kai | K03 | – (switch is M3's `VLLM_SM70_MODELOPT_NVFP4_GEMV`) | fork tests listed in the carry map |
| K12 | 3 | Flint NVFP4 MoE QPN decode (op only; M4 feature-detects it) | Kai | K03 | – (switch is M4's `VLLM_SM70_FLINT_MOE_QPN_DECODE`) | fork tests listed in the carry map |
| K13 | 1 | Flint fused router (op) | Kai | K01, U3 | – (switch is M10's) | fork tests; greedy-identical with the switch off |
| K14 | 3 | Flint one-launch sink decode + counter fix (fork 7f12d21b8) | Kai | K06 | `VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0` | GPU: `test_decode_sinks_descending_capture_regression`; max-num-seqs sweep 2/3/4/8/20 coherent, each prompt twice |
| K15 | 1 | Flint q/k-norm + RoPE + KV write, Triton (op; M5 calls it) | Kai | K01 | – (switch is M5's `VLLM_SM70_FLINT_QK_ROPE_KV`) | GPU: vs unfused ops |
| K16 | 1 | Flint top-k candidate sampler (op) | Kai | K01, U4 | – (switch is M10's) | fork tests; greedy-identical with the switch off |
| M10 | 6 | Wire K13 (router) and K16 (sampler) into M5 | Mara | M5, K13, K16 | `VLLM_SM70_FLINT_FUSED_ROUTER=0`, `VLLM_SM70_FLINT_SAMPLER=0` | GPU e2e: greedy-identical with each switch off |
| K04 | 3 | opt: AWQ/int4 | Kai | K03 | `VLLM_SM70_TURBOMIND_AWQ=0` | GPU: AWQ cases in `test_sm70_turbomind_adapter` |
| U5 | 0 | TP2 fused all-reduce + add + RMSNorm, stable ABI. **Lands last** (D3) | Kai | – | `VLLM_SM70_FLINT_AR_RMSNORM=0` | GPU, 2 GPUs: `test_sm70_tp2_ar_fused_add_rms_norm` bitwise-equal to unfused |
| K17 | 1 | Flint AR+RMSNorm fusion pattern (composite post-grad pass) | Kai | K01, U5 | `VLLM_SM70_FLINT_AR_RMSNORM=0` (same switch as U5) | GPU: pass fires on the Flint graph; output greedy-identical with the switch off |

**Phase 4 exit:** Flint TP2 4K decode is at least **170 tok/s** on c4130, with every fast path on. Every kill switch set to 0 gives greedy-identical output.

## Phase 5: cutover

| node | what | owner | depends | exit test |
|---|---|---|---|---|
| C1 | Side-by-side bench and quality run vs the fork (cf9492a) on c4130: `verify_vllm.py`, `bench_vllm.py`, `kvtier_test.py`, and the SWE-bench slice (fork ≈37/50) | Astro + Mara | phase 2–4 exits | report attached to the cutover PR |
| C2 | LMCache 0.5.6 from sdist (cu12, sm_70) in the flake, loaded via `kv_connector_module_path`, no vLLM patch | Theo + Felix | N2 | `kvtier_test.py` passes |
| C3 | Switch `hosts/c4130/vllm` in charlie-horse/flake to this flake's `latest` branch (gated per `docs/release.md`; after the GC of old generations c4130 needs first) | Felix + Charlie | C1, C2 | `nixos-rebuild switch` succeeds; etcd health OK; service serves |
| C4 | Archive the fork (charlie-horse/vLLM) as reference only | Charlie | C3 | – |

## Series order for `patches/`

1. 0001–0099 build: 0001 trims the external projects that build nothing for sm_70 (`VLLM_SM70_TRIM_EXTERNALS=1`, set by nix/package.nix; unset or ≠1 → upstream). The remaining FetchContent sources are fed by Nix.
2. 0100–0199 platform: U1, U2, U3, U4, then U5 last.
3. 0500–0699 models: none. The M7 subclass route works without a patch.
4. 0700–0799 kv: none.

## Settled

- **Plugin route** for M nodes, so the models area adds 0 upstream lines (Charlie).
- **Kill-switch placement** (Kai and Mara): one switch per feature, read where the path is chosen, normally in M wiring. K14 (attention) and K17 (pass) are the exceptions, because their path is chosen inside Kai's code. K ops carry no runtime switch. Only `0` acts; env never turns a feature on. The seven Flint switch names stay exactly as the fork's operator docs have them: `MODELOPT_NVFP4_GEMV`, `FLINT_MOE_QPN_DECODE`, `FLINT_QK_ROPE_KV`, `FLINT_FUSED_ROUTER`, `FLINT_SAMPLER`, `FLINT_ATTN_FUSED_DECODE`, `FLINT_AR_RMSNORM` (each `VLLM_SM70_`-prefixed).
- **K15 is the op and M5 calls it,** with `VLLM_SM70_FLINT_QK_ROPE_KV` as the only switch. `VLLM_SM70_FLINT_FUSIONS` is dropped.
- **M6-M (Marlin-SM70 FP8) is dropped for v1.** Revisit only if M6 is blocked.
- **D2** is deferred to phase 4, measured before K16. **D3:** U5 ships, last in phase 4.
- **kv:** no vLLM-side patches (Theo).
