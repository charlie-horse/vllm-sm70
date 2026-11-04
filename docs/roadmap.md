# Roadmap

This is the ordered plan from upstream vLLM v0.33.0 to cutover on c4130 (2× V100, TP2). It merges Kai's kernel plan (U1–U5, K01–K17; detail in his kernel-carry-map) and Mara's model plan (M1–M10). Infra nodes are added for the flake, CI and cutover.

- **Kinds:**
  - **U** is an upstream edit (in `patches/`).
  - **K** is a kernel or plugin feature (in `sm70/`).
  - **M** is a model or quant feature (in `sm70/`, plus a small `models`-area patch where it lists upstream files).
  - **N** is Nix or CI. **C** is cutover.
- **Level** is the longest dependency chain below a node. Everything on one level can be built in parallel.
- **Kill switches** are `VLLM_SM70_<THING>=0`. Each one turns its feature off, back to the upstream path.
- **Phases:**
  - 1: unpatched build;
  - 2: tier A, run-correct;
  - 3: Flash-V100 and GDN speed, plus Qwen decode speed;
  - 4: Flint fast paths;
  - 5: cutover.
- **Opt** marks an optional node.

## Phase 1: unpatched build

| node | what | owner | depends | exit test |
|---|---|---|---|---|
| N0 | Flake builds stock v0.33.0 for sm_70 (all FetchContent fed via `*_SRC_DIR`), PR #2 | Felix | – | `nix build .#vllm` on r440; `import vllm` on c4130 with `cuda.get_device_capability()==(7,0)` |
| N1 | CI: `--check`, series lint, CPU unit tests, nix eval and `nix build .#source` | Penny | – | CI green on main |

## Phase 2: tier A, run-correct

The first runnable cut is **K01 + U1 + K02 + K03**. Attention is upstream TRITON_ATTN until K06.

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| K01 | 0 | sm70 extension (`torch.ops._sm70_C`) and plugin skeleton (general + platform plugins) | Kai | N0 | plugin not loaded | CPU: plugin entry points load, and `VLLM_PLUGINS=` (empty) leaves stock behaviour. GPU: `_sm70_C` loads on cc 7.0 |
| N2 | 1 | Flake builds `sm70/` (extension + package) into the venv | Felix | K01, N0 | – | `nix build .#vllm`; `python -c "import vllm_sm70, torch; torch.ops._sm70_C"` |
| U1 | 0 | Platform hooks: unquantized GEMM + warmup (`layers/utils.py:682`, `platforms/interface.py`, `warmup/kernel_warmup.py:164`) | Kai + Mara | – | hook returns None → upstream | CPU: a fake platform hook is called, and None falls through to upstream |
| M1 | 0 | Typed `Sm70Config` (7 fields) + `sm70_config()`; `config/kernel.py` field `sm70` (+3) | Mara | – | n/a (defaults = current behaviour) | CPU: config round-trips; env kill-switches override only their field |
| U2 | 0 | Volta launch schedule for FLA recurrent GDN (`fla/ops/fused_sigmoid_gating.py`, `fused_recurrent.py`, ~8 lines) | Kai | – | `VLLM_SM70_FLA_RECURRENT_SCHED=0` | GPU: Qwen3.6 GDN layer output matches the fp32 reference (tolerance in the test); no launch error on sm_70 |
| K02 | 1 | Vendored TurboMind GEMM core (lean lmdeploy, ~20K) | Kai | K01 | required | build only; symbols present in `_sm70_C` |
| M2 | 1 | Quant gates: fp8 and modelopt min-capability 70 when cc==70 and `_sm70_C` is loaded (`fp8.py:146`, `modelopt.py:772`) | Mara | K01 | `VLLM_SM70_QUANT=0` (upstream refuses on sm70) | CPU: gate is 75 without the plugin, 70 with it |
| M7 | 1 | DFlash2 draft: BF16-range emulation in fp16 (`qwen3_dflash2.py`, `qwen3_dflash.py`) | Mara | M1 | `VLLM_SM70_DFLASH2_EMULATION=0` | GPU: draft forward in fp16 has no inf/NaN (plain cuBLAS fp16). Acceptance-rate e2e runs after M6 |
| K03 | 2 | TurboMind fp16/FP8/NVFP4 dense + MoE ops (~5.8K), `vllm_sm70/ops.py` | Kai | K02, U1 | `VLLM_SM70_TURBOMIND=0` (fp16 path only) | GPU: each op vs a torch reference within tolerance, over the shapes of Flint and Qwen |
| M3 | 3 | NVFP4 dense linear → TurboMind GEMM + M≤2 GEMV (`modelopt.py` LINEAR_METHOD_BUILDERS) | Mara | M2, K03 | `VLLM_SM70_NVFP4_GEMV=0` (GEMM only) ⚠ see open item 1 | GPU: layer output vs dequantized fp16 reference |
| M4 | 4 | NVFP4 MoE → TurboMind MoE (dense stage + M≤2 decode) | Mara | M3, K03 | `VLLM_SM70_MOE_DECODE_FASTPATH=0` ⚠ open item 1 | GPU: MoE block vs reference; needs `_moe_C.moe_permute_metadata_with_scratch` |
| M5 | 5 | `FlintSm70ForCausalLM` (FP16 rescale, fused qk-norm/RoPE/KV); selected with `--hf-overrides architectures` | Mara | M3, M4, U1 | `VLLM_SM70_FLINT_FUSIONS=0` (unfused) ⚠ open item 2 | GPU e2e: Flint TP2 serves, and the quality suite matches the fork (cf9492a) within agreed bounds |
| M6 | 5 | Qwen FP8 block-128 dense + MoE → TurboMind FP8 weight-only | Mara | M2, M4, K03 | `VLLM_SM70_FP8_TURBOMIND=0` (→ M6-M if present, else refuse) | GPU e2e: Qwen3.6 FP8 TP2 serves; quality suite vs the fork |
| M6-M | 3 | opt: FP8 via Marlin-SM70 | Mara | M2, **Kai sm70_marlin (no K node yet, open item 3)** | `VLLM_SM70_FP8_MARLIN=0` | GPU: both fallback branches of M6 tested |

**Phase 2 exit:** Flint and Qwen3.6 serve on c4130 TP2 with `--dtype half` on TRITON_ATTN, and the quality suite passes against the fork. Speed isn't gated yet.

## Phase 3: Flash-V100, GDN and Qwen decode speed

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| K05 | 1 | Flash-V100 kernels, lean (~8.7K) | Kai | K01 | – | GPU: kernel vs reference attention (fp16) over head dims and lengths |
| K06 | 2 | Flash-V100 attention backend (CUSTOM slot) | Kai | K05 | `VLLM_SM70_FLASH_V100=0` → TRITON_ATTN | GPU e2e: same outputs as TRITON_ATTN (greedy) and faster prefill/decode |
| K07 | 3 | Flash-V100 spec-decode verify for Qwen DFlash2 | Kai | K06 | `VLLM_SM70_FLASH_V100_SPEC=0` | GPU: verify path matches the non-spec path; acceptance rate within bounds of the fork |
| K08 | 1 | GDN FlashQLA native prefill + decode | Kai | K01 | `VLLM_SM70_GDN_DECODE_FLASHQLA=0`, `VLLM_SM70_GDN_FLASHQLA_PREFILL=0` | GPU: GDN vs U2 path within tolerance; faster |
| K10 | 2 | opt: GDN TileLang prefill (needs a Nix tilelang) | Kai | K08 | `VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0` (default 0) | as K08 |
| K09 | 3 | QPN8 block-FP8 decode GEMV | Kai | K03 | `VLLM_SM70_FP8_QPN8_BLOCK=0` ⚠ open item 1 | GPU: GEMV vs K03 FP8 GEMM at M≤2 |
| M8 | 6 | Wire QPN8 FP8 decode into Qwen | Mara | M6, K09 | `VLLM_SM70_FP8_QPN8=0` ⚠ open item 1 | GPU: Qwen decode tok/s ≥ fork at 4K; outputs unchanged |
| M9 | 6 | DFlash2 M8 verifier GEMVs. **Out of the series until its Upstream-files are named** | Mara | M6, M7 | `VLLM_SM70_DFLASH2_VERIFIER=0` | GPU: spec decode with DFlash2 ≥ fork (fork reference: 111 tok/s at 4K on the 3.6-35B) |

**Phase 3 exit:** Qwen3.6 decode and prefill on c4130 are at or above the fork's numbers from `bench_vllm.py`, with Flash-V100 and DFlash2 on.

## Phase 4: Flint fast paths

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| U3 | 0 | MoE router/gate provider hook (`fused_moe/runner/moe_runner.py`) | Kai | – | no provider → upstream | CPU: no provider → upstream path |
| U4 | 0 | V1 sampler hook (`v1/worker/gpu_model_runner.py`). **Pending D2: V1 vs V2 runner** | Kai | – | no hook → upstream | CPU: no hook → upstream path |
| U5 | 0 | opt: TP2 fused all-reduce + add + RMSNorm, stable ABI (`csrc/custom_all_reduce.cu` +240 etc.). **Pending D3; lands last** | Kai | – | `VLLM_SM70_FLINT_AR_RMSNORM=0` | GPU TP2: fused result = unfused AR→add→RMSNorm |
| K11 | 3 | Flint NVFP4 M≤2 GEMV | Kai | K03 | `VLLM_SM70_MODELOPT_NVFP4_GEMV=0` ⚠ open item 1 | GPU: vs K03 GEMM at M≤2 |
| K12 | 3 | Flint NVFP4 MoE QPN decode | Kai | K03 | `VLLM_SM70_FLINT_MOE_QPN_DECODE=0` ⚠ open item 1 | GPU: vs M4 MoE at M≤2 |
| K13 | 1 | Flint fused router | Kai | K01, U3 | `VLLM_SM70_FLINT_FUSED_ROUTER=0` | GPU: same top-k and weights as upstream router |
| K14 | 3 | Flint one-launch sink decode + counter fix (fork 7f12d21b8) | Kai | K06 | `VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0` | GPU: same as the K06 decode path |
| K15 | 1 | Flint q/k-norm + RoPE + KV write (Triton) | Kai | K01 | `VLLM_SM70_FLINT_QK_ROPE_KV=0` ⚠ open item 2 | GPU: vs unfused ops |
| K16 | 1 | Flint top-k candidate sampler | Kai | K01, U4 | `VLLM_SM70_FLINT_SAMPLER=0` | GPU: distribution test vs upstream sampler (fixed seed) |
| K17 | 1 | Flint AR+RMSNorm fusion pattern (post-grad pass) | Kai | K01, U5 | `VLLM_SM70_FLINT_AR_RMSNORM=0` | GPU: pass fires on Flint graph; outputs unchanged |
| K04 | 3 | opt: AWQ/int4 | Kai | K03 | `VLLM_SM70_TURBOMIND_AWQ=0` | GPU: AWQ layer vs dequant reference |
| M10 | 6 | Wire the Flint fused router and sampler into M5. **Deferred, out of the series until its files are named** | Mara | M5, K13, K16 | `VLLM_SM70_FLINT_FUSIONS=0` | GPU e2e: Flint tok/s ≥ fork |

**Phase 4 exit:** Flint on c4130 runs at or above the fork's speed, with every fast path on, and each kill switch tested to give identical output.

## Phase 5: cutover

| node | what | owner | depends | exit test |
|---|---|---|---|---|
| C1 | Side-by-side bench and quality run vs the fork (cf9492a) on c4130: `verify_vllm.py`, `bench_vllm.py`, `kvtier_test.py` | Astro + Mara | phase 2–4 exits | report attached to the cutover PR |
| C2 | LMCache 0.5.6 from sdist (cu12, sm_70) in the flake; KV tiering works | Theo + Felix | N2 | `kvtier_test.py` passes |
| C3 | Switch `hosts/c4130/vllm` in charlie-horse/flake to this flake (after the GC of old generations c4130 needs first) | Felix + Charlie | C1, C2 | `nixos-rebuild switch` succeeds; etcd health OK; service serves |
| C4 | Archive the fork (charlie-horse/vLLM) as reference only | Charlie | C3 | – |

## Series order for `patches/` (upstream edits only)

1. 0001–0099 build: none yet. FetchContent is fed by Nix, so trimming it is optional.
2. 0100–0199 platform: U1, U2, U3, U4, then U5 last.
3. 0500–0699 models (only the upstream-file parts):
   - M1 `config/kernel.py`
   - M2 quant gates
   - M7 dflash
   - M3/M4 `modelopt.py` registration
   - M5 `registry.py` + `models/config.py`
   - M6 `fp8.py`
   - M6-M marlin
4. 0700–0799 kv: none yet (Theo).

## Open items (owners to resolve before the affected PRs)

1. **Duplicate kill switches for one feature.** Pick one name per feature, and decide whether the switch lives in K (kernel availability) or M (model wiring):
   - M3 `VLLM_SM70_NVFP4_GEMV` vs K11 `VLLM_SM70_MODELOPT_NVFP4_GEMV`;
   - M4 `VLLM_SM70_MOE_DECODE_FASTPATH` vs K12 `VLLM_SM70_FLINT_MOE_QPN_DECODE`;
   - M8 `VLLM_SM70_FP8_QPN8` vs K09 `VLLM_SM70_FP8_QPN8_BLOCK`.

   Kai and Mara own this.
2. **Overlap between M5 and K15.** Both describe the fused Flint q/k-norm + RoPE + KV write. Proposal: K15 is the op, M5 calls it, and `VLLM_SM70_FLINT_QK_ROPE_KV` is the only switch for it. `VLLM_SM70_FLINT_FUSIONS` then covers only what's left in M5. Kai and Mara own this.
3. **M6-M needs a Marlin-SM70 kernel node.** Kai's list has none. Either add K18 or drop M6-M. Kai owns this.
4. **Pending decisions:** D2 (V1 vs V2 sampler hook, U4) and D3 (U5 at all). Astro and Charlie own these.
