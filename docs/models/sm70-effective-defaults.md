# R5: effective SM70 knob values on c4130 (FORK cf9492a, static)

Refs: FORK `cf9492a` unless HOST. eff = effective, dflt = plain envs.py/field default, ⚠ = effective ≠ default.

## Inputs and machinery

- **Unit env.** `nix/module.nix:198-207` sets HOME, HF_HOME, XDG_CACHE_HOME, VLLM_CACHE_ROOT, TRITON_CACHE_DIR, TILELANG_CACHE_DIR, VLLM_NO_USAGE_STATS and DO_NOT_TRACK. `:329` adds `HF_HUB_OFFLINE=1` and the model `env`. Flint's model env (`:69-77`) is CUDA_DEVICE_ORDER and VLLM_SKIP_P2P_CHECK=0. The wrapper (`nix/package.nix:135-144`) sets CUDA_HOME, CUDA_PATH, NVCC_PREPEND_FLAGS, TRITON_LIBCUDA_PATH, CC, CXX, PATH and LD_LIBRARY_PATH. HOST `hosts/c4130/vllm/default.nix` sets no env. So **no VLLM_SM70_\* is set**, which matches Bea's baseline.
- **Serve args.** Common args are in `module.nix:162-196`. Flint adds `--kv-cache-dtype auto --kernel-config.sm70_skinny_moe false`, `asyncScheduling=true`, `max-num-seqs 20` (HOST default.nix:21) and `gpu-mem 0.90`. Qwen adds `--language-model-only --mamba-cache-mode align --speculative-config {method:dflash, k=7, kv auto, FLASH_ATTN_V100, probabilistic}`. Qwen3.8 also gets `--attention-backend FLASH_ATTN_V100 --block-size 2048`.
- **What actually overrides defaults:** `platforms/runtime_defaults.py:254 apply_runtime_policy_defaults`, called from `config/vllm.py:1426`. That runs after the model hooks at `vllm.py:1140` (`FlintModelConfig.verify_and_update_config`, `models/config.py:341-404`). Writes go through `PolicyDefaults.__setitem__` (`config/policy_defaults.py:116-140`) into the typed owners, never into os.environ.
  - `model_executor/models/runtime_defaults.py:194` is **GLM-5.3 DFlash2 TP8/PP1 only** (`:165-180` requires model_type glm5_next, modelopt_fp4, TP8). It does not apply. The same holds for `:125` (GLM5 TP4 push AR), `:217` (GLM5 TP4/PP2) and `:70-121` (Qwen4Exp FP16 lane, arch Qwen4Exp*). The graph_cache rule (`:43-67`) needs NVFP4, TP4 and fp8_e4m3 KV, so it doesn't apply either.
  - `sm70_profiles/acceleration.py:752` is **report-only**: `build_report` and `log_and_validate` (`:1010-1039`) only warn, and raise only if `require_profile_acceleration`. The release profile JSON (`sm70_profiles/profile.py`) only generates argv. Neither changes any knob.

## (a) The Sm70Config fields (8 at M1, + 2 at M8, + 1 for F12)

| Field | Flint (NVFP4 W4A16, TP2) | Qwen3.6-35B-A3B FP8 | Qwen3.8-27B FP8 | Evidence chain |
|---|---|---|---|---|
| quant_backend | auto → **turbomind** | auto → **turbomind** (W8A16) | auto → **turbomind** | VLLM_SM70_QUANT_BACKEND='auto' envs.py:3276; NVFP4_TURBOMIND=1 envs.py:8546 / FP8_TURBOMIND=1 envs.py:8485; `use_turbomind` kernel.py:232-234. No override |
| nvfp4_gemv | **True**, for M≤2 on the 6 TP2 (K,N) shapes; M>2 uses the TM GEMM | n/a | n/a | MODELOPT_NVFP4_GEMV=1 envs.py:8669 → loader alias kernel.py:62 → sm70_turbomind.py:560-579. No override |
| flint_moe_qpn_decode | **True**, M≤2, contract ∈ `_DECODE_QPN_CONTRACTS`, finite scales; persistent off | n/a | n/a | envs.py:5079 → sm70_moe.py:472 → nvfp4_sm70_moe.py:230-290; PERSISTENT=0 envs.py:5100. No override. `sm70_skinny_moe=false` (serve arg) makes TurboMind take M>2 ⚠ (fork default True, kernel.py:814) |
| flint_moe_compact | **True**: dense-stage calls with M·top_k ≤ 80 (Flint M ≤ 13) run one compact grouped TurboMind launch per stage (also the Inkling 2 TurboMind experts, per slot chunk) | n/a | n/a | fork `_use_compact_grouped` nvfp4_sm70_moe.py:718-721 → :1619-1636 (`_prepare_compact_slot_groups`, :570-618), always on, no fork switch. Kill `VLLM_SM70_FLINT_MOE_COMPACT=0`: new, not in fork, approved by Astro 2026-11-08 as the eighth Flint switch |
| fp8_moe_decode | n/a | **True** (M==1: dense W13 (compact exact layout) + indexed W2 + **weighted** native reduce) | n/a (dense, unverified) | sm70_moe.py:16-71; FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT=1, FP8_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH=1, MOE_SINGLE_TOKEN_UNPERMUTE_FASTPATH=1 (envs.py:9262) → `single_token_reduce="weighted"` (sm70_moe.py:393-397; consumer fp8_sm70_moe.py:54). No override |
| flint_qk_rope_kv | **True** (fp16, FP16 KV) | n/a | n/a | envs.py:8622 → kernel.py:774-781 (resolved at vllm.py:1188) → flint/model.py:312-330. No override |
| fp8_qpn8 (auto) | n/a | **False** (not qualified) | **True** ⚠ | dflt 0 envs.py:3880. Override: platforms/runtime_defaults.py:497-513 sets `sm70_fp8.qpn8 = target_fp8_qpn8` when the verifier contract qualifies and quantization=="fp8"; VERIFIER_DEFAULTS sm70_dflash2.py:27 = "1" |
| fp8_block_qpn8 (M8) | n/a | flag **True**; effective **off at prod** ⚠ | flag **True**; effective **off at prod** ⚠ | **Effective at prod (max_num_seqs 4, DFlash2 k=7 → 32 rows): neither Qwen model takes block QPN8 by default; both stay on M6 TurboMind**, because the copy policy below admits QPN8 only when rows ≤ K09 `qpn8_max_m(block_scaled=True)` = 8 (32 only with VLLM_SM70_FP8_QPN8_M32=1). Signed-off deviation (Astro, PR #65); the Nov 7 A/B (M32=1, copy=true) decides whether the default changes. Fork flag: `kernel_config.sm70_fp8.block_qpn8 = True` kernel.py:386; QPN8Fp8BlockScaledMMLinearKernel first in `_POSSIBLE_FP8_BLOCK_KERNELS` (kernels/linear/__init__.py:344); can_implement fp8_block.py:160-195 gates only on the flag, fp16, static 128×128 block, rank 2, N,K%128, native ops. Not the `qpn8` selector above |
| fp8_block_qpn8_turbomind_copy (M8) | n/a | None (safe) | None (safe) | **Deliberately differs from the fork.** Fork `block_qpn8_volta_turbomind_prefill = None` kernel.py:388 → `_keep_volta_turbomind_copy` fp8_block.py:204-213 kept a second TurboMind copy iff max_num_seqs × (1 + k) > 8. Ours: None admits block QPN8 with no copy iff rows ≤ K09 `qpn8_max_m(block_scaled=True)` (8; 32 with VLLM_SM70_FP8_QPN8_M32=1), else stays M6. True reproduces the fork's two copies; False = QPN8, no copy. At prod settings this keeps both Qwen models on M6 TurboMind (see the row above) |
| dflash2_verifier (auto) | n/a (no spec) | **off** | **on** ⚠ | Qualification is in models/config.py:1087-1134: arch Qwen3_5ForConditionalGeneration/CausalLM (Qwen3.6-35B-A3B is `Qwen3_5Moe*`, so it fails), fp16, hidden 5120, heads 24/4, head_dim 256, dflash, k=7, draft selector_top_k=16, PP1. Applied in sm70_dflash2.py:264-305 |
| shared_moe_overlap | n/a | **False** | n/a | VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP=0 envs.py:13347 → qwen3_next.py:245-249 disables the stream. It is forced to 1 only for Qwen4Exp (models/runtime_defaults.py:110-111). **This corrects my earlier proposal of True.** |
| greedy_token_fastpath (F12) | **True**: all-greedy batches (temperature 0) with no logprobs, penalties, allowed/bad words, thinking budget, active min-tokens/logit-bias/custom processors, structured output or spec decode take a vocab-parallel top-1 (per-rank argmax, all-gather of one (value, id) pair per row and rank) instead of the full-logit all-gather + Sampler; same token as upstream argmax | n/a in prod (DFlash2 spec decode never reaches the U4 hook; the model does not opt in) | n/a (same) | VLLM_SM70_GREEDY_TOKEN_FASTPATH=1 envs.py:7033 → `kernel_config.layer_execution.greedy_token_fastpath` (gpu_model_runner.py:1314); gate `_can_use_greedy_token_fastpath` :7646-7695, call `_sample_greedy_token_fastpath` :7845-7862 → Flint `get_top_tokens(fused_top1=False)` flint/model.py:571-578, :597-603 (the fused LM-head top-1 stays off: LM_HEAD_TOP1=0, see below). No override. Here: `vllm_sm70/sample_greedy.py` on the U4 hook (`sample_v1.py`), kill `VLLM_SM70_GREEDY_TOKEN_FASTPATH=0` (sampler section); not a graph field. Pair exchange: **NCCL all-gather** (fork: custom-AR kernel). **Deviation:** the custom-AR kernel (`_sm70_C::custom_ar_top1_argmax`, csrc/custom_ar/top1_custom_ar.cu) is off by default, opt-in `VLLM_SM70_TOP1_CUSTOM_AR=1` (env-only, sampler section): on c4130 it returned a different token than the all-gather on a checkpoint near-tie (m10 G3: 11749 vs 3426, the full-logit argmax), cause not found; the all-gather costs within noise (32K greedy 162.3 vs 164.0 tok/s). Opted in, it runs when TP>1 on CUDA, 1-128 rows, upstream's custom all-reduce enabled on the TP group (2/4/6/8 ranks, one node) and `_sm70_C` built with custom_ar. Fork VLLM_SM70_TOP1_CUSTOM_AR=1 envs.py:8315, policy_defaults.py:328 (pre-Ampere, TP>1) → logits_processor.py:817-847 → custom_all_reduce.py:977-998 → custom_all_reduce.cuh:1720-1750. `VLLM_SM70_TOP1_DEBUG=1` logs each rank's local pairs and the exchanged tokens per greedy step |

**Assumptions:** Qwen3.8-27B satisfies 5120/24/4/256 and its draft `incoai/Qwen3.8-27B-DFlash2@dedf8df` has selector_top_k=16. The evidence is the fork's release profile `sm70_profiles/qwen38_27b_nvfp4_dflash2.json`, which uses that draft revision and expects `dflash2_verifier`, plus `tests/models/gguf/dflash2_metadata.json:39` (selector_top_k 16). The architecture string is unverified, since no target config.json is available offline. **Caveat:** today the Qwen entries run the 1Cat 1.5.1 wheel, not cf9492a (HOST CLAUDE.md:73). The Qwen columns describe "same tree", as asked.

### Sub-features inside the bundled fields

**Flint fusions at cf9492a** (only qk_rope_kv belongs to M1; the others are recorded for M10 and Kai). All six are on, each behind its own runtime gate:

| Fusion | eff | Runtime rule | Evidence |
|---|---|---|---|
| qk-norm+RoPE+KV | on | fp16, FP16 KV, shape checks | envs.py:8622; flint/model.py:312-330 |
| fused router | on | M≤2, renormalized sigmoid E256/K6, no DP/PCP/EP, no gate fusion | envs.py:5400; sm70_moe.py:668; fused_moe/router/sm70_flint_router.py:160-230 |
| greedy top-1 (F12) | on | all-greedy, no logprobs/penalties/masks/bad words/active non-argmax processors, no structured output or spec decode; any batch size | envs.py:7033; gpu_model_runner.py:7646-7695, :7845-7862; flint/model.py:571-578, :597-603 |
| top-1 custom-AR exchange (F12) | on | TP>1, ≤128 rows, custom all-reduce enabled (single node, 2/4/6/8 ranks); else the pair all-gather | envs.py:8315; policy_defaults.py:328; logits_processor.py:817-847; custom_all_reduce.py:977-998; custom_all_reduce.cuh:1720-1750 |
| native sampler | on | ≤2 rows (MAX_ROWS=2), all-random, 1≤top_k≤20 (MAX_TOP_K=20), no logprobs, penalties, masks or bad words; one generator. Service default top_k=20 qualifies | envs.py:8749; execution_policy.py:555; flint/model.py:570; v1/sample/ops/sm70_topk_candidates.py:53-55,118-160 |
| TP2 AR+RMSNorm | on | TP2, PP1, fp16, hidden 2048, custom AR enabled, `fuse_allreduce_rms` not False → pass_config fuse_allreduce_rms/…_add_rms_tp2=True ⚠ (pass config) | envs.py:8687; execution_policy.py:720; models/config.py:308-370 |
| sink-fused decode | on | Requires FLASH_ATTN_V100, chosen at models/config.py:384-394 when `flash_attn_v100.SUPPORTS_SINKS` (built extension), head_dim 128 (:209), FP16 KV and no spec decoding. Otherwise TRITON_ATTN and no fusion | envs.py:10514; flash_v100.py:869,1313 |

**DFlash2 flags: Qwen3.8 (qualified) vs Qwen3.6 (not qualified).** These come from `SM70_DFLASH2_VERIFIER_DEFAULTS` (sm70_dflash2.py:20-39) applied at :281-305; every dflt is 0.

| Flag | 3.8 | 3.6 |
|---|---|---|
| FUSED_GDN_VERIFY, FUSED_GDN_COMBINED_SPLIT, CONTEXT_PIPELINE, CONTEXT_KV_GRAPH, QUANT_LM_HEAD, FP32_LOGITS, QPN8_RERANK, VERIFY_FASTPATH, FUSED_GDN_METADATA, FUSED_GDN_NORM, FUSED_GDN_SPLIT, FUSED_GEMMA_RMS, FIXED_GEMMA_RMS, FUSED_SMALLQ_METADATA, GROUPED_SMALLQ_METADATA, SPARSE_TARGET_REJECTION, SHARDED_CONTEXT_FC (17) + FP8_QPN8 | **1** ⚠ | 0 |
| TP2_GDN_BV2, FUSED_QKV_PACK, QPN8_RERANK_SHADOW | 0 | 0 |
| BF16_EMULATION (→ env-only VLLM_SM70_DFLASH2_EMULATION, M7) | **on** ⚠: declared None, but `read_execution_legacy` reads None as "1" (execution_policy.py:36-42) | on |
| PROPOSAL_TEMPERATURE_SCALE / TOP_P | 1.0 / 1.0 (the 0.8/0.9/0.95 overrides are GLM5-only: sm70_dflash2.py:41-61 and models/runtime_defaults.py:241-244) | same |

## (b) Knobs frozen as constants (`CONSTANTS` in tree/vllm_sm70/config/__init__.py)

When an env var is unset, the native ABI tuple passes `UNSET` and the **C++ default applies** (sm70_native.py:393-395 captures `UNSET`, :419-421 passes it). The envs.py declared defaults below are assumed to mirror the C++ defaults (unverified against csrc/sm70_policy.h). Overridden values are passed explicitly through EXTRA_BINDINGS (policy_defaults.py:197-215).

| Knob | dflt | **eff** | Changed? / source |
|---|---|---|---|
| VLLM_SM70_FP8_DENSE_TUNE_MAX_M | 16 (envs.py:5520) | **64** | ⚠ platforms/runtime_defaults.py:42, applied :263 at cc 7.0 (affects Qwen FP8 tuning: TurboMind autotunes shapes up to M=64) |
| VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M | 16 (envs.py:5555) | **64** | ⚠ platforms/runtime_defaults.py:43 (Flint) |
| VLLM_SM70_BATCH_GEMM_LAYOUTS | 0 (envs.py:9441) | **1** | ⚠ platforms/runtime_defaults.py:40. Batch GEMM layouts are prepared (sm70_turbomind.py:159); local shape checks still apply |
| VLLM_SM70_AWQ_WARMUP_MAX_M | – | 64 | :41 (AWQ only, n/a) |
| F16_DENSE_MAX_M | 64 (envs.py:8354) | 64 | no |
| NVFP4_MOE_TUNE_MAX_TOKENS | 128 | 128 | no |
| NVFP4_QPN2_PREFILL_MIN_M | 256 | 256 (QPN2 off) | no |
| FP8/NVFP4_TUNE_SMALL_SHAPES; AWQ_TUNE_SMALL_SHAPES (fp8 family) | 1; 0 | 1; 0 | no (vllm-sm70: 0 since #77; see "How vllm-sm70 carries T2, T8 and T9") |
| FP8 native: GROUPED_BMM_DECODE, PREFILL_FAST_SELECTOR, PREFILL_CUTLASS, PRESERVE_DEFAULT_SPLITS, QPN8_M16, QPN8_M32_CHUNKED, QPN8_M32_NATIVE | 1 | 1 | no (QPN8_* matter only on Qwen3.8, where qpn8 is on) |
| FP8 native: PRESERVE_DEFAULT_SPLITS_ONLY, REUSE_IMPORTED_CACHE, SAFE_FAST_SELECTOR | 0 | 0 | no |
| FP8 native: 0DOT3_DENSE_SELECTOR, MOE_PREPARE_VEC, MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH | None | C++ default | no (UNSET) |
| Sm70Fp8Config.block_qpn8 | True (kernel.py:386) | True (capability-gated) | no |
| NVFP4 native: QPN2_M16_NATIVE, QWEN38_TP4_M1_FAST_SELECTOR (TP4 only → inert) | 1 | 1 | no |
| FP8 MoE: BATCHED_GEMM 1, W13/W2_PER_EXPERT 0, PERMUTE_WITH_SCRATCH 1, LEGACY_SINGLE_TOKEN_COMPACT 1 (compact exact layout 1, native-unpermute 0, decomposed 0), single-token W13 dense / W2 indexed / reduce **weighted**, PERMUTE_FASTPATH 0, MOE_SINGLE_TOKEN_FASTPATH 0 | as listed | same | no |
| NVFP4 MoE routes (GROUPED_PREFILL, FAST_PREFILL, FUSED_SWIGLU_PREFILL, INDEXED_PREFILL, FUSED_BATCH_W13/W2, W2_DIRECT_REDUCE, QPN_M1/BATCH/MTP5, GROUPED_MTP5 = 1; GROUPED_DECODE, QPN_DYNAMIC, RAW_SCALE, GROUPED_EXPERT_ROWS = 0) | as listed | same | no. Most are exact Qwen3.8-NVFP4-TP4 contracts and stay inert for Flint TP2 (rule: the selector checks experts, top_k and TP shape) |
| FLINT_MOE_PERSISTENT | 0 | 0 | no |
| DFlash2 lookup (sm70_dflash2.py:144): adaptive 1, agree 0, cheap_context 0, entry_streak 2, long_min 6, nmin_tail 4, nstrong 6, search 2^30, sticky 3 | as listed | same | no |
| sm70_turbomind_workspace_bytes | 32 MiB (kernel.py:934) | 32 MiB | no |
| VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q | 16 (envs.py:10980) | 16 | Rule: raised to k+1 only if that exceeds 16 (platforms/runtime_defaults.py:559-565). DFlash2 k=7 gives 8, so no change |

### Behaviour-changing overrides outside Sm70Config (for Kai's compile/attention patches)

These are all applied because the Flash-V100 baseline holds: cc 7.0, VLLM_SM70_FLASH_ATTN_V100=1 (envs.py:9423), and the backend is None or FLASH_ATTN_V100 (platforms/runtime_defaults.py:358-367). The baseline holds for all three models, and for Flint only if SUPPORTS_SINKS is true (see above).

| Knob | dflt | eff | Line |
|---|---|---|---|
| VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH | 0 | **1** ⚠ | :413 (enforce_eager false) |
| VLLM_USE_AOT_COMPILE | 0 | **1** ⚠ | :657 and policy_defaults.py:156 |
| VLLM_SM70_LM_HEAD_TOP1 | 1 | **0** ⚠ | :647 |
| VLLM_SM70_GEMMA_RMS_NORM_COMPILE_NATIVE | – | **1** ⚠ | :407 |
| VLLM_MQ_BROADCASTER_MAX_CHUNKS | – | **64** ⚠ | :632 |
| VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS | – | **0** ⚠ (V1 runner only) | :641 |
| compile mode / cudagraph_mode | None | VLLM_COMPILE / FULL_AND_PIECEWISE ⚠ | :543-548 |
| capture sizes | None | SM70 no-MTP sizes for max_num_seqs; speculative sizes for k+1=8 ⚠ | :549-603 |
| ir_op_priority rms_norm / fused_add_rms_norm | – | [vllm_c, native] ⚠ | :613-620 |
| inductor combo_kernels / benchmark_combo_kernel | – | True / True ⚠ | :688-691 |

### How vllm-sm70 carries T2, T8 and T9 (2026-11-16)

| Fork row | vllm-sm70 | Default | Off / on |
|---|---|---|---|
| T8 capture sizes (`:200-251`, `:549-603`) | `vllm_sm70/graph_policy.py`, from `Sm70CudaPlatform.apply_config_platform_defaults` (UP `config/vllm.py:1825`, before `_set_cudagraph_sizes` at `:2013`) | **on**, fork tables | `VLLM_SM70_GRAPH_CAPTURE_SIZES=0` → upstream sizes |
| T9 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (`:636-642`, V1 only) | same hook; written to `os.environ` (the fork wrote PolicyDefaults) when unset, so the spawned workers read 0 | **on** (V1 runner only) | `VLLM_SM70_GRAPH_MEMORY_POLICY=0` → upstream estimate; an explicit `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` always wins |
| T2 `{NVFP4,FP8}_TUNE_SMALL_SHAPES=1` | C++ default 0 since #77; `services.vllm-sm70.turbomind.{tuneSmallShapes,gemmLutPath}` (`nix/module.nix`) set it with a persisted LUT | **off** | needs the c4130 T4c + n = 1/4/8/20 A/B (`sm70/csrc/turbomind/PROVENANCE.md`, "Nix (fork parity T2/G3)") |

T8/T9 apply under the fork's Flash-V100 baseline: Flash-V100 not killed
(`VLLM_SM70_FLASH_V100`), no other `--attention-backend`, no
`enforce_eager`, cudagraph mode not NONE, `TORCH_COMPILE_DISABLE` and
`VLLM_USE_BREAKABLE_CUDAGRAPH` off. Explicit `cudagraph_capture_sizes` or
`max_cudagraph_capture_size` in `--compilation-config` keep the user's sizes;
a dynamic speculative-decoding schedule keeps upstream's.

Sizes (`max-num-batched-tokens 8192`):

| Engine | Upstream v0.33 | SM70 (fork) |
|---|---|---|
| Flint, `max-num-seqs 20` (prod) | 1, 2, 4, 8, 16, 20, 24, 32, 40 | 1, 2, 4, 8, 16, 20 |
| no spec, `max-num-seqs 2` | 1, 2, 4 | 1, 2 |
| no spec, `max-num-seqs 64` | 1, 2, 4, then every 8 up to 128 (19 sizes) | 1, 2, 4, 8, 16, 32 |
| Qwen DFlash2 k=7, `max-num-seqs 8` (qwen3.8) | 1, 2, 4, then every 8 up to 128 (19 sizes) | 1, 2, 4, 8, 9, 16, 18, 24, 32, 48, 64, 96, 128, 160, 192, 256 |
| Qwen DFlash2 k=7, `max-num-seqs 20` (qwen3.6) | 1, 2, 4, every 8 up to 256, every 16 up to 320 (39 sizes) | 1, 2, 4, 8, 9, 16, 18, 24, 32, 48, 64, 96, 128, 160, 192, 256 |

Not ported: the fork's MTP split-draft sizes (`VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS`,
default off there) and the Flash-V100 small-q auto-raise (only when k + 1 > 16).
The V1 runner still rounds speculative sizes to multiples of k + 1 under FULL
decode graphs (UP `config/compilation.py:1502-1508`).

GPU gate before `latest`: Flint greedy identity and 4K/32K decode, n = 3..20
decode, startup with LMCache at `gpu-mem 0.90` and a 32K prefill (T9 takes
graph memory out of the 10% headroom that LMCache's ~2.2 GiB also uses), and
the KV token count in the log (it grows with T9).

## Where effective ≠ default (summary)

1. FP8 and NVFP4 DENSE_TUNE_MAX_M go 16→64, and BATCH_GEMM_LAYOUTS goes 0→1 (all models, cc 7.0).
2. On Qwen3.8 the 17-flag DFlash2 verifier bundle plus FP8_QPN8 go 0→1; Qwen3.6 stays at 0.
3. DFLASH2_BF16_EMULATION is declared None but is effectively on.
4. shared_moe_overlap is effectively False (my earlier proposal said True).
5. `sm70_skinny_moe` goes True→False through the serve arg.
6. Flint gets TP2 AR+RMSNorm pass flags and the FLASH_ATTN_V100 backend.
7. The compile policy changes: 0.0.3 graph, AOT=1, LM_HEAD_TOP1=0, combo kernels, vllm_c RMSNorm.

## How the plugin carries these values (M1 on K01)

- `sm70/vllm_sm70/config/__init__.py` freezes the values above as the `Sm70Config` defaults. The only overrides are `additional_config["sm70"]` (validated, unknown keys rejected) and the kill switches.
- The kill switches live in the `quant` and `models` sections of K01's `sm70/vllm_sm70/envs.py`, built with `kill_switch(name)`. Only `"0"` acts; any other value, including `"1"`, leaves the feature at its default. The environment can never turn a feature on, except the opt-in switches named below. `vllm_sm70.register()` merges them into `vllm.envs`, so they also enter `compile_factors()`.
  - quant: `VLLM_SM70_QUANT` (M2), `VLLM_SM70_FP8_TURBOMIND`, `VLLM_SM70_MODELOPT_NVFP4_GEMV`, `VLLM_SM70_FP8_MOE_DECODE`, `VLLM_SM70_FP8_QPN8`.
  - models: `VLLM_SM70_FLINT_MOE_QPN_DECODE`, `VLLM_SM70_FLINT_MOE_COMPACT` (new, not in fork; approved), `VLLM_SM70_FLINT_QK_ROPE_KV`, `VLLM_SM70_DFLASH2_VERIFIER`, `VLLM_SM70_DFLASH2_EMULATION` (M7), `VLLM_SM70_SHARED_MOE_OVERLAP`, `VLLM_SM70_INKLING_NVFP4_TM` (M12; env-only, no field; `"0"` keeps the Inkling 2 NVFP4 experts on the fp16 torch path, which holds every expert in fp16. By default they run on the TurboMind dense stage wherever `moe_tm.rejection` admits them: TP2, fp16, the Inkling 2 Mini expert shape and a loaded `_sm70_C` with the TurboMind ops on CUDA. Every other layer stays on torch and the load log names the reason), `VLLM_SM70_INKLING_TRITON_SCONV` (M12; env-only, no field, read on every eager call and fixed at capture in a replayed CUDA graph: `"0"` sends the Inkling 2 paged short convs and the prologue's K/V convs back from the kernels of `inkling2_sm70/ops/paged_conv.py`, one launch per call, to the torch ops; tensors off the GPU always take the torch ops), `VLLM_SM70_INKLING_TRITON_DECODE_ATTN` (M12; env-only, no field, read on every eager call and fixed at capture in a replayed CUDA graph: `"0"` sends the one-query rows of Inkling 2 attention back from the kernel of `inkling2_sm70/ops/paged_attention.py`, one launch per layer visit that reads only each row's own keys, to the torch ops; tensors off the GPU always take the torch ops), `VLLM_SM70_INKLING_DECODE_GRAPH` (M12; env-only, no field: `"0"` marks no Inkling 2 batch decode-only, so decode steps take the ops of mixed batches, the cache backends declare no CUDA-graph support, and the engine-config hook neither picks nor accepts a mode with FULL graphs).
  - models, opt-in (only `"1"` acts; default off, env-only, no field): `VLLM_SM70_INKLING_TRITON_NORMS` (M12; read on every eager call and fixed at capture in a replayed CUDA graph: the Inkling 2 norms, AttnRes mixes and message blends run the vendored Triton kernels `inkling2_sm70/ops/norm.py`, `attn_res.py`, `loop.py` instead of the torch ops; tensors off the GPU always take the torch ops).
  - sampler (Kai's section, but a `Sm70Config` field): `VLLM_SM70_GREEDY_TOKEN_FASTPATH` → `greedy_token_fastpath` (F12). Not in `GRAPH_FIELDS`: sampling runs eagerly. `VLLM_SM70_TOP1_CUSTOM_AR` (F12 pair exchange over the custom all-reduce buffers; opt-in, default off; env-only, no field) and `VLLM_SM70_TOP1_DEBUG` (diagnostic log) sit next to it.
- `Sm70Config.inkling2` is the Inkling 2 section (M12). Its fields are set only with `--additional-config '{"sm70": {"inkling2": {"<field>": <value>}}}'`; they have no env variable and are not in `GRAPH_FIELDS`. `max_image_encoder_tokens` (default 10817) is the largest vision-encoder sequence of one image, `merge * merge` patches per image token plus the class token: 2704 image tokens for Inkling 2 Mini. A request with a larger image is rejected, never resized. `visit_dump_dir` (default unset: no dump) names the directory for the per-visit state dump. Each forward writes `<dir>/rank<r>/forward<n>/visit<v>_layer<l>_pass<p>.pt`: the residual stream `[tokens, hidden]` after that visit, padded batch rows included.
- The Inkling 2 engine-config hook sets `--block-size 32` and `--no-async-scheduling` when they are unset and refuses any other block size, `--async-scheduling`, a KV cache dtype other than `auto`/`float16` and pipeline parallelism. An unset `cudagraph_mode` becomes `FULL_DECODE_ONLY`. With any mode that has FULL decode graphs (`FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`) an unset `max_cudagraph_capture_size` (with no capture sizes) becomes the smaller of 8 and `--max-num-seqs`, and set capture sizes above 8 are refused: decode-only batches up to that size replay a FULL CUDA graph and every other batch runs eagerly. Only a batch that replays takes the static decode ops; the metadata builders read that from `cudagraph_mode`, `--enforce-eager` and `max_cudagraph_capture_size`. `FULL` is refused, because the ops of mixed batches branch on the batch on the host. `PIECEWISE` and the mixed part of `FULL_AND_PIECEWISE` capture the model's ops as eager breaks, only with `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. With `VLLM_SM70_INKLING_DECODE_GRAPH=0` or a `visit_dump_dir`, every mode with FULL graphs is refused and an unset mode becomes `PIECEWISE` with `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `NONE` without it. `--enforce-eager` or `-cc.cudagraph_mode=none` turns graphs off. Its cache backends declare the `LBNHC` KV cache layout and full-graph support for uniform one-token decode batches.
- The TurboMind tuning constants (TUNE_MAX_M=64, BATCH_GEMM_LAYOUTS=1, QPN8 tiles, workspace) are op defaults in K02/K03. The Flint sink decode and AR+RMSNorm records belong to Kai (K14/K17); the router and sampler come in M10.
