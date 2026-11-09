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
| U3 | +20 | `fused_moe/runner/moe_runner.py:630, :907`, `platforms/interface.py` |
| U4 | +35 | `v1/worker/gpu_model_runner.py:563, :4208, :4306`, `platforms/interface.py` |
| U5 | 0 | none: K17's kernel is in `_sm70_C` and includes upstream's `csrc/custom_all_reduce.cuh` |
| **total** | **~75** (U5 adds none) | |

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
| M4 | 4 | NVFP4 MoE → TurboMind MoE via `FusedMoEMethodCls`; the decode fast path is wired only if K12's op is present; dense-stage calls with M·top_k ≤ 80 run compact slot groups, one TurboMind launch per stage (fork `nvfp4_sm70_moe.py:1619-1636`, #97) | Mara | M3, K03 | `VLLM_SM70_FLINT_MOE_QPN_DECODE=0`; `VLLM_SM70_FLINT_MOE_COMPACT=0` (new, not in fork) | GPU: MoE block vs reference; K14 sweep n = 3..10 near the fork (no batch>2 cliff) |
| M5 | 5 | `FlintSm70ForCausalLM` (FP16 rescale), registered with `register_model` and selected with `--hf-overrides architectures`; fused qk-norm/RoPE/KV is used only if K15 is present | Mara | M3, M4, U1 | `VLLM_SM70_FLINT_QK_ROPE_KV=0` (unfused) | GPU e2e quality suite: math 10/10, code 5/5, needle@32K 3/3, no gibberish at concurrency 20, glm47 tool/reasoning parse, LMCache prefix reuse |
| M6 | 5 | Qwen FP8 block-128 dense + MoE → TurboMind FP8 weight-only | Mara | M2, M4, K03 | `VLLM_SM70_FP8_TURBOMIND=0` → refuse with a clear error; `VLLM_SM70_FP8_MOE_DECODE=0` turns off the FP8 MoE single-token decode route | GPU e2e: Qwen3.6-35B-A3B and Qwen3.8-27B FP8 at TP2 within 32 GB/GPU; quality suite |

**Phase 2 exit:** Flint and Qwen3.6/3.8 serve on c4130 TP2 with `--dtype half` on TRITON_ATTN, and the M5 and M6 quality suites pass. Speed isn't gated yet.

## Phase 3: Flash-V100, GDN and Qwen speed

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| K05 | 1 | Flash-V100 kernels, lean (~8.7K) | Kai | K01 | – | GPU: `test_sm70_flash_v100_sinks_prefill` and `test_sm70_flash_v100_splitkv_n` pass |
| K06 | 2 | Flash-V100 attention backend (CUSTOM slot) | Kai | K05 | `VLLM_SM70_FLASH_V100=0` → TRITON_ATTN | GPU: Flint TP2 4K gives a greedy-identical 256-token continuation to TRITON_ATTN on 5 prompts; decode tok/s ≥ the fork's grouped two-launch number; LBNHC stride test passes |
| K07 | 3 | Flash-V100 spec-decode verify for Qwen DFlash2 | Kai | K06 | `VLLM_SM70_FLASH_V100_SPEC=0` | GPU: smallq exactness test; Qwen DFlash2 `accept_len` with Flash-V100 spec verify on within 2% (relative) of `VLLM_SM70_FLASH_V100_SPEC=0`, same prompts, rows 2 and 4 of `sm70/scripts/qwen_phase2/qwen_matrix.sh` (A/B in our build, not against the fork) |
| K08 | 1 | GDN FlashQLA native prefill + decode | Kai | K01 | `VLLM_SM70_GDN_DECODE_FLASHQLA=0`, `VLLM_SM70_GDN_FLASHQLA_PREFILL=0` | GPU: prefill/decode state vs upstream Triton, rel-L2 < 1e-3 |
| M11 | 2 | Qwen GDN layer subclass (`vllm_sm70/gdn/layer.py`, ~60 lines). Overrides `_forward_core_decode_non_spec` (upstream `qwen_gdn_linear_attn.py:1981`) and `_warmup_prefill_kernels` (:1308-1415) to call K08 | Mara | K08 | K08's switches | GPU: GDN layer output vs the upstream path within tolerance |
| K10 | 2 | opt: GDN TileLang prefill (needs a Nix tilelang) | Kai | K08 | `VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0` (default 0) | GPU: TileLang vs native state rel-L2 < 4e-4 |
| K09 | 3 | QPN8 block-FP8 decode GEMV (op only) | Kai | K03 | – (switch is M8's) | GPU: vs TM FP8 max-abs-rel < 1e-2 |
| K09b | 4 | **Paused, upside only** (Astro, 2026-11-09): the Phase 3 floors are met without it, and the fork's fused projection never runs on our block-FP8 Qwen3.8-27B-FP8 checkpoint at any TP (FORK `sm70_gdn_ba_verify.py:25` needs channel scales `(1, N)`; QPN8 prepare gives `[K/128, N/32]` for block scales, `fp8_qpn8_sm70.cu:1273-1310`). Fused QPN8 GDN verify projection (op only; M9b calls it): restore the fork's `fp8_qpn8_dispatch_ba_split_sm70_out` and `launch_fp8_qpn8_ba_split_sm70` (FORK `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu:1715`; K09 dropped them), with the TP2 Qwen3.8 shapes (qkvz N 8192, ba N 48) | Kai | K09 | – (switch is M9's) | GPU: fused qkv/z/b/a vs separate QPN8 qkvz GEMV + FP16 ba GEMM at x = [8, 5120] on both TP2 shards, max-abs-rel < 1e-2 |
| K18 | 3 | DFlash2 draft fp16 M≤8 GEMM + compact half top-k (`dflash2_fp16_m8_sm70.cu`, `compact_half_topk_sm70.cu`, `vllm_sm70/ops/dflash2.py`). **Default on for the qualified Qwen3.8 DFlash2 contract only** (Astro, 2026-11-09, on E3: −1.1 ms/step alone, −2.0 with M9a; probe accepted/step unchanged with K18 alone, −1.6% with M9a, net +3.9% per token); every other engine keeps upstream dispatch | Kai | K03 | `VLLM_SM70_DFLASH2_FP16=0` | GPU: vs `torch.matmul` max-abs-rel < 1e-2; draft acceptance unchanged |
| M8 | 6 | Wire QPN8 FP8 decode into Qwen (TP2 only: the fork's TP4 meaning of the switch is not carried) | Mara | M6, K09 | `VLLM_SM70_FP8_QPN8=0` | GPU: Qwen decode tok/s not worse than M6 alone; outputs unchanged |
| M9 | 7 | DFlash2 M8 verifier GEMVs, two parts, **0 upstream lines** (see "M9 hook points" below). **M9a:** the draft decoder's channel-FP8 QPN8 projections at M = 8/16/24/32, through U1 inside K18's dispatcher, flagged by M7's `decoder_layer_cls`. **M9b:** the target GDN fused `in_proj_qkvz` (FP8 QPN8) + `in_proj_ba` (FP16) M8 verify projection, as an extension of M11's `PluggableLayer` OOT class for `QwenGatedDeltaNetAttention` (#96), not a second registration. The port gates both on the DFlash2 verifier contract, a deliberate deviation (the fork gates draft FP8 on k == 7 plus its TP4 shape table). Either way only Qwen3.8-27B qualifies, not Qwen3.6-35B-A3B. **M9a is on by default** with K18 (E3 arm C, 2026-11-09). **Exit: A/B passed (E3); the ≥ 80 floor needs Block F.** **M9b is paused, upside only** (see K09b) | Mara (M9a draft PR: Kai) | M9a: M1, M7, K09, K18. M9b: M8, M11 (#96), K09b | `VLLM_SM70_DFLASH2_VERIFIER=0` (M9a rides on K18's hook, so `VLLM_SM70_DFLASH2_FP16=0` turns both off) | GPU, Qwen3.8-27B DFlash2 TP2, 4K decode: ≥ the K18-only A/B arm **and** ≥ the 1Cat 1.5.1 documented Qwen3.8 number (citation: Mara); acceptance rate unchanged within noise; `VLLM_SM70_DFLASH2_VERIFIER=0` restores K18 exactly |

### M9 hook points

The "M8 verifier GEMVs" at the fork (cf9492a) are three routes. All three serve the Qwen3.8-27B DFlash2 contract only: `sm70_dflash2_verifier_qualified` (FORK `models/config.py:1085-1134`) needs a `Qwen3_5ForConditionalGeneration`/`Qwen3_5ForCausalLM` target with hidden 5120, 24/4 heads × 256 and k = 7. Qwen3.6-35B-A3B is `Qwen3_5Moe*` with hidden 2048, so it never qualifies. Every shape table in these routes is also a Qwen3.8 TP4 shard.

| fork route | fork call site | what it is | upstream v0.33.0 path | hook (0 upstream lines) |
|---|---|---|---|---|
| draft FP16 M8 | `layers/linear.py:404-410` → `quantization/sm70_dflash2_fp16.py`; flagged at `models/qwen3_dflash2.py:278-297` (TP4 only) | DFlash2 draft qkv/o/gate_up/down/conv projections, M ≤ 8 | `UnquantizedLinearMethod.apply` → `self._gemm_impl` from `dispatch_unquantized_gemm` (`layers/linear.py:176`, `layers/utils.py:682`) | **Already landed as K18**: U1 `get_unquantized_gemm` → `ops/dflash2.py:dflash2_unquantized_gemm`, flagged by M7's `DFlash2Qwen3Model.decoder_layer_cls` (`qwen3_dflash.py:388`) |
| draft channel-FP8 M8 | `layers/linear.py:396-403` → `quantization/sm70_dflash2_fp8.py` (tried before FP16; prepare at `linear.py:344-352`); flag needs k == 7 (`qwen3_dflash2.py:295-297`) | the same draft projections quantized per output channel to E4M3, K09's QPN8 GEMV, M = 8/16/24/32 (1-4 requests) | same as above | **M9a**: the same U1 hook and K18 dispatcher (FP8 first, then FP16, then `F.linear`); M7 passes `fp8=True` to `mark_draft_linears` when the verifier resolves on. Ops are K09's `fp8_qpn8_prepare_sm70` and `fp8_qpn8_gemm_sm70_out` (channel scales, M ≤ 32) |
| target GDN fused verify projection | `layers/mamba/gdn/qwen_gdn_linear_attn.py:3079-3083` (also `:5876`, `:6033`, `models/qwen3_5.py:431`) → `quantization/sm70_gdn_ba_verify.py` | one launch for `in_proj_qkvz` (FP8 channel QPN8) + `in_proj_ba` (FP16) at x = [8, 5120]; shapes are the TP4 shard (qkvz N 4096, ba N 24) | `QwenGatedDeltaNetAttention.forward_cuda` Part 1, `qwen_gdn_linear_attn.py:1147-1148`; `forward_cuda` is bound to `_forward_method` at `:593` | **M9b**: `PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")` (`model_executor/custom_op.py:83`). `PluggableLayer.__new__` looks the OOT class up by the in-tree class *name*, not by the registered `"qwen_gated_delta_net_attention"` (`custom_op.py:46-59`). The subclass overrides `forward_cuda` and runs the fused op for an [8, K] fp16 input, else `super()`. One OOT class per name (a duplicate asserts, `:86`), so M9b extends M11's OOT class (#96) rather than registering its own |

**Why no U patch.** U1 already reaches every unquantized draft projection. `PluggableLayer` already lets a plugin replace the whole GDN layer class at construction. M7's subclass shows the factory route works for the draft. The `LINEAR_METHOD_BUILDERS` route doesn't apply: the draft is unquantized, and the target's qkvz is FP8 (M6/M8), not ModelOpt.

**What each part still needs.**
- M9a: no kernel work (K09 ships channel-scaled M ≤ 32). The draft PR adds `vllm_sm70/ops/dflash2_fp8.py` and about 20 lines in `ops/dflash2.py` and M7's layer. CPU tests cover row admission, split-K choice, the quantizer, the kill switch, route order, FP8 replacing the FP16 packing, the verifier gate and the op fake. Like K18 it rides K18's opt-in, because the fork never ran it at TP2. **Deliberate deviation:** the port flags draft FP8 on the verifier contract plus k == 7. The fork flags it on k == 7 alone (`qwen3_dflash2.py:295-297`) and limits it to Qwen3.8 through its TP4 `_SHAPES` table (`sm70_dflash2_fp8.py:23-28`). The covered models are the same; the gate is explicit and TP-independent.
- M9b: K09b (its own row above) restores the fork's `fp8_qpn8_dispatch_ba_split_sm70_out` and `launch_fp8_qpn8_ba_split_sm70` (FORK `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu:1715` and the launcher; K09 dropped both, see `sm70/csrc/qpn8/PROVENANCE.md`). It also needs qkvz to carry channel-scaled QPN8 codes (the M1 `fp8_qpn8` selector, which is reserved for this) and new TP2 shapes (qkvz N 8192, ba N 48). It lands after M11 (#96), as an extension of M11's OOT class. More than a day of work, plus GPU time. **Paused, upside only** (Astro, 2026-11-09): on the block-FP8 checkpoint the fork's route never runs, so it would need a channel-scale requantization of qkvz that neither the fork nor 1Cat does for this checkpoint. E2 puts the whole verify projection inside the 21.7 ms/step target FP8 GEMM bucket, of which the fusion saves only a launch.

**The exit test moves to Qwen3.8-27B.** No M9 route runs on Qwen3.6-35B-A3B at the fork, so the old reference (111 tok/s at 4K on the 3.6-35B) can't measure M9. At the fork's TP2 default even Qwen3.8 runs neither draft route, because both are flagged only at TP4 (`qwen3_dflash2.py:280`). So TP2 fork parity doesn't need M9: M9 is speed beyond the fork. **Decided (Astro, 2026-11-08):** the pass bar is DFlash2 4K decode ≥ the K18-only A/B arm and ≥ the 1Cat 1.5.1 documented Qwen3.8 number (Qwen speed is measured against the 1Cat docs, with no baseline runs; Mara has the citation). The acceptance rate must be unchanged within noise, and `VLLM_SM70_DFLASH2_VERIFIER=0` must restore K18 exactly.

**Phase 3 exit:** Qwen3.6/3.8 4K decode on c4130, as a 3-run median, is at or above the documented 1Cat 1.5.1 numbers, with Flash-V100, GDN FlashQLA and DFlash2 on: Qwen3.6 row 1 >= 66 tok/s and row 2 (with DFlash2) >= 111 tok/s; Qwen3.8 row 4 >= 80 tok/s (floor). Row 3 is record-only. Qwen3.8 cold 100K-token TTFT <= 66 s (the documented 1Cat number, per Mara's run list, section F). No fork or 1Cat baseline runs (Charlie, 2026-11-08).

## Phase 4: Flint fast paths

**D2 (decided): Flint runs on the V1 model runner.** Reasons: parity with the fork, U4 and K16 are V1-only, and V1 shows no host-side gap at batch size 1. The wrapper defaults `VLLM_USE_V2_MODEL_RUNNER=0` (#69). **D3:** U5 needs no upstream patch. K17 builds the fused kernel in `_sm70_C` against upstream's own `custom_all_reduce.cuh` and takes `ca_comm._ptr` as an int. R5 (the FULL-capture `cudaIpcGetMemHandle` crash under expandable segments) needs no upstream line either. K17's op copies its input into the communicator's eager buffer, which is IPC-registered at init, then publishes a fenced start flag (`staged_barrier_at_start`, fork 9f175fc6e). So K17's op never registers graph buffers. Upstream's plain all-reduce still does, unchanged.

**Upstream contract K17 depends on** (a rebase must keep it; the build fails loudly if the C++ side moves):
- Python `CustomAllreduce` (`vllm/distributed/device_communicators/custom_all_reduce.py`): `_ptr`, `disabled`, `rank`, `world_size`, `buffer_ptrs`, `max_size`, `should_custom_ar`, `custom_all_reduce`.
- C++ layout of `CustomAllreduce`, `Signal`, `RankData` and `RankSignals` in `csrc/custom_all_reduce.cuh`: pinned by `static_assert`. CMake and a CPU test also check the sha256 of the three upstream headers K17 includes against BASE+series.

| node | level | what | owner | depends | kill switch | exit test |
|---|---|---|---|---|---|---|
| U3 | 0 | MoE router/gate provider hook | Kai | – | no provider → upstream | CPU: with no provider, upstream behaviour is bit-identical |
| U4 | 0 | V1 sampler hook (V1 per D2) | Kai | – | no hook → upstream | CPU: with no hook, upstream behaviour is bit-identical |
| K11 | 3 | Flint NVFP4 M≤2 GEMV (op only; M3 feature-detects it) | Kai | K03 | – (switch is M3's `VLLM_SM70_MODELOPT_NVFP4_GEMV`) | fork tests listed in the carry map |
| K12 | 3 | Flint NVFP4 MoE QPN decode (op only; M4 feature-detects it) | Kai | K03 | – (switch is M4's `VLLM_SM70_FLINT_MOE_QPN_DECODE`) | fork tests listed in the carry map |
| K13 | 1 | Flint fused router (op) | Kai | K01, U3 | – (switch is M10's) | fork tests; greedy-identical with the switch off |
| K14 | 3 | Flint one-launch sink decode + counter fix (fork 7f12d21b8) | Kai | K06 | `VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0` | GPU: `test_decode_sinks_descending_capture_regression`; max-num-seqs sweep 2/3/4/8/20 coherent, each prompt twice |
| K15 | 1 | Flint q/k-norm + RoPE + KV write, Triton (op; M5 calls it) | Kai | K01 | – (switch is M5's `VLLM_SM70_FLINT_QK_ROPE_KV`) | GPU: vs unfused ops |
| K16 | 1 | Flint top-k candidate sampler (op) | Kai | K01, U4 | – (switch is M10's) | fork tests; greedy-identical with the switch off |
| M10 | 6 | Wire K13 (router) and K16 (sampler) into M5 | Mara | M5, K13, K16 | `VLLM_SM70_FLINT_FUSED_ROUTER=0`, `VLLM_SM70_FLINT_SAMPLER=0` | GPU e2e: greedy-identical with each switch off |
| K04 | 3 | opt: AWQ/int4 | Kai | K03 | `VLLM_SM70_TURBOMIND_AWQ=0` | GPU: AWQ cases in `test_sm70_turbomind_adapter` |
| U5 | 0 | No patch (D3). Guards in K17: `static_assert` on the layout of `CustomAllreduce`, `Signal`, `RankData` and `RankSignals`; a CMake check and a CPU test that the three upstream headers hash to BASE+series | Kai | – | – (K17's `VLLM_SM70_FLINT_AR_RMSNORM=0`) | CPU: header-hash test; GPU: FULL capture with expandable segments (R5) |
| K17 | 1 | Flint TP2 fused all-reduce + add + RMSNorm: kernel in `_sm70_C` (upstream `custom_all_reduce.cuh`, fork kernel ported to its layout) plus the composite post-grad pass. Needs sm70-ext to take vLLM's `csrc` include dir (Felix) | Kai | K01, U5 | `VLLM_SM70_FLINT_AR_RMSNORM=0` | GPU, 2 GPUs: `test_sm70_tp2_ar_fused_add_rms_norm` bitwise-equal to unfused; pass fires on the Flint graph; greedy-identical with the switch off |

**Phase 4 exit:** Flint TP2 4K decode is at least **170 tok/s** on c4130, with every fast path on. Kill switches (Charlie, 2026-11-08, "Teacher-forced envelope"): the 5 kernel-changing switches (`MODELOPT_NVFP4_GEMV`, `FLINT_MOE_QPN_DECODE`, `FLINT_QK_ROPE_KV`, `FLINT_FUSED_ROUTER`, `FLINT_ATTN_FUSED_DECODE`) pass if switch=0 vs default, teacher-forced on the M5 sequences, stays within the fork envelope (as #86 for K13). `MOE_COMPACT` (Astro, 2026-11-08) joins the teacher-forced group, since it changes kernels and #97 was gated teacher-forced (G2c, #100). `SAMPLER` and `AR_RMSNORM` stay greedy-identical.

## Phase 5: cutover

| node | what | owner | depends | exit test |
|---|---|---|---|---|
| C1 | Side-by-side bench and quality run vs the fork (cf9492a) on c4130: `verify_vllm.py`, `bench_vllm.py`, `kvtier_test.py`, and the SWE-bench slice (fork ≈37/50) | Astro + Mara | phase 2–4 exits | report attached to the cutover PR |
| C2 | LMCache 0.5.6 from sdist (cu12, sm_70) in the flake, loaded via `kv_connector_module_path`, no vLLM patch | Theo + Felix | N2 | `kvtier_test.py` passes |
| C3 | Switch `hosts/c4130/vllm` in charlie-horse/flake to this flake's `latest` branch (gated per `docs/release.md`; after the GC of old generations c4130 needs first) | Felix + Charlie | C1, C2 | `nixos-rebuild switch` succeeds; etcd health OK; service serves |
| C4 | Archive the fork (charlie-horse/vLLM) as reference only | Charlie | C3 | – |

## Series order for `patches/`

1. 0001–0099 build: 0001 trims the external projects that build nothing for sm_70 (`VLLM_SM70_TRIM_EXTERNALS=1`, set by nix/package.nix; unset or ≠1 → upstream). The remaining FetchContent sources are fed by Nix.
2. 0100–0199 platform: U1, U2, U3, U4. U5 has no patch.
3. 0500–0699 models: none. The M7 subclass route works without a patch.
4. 0700–0799 kv: none.

## Settled

- **Plugin route** for M nodes, so the models area adds 0 upstream lines (Charlie).
- **Kill-switch placement** (Kai and Mara): one switch per feature, read where the path is chosen, normally in M wiring. K14 (attention) and K17 (pass) are the exceptions, because their path is chosen inside Kai's code. K ops carry no runtime switch. Only `0` acts; env never turns a feature on. The seven Flint switch names stay exactly as the fork's operator docs have them: `MODELOPT_NVFP4_GEMV`, `FLINT_MOE_QPN_DECODE`, `FLINT_QK_ROPE_KV`, `FLINT_FUSED_ROUTER`, `FLINT_SAMPLER`, `FLINT_ATTN_FUSED_DECODE`, `FLINT_AR_RMSNORM` (each `VLLM_SM70_`-prefixed). An eighth, `FLINT_MOE_COMPACT` (M4 compact slot groups, #97), is **new, not in the fork**: the fork runs that route always on with no switch, so the default (on) is fork parity (Astro, 2026-11-08).
- **Phase 4 kill-switch criterion** (Charlie, 2026-11-08, "Teacher-forced envelope"): the 5 kernel-changing Flint switches pass if switch=0 vs default, teacher-forced on the M5 sequences, stays within the fork envelope (as #86 for K13). `MOE_COMPACT` (Astro, 2026-11-08) joins the teacher-forced group, since it changes kernels and #97 was gated teacher-forced (G2c, #100). `SAMPLER` and `AR_RMSNORM` stay greedy-identical. Supersedes the "greedy-identical" wording for the 5 switches.
- **K18 + M9a default on for Qwen3.8** (Astro, 2026-11-09): E3 on row 4 at TP2 (our flags, 5 bench runs and 3 probes per arm) measured 43.1 / 42.0 / 41.1 ms per step for default / K18 / K18+M9a, with non-overlapping ranges. Probe accepted/step (deterministic prompt) is 1.543 / 1.543 / 1.519: unchanged with K18 alone, −1.6% with M9a, so net +3.9% per token. Extra memory per GPU: ~0.85 GiB by default (M9a's FP8 copy; E3 KV 5.12 → 4.27 GiB), ~1.63 GiB with `VLLM_SM70_DFLASH2_VERIFIER=0` (FP16 copy). `VLLM_SM70_DFLASH2_FP16` moved from opt-in to kill switch; the default applies only to the qualified Qwen3.8 DFlash2 contract. K09b/M9b are paused, upside only.
- **K15 is the op and M5 calls it,** with `VLLM_SM70_FLINT_QK_ROPE_KV` as the only switch. `VLLM_SM70_FLINT_FUSIONS` is dropped.
- **M6-M (Marlin-SM70 FP8) is dropped for v1.** Revisit only if M6 is blocked.
- **D2:** V1 model runner for Flint (fork parity; U4/K16 are V1-only; no host gap at bs=1). `VLLM_USE_V2_MODEL_RUNNER=0` is the wrapper default. **D3:** U5 is no patch; K17 carries the kernel and the pass; R5 needs no upstream line.
- **kv:** no vLLM-side patches (Theo).
