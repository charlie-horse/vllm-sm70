# TurboMind SM70 GEMM core and ops: provenance (patches K02, K03)

Source: charlie-horse/vLLM at **cf9492a01**. The lmdeploy upstream commit isn't recorded in the fork (see `lmdeploy/NOTICE`). Line counts are `wc -l`. "fork last-touch" is `git log -1 --format=%h -- <fork path>`. Re-verify against a fork checkout with:

```
sm70/tools/check_turbomind_provenance.py --fork <fork checkout at cf9492a>
```

## Status values
| status | meaning |
|---|---|
| verbatim | byte-identical to the fork file, so `sha256` == `fork sha256` |
| trimmed | deletions only, plus the two `#ifdef VLLM_SM70_TM_AWQ` / `#endif` guard lines in `sm70_884_4.cu` |
| rewritten | edited, see the per-function list below |
| new | no fork counterpart |
| external | taken from upstream lmdeploy, not from the fork |

## What changed, function by function (for review against cf9492a)

### `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu` (trimmed)
- **Deleted:**
  - `#include "sm70_policy.h"`;
  - `PaddedActivationConfig` (GGUF only);
  - `GgufLatticeNarrowKernelImpl`, `GgufPrefillNarrowKernelImpl`;
  - `Qwen38Nvfp4W2CacheBKernelImpl`, `Qwen38Nvfp4W13TailN64KernelImpl`.
- **In `Registry::sm70_884_4()`:**
  - Deleted the GGUF lattice/LUT/bit-plane blocks, the `Config_U2_Affine` block and the GGUF `Config_U4_d` group-32 block (fork lines 111-342).
  - Deleted the `Config_MXF4` block (434-444).
  - Deleted the `Qwen38CacheB` / `Qwen38W13TailN64` entries of the `Config_NVF4` block (453-456).
  - Wrapped the `Config_U4_d` g128 and `Config_U4_g` blocks (370-432) in `#ifdef VLLM_SM70_TM_AWQ`.
- **Unchanged:**
  - `QuantizedBatch<fp4>` block;
  - the remaining NVF4 entries (incl. `ExactMnkKernelImpl` 8704x5120 / 5120x4352);
  - `Sm70Nvfp4PrescaledCounterpart`.

### `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` (trimmed)
- **In `Registry::sm70_884_8()`:** deleted the `Config_U8_Affine` dense and grouped block (fork lines 67-80).
- **Unchanged:** the fp8 `QuantizedBatch`, `Config_E4M3` and `Config_E4M3_Prescaled` blocks.

### `ops/gemm_runtime.h` (trimmed)
- **`TuneKeyKind`:** deleted `kMxfp4Dense`=6, `kMxfp4Moe`=8 and `kGguf*`=10..17. The remaining values keep their numbers.
- **Declarations deleted:**
  - `mxfp4_moe_compact_grouped_decode_enabled`, `mxfp4_moe_broadcast_input_decode_enabled`;
  - `mxfp4_moe_grouped_m8_enabled`, `mxfp4_moe_grouped_verifier_enabled`, `mxfp4_moe_grouped_m8_expert_rows_enabled`;
  - `select_mxfp4_dense_dispatch_policy`, `select_mxfp4_moe_dispatch_policy`.

### `ops/gemm_runtime.cpp` (rewritten)
- **Include:** `#include "sm70_policy.h"` became `#include "tm_config.h"`.
- **Tune keys** (`DenseTuneKey::operator==`, `DenseTuneKeyHash::operator()`, `MoeTuneKey::operator==`, `MoeTuneKeyHash::operator()`): dropped the `policy` member (`vllm::sm70::active_policy_key`) from the key, equality and hash.
- **`struct TurboMindRuntime`:**
  - no longer derives from `vllm::sm70::RuntimeResource`;
  - `close()` loses `override`;
  - `gemm_cache` is keyed by `int device`, not `(device, policy)`;
  - `imported_cache_devices` is a `std::set<int>`.
- **`runtime()`:** returns one function-static `TurboMindRuntime`. It no longer looks up a per-engine `current_native_runtime().resources["turbomind"]`.
- **Deleted functions:**
  - `mxfp4_tune_small_shapes_enabled`, `mxfp4_moe_compact_grouped_decode_enabled`, `mxfp4_moe_broadcast_input_decode_enabled`;
  - `mxfp4_moe_grouped_m8_enabled`, `mxfp4_moe_grouped_verifier_enabled`, `mxfp4_moe_grouped_m8_expert_rows_enabled`, `mxfp4_moe_grouped_m8_fast_selector_enabled`;
  - `mxfp4_dense_tune_max_m`;
  - `select_mxfp4_dense_dispatch_policy`, `select_mxfp4_moe_dispatch_policy`.
- **`has_imported_cache`, `select_dense_dispatch_policy_impl`, `select_moe_dispatch_policy_impl`:** imported-cache lookup by `device` only.
- **`get_workspace`:** dropped the `cached_owner` / `current_native_runtime().id` part of the thread-local fast path.
- **`get_gemm`:** dropped the owner and policy parts of the thread-local fast path and of the cache key.
- **`configure_gemm_workspace`:** dropped `TORCH_CHECK(vllm::sm70::active_runtime, ...)`. There's no engine owner any more.
- **`sm70_gemm_import_cache`:** `imported_cache_devices.insert(device)`.
- **`fp8_dense_tune_max_m`, `nvfp4_dense_tune_max_m`:** fallback 16 became `vllm::sm70::k{Fp8,Nvfp4}DenseTuneMaxMDefault` (64). These are the fork's c4130 auto-defaults.
- **Op registration:** `TORCH_LIBRARY_FRAGMENT(_C, m)` became `TORCH_LIBRARY_FRAGMENT(_sm70_C, m)` for `sm70_gemm_configure_workspace` and `sm70_gemm_workspace_storage`. This follows the `_sm70_C` decision in section A. The other K02 op entry points (`sm70_gemm_import_cache`, `sm70_gemm_export_cache`) have no registration here; K03's bindings file registers them.
- **Unchanged:**
  - all other fp8/nvfp4/awq/f16 knob readers (they now read `tm_config.h`);
  - `maybe_preserve_*`, `dispatch_policy_override`;
  - the `select_{dense,awq,fp8,nvfp4}_dense_*`, `select_{prefill_,fp8_,nvfp4_,}moe_*` selectors;
  - workspace allocation, the f16 weight cache, `sm70_gemm_export_cache`.

### `ops/tm_config.h` (rewritten from `csrc/sm70_policy.h`, `sm70_policy_fields.inc`, `sm70_runtime_state.h`)
- **42 knobs** (fork env names): 41 used by the K02 files, plus `batch_gemm_layouts` for the K03 Python. Defaults stay at the call sites.
- **c4130 defaults:** `kFp8DenseTuneMaxMDefault`=64 and `kNvfp4DenseTuneMaxMDefault`=64 (fork call-site default 16), `kBatchGemmLayoutsDefault`=true (fork env default 0), `kAwqWarmupMaxMDefault`=64 (recorded only). Source: fork `vllm/platforms/runtime_defaults.py:36-44`, applied for cc 7.0 via `:254` from `vllm/config/vllm.py:1426`.
- **One cached `getenv` pass:** `knobs()`, a function-local static.
- **Kept API** (fork names): `PolicyField`, `policy_name`, `policy_value`, `policy_atoi`, `policy_exact_one`, `DispatchOverride` / `policy_dispatch_override`, `GemmTargetPolicy` / `parse_gemm_targets` (copied) / `policy_gemm_targets`, `diagnostic_counter`.
- **Dropped:**
  - policy tokens and the ABI (`prepared_policy`, `PolicyScope`, `with_policy`, `legacy_policy`, `policy_key`, `active_policy*`);
  - `RuntimeState` / `active_runtime` / `enter_runtime` / `exit_runtime`;
  - the Marlin policy.
- **Behaviour change:** the fork's `dynamic_legacy=true` re-read of `TM_GEMM_CACHE_SUMMARY` and `VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC` on every call is gone. All knobs are fixed at first use.

### `ops/sm70_policy.h` (new)
A shim, `#include "tm_config.h"`, so `gemm.cu`, `dispatch_cache.cu` and `sm70_dflash_context.h` stay verbatim.

## K03: TurboMind ops moved out of `awq_sm70_gemm.cu`

Fork `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` at cf9492a (10,913 lines, last touch `899e9ed33`) is split into the files below. Commit "move ... verbatim" carries every function body byte for byte: `scripts/verify-port.py --port <tm files> --fork <fork awq_sm70_gemm.cu>` reports IDENTICAL=120 on that commit (with the `ns:` rules off, since these files keep `vllm::awq_sm70`). The next commit makes the adaptations listed after the move table. Line ranges include the comment block directly above each definition.

### Move table (fork `awq_sm70_gemm.cu` lines → new file)

**`ops/tm_ops_common_sm70.cuh`** (4 definitions)

| fork lines | definition |
|---|---|
| 46-62 | `sm70_silu_and_mul_fp16_kernel` |
| 64-75 | `sm70_silu_and_mul_fp16_out` |
| 77-94 | `sm70_silu_and_mul_interleaved_fp16_kernel` |
| 96-108 | `sm70_silu_and_mul_interleaved_fp16_out` |

**`ops/tm_f16_sm70.cu`** (15 definitions)

| fork lines | definition |
|---|---|
| 193-208 | `silu_and_mul_interleaved` |
| 971-976 | `validate_f16_weight` |
| 978-994 | `validate_f16_input` |
| 996-1025 | `validate_f16_gate_mul_input` |
| 1115-1120 | `warp_reduce_sum` |
| 1690-1749 | `sm70_f16_gate_mul_kernel` |
| 2418-2426 | `interleave_gated_silu_cols` |
| 4051-4061 | `sm70_f16_prepare` |
| 4849-4936 | `sm70_f16_gemm_out` |
| 6016-6039 | `sm70_f16_gate_mul_out` |
| 6053-6073 | `sm70_f16_gemm` |
| 6249-6251 | `sm70_f16_prepare` |
| 6260-6262 | `sm70_f16_gemm` |
| 6391-6394 | `sm70_f16_gemm_out` |
| 6550-6553 | `sm70_f16_gate_mul_out` |

**`ops/tm_fp8_sm70.cu`** (20 definitions)

| fork lines | definition |
|---|---|
| 957-969 | `validate_fp8_inputs` |
| 2318-2318 | `kFp8PrefillDequantThreads` |
| 2319-2319 | `kFp8QuantValuesPerWord` |
| 2320-2320 | `kFp8OutputColumnsPerTile` |
| 2322-2371 | `fp8_sm70_dequantize_kernel` |
| 2373-2416 | `fp8_sm70_dequantize_out` |
| 3679-3810 | `fp8_sm70_prepare` |
| 4254-4353 | `fp8_gemm_sm70_out` |
| 4355-4362 | `sm70_fp8_prefill_cutlass_gated_silu_enabled` |
| 4364-4400 | `fp8_gemm_sm70_prefill_dispatch_out` |
| 4776-4834 | `fp8_gemm_sm70_out_auto` |
| 4836-4847 | `fp8_gemm_sm70_out_meta` |
| 6218-6224 | `fp8_sm70_prepare` |
| 6226-6231 | `fp8_sm70_dequantize_out` |
| 6287-6294 | `fp8_gemm_sm70_out` |
| 6296-6304 | `fp8_gemm_sm70_prefill_prescaled_out` |
| 6306-6315 | `fp8_gemm_sm70_prescaled_m1_out` |
| 6317-6324 | `fp8_gemm_sm70_prefill_dispatch_out` |
| 6376-6381 | `fp8_gemm_sm70_out_auto` |
| 6383-6389 | `fp8_gemm_sm70_out_meta` |

**`ops/tm_nvfp4_sm70.cu`** (6 definitions)

| fork lines | definition |
|---|---|
| 943-955 | `validate_nvfp4_inputs` |
| 3933-4049 | `nvfp4_sm70_prepare` |
| 4479-4563 | `nvfp4_gemm_sm70_out` |
| 6241-6247 | `nvfp4_sm70_prepare` |
| 6334-6340 | `nvfp4_gemm_sm70_out` |
| 6342-6348 | `nvfp4_gemm_sm70_prescaled_out` |

**`ops/tm_moe_sm70.cu`** (55 definitions)

| fork lines | definition |
|---|---|
| 110-112 | `sm70_profile_trace_enabled` |
| 114-125 | `sm70_scalar_type_name` |
| 127-138 | `sm70_capture_status_name` |
| 140-151 | `sm70_capture_status_bit` |
| 153-174 | `maybe_log_sm70_moe_route_once` |
| 6082-6092 | `select_awq_moe_dispatch_policy` |
| 6264-6267 | `sm70_fp8_moe_prepare_vec_enabled` |
| 6563-6577 | `awq_moe_fill_strided_ptrs_kernel` |
| 6581-6633 | `awq_moe_build_strided_ptrs` |
| 6793-6860 | `awq_moe_single_token_exact_layout_prepare_kernel` |
| 6862-6935 | `awq_moe_single_token_exact_layout_prepare` |
| 6937-6942 | `awq_moe_gemm_sm70_out` |
| 6944-6948 | `awq_moe_gemm_sm70_per_expert_dispatch_out` |
| 6950-6957 | `awq_moe_gemm_sm70_out_impl` |
| 6959-7099 | `awq_moe_single_token_prepare_kernel` |
| 7318-7333 | `awq_moe_single_token_weighted_reduce_kernel` |
| 7335-7366 | `awq_moe_single_token_weighted_reduce_half2_kernel` |
| 7368-7442 | `awq_moe_single_token_weighted_reduce_out` |
| 7444-7448 | `fp8_e4m3_to_float` |
| 7450-7499 | `fp8_moe_w2_direct_reduce_kernel` |
| 7501-7580 | `fp8_moe_w2_direct_reduce_sm70_out` |
| 7582-7761 | `awq_moe_single_token_sm70_out` |
| 7763-7946 | `awq_moe_gemm_sm70_out_impl` |
| 7948-7957 | `awq_moe_gemm_sm70_out` |
| 7959-7967 | `awq_moe_gemm_sm70_per_expert_dispatch_out` |
| 7969-8013 | `awq_moe_indexed_dense_w13_sm70_out` |
| 8030-8078 | `awq_moe_dense_stage_sm70_out` |
| 8224-8254 | `awq_moe_build_active_expert_segments_kernel` |
| 8335-8411 | `awq_moe_active_dense_stage_sm70_out` |
| 8415-8426 | `awq_moe_lower_bound_source_row` |
| 8428-8448 | `awq_moe_chunk_ranges_kernel` |
| 8450-8458 | `awq_moe_prefix_chunk_offsets_kernel` |
| 8460-8477 | `awq_moe_fill_chunk_indices_kernel` |
| 8479-8504 | `awq_moe_chunk_weighted_reduce_kernel` |
| 8508-8676 | `awq_moe_chunked_w2_sm70_out` |
| 8678-8721 | `awq_moe_single_token_dense_stage_sm70_out` |
| 8723-8764 | `awq_moe_single_token_indexed_dense_stage_sm70_out` |
| 8766-8860 | `awq_moe_single_token_dense_w13_sm70_out` |
| 8862-8957 | `awq_moe_single_token_indexed_dense_w13_sm70_out` |
| 8959-9067 | `awq_moe_single_token_compact_dense_w13_sm70_out` |
| 9069-9270 | `fp8_moe_gemm_sm70_out_impl` |
| 9272-9282 | `fp8_moe_gemm_sm70_out` |
| 9284-9293 | `fp8_moe_gemm_sm70_per_expert_dispatch_out` |
| 9295-9345 | `fp8_moe_dense_stage_sm70_out` |
| 9570-9709 | `nvfp4_moe_gemm_sm70_out_impl` |
| 9711-9756 | `nvfp4_moe_indexed_dense_stage_sm70_out_impl` |
| 9758-9766 | `nvfp4_moe_indexed_dense_stage_sm70_out` |
| 9768-9776 | `nvfp4_moe_indexed_fused_swiglu_sm70_out` |
| 9778-9858 | `nvfp4_moe_dense_stage_sm70_out` |
| 9979-10024 | `fp8_moe_single_token_dense_stage_sm70_out` |
| 10026-10069 | `fp8_moe_single_token_indexed_dense_stage_sm70_out` |
| 10071-10166 | `fp8_moe_single_token_dense_w13_sm70_out` |
| 10168-10264 | `fp8_moe_single_token_indexed_dense_w13_sm70_out` |
| 10266-10376 | `fp8_moe_single_token_compact_dense_w13_sm70_out` |
| 10378-10707 | `fp8_moe_single_token_sm70_out` |

**`ops/tm_awq_sm70.cu`** (24 definitions)

| fork lines | definition |
|---|---|
| 40-41 | `unpack_awq_gemm` |
| 898-910 | `validate_awq_inputs` |
| 912-927 | `validate_uint4_inputs` |
| 2229-2229 | `kAwqPrefillDequantThreads` |
| 2230-2230 | `kAwqQuantValuesPerWord` |
| 2231-2231 | `kAwqOutputColumnsPerTile` |
| 2233-2237 | `awq_prefill_physical_nibble` |
| 2239-2272 | `awq_sm70_dequantize_kernel` |
| 2274-2316 | `awq_sm70_dequantize_out` |
| 2428-2439 | `pack_awq_compact_source_kernel` |
| 2441-2452 | `compact_awq_stats_kernel` |
| 3327-3520 | `awq_sm70_prepare_impl` |
| 3522-3529 | `awq_sm70_prepare` |
| 3531-3536 | `awq_sm70_prepare_compact` |
| 3538-3677 | `uint4_sm70_prepare` |
| 4063-4164 | `awq_gemm_sm70_out` |
| 4166-4172 | `awq_gemm_sm70_out` |
| 6041-6051 | `awq_gemm_sm70` |
| 6186-6193 | `awq_sm70_prepare` |
| 6195-6200 | `awq_sm70_prepare_compact` |
| 6202-6207 | `awq_sm70_dequantize_out` |
| 6209-6216 | `uint4_sm70_prepare` |
| 6253-6258 | `awq_gemm_sm70` |
| 6269-6275 | `awq_gemm_sm70_out` |

`ops/qwen38_prefill_cutlass.cuh` is verbatim. `ops/qwen38_prefill_cutlass.cu` is verbatim apart from its first line (`sm70_policy.h` → `tm_config.h`).

### Not carried from `awq_sm70_gemm.cu`

| bucket | fork lines (main ranges) | where it goes |
|---|---|---|
| lm_head top1/top20, rerank, sampler, dynamic draft vocab | 1027-1113, 1751-2226, 4938-6014, 6442-6548 | not K03 (Rosa's sampler, or dropped) |
| GGUF | 2454-3326, 6097-6184 | dropped |
| GLM mhc/kda, glm53 MoE permute | 889-896, 1129-1689, 6396-6440, 8082-8333 | dropped |
| MXFP4 dense + MoE | 929-941, 3812-3931, 4402-4477, 9351-9568, 9860-9977, 6233-6239, 6326-6332 | dropped |
| NVFP4 raw/warp/h2 GEMV | 565-885, 4565-4774, 6350-6374 | dropped (K11 carries `nvfp4_tm_gemv`) |
| `awq_gemm_sm70_out_tile_reduce` | 4174-4252, 6277-6285 | dropped (needs custom all-reduce) |
| `fp8_moe_single_token_router_sm70_out` | 10709-10911 | dropped (`VLLM_ENABLE_SM70_TURBOMIND_EXPERIMENTAL_MOE` only) |
| global `silu_and_mul` | 178-191 | dropped (upstream has the op) |
| `Top1Epilogue`/`Top20Epilogue`, `Sm70_s884_top1/top20` | 212-557 | dropped (lm_head top-k) |

### Adaptations (commit after the moves)
- **`ops/tm_ops_common_sm70.cuh`:** the fork's include block (lines 1-37) with three changes: `tm_config.h` replaces `sm70_policy.h`; `custom_all_reduce.cuh` is dropped (only the dropped tile-reduce used it); `<cutlass/half.h>` is dropped (no `cutlass::` use left, so only `qwen38_prefill_cutlass.cu` needs CUTLASS).
- **`ops/tm_f16_sm70.cu`:** the global `silu_and_mul_interleaved` (fork 193-208) is now `vllm_sm70::silu_and_mul_interleaved`, with the body unchanged, so it can't clash with upstream symbols.
- **Every `ops/tm_*_sm70.cu`:**
  - `#include <torch/library.h>`;
  - one `TORCH_LIBRARY_FRAGMENT(_sm70_C, m)` with the fork schemas (fork `csrc/torch_bindings.cpp` line in a comment above each def) minus `str? native_policy=None`;
  - one `TORCH_LIBRARY_IMPL(_sm70_C, CUDA, m)`;
  - the AWQ ops are registered only under `VLLM_SM70_TM_AWQ`.
  - Schema change (Mara's K03 contract): `nvfp4_sm70_prepare` defaults `group_size=16, interleave_gated_silu=False`, and `nvfp4_gemm_sm70_out` defaults `gated_silu=False`. Call sites that pass every argument are unaffected.
- **`ops/tm_moe_sm70.cu`:** the fork's `#if defined(ENABLE_SM70_TURBOMIND)` wrapper (6559/10913) is gone, since the plugin always builds TurboMind. AWQ-only MoE ops sit under `#ifdef VLLM_SM70_TM_AWQ` in their fork order.
- **`ops/tm_awq_sm70.cu`:** the AWQ dense ops, entirely under `#ifdef VLLM_SM70_TM_AWQ` (K04).
- **`ops/tm_config.h`:** five more knobs.
  - K03: `profile_trace` (`VLLM_SM70_PROFILE_TRACE`), `fp8_prefill_cutlass`, `fp8_moe_prepare_vec`.
  - K04: `awq_moe_batched_active_exact_w2`, `awq_qwen38_moe_compact_grouped_decode`.
  - Env names are the fork's (`sm70_policy_fields.inc`).
- **`ops/gemm_runtime.cpp`** (K03 changes on top of K02):
  - `TurboMindRuntime` gains `generation` and `release()`.
  - `get_workspace()` and `get_gemm()` drop their thread-local fast-path pointers when `generation` changes.
  - New `tm_runtime_close()`.
  - Registrations: `sm70_gemm_import_cache` / `sm70_gemm_export_cache` (fork `torch_bindings.cpp:1155,1161`, minus `native_policy`) with CUDA impls, and `tm_runtime_close() -> ()` with a CompositeExplicitAutograd impl.
  - `#include <torch/library.h>`, `<atomic>`.
- **`sources.cmake`:** now appends to K01's `SM70_CXX_SRCS` / `SM70_CUDA_SRCS` / `SM70_INCLUDE_DIRS`. The K02 version set only `VLLM_SM70_TM_*`, which K01 never read. It adds the K03 sources and requires `VLLM_SM70_CUTLASS_DIR`.

### Python: `vllm_sm70/ops/tm.py`
Ported from fork `vllm/_sm70/{auxiliary,linear,moe}.py`. Each block is marked `# fork vllm/_sm70/<file>:<lines>`.

- **Mechanical rewrites** (the `scripts/verify-port.toml` python rules):
  - `call_native(_op(x), native_policy, ...)` → `_op(x)(...)`;
  - `native_policy` parameters dropped;
  - `"_C::"` → `"_sm70_C::"`;
  - `hasattr(torch.ops._C, x)` → `has_op(x)`;
  - `@direct_native("_C")` removed.
- **Real changes:**
  - The six prepare fakes create `meta` on CPU, as the C++ ops do; the fork faked it on the weight's device.
  - `nvfp4_sm70_prepare(..., group_size=16)` gets a default and a docstring for Mara's contract.
  - The `nvfp4` fakes take the schema defaults.
  - New fakes for `fp8_gemm_sm70_out_auto`, `fp8_gemm_sm70_out_meta`, `sm70_gemm_import_cache`, `sm70_gemm_export_cache`, `sm70_gemm_configure_workspace`, `sm70_gemm_workspace_storage` and `tm_runtime_close`.
  - New wrappers for the last three.

| fork file | lines | op | kind |
|---|---|---|---|
| `vllm/_sm70/auxiliary.py` | 10-11 | `silu_and_mul_interleaved` | wrapper |
| `vllm/_sm70/auxiliary.py` | 14-19 | `silu_and_mul_interleaved` | fake |
| `vllm/_sm70/auxiliary.py` | 706-711 | `sm70_f16_gate_mul_out` | wrapper |
| `vllm/_sm70/auxiliary.py` | 714-722 | `sm70_f16_gate_mul_out` | fake |
| `vllm/_sm70/linear.py` | 78-94 | `awq_sm70_prepare` | wrapper |
| `vllm/_sm70/linear.py` | 97-118 | `awq_sm70_prepare` | fake |
| `vllm/_sm70/linear.py` | 121-137 | `awq_sm70_prepare_compact` | wrapper |
| `vllm/_sm70/linear.py` | 140-161 | `awq_sm70_prepare_compact` | fake |
| `vllm/_sm70/linear.py` | 164-173 | `awq_sm70_dequantize_out` | wrapper |
| `vllm/_sm70/linear.py` | 176-187 | `awq_sm70_dequantize_out` | fake |
| `vllm/_sm70/linear.py` | 190-206 | `uint4_sm70_prepare` | wrapper |
| `vllm/_sm70/linear.py` | 209-235 | `uint4_sm70_prepare` | fake |
| `vllm/_sm70/linear.py` | 238-252 | `fp8_sm70_prepare` | wrapper |
| `vllm/_sm70/linear.py` | 255-276 | `fp8_sm70_prepare` | fake |
| `vllm/_sm70/linear.py` | 279-288 | `fp8_sm70_dequantize_out` | wrapper |
| `vllm/_sm70/linear.py` | 291-302 | `fp8_sm70_dequantize_out` | fake |
| `vllm/_sm70/linear.py` | 350-364 | `nvfp4_sm70_prepare` | wrapper |
| `vllm/_sm70/linear.py` | 367-392 | `nvfp4_sm70_prepare` | fake |
| `vllm/_sm70/linear.py` | 395-398 | `sm70_f16_prepare` | wrapper |
| `vllm/_sm70/linear.py` | 401-408 | `sm70_f16_prepare` | fake |
| `vllm/_sm70/linear.py` | 432-450 | `awq_gemm_sm70` | wrapper |
| `vllm/_sm70/linear.py` | 453-470 | `awq_gemm_sm70` | fake |
| `vllm/_sm70/linear.py` | 473-495 | `awq_gemm_sm70_out` | wrapper |
| `vllm/_sm70/linear.py` | 498-512 | `awq_gemm_sm70_out` | fake |
| `vllm/_sm70/linear.py` | 572-610 | `fp8_gemm_sm70_out` | wrapper |
| `vllm/_sm70/linear.py` | 613-628 | `fp8_gemm_sm70_out` | fake |
| `vllm/_sm70/linear.py` | 631-651 | `fp8_gemm_sm70_prefill_prescaled_out` | wrapper |
| `vllm/_sm70/linear.py` | 654-667 | `fp8_gemm_sm70_prefill_prescaled_out` | fake |
| `vllm/_sm70/linear.py` | 670-690 | `fp8_gemm_sm70_prescaled_m1_out` | wrapper |
| `vllm/_sm70/linear.py` | 693-706 | `fp8_gemm_sm70_prescaled_m1_out` | fake |
| `vllm/_sm70/linear.py` | 1672-1698 | `fp8_gemm_sm70_prefill_dispatch_out` | wrapper |
| `vllm/_sm70/linear.py` | 1701-1718 | `fp8_gemm_sm70_prefill_dispatch_out` | fake |
| `vllm/_sm70/linear.py` | 1763-1785 | `nvfp4_gemm_sm70_out` | wrapper |
| `vllm/_sm70/linear.py` | 1788-1802 | `nvfp4_gemm_sm70_out` | fake |
| `vllm/_sm70/linear.py` | 1848-1870 | `nvfp4_gemm_sm70_prescaled_out` | wrapper |
| `vllm/_sm70/linear.py` | 1873-1887 | `nvfp4_gemm_sm70_prescaled_out` | fake |
| `vllm/_sm70/linear.py` | 2001-2010 | `fp8_gemm_sm70_out_auto` | wrapper |
| `vllm/_sm70/linear.py` | 2013-2031 | `fp8_gemm_sm70_out_meta` | wrapper |
| `vllm/_sm70/linear.py` | 2034-2037 | `sm70_f16_gemm` | wrapper |
| `vllm/_sm70/linear.py` | 2040-2052 | `sm70_f16_gemm` | fake |
| `vllm/_sm70/linear.py` | 2055-2065 | `sm70_f16_gemm_out` | wrapper |
| `vllm/_sm70/linear.py` | 2068-2079 | `sm70_f16_gemm_out` | fake |
| `vllm/_sm70/linear.py` | 2103-2106 | `sm70_gemm_import_cache` | wrapper |
| `vllm/_sm70/linear.py` | 2109-2112 | `sm70_gemm_export_cache` | wrapper |
| `vllm/_sm70/moe.py` | 837-863 | `nvfp4_moe_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 866-882 | `nvfp4_moe_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 885-913 | `nvfp4_moe_indexed_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 916-933 | `nvfp4_moe_indexed_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 936-964 | `nvfp4_moe_indexed_fused_swiglu_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 967-984 | `nvfp4_moe_indexed_fused_swiglu_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1083-1099 | `awq_moe_build_strided_ptrs` | wrapper |
| `vllm/_sm70/moe.py` | 1102-1116 | `awq_moe_build_strided_ptrs` | fake |
| `vllm/_sm70/moe.py` | 1119-1145 | `awq_moe_gemm_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1148-1174 | `awq_moe_gemm_sm70_per_expert_dispatch_out` | wrapper |
| `vllm/_sm70/moe.py` | 1177-1193 | `awq_moe_gemm_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1196-1212 | `awq_moe_gemm_sm70_per_expert_dispatch_out` | fake |
| `vllm/_sm70/moe.py` | 1215-1242 | `awq_moe_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1245-1261 | `awq_moe_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1264-1293 | `awq_moe_indexed_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1296-1313 | `awq_moe_indexed_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1316-1345 | `awq_moe_active_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1348-1365 | `awq_moe_active_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1368-1417 | `awq_moe_chunked_w2_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1420-1447 | `awq_moe_chunked_w2_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1450-1477 | `awq_moe_single_token_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1480-1496 | `awq_moe_single_token_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1499-1526 | `awq_moe_single_token_indexed_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1529-1545 | `awq_moe_single_token_indexed_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1548-1583 | `awq_moe_single_token_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1586-1606 | `awq_moe_single_token_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1609-1644 | `awq_moe_single_token_indexed_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1647-1667 | `awq_moe_single_token_indexed_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1670-1709 | `awq_moe_single_token_compact_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1712-1734 | `awq_moe_single_token_compact_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1737-1758 | `awq_moe_single_token_exact_layout_prepare` | wrapper |
| `vllm/_sm70/moe.py` | 1761-1774 | `awq_moe_single_token_exact_layout_prepare` | fake |
| `vllm/_sm70/moe.py` | 1777-1796 | `awq_moe_single_token_weighted_reduce_out` | wrapper |
| `vllm/_sm70/moe.py` | 1799-1811 | `awq_moe_single_token_weighted_reduce_out` | fake |
| `vllm/_sm70/moe.py` | 1860-1915 | `awq_moe_single_token_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 1918-1973 | `awq_moe_single_token_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 1976-2002 | `fp8_moe_gemm_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2005-2031 | `fp8_moe_gemm_sm70_per_expert_dispatch_out` | wrapper |
| `vllm/_sm70/moe.py` | 2034-2050 | `fp8_moe_gemm_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2053-2069 | `fp8_moe_gemm_sm70_per_expert_dispatch_out` | fake |
| `vllm/_sm70/moe.py` | 2072-2099 | `fp8_moe_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2102-2118 | `fp8_moe_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2121-2148 | `fp8_moe_single_token_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2151-2167 | `fp8_moe_single_token_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2170-2197 | `fp8_moe_single_token_indexed_dense_stage_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2200-2216 | `fp8_moe_single_token_indexed_dense_stage_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2219-2254 | `fp8_moe_single_token_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2257-2277 | `fp8_moe_single_token_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2280-2315 | `fp8_moe_single_token_indexed_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2318-2338 | `fp8_moe_single_token_indexed_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2341-2380 | `fp8_moe_single_token_compact_dense_w13_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2383-2405 | `fp8_moe_single_token_compact_dense_w13_sm70_out` | fake |
| `vllm/_sm70/moe.py` | 2408-2485 | `fp8_moe_single_token_sm70_out` | wrapper |
| `vllm/_sm70/moe.py` | 2488-2529 | `fp8_moe_single_token_sm70_out` | fake |

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---|---|---|---|---|
| `sm70/csrc/turbomind/lmdeploy/LICENSE` | `-` | external | 201 | - | - | `811e05b43b0d985814c7ffd0a1d0b93e9166e6f0271245d540302505734528ba` | `-` |
| `sm70/csrc/turbomind/lmdeploy/NOTICE` | `-` | new | 34 | - | - | `1e5a97d67599a0be5bafd15ff9a5670f0e5fc4289073495dc86d98c11ce8e020` | `-` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/allocator.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/allocator.cc` | verbatim | 159 | 159 | ee0bc6a87 | `2805256f5ad037c0cf1e729bd1f292a226fd3f760377482eedee7b2e777128be` | `2805256f5ad037c0cf1e729bd1f292a226fd3f760377482eedee7b2e777128be` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/allocator.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/allocator.h` | verbatim | 247 | 247 | ee0bc6a87 | `a8ab1702fdd9a1488a30c778b1c32067aef50d500c52c4cfe60348729e8f22fc` | `a8ab1702fdd9a1488a30c778b1c32067aef50d500c52c4cfe60348729e8f22fc` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/buffer.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/buffer.cc` | verbatim | 98 | 98 | ee0bc6a87 | `2eac88e78394f5ebdd885496801ca2f8d5a83e32e53d80692575b08f87f5e13a` | `2eac88e78394f5ebdd885496801ca2f8d5a83e32e53d80692575b08f87f5e13a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/buffer.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/buffer.h` | verbatim | 390 | 390 | ee0bc6a87 | `9a95cd976450e9c7491b62b7eea3d520b1733d97d7ac914df40c26851c258f06` | `9a95cd976450e9c7491b62b7eea3d520b1733d97d7ac914df40c26851c258f06` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/check.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/check.cc` | verbatim | 90 | 90 | ee0bc6a87 | `9eac0ea84a6b0c16f1593bc24b0bca3a9dc6bac667b64010753a23471ff81bf6` | `9eac0ea84a6b0c16f1593bc24b0bca3a9dc6bac667b64010753a23471ff81bf6` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/check.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/check.h` | verbatim | 143 | 143 | ee0bc6a87 | `830fdf749d3b4b86f7802dd5bb01d65c09ca7a877381977f52884c469de60038` | `830fdf749d3b4b86f7802dd5bb01d65c09ca7a877381977f52884c469de60038` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/common.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/common.h` | verbatim | 24 | 24 | ee0bc6a87 | `996d7ba38ab4fcf1b0ec97162e1c2c1053a7568b2954e968082b0571621c159a` | `996d7ba38ab4fcf1b0ec97162e1c2c1053a7568b2954e968082b0571621c159a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/context.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/context.cc` | verbatim | 144 | 144 | ee0bc6a87 | `f06b050b5fde7363aa4619011575db2ae6c77d6e2293a0c84b7cabe24278395a` | `f06b050b5fde7363aa4619011575db2ae6c77d6e2293a0c84b7cabe24278395a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/context.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/context.h` | verbatim | 43 | 43 | ee0bc6a87 | `12d8d8bae0e50cc9e2a1106e6c0a24bf80e80a5bd772837b2fc3ca6552d02662` | `12d8d8bae0e50cc9e2a1106e6c0a24bf80e80a5bd772837b2fc3ca6552d02662` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/copy.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/copy.h` | verbatim | 140 | 140 | ee0bc6a87 | `990a3f818974fdb314a889ee41d9d1750bcbc6c024bbf1fdd952dc2567afe6e4` | `990a3f818974fdb314a889ee41d9d1750bcbc6c024bbf1fdd952dc2567afe6e4` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/core.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/core.h` | verbatim | 31 | 31 | ee0bc6a87 | `440d8f60b6b5f9ed33b053cb37d15d7d2f72096ef68d82c12a800c7a79319cc5` | `440d8f60b6b5f9ed33b053cb37d15d7d2f72096ef68d82c12a800c7a79319cc5` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/data_type.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/data_type.h` | verbatim | 350 | 350 | ee0bc6a87 | `49adb973b870bb89e7572da307e196edb9b9cbfd5f17a941c940faa19b34ba6a` | `49adb973b870bb89e7572da307e196edb9b9cbfd5f17a941c940faa19b34ba6a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/layout.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/layout.cc` | verbatim | 153 | 153 | ee0bc6a87 | `44919c15b4da0f7101bb2d21ae15077f29496cb8036266904e1921ebd8b78ea3` | `44919c15b4da0f7101bb2d21ae15077f29496cb8036266904e1921ebd8b78ea3` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/layout.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/layout.h` | verbatim | 195 | 195 | ee0bc6a87 | `c64b72c76e8a8422e1b558c100beb1d0a47eb8fb2953bb23def423fd98751114` | `c64b72c76e8a8422e1b558c100beb1d0a47eb8fb2953bb23def423fd98751114` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/ranges.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/ranges.h` | verbatim | 35 | 35 | ee0bc6a87 | `3d6f72f3cf47b515d0ddb7b03e57efcf6becc20455b2bf1150e70bcf13c79c86` | `3d6f72f3cf47b515d0ddb7b03e57efcf6becc20455b2bf1150e70bcf13c79c86` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/serdes.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/serdes.h` | verbatim | 278 | 278 | ee0bc6a87 | `8273a7b3ccd5173b4759f2c165ee3315241e75960047f188f1f6e2d39acf1629` | `8273a7b3ccd5173b4759f2c165ee3315241e75960047f188f1f6e2d39acf1629` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/stream.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/stream.cc` | verbatim | 19 | 19 | ee0bc6a87 | `e1c4500f12402702802cdf889a34ab194093db30f9481493d444fef9f36eff7e` | `e1c4500f12402702802cdf889a34ab194093db30f9481493d444fef9f36eff7e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/stream.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/stream.h` | verbatim | 160 | 160 | ee0bc6a87 | `d157f2bdd2ad6f8eb988fbec35ff6808d63ddb90f4568e6ad9f6f0cfc16be934` | `d157f2bdd2ad6f8eb988fbec35ff6808d63ddb90f4568e6ad9f6f0cfc16be934` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/core/tensor.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/core/tensor.h` | verbatim | 415 | 415 | ee0bc6a87 | `f45aa75ae734d7ecbff0eb4180c54fc7ae60d4670fce5e59fa4eb2572bb75d2a` | `f45aa75ae734d7ecbff0eb4180c54fc7ae60d4670fce5e59fa4eb2572bb75d2a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/attention/quantization.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/attention/quantization.h` | verbatim | 871 | 871 | 58beb3433 | `997ca2c16bc047c7c219039800bca1d3b0a75668c95c0fce88d289e1a4d50419` | `997ca2c16bc047c7c219039800bca1d3b0a75668c95c0fce88d289e1a4d50419` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/array.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/array.h` | verbatim | 219 | 219 | b91108bb2 | `0b05361e2de4b6a13bcb3761f1e743c168d1c98dbba3463854d711e74eb2361c` | `0b05361e2de4b6a13bcb3761f1e743c168d1c98dbba3463854d711e74eb2361c` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/array_ops.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/array_ops.h` | verbatim | 511 | 511 | ee0bc6a87 | `4e8a07ae04dc1d4cc194357ded24e441c39a3f735442c4dd4624fe802f5140ea` | `4e8a07ae04dc1d4cc194357ded24e441c39a3f735442c4dd4624fe802f5140ea` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/common.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/common.h` | verbatim | 76 | 76 | ee0bc6a87 | `cc57ba9e7806a4348f44081625c721c6f683e1b49749f7271dedeb13e1e42246` | `cc57ba9e7806a4348f44081625c721c6f683e1b49749f7271dedeb13e1e42246` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/data_type.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/data_type.h` | verbatim | 36 | 36 | b91108bb2 | `707fce6edf227e94a32444491be4c6af3fa1cc85ace561c8d3174137013c7730` | `707fce6edf227e94a32444491be4c6af3fa1cc85ace561c8d3174137013c7730` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/layout.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/layout.h` | verbatim | 148 | 148 | ee0bc6a87 | `233251bfea729014be2168b508a325e2c3ab31fac5ce71eb58a10ab05b201139` | `233251bfea729014be2168b508a325e2c3ab31fac5ce71eb58a10ab05b201139` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/math.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/math.h` | verbatim | 86 | 86 | ee0bc6a87 | `1d5a444fac3663b41b30571a9dd18381c930cb7d341bc05b5c1e7d51e36b41a1` | `1d5a444fac3663b41b30571a9dd18381c930cb7d341bc05b5c1e7d51e36b41a1` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/meta.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/meta.h` | verbatim | 52 | 52 | ee0bc6a87 | `9b70b53f5f49695c7880be3caa9c021b36b5580c1be2a2ea28efb520f54cb8e7` | `9b70b53f5f49695c7880be3caa9c021b36b5580c1be2a2ea28efb520f54cb8e7` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/mma.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/mma.h` | verbatim | 212 | 212 | ee0bc6a87 | `c978b646408d9ddb9ebb8ec4b7e07c68725c63e076be77f7af877ca8ee61bc70` | `c978b646408d9ddb9ebb8ec4b7e07c68725c63e076be77f7af877ca8ee61bc70` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/smem.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/smem.h` | verbatim | 106 | 106 | ee0bc6a87 | `c713377bdcd0b9b0a619e1eed6e0b06782015421e085533e392f38d259076a46` | `c713377bdcd0b9b0a619e1eed6e0b06782015421e085533e392f38d259076a46` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/sub_byte_ptr.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/sub_byte_ptr.h` | verbatim | 51 | 51 | ee0bc6a87 | `a10a8c8f127c8136c133c65f222f4ac80e2e26bac142c110eff8a5c015c7c95a` | `a10a8c8f127c8136c133c65f222f4ac80e2e26bac142c110eff8a5c015c7c95a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/core/sync.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/core/sync.h` | verbatim | 65 | 65 | 4f90bd6f5 | `b280b9c98f980980f0ae4f2f42f7d919048f4e896f5fdc73a71018b2628b6e05` | `b280b9c98f980980f0ae4f2f42f7d919048f4e896f5fdc73a71018b2628b6e05` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h` | verbatim | 230 | 230 | 49a9378dd | `902560efd78cda8bbb24c67ab0fa7b00024eb9e9dd899e064f47153be25cef10` | `902560efd78cda8bbb24c67ab0fa7b00024eb9e9dd899e064f47153be25cef10` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/mma_sm70.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/mma_sm70.h` | verbatim | 76 | 76 | ee0bc6a87 | `e885b4d2f0ffc95776589cbe3307559341e825e920a471b4208d6aeaa8c73f6a` | `e885b4d2f0ffc95776589cbe3307559341e825e920a471b4208d6aeaa8c73f6a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/operand_simt.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/operand_simt.h` | verbatim | 175 | 175 | ee0bc6a87 | `2cbef319d164be7fb5ab7df415262264d0d14bc01048422023c07f22e6b4be04` | `2cbef319d164be7fb5ab7df415262264d0d14bc01048422023c07f22e6b4be04` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/operand_sm70_s884.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/operand_sm70_s884.h` | verbatim | 247 | 247 | a42e887cf | `a0f9ac1925113a7e9669b37f27d655444b25fa942a42ce05d48b5e34e0844e0c` | `a0f9ac1925113a7e9669b37f27d655444b25fa942a42ce05d48b5e34e0844e0c` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h` | verbatim | 249 | 249 | ee0bc6a87 | `fac23c12dfcedfaba0f3acdbbd3ec13b2b900aff2af3bc12f0d5517afe566c8e` | `fac23c12dfcedfaba0f3acdbbd3ec13b2b900aff2af3bc12f0d5517afe566c8e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_simt.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_simt.h` | verbatim | 102 | 102 | ee0bc6a87 | `73eab9555cdef9413e996238c17e51c60c4cb45799c96f6615438f72f0be05ca` | `73eab9555cdef9413e996238c17e51c60c4cb45799c96f6615438f72f0be05ca` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm70.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm70.h` | verbatim | 113 | 113 | ee0bc6a87 | `408368b786b93cd88e498513ac71fa9470e8854bdeebe00217d333db520b8482` | `408368b786b93cd88e498513ac71fa9470e8854bdeebe00217d333db520b8482` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm80.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm80.h` | verbatim | 207 | 207 | ee0bc6a87 | `8aac917b3b9260eb426778da6e732193fa32f02875b49fc7693ac41832ec8287` | `8aac917b3b9260eb426778da6e732193fa32f02875b49fc7693ac41832ec8287` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch.h` | verbatim | 51 | 51 | ee0bc6a87 | `097ea44bbf387e5f8c9b7e29c3dc21819f39c9a32a27252d1c8eaa28544993e4` | `097ea44bbf387e5f8c9b7e29c3dc21819f39c9a32a27252d1c8eaa28544993e4` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/batch_kernel_sm70.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/batch_kernel_sm70.h` | verbatim | 34 | 34 | a42e887cf | `f90520e58da79ae866b80e5bc74f28e1280455f3e8cd35fc76b246c25c360f4f` | `f90520e58da79ae866b80e5bc74f28e1280455f3e8cd35fc76b246c25c360f4f` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/cast.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/cast.cu` | verbatim | 259 | 259 | ee0bc6a87 | `dc6916f87158cbb4f5793ba4f6e819f9669cbea4e0b3893278349db841ec999d` | `dc6916f87158cbb4f5793ba4f6e819f9669cbea4e0b3893278349db841ec999d` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/cast.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/cast.h` | verbatim | 47 | 47 | ee0bc6a87 | `a789b2077086b04e15f4477773d387509ce8db3551dfd0dacdf4f70c1cf19c8e` | `a789b2077086b04e15f4477773d387509ce8db3551dfd0dacdf4f70c1cf19c8e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/context.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/context.cu` | verbatim | 249 | 249 | ee0bc6a87 | `d01208e44c4f1b042016fc844e8bf75d38f83b620392d50cdd204be243e4a22f` | `d01208e44c4f1b042016fc844e8bf75d38f83b620392d50cdd204be243e4a22f` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/context.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/context.h` | verbatim | 68 | 68 | ee0bc6a87 | `7733c7d48a52ff119cba30c0da663f2c935d49910c2df293651fa2ca12e63dc9` | `7733c7d48a52ff119cba30c0da663f2c935d49910c2df293651fa2ca12e63dc9` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/convert.cuh` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/convert.cuh` | verbatim | 294 | 294 | ee0bc6a87 | `a064561b17a51a5bf8cda33938a6b88da6a790868da9b2badad1d8287b384395` | `a064561b17a51a5bf8cda33938a6b88da6a790868da9b2badad1d8287b384395` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/convert.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/convert.h` | verbatim | 40 | 40 | 097e3ada5 | `4a82906344f11f3ecfeadbbbb7c3d4492cbeefba8c2c445ff52d9193e6d411ee` | `4a82906344f11f3ecfeadbbbb7c3d4492cbeefba8c2c445ff52d9193e6d411ee` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/convert_v3.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/convert_v3.cu` | verbatim | 297 | 297 | 097e3ada5 | `ae7dac8138c29f2755d1d436ae72dd6aa41269c690d2b564fcba50c1ab36bced` | `ae7dac8138c29f2755d1d436ae72dd6aa41269c690d2b564fcba50c1ab36bced` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/cp_async.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/cp_async.h` | verbatim | 211 | 211 | ee0bc6a87 | `db83410c0c2802ecf208d384129eab5a2c37e8425b3b78785b5e06ddd927a5ed` | `db83410c0c2802ecf208d384129eab5a2c37e8425b3b78785b5e06ddd927a5ed` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/cta_map.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/cta_map.h` | verbatim | 259 | 259 | ee0bc6a87 | `4e5b1659fbab95c83dc82f8764778a96ba955a13456bd38813a8fe195552c9be` | `4e5b1659fbab95c83dc82f8764778a96ba955a13456bd38813a8fe195552c9be` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/desc.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/desc.h` | verbatim | 204 | 204 | ee0bc6a87 | `622999b327471c94e443d081b196c8ede1f54d4065fc9d6f8b35bfb193012460` | `622999b327471c94e443d081b196c8ede1f54d4065fc9d6f8b35bfb193012460` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu` | verbatim | 308 | 308 | 495fb442a | `9164f65eee33b5aa6beb4bec7e387daf9850f584fe5105c0c825cbd0dd37b81b` | `9164f65eee33b5aa6beb4bec7e387daf9850f584fe5105c0c825cbd0dd37b81b` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.h` | verbatim | 32 | 32 | ee0bc6a87 | `bd6f75686c5221a81f1ac5774169ca259ce14b0645baee756ed6a4fbd314f8e2` | `bd6f75686c5221a81f1ac5774169ca259ce14b0645baee756ed6a4fbd314f8e2` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h` | verbatim | 857 | 857 | 4f90bd6f5 | `aa8e8173a4fa00f689fcc38df8f6b90bae42ade04bf79d6e47961ffa4c1dba45` | `aa8e8173a4fa00f689fcc38df8f6b90bae42ade04bf79d6e47961ffa4c1dba45` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/format.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/format.h` | verbatim | 97 | 97 | b91108bb2 | `78db090a5d90d71ef91e97bc8e31836cdb8d0bf08edeb972a2fc82e181fa2e50` | `78db090a5d90d71ef91e97bc8e31836cdb8d0bf08edeb972a2fc82e181fa2e50` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.cu` | verbatim | 1152 | 1152 | 0e8e3c46d | `9ba3fec2abc335ea6adb5ce5d21a3b0dff0e2ca1f99c09d2906d7eb4df85361e` | `9ba3fec2abc335ea6adb5ce5d21a3b0dff0e2ca1f99c09d2906d7eb4df85361e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.h` | verbatim | 52 | 52 | ee0bc6a87 | `208e7b625bb5e7f33985de3be3eaee8c5dc856197d10d843131bb9c21d6b6eec` | `208e7b625bb5e7f33985de3be3eaee8c5dc856197d10d843131bb9c21d6b6eec` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm_universal.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm_universal.h` | verbatim | 252 | 252 | 15ace0f15 | `2e7961392137a5a5ca0395fea6f388a95d96593c0c6bbadfb3f7a09ddb0bc3d4` | `2e7961392137a5a5ca0395fea6f388a95d96593c0c6bbadfb3f7a09ddb0bc3d4` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/iterator.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/iterator.h` | verbatim | 62 | 62 | ee0bc6a87 | `9a96272032696dad66e28be009012561fd98d9796dea47bade1e8a1bd16dd6b7` | `9a96272032696dad66e28be009012561fd98d9796dea47bade1e8a1bd16dd6b7` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h` | verbatim | 319 | 319 | 382a00204 | `b7db284715a1f180dec573074896186d8355c082c20b86ec79a8e22eb306d12c` | `b7db284715a1f180dec573074896186d8355c082c20b86ec79a8e22eb306d12c` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu` | verbatim | 35 | 35 | ee0bc6a87 | `99500bb46167755c497f1fcdfc0108251d849299a8a000548b801701571a3d8d` | `99500bb46167755c497f1fcdfc0108251d849299a8a000548b801701571a3d8d` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu` | trimmed | 244 | 558 | e150e9da8 | `7b7ebc846a08a628eb46ffd14f784b94268ede1196a7492a86ed79989d8e7aee` | `b63a7f6988a29490ffd30309ca755f5bf419d96ae7c77bb14b78e61fefc4c0d7` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` | trimmed | 128 | 142 | de0dd17fb | `ee25256ffbb6a8a41bf86d86bd9a99971a3bfd8552ab7e90d9df517dd57f3fe0` | `117b5a43c710c86293848addeabdcdbb9c81e28d6ac9c4754c256a85426f19b2` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel.cu` | verbatim | 305 | 305 | ee0bc6a87 | `3c7e07df633e87924f08bbd2f03295402baf1105e3358e01a31609be3405cc05` | `3c7e07df633e87924f08bbd2f03295402baf1105e3358e01a31609be3405cc05` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel.h` | verbatim | 122 | 122 | ee0bc6a87 | `b477aff18366692b98889ab73078f516a9810001a98a43f0a5729c5f7f9e3390` | `b477aff18366692b98889ab73078f516a9810001a98a43f0a5729c5f7f9e3390` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h` | verbatim | 309 | 309 | ed7a9e9be | `f73d3064a31045c921f7154d27fd11712c43cb17597498e36e53b340074b67bd` | `f73d3064a31045c921f7154d27fd11712c43cb17597498e36e53b340074b67bd` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/lattice_codebooks.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/lattice_codebooks.h` | verbatim | 1126 | 1126 | f1c11f7b3 | `f5b8962a7a2722a44d0a209a9fbafee4f1482fd2ad7953a8d61c9212ececa22c` | `f5b8962a7a2722a44d0a209a9fbafee4f1482fd2ad7953a8d61c9212ececa22c` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/lattice_transform.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/lattice_transform.h` | verbatim | 86 | 86 | 5892da031 | `9cab95b6918f42504b4a0b73831cdff052a25749b9f97344ab372f8541d224d1` | `9cab95b6918f42504b4a0b73831cdff052a25749b9f97344ab372f8541d224d1` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h` | verbatim | 425 | 425 | 8f47970a0 | `45fc6801338ab706a2e8393e4d8cda24416ae0cc1101061771a8eaeccdb449ec` | `45fc6801338ab706a2e8393e4d8cda24416ae0cc1101061771a8eaeccdb449ec` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/matrix_ptr.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/matrix_ptr.h` | verbatim | 114 | 114 | ee0bc6a87 | `71184e0e4529902b48cc3edf50c08c53d518024849a86262f78c4018173a6b6e` | `71184e0e4529902b48cc3edf50c08c53d518024849a86262f78c4018173a6b6e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/moe_utils_v2.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/moe_utils_v2.h` | verbatim | 64 | 64 | ee0bc6a87 | `c9ec92b8898106ddcf52273c49b1534fb909b4bafaac9d4d6776b1650e74cf4a` | `c9ec92b8898106ddcf52273c49b1534fb909b4bafaac9d4d6776b1650e74cf4a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/operand.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/operand.h` | verbatim | 66 | 66 | ee0bc6a87 | `4f68b196a997b3ca4187b31c3a831b6ce20937883e0ebb374bf4d48e97d84fb1` | `4f68b196a997b3ca4187b31c3a831b6ce20937883e0ebb374bf4d48e97d84fb1` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/predicate.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/predicate.h` | verbatim | 55 | 55 | ee0bc6a87 | `a7dc394324dc6ca8e78b56fd1098e2b9e54595c609221f733c143d214a6031d7` | `a7dc394324dc6ca8e78b56fd1098e2b9e54595c609221f733c143d214a6031d7` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/registry.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/registry.h` | verbatim | 59 | 59 | aeede18c5 | `764e94d5e32c8578f482b304d16829d0e36778cf4998388bb5d9b38cdc208fa8` | `764e94d5e32c8578f482b304d16829d0e36778cf4998388bb5d9b38cdc208fa8` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/scheduler_sm70.cuh` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/scheduler_sm70.cuh` | verbatim | 272 | 272 | 992339a4c | `2aef17b2ac33bf1860b102b2c17d2ce11d2e7ee9dbe7c90c9b525be8d3deaa0a` | `2aef17b2ac33bf1860b102b2c17d2ce11d2e7ee9dbe7c90c9b525be8d3deaa0a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/simt.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/simt.h` | verbatim | 19 | 19 | ee0bc6a87 | `bc900c794e2c8011f695ac6883a93d745abb2fb4f6f941a683780fb7eb793fb8` | `bc900c794e2c8011f695ac6883a93d745abb2fb4f6f941a683780fb7eb793fb8` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/sm70_dflash_context.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/sm70_dflash_context.h` | verbatim | 24 | 24 | 495fb442a | `0c8a8996c2190cdf77349e0b41b145ef23e583be668840cc6df4e2c1c761e388` | `0c8a8996c2190cdf77349e0b41b145ef23e583be668840cc6df4e2c1c761e388` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/smem_copy.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/smem_copy.h` | verbatim | 200 | 200 | ee0bc6a87 | `c8be68e9814f9704b7e39bca229f3b65595e94862008c0ca14f82e062e5f97c4` | `c8be68e9814f9704b7e39bca229f3b65595e94862008c0ca14f82e062e5f97c4` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/thread_group_map.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/thread_group_map.h` | verbatim | 117 | 117 | ee0bc6a87 | `255926f390bf9adb7410a19cef03619fb98eb51f3bce655c2350bf1b11f4ab9d` | `255926f390bf9adb7410a19cef03619fb98eb51f3bce655c2350bf1b11f4ab9d` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/thread_map.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/thread_map.h` | verbatim | 246 | 246 | ee0bc6a87 | `9ef1489aee635e4db5b8973d0a0edccd0a00af7e8eb3f396fb21ae50fed16019` | `9ef1489aee635e4db5b8973d0a0edccd0a00af7e8eb3f396fb21ae50fed16019` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tiled_mma.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tiled_mma.h` | verbatim | 236 | 236 | ee0bc6a87 | `8e2920c10cc60d808a10099ae6cac17a44e97e3a53a2ff4b7fd5a6744ababe06` | `8e2920c10cc60d808a10099ae6cac17a44e97e3a53a2ff4b7fd5a6744ababe06` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/transform.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/transform.h` | verbatim | 413 | 413 | f01116d3a | `e783882dba8a4b817e9b7eedb6c03f16327e6ade4263250a17dce2493afe576a` | `e783882dba8a4b817e9b7eedb6c03f16327e6ade4263250a17dce2493afe576a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/cache_utils.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/cache_utils.cu` | verbatim | 41 | 41 | 622c1b993 | `24bafc9c9c939351f665c1c131c464410b330fede344bd3cac08aa41d5e8cf8a` | `24bafc9c9c939351f665c1c131c464410b330fede344bd3cac08aa41d5e8cf8a` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/cache_utils.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/cache_utils.h` | verbatim | 21 | 21 | ee0bc6a87 | `834a84414c9beba8035775a5f04e39838cf853d4c6ff7bab39d747420efdaaae` | `834a84414c9beba8035775a5f04e39838cf853d4c6ff7bab39d747420efdaaae` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/measurer.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/measurer.cu` | verbatim | 89 | 89 | ee0bc6a87 | `9728cc35d7e299cda581428b17eb449f9bc8a84e6762f83e4659172c130dfd0c` | `9728cc35d7e299cda581428b17eb449f9bc8a84e6762f83e4659172c130dfd0c` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/measurer.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/measurer.h` | verbatim | 41 | 41 | ee0bc6a87 | `d4dce95c925d643e419cb6300c756d2526e8ffee5c5561d6c3635f3f289294b0` | `d4dce95c925d643e419cb6300c756d2526e8ffee5c5561d6c3635f3f289294b0` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/params.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/params.cc` | verbatim | 104 | 104 | ee0bc6a87 | `2c86d113c8c61c67e0cebfd4fcf38d843ce3b18982e6ed80a927310370bc4b87` | `2c86d113c8c61c67e0cebfd4fcf38d843ce3b18982e6ed80a927310370bc4b87` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/params.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/params.h` | verbatim | 41 | 41 | ee0bc6a87 | `2a6c8c3ca72a66627aa4fed5d388afc837ca931fc7cb93fe493b81e61a5de792` | `2a6c8c3ca72a66627aa4fed5d388afc837ca931fc7cb93fe493b81e61a5de792` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/sampler.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/sampler.cu` | verbatim | 79 | 79 | ee0bc6a87 | `0f4b3169df0cf207cb8cc9821dad0780ac38003b83781e327dbd433bc09b7125` | `0f4b3169df0cf207cb8cc9821dad0780ac38003b83781e327dbd433bc09b7125` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/sampler.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/sampler.h` | verbatim | 23 | 23 | ee0bc6a87 | `491d2960e6afdb71b104fa107e0fddfb568cf131992de15cba322f560065e38e` | `491d2960e6afdb71b104fa107e0fddfb568cf131992de15cba322f560065e38e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/stats.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/stats.h` | verbatim | 46 | 46 | ee0bc6a87 | `145f22f12fb59010489706d7cbf83921c1f81b919d20804f6008613bda1afb24` | `145f22f12fb59010489706d7cbf83921c1f81b919d20804f6008613bda1afb24` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/stopping_criterion.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/stopping_criterion.cc` | verbatim | 36 | 36 | ee0bc6a87 | `e3e58c9e349f4407f65a04d0400677ee7c2f811c8de534ca9295ebb9830c039d` | `e3e58c9e349f4407f65a04d0400677ee7c2f811c8de534ca9295ebb9830c039d` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/stopping_criterion.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/tuner/stopping_criterion.h` | verbatim | 16 | 16 | ee0bc6a87 | `c8a36e5f5ee3c4f5c5d44a97343d506b5658fb6e162a34f0cedaf94e91622ed9` | `c8a36e5f5ee3c4f5c5d44a97343d506b5658fb6e162a34f0cedaf94e91622ed9` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/types.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/types.h` | verbatim | 278 | 278 | 097e3ada5 | `b0798ea7459eada8e94ec8e0d55cc457b1b1f1b0ac32b6535e07bb85371f1a79` | `b0798ea7459eada8e94ec8e0d55cc457b1b1f1b0ac32b6535e07bb85371f1a79` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/unpack.cu` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/unpack.cu` | verbatim | 117 | 117 | ee0bc6a87 | `7e1560f2e5c6de54472ef936c0a73a6496bc34d3b9a09c0a00995e4cc65766fb` | `7e1560f2e5c6de54472ef936c0a73a6496bc34d3b9a09c0a00995e4cc65766fb` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/kernels/gemm/utils.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/utils.h` | verbatim | 150 | 150 | ee0bc6a87 | `f5abc2ac819a84a3e9a0cef5b8996f58624ddc68d40213300797dc04f9308b9e` | `f5abc2ac819a84a3e9a0cef5b8996f58624ddc68d40213300797dc04f9308b9e` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/macro.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/macro.h` | verbatim | 9 | 9 | ee0bc6a87 | `14e4c9f91a57249d14a3cb8602d35eb99d5af705c869295dff12748a923ec0ad` | `14e4c9f91a57249d14a3cb8602d35eb99d5af705c869295dff12748a923ec0ad` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/cuda_bf16_wrapper.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/cuda_bf16_wrapper.h` | verbatim | 21 | 21 | ee0bc6a87 | `44075cca67013e32baaf5818699505039e2f8d22716db62c390cf5546ae0a7ee` | `44075cca67013e32baaf5818699505039e2f8d22716db62c390cf5546ae0a7ee` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/cuda_utils.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/cuda_utils.h` | verbatim | 192 | 192 | ee0bc6a87 | `bdfc84595c32162a60e4b52c01a7755dcb89363615156e5db757d1c284ca3564` | `bdfc84595c32162a60e4b52c01a7755dcb89363615156e5db757d1c284ca3564` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/logger.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/logger.cc` | verbatim | 65 | 65 | ee0bc6a87 | `b15fd4531c0e186a611ee3f59bb294b243c3e7ac291aa6e81926a969662d35cc` | `b15fd4531c0e186a611ee3f59bb294b243c3e7ac291aa6e81926a969662d35cc` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/logger.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/logger.h` | verbatim | 124 | 124 | ee0bc6a87 | `14c40ed2f7fe82a913cc333d5043221caa0fff6de3bc850be52a360008c0cc2d` | `14c40ed2f7fe82a913cc333d5043221caa0fff6de3bc850be52a360008c0cc2d` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/monotonic.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/monotonic.h` | verbatim | 43 | 43 | ee0bc6a87 | `519deaf54ae7202e451bb5f90fb192610d3323cbaf56e8419d89c8bf63b26eaa` | `519deaf54ae7202e451bb5f90fb192610d3323cbaf56e8419d89c8bf63b26eaa` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/parser.cc` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/parser.cc` | verbatim | 39 | 39 | ee0bc6a87 | `cf1167c34048cd0e1cf78ce823f95aefd406059c5715c6a0850fec41d82220a6` | `cf1167c34048cd0e1cf78ce823f95aefd406059c5715c6a0850fec41d82220a6` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/parser.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/parser.h` | verbatim | 30 | 30 | ee0bc6a87 | `5644190d0b80adddedcbac37934b72c78b74d251c1f69e4563985c7876aa0773` | `5644190d0b80adddedcbac37934b72c78b74d251c1f69e4563985c7876aa0773` |
| `sm70/csrc/turbomind/lmdeploy/src/turbomind/utils/string_utils.h` | `csrc/sm70_turbomind/lmdeploy/src/turbomind/utils/string_utils.h` | verbatim | 84 | 84 | ee0bc6a87 | `7caae49d69950416a3c227150d73490fc70abe77ea660f873f95275ec3b3befe` | `7caae49d69950416a3c227150d73490fc70abe77ea660f873f95275ec3b3befe` |
| `sm70/csrc/turbomind/ops/gemm_runtime.cpp` | `csrc/sm70_turbomind/ops/gemm_runtime.cpp` | rewritten | 890 | 895 | 4db646d45 | `3f203e3b8d73167a3229076129871d56e7e73c60d17c4ded357f7e500d19881c` | `55d4e35e7c7f9cbb439d4c1126c3647da18d62a8d02dc2008c05d5d3ff8c7085` |
| `sm70/csrc/turbomind/ops/gemm_runtime.h` | `csrc/sm70_turbomind/ops/gemm_runtime.h` | trimmed | 86 | 113 | 4db646d45 | `1082262db91911f54016bbdb4cd2626bb5b3c810154ef794eea840be33250021` | `305543af6e3e3850f62c82e245619894aee634ec02b53770ad459cf366ecbc6f` |
| `sm70/csrc/turbomind/ops/packed_gemm.h` | `csrc/sm70_turbomind/ops/packed_gemm.h` | verbatim | 76 | 76 | 156ec4deb | `da56a140b05e0c21baa37400110871a0600236b536102997ef86f1ac155b5b1f` | `da56a140b05e0c21baa37400110871a0600236b536102997ef86f1ac155b5b1f` |
| `sm70/csrc/turbomind/ops/qwen38_prefill_cutlass.cu` | `csrc/sm70_turbomind/ops/qwen38_prefill_cutlass.cu` | rewritten | 127 | 127 | 495fb442a | `c99a0a489487d381a9809dde7c1ce54a6f966c02f56eb94e3b6f3af41ba39e92` | `4934d496d3aa5da5e0ddb8f35f867cd1f79c30a604eff9bbca3fda3780a5dd2d` |
| `sm70/csrc/turbomind/ops/qwen38_prefill_cutlass.cuh` | `csrc/sm70_turbomind/ops/qwen38_prefill_cutlass.cuh` | verbatim | 13 | 13 | 58beb3433 | `51aaa6486cebfbbb82ab0a8edd3296ed3bf4dcdca7fd7f653fc96662dbbbdfa0` | `51aaa6486cebfbbb82ab0a8edd3296ed3bf4dcdca7fd7f653fc96662dbbbdfa0` |
| `sm70/csrc/turbomind/ops/sm70_policy.h` | `-` | new | 9 | - | - | `e22e683997b19651e2d31f7bf9597a39e97574aa34348c7a0a0be05d4dec02df` | `-` |
| `sm70/csrc/turbomind/ops/tm_awq_sm70.cu` | `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | rewritten | 744 | 10913 | 899e9ed33 | `0d179aa7673d8b1a21105012bbb3d1dc3af8285119ed2a7869e11fc30d5bce08` | `69d722af79000d1df70bf1ac3172903435d3ab4ec26a6e6c9067c4c2697f9e89` |
| `sm70/csrc/turbomind/ops/tm_config.h` | `csrc/sm70_policy.h` | rewritten | 267 | 315 | c43f5d78c | `a5fb816fb7d5ba9485e7839d85d7e8e0662ea8f7e47cacc665192b3cf63f760a` | `b2c935891ef32cc8c70367fb109c65f574beaff64b193b08565164b0a9e53981` |
| `sm70/csrc/turbomind/ops/tm_f16_sm70.cu` | `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | rewritten | 375 | 10913 | 899e9ed33 | `39219d6437ea5ebd06a5468731caa7e6a5e16496a1e1342de3fa336244c74da5` | `69d722af79000d1df70bf1ac3172903435d3ab4ec26a6e6c9067c4c2697f9e89` |
| `sm70/csrc/turbomind/ops/tm_fp8_sm70.cu` | `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | rewritten | 615 | 10913 | 899e9ed33 | `493ed6e2d7c920bb8f6c82d8c8046feb0fb566c4318c6f0b5e9cd5fa1daf790b` | `69d722af79000d1df70bf1ac3172903435d3ab4ec26a6e6c9067c4c2697f9e89` |
| `sm70/csrc/turbomind/ops/tm_moe_sm70.cu` | `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | rewritten | 3549 | 10913 | 899e9ed33 | `794b33738da06b5a507d539085226a803b8bab6a2df889763d9ff9b508a14eaf` | `69d722af79000d1df70bf1ac3172903435d3ab4ec26a6e6c9067c4c2697f9e89` |
| `sm70/csrc/turbomind/ops/tm_nvfp4_sm70.cu` | `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | rewritten | 288 | 10913 | 899e9ed33 | `98028afde445b8ecc1827be9ac32982bbbbf6afc6ae6cb3a269ba9423ec97c0f` | `69d722af79000d1df70bf1ac3172903435d3ab4ec26a6e6c9067c4c2697f9e89` |
| `sm70/csrc/turbomind/ops/tm_ops_common_sm70.cuh` | `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | rewritten | 119 | 10913 | 899e9ed33 | `f00da8f4ef8450a75bbd8cae4d798fa12be683d0e406606410b4db97ccd96bb7` | `69d722af79000d1df70bf1ac3172903435d3ab4ec26a6e6c9067c4c2697f9e89` |
| `sm70/csrc/turbomind/ops/tm_registry_sm70.cu` | `csrc/sm70_turbomind/ops/tm_registry_sm70.cu` | verbatim | 38 | 38 | 58beb3433 | `9025732862826d2bd1322da9be4157578c7e3db45b13cde725f88f3f57af73d8` | `9025732862826d2bd1322da9be4157578c7e3db45b13cde725f88f3f57af73d8` |
| `sm70/csrc/turbomind/sources.cmake` | `-` | new | 102 | - | - | `0e23a50120f954ac68368e8bdb36be37d5836e6b536078fd732f8404f90509ce` | `-` |
| `sm70/csrc/sm70_tile_runtime_signal.cuh` | `csrc/sm70_tile_runtime_signal.cuh` | verbatim | 49 | 49 | 58beb3433 | `91f2977cbdd206c7bbf37f40330fcd99a1266f04da585b713f1c4e514982c350` | `91f2977cbdd206c7bbf37f40330fcd99a1266f04da585b713f1c4e514982c350` |
