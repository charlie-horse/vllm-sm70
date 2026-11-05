# DFlash2 draft FP16 M<=8 GEMM + compact FP16 top-64: provenance (patch K18)

Source: charlie-horse/vLLM at **cf9492a01**. Line counts are `wc -l`. "fork last-touch" is `git log -1 --format=%h -- <fork path>`. Carry map: R9 (no tier owned these; K18 takes them), I9/U1 (fp16 GEMM routing through the platform hook). Re-verify the function-by-function comparison with:

```
scripts/verify-port.py --port sm70/csrc/dflash2/dflash2_fp16_m8.cu sm70/csrc/dflash2/compact_half_topk.cu \
  --fork <fork>/csrc/sm70_turbomind/ops/dflash2_fp16_m8_sm70.cu <fork>/csrc/sm70_turbomind/ops/compact_half_topk_sm70.cu
scripts/verify-port.py --port sm70/vllm_sm70/ops/dflash2.py \
  --fork <fork>/vllm/model_executor/layers/quantization/sm70_dflash2_fp16.py <fork>/vllm/model_executor/layers/sm70_compact_topk.py
```

At the K18 commit the CUDA run reports IDENTICAL=9, CHANGED=2, NEW=7 (the NEW rows are the `_sm70_C` registrations and `aligned16`). Every CHANGED row is listed below.

## Files

| file | lines | status | sha256 | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---|---:|---|---|
| `dflash2_fp16_m8.cu` | 187 | changed | `53a3faacd91b31ae988102299b6316cbba08f68f42f4398f71393965d4649e93` | `csrc/sm70_turbomind/ops/dflash2_fp16_m8_sm70.cu` | 146 | `bf6dc0a3198467b3ff207f3f93a2bc89849b270005dcf0a481cca9f09ad427c4` | f25139144 |
| `compact_half_topk.cu` | 99 | namespace only | `999805f60fd639a5e7b165586ae2b67782521e9d153317d1ac09ec1e62975162` | `csrc/sm70_turbomind/ops/compact_half_topk_sm70.cu` | 96 | `bbad9956baf6e63935205faaa450de5557705b09b17ed3654636c2bb642d46df` | fb52756f4 |
| `LICENSE.v100-skinny` | 21 | verbatim | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` | 21 | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | 58beb3433 |
| `sources.cmake` | 21 | new (replaces fork `CMakeLists.txt:407-408` in `VLLM_SM70_TURBOMIND_SRCS`) | `6bb04ae442fa9ed8b075352110cedbce876b403c5dd2fbd24561a62fda5e5f61` | – | – | – | – |

The registrations in the fork lived in `csrc/torch_bindings.cpp:713-721` (`ops.def`/`ops.impl` in `_C`, declarations in `csrc/ops.h:201-206`) for the GEMM ops and in the file itself (`TORCH_LIBRARY_FRAGMENT(_C)`) for the top-k. Here each `.cu` has its own `TORCH_LIBRARY_FRAGMENT(_sm70_C)` + `TORCH_LIBRARY_IMPL(_sm70_C, CUDA)`. Schemas are unchanged (op names kept, no `native_policy` arg existed).

## `dflash2_fp16_m8.cu`, function by function

| port | fork | status |
|---|---|---|
| `MMA` macro | :14-21 | identical |
| `dflash2_fp16_m8_kernel<Warps, Tile>` | :25-79 | identical |
| `aligned16` | – | **new**: 16-byte base-pointer check |
| `sm70_dflash2_fp16_m8_out` | :83-120 | changed: one added `TORCH_CHECK(aligned16(input) && aligned16(packed))`. The kernel loads input rows and packed weights as `uint4`; the fork assumed alignment (true for fresh allocations, false for a contiguous view at an odd offset). Everything else identical |
| `sm70_dflash2_fp16_dispatch_out` | :122-146 | changed: (a) **M 1..7** now runs the M8 kernel on a zero-padded 8-row copy and copies rows [0, M) back (fork: only M == 8, else `mm`). Rows are independent and the reduction order is fixed, so the result is bitwise the M8 kernel's rows (GPU test `test_m_below_8_is_padded_m8`). (b) a misaligned input or packed buffer takes `at::mm_out` instead of failing. (c) the `OptionalCUDAGuard` now covers the padded path too |

## `compact_half_topk.cu`

`select_compact_half_keys<Items, Merge>` (:13-50) and `sm70_compact_half_topk_out` (:51-88) are identical. Only `TORCH_LIBRARY_FRAGMENT(_C)`/`TORCH_LIBRARY_IMPL(_C, CUDA)` became `_sm70_C`, and a 3-line provenance comment was added.

## Python counterpart (`vllm_sm70/ops/dflash2.py`)

| port | fork | status |
|---|---|---|
| `GEOMETRY`, `DRAFT_SHAPES_TP2` | `quantization/sm70_dflash2_fp16.py:14-20` `_GEOMETRY` | changed: see "Admitted shapes" |
| `pack_weight` | `sm70_dflash2_fp16.py:44-56` | split out of `prepare_dflash2_fp16_m8`, body identical |
| `prepare_dflash2_fp16_m8` | `sm70_dflash2_fp16.py:23-67` | changed: `hasattr(torch.ops._C, ...)` → `enabled()` (kill switch + `_sm70_C` ops); `current_platform.is_device_capability(70)` → `torch.cuda.get_device_capability(dev) == (7, 0)`; dropped the `VLLM_BATCH_INVARIANT` check (UP `UnquantizedLinearMethod.apply`, linear.py:238-241, handles batch-invariant mode before calling the U1 GEMM); logs unadmitted shapes |
| `apply_dflash2_fp16_m8` | `sm70_dflash2_fp16.py:70-92` | changed: eager admits M 1..8 (fork: M == 8); dropped the `_sm70_f16_prepared` (TurboMind F16 dense, not ported) and `VLLM_BATCH_INVARIANT` checks |
| `dflash2_unquantized_gemm` | `linear.py:404-411` (`UnquantizedLinearMethod.apply` calls it first) | new home: the U1 callable; falls back to `F.linear` (UP `default_unquantized_gemm`) |
| `select_unquantized_gemm` | – | new: what `Sm70CudaPlatform.get_unquantized_gemm` returns; None (upstream dispatch) when the kill switch is 0, the ops are missing, or an explicit FlashInfer BF16 backend was chosen |
| `mark_draft_linears` | `models/qwen3_dflash2.py:280-298` | changed: no `get_tensor_model_parallel_world_size() == 4` condition (admission is by shard shape); the FP8 M8 flag (`_sm70_dflash2_fp8_m8`) is not set (FP8 draft path not ported) |
| `prepare_marked`, `mark_and_prepare` | `linear.py:343-352` (called from `process_weights_after_loading`) | new home: M7's `DFlash2SM70Qwen3ForCausalLM.load_weights` packs after `super().load_weights` (UP `process_weights_after_loading` does nothing on CUDA, and packing must precede profiling/capture, carry-map I11) |
| `compact_half_topk` | `layers/sm70_compact_topk.py:10-39` | changed: `current_platform.is_device_capability(70)` → `_is_sm70`; `torch.ops._C` → `torch.ops._sm70_C` |
| fakes `_fake_m8`, `_fake_dispatch`, `_fake_topk` | `sm70_dflash2_fp16.py:95-106`, `sm70_compact_topk.py:42-46` | `_C::` → `_sm70_C::`, registered only if `has_op(name)` |

Hook sites: `vllm_sm70/_cuda_platform.py` `get_unquantized_gemm` (U1, patches/0002); `vllm_sm70/models/qwen/dflash_sm70.py` (M7) `DFlash2SM70Qwen3DecoderLayer.__init__` marks, `DFlash2SM70Qwen3ForCausalLM.load_weights` packs. Marking happens only when M7's BF16-range emulation is on for the layer, as in the fork (`use_sm70_bf16_emulation and ...`).

## Admitted shapes (TP2, 2x V100)

`(N, K)` is the local weight shard `[out, in]`. The fork's table held the Qwen3.8-27B draft at TP4 only (and the fork marked layers only at TP4), so at TP2 nothing would have been admitted. Shards from the drafts' public `config.json` (hidden H, 32 q / 8 kv heads x 128, intermediate I, conv kernel 2, conv group 16):

| draft | qkv_proj | o_proj | gate_up_proj | down_proj | conv kernel_projection (x2, replicated) |
|---|---|---|---|---|---|
| Qwen3.8-27B-DFlash2 (H 5120, I 17408) | 3072x5120 → (32,16) | 5120x2048 → (32,8) | 17408x5120 → (32,16) | 5120x8704 → (32,16) | 1280x5120 → (16,16) **measured** |
| Qwen3.6-35B-A3B-DFlash2 (H 2048, I 6144) | 3072x2048 → (32,8) | 2048x2048 → (16,16) | 6144x2048 → (32,8) | 2048x3072 → (16,16) | 512x2048 → (16,16) |

`(tile, warps)`; kernel constraints `N % tile == 0`, `K % (16 * warps * (tile == 16 ? 2 : 1)) == 0` hold for all ten (CPU test `test_tp2_shards_admitted`). Only (1280, 5120) was measured in the fork; the other nine geometries are **provisional**, picked by the fork's pattern (tile 16 when N/32 < 80 SMs; 16 warps for K >= 5120), and need a bench pass. Any other marked shard keeps `F.linear` (logged once per layer at load).

Memory: packing keeps a second copy of each admitted weight (as the fork did). Qwen3.8 draft at TP2: ~346 MB/layer, ~1.7 GB/GPU for 5 layers; Qwen3.6 draft: ~0.38 GB/GPU. It is allocated at load, so the KV-cache budget accounts for it.

## Dropped (and why)

- The four TP4-only `_GEOMETRY` entries (1536,5120), (5120,1024), (8704,5120), (5120,4352): TP4-only (brief rule). The kernel and its three geometries are unchanged, so re-adding a shape is a table row.
- The TP==4 condition on marking (FORK `qwen3_dflash2.py:280-284`).
- `_sm70_dflash2_fp8_m8` marking and `prepare/apply_dflash2_fp8_m8` (FORK `sm70_dflash2_fp8.py`): a separate FP8 draft path, not part of K18.
- The fork test `test_target_probe_preserves_dense_token_mask` (FORK `tests/kernels/core/test_sm70_compact_half_topk.py:59-120`): it needs the fork-only DFlash2 sparse-rejection sampler (`v1/worker/gpu/spec_decode/dflash2/sparse_rejection.py`), which upstream v0.33.0 does not have.

## Known limits

- `compact_half_topk` has **no consumer** in upstream v0.33.0 (the fork's was the DFlash2 target probe in `logits_processor.py:620-631`), and its 16-bit index packing caps the width at 65535: the 248320-token vocab is 62080 per rank at TP4 but 124160 at TP2, so at TP2 the guard returns None (GPU test `test_tp2_vocab_shard_falls_back`). Ported as an op for a future consumer.
- The packed copy is built once in `load_weights`; a later in-place weight update (weight reload / RLHF sync) would leave it stale.
- M > 8 (several draft requests per step) keeps `F.linear`.
