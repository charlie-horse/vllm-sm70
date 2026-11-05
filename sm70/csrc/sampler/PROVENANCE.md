# Flint top-k candidate sampler: provenance (patch K16)

Source: charlie-horse/vLLM at **cf9492a01**. Carry-map row K16 (C7 in the
Flint section), plus I16 (V1 vs V2, `use_fp64_gumbel`) and R4 (capture-time
state). "Fork last-touch" is `git log -1 --format=%h -- <fork path>`.

| fork file | lines | last-touch | here |
|---|---|---|---|
| `csrc/sm70_turbomind/ops/sm70_topk_candidates_sampler.cu` | 448 | `ae002efc7` | `csrc/sampler/topk_candidates_sampler.cu` (verbatim) |
| `csrc/sm70_turbomind/ops/sm70_top20_philox_fragment.cpp` | 93 | `ae002efc7` | `csrc/sampler/sampler_ops.cpp` (decls :34-41, defs :82-92 only; adapted) |
| `vllm/v1/sample/ops/sm70_topk_candidates.py` | 347 | `484216b2a` | `vllm_sm70/sample.py` (adapted) + `vllm_sm70/ops/sampler.py` (new: op wrappers, fakes) |
| `vllm/v1/worker/gpu_model_runner.py` | :7104-7125, :7698-7800, :7828-7843 | — | `vllm_sm70/sample_v1.py` (the U4 consumer; new shape, same gate) |
| `vllm/models/flint/model.py` | :570-577, :605-614 | — | opt-in attribute `supports_sm70_topk_sampler` (M10 sets it on the Flint model) |
| `tests/kernels/core/test_sm70_topk_candidates_sampler.py` | 230 | `ae002efc7` | `tests/sampler/test_k16_topk_candidates_sampler_gpu.py` |
| `tests/v1/sample/test_sm70_topk_candidates_admission.py` | 111 | `484216b2a` | `tests/sampler/test_k16_topk_candidates_admission.py` |

## Function map

### CUDA (`topk_candidates_sampler.cu`, byte-identical to the fork file)

| function | fork line | status |
|---|---|---|
| `half_order` | 69-77 | verbatim |
| `local_top32_tile_kernel` | 79-129 | verbatim |
| `local_top32_merge_kernel` | 131-177 | verbatim |
| `native_exponential_for_index` | 179-199 | verbatim |
| `sample_topk_candidates_kernel` | 201-321 | verbatim |
| `sm70_local_top32_half` (host) | 325-371 | verbatim |
| `sm70_sample_topk_candidates` (host) | 373-448 | verbatim |

### Registration (`sampler_ops.cpp`)

The two defs keep the fork schemas character for character; the namespace is
`_sm70_C` (TORCH_LIBRARY_FRAGMENT) instead of `_C`, and the impls move to a
`TORCH_LIBRARY_IMPL(_sm70_C, CUDA)` block (the fork passed `torch::kCUDA` to
`ops.impl`).

### Python (`vllm_sm70/sample.py`)

| function | fork line | status / change |
|---|---|---|
| constants (`LOCAL_K`, `WIRE_WIDTH`, `MAX_TOP_K`, `MAX_ROWS`, ...) | 48-60 | same values; the kernel ones come from `ops/sampler.py` |
| `ops_available` | 63-69 | `ops.sampler.available()` (`_sm70_C` instead of the `vllm._sm70_ops` sidecar) |
| `_is_sm70` | 72-74 | `torch.cuda.get_device_capability` instead of `current_platform` |
| `_OverflowMonitor` | 77-100 | `after_call` skips host bookkeeping while a capture runs |
| `_overflow_monitor` | 106-110 | **lookup only**: never creates (R4); new `prepare(device)` creates, refuses during capture |
| `overflow_count` | 113-115 | raises when `prepare()` was not called |
| `metadata_reject_reason` | 118-160 | takes the host `top_k_cpu` as an argument (upstream `SamplingMetadata` has no `top_k_cpu`/`temperature_cpu`); host temperature check reduced to the device tensor being present |
| `argmax_invariant_reject_reason` | 163-173 | verbatim |
| `admission_reject_reason` | 176-196 | passes `top_k_cpu` through |
| `lm_head_topk_candidates` | 199-227 | admission moved to the new rank-uniform `lm_head_reject_reason` (see shape audit); vLLM imports are local |
| `sample_lm_head_candidates` | 230-254 | `logger.info_once` -> module `_log_once` |
| `local_topk_candidates` | 257-267 | calls `ops.sampler.local_top32_half` |
| `shard_supported` | 270-282 | adds the kernel's row-stride check (`.cu`:341-343) |
| `sample_topk_candidates` | 285-347 | passes `overflow=None` when no counter was prepared |

### Runner gate (`vllm_sm70/sample_v1.py`)

| function | fork origin | change |
|---|---|---|
| `non_argmax_logits_processors_inactive` | runner :7828-7843 | verbatim logic |
| `Sm70HiddenStateSampler.reject_reason` | runner :7770-7803 (`_can_use_topk_candidate_sampler`, `_reject_topk_candidate_sampler`) | pooling / PP broadcast checks dropped (U4 only calls on the last-rank, non-pooling path); adds `not_prepared`, `model` (opt-in), `fp64_gumbel`, `capturing`, `batch_size` (hidden rows vs `num_reqs`) |
| `Sm70HiddenStateSampler.try_sample` | runner :7104-7125 | returns `SamplerOutput(sampled[:, None], None)` instead of the runner doing it |
| `Sm70HiddenStateSampler.prepare` / `prepare_worker` | — (new) | run from `Sm70CudaPlatform.kernel_warmup` (U1): bind the runner, read `use_fp64_gumbel`, create the overflow counter before capture (R4) |

## Dropped, and why

| fork code | why |
|---|---|
| `sm70_top20_philox_sampler.cu` (289) and its three defs (`sm70_sample_sorted_top20_philox_out`, `..._token_out`, `sm70_sample_chunked_top20_philox_token_out`) | the older compact/chunked top-k20 routes (`VLLM_SM70_COMPACT_TOPK20_SAMPLER`, `..._TP_LOCAL_TOPK20_SAMPLER`, both default off at cf9492a); the Flint route is the candidate sampler. Runner branch :7108-7120 (`get_topk_tokens_and_logits` + `sm70_compact_topk20_pairs_sample`) dropped with them |
| `sm70_sampler_graph_attach_branch`, `sm70_sampler_graph_prepare_collective` (`sm70_sampler_graph_branch.cu`) | sampler CUDA-graph branch experiment, not on the Flint route |
| runner `_can_use_sm70_compact_topk20_tokens` :7698-7768 (minus the :7703-7706 delegation) | the compact route above (fixed vocab 248320, top-p 0.95, temperature 1.0 only) |
| fork `_sm70_sampler_C` module and its CMake (`CMakeLists.txt`:762-800) | one `.so` (`_sm70_C`); C++20 instead of the fork's C++17 target properties, no separate compilation |
| `layer_policy(...).value("flint_sampler")` on the model | kill switch is read in the platform getter instead |

Nothing in the kept code is TP4-only.

## Admitted shape set (audited for Flint TP2 on 2x V100)

Kernel limits (`.cu`:51-60, mirrored in `ops/sampler.py` and checked by
`test_kernel_constants_match_python`): `kLocalK` 32 candidates per rank,
`kMaxTiles` 128 x 2048 = 262144 logits per rank, `kMaxCandidates` 128, so
TP <= 4, `kMaxRows` 8, token ids < 2^24 (exact in FP32).

| quantity | admitted | Flint TP2 |
|---|---|---|
| TP size | 1, 2, 4 (`lm_head.tp_size` must equal the TP world size) | 2 |
| logits per rank (padded shard) | 32 .. 262144, last rank >= 32 valid | 75072 (37 tiles), both ranks fully valid |
| vocabulary | <= 2^24, no added (LoRA) vocab | 150144 |
| request rows | 1..2 (`MAX_ROWS`; kernel takes 1..8) | 1..2 |
| top-k | 1..20 per row (12 spare candidates per rank for ties) | 20 |
| top-p | any per row, or none | 0.95 |
| logits | FP16, row-major (any row stride) from an FP16 `lm_head`, no bias, no soft cap, scale 1.0, `head_dtype` unset/FP16 | FP16 (Flint folds `weight_scale_2` into the FP16 weight, M5) |
| generators | none, or exactly one seeded request with 1 row | — |
| noise | FP32 exponential only (`use_fp64_gumbel` off) | off (upstream default) |

`lm_head_reject_reason` decides the LM-head part from configuration only, for
every rank's shard (the last rank holds the fewest valid logits), so all
ranks take the same branch before the all-gather.

Tested: GPU parity at TP {1, 2, 4} x rows {1, 2} with the Flint vocabulary,
TP2 with a padded last shard, k=20; CPU guards for TP {1, 2, 4} and the
rejections (TP 8, > 262144 logits per rank, last rank < 32 valid).

## R4

The fork created the overflow counter lazily on the first sampler call
(`_overflow_monitor`, fork :106-110). Here only `prepare(device)` creates it;
the V1 hook calls it from `kernel_warmup`, after model load and before CUDA
graph capture, and `prepare` raises while a capture runs. The ops allocate
only their outputs and one scratch tensor per call (caching allocator).
`test_descending_capture_regression` ports C4's pattern (FORK
`tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py:764`): capture
batches 8, 4, 2, 1 on a poisoned shared pool with no eager call first, replay
batch 1 first, compare each replay to an eager call from the same RNG state,
and check the counter was not recreated.

## I16 (V1 / V2, FP64 Gumbel)

The hook is V1-only (patch U4). The sampling code (`sample.py`) takes plain
arguments and does not touch the runner, so a V2 hook can reuse it. Upstream
v0.33 added `use_fp64_gumbel`; the kernel reproduces the FP32 draw, so the
hook returns None when it is on, and the parity tests call `random_sample`
with `use_fp64_gumbel=False`.
