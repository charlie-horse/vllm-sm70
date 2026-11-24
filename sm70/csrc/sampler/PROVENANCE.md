# Top-k candidate sampler and greedy top-1: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`. License: Apache-2.0.
"last-touch" is `git log -1 --format=%h -- <fork path>`.

| fork file | lines | last-touch | here |
|---|---|---|---|
| `csrc/sm70_turbomind/ops/sm70_topk_candidates_sampler.cu` | 448 | `ae002efc7` | `csrc/sampler/topk_candidates_sampler.cu` (verbatim) |
| `csrc/sm70_turbomind/ops/sm70_top20_philox_fragment.cpp` | 93 | `ae002efc7` | `csrc/sampler/sampler_ops.cpp` (decls :34-41, defs :82-92 only; adapted) |
| `vllm/v1/sample/ops/sm70_topk_candidates.py` | 347 | `484216b2a` | `vllm_sm70/sample.py` (adapted) + `vllm_sm70/ops/sampler.py` (new: op wrappers, fakes) |
| `vllm/v1/worker/gpu_model_runner.py` | :7104-7125, :7646-7862 | — | `vllm_sm70/sample_v1.py`, `vllm_sm70/sample_greedy.py` |
| `vllm/model_executor/layers/logits_processor.py` | :281-386, :817-847 | — | `vllm_sm70/sample_greedy.py` |
| `vllm/envs.py` | `VLLM_SM70_GREEDY_TOKEN_FASTPATH`, `VLLM_SM70_TOP1_CUSTOM_AR` | — | `vllm_sm70/envs.py`, "sampler" section |

## CUDA (`topk_candidates_sampler.cu`, byte-identical to the fork file)

| function | fork line | status |
|---|---|---|
| `half_order` | 69-77 | verbatim |
| `local_top32_tile_kernel` | 79-129 | verbatim |
| `local_top32_merge_kernel` | 131-177 | verbatim |
| `native_exponential_for_index` | 179-199 | verbatim |
| `sample_topk_candidates_kernel` | 201-321 | verbatim |
| `sm70_local_top32_half` (host) | 325-371 | verbatim |
| `sm70_sample_topk_candidates` (host) | 373-448 | verbatim |

## Registration (`sampler_ops.cpp`)

The two defs keep the fork schemas character for character. The namespace is
`_sm70_C` (`TORCH_LIBRARY_FRAGMENT`) instead of `_C`, and the impls are in a
`TORCH_LIBRARY_IMPL(_sm70_C, CUDA)` block.

## Python (`vllm_sm70/sample.py`)

| function | fork line | change |
|---|---|---|
| constants (`LOCAL_K`, `WIRE_WIDTH`, `MAX_TOP_K`, `MAX_ROWS`, ...) | 48-60 | same values; the kernel ones come from `ops/sampler.py` |
| `ops_available` | 63-69 | `ops.sampler.available()` |
| `_is_sm70` | 72-74 | `torch.cuda.get_device_capability` instead of `current_platform` |
| `_OverflowMonitor` | 77-100 | `after_call` skips host bookkeeping during capture |
| `_overflow_monitor` | 106-110 | lookup only; `prepare(device)` creates and refuses during capture |
| `overflow_count` | 113-115 | raises when `prepare()` was not called |
| `metadata_reject_reason` | 118-160 | takes the host `top_k_cpu` as an argument |
| `argmax_invariant_reject_reason` | 163-173 | verbatim |
| `admission_reject_reason` | 176-196 | passes `top_k_cpu` through |
| `lm_head_topk_candidates` | 199-227 | admission in `lm_head_reject_reason`; vLLM imports are local |
| `sample_lm_head_candidates` | 230-254 | `logger.info_once` -> module `_log_once` |
| `local_topk_candidates` | 257-267 | calls `ops.sampler.local_top32_half` |
| `shard_supported` | 270-282 | adds the kernel's row-stride check |
| `sample_topk_candidates` | 285-347 | passes `overflow=None` when no counter was prepared |

## Runner and greedy route (`sample_v1.py`, `sample_greedy.py`)

| fork code | here | change |
|---|---|---|
| `non_argmax_logits_processors_inactive` (runner :7828-7843) | `sample_v1.py` | verbatim logic |
| `_can_use_topk_candidate_sampler`, `_reject_topk_candidate_sampler` (runner :7770-7803) | `Sm70HiddenStateSampler.reject_reason` | pooling / PP checks dropped; adds `not_prepared`, `model`, `fp64_gumbel`, `capturing`, `batch_size` |
| runner :7104-7125 | `Sm70HiddenStateSampler.try_sample` | returns the `SamplerOutput` itself |
| — | `Sm70HiddenStateSampler.prepare` / `prepare_worker` | new; run from `kernel_warmup` before capture |
| `_can_use_greedy_token_fastpath` (runner :7646-7695) | `reject_reason` + `sample_greedy.*_reject_reason` | same rejections, plus thinking budget, capture, row count and rank-uniform LM-head checks |
| `_sample_greedy_token_fastpath` (runner :7845-7862) | `try_sample` greedy branch | returns int32 `[rows, 1]` |
| `get_top_tokens` pair build (logits_processor :304-352) | `sample_greedy.local_pair` | upstream v0.33 `get_top_tokens` first half |
| `_maybe_custom_top1_argmax` (logits_processor :817-847), `CustomAllreduce.top1_argmax` | `sample_greedy.custom_ar_reject_reason`, `greedy_tokens`, `exchange_custom_ar` | on upstream's `ca_comm`; no capture branch; staged exchange (`../custom_ar/PROVENANCE.md`) |

## Dropped

- `sm70_top20_philox_sampler.cu` and its three defs, the compact/chunked
  top-k20 routes and their runner branches.
- `sm70_sampler_graph_branch.cu`.
- The fork `_sm70_sampler_C` module and its CMake target.
- The `flint_sampler` layer policy (a kill switch replaces it).
- DDTree greedy sampling (spec decode).
- The fork `CudaCommunicator` creating a `CustomAllreduce` only for top-1.
