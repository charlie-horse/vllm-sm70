# Flint fused SM70 decode router: provenance (patch K13)

Source: charlie-horse/vLLM at **cf9492a01**. Line counts are `wc -l`. "fork last-touch" is `git log -1 --format=%h -- <fork path>`.

**Depends on U3 (Penny's upstream PR, not in this branch).** K13 implements the fork's router-provider contract at the gate site of UP `model_executor/layers/fused_moe/runner/moe_runner.py:907` (fork `moe_runner.py:205-216,307-309,351-369,995-1028`):

- `MoERunner.router_provider: Callable[[MoERunner], PrecomputedRoute | None]`, bound once per layer on the first eager forward and never during CUDA graph capture.
- `PrecomputedRoute.try_route(hidden_states) -> (topk_weights fp32, topk_ids int32) | None`. The runner skips the gate GEMM and `select_experts` when it returns a pair, and hands the pair to the quant method as `precomputed_topk`.
- A platform getter that is probably `Platform.get_moe_router_provider()`. `Sm70CudaPlatform` implements it.

Without U3 everything in this patch is inert: no runner calls the provider, and `prepare_model` binds nothing because no module has `router_provider`. The CPU hook test (`test_u3_runner_hook_admits_once_outside_capture`) skips on a vLLM without U3. It was run against a local copy of the contract, which is not shipped.

## Files

| file | lines | status | sha256 | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---|---:|---|---|
| `flint_router.cuh` | 186 | verbatim | `e9c8a1885b203ebdfd9c2b1739d3a7b8aeb0808949f01aa11677babe30fe25ce` | `csrc/sm70_turbomind/ops/sm70_flint_router.cuh` | 186 | `e9c8a1885b203ebdfd9c2b1739d3a7b8aeb0808949f01aa11677babe30fe25ce` | dbd60bff2 |
| `flint_router_ops.cu` | 102 | trimmed | see git | `csrc/sm70_turbomind/ops/sm70_router.cu` | 139 | `40cb928879535351b6a759c6da524d99bfcfa0f93ca3bbb25840003e140f4ac8` | dbd60bff2 |
| `sources.cmake` | 18 | new (replaces fork `CMakeLists.txt:841-846`, the `_sm70_router_C` target) | see git | – | – | – | – |
| `vllm_sm70/router.py` | ~300 | adapted | see git | `vllm/model_executor/layers/fused_moe/router/sm70_flint_router.py` | 205 | `1258900fea41765a88bdbeaa2e2c248f42c47631e13cd4b84a27f90aa9adfcf7` | 3629e29f6 |

## Function map

| port | fork | change |
|---|---|---|
| `flint_router.cuh`: `warp_sum`, `sigmoid_score`, `select_topk`, `route_kernel<M>` | `sm70_flint_router.cuh` (all) | none (byte-identical) |
| `flint_router_ops.cu` `flint_route` | `sm70_router.cu:72-123` | none in the body. It now lives in `namespace vllm_sm70::router` |
| `flint_router_ops.cu` `check_sm70` | `sm70_router.cu:27-31` | none |
| schema `_sm70_C::flint_route(Tensor(a!) weights, Tensor(b!) ids, Tensor x, Tensor gate, Tensor(c!) logits, Tensor(d!) counter) -> ()` | `sm70_router.cu:131-133` (`vllm_sm70_router::flint_route`) | namespace `_sm70_C`, `TORCH_LIBRARY_FRAGMENT`, impl under `CUDA` |
| `router.py` `FlintFusedRouter.__init__`, `accepts`, `route` | `sm70_flint_router.py:74-81, 83-93, 112-128` | `__init__` takes the runner, not the router |
| `router.py` `FlintFusedRouter.try_route` | `:95-110` | the id-dtype check reads UP `runner._quant_method.topk_indices_dtype` (the fork used `router._get_indices_type()`, which is gone in UP). It also defers when UP's `router._routing_replay_out` is set (new in UP, `router/fused_moe_router.py:18,95`) |
| `router.py` `_rejection` | `:131-176` | the gate check uses UP `GateLinear.is_unquantized` in place of the fork's `allow_dsv3_router_gemm`/`allow_cublas_router_gemm`. On SM70 every UP specialized router-GEMM tier is off anyway (Hopper/Blackwell only, UP `gate_linear.py:83-88`) |
| `router.py` `admit_flint_fused_router` | `:179-205` | the kill switch is read from `vllm_sm70.envs` (fork: `Sm70MoERoutingPolicy`). Admission refuses under capture |
| `router.py` `_flint_route_op`, `has_op` | `:42-51` | `_ext.load()` plus `torch.ops._sm70_C` in place of `vllm._sm70_router_C` |
| `router.py` `_is_exact_sm70` | `:60-64` | cc 7.0 only (the fork also accepted 7.2) |
| `router.py` `provider`, `prepare_model`, `register`, `_register_fake`, `_capturing` | – | new: the platform getter, the warmup-time admission (R4), the fake |
| `Sm70CudaPlatform.get_moe_router_provider`, `.kernel_warmup` | fork `models/flint/model.py:229` bound the provider per layer in the model | the plugin can't edit the model, so the platform getter (U3) and the U1 warmup hook bind it |

## Dropped

- `top10` and `select_quantize` with their checks (`sm70_router.cu:13-71`) and `sm70_router.cuh`. That is the GGUF E512 router, which neither Flint nor Qwen uses.
- `TORCH_LIBRARY(vllm_sm70_router)` and `PYBIND11_MODULE` (`:125-139`). The op is a fragment of `_sm70_C` instead.
- `_policy_enabled` / `vllm.config.sm70_moe` (`sm70_flint_router.py:54-57`), replaced by the kill switch.
- The fork admission test `test_kill_switch_reads_routing_policy`, because its policy object is gone. `test_kill_switch_only_zero_disables` replaces it.

## Admitted shape set

The decision is made once per layer, at admission:
- gate: an unquantized `GateLinear` with an **FP32** weight `[256, 2048]`, contiguous, 16-byte aligned, no bias, `out_dtype == float32`
- router: exactly `FusedTopKRouter` with sigmoid, renormalize, top_k 6, 256 global experts, no EPLB
- layer: no fused shared gate, no routed-input transform, no DBO, not monolithic
- parallelism: `dp_size == pcp_size == 1`, no EP. Any TP size works: the gate is replicated, so TP2 on 2×V100 is admitted
- device: cc 7.0

Then on each call, from tensor metadata only (fixed for a captured graph):
- hidden states FP16 `[M, 2048]` with M ∈ {1, 2}, unit inner stride, row stride a multiple of 4, 8-byte aligned base
- no routed-id `capture_fn`, no routing replay, id dtype None or int32

Everything else takes the generic route. The tests cover each rejection (`sm70/tests/router/test_k13_router_cpu.py`).

**Flint at `--dtype half` on upstream v0.33.0 has an FP16 gate** (UP `models/flint/model.py:149-154` builds `GateLinear` with no `params_dtype` or `out_dtype`), so it is **not admitted** as is. The fork built the gate with `out_dtype=torch.float32, force_fp32_compute=True` (fork `models/flint/model.py:196-202`), which keeps an FP32 weight on SM70. Mara's Flint adapter has to do the same for K13 to engage. `test_other_layers_keep_generic_route[gate5]` pins this.

## Capture safety (R4)

- The per-layer workspace (FP32 logits `[2*256]`) and the INT32 arrival counter are allocated only at admission, which happens in `prepare_model` (the U1 `kernel_warmup` hook, eager, before capture) or on the first eager forward (U3). Admission returns None under capture.
- The kernel's last CTA re-arms the counter, so graphs of any batch size can replay in any order.
- Regression test: `test_descending_capture_regression` (GPU). It captures batch 2 then batch 1 on a fresh stream and a poisoned shared pool, replays batch 1 first, and checks that capture created no new tensors.
