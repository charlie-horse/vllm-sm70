# Flint fused SM70 decode router: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`. License: Apache-2.0. Line
counts are `wc -l`; "fork last-touch" is `git log -1 --format=%h -- <fork
path>`.

## Files

| file | lines | status | sha256 | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---|---:|---|---|
| `flint_router.cuh` | 186 | verbatim | `e9c8a1885b203ebdfd9c2b1739d3a7b8aeb0808949f01aa11677babe30fe25ce` | `csrc/sm70_turbomind/ops/sm70_flint_router.cuh` | 186 | `e9c8a1885b203ebdfd9c2b1739d3a7b8aeb0808949f01aa11677babe30fe25ce` | dbd60bff2 |
| `flint_router_ops.cu` | 97 | adapted | see git | `csrc/sm70_turbomind/ops/sm70_router.cu` | 139 | `40cb928879535351b6a759c6da524d99bfcfa0f93ca3bbb25840003e140f4ac8` | dbd60bff2 |
| `sources.cmake` | 18 | new (replaces fork `CMakeLists.txt:841-846`, the `_sm70_router_C` target) | see git | – | – | – | – |
| `vllm_sm70/router.py` | ~300 | adapted | see git | `vllm/model_executor/layers/fused_moe/router/sm70_flint_router.py` | 205 | `1258900fea41765a88bdbeaa2e2c248f42c47631e13cd4b84a27f90aa9adfcf7` | 3629e29f6 |

## Function map

| port | fork | change |
|---|---|---|
| `flint_router.cuh`: `warp_sum`, `sigmoid_score`, `select_topk`, `route_kernel<M>` | `sm70_flint_router.cuh` (all) | none (byte-identical) |
| `flint_router_ops.cu` `flint_route` | `sm70_router.cu:72-123` | none in the body; in `namespace vllm_sm70::router` |
| `flint_router_ops.cu` `check_sm70` | `sm70_router.cu:27-31` | none |
| schema `_sm70_C::flint_route(Tensor(a!) weights, Tensor(b!) ids, Tensor x, Tensor gate, Tensor(c!) logits, Tensor(d!) counter) -> ()` | `sm70_router.cu:131-133` (`vllm_sm70_router::flint_route`) | namespace `_sm70_C`, `TORCH_LIBRARY_FRAGMENT`, impl under `CUDA` |
| `router.py` `FlintFusedRouter.__init__`, `accepts`, `route` | `sm70_flint_router.py:74-81, 83-93, 112-128` | `__init__` takes the runner, not the router |
| `router.py` `FlintFusedRouter.try_route` | `:95-110` | the id-dtype check reads upstream `runner._quant_method.topk_indices_dtype`. It also defers when upstream's `router._routing_replay_out` is set |
| `router.py` `_rejection` | `:131-176` | the gate check uses upstream `GateLinear.is_unquantized` in place of the fork's `allow_dsv3_router_gemm`/`allow_cublas_router_gemm` |
| `router.py` `admit_flint_fused_router` | `:179-205` | the kill switch is read from `vllm_sm70.envs`. Admission refuses under capture |
| `router.py` `_flint_route_op`, `has_op` | `:42-51` | `_ext.load()` plus `torch.ops._sm70_C` in place of `vllm._sm70_router_C` |
| `router.py` `_is_exact_sm70` | `:60-64` | cc 7.0 only |
| `router.py` `flint_router_provider`, `provider`, `prepare_model`, `register`, `_register_fake`, `_capturing` | fork `moe_runner.py:351-369` (`_maybe_precomputed_route`: decide once, never under capture) | the decide-once logic is in the provider; `provider()` is the platform getter's answer, with the kill switch read once |
| `Sm70CudaPlatform.get_moe_router_provider`, `.kernel_warmup` | fork `models/flint/model.py:229` bound the provider per layer in the model | bound by the platform getter and the warmup hook |

## Dropped

- `top10` and `select_quantize` with their checks (`sm70_router.cu:13-71`)
  and `sm70_router.cuh` (the GGUF E512 router).
- `TORCH_LIBRARY(vllm_sm70_router)` and `PYBIND11_MODULE`; the op is a
  fragment of `_sm70_C`.
- `_policy_enabled` / `vllm.config.sm70_moe`, replaced by the kill switch.
- The fork admission test `test_kill_switch_reads_routing_policy`.
