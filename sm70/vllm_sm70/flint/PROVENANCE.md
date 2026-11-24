# Flint fused q/k-norm + RoPE + KV write: provenance (patch K15)

Source: charlie-horse/vLLM at **cf9492a01**. The fork item is C5 (`ebff72294`). Line counts are `wc -l`. "fork last-touch" is `git log -1 --format=%h -- <fork path>`. This is Python/Triton only: there is no CUDA source and nothing goes into `_sm70_C`.

## Files

| file | lines | status | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---:|---|---|
| `vllm_sm70/flint/qk_norm_rope_kv.py` | 445 | ported, changes below | `vllm/models/flint/qk_norm_rope_kv.py` | 369 | `0af10b21d323ece09bf7fdcd838201f57505650b5226a131d545935a8ec68267` | ebff72294 |
| `vllm_sm70/flint/__init__.py` | 54 | new (op name + `install_splitting_op`) | replaces `vllm/config/compilation.py:1172-1174` | – | – | – |
| `tests/flint/test_k15_qk_norm_rope_kv_gpu.py` | 229 | ported (`gpu`) | `tests/kernels/core/test_sm70_flint_qk_norm_rope_kv.py` | 197 | `c678e7ac422d30524164efb105803a04caad76ab6e877cebbddf2b31ec709ac6` | ebff72294 |
| `tests/flint/test_k15_qk_norm_rope_kv_admission.py` | 195 | ported (CPU) | `tests/models/flint/test_qk_norm_rope_kv_admission.py` | 117 | `97ced958066932531db46d7a32e4c1724b3a9925af9ee30571ccba1fe53570d8` | ebff72294 |
| `tests/flint/test_k15_qk_norm_rope_kv_cpu.py` + `conftest.py` | 432 + 70 | new (CPU, Triton interpreter) | – | – | – | – |
| `tests/plugin/test_platform_class.py` | +33 | new test for the platform hook | – | – | – | – |
| `vllm_sm70/_cuda_platform.py` | +10 | `check_and_update_config` calls `install_splitting_op` | – | – | – | – |

The kill switch `VLLM_SM70_FLINT_QK_ROPE_KV` and `Sm70Config.flint_qk_rope_kv` were already in `envs.py` and `config/__init__.py` (M1). They replace FORK `envs.py:567,8622` and `KernelConfig.sm70_flint_qk_rope_kv` / `resolve_flint_qk_rope_kv` (FORK `config/kernel.py:766-781`).

## Function by function (`qk_norm_rope_kv.py`)

| port | fork | status |
|---|---|---|
| `_qk_norm_rope_kv_kernel` | `:33-125` | **identical** |
| `qk_norm_rope_kv` | `:128-215` | changed: the RoPE table may be FP16 **or FP32** (`COS_SIN_DTYPES`). The kernel already widened it to FP32, so only the assert changed. Docstring updated. |
| `fused_route_rejection` | `:218-265` | changed: admits an FP32 RoPE table, and `kv_cache_dtype` `"float16"` next to `"auto"` (I7 lists both for V100). The 128 is now the `HEAD_DIM` constant. |
| `_direct_cache_views` | `:268-294` | **rewritten for the upstream KV layout** (next section). Also returns None when `impl._set_kv_ptr_range` is set (ROCm only; never on SM70) and takes `head_dim` from the caller. |
| `qk_norm_rope_kv_cache_update` | `:297-345` | identical apart from the `_direct_cache_views` call |
| `qk_norm_rope_kv_cache_update_fake` | `:348-359` | identical |
| `_register()` | `:364-369` | the same `direct_register_custom_op(mutates_args=["q_out","k_out"])`, now idempotent (skips if already defined). The op is renamed `vllm::sm70_qk_norm_rope_kv_cache_update` (fork: `vllm::qk_norm_rope_kv_cache_update`, `:27-30`) so it cannot clash with upstream. |
| `attention_after_fused_write` | FORK `model_executor/layers/attention/attention.py:535,583,604` (`Attention.forward(..., kv_cache_updated=True)`) | **new**: upstream `Attention.forward` (UP `attention.py:490-581`) has no `kv_cache_updated` flag. This helper does the same as that forward for an unquantized cache, minus `unified_kv_cache_update`: it calls `unified_attention_with_output` directly (or through `torch.ops`). Upstream Flint calls it directly as well (UP `models/flint/model.py:353`). |

`flint/__init__.py` `install_splitting_op` replaces FORK `config/compilation.py:1172-1174`; see "splitting_ops" below.

Dropped:
- `benchmarks/kernels/benchmark_sm70_flint_qk_norm_rope_kv.py`. It is a benchmark, not runtime code. The fork's numbers (2.3 µs/layer fused vs 9.3–10.3 µs unfused) are in the carry map, C5.
- The model wiring (FORK `models/flint/model.py:312-394`: the route decision in `FlintAttention.__init__` and `_forward_fused`). That belongs to the SM70 Flint model (`vllm_sm70/models/flint_sm70`, Mara), which calls `fused_route_rejection`, `torch.ops.vllm.sm70_qk_norm_rope_kv_cache_update` and `attention_after_fused_write`.
- No TP4-only or persistent variants existed in this file.

## Upstream v0.33.0 vs the fork base

### KV-cache layout: changed (hard break, carry map I3)
- **Fork:** the per-layer cache is `[blocks, 2, page, Hkv, D]`. K and V are `kv_cache.unbind(1)` (FORK `v1/attention/backends/triton_attn.py:896`, `get_kv_cache_shape` `:468`).
- **Upstream:** the per-layer cache arrives as logical `(blocks, Hkv, page, 2*D)`, with a physical layout per `KVCacheLayout` (UP `v1/kv_cache_layout.py:15-28`). Flash-V100 declares LBNHC (`attention/flash_v100/backend.py`), which means K and V are interleaved per head. `TritonAttentionImpl.do_kv_cache_update` takes `kv_cache.transpose(1, 2).split(head_size, -1)` → K and V `(blocks, page, Hkv, D)` (UP `triton_attn.py:828`).
- **Port:** `_direct_cache_views` builds exactly that view. It admits `ndim == 4`, `shape[1] == Hkv`, `shape[-1] == 2*D`, fp16, and unit inner stride. The kernel already took explicit block, page and head strides for K and V separately, so it is unchanged. With LBNHC, `head_stride = 2*D` and V sits at `+D` from K. The CPU test checks LBNHC and LBHNC against upstream's own `triton_reshape_and_cache_flash`, which must produce bit-identical cache bytes.
- **The writer gate is unchanged:** `type(impl).do_kv_cache_update is TritonAttentionImpl.do_kv_cache_update`. `FlashAttnV100Impl` (K06) inherits that method, so this kernel writes directly on **both Flash-V100 and plain TRITON_ATTN** (FORK gate `:279`). Any other backend, a per-token-head quantized cache, or an FP8 cache falls back to the backend's own `do_kv_cache_update`. Tests cover each case.

### slot_mapping semantics: unchanged
- `get_attention_context` is line-for-line the same (FORK `attention.py:779-819` vs UP `:685-726`). It returns the per-layer `forward_context.slot_mapping[layer_name]`, or None when it is absent (profiling).
- A slot is `block * page + offset`, and negative slots are padding: UP `gpu_model_runner.py:3870` fills the padded rows with -1, and upstream's writer skips `slot < 0` (UP `triton_reshape_and_cache_flash.py:63-65`).
- Upstream's writer launches over `slot_mapping.shape[0]`. The kernel's `token < num_slots` mask gives the same "rows past the mapping are not written" rule. This is tested.

### RoPE cache API: unchanged, but the table dtype upstream Flint uses changed
- `rotary_embedding/base.py` is byte-identical between fork and UP: `cos_sin_cache` is `[max_pos, rotary_dim]` = `cat(cos, sin)` (UP `base.py:94-102`), and the constructor signature is the same.
- `get_rope` gained `mrope_positions_factor` (mrope only), and the mrope cache sizing changed. Neither matters here: the gate admits only an exact `RotaryEmbedding` (no subclass) with NeoX style and `rotary_dim == head_size == 128`.
- **Behaviour change:** the fork built Flint's table in the model dtype (FP16, FORK `models/flint/model.py:281-286`). Upstream Flint builds it in FP32 (`dtype=torch.float32`, UP `models/flint/model.py:283-291`). The fork's gate would have rejected an FP32 table and silently fallen back to the unfused route. The port admits FP16 and FP32: the kernel reads either and widens it to FP32. With an FP32 table the fused result is slightly more accurate than upstream's unfused path, which casts the table to the query dtype (UP `base.py:105-131`). The GPU and CPU tests check both dtypes against a reference that uses the same table.
- Upstream also has a backend hook, `do_qk_norm_rope_kvcache_update`, with the `fuse_qk_norm_rope_kvcache` pass (UP `v1/attention/backend.py:945-990`). It is ROCm/AITER-only and needs Inductor partitioning (carry map I10). It is not used.

### splitting_ops: the platform hook runs earlier than the fork's code
- The fork appended its op inside `CompilationConfig.set_splitting_ops_for_v1`, right after `unified_kv_cache_update`. Upstream runs the platform hook first (UP `config/vllm.py:2067`) and builds `splitting_ops` later (`:2161`), so at hook time the list is usually still None.
- `install_splitting_op` handles this. When `mode == VLLM_COMPILE`, `splitting_ops is None`, and neither Inductor graph partition nor `fuse_attn_quant` is on (the conditions under which upstream would split the KV update), it materialises upstream's default list by calling upstream's own `set_splitting_ops_for_v1`. It then appends `vllm::sm70_qk_norm_rope_kv_cache_update`. Upstream's later call keeps a non-None list as is; this is tested.
- A user-given list is extended only if it already contains `vllm::unified_kv_cache_update`.
- `VLLM_SM70_FLINT_QK_ROPE_KV=0` leaves `splitting_ops` untouched.
- One side effect: on the first call the DeepEP cudagraph check is skipped (`all2all_backend=""`). Upstream's own call runs it later.

## Admitted shape set (Flint TP2 audit)

Flint has 32 query heads, 4 KV heads and head dim 128 (FORK `docs/design/sm70_flint_tp2_pcie.md:34`). At **TP2 each rank has Hq = 16 and Hkv = 2**; TP1 has 32 and 4.

| guard | where | admits |
|---|---|---|
| `head_dim == 128` | `fused_route_rejection` | 128 only (`tl.arange` needs a power of two). Flint is 128 at every TP. |
| `qkv.shape[1] == (Hq + 2*Hkv) * D` | `qk_norm_rope_kv` | any Hq ≥ 1 and Hkv ≥ 1. The grid is `(tokens, Hq + Hkv)`, and the head counts are constexpr per layer. |
| `key_cache.shape[2:] == (Hkv, D)`, `kv_cache.shape[1] == Hkv` | `qk_norm_rope_kv` / `_direct_cache_views` | rank-local Hkv (2 at TP2) |
| page size | runtime `page_size = key_cache.shape[1]` | any; Flash-V100 uses multiples of 16 |
| tokens | grid dim 0 | 0 (no-op) up to the chunked-prefill size |

The shape guards have no TP-specific constants: nothing assumes TP4 (Hq 8 / Hkv 1). Tested shapes:
- **(16, 2) [TP2]** and (32, 4) [TP1].
- tokens 1/2/7/19 on CPU, 1/2/7/300 on GPU.
- pages 16 and 32.
- RoPE none, FP16 and FP32.
- LBNHC and LBHNC caches.
- a padding slot (-1), a short slot_mapping, negative positions, and zero tokens.
