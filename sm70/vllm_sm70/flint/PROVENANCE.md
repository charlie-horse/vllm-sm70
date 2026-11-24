# Flint fused q/k-norm + RoPE + KV write: provenance

## Source

- Repository: charlie-horse/vLLM (a vLLM fork), commit `cf9492a01`.
- Path: `vllm/models/flint/qk_norm_rope_kv.py`, sha256
  `0af10b21d323ece09bf7fdcd838201f57505650b5226a131d545935a8ec68267`.
- `vllm_sm70/flint/__init__.py` (`install_splitting_op`) is new; it does
  what the fork did in `vllm/config/compilation.py` (`set_splitting_ops_for_v1`).

## License

Apache-2.0, copyright contributors to the vLLM project (SPDX header kept).

## Changes from the source (`qk_norm_rope_kv.py`)

| item | change |
|---|---|
| `_qk_norm_rope_kv_kernel` | none |
| `qk_norm_rope_kv` | the RoPE table may be FP16 or FP32 (`COS_SIN_DTYPES`); the kernel widens it to FP32 |
| `fused_route_rejection` | admits an FP32 RoPE table and `kv_cache_dtype` `"float16"` next to `"auto"`; 128 is the `HEAD_DIM` constant |
| `_direct_cache_views` | rewritten for vLLM's logical `(blocks, Hkv, page, 2*D)` cache, viewed as K and V `(blocks, page, Hkv, D)` as `TritonAttentionImpl.do_kv_cache_update` does; returns None when `impl._set_kv_ptr_range` is set; takes `head_dim` from the caller |
| `qk_norm_rope_kv_cache_update` | only the `_direct_cache_views` call |
| `qk_norm_rope_kv_cache_update_fake` | none |
| `_register()` | idempotent; the op is named `vllm::sm70_qk_norm_rope_kv_cache_update` |
| `attention_after_fused_write` | new: replaces the fork's `Attention.forward(..., kv_cache_updated=True)`, which vLLM does not have |

Not taken: the benchmark script and the model wiring (the SM70 Flint model
in `vllm_sm70/models/flint_sm70` calls these functions).
