# Provenance: flint_sm70

Ported from charlie-horse/vLLM (a vLLM fork) at commit `cf9492a`; not
byte-identical copies. License: Apache-2.0, as vLLM (SPDX headers kept).

| File | Source path |
|---|---|
| `model.py`, `fp16_rescale.py` | `vllm/models/flint/model.py` |
| `config.py` | `vllm/model_executor/models/config.py` (`FlintModelConfig`, Flash-V100 sink admission) |

Changes from the source, on upstream vLLM v0.33.0 layers:

| Source | Here |
|---|---|
| `FusedMoE(expert_mapping=...)`, `_expert_params_mapping`, `get_expert_mapping` | `FusedMoEFactory`; `RoutedExperts.load_weights` builds the mapping |
| `runner.router_provider` (fused router hook) | the platform's MoE router getter; no model hook |
| `Attention.forward(kv_cache_updated=)` | the `vllm_sm70/flint/` op, then `attention_after_fused_write` |
| `kernel_config.sm70_flint_qk_rope_kv` | `sm70_config(vllm_config).flint_qk_rope_kv` |
| sampler hooks `_get_top_tokens`, `_sample_topk_candidates` | `supports_sm70_topk_sampler = True` (platform sampler hook) |
| `AutoWeightsLoader(skip_prefixes=)` | `lm_head.*` filtered when tied |
| FP16 rescale helpers in `model.py` | `fp16_rescale.py` |
| `attention_config.flash_v100.enabled`, `flash_attn_v100.SUPPORTS_SINKS` | `VLLM_SM70_FLASH_V100`, `vllm_sm70.ops.fv100.supports_sinks()` |
| Flash-V100 sink admission | also requires a KV block size that is a multiple of 16 and that CUSTOM resolves to the Flash-V100 backend |
| TP2 all-reduce + RMSNorm fusion | not ported |
