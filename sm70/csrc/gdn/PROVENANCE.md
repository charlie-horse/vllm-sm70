# FlashQLA SM70 GDN kernels: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`,
`flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu` (FlashQLA by
the Qwen team, Alibaba Group). License: MIT, `LICENSE`; origin and notice in
`NOTICE`. Line counts are `wc -l`; "fork last-touch" is
`git log -1 --format=%h -- <fork path>`.

## Files

| file | lines | status | sha256 | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---|---:|---|---|
| `gdn_forward.cu` | 1021 | rewritten | `428ae1a67c60b419dfe21df53589e15919002225b6274b00893cdf1bbd9266e1` | `flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu` | 1959 | `910e050bdd2600b10c50fc14480f77b01d48dad8a5347261c525b4f09c13e408` | ec1484d7e |
| `LICENSE` | 21 | verbatim | `d0b7a111f48dc0207c7e057c810ae0e2fdbc807ba654eb669ade998b2cdc3438` | `flash_qla/LICENSE` | 21 | `d0b7a111f48dc0207c7e057c810ae0e2fdbc807ba654eb669ade998b2cdc3438` | 25733d9cf |
| `NOTICE` | 24 | new | `ca89dc0d0f9c91273f07baa336c40cac8f78d6d7c0b4389b55b7a3a489200009` | – | – | – | – |
| `sources.cmake` | 14 | new (replaces fork `setup.py:262-300` `bundle_flash_qla_sm70`) | `0df254c465d25341321734167b9ad6d5325f9dc5710102d01af239ef07b33533` | – | – | – | – |

## Changes in `gdn_forward.cu`

- Deleted (fork lines): `gdn_forward_kernel` (76-233),
  `gdn_decode_mixed_qkv_ddtree_state_kernel` (582-806),
  `launch_gdn_forward_typed`/`_kv` (807-892), the DDTree launchers
  (1059-1161), `get_ddtree_column_groups_per_block` (1252-1265), host
  `gdn_forward` (1272-1412), host `gdn_decode_mixed_qkv_ddtree_state`
  (1734-1909), `GdnPolicy`, `GdnPolicyScope`, `configured_gdn` and
  `PYBIND11_MODULE` (1910-1959).
- Includes: `<torch/all.h>` + `<torch/library.h>` instead of
  `<torch/extension.h>`; SPDX/MIT header added.
- `get_column_groups_per_block`: the thread-local policy and the
  `FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK` override removed; the shape
  heuristic is unchanged.
- `resolve_column_groups_per_block` -> `gdn_resolve_column_groups_per_block`
  (`int64_t` in and out).
- `gdn_forward_vlk_varlen` -> `gdn_forward_vlk_varlen_out`: caller-allocated
  `output` and `final_state` (`Tensor(a!)`, `Tensor(b!)`, fp32
  `[N, Hv, V, K]` checked); `output_final_state`, `validate_cu_seqlens` and
  `output_arg` dropped; `std::optional`.
- `gdn_decode_mixed_qkv_global_state_kernel`: invalid slot is `slot <= 0`
  (was `< 0`), as upstream's packed decode skips `state_idx <= 0`
  (`NULL_BLOCK_ID` is 0).
- Registration: `TORCH_LIBRARY_FRAGMENT(_sm70_C)` with three defs; CUDA impls
  for the tensor ops, `CompositeExplicitAutograd` for
  `gdn_resolve_column_groups_per_block`.
- Comments edited.

## Python counterpart

| port | fork |
|---|---|
| `vllm_sm70/gdn/kernels.py` `gdn_forward_vlk_varlen_out`, `gdn_decode_mixed_qkv_global_state`, `gdn_resolve_column_groups_per_block` (wrappers + fakes) | `flash_qla/ops/gated_delta_rule/chunk/sm70/fused_fwd.py` `_load_ext` (:29-91) plus the pybind calls in `chunk_gated_delta_rule_fwd_sm70_vlk_varlen` (:269-322) and `gdn_decode_mixed_qkv_global_state_sm70` (:325-417) |
| `kernels.py` `flashqla_chunk_gated_delta_rule`, `prefill_rejection` | `vllm/model_executor/layers/fla/ops/sm70/gdn_prefill.py` `flashqla_sm70_chunk_gated_delta_rule`, non-TileLang branch (:69-247). Admission comes from `fla/ops/gdn_selector.py:123-180` (sm70, fp16, head_k_dim 128) |
| `kernels.py` `flashqla_decode`, `decode_rejection` | `vllm/model_executor/layers/fla/ops/sm70/gdn_decode.py` `flashqla_decode_rejection`, `flashqla_decode` (:60-145), plus the checks in `fused_fwd.py:338-399` |
| `kernels.py` `register_chunk_gated_delta_rule` (OOT `ChunkGatedDeltaRule`) | fork `qwen_gdn_linear_attn.py:1170-1201` (`ChunkGatedDeltaRule` with the `flashqla_sm70` backend) |

Dropped on the Python side: the pybind loader and JIT fallback,
`bind_flashqla_native_policy` and `GdnPolicy`, the TileLang branch,
`GdnPrefillProfiler`, the DDTree decode.
