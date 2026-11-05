# FlashQLA SM70 GDN kernels: provenance (patch K08)

Source: charlie-horse/vLLM at **cf9492a01**. Line counts are `wc -l`. "fork last-touch" is `git log -1 --format=%h -- <fork path>`. Re-verify the function-by-function comparison with:

```
scripts/verify-port.py --port sm70/csrc/gdn/gdn_forward.cu \
  --fork <fork>/flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu
```

At the K08 commit it reports IDENTICAL=20, CHANGED=3, NEW=10, DROPPED=23. Every CHANGED, NEW and DROPPED row is listed below.

## Files

| file | lines | status | sha256 | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---|---:|---|---|
| `gdn_forward.cu` | 1025 | rewritten | `5469e0ffa606a8a300e9ace40636f2b58c011cc377f8bd2ec278a58e937c5c02` | `flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu` | 1959 | `910e050bdd2600b10c50fc14480f77b01d48dad8a5347261c525b4f09c13e408` | ec1484d7e |
| `LICENSE` | 21 | verbatim | `d0b7a111f48dc0207c7e057c810ae0e2fdbc807ba654eb669ade998b2cdc3438` | `flash_qla/LICENSE` | 21 | `d0b7a111f48dc0207c7e057c810ae0e2fdbc807ba654eb669ade998b2cdc3438` | 25733d9cf |
| `NOTICE` | 24 | new | `ca89dc0d0f9c91273f07baa336c40cac8f78d6d7c0b4389b55b7a3a489200009` | – | – | – | – |
| `sources.cmake` | 16 | new (replaces fork `setup.py:262-300` `bundle_flash_qla_sm70`) | `31a9492a0141430b561c651dfa49147f6eb848a3750d75146b319b63681745de` | – | – | – | – |

The history on branch `kernels/k08-gdn` has four commits for `gdn_forward.cu`: a byte-identical import (sha256 = fork sha256), deletions only, the registration, then the null-block fix. Review each one against the fork separately.

## Python counterpart

| port | fork |
|---|---|
| `vllm_sm70/gdn/kernels.py` `gdn_forward_vlk_varlen_out`, `gdn_decode_mixed_qkv_global_state`, `gdn_resolve_column_groups_per_block` (wrappers + fakes) | `flash_qla/ops/gated_delta_rule/chunk/sm70/fused_fwd.py` `_load_ext` (:29-91) plus the pybind calls in `chunk_gated_delta_rule_fwd_sm70_vlk_varlen` (:269-322) and `gdn_decode_mixed_qkv_global_state_sm70` (:325-417) |
| `kernels.py` `flashqla_chunk_gated_delta_rule`, `prefill_rejection` | `vllm/model_executor/layers/fla/ops/sm70/gdn_prefill.py` `flashqla_sm70_chunk_gated_delta_rule`, non-TileLang branch (:69-247). Admission comes from `fla/ops/gdn_selector.py:123-180` (sm70, fp16, head_k_dim 128) |
| `kernels.py` `flashqla_decode`, `decode_rejection` | `vllm/model_executor/layers/fla/ops/sm70/gdn_decode.py` `flashqla_decode_rejection`, `flashqla_decode` (:60-145), plus the checks in `fused_fwd.py:338-399` |
| `kernels.py` `register_chunk_gated_delta_rule` (OOT `ChunkGatedDeltaRule`) | fork `qwen_gdn_linear_attn.py:1170-1201` (`ChunkGatedDeltaRule` with the `flashqla_sm70` backend) |

Dropped on the Python side:
- The pybind loader, the prebuilt-path env and the JIT fallback (`fused_fwd.py:16-91`).
- `bind_flashqla_native_policy` and `GdnPolicy`.
- The TileLang branch (that is K10).
- `GdnPrefillProfiler`.
- The DDTree decode (`fused_fwd.py:420-545`).

## What changed in `gdn_forward.cu`, function by function

### Commit 2: deletions only (1959 → 1005 lines)
Deleted (fork lines):
- `gdn_forward_kernel` (76-233). It was the fixed-batch forward and is used only by the fork's standalone `chunk_gated_delta_rule_fwd_sm70`.
- `gdn_decode_mixed_qkv_ddtree_state_kernel` (582-806).
- `launch_gdn_forward_typed`, `launch_gdn_forward_kv` (807-892).
- `launch_gdn_decode_mixed_qkv_ddtree_state_typed`, `launch_gdn_decode_mixed_qkv_ddtree_state_kv` (1059-1161).
- `get_ddtree_column_groups_per_block` (1252-1265).
- Host `gdn_forward` (1272-1412).
- Host `gdn_decode_mixed_qkv_ddtree_state` (1734-1909).
- `GdnPolicy`, `GdnPolicyScope`, `configured_gdn`, `PYBIND11_MODULE` (1910-1959).

### Commit 3: registration (rewritten)
- **Includes:** `#include <torch/extension.h>` became `<torch/all.h>` + `<torch/library.h>`, and an SPDX/MIT header was added.
- **`get_column_groups_per_block`:**
  - Removed the `thread_local active_column_groups` policy and the `FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK` getenv/atoi override.
  - The shape heuristic is unchanged.
- **`resolve_column_groups_per_block` → `gdn_resolve_column_groups_per_block`:** takes `int64_t` (schema `int`) and returns `int64_t`.
- **`gdn_forward_vlk_varlen` → `gdn_forward_vlk_varlen_out`:**
  - `output` and `final_state` are leading caller-allocated arguments (`Tensor(a!)`, `Tensor(b!)`).
  - Dropped `output_final_state`, `validate_cu_seqlens` (its two `.item()` host syncs) and `output_arg`.
  - `final_state` is validated to be fp32 `[N, Hv, V, K]`, and nothing is returned.
  - `c10::optional` became `std::optional`.
  - Kernel launch and dtype dispatch are unchanged.
- **`gdn_decode_mixed_qkv_global_state`:** host function unchanged.
- **New:** `TORCH_LIBRARY_FRAGMENT(_sm70_C)` with the three defs; `TORCH_LIBRARY_IMPL(_sm70_C, CUDA)` for the two tensor ops; `CompositeExplicitAutograd` for `gdn_resolve_column_groups_per_block`.

### Commit 4: null-block slot (rewritten)
- **`gdn_decode_mixed_qkv_global_state_kernel`:** the invalid-slot test went from `slot < 0` to `slot <= 0`.
  - Slot 0 is vLLM v0.33's `NULL_BLOCK_ID` (UP `vllm/v1/attention/backends/utils.py:55`), and upstream's packed decode skips `state_idx <= 0` (UP `third_party/flash_linear_attention/ops/fused_recurrent.py`, packed decode kernel).
  - With the fork's test, CUDA-graph padding rows would write the null block's state and a non-zero output.

Everything else is unchanged: `gdn_forward_vlk_varlen_kernel`, the load/store helpers, the subgroup reductions, the `launch_*_vlk_varlen_*` and `launch_*_global_state_*` launchers, and the `validate_*` helpers.
