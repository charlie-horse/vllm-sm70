# QPN8 block-FP8 decode GEMV: provenance (patch K09)

Source: charlie-horse/vLLM at **cf9492a01**. Line counts are `wc -l`. "fork last-touch" is `git log -1 --format=%h -- <fork path>`. Re-verify with:

```
sm70/tools/check_qpn8_provenance.py --fork <fork checkout at cf9492a>
scripts/verify-port.py --port sm70/csrc/qpn8/fp8_qpn8_sm70.cu \
  --fork <fork>/csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu
```

At the K09 head verify-port reports IDENTICAL=21, CHANGED=8, NEW=2, DROPPED=41. Each CHANGED and NEW row is explained below. The DROPPED rows are the list under "Commit 2", minus what commit 4 restored.

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---|---|---|---|---|
| `sm70/csrc/qpn8/fp8_qpn8_sm70.cu` | `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | rewritten | 978 | 2248 | 495fb442a | `527c322ba9edd252b3f1a3e9797bad135073c7d736121a5f8965dfccb2f3ba96` | `7900034f1057a66f99ada9673d64acc98a0bc40ed6069c1ade361ffc02c5ce6d` |
| `sm70/csrc/qpn8/LICENSE.v100-skinny` | `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` | verbatim | 21 | 21 | 58beb3433 | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` |
| `sm70/csrc/qpn8/qpn_pair_sm70.cuh` | `csrc/sm70_turbomind/ops/qpn_pair_sm70.cuh` | verbatim | 135 | 135 | aeede18c5 | `ed51466066679d5291129f0a9a7f64ef565632f78f0489054acd217cad773a32` | `ed51466066679d5291129f0a9a7f64ef565632f78f0489054acd217cad773a32` |
| `sm70/csrc/qpn8/activation_pack_sm70.cuh` | `csrc/sm70_turbomind/ops/activation_pack_sm70.cuh` | verbatim | 23 | 23 | aeede18c5 | `dda71dbdd6313c2000c859cea2e990b39017ee92e83304a31c4fbdd8b68f7cbb` | `dda71dbdd6313c2000c859cea2e990b39017ee92e83304a31c4fbdd8b68f7cbb` |
| `sm70/csrc/qpn8/sources.cmake` | - | new | 17 | - | - | `49c86b2692cfa70f3c58dfba8a937a9e69184e6e0873eb9d79042d8ea23c62a8` | - |

Not carried: `sm70_policy.h` (the fork's native policy). `qpn_pair_sm70.cuh` and `activation_pack_sm70.cuh` were dropped in commit 2 and restored verbatim in commit 4; they serve channel-scaled M9..32 only.

## Python counterpart (`sm70/vllm_sm70/ops/qpn8.py`)

| port | fork |
|---|---|
| `fp8_qpn8_prepare_sm70`, `fp8_qpn8_dequantize_sm70_out`, `fp8_qpn8_gemm_sm70_out` wrappers and `register_fake`s | `vllm/_sm70/linear.py` wrappers of the same ops (with `native_policy`) and `csrc/torch_bindings.cpp:723-758` (schemas) |
| `QPN8_CONFIGS`, `qpn8_config()` | `vllm/model_executor/kernels/linear/qpn/fp8.py:55-63` (`_SM70_FP8_QPN8_CONFIGS`) and `:208-211` (`_sm70_fp8_qpn8_config`), plus a validity check against the C++ guards |
| `qpn8_rejection()` | `qpn/fp8_block.py:162-195` (`can_implement`, shape part) |
| `QPN8_MAX_M = 32`, `needs_turbomind_fallback()` | `qpn/fp8_block.py:45` (`_QPN8_MAX_M = 8`) and `:204-213` (`_keep_volta_turbomind_copy`). The cap is 32 here: the kernel serves block-scaled M9..32 (commit 5). It is the single M cap; `test_m_cap_has_one_source` checks that the C++ bound, the dispatch and the TurboMind rule all match it |
| `qpn8_prepare()`, `Qpn8Prepared` | `qpn/fp8_block.py:215-277` (`process_weights_after_loading`); buffers become dataclass fields |
| `torch.ops.vllm_sm70.qpn8_native_linear` (custom op + fake), `qpn8_linear()` | `qpn/fp8_block.py:50-134` (`sm70_fp8::qpn8_native_linear`) and `:284-309` (`apply_weights`). The `native_policy` argument is gone. The M>QPN8_MAX_M branch without TurboMind replaces `fp8_qpn8_prefill_sm70_out(int dense_weight_ptr)` with `fp8_qpn8_dequantize_sm70_out` into a caller `dense_scratch` (`Tensor(a!)?`) or a per-call tensor, then `torch.mm` |

Not ported: the fork's TP4 suffix/shape admission (`qpn/fp8.py:214-232` `_is_sm70_fp8_qpn8_layer`, `_SM70_FP8_PREFILL_DENSE_SHAPES`, `_SM70_FP8_QPN8_EXTRA_SHAPES`, the PP2/TP4 tables). The block kernel admits any N, K divisible by 128.

## What changed in `fp8_qpn8_sm70.cu`, function by function

### Commit 1: verbatim import
sha256 equals the fork sha256 (`7900034f...`).

### Commit 2: deletions only (2248 → 697 lines)
Deleted (fork lines):
- **Includes:** `sm70_policy.h`, `activation_pack_sm70.cuh`, `qpn_pair_sm70.cuh` (24-26).
- **Kernels and launchers:**
  - `fp8_qpn8_silu_and_mul_sm70_kernel` (156-171) and `fp8_qpn8_ba_split_copy_sm70_kernel` (181-210).
  - Everything from `Fp8PairReader` to the end of the anonymous namespace (466-1117): `fp8_qpn8_m32_twophase_sm70_kernel` and its launcher, `launch_fp8_qpn8_ba_split_sm70`, `fp8_qpn8_gated_pair_sm70_kernel` and its launcher, the split-CTA M1 stage1/reduce kernels, and the HC down/up/transform/gate-mix kernels.
- **Host ops:**
  - `fp8_qpn8_split_cta_m1_sm70_out`, `fp8_qpn8_hc_down_silu_sm70_out`, `fp8_qpn8_hc_up_gate_mix_sm70_out` (1120-1272).
  - `fp8_qpn8_prefill_sm70_out` (1383-1421; it took a raw `int dense_weight_ptr`).
- **`fp8_qpn8_gemm_sm70_out` branches:** the M9..16 `launch_qpn_pair_m16` branch and the M17..32 two-phase branch (1499-1547).
- **After the GEMM (1640-2248):** `fp8_qpn8_gemm_ba_split_sm70_out`, both `fp8_qpn8_dispatch_sm70_out` overloads, `fp8_qpn8_dispatch_ba_split_sm70_out`, `fp8_qpn8_gated_pair_sm70_out`, `fp8_qpn8_hc_dispatch_sm70_out`, and the `#ifdef VLLM_QPN8_STANDALONE` `TORCH_LIBRARY_FRAGMENT(_C_qwen38/_C)` block.

### Commit 3: adaptation (rewritten)
- **Header comment:** added the vllm-sm70 note.
- **`fp8_qpn8_gemm_sm70_out` (CHANGED):**
  - `TORCH_CHECK(m >= 1 && m <= 32)` became `m <= 8`.
  - Removed the three M>8-only guards: "M=9..32 requires channel scales", "M=9..32 does not support split_k 32", "M=17..32 requires split_k 12/16 ...".
  - The remaining guards are shape-generic: N%32, block K%128 (channel K%16), `codes.numel() == N*K`, split_k ∈ {4, 8, 12, 16, 32}, `(K/16) % split_k == 0`, chains ∈ {1, 2}, prefetch ⇒ fast decoder, split 12 ⇒ 2 chains, fast decoder and no prefetch. None names a model or TP shape.
  - The M=1 split-12/16 specialisation is unchanged.
- **`VLLM_LAUNCH_QPN8` (CHANGED):** keeps only the one-row-tile (`RowTiles=1`) launch. The `RowTiles=2` M9..16 launch and its split-32 refusal are gone.
- **Registration (NEW):**
  - `TORCH_LIBRARY_FRAGMENT(_sm70_C)` defines `fp8_qpn8_prepare_sm70(Tensor qweight, Tensor scales) -> Tensor[]`, `fp8_qpn8_dequantize_sm70_out(Tensor(a!) out, ...)` and `fp8_qpn8_gemm_sm70_out(Tensor(a!) out, ...)`, each without the fork's trailing `str? native_policy`.
  - `TORCH_LIBRARY_IMPL(_sm70_C, CUDA)` holds the three impls, without `with_policy`.
  - The opdef/opimpl rows that verify-port reports CHANGED are these schema and namespace edits.

### Commit 4: restore M9..32 verbatim
- Adds `qpn_pair_sm70.cuh` and `activation_pack_sm70.cuh` byte for byte.
- Puts back, unchanged:
  - their includes (fork 25-26);
  - `Fp8PairReader`, `fp8_qpn8_m32_twophase_sm70_kernel` and `launch_fp8_qpn8_m32_twophase_sm70` (fork 466-658);
  - the fork's whole `fp8_qpn8_gemm_sm70_out` (fork 1422-1639): M in [1, 32], the M9..16 pair and M17..32 two-phase branches, the M>8 guards and the `RowTiles=2` launch.
- This undoes the commit 3 edits to the GEMM and the macro.

### Commit 5: block-scaled M9..32 (rewritten)
In `fp8_qpn8_gemm_sm70_out` (CHANGED):
- **Removed guard:** `TORCH_CHECK(m <= 8 || channel_scales, "M=9..32 requires channel scales")`. The fork served M>8 only for channel scales; block-FP8 checkpoints carry block scales `[K/128, N/32]`.
- **M17..32 guard:** "split_k 12/16, fast decoder, 2 chains, no prefetch" now applies only when `channel_scales`.
- **Two-phase branch:** `if (m > 16)` became `if (m > 16 && channel_scales)`. The pair branch already required `channel_scales`.
- **`VLLM_LAUNCH_QPN8` (CHANGED):**
  - The `RowTiles=2` launch passes the real `channel_scales` flag instead of a hard-coded `true`. `fp8_qpn8_sm70_kernel` loads the block scale per (K/128 group, 32-column tile) the same way for any row-tile count, so block scales work at M9..16.
  - The macro reads `chunk_in` / `chunk_out` / `chunk_m`.
- **Chunk loop:** the generic dispatch runs in a loop over 16-row chunks. It is one pass for M≤16, or for channel scales, which return earlier above 16. Block-scaled M17..32 makes two passes, each re-reading the codes.
- **Guards:** the remaining guards are the fork's and shape-generic: N%32, block K%128 (channel K%16), `codes.numel() == N*K`, split_k ∈ {4, 8, 12, 16, 32}, `(K/16) % split_k == 0`, M>8 ⇒ split_k ≤ 16, chains ∈ {1, 2}, prefetch ⇒ fast decoder, split 12 ⇒ 2 chains + fast decoder + no prefetch. None names a model or TP shape.
- **Shape-dependent `if` heuristics:** the pair path needs N≥2048, K≥1536; the packed two-phase path needs N≥2048 and K≥4096 or K≥1536. Both are channel-only and never route block-scaled weights.
- **Split-K at TP2:** the Python split-K table (`QPN8_CONFIGS`) holds only TP4 shapes. All 12 TP2 Qwen3.6/3.8 dense shapes use the generic split 16 (8 if K%256), which is always ≤16, so M up to 32 is admitted.

Everything else is unchanged:
- `fp8x8_to_half2x4` and `_fast`, the lane/column/physical-K helpers, `fp8_qpn8_prepack_sm70_kernel`, both scale kernels, `fp8_qpn8_dequantize_sm70_kernel`, `VLLM_SM70_MMA_8N8K4`.
- `fp8_qpn8_sm70_kernel`, including its now-unused `FusedBA` / `SplitOutputs` / `RowTiles=2` template branches, kept so the instantiated M≤8 code is byte-identical.
- `launch_fp8_qpn8_sm70`, `fp8_qpn8_prepare_sm70`, `fp8_qpn8_dequantize_sm70_out`.
