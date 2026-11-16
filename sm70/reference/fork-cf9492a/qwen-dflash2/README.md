# Fork reference: Qwen DFlash2 verifier bundle and Flash-V100 Qwen XQA variants

**Not ported yet: port when Qwen3.8 is activated (Phase 3 speed bar).**

This directory keeps source from the retired fork charlie-horse/vLLM, commit
`cf9492a01` (the port base), so that a later port does not need the fork. The
code is Qwen-only. Flint, the active model, does not use it. vllm-sm70 left it
out on purpose: parity-matrix rows Q9, Q10 and Q13 are marked **D** (dropped),
and gap G6 is "Qwen DFlash2 verifier bundle".

## Inert by construction

Nothing here is built, imported, tested or linted:

- **Build.** `sm70/CMakeLists.txt` lists its sources explicitly
  (`csrc/*/sources.cmake`), with no globbing. `nix/sm70-ext.nix` removes
  `sm70/reference` from the extension's source fileset, so changes here do
  not rebuild anything. setuptools packages only `vllm_sm70*`.
- **Import.** The tree is outside `vllm_sm70/`, and the `vllm/` paths below are
  the fork's layout. Nothing puts them on `sys.path`.
- **pytest.** `sm70/pyproject.toml` has `testpaths = ["tests"]` and lists
  `reference` in `norecursedirs`. CI runs `pytest sm70/tests tests`. No file
  here is named `test_*.py`.
- **ruff.** `sm70/pyproject.toml` lists `reference` in `[tool.ruff] extend-exclude`.
- **Patch series.** `patches/` applies to upstream vLLM only and never reads
  `sm70/`.

Files are byte-identical copies of the fork at `cf9492a01`, at the fork's
relative paths. The two `*.excerpt` files are the exception: they are verbatim
line ranges, and each block is headed by its original path and line numbers.
`flash_decode_paged.cu` (342 KiB) is copied whole, because it is under the
1 MB limit for whole-file copies. The Qwen ranges are listed below. Everything
was checked for secrets, tailnet addresses (100.x) and `ts.net` names, and none
were found.

## Switches (fork defaults)

The `VLLM_SM70_DFLASH2_*` getters all default to `0`. They became deprecated
aliases of `speculative_config.sm70_dflash2.<field>`. The fork turned them on
for the qualified Qwen3.8 verifier contract through
`SM70_DFLASH2_VERIFIER_DEFAULTS` in `vllm/config/sm70_dflash2.py:20-39`, which
sets each one to `1`. So the switches are effectively **on for Qwen3.8 DFlash2**
and off everywhere else. Full definitions are in `vllm/envs.py.excerpt`.

| Row | Switch (`VLLM_SM70_DFLASH2_…`) | Default (env / Qwen3.8 contract) | Where it acts (files here) |
|---|---|---|---|
| Q9 | `FUSED_GDN_VERIFY` | 0 / 1 | `csrc/sm70_turbomind/ops/gdn_verify_sm70.cu` (`sm70_gdn_verify_out`): the GDN target-verify recurrent update (sigmoid gating + gated delta rule) in one CUDA kernel, replacing the Triton `fused_sigmoid_gating_delta_rule_update` on the spec-verify route. Called from `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`. Op registration: `csrc/bindings.excerpt` |
| Q9 | `FUSED_GDN_METADATA` | 0 / 1 | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`: builds all GDN state-index metadata for the verifier with one pointer-table Triton launch. The fork's `vllm/v1/attention/backends/gdn_attn.py` also reads it (not copied: 1 reference in a 97 KB file) |
| Q9 | `FUSED_GDN_NORM`, `FUSED_GDN_SPLIT`, `FUSED_GDN_COMBINED_SPLIT` | 0 / 1 | `qwen_gdn_linear_attn.py`: fused gated norm and fused qkvz/ba split (combined variant) on the verify path. `vllm/model_executor/models/qwen3_5.py` also wires the split flags (not copied) |
| Q9 | `FUSED_GEMMA_RMS`, `FIXED_GEMMA_RMS` | 0 / 1 | `vllm/model_executor/kernels/norm/sm70.py`: fused or fixed-order Gemma RMSNorm (Triton) for verifier rows. It also binds `sm70_rmsnorm_gated_exact_out` from `csrc/sm70_turbomind/ops/rmsnorm_gated_exact_sm70.cu`, an exact-order gated RMSNorm (also used from the fork's `layernorm.py`, not copied) |
| Q9 | `CONTEXT_PIPELINE`, `CONTEXT_KV_GRAPH` | 0 / 1 | `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`: pipelines the drafter's context (target hidden-state) projection with verification, and captures its context-KV write in a CUDA graph |
| Q9 | `SPARSE_TARGET_REJECTION` | 0 / 1 | `speculator.py` + `vllm/v1/worker/gpu/spec_decode/dflash2/sparse_rejection.py`: strictly gated compact (sparse) target rejection sampler. The fork's GLM contract sets it to 0 |
| Q9 | `QUANT_LM_HEAD`, `SHARDED_CONTEXT_FC` | 0 / 1 | `vllm/model_executor/models/qwen3_dflash2.py`: quantized draft LM head, and a TP-sharded context FC (TP=4 and (25600,5120) or (20480,4096) only; the fork's TP2 never admits it). vllm-sm70's `tm_config.h` registers the `SHARDED_CONTEXT_FC` name but nothing consumes it |
| Q9 | `VERIFY_FASTPATH` | 0 / 1 | Umbrella gate for the selector-based verifier fast paths (`mamba_hybrid.py`, `sm70_dflash2.py`) |
| Q9 | none (shape-gated) | always built | `csrc/sm70_turbomind/ops/qwen38_gdn_input_batch_sm70.cu`: packed checkpoint-FP16 GDN input, router, shared-expert and dense batch kernels for M = 2..16 decode (TP4). Consumer: `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` |
| Q9 | none (separate extension `_sm70_exact_reduce_C`) | always built | `csrc/sm70_turbomind/ops/exact_row_reduce.cu`: a deterministic peer-to-peer row all-reduce with fixed order (`--fmad=false`, IPC handles). Wrapped by `vllm/model_executor/layers/sm70_collectives.py`. At cf9492a its only caller is the fork's MiniMax-H3 collectives, so the parity matrix's Q9 grouping is kept here but no DFlash2 caller was found |
| Q10 | none; needs `FUSED_GDN_VERIFY` + `VLLM_SM70_FP8_QPN8` + layer `enable_sm70_gdn_ba_verify` | on when admitted | `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu`, ops `fp8_qpn8_gemm_ba_split_sm70_out` / `fp8_qpn8_dispatch_ba_split_sm70_out` (lines ~181-205, 660, 1641-1760, registration ~2207-2214): one launch for the qkvz QPN8 channel-FP8 GEMM plus the FP16 `ba` projection, writing qkv/z/b/a split. Python gate: `vllm/model_executor/layers/quantization/sm70_gdn_ba_verify.py` (only x = (8, 5120) fp16). vllm-sm70's `sm70/csrc/qpn8/fp8_qpn8_sm70.cu` has no ba_split. The fork route never runs on the block-FP8 checkpoint (paused K09b/M9b) |

Q13: Flash-V100 Qwen XQA variants, `flash-attention-v100/kernel/flash_decode_paged.cu`:

- Lines **2893-3361**: the split-reduce XQA kernels. A stats kernel and a
  cross-partition reducer keep the original max/sum tree and the per-dimension
  ascending-partition FMA order. Switches: `VLLM_FLASH_V100_XQA_SPLIT_REDUCE`
  (default 0), `VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE` (1),
  `VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE` (native default 8).
- Lines **6619-7148**: XQA routing for GQA group 6 (G6). Covers the G6
  dual-CTA (`XQA_G6_DUAL_CTA` 0, `XQA_E5M2_G6_DUAL_CTA` 1,
  `XQA_E4M3_G6_DUAL_CTA_BEGIN` 32768) and staged/QK-pipeline XQA
  (`XQA_G6_QK_PIPELINE` 1, warps 8). Also the p1024 sawtooth
  (`XQA_G6_P1024_SAWTOOTH` 1, `XQA_G6_P1024_AUTO` native 1) and
  page-specialised layouts: block-784 index (`XQA_BLOCK784_INDEX` native 1),
  block-16 (`XQA_BLOCK16_LAYOUT` native 0) and the 4-page pair load
  (`XQA_E5M2_PAIR_LOAD` 1). Then MTP5 (`XQA_MTP5_DUAL_CTA`,
  `XQA_MTP5_PARTITION_SIZE`, unset) and the E4M3/E5M2 KV variants
  (`E4M3_BATCH_XQA` 1, `XQA_E5M2_*`). All switches are
  `VLLM_FLASH_V100_*`. Each one's definition and defaults are in
  `vllm/envs.py.excerpt`.
- vllm-sm70's `sm70/csrc/flash_v100/flash_decode_paged.cu` is the lean carve
  (K05) without these. The parity matrix never checked which of them the fork
  selected by default for the Qwen3.8 TP2 GQA6 shapes. Audit that first.

`vllm/config/sm70_dflash2.py` is the fork's per-engine verifier policy (field
names, legacy env aliases, contract defaults). The main-only follow-ups A1/A2/B1/B2
(rows Q6-Q8) were ported to vllm-sm70 `main` (not on `latest` yet) and are not repeated here.

## License

- The fork is a fork of 1CatAI/1Cat-vLLM, itself derived from vLLM. Files with
  `SPDX-License-Identifier: Apache-2.0` (every `.py`, and the `csrc/` `.cu`
  files) are Apache-2.0, Copyright contributors to the vLLM project. The fork's
  top-level `LICENSE` is copied unchanged as `LICENSE`.
- `flash-attention-v100/kernel/flash_decode_paged.cu` has no SPDX header. It
  belongs to the fork's `flash-attention-v100/` tree, which is BSD 3-Clause,
  Copyright (c) 2025 D.Skryabin. That tree's `LICENSE` is copied unchanged as
  `flash-attention-v100/LICENSE`. The fork's `AUTHORS` file there is empty and
  is not copied.
- `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu`: the QPN8 layout is derived from
  dnv2003/v100-skinny (MIT) and haohervchb/sglang-V100. The retained MIT notice
  is copied unchanged as `csrc/sm70_turbomind/ops/LICENSE.v100-skinny`, the same
  file as `sm70/csrc/qpn8/LICENSE.v100-skinny`.
- A port must carry these notices along with the code, as `sm70/csrc/gdn/NOTICE`
  and `sm70/csrc/flash_v100/LICENSE` do, and record each change in a
  PROVENANCE.md.
