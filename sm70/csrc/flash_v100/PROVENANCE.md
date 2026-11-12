# Flash-V100 attention, lean subset: provenance (patches K05, K14)

Source: charlie-horse/vLLM at **cf9492a01**, directory `flash-attention-v100/`
(D. Skryabin's Flash-V100, BSD-3-Clause, `LICENSE`). Scope: Tier B section B
(Ivy) of the SM70 carry map. Line counts are `wc -l`; "fork last-touch" is
`git log -1 --format=%h -- <fork path>`. Re-verify with:

```
sm70/tools/check_flash_v100_provenance.py --fork <fork checkout at cf9492a>
```

Commit order on `kernels/k05-flash-v100`: (1) verbatim import of the seven
fork files, (2) the carve (deletions only, plus removing the `} else `
prefix of the last XQA dispatch branch), (3) the adaptations below, the new
files, the build hook and the tests.

## Status values
| status | meaning |
|---|---|
| verbatim | byte-identical to the fork file / definition |
| trimmed | deletions only (a line subsequence of the fork text) |
| adapted | trimmed plus the edits listed under *Adaptations* |
| new | no fork counterpart; the fork path is the file it replaces |

## What the lean subset keeps

| path | kept instantiations |
|---|---|
| decode, generic paged (`flash_attention_decode_partition_kernel` + `flash_attention_decode_reduce_kernel`) | D {128, 256} x partition {256, 512, 1024} x fp16 KV, not anchored; sinks and sliding window are runtime arguments |
| decode, Flint grouped two-launch (`flash_attention_decode_gqa8_d128_{partition,reduce}_kernel`) | D128, GQA 8, fp16, with sinks, partition {256, 512, 1024} |
| decode, one-launch fused sink decode (`flash_attention_decode_gqa8_d128_fused_kernel<256>`) | K14: D128, GQA 8, fp16 KV, with sinks, not anchored; under `#ifdef VLLM_SM70_FV100_FUSED_DECODE`, which `sources.cmake` now defines by default (see *K14* below) |
| decode, Qwen tensor-core XQA (`flash_attention_decode_xqa_tc_partition_kernel_256_wide` + reducer) | D256 x GQA {4, 6, 8} x partition {256, 512, 1024} x padded smem {G6/P256 only, otherwise off}, fp16 KV, generic page size |
| prefill, generic paged (`flash_attention_forward_kernel_paged`) | D {128, 256} x causal {0, 1} x fp16; the only path with sinks; window is runtime |
| prefill, D256 low-SMEM | `<LOW_SMEM, CONTIG_FAST>` for page 16, `<LOW_SMEM>` otherwise, `<LOW_SMEM, OUTPUT_STRIDE_268, SW_PIPELINE_QK, SW_PIPELINE_PV>` for page 784 (the fork defaults) |
| prefill, D256 BM32 phase | causal {0, 1} x ALL_P x PAIR_SCRATCH (the fork defaults) |

KV cache views: the kernels take explicit block/token/head strides, so the
LBNHC layout of upstream v0.33.0 (K and V interleaved per head, stride(2) =
2*D) works without copies. `vllm_sm70/ops/fv100.py:check_kv_cache_views`
checks it on the CPU.

## Deleted regions (fork line numbers)

`flash-attention-v100/kernel/flash_decode_paged.cu` (7382 -> 3254 lines):

| fork lines | what |
|---|---|
| 51-52 | G6 dual-CTA / 8-warp pipeline thread counts |
| 138-147 | staged-PV shared layout |
| 153-531 | 66 env-policy probe and trace helpers, batch-context routing |
| 1812-2784 | grouped DFlash2 / E4M3 verifier kernels |
| 2893-3361 | staged XQA (G6) kernels |
| 3479-3551 | `qk_scores` debug kernel |
| 4513-4514 | anchored-SWA partition launch |
| 4518-4535 | FP8 scalar decode branch (incl. TP2 E4M3 fast) |
| 4697-4710, 4712 | FP8 branches of the XQA launcher |
| 4753-5294 | E4M3 aligned-page dispatch, staged launcher, `qk_scores` launcher, grouped sparse page4 kernels |
| 5297-5877 | host entries: grouped sparse page4 (+plan), E4M3 FP32 grouped, version probes, grouped verify |
| 6057-6073 | FP8 cases of the sink-decode dispatch |
| 6098-6109 | head dims 64/80/96/112 of the sink-decode dispatch |
| 6282-6614 | E4M3 XQA host routes |
| 6619-6841, 6877-7148 | page-specialised XQA routes (dual CTA, p1024 sawtooth/auto, block 4/16/784, E5M2, MTP5, QSA page4) |
| 7163-7382 | staged XQA host entry, `qk_scores` host entry |

`flash-attention-v100/kernel/fused_mha_forward_paged.cu` (4228 -> 3065 lines):

| fork lines | what |
|---|---|
| 1848-1876 | anchored-SWA launch in `launcher_flash_attention_forward_paged_impl` |
| 1921-2109 | split-KV merge kernel and launcher |
| 2117 | unused page-size constant |
| 2962-2981, 3015-3134 | split-KV3 BM32 kernels and launcher |
| 3141-3147, 3149-3150, 3155-3166, 3170-3171, 3176-3187 | BM32 phase non-default (ALL_P / pair-scratch off) branches |
| 3202-3220 | anchored branch of `launcher_flash_attention_forward_paged` |
| 3261-3264, 3275-3288 | knob reads of the removed D256 variants |
| 3290-3317, 3325-3326, 3328-3355, 3357-3358, 3367-3393, 3403-3404 | D256 low-SMEM variants behind non-default knobs (scalar QK, BM32, contig-fast off, partial SW pipeline) |
| 3426-3574 | split-KV prefill host entry |
| 3681-3686 | FP8 cases of the prefill dispatch |
| 3693-3701 | head dims 16/32/64 |
| 3747-4228 | BM32 pair-scratch / split-KV3 hosts, BFLA prefill, WMMA decode, DFlash2 BMHD |

`flash-attention-v100/include/fused_mha.h`: declarations of every removed
entry point (fork lines 10-20, 72-141, 160-219).

Not imported at all: `fused_mha_forward.cu` (dense `fwd`), `fused_mha_backward.cu`,
`flash_decode_turboquant.cu`, `fp8_kv_bridge.cu`, `paged_to_contiguous*.cu`,
`contiguous_to_paged.cu`, `paged_kv_utils.cuh`, `h3/`, the Python package
(re-written as `vllm_sm70/ops/fv100.py`), and the external FA2 stack
(`cmake/patches/sm70_flash_attn_d256_*`, `csrc/attention/sm70_{v37,79t,grouped_long}`).

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---|---|---|---|---|
| `sm70/csrc/flash_v100/LICENSE` | `flash-attention-v100/LICENSE` | verbatim | 28 | 28 | ee0bc6a87 | `f5bb48b59e45ad3f1481627d8906448b456ebed0d31a874cdf2a12f1e3a8854f` | `f5bb48b59e45ad3f1481627d8906448b456ebed0d31a874cdf2a12f1e3a8854f` |
| `sm70/csrc/flash_v100/flash_v100_traits.cuh` | `flash-attention-v100/kernel/flash_v100_traits.cuh` | verbatim | 118 | 118 | 58beb3433 | `a2fcf0385362a5546f475479efcac49f49a4d7636dc57c12edd52fbbeae38d5c` | `a2fcf0385362a5546f475479efcac49f49a4d7636dc57c12edd52fbbeae38d5c` |
| `sm70/csrc/flash_v100/fp8_kv_utils.cuh` | `flash-attention-v100/kernel/fp8_kv_utils.cuh` | verbatim | 117 | 117 | ca0ea462c | `f885710cd191d858cd532fa95582ee39476376dd6be079fe2ee414e906a803ab` | `f885710cd191d858cd532fa95582ee39476376dd6be079fe2ee414e906a803ab` |
| `sm70/csrc/flash_v100/fused_mma.h` | `flash-attention-v100/include/fused_mma.h` | verbatim | 442 | 442 | 8235544d4 | `6de64878a322cc93cbb98ef5b8a435450c273cca41dbcf5a378627c2a0560201` | `6de64878a322cc93cbb98ef5b8a435450c273cca41dbcf5a378627c2a0560201` |
| `sm70/csrc/flash_v100/fused_mha.h` | `flash-attention-v100/include/fused_mha.h` | trimmed | 79 | 220 | 38f49201f | `e1ef557dbfe10354e12c77c7e018cbe3426c1742b0c43c2a9de7d3f5fff386a4` | `5557b25373de90406de39d6d404b76a79fd2f60de95c40574d80de8fc7f9e220` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash-attention-v100/kernel/flash_decode_paged.cu` | adapted | 3254 | 7382 | 38f49201f | `eddf0f86e370664b0487423261b8c48136f7b21f539c4295a7b09721241e6af7` | `ede98fc9d28bea2812b4a871fa3ca3e0b12cd2ae783b4a76e95c8aeac0144cbe` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash-attention-v100/kernel/fused_mha_forward_paged.cu` | adapted | 3065 | 4228 | 5f9c273a8 | `8393eff871d5120e7c9f64e97f97235084163913e4798a5d8fb6dc549c3949dd` | `12d5b8a5118a47b040fb1963040e5db8c1805b0145fda67733c867460e06b630` |
| `sm70/csrc/flash_v100/fv100_config.h` | `flash-attention-v100/include/flash_v100_policy.h` | new | 56 | 466 | c8efae47b | `76ee201719ee9646c4b3e057a6be3912842de8220a1ae2ace349cf44669f0788` | `1fabfd62dfe0284695af025fc1cd0478451a7d123703f6035befd57fa98231a0` |
| `sm70/csrc/flash_v100/fv100_ops.cpp` | `flash-attention-v100/kernel/fused_mha_api.cpp` | new | 286 | 166 | 38f49201f | `c3b69f7777f2f61720188480aef5f6bc92b25b60716233fc796237c23a6abda2` | `a83649ace10b975ca1a15ef0c93878ba4243c8a507f930793a31edf97ad9b165` |
| `sm70/csrc/flash_v100/sources.cmake` | `flash-attention-v100/setup.py` | new | 64 | 143 | 8235544d4 | `ae5dcd7e2a34b911a54a5dd0ab819f6cdd3591bb00bb51ab4e52950a1c10c00c` | `b926a26015b31bb8a6f1d135b8a1cbb2013f14a32a2965511a6f2cbc047a59a7` |
| `sm70/csrc/flash_v100/verify_decode.cu` | - | new | 546 | - | - | `02be03d5e4d7c720720c0c8a2e6a5628bac36328d3780e5d6131bc732c5ef290` | - |

## Adaptations (commit 3)

- `flash_decode_paged.cu`, `fused_mha_forward_paged.cu`: line 1
  `#include "flash_v100_policy.h"` -> `#include "fv100_config.h"`.
- `fv100_config.h` (new, replaces `include/flash_v100_policy.h`): the six
  `flash_v100::policy::Field`s the kept code reads, as `constexpr value()`
  returning the fork's unset-variable defaults. No environment is read; the
  fork's `PreparedPolicy`, `Scope`, `with_policy` and `observation()` are gone.
- `flash_decode_paged.cu`:
  - `#ifdef VLLM_SM70_FV100_FUSED_DECODE` ... `#endif` around the fused
    kernel and `gqa8_d128_fused_ctas_per_head` (fork 3967-4377), around the
    fused branch of `launch_flash_attention_decode_paged` (fork 4467-4493;
    its `} else if (` is split into `} else` / `#endif` / `if (`), and
    around `flash_attention_decode_paged_sinks_fused` (fork 6145-6162).
  - `decode_paged_sinks_impl`: `TORCH_CHECK(!use_anchored, ...)` after
    `use_anchored` (the anchored kernels are not compiled).
  - `flash_attention_decode_paged_xqa`: `TORCH_CHECK(kv_dtype_code ==
    KV_CACHE_DTYPE_FP16, ...)` (only fp16-KV XQA is compiled).
- `fused_mha_forward_paged.cu`, `prefill_paged_forward`:
  `TORCH_CHECK(!use_anchored, ...)`.
- `fv100_ops.cpp` (new, replaces the pybind module `kernel/fused_mha_api.cpp`):
  five `TORCH_LIBRARY_FRAGMENT(_sm70_C)` out ops, `fv100_{decode_paged,
  decode_paged_sinks, decode_paged_xqa, prefill_paged, prefill_paged_sinks}`,
  with CUDA impls that call the fork entry points (`out` passed as the
  optional output, anchored arguments `std::nullopt, 0`).
- `sources.cmake` (new, replaces `setup.py`'s `CUDAExtension`): the two
  kernel files and `fv100_ops.cpp`, with the fork's nvcc flags.

## K14 (C4): one-launch sink decode and the counter fix

Branch `kernels/k14-fused-sink-decode`. The kernel came over verbatim in K05
(rows `kGqaFusedMaxChunk`, `flash_attention_decode_gqa8_d128_fused_kernel`,
`gqa8_d128_fused_ctas_per_head`, `flash_attention_decode_paged_sinks_fused`
and the fused branch of `launch_flash_attention_decode_paged` below; fork
`38f49201f`), compiled out. K14 changes no `.cu` line; it builds it and wires
it up. Fork commits carried: `38f49201f` (fused kernel), `7f12d21b8` (never
create counters under capture, two-launch fallback, prepare hook,
`_ZERO_COUNTERS`), `89f751847` (per-device counters, capture report),
`7e1c8c089` (quiet kill switch, cheap hook), `7b2bacefd` (INFO once, DEBUG
as totals grow).

| new location | fork cf9492a | status |
|---|---|---|
| `sources.cmake`: `option(VLLM_SM70_FV100_FUSED_DECODE ... ON)` | `flash-attention-v100/setup.py` (always built) | new; `-DVLLM_SM70_FV100_FUSED_DECODE=OFF` restores the K05 set |
| `fv100_ops.cpp`: `decode_paged_sinks_fused` + `fv100_decode_paged_sinks_fused` def/impl, all under the `#ifdef` | `kernel/fused_mha_api.cpp:29-30` `decode_paged_sinks_fused_fwd` | new (out op; `Tensor sinks`, `Tensor(e!) counters`; no anchored args) |
| `vllm_sm70/ops/fv100.py`: `FUSED_DECODE_COUNTER_CAPACITY`, `_FusedDecodeCounters`, counter cache, capture counts | `flash_attn_v100/flash_attn_interface.py:30-36, 59-71` | ported |
| `fv100.py`: `fused_decode_enabled`, `_fused_decode_zero_counters_enabled` | `flash_attn_v100/legacy_policy.py:42-47`, `flash_attn_interface.py:141-150` | ported; `_runtime` policy dropped, env read per call |
| `fv100.py`: `_FUSED_DECODE_OFF*`, `_warn_fused_decode_once`, `_normalized_device` | `flash_attn_interface.py:153-185` | ported |
| `fv100.py`: `_fused_decode_counters` | `flash_attn_interface.py:198-233` | ported; `_fused_decode_counter_cache_for(_runtime)` (:188-195) dropped |
| `fv100.py`: `fused_decode_available` | `flash_attn_interface.py:236-240` | ported (`has_op` instead of `hasattr(native, ...)`) |
| `fv100.py`: `flash_attn_decode_paged_sinks_prepare` | `flash_attn_interface.py:243-259` | ported |
| `fv100.py`: `flash_attn_decode_paged_sinks_capture_counts` | `flash_attn_interface.py:272-275` | ported; `_fused_decode_capture_counts_for(_runtime)` (:262-269) dropped |
| `fv100.py`: `fused_decode_eligible` + fused branch of `flash_attn_decode_paged` | `flash_attn_interface.py:1366-1391, 1428-1433` | ported; `e4m3_fp32` and anchored terms dropped (not in the lean build) |
| `vllm_sm70/attention/flash_v100/dispatch.py`: `_SINK_CAPTURE_*`, `_log_sink_capture_summary` | `vllm/v1/attention/backends/flash_v100/impl.py:49-70` | ported |
| `dispatch.py`: `fused_decode_hook` | `impl.py:894-920` (`_prepare_fused_sink_decode`) + `:942-948` | ported onto the K06 hook point |
| `dispatch.py`: `init_fused_decode` (called from `FlashAttnV100Impl.__init__`) | none | new (R4, below) |
| `vllm_sm70/envs.py`: `VLLM_SM70_FLINT_ATTN_FUSED_DECODE` (kill switch), `..._ZERO_COUNTERS` (`opt_in`, only "1" acts) | `vllm/envs.py:689-690, 10514-10535` | fork names and defaults |

Dropped: the `_runtime`/`PythonPolicy` plumbing (`sink_fused_decode`,
`sink_fused_zero_counters` fields, `config/flash_v100.py:869-872, 1313-1316`),
the `*_configured` twin of the fused op, and `ops.py:get_flash_decode_sinks_prepare_ops`
(the impl calls `fv100` directly).

**R4 (capture-time state).** The fork prepared the counters on a sink layer's
first eager forward, counting on the profile run. In v0.33.0 the profile run
has no attention metadata (the K06 impl returns before its hook), so the
first eager attention forward would be the warmup of the largest capture
size. K14 therefore also prepares at layer construction
(`init_fused_decode`), on the current CUDA device: the counters exist before
any capture, and no zero-dependent state is ever created under capture (a
capture that finds none, or too few, takes the two-launch kernels and warns
once). The kernel's last CTA re-zeroes its counter, so graphs of every batch
size share one per-device buffer; `retired` keeps outgrown buffers alive.

**Admitted shape set (2xV100).** `fused_decode_eligible` (Python) and
`gqa8_d128_decode_eligible` + `gqa8_d128_decode_aligned(out)` (C++):
head dim 128, `H_q == 8 * H_kv`, sinks present (fp32 `[H_q]`), fp16 KV
(`auto`/`float16`), not anchored, q/K/V/out/tmp_out 16-byte aligned with
strides % 8. Flint per rank: TP1 32q/4kv, **TP2 16q/2kv**, TP4 8q/1kv; any
`H_kv` works (grid `(B, H_kv, Z)`, counters `B * H_kv`). LBNHC K/V views
(stride(2) = 2 * D) pass. Everything else (GQA 4/6, D256, no sinks, FP8)
takes the two-launch or XQA kernels. `Z = min(ceil(SMs * occupancy / (B *
H_kv)), tmp_out.size(2), 256)`; it never depends on sequence lengths, so
captured grids replay at any length.

## Definitions (fork function -> new file)

Top-level definitions of the kernel sources, matched to the fork by name
and occurrence. `lines` excludes the comment block above a definition;
`sha256` is over those lines joined with `\n`. Rows marked `verbatim` are
byte-identical to the fork's `fork lines`, `trimmed` rows are deletions
only, `adapted` rows are listed under *Adaptations* above.

| new file | definition | lines | fork lines | status | sha256 |
|---|---|---|---|---|---|
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `kv_cache_dtype_code_from_string` | 25-37 | 25-37 | verbatim | `88f1f41a73196138823fe41161403c6bad0fc83cfea0dc17c1ead571a3fe6c9b` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `kWarpSize` | 39-68 | 39-70 | trimmed | `0bc0f2c848b1e598b38bef91fa7dddc51339edccbcd1bb5f9c5636c79acc2a7b` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `XQATCSmem256WideLayout` | 70-98 | 72-100 | verbatim | `ed29da626896e0a368477287673e4f5b0748c9a76c5bdf445edd108f5da125dd` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `XQATCQKPipelineSmem256WideLayout` | 100-134 | 102-136 | verbatim | `b44dee9a7681d5a1ccc4f40c54e8ea6a7d55f9f29381cdb364fa3d3cf4a96a1d` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `xqa_padded_smem_enabled` | 136-139 | 148-151 | verbatim | `09ee02e717927fe467357487a71bfbaad7601ae29862c62f8640664bf4391a95` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `xqa_seq_len_route_active` | 141-165 | 532-556 | verbatim | `295b8ae1d842e2ac461594c565093633f1b7dabfe466c82c9c87e92c243f5230` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `xqa_sawtooth_partition_size` | 167-174 | 558-565 | verbatim | `07d82ffc8c269d4ca5dcd1f08aff9fd5ffc992b88a31c13ace4d08470786f2ca` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `warp_reduce_sum` | 176-182 | 567-573 | verbatim | `35b830f39f828268b020e918973855ee61b93d131d3add5a87b33aa849720735` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `warp_reduce_max` | 184-190 | 575-581 | verbatim | `fa10acf8e1f61fe179dc4f1e61f5620c65a8faf4dec634c35f604a54c7785848` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `block_reduce_sum` | 192-214 | 583-605 | verbatim | `583ff863f03d0ede76d5da44257bb95567ed7531cd9429f8227455d7b3b9571a` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `block_reduce_max` | 216-238 | 607-629 | verbatim | `2542104896e2a72c10eb44f190ac71cebe69cb060cacb40761447bbc007202a7` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e5m2_pair_to_half2_bits` | 240-244 | 631-635 | verbatim | `2d3c1cd8d73b2712456c21044dc9112c22a49980759d545adbd4922bcae0a953` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e5m2_vector_to_half8` | 246-252 | 637-643 | verbatim | `9cae020e38b1094cf2952420077fa192b09a1c8fc34572879fe71cde98704b5c` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e4m3fn_to_half_bits` | 254-277 | 645-668 | verbatim | `90341db8049dfbc1c5b4e61bb6d9fa9d880d90356b10a518ae3a5d5eb739b8e0` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e4m3fn_pair_to_half2_bits` | 279-286 | 670-677 | verbatim | `92f2fa5cba8326018d090979ddea8261e3577f97c61bf47ee5e9503e43d067de` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e4m3fn_pair_to_half2_bits_fast` | 288-310 | 679-701 | verbatim | `116477f250469cb733e8c7df24e680909d6c2ea032e0148375565d3af22b6fc6` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e4m3fn_vector_to_half8` | 312-319 | 703-710 | verbatim | `99a78ea325671f203cd1efd4f537644ccf18236443c7ffe5db0758cddd455fae` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e4m3fn_vector_to_half8_fast` | 321-328 | 712-719 | verbatim | `9842163ea523191c71aeeac7c35681bbdaa005f61023eed01d644e99f7c49cff` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `fp8_e4m3fn_vector_to_half8_lut` | 330-341 | 721-732 | verbatim | `da7b4048b3cac2e1b88865238602234e2f40ccd55acce710a905fad594f3c94b` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `load_xqa_tc_kv_vector` | 343-431 | 734-822 | verbatim | `00dfe634038ed7733e8f87cc00c84dba0b24cb9db1519398c72cd67c424136d7` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `load_xqa_tc_kv_panel` | 433-532 | 824-923 | verbatim | `8491bf60b22e0a922fe731c9fc126a70d930bbfe27a696aa37888139a22b0ed3` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `load_xqa_tc_kv_panel_and_zero` | 534-582 | 925-973 | verbatim | `a2e31b684bb2997899854dd404ddfa4873f34a8192fd9dd6ade8ceecea4fff26` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `dot_qk_half2` | 584-601 | 975-992 | verbatim | `6546bb91f5e84b456a45b0af0f5b6ebe651743bce7a4a8c9db62ddaec9892033` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `dot_qk_cache` | 603-638 | 994-1029 | verbatim | `41dca8eccfde417615221625e7a60278335fd7752fe99ba6c078a839b861eb35` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_partition_kernel` | 640-858 | 1031-1249 | verbatim | `9c1b0258011e3ae3f58fcc14b2335c8b90c7a285044c482e2b187f5e5ae168f7` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_xqa_tc_partition_kernel_256_wide` | 860-1420 | 1251-1811 | verbatim | `bdadbd82ff45aa293b564509ee70c2dbbe1d9fdc2dba9477e27dbea0799412ab` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_reduce_kernel` | 1421-1527 | 2785-2891 | verbatim | `b64d4d10d28d35ad46fc3ede83ab7498b9edefcf2c13ee36c9620482d58214c0` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_xqa_reduce_stats_kernel` | 1529-1584 | 3362-3417 | verbatim | `371eaa617a1cd56b7e834d4416997f89f71eb94c523e7f79a30e5307686728b0` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_xqa_reduce_output_kernel` | 1586-1644 | 3419-3477 | verbatim | `7f501758ed4dc6c9f3771164d63464c03fa6d61d541dcdec67f393507f9829b2` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `kGqaDecodeGroup` | 1647-1648 | 3553-3554 | verbatim | `dc5e6b6c2b89f6ab4d4943c7e5190254d35277e42eaf10daca20e75daec99f15` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `warp_allreduce_sum` | 1650-1656 | 3556-3562 | verbatim | `669de0ce154b9d1de1fb90765daaab3ef39596c8ce5790cf4883add5428e6845` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `warp_allreduce_max` | 1658-1664 | 3564-3570 | verbatim | `672653500287bfc40b18ef9fc68546dd173a565ffb82f9277161484950db5ee5` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `load_half4_as_float` | 1666-1675 | 3572-3581 | verbatim | `3ee875d7c63c6347adc654c302d548537d05aa352ceb51f36dd4d2f521d2a571` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `load_half8_as_float` | 1677-1688 | 3583-3594 | verbatim | `10a0cd8d9e7d5082fd923338120ed4c79d89cbca51e3d09113f5c8643c70b2f2` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_gqa8_d128_partition_kernel` | 1703-1949 | 3609-3855 | verbatim | `81f24f9796b0afc30dda250d65305c3c22b82341edf33aa4b8753c7469188ca4` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_gqa8_d128_reduce_kernel` | 1957-2059 | 3863-3965 | verbatim | `d734ed6fbe6513bf02387600cf12e02d84c46fe04483390d1643f5e3bbdfef00` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `kGqaFusedMaxChunk` | 2083-2085 | 3988-3990 | verbatim | `6c02a92677f4f8808e4ece70873a77903075f6f0c4bf64b6749f4720d33130c9` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_gqa8_d128_fused_kernel` | 2087-2445 | 3992-4350 | verbatim | `29f9f877aec0fb79089b356da32cb98ef1ebc0cdc0ca3b2adbe8c585da2779b0` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `gqa8_d128_fused_ctas_per_head` | 2451-2471 | 4356-4376 | verbatim | `b78defc80ef3ba8a1a3c23a16011ef57b72dbfae6eec9bfc8681d89536a2e5f2` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `gqa8_d128_decode_aligned` | 2475-2485 | 4379-4389 | verbatim | `7a135e12d6714f570f919194b273f3a157ca36017c6d5fae660a99e8c9aad491` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `gqa8_d128_decode_eligible` | 2487-2504 | 4391-4408 | verbatim | `2b38d6fa4b58e5775be6f20061c1d011e96fae29093590f0567be4f30db65cd3` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `launch_flash_attention_decode_paged` | 2506-2645 | 4410-4566 | adapted | `7e4d275676f35010bd2e9c15bb8f0c3c6dc3c1ef3a472f2408f637e6a67403f3` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `launch_flash_attention_decode_xqa_split_reduce` | 2647-2700 | 4568-4621 | verbatim | `066667ff7a2ffc5ac36004661d106009fb442b43fa22e2b56829a9dd51d0097b` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `launch_flash_attention_decode_paged_xqa_tc_256_wide` | 2702-2815 | 4623-4751 | trimmed | `3695bedac76b901d29cf7471abae65784e45a89a354bf396eb473ed5fcb437c2` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `decode_paged_sinks_impl` | 2819-3041 | 5878-6126 | adapted | `18d5b7ea2d4f5096b6338eb7b50f3cc5f1562351a2713a3ee1d5e1c695ef31e5` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged_sinks` | 3043-3058 | 6128-6143 | verbatim | `f77b41627fcf367f9f71c060d8e1bfa2e5f4bc8a1162f94e9851fcc241728702` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged_sinks_fused` | 3061-3077 | 6145-6161 | verbatim | `58bfce6720f0558e3c87c0ee6046d5b244381b31f7223b068e559719a3215a3d` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged` | 3081-3096 | 6164-6179 | verbatim | `8f6ee10b962968b5a730f7c17e416c51941b0bfd5c7feef7b9c6517c5ef5438c` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged_xqa` | 3098-3254 | 6181-7162 | adapted | `66a7455964b2f5fe64a041bc59b4fad6b2b39484a85ad3cf30ee385162061b56` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `using namespace nvcuda::wmma;` | 19-19 | 19-19 | verbatim | `3761f13bbff96f2d5d0e3dfdeb62778ea918b7186f9df805cd66de980155e59c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `kv_cache_dtype_code_from_string` | 40-51 | 40-51 | verbatim | `253b83ce8a1309cf517960ffda0072eadcaff4737522fa60eb3eabfe2f035fcd` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `KernelConfig` | 83-173 | 83-173 | verbatim | `0575f1562e5473bed5c06bb79b65972f3a147ad240a5476486756198f9344ccd` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `init_smem` | 175-189 | 175-189 | verbatim | `a615b46f9fc99d097216a923a3e6fd3392d4399bfd1a13cd81f91e1c21aabb97` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `pipeline_swizzled_q_row_slot` | 191-193 | 191-193 | verbatim | `d400ae2aa13882b76439b52e5e63d6c7ce9a00a5061edb4cd6d3b7f9e23b636c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `pipeline_swizzled_matrix_a_offset` | 195-203 | 195-203 | verbatim | `5139224a831277082186f04a365ce7e39915b77b4d96b238b91497e7da78d453` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `load_pipeline_swizzled_matrix_a_fragment` | 205-224 | 205-224 | verbatim | `fa8079c74edb5286c8e2346695379d9651798c742ff499d1f699e20954fef6c8` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `pipeline_qk_slice` | 226-313 | 226-313 | verbatim | `a282d68564072878177448ea1cc947476d318fbd8cf77bd89d6bc77f537a2a88` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `pipeline_pv_tile` | 315-350 | 315-350 | verbatim | `45833340f3b6f4f82d5203319efde0bd86c262315d33d51d941c66c8c847ea0c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_forward_kernel_paged` | 352-1805 | 352-1805 | verbatim | `46b79c4698b420c1db7670a7769bd2ab41fdd94def66d6cb613a8f816f14503f` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `launcher_flash_attention_forward_paged_impl` | 1807-1890 | 1807-1919 | trimmed | `83dc2eda0e206e9b2812ea371a7e380028f21c96f6d1efb693bb024d645d3255` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `D256_BM32_PHASE_BLOCK_M` | 1892-1905 | 2110-2124 | trimmed | `6d15de648d6afa1f1e6d04b68dcc6685c6c6444fc21bfbbb4546f576658a9ba3` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `D256BM32PhaseSharedStorage` | 1907-1926 | 2126-2145 | verbatim | `1ee3657f19d7092e1d0d23646b2e9d6540c092924449c71865eb272ed6cd0075` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `D256BM32PhaseSplitSharedStorage` | 1928-1932 | 2147-2151 | verbatim | `628029adddd23b0a832dd29821420e7eda8c7e5f4614e006f12438a8988dd9b7` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `D256BM32PhaseSharedStorageSelector` | 1934-1937 | 2153-2156 | verbatim | `3138fd25003f008418a00e204b0bb2232d1e6616574dce19f1042fed346ea7ff` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `D256BM32PhaseSharedStorageSelector` | 1939-1942 | 2158-2161 | verbatim | `dcf5ad038a4d9cf20ce30ff22ba401ef4f098adc8bc6565431a14ab3c589a73e` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `using D256BM32PhaseSharedStorageSingle =` | 1944-1946 | 2163-2165 | verbatim | `557b01742ee3326662531272a2298347bf85938b4f7a9af60bcab7228be41d40` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_swizzled_row_slot` | 1948-1969 | 2167-2188 | verbatim | `eee9a5d670870ef8d05f983f5dada76eb2544d1ca2bec035b1e9641a2a61ad21` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_matrix_a_offset` | 1971-1979 | 2190-2198 | verbatim | `65c72515b67ea03c54e4bcae503232d4957d76ed5f72b1a7c125426ecfa91eae` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_stage_query` | 1981-2006 | 2200-2225 | verbatim | `e4d44934e7d51ef9e368b845c9e18305d3ea2d3fa01c9b84fd5964bb54ff1d4c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_stage_query_partial` | 2008-2037 | 2227-2256 | verbatim | `dda467c6ab4e548b6626fb1aa57e849205e0b722b4d12af5b97165f47977851b` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_load_matrix_a` | 2039-2058 | 2258-2277 | verbatim | `505d3b72bcd86094265da404724b68faa2aa6987f38aaf568a50f82868322bbd` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_accumulator_row` | 2060-2065 | 2279-2284 | verbatim | `67a640ffc9133c88b84bdea43d03a4568f2208f581be8956a1ec0ace60983b3d` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_accumulator_column` | 2067-2071 | 2286-2290 | verbatim | `513f4ceae9051774d3d7ae88a222f89a32aec5a3aa13fe45a68fc01228696f71` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_spill_pair_scratch` | 2073-2105 | 2292-2324 | verbatim | `99889c621a347f74833cf8edfd8cc879d7ff995140902bf59385a98dee421c58` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_reload_pair_scratch` | 2107-2141 | 2326-2360 | verbatim | `754fe4e7f726c29f32bbff95e0ca45b36df33353eefa82bb19be3c412ba83f97` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_sync_warp_pair` | 2143-2146 | 2362-2365 | verbatim | `031fb573ab04be2d7be5a8d99a8438ec379d893095d4737adeb7393104f96634` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_make_probability_row` | 2148-2196 | 2367-2415 | verbatim | `096d75db972877fa0f11f280d4272848126d550a787a520a38d946e79d76f74c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_make_split_probability_row` | 2198-2222 | 2417-2441 | verbatim | `952f111a515cb9c16bd897182bab196f80cf07944021896edfb638c882bb3450` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_scale_accumulator_rows` | 2224-2235 | 2443-2454 | verbatim | `bc3c6291e2461d36c3902dbb83065a29b29d6025f915da337fe82dc78a94a6d2` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_scale_accumulators` | 2237-2247 | 2456-2466 | verbatim | `08314b538c1eafbc8c66bcacbd9c14b0752492bdf1895d17d2b6dcb828a1c71a` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_update_pv_panel` | 2249-2271 | 2468-2490 | verbatim | `bf9cd85ba70bee080316db8bb97dd44002e7195a742b3a37c546e051c1e8679f` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_store_output` | 2273-2287 | 2492-2506 | verbatim | `83b35a7445296c293c6ad7f19e6907585529cf476a741e06e2a34f5babf588c1` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_store_output_partial` | 2289-2305 | 2508-2524 | verbatim | `745f10f843a4878fca0ecee32e70b570c18415b7a64a78b33dd130a0d83b2371` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `d256_bm32_phase_store_unnormalized_partial` | 2307-2320 | 2526-2539 | verbatim | `494eaf7830c08e1091e3fb0e43cec124bd735ce3271166ef0cc1866174213966` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_forward_paged_d256_bm32_phase_body` | 2322-2723 | 2541-2942 | verbatim | `772acde34052ac513eb85d432e2cb2f5ddba9d2249074417f1d1650e38e0a0e1` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_forward_paged_d256_bm32_phase_kernel` | 2725-2741 | 2944-2960 | verbatim | `e8dd85dc4ed4904e84bf8131e3144eadf19ce2e5d3473203004803dea8cb577c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `launch_flash_attention_forward_paged_d256_bm32_phase_kernel` | 2743-2774 | 2982-3013 | verbatim | `17f26751957c3eaacc16382e19f333986ec46bab87e1c431e6bd75b00abf02ae` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `launcher_flash_attention_forward_paged_d256_bm32_phase` | 2776-2794 | 3135-3188 | trimmed | `4397d01de46247908a77ad3723ae60c0f4b5985a31bb67559297b247132c1467` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `launcher_flash_attention_forward_paged` | 2796-2904 | 3190-3424 | trimmed | `4f761b95c465740bd5338516bcb8aa60a37d806add6023f5d4f6c8ce5f995133` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `prefill_paged_forward` | 2906-3035 | 3575-3716 | adapted | `55eb927277e48b5e790635c2f4c608c175a5880e705cf2b7fdb1bb52b79643d9` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_prefill_paged` | 3037-3049 | 3718-3730 | verbatim | `c2ff9c56c670982c3b0dc9a9727e82220a3668262d4c644bd046b0c20393e85c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_prefill_paged_sinks` | 3053-3065 | 3734-3746 | verbatim | `420fea52ad1b20e81bc9d1d82428fa130257f7a60c72a4e99ec72c1390a0c4f1` |
