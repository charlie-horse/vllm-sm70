# Flash-V100 attention: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`, directory
`flash-attention-v100/` (D. Skryabin's Flash-V100). License: BSD-3-Clause,
`LICENSE` in this directory. Line counts are `wc -l`; "fork last-touch" is
`git log -1 --format=%h -- <fork path>`. Check with
`sm70/tools/check_flash_v100_provenance.py [--fork <checkout at cf9492a>]`.

| status | meaning |
|---|---|
| verbatim | byte-identical to the fork file / definition |
| trimmed | deletions only (a line subsequence of the fork text) |
| adapted | trimmed plus the edits listed under *Changes from the source* |
| new | no fork counterpart; the fork path is the file it replaces |

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---|---|---|---|---|
| `sm70/csrc/flash_v100/LICENSE` | `flash-attention-v100/LICENSE` | verbatim | 28 | 28 | ee0bc6a87 | `f5bb48b59e45ad3f1481627d8906448b456ebed0d31a874cdf2a12f1e3a8854f` | `f5bb48b59e45ad3f1481627d8906448b456ebed0d31a874cdf2a12f1e3a8854f` |
| `sm70/csrc/flash_v100/flash_v100_traits.cuh` | `flash-attention-v100/kernel/flash_v100_traits.cuh` | verbatim | 118 | 118 | 58beb3433 | `a2fcf0385362a5546f475479efcac49f49a4d7636dc57c12edd52fbbeae38d5c` | `a2fcf0385362a5546f475479efcac49f49a4d7636dc57c12edd52fbbeae38d5c` |
| `sm70/csrc/flash_v100/fp8_kv_utils.cuh` | `flash-attention-v100/kernel/fp8_kv_utils.cuh` | verbatim | 117 | 117 | ca0ea462c | `f885710cd191d858cd532fa95582ee39476376dd6be079fe2ee414e906a803ab` | `f885710cd191d858cd532fa95582ee39476376dd6be079fe2ee414e906a803ab` |
| `sm70/csrc/flash_v100/fused_mma.h` | `flash-attention-v100/include/fused_mma.h` | verbatim | 442 | 442 | 8235544d4 | `6de64878a322cc93cbb98ef5b8a435450c273cca41dbcf5a378627c2a0560201` | `6de64878a322cc93cbb98ef5b8a435450c273cca41dbcf5a378627c2a0560201` |
| `sm70/csrc/flash_v100/fused_mha.h` | `flash-attention-v100/include/fused_mha.h` | trimmed | 79 | 220 | 38f49201f | `e1ef557dbfe10354e12c77c7e018cbe3426c1742b0c43c2a9de7d3f5fff386a4` | `5557b25373de90406de39d6d404b76a79fd2f60de95c40574d80de8fc7f9e220` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash-attention-v100/kernel/flash_decode_paged.cu` | adapted | 3246 | 7382 | 38f49201f | `82822fbcaf655ff64484252ba569ae499838f5ab690e9b743ac4f879b2f2f91b` | `ede98fc9d28bea2812b4a871fa3ca3e0b12cd2ae783b4a76e95c8aeac0144cbe` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash-attention-v100/kernel/fused_mha_forward_paged.cu` | adapted | 3063 | 4228 | 5f9c273a8 | `5fe710fc568913b552e383f8e8d7a1fa86b4f70988333b0a5ee3e71cbbd902ab` | `12d5b8a5118a47b040fb1963040e5db8c1805b0145fda67733c867460e06b630` |
| `sm70/csrc/flash_v100/fv100_config.h` | `flash-attention-v100/include/flash_v100_policy.h` | new | 46 | 466 | c8efae47b | `ee5d6e3f539b8832110d1e9e46160f71d9151fef263a6ce072be3deab114d17b` | `1fabfd62dfe0284695af025fc1cd0478451a7d123703f6035befd57fa98231a0` |
| `sm70/csrc/flash_v100/fv100_ops.cpp` | `flash-attention-v100/kernel/fused_mha_api.cpp` | new | 266 | 166 | 38f49201f | `701e4dfb4ced2f387ecbd28ff8166bc5382ee25855a56705b04d17f5e67ce3cb` | `a83649ace10b975ca1a15ef0c93878ba4243c8a507f930793a31edf97ad9b165` |
| `sm70/csrc/flash_v100/sources.cmake` | `flash-attention-v100/setup.py` | new | 51 | 143 | 8235544d4 | `ad1a0f195f737bcd4b5abf356cd0aa262dbec42c6b44587a27530b7cc9ac0b71` | `b926a26015b31bb8a6f1d135b8a1cbb2013f14a32a2965511a6f2cbc047a59a7` |
| `sm70/csrc/flash_v100/verify_decode.cu` | - | new | 540 | - | - | `440cd55dbce6cc2792a8c760dbe7952f15e885340d584364545c750f13bde54a` | - |

## Changes from the source

- `flash_decode_paged.cu`, `fused_mha_forward_paged.cu`: only the fp16-KV
  decode, XQA (D256, GQA 4/6/8), grouped GQA-8 D128 sink decode and paged
  prefill (D128/D256) paths are kept. Deleted: FP8 routes, anchored-SWA
  kernels, split-KV prefill, staged/dual-CTA/page-specialised XQA routes,
  DFlash2/E4M3 verifier kernels, debug kernels, env-policy probes and the
  host entries of all of these. `#include "flash_v100_policy.h"` became
  `#include "fv100_config.h"`.
- `flash_decode_paged.cu`: the fused sink-decode kernel, its launch branch
  and `flash_attention_decode_paged_sinks_fused` sit under
  `#ifdef VLLM_SM70_FV100_FUSED_DECODE`. `decode_paged_sinks_impl` and
  `flash_attention_decode_paged_xqa` refuse anchored windows and non-fp16
  XQA KV with a `TORCH_CHECK`.
- `fused_mha_forward_paged.cu`, `prefill_paged_forward`: refuses anchored
  windows with a `TORCH_CHECK`.
- `fused_mha.h`: declarations of the removed entry points deleted.
- `fv100_config.h` replaces `include/flash_v100_policy.h`: the six policy
  fields the kept code reads, as `constexpr` values equal to the fork's
  defaults with the variable unset. No environment is read.
- `fv100_ops.cpp` replaces the pybind module `kernel/fused_mha_api.cpp`:
  `TORCH_LIBRARY_FRAGMENT(_sm70_C)` out ops `fv100_*` that call the fork
  entry points (anchored arguments `std::nullopt, 0`).
- `sources.cmake` replaces the `setup.py` `CUDAExtension`, with its nvcc
  flags.
- `verify_decode.cu`: new code, no fork counterpart.
- Comments edited outside verbatim definitions.
- Not imported: dense/backward kernels, TurboQuant and FP8 bridge kernels,
  paged/contiguous converters, `h3/`, the Python package (rewritten as
  `vllm_sm70/ops/fv100.py`).

## Definitions

Top-level definitions of the kernel sources, matched to the fork by name
and occurrence. `lines` excludes the comment block above a definition;
`sha256` is over those lines joined with `\n`.

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
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `kGqaFusedMaxChunk` | 2075-2077 | 3988-3990 | verbatim | `6c02a92677f4f8808e4ece70873a77903075f6f0c4bf64b6749f4720d33130c9` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_gqa8_d128_fused_kernel` | 2079-2437 | 3992-4350 | verbatim | `29f9f877aec0fb79089b356da32cb98ef1ebc0cdc0ca3b2adbe8c585da2779b0` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `gqa8_d128_fused_ctas_per_head` | 2443-2463 | 4356-4376 | verbatim | `b78defc80ef3ba8a1a3c23a16011ef57b72dbfae6eec9bfc8681d89536a2e5f2` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `gqa8_d128_decode_aligned` | 2467-2477 | 4379-4389 | verbatim | `7a135e12d6714f570f919194b273f3a157ca36017c6d5fae660a99e8c9aad491` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `gqa8_d128_decode_eligible` | 2479-2496 | 4391-4408 | verbatim | `2b38d6fa4b58e5775be6f20061c1d011e96fae29093590f0567be4f30db65cd3` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `launch_flash_attention_decode_paged` | 2498-2637 | 4410-4566 | adapted | `7e4d275676f35010bd2e9c15bb8f0c3c6dc3c1ef3a472f2408f637e6a67403f3` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `launch_flash_attention_decode_xqa_split_reduce` | 2639-2692 | 4568-4621 | verbatim | `066667ff7a2ffc5ac36004661d106009fb442b43fa22e2b56829a9dd51d0097b` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `launch_flash_attention_decode_paged_xqa_tc_256_wide` | 2694-2807 | 4623-4751 | trimmed | `3695bedac76b901d29cf7471abae65784e45a89a354bf396eb473ed5fcb437c2` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `decode_paged_sinks_impl` | 2811-3033 | 5878-6126 | adapted | `4792869dcff0fe2529bf291e309024f85907e1c21593964059ff6fac2dcf7a7e` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged_sinks` | 3035-3050 | 6128-6143 | verbatim | `f77b41627fcf367f9f71c060d8e1bfa2e5f4bc8a1162f94e9851fcc241728702` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged_sinks_fused` | 3053-3069 | 6145-6161 | verbatim | `58bfce6720f0558e3c87c0ee6046d5b244381b31f7223b068e559719a3215a3d` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged` | 3073-3088 | 6164-6179 | verbatim | `8f6ee10b962968b5a730f7c17e416c51941b0bfd5c7feef7b9c6517c5ef5438c` |
| `sm70/csrc/flash_v100/flash_decode_paged.cu` | `flash_attention_decode_paged_xqa` | 3090-3246 | 6181-7162 | adapted | `193a8eb743e8c3e26010d467e0f9ee257ae977b239d5c607ee54cb15b03c0199` |
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
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `launcher_flash_attention_forward_paged` | 2796-2902 | 3190-3424 | trimmed | `3a3ea295b6f1c1c141679157b508eaa232324b0c97510d7ac7742df101705a34` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `prefill_paged_forward` | 2904-3033 | 3575-3716 | adapted | `d6347f4c5334b472ee3bccfacc7d9340e6427caa01477866a3f974f94d66b104` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_prefill_paged` | 3035-3047 | 3718-3730 | verbatim | `c2ff9c56c670982c3b0dc9a9727e82220a3668262d4c644bd046b0c20393e85c` |
| `sm70/csrc/flash_v100/fused_mha_forward_paged.cu` | `flash_attention_prefill_paged_sinks` | 3051-3063 | 3734-3746 | verbatim | `420fea52ad1b20e81bc9d1d82428fa130257f7a60c72a4e99ec72c1390a0c4f1` |
