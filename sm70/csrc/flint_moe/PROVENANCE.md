# Flint NVFP4 MoE M<=2 QPN decode: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`,
`csrc/sm70_turbomind/ops/nvfp4_moe_decode_qpn_sm70.cu`. License: Apache-2.0;
the quadpair-N layout is derived from dnv2003/v100-skinny (MIT,
`LICENSE.v100-skinny`). Line counts are `wc -l`; "fork last-touch" is
`git log -1 --format=%h -- <fork path>`. Check with
`sm70/tools/check_flint_moe_provenance.py [--fork <checkout at cf9492a>]`.
Status values as in `../flash_v100/PROVENANCE.md`.

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---:|---:|---|---|---|
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `csrc/sm70_turbomind/ops/nvfp4_moe_decode_qpn_sm70.cu` | adapted | 627 | 882 | 4f332bcba | `45878b699462550b83951b513778d35ffe2a9a0d4d7a8e597745e6485d441e67` | `699096dd8996285f5ad7c869e8da56fa175a99bfc258d98702d4bdad5df81dd5` |
| `sm70/csrc/flint_moe/LICENSE.v100-skinny` | `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` | verbatim | 21 | 21 | 58beb3433 | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` |
| `sm70/csrc/flint_moe/sources.cmake` | - | new | 27 | - | - | `9cfa73b4482464bf339192bfb376d1ba1cacaf6c894235b4d6cdec38efb587c5` | - |

## Definitions

Spans are in the new file and the fork file.

| new file | definition | lines | fork lines | status | sha256 |
|---|---|---|---|---|---|
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::dequant_e2m1x8` | 44-58 | 57-71 | verbatim | `c39d9dd99e14c8bf34254d7eaf012f9a4053545195fbe598a8a7e8d9d03ea24c` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::mma_m8n8k4` | 60-69 | 73-82 | verbatim | `72dc6100e3181b00cc43af59c1a0c9c7b41059de24ff2fc50903d0009454cea0` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::packed_column` | 71-73 | 84-86 | verbatim | `126e375ea38b7471370f28e5e4b90b6814053e6e25be1f0c3d780c00df46c4ab` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::TileFragment` | 77-81 | 90-94 | verbatim | `3238c3f3a80d322d0d83fdeedf5966f0de2e8162f442c0cea63e46c429b6dc15` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::load_tile` | 85-101 | 98-114 | verbatim | `5495a7e8e1d3eb2528c0085ceb68d8c5b5559928bd53a9f0b0cba61885370f89` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::is_input_lane` | 105-107 | 118-120 | verbatim | `d98e5e7be3282ee59e2d0c05a7eba6e5b7a810668dd201225b4c71d81a39cc81` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::InputFragment` | 110-113 | 123-126 | verbatim | `a4b43e97828e946ff3f6c50399a1881dadd1eff4fbdece7879ae03832456472b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::load_input` | 115-130 | 128-143 | verbatim | `08337368a5ac7928c4955ad88d11c9fa5ca01b18a286abf964b9eb800bbe1c21` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::mma_group` | 134-149 | 147-162 | verbatim | `20289120bb986dc0de2cf8455eb53437225f5cbd8ee767356ccd3d6a4ca387cf` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::mma_tile` | 154-169 | 167-182 | verbatim | `470e672ff9bea57661917c6e2297ebba20a8c9ae31ec3d1e49ccf1078463585d` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::store_row0` | 172-185 | 213-226 | verbatim | `f72662c173f8a41a175b2ed312c4a8db971b7efd2b5dfcab9c6ba92e485e5886` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::scale_multiplier` | 187-189 | 228-230 | verbatim | `4068cad5517400b19485ddcf3530f02e7bca243488f45c3f2c45b2237ea54ba9` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::swiglu_epilogue` | 193-200 | 234-241 | verbatim | `901e06be64048c78a1fbffe0de99e5090e4868d16d673de3b06894dcc98d6ca3` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w2_weighted_sum` | 205-222 | 246-263 | verbatim | `3cda128a5a7aff803a0a9411e330085872df6069b3b2f5ae61f50af5c2c8cbcc` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::task_sync` | 227-240 | 268-281 | verbatim | `d697d9764dc179456527064224c4800ab33ae16436e4c6756b2094f803427398` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::W13Shared` | 242-247 | 292-297 | verbatim | `4778836336ea1808693df901c78f5dc0d974d931787b53110b3fe3366d0c639b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w13_swiglu_task` | 256-353 | 306-403 | verbatim | `9c3ac216c7665431f5ab7ac8b5cdac57f26b428b027e9c1d5e2126f3fb1bda61` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w13_swiglu_kernel` | 356-372 | 406-422 | verbatim | `9f04c2a77141a97e7ba602b5268f66ef8b35a80840e48ddec3d0e3043ae8cdca` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w2_reduce_kernel` | 375-420 | 425-470 | verbatim | `7cdb01efc427047a5b5300b7bf19fe974e729f6976e7cc30eb13d6f743e45cc6` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintHidden` | 423-423 | 618-618 | verbatim | `b39deb063b31f143de7ec35638699f68d29a8767410bb45e1dca60e41d4eb4fd` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintIntermediate` | 424-424 | 619-619 | verbatim | `d4b16cbff3409a5b5abf8fde54b7b475be4543e132b290fc262a6af5452cd6e2` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintTopK` | 425-425 | 620-620 | verbatim | `ece155940224f864df78b4cad275b205418d70cdc05cee8030d955ad1cd18c04` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintExperts` | 426-426 | 621-621 | verbatim | `d6e64b0c853c1286188c186d13b7555ae813d6253324f1a920d68bc9021cc136` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintTiles` | 427-427 | 622-622 | verbatim | `c48fc716d9316cde9dbee4be396ce4de1efe23bed488d2dbb8202f9c13b6f5f1` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintMaxTokens` | 428-428 | 623-623 | verbatim | `58ae9c191ea23715fb3784728349c776446758ee49d3d3b28c89deda997b4e5e` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintW13SplitWarps` | 431-431 | 627-627 | verbatim | `b29a431ab0925c8a95ac7aca583eaea811363647e57767164667c67782efe283` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintW13SplitCta` | 432-432 | 628-628 | verbatim | `36051f2b70372b183bc9bcc46098badc5843a4fe8ec5b98f8eb0aa77af0fbe28` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintW2SplitWarps` | 433-433 | 629-629 | verbatim | `fc73e88a1ecd98c673991db37b7cf2263835b3d3cec1a2be114b9e9668dbdbde` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintCounters` | 437-437 | 633-633 | verbatim | `aa22ca99df4f12d70cbc89f90e4bdd24c31de7ebb47bfff5dff6e9f34a8e9109` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintPartials` | 438-438 | 635-635 | verbatim | `b5f74cc66f831b232a7cfdd78c40754e1b23664363ad9d8f450775027220687d` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::check_bank` | 442-472 | 655-682 | adapted | `010edf5ade06cf4428769666ce366c2fb38b70c409661712c2d6130d4c327858` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::check_w13` | 475-506 | 685-716 | verbatim | `29c5706b2043476a83c10c073a9db0915da47593ce744b2ffb3c515361ec9f66` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::check_w2` | 508-533 | 718-743 | verbatim | `d259dc7f3a10c9f2c8870c984ae9644b44a532b049f51228f7bd9e481cbf9c35` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::launch_w13` | 535-553 | 745-763 | verbatim | `696d350c1d1162a9caf4df3fdb8509092ad5bccb0a4893209853f1d7d7c0a8c2` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::launch_w2` | 555-572 | 765-782 | verbatim | `5f2fef60047592156defe2773f4d43a86a874c43ab1add1d2e3d3a935dbc0a66` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `vllm_sm70::nvfp4_moe_decode_w13_swiglu_sm70_out` | 579-591 | 808-820 | verbatim | `b6f84a04f0ebbfe7f30e5dfc2b31ab9960468ce7054f1b71ed890444e2ced63b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `vllm_sm70::nvfp4_moe_decode_w2_reduce_sm70_out` | 593-606 | 822-835 | verbatim | `b9907b19a96477e51dc3279995cc50caeae31f6058555c01291f8fdba52df135` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `TORCH_LIBRARY_FRAGMENT(_sm70_C,m)` | 611-620 | - | new | `986efae4dd9b76970e8de756d1c7b4bcd5b030e4143ce8c583b198a7e305ec6d` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `def:nvfp4_moe_decode_w13_swiglu_sm70_out` | 612-615 | - | new | `6402fa7f4836f767b44a5bc6a12429765ce9b9ff192dee2fb61ded5d67c314f9` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `def:nvfp4_moe_decode_w2_reduce_sm70_out` | 616-619 | - | new | `e84eb01752fe99f76b1e351ddec708c38602c5ba2a697f537c9f42a04f9250ed` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `TORCH_LIBRARY_IMPL(_sm70_C,CUDA,m)` | 622-627 | - | new | `133120e2cb5b9fa4ecfb0abd9b7a0962818aeba22b5895feda28ab7dea97eea4` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `impl:nvfp4_moe_decode_w13_swiglu_sm70_out` | 623-624 | - | new | `19d7d73c3dc90ea5d0bb96b11f47cb449e72b65f80ba079bd48c895dbe38ea6b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `impl:nvfp4_moe_decode_w2_reduce_sm70_out` | 625-626 | - | new | `f39c574d1cde4f64014d747fbf643d80059cb49e59b7e6d61cbdb885a89d7e64` |

## Changes from the source

- Dropped: the single-launch path (`fused_decode_kernel`,
  `mma_tile_shared`, `load_acquire_gpu`, `kFlintSyncCounters`,
  `kFlintFused*`, `kFlintW13TasksPerBlock`, `kFlintW2Tiles`,
  `FLINT_FUSED_DECODE_KERNEL`, `fused_resident_blocks`,
  `nvfp4_moe_decode_fused_sm70_out` and its binding). The counters
  workspace is 96 int32.
- Includes: `<atomic>` dropped; `<torch/library.h>` and `<cstdint>` added.
- `check_bank` (adapted): `input` must be 16-byte aligned.
- Host ops: same bodies, in `namespace vllm_sm70`.
- Registration (new): `TORCH_LIBRARY_FRAGMENT(_sm70_C)` with the fork
  schemas minus `str? native_policy`; CUDA impls only.
- Comments edited outside verbatim definitions.
- `sources.cmake` (new) replaces the fork's `CMakeLists.txt` entry; its `-U`
  flags match upstream's torch nvcc flags.
- Python side (`sm70/vllm_sm70/ops/flint_moe.py`): wrappers and fakes from
  fork `vllm/_sm70/moe.py`; contract, scale shift, admission and workspace
  from fork `quantization/nvfp4_sm70_moe.py`, without the persistent path.
