# Flint NVFP4 MoE M<=2 QPN decode: provenance (patch K12)

Source: charlie-horse/vLLM at **cf9492a01** (carry-map row K12, Flint fast path C2, fork kill switch `VLLM_SM70_FLINT_MOE_QPN_DECODE`, envs.py:354/5079). Line counts are `wc -l`; "fork last-touch" is `git log -1 --format=%h -- <fork path>`. Check this file with:

```
sm70/tools/check_flint_moe_provenance.py --fork <fork@cf9492a>
scripts/verify-port.py --port sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu \
  --fork <fork>/csrc/sm70_turbomind/ops/nvfp4_moe_decode_qpn_sm70.cu
scripts/verify-port.py --port sm70/vllm_sm70/ops/flint_moe.py --fork <fork>/vllm/_sm70/moe.py
```

At the K12 commit verify-port reports, for the .cu: IDENTICAL=34, CHANGED=1 (`check_bank`, below), MOVED=2 (the two host ops, into `namespace vllm_sm70`), NEW=6 (registration), DROPPED=11 (the single-launch path, below). For the Python wrappers and fakes against fork `vllm/_sm70/moe.py`: the four K12 entities are IDENTICAL.

Branch history (review each against the fork separately): a byte-identical import, deletions only (the file is a line subsequence of the fork), then this registration commit.

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---:|---:|---|---|---|
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `csrc/sm70_turbomind/ops/nvfp4_moe_decode_qpn_sm70.cu` | adapted | 643 | 882 | 4f332bcba | `83e23afc0adb47861782176a658c37292516616591bc5c3a4694b9948588f4d1` | `699096dd8996285f5ad7c869e8da56fa175a99bfc258d98702d4bdad5df81dd5` |
| `sm70/csrc/flint_moe/LICENSE.v100-skinny` | `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` | verbatim | 21 | 21 | 58beb3433 | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` |
| `sm70/csrc/flint_moe/sources.cmake` | - | new | 29 | - | - | `6e3c731d9a12a19ea504e07bd85a7ab030a2cce2d445abbc799076e21d85e272` | - |

`sources.cmake` replaces the fork's `CMakeLists.txt:410` entry in `VLLM_SM70_TURBOMIND_SRCS`. The TU includes no TurboMind header, so it is its own area. Flags: the fork built it with upstream's torch nvcc flags minus `-D__CUDA_NO_HALF*__` (fork `cmake/utils.cmake:131-137`); the `-U` flags here do the same. No `--use_fast_math` (the SwiGLU `expf` must match eager FP16 SiLU).

## Definitions

Spans are in the new file and the fork file. "adapted" means changed on purpose (listed below); "new" has no fork counterpart.

| new file | definition | lines | fork lines | status | sha256 |
|---|---|---|---|---|---|
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::dequant_e2m1x8` | 58-72 | 57-71 | verbatim | `c39d9dd99e14c8bf34254d7eaf012f9a4053545195fbe598a8a7e8d9d03ea24c` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::mma_m8n8k4` | 74-83 | 73-82 | verbatim | `72dc6100e3181b00cc43af59c1a0c9c7b41059de24ff2fc50903d0009454cea0` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::packed_column` | 85-87 | 84-86 | verbatim | `126e375ea38b7471370f28e5e4b90b6814053e6e25be1f0c3d780c00df46c4ab` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::TileFragment` | 91-95 | 90-94 | verbatim | `3238c3f3a80d322d0d83fdeedf5966f0de2e8162f442c0cea63e46c429b6dc15` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::load_tile` | 99-115 | 98-114 | verbatim | `5495a7e8e1d3eb2528c0085ceb68d8c5b5559928bd53a9f0b0cba61885370f89` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::is_input_lane` | 119-121 | 118-120 | verbatim | `d98e5e7be3282ee59e2d0c05a7eba6e5b7a810668dd201225b4c71d81a39cc81` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::InputFragment` | 124-127 | 123-126 | verbatim | `a4b43e97828e946ff3f6c50399a1881dadd1eff4fbdece7879ae03832456472b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::load_input` | 129-144 | 128-143 | verbatim | `08337368a5ac7928c4955ad88d11c9fa5ca01b18a286abf964b9eb800bbe1c21` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::mma_group` | 148-163 | 147-162 | verbatim | `20289120bb986dc0de2cf8455eb53437225f5cbd8ee767356ccd3d6a4ca387cf` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::mma_tile` | 168-183 | 167-182 | verbatim | `470e672ff9bea57661917c6e2297ebba20a8c9ae31ec3d1e49ccf1078463585d` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::store_row0` | 186-199 | 213-226 | verbatim | `f72662c173f8a41a175b2ed312c4a8db971b7efd2b5dfcab9c6ba92e485e5886` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::scale_multiplier` | 201-203 | 228-230 | verbatim | `4068cad5517400b19485ddcf3530f02e7bca243488f45c3f2c45b2237ea54ba9` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::swiglu_epilogue` | 207-214 | 234-241 | verbatim | `901e06be64048c78a1fbffe0de99e5090e4868d16d673de3b06894dcc98d6ca3` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w2_weighted_sum` | 219-236 | 246-263 | verbatim | `3cda128a5a7aff803a0a9411e330085872df6069b3b2f5ae61f50af5c2c8cbcc` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::task_sync` | 241-254 | 268-281 | verbatim | `d697d9764dc179456527064224c4800ab33ae16436e4c6756b2094f803427398` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::W13Shared` | 256-261 | 292-297 | verbatim | `4778836336ea1808693df901c78f5dc0d974d931787b53110b3fe3366d0c639b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w13_swiglu_task` | 270-367 | 306-403 | verbatim | `9c3ac216c7665431f5ab7ac8b5cdac57f26b428b027e9c1d5e2126f3fb1bda61` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w13_swiglu_kernel` | 370-386 | 406-422 | verbatim | `9f04c2a77141a97e7ba602b5268f66ef8b35a80840e48ddec3d0e3043ae8cdca` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::w2_reduce_kernel` | 389-434 | 425-470 | verbatim | `7cdb01efc427047a5b5300b7bf19fe974e729f6976e7cc30eb13d6f743e45cc6` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintHidden` | 437-437 | 618-618 | verbatim | `b39deb063b31f143de7ec35638699f68d29a8767410bb45e1dca60e41d4eb4fd` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintIntermediate` | 438-438 | 619-619 | verbatim | `d4b16cbff3409a5b5abf8fde54b7b475be4543e132b290fc262a6af5452cd6e2` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintTopK` | 439-439 | 620-620 | verbatim | `ece155940224f864df78b4cad275b205418d70cdc05cee8030d955ad1cd18c04` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintExperts` | 440-440 | 621-621 | verbatim | `d6e64b0c853c1286188c186d13b7555ae813d6253324f1a920d68bc9021cc136` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintTiles` | 441-441 | 622-622 | verbatim | `c48fc716d9316cde9dbee4be396ce4de1efe23bed488d2dbb8202f9c13b6f5f1` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintMaxTokens` | 442-442 | 623-623 | verbatim | `58ae9c191ea23715fb3784728349c776446758ee49d3d3b28c89deda997b4e5e` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintW13SplitWarps` | 446-446 | 627-627 | verbatim | `b29a431ab0925c8a95ac7aca583eaea811363647e57767164667c67782efe283` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintW13SplitCta` | 447-447 | 628-628 | verbatim | `36051f2b70372b183bc9bcc46098badc5843a4fe8ec5b98f8eb0aa77af0fbe28` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintW2SplitWarps` | 448-448 | 629-629 | verbatim | `fc73e88a1ecd98c673991db37b7cf2263835b3d3cec1a2be114b9e9668dbdbde` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintCounters` | 452-452 | 633-633 | verbatim | `aa22ca99df4f12d70cbc89f90e4bdd24c31de7ebb47bfff5dff6e9f34a8e9109` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::kFlintPartials` | 453-453 | 635-635 | verbatim | `b5f74cc66f831b232a7cfdd78c40754e1b23664363ad9d8f450775027220687d` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::check_bank` | 457-487 | 655-682 | adapted | `010edf5ade06cf4428769666ce366c2fb38b70c409661712c2d6130d4c327858` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::check_w13` | 490-521 | 685-716 | verbatim | `29c5706b2043476a83c10c073a9db0915da47593ce744b2ffb3c515361ec9f66` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::check_w2` | 523-548 | 718-743 | verbatim | `d259dc7f3a10c9f2c8870c984ae9644b44a532b049f51228f7bd9e481cbf9c35` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::launch_w13` | 550-568 | 745-763 | verbatim | `696d350c1d1162a9caf4df3fdb8509092ad5bccb0a4893209853f1d7d7c0a8c2` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `decode_qpn::launch_w2` | 570-587 | 765-782 | verbatim | `5f2fef60047592156defe2773f4d43a86a874c43ab1add1d2e3d3a935dbc0a66` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `vllm_sm70::nvfp4_moe_decode_w13_swiglu_sm70_out` | 594-606 | 808-820 | verbatim | `b6f84a04f0ebbfe7f30e5dfc2b31ab9960468ce7054f1b71ed890444e2ced63b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `vllm_sm70::nvfp4_moe_decode_w2_reduce_sm70_out` | 608-621 | 822-835 | verbatim | `b9907b19a96477e51dc3279995cc50caeae31f6058555c01291f8fdba52df135` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `TORCH_LIBRARY_FRAGMENT(_sm70_C,m)` | 627-636 | - | new | `986efae4dd9b76970e8de756d1c7b4bcd5b030e4143ce8c583b198a7e305ec6d` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `def:nvfp4_moe_decode_w13_swiglu_sm70_out` | 628-631 | - | new | `6402fa7f4836f767b44a5bc6a12429765ce9b9ff192dee2fb61ded5d67c314f9` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `def:nvfp4_moe_decode_w2_reduce_sm70_out` | 632-635 | - | new | `e84eb01752fe99f76b1e351ddec708c38602c5ba2a697f537c9f42a04f9250ed` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `TORCH_LIBRARY_IMPL(_sm70_C,CUDA,m)` | 638-643 | - | new | `133120e2cb5b9fa4ecfb0abd9b7a0962818aeba22b5895feda28ab7dea97eea4` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `impl:nvfp4_moe_decode_w13_swiglu_sm70_out` | 639-640 | - | new | `19d7d73c3dc90ea5d0bb96b11f47cb449e72b65f80ba079bd48c895dbe38ea6b` |
| `sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu` | `impl:nvfp4_moe_decode_w2_reduce_sm70_out` | 641-642 | - | new | `f39c574d1cde4f64014d747fbf643d80059cb49e59b7e6d61cbdb885a89d7e64` |

## What changed, function by function

- **Header comment:** the paragraph on the single-launch op (fork :27-32) is gone; a port note was added.
- **Includes:** `<atomic>` dropped (only `fused_resident_blocks` used it); `<torch/library.h>` and `<cstdint>` added.
- **`check_bank` (adapted):** one new guard, `input` must be 16-byte aligned. `load_input` reads each activation row with `uint4` loads; a misaligned `x` (a view at an odd offset) used to fault inside the kernel and now fails with a clear error. The host mirror is `flint_moe.call_rejection`.
- **Workspace comment above `kFlintCounters`:** it pointed at fork `nvfp4_sm70_moe.py` and the single-launch flags; it now points at `vllm_sm70/ops/flint_moe.py`. The values are unchanged (96 counters, 98304 partials).
- **Host ops:** byte-identical bodies, now in `namespace vllm_sm70` (they were global in the fork).
- **Registration (new):** `TORCH_LIBRARY_FRAGMENT(_sm70_C)` with the fork schemas (`torch_bindings.cpp:1463-1478`) minus `str? native_policy`; CUDA impls only. No fake is needed in C++; the Python fakes are in `flint_moe.py`.

Kept verbatim although dead in this build: the `kGrouped` branch of `task_sync` and the `ready` parameter of `w13_swiglu_task`. The two-kernel path always passes `kGrouped=false` and `ready=nullptr`. Leaving them keeps the W13 task byte-identical to the fork (the fork tested the two paths bitwise equal); removing them is a reviewable follow-up, not a port step.

## Dropped (fork lines) and why

- `fused_decode_kernel` (:472-615), `mma_tile_shared` (:184-210), `load_acquire_gpu` (:283-290): single-launch ("persistent", fork `VLLM_SM70_FLINT_MOE_PERSISTENT`, `4f332bcba`) only. The carry map drops it. It needed a device-wide wait with a forward-progress argument tied to the 80-SM occupancy and was default off in the fork.
- `kFlintSyncCounters` and its `static_assert` (:634, :638-639); `kFlintFusedThreads`, `kFlintFusedMinBlocks`, `kFlintW13TasksPerBlock`, `kFlintW2Tiles`, `FLINT_FUSED_DECODE_KERNEL` (:640-653); `fused_resident_blocks` (:784-803); `nvfp4_moe_decode_fused_sm70_out` (:837-882) and its binding (`torch_bindings.cpp:1480-1489`): same reason. The counters workspace shrinks from 96 + 32 to 96 int32.

## Python counterpart (`sm70/vllm_sm70/ops/flint_moe.py`)

| port | fork |
|---|---|
| `nvfp4_moe_decode_w13_swiglu_sm70_out`, `nvfp4_moe_decode_w2_reduce_sm70_out` (wrappers) | `vllm/_sm70/moe.py:345-389`, minus `native_policy`/`call_native` |
| `_nvfp4_moe_decode_*_fake` (registered only if `has_op`) | `vllm/_sm70/moe.py:450-481` |
| `FLINT_TP2`/`CONTRACTS`, `MAX_TOKENS`, `DECODE_PARTIALS`, `DECODE_COUNTERS` (96, not 128), `MAX_SCALE_SHIFT` | `quantization/nvfp4_sm70_moe.py:120-130` |
| `scale_shift` | `nvfp4_sm70_moe.py:180-195` (`_decode_qpn_scale_shift`), unchanged arithmetic |
| `contract_rejection`, `bank_rejection`, `admit_layer`, `DecodePlan` | `nvfp4_sm70_moe.py:218-290` (`_prepare_decode_qpn`), without the persistent admission (:293-310) or the "explicit request fails closed" branch (the env can only turn the decode off here) |
| `allocate_workspace`, `DecodeWorkspace` | the workspace half of `_prepare_decode_qpn` (:270-277), plus an int32 `ids` scratch for int64 router IDs, plus a refusal under graph capture (R4) |
| `call_rejection`, `can_decode` | `nvfp4_sm70_moe.py:198-215` (`_use_decode_qpn`), extended to the full C++ guard set (shape, dtype, contiguity, alignment) on metadata only |
| `flint_moe_decode_out` | `fused_moe/sm70/fp4_codec.py:158-167` + `:252-260` (the `DECODE_QPN` -> `DECODE_REDUCE` plan, `nvfp4_sm70_moe.py:1443-1450`) |

Dropped on the Python side: `Fp4MoECodec`/stage-plan indirection, `Sm70NvFp4MoEConfig` policy fields (`flint_qpn_decode`, `flint_persistent_decode`, `explicit_fields`), `decode_fused`, `_admit_decode_persistent`, and `has_nvfp4_moe_decode_{,fused_}dispatch`. `has_op` and `available` replace them.

## Admitted shape set (audit for Flint TP2 on 2x V100)

The kernels are compiled for one shape (`kFlint*`). The C++ guards (`check_w13`, `check_w2`) admit only this:

| quantity | admitted | Flint source |
|---|---|---|
| tokens M | 1 or 2 | decode steps with 1-2 tokens (M>2 stays on the dense stage) |
| hidden H | 2048 | Flint hidden_size |
| local intermediate I | 256 | moe_intermediate_size 512 / TP 2 |
| experts E | 256 (replicated, no EP) | n_routed_experts |
| top-k | 6 | num_experts_per_tok (sigmoid, renormalized) |
| W13 bank | int32 [256, 2048, 64] + FP16 [256, 128, 512], gate cols then up cols (not interleaved) | `tm.nvfp4_sm70_prepare(..., 16, False)` stacked, as M4 does |
| W2 bank | int32 [256, 256, 256] + FP16 [256, 16, 2048] | same |
| x / out | FP16 [M, 2048] contiguous; x 16-byte aligned | |
| intermediate | FP16 [6M, 256] contiguous | |
| expert IDs | int32 [6M] contiguous; out-of-range IDs contribute 0 | int64 router IDs are copied into `DecodeWorkspace.ids` |
| topk_weights | FP32 [M, 6] contiguous | |
| scale_shift | 0..14, chosen at load | |

The guards refuse Flint TP1 (I=512), TP4 (I=128), Qwen3.6 top-8, M=3 and misaligned input with `RuntimeError`, never with an out-of-bounds launch. This is tested on GPU (`test_guards_reject_other_shapes`, `test_guards_reject_misaligned_input_and_short_workspace`) and mirrored on CPU (`contract_rejection`, `bank_rejection`, `call_rejection`). This is the TP2 shape the fork tuned (V100-PCIE-32GB, CUDA graph). The split-K choice (W13 16 CTAs x 2 warps, W2 2 warps) needs no TP2 re-tune.

## CUDA graphs (R4)

- **Only state:** `partials` (scratch) and `counters` (zero between launches). The last CTA of each (route, tile) re-arms its own counter, so replays need no reset launch and no host work.
- **Allocation:** `allocate_workspace()` allocates and zeroes both at load and raises if called during capture. The module has no caches, and `flint_moe_decode_out` allocates nothing.
- **Sharing:** one workspace may serve several layers that run in sequence on one stream. It may not be shared across streams.
- **Regression tests:**
  - `test_descending_capture_regression` ports the C4 pattern (fork `test_decode_sinks_descending_capture_regression`): capture M=2 then M=1 on a poisoned shared pool with no eager call, replay M=1 first, compare bitwise against eager and against the FP32 oracle, and check the counters are zero after every replay.
  - `test_shared_workspace_two_layers_in_one_graph`.

## Callable contract for the M4 wiring (`vllm_sm70/quant/nvfp4_moe.py`)

M4 already detects the ops by name with `tm.has_op` (true once `_sm70_C` has K12, because the ops share the `_sm70_C` namespace). Its `decode_ops_available()` then requires `callable(getattr(tm, name))`. The wrappers live in `vllm_sm70.ops.flint_moe`, not in `tm`, so the fast path stays off until M4 switches to the following. Everything comes from `from vllm_sm70.ops import flint_moe`.

At load (`_prepare_decode`, after the banks are stacked):
```python
plan = flint_moe.admit_layer(
    (num_experts, hidden, intermediate, top_k),   # local, per TP rank
    layer.w13_tm_weight, layer.w13_tm_scales,
    layer.w2_tm_weight, layer.w2_tm_scales,
    policy_enabled=self.decode_enabled,           # Sm70Config.flint_moe_qpn_decode
)                                                 # -> DecodePlan | None
if plan is not None:
    layer.sm70_nvfp4_decode_plan = plan
    layer.sm70_nvfp4_decode_ws = flint_moe.allocate_workspace(device)  # never under capture
```
`admit_layer` covers the kill switch, op presence, the contract (Flint TP2 only), the bank shapes and dtypes, and the scale pre-shifts. It reads scales back to the host, so call it at load only.

In `apply`, before the dense route:
```python
if plan is not None and flint_moe.can_decode(x, topk_weights, topk_ids):
    out = torch.empty(m, 2048, dtype=x.dtype, device=x.device)  # fully overwritten
    return flint_moe.flint_moe_decode_out(
        out, buffers["intermediate"], x,
        layer.w13_tm_weight, layer.w13_tm_scales,
        layer.w2_tm_weight, layer.w2_tm_scales,
        topk_weights, topk_ids, layer.sm70_nvfp4_decode_ws, plan)
```
- `can_decode` is metadata only (no sync; safe under capture and torch.compile). It admits int32 or int64 `topk_ids` (int64 IDs are copied into the workspace, with no allocation), FP32 `topk_weights`, and FP16 `x` [M<=2, 2048] that is contiguous and 16-byte aligned.
- The return value is the local, un-reduced [M, 2048] sum, as the dense route returns it.
- Raw ops, if M4 prefers them: `flint_moe.nvfp4_moe_decode_w13_swiglu_sm70_out(out[6M,256], x, w13_w, w13_s, ids_i32[6M], partials, counters, w13_shift)`, then `flint_moe.nvfp4_moe_decode_w2_reduce_sm70_out(out[M,2048], act, w2_w, w2_s, ids_i32, topk_weights[M,6], w2_shift)`.

Differences from M4's current pre-wiring:
- **Counters:** 96 are needed (M4 allocates 128; still accepted).
- **int64 IDs:** M4's `_use_decode` refuses them and sends them to the dense route; `can_decode` accepts them.
- **Output buffer:** the output needs no `torch.zeros`.
- **Sharing the workspace:** one workspace can serve every layer on the device, since they run in sequence on one stream (saves 384 KiB per MoE layer). Per-layer workspaces are also correct.
