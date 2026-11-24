# DFlash2 draft FP16 M<=8 GEMM + compact FP16 top-64: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`. License: Apache-2.0; the
M8 layout follows the QPN8 kernels derived from dnv2003/v100-skinny (MIT,
`LICENSE.v100-skinny`). Line counts are `wc -l`; "fork last-touch" is
`git log -1 --format=%h -- <fork path>`.

## Files

| file | lines | status | sha256 | fork path | fork lines | fork sha256 | fork last-touch |
|---|---:|---|---|---|---:|---|---|
| `dflash2_fp16_m8.cu` | 185 | changed | `972b13cee463988f4690a9398902be3bf1bda848ba5e176c758d23d929e5681c` | `csrc/sm70_turbomind/ops/dflash2_fp16_m8_sm70.cu` | 146 | `bf6dc0a3198467b3ff207f3f93a2bc89849b270005dcf0a481cca9f09ad427c4` | f25139144 |
| `compact_half_topk.cu` | 97 | changed | `9a31f076e30e60deb5d6ba674e55cae6604b52d64a4fa988b5afbb5803e7d35c` | `csrc/sm70_turbomind/ops/compact_half_topk_sm70.cu` | 96 | `bbad9956baf6e63935205faaa450de5557705b09b17ed3654636c2bb642d46df` | fb52756f4 |
| `LICENSE.v100-skinny` | 21 | verbatim | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` | 21 | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | 58beb3433 |
| `sources.cmake` | 19 | new (replaces fork `CMakeLists.txt:407-408` in `VLLM_SM70_TURBOMIND_SRCS`) | `a53ae93f6578c2494955a9391f71350438c19c1714ad8f9bb8b3ea5d3b3b500b` | – | – | – | – |

Both `.cu` files register in `_sm70_C` (`TORCH_LIBRARY_FRAGMENT` +
`TORCH_LIBRARY_IMPL(_sm70_C, CUDA)`) instead of the fork's `_C`; schemas and
op names are unchanged. Comments edited.

## `dflash2_fp16_m8.cu`

| port | fork | status |
|---|---|---|
| `MMA` macro | :14-21 | identical |
| `dflash2_fp16_m8_kernel<Warps, Tile>` | :25-79 | identical |
| `aligned16` | – | **new**: 16-byte base-pointer check |
| `sm70_dflash2_fp16_m8_out` | :83-120 | changed: one added `TORCH_CHECK(aligned16(input) && aligned16(packed))`. The kernel loads input rows and packed weights as `uint4` |
| `sm70_dflash2_fp16_dispatch_out` | :122-146 | changed: (a) **M 1..7** runs the M8 kernel on a zero-padded 8-row copy and copies rows [0, M) back (fork: only M == 8, else `mm`). Rows are independent and the reduction order is fixed, so the result is bitwise the M8 kernel's rows. (b) a misaligned input or packed buffer takes `at::mm_out` instead of failing. (c) the `OptionalCUDAGuard` covers the padded path |

## `compact_half_topk.cu`

`select_compact_half_keys<Items, Merge>` (:13-50) and
`sm70_compact_half_topk_out` (:51-88) are identical; only the registration
namespace changed.

## Python (`vllm_sm70/ops/dflash2.py`)

Ported from fork `quantization/sm70_dflash2_fp16.py`,
`layers/sm70_compact_topk.py`, `models/qwen3_dflash2.py:280-298` and
`linear.py:343-411`. Changes: eager admission of M 1..8; repack in place on
reload; `torch.ops._sm70_C`; marking by shard shape instead of TP==4; TP2
shard geometries; selection through `Sm70CudaPlatform.get_unquantized_gemm`.
Dropped: the TP4-only geometries, the FP8 draft path
(`sm70_dflash2_fp8.py`), the `VLLM_BATCH_INVARIANT` checks and the fork test
`test_target_probe_preserves_dense_token_mask`.
