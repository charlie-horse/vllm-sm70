# QPN8 block-FP8 decode GEMV: provenance

Source: charlie-horse/vLLM at commit `cf9492a01`,
`csrc/sm70_turbomind/ops/`. License: Apache-2.0; the QPN8 layout is derived
from dnv2003/v100-skinny (MIT, `LICENSE.v100-skinny`) and its block-scale
adaptation in haohervchb/sglang-V100. Line counts are `wc -l`; "fork
last-touch" is `git log -1 --format=%h -- <fork path>`. Check with
`sm70/tools/check_qpn8_provenance.py [--fork <checkout at cf9492a>]`.

## Files

| new path | fork path | status | lines | fork lines | fork last-touch | sha256 | fork sha256 |
|---|---|---|---|---|---|---|---|
| `sm70/csrc/qpn8/fp8_qpn8_sm70.cu` | `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | rewritten | 972 | 2248 | 495fb442a | `79afe9c39eede56d7f1dcf4790efa52f89a491ce464599b4a2d2873f2c758a7c` | `7900034f1057a66f99ada9673d64acc98a0bc40ed6069c1ade361ffc02c5ce6d` |
| `sm70/csrc/qpn8/LICENSE.v100-skinny` | `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` | verbatim | 21 | 21 | 58beb3433 | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` | `d83507dc68f1bfa4b0ce4df4f0a82dca45e3e7df884b022519fb9d74cc769bf2` |
| `sm70/csrc/qpn8/qpn_pair_sm70.cuh` | `csrc/sm70_turbomind/ops/qpn_pair_sm70.cuh` | verbatim | 135 | 135 | aeede18c5 | `ed51466066679d5291129f0a9a7f64ef565632f78f0489054acd217cad773a32` | `ed51466066679d5291129f0a9a7f64ef565632f78f0489054acd217cad773a32` |
| `sm70/csrc/qpn8/activation_pack_sm70.cuh` | `csrc/sm70_turbomind/ops/activation_pack_sm70.cuh` | verbatim | 23 | 23 | aeede18c5 | `dda71dbdd6313c2000c859cea2e990b39017ee92e83304a31c4fbdd8b68f7cbb` | `dda71dbdd6313c2000c859cea2e990b39017ee92e83304a31c4fbdd8b68f7cbb` |
| `sm70/csrc/qpn8/sources.cmake` | - | new | 15 | - | - | `ecd99e99a4856f655d1f0a99df231e522fdb57b3238664e43b94caadb212bbb1` | - |

## Changes from the source (`fp8_qpn8_sm70.cu`)

- Kept: prepare, dequantize and the M<=32 GEMM (`fp8_qpn8_sm70_kernel`,
  its launcher, `Fp8PairReader`, the M17..32 two-phase kernel and the
  helpers), with `qpn_pair_sm70.cuh` and `activation_pack_sm70.cuh`.
- Dropped: `sm70_policy.h`; SiLU-and-mul, BA-split, gated-pair, split-CTA
  M1 and HC kernels and host ops; `fp8_qpn8_prefill_sm70_out`; the
  dispatch ops; the standalone `TORCH_LIBRARY_FRAGMENT(_C)` block.
- `fp8_qpn8_gemm_sm70_out`: block-scaled weights are admitted at M9..32 (the
  fork admitted M>8 for channel scales only). The `RowTiles=2` launch passes
  the real `channel_scales` flag; block-scaled M17..32 runs as 16-row
  chunks; the M17..32 split/decoder guard and the two-phase branch apply to
  channel scales only.
- Registration: `TORCH_LIBRARY_FRAGMENT(_sm70_C)` /
  `TORCH_LIBRARY_IMPL(_sm70_C, CUDA)` for `fp8_qpn8_prepare_sm70`,
  `fp8_qpn8_dequantize_sm70_out` and `fp8_qpn8_gemm_sm70_out`, without the
  fork's trailing `str? native_policy`; outputs are `Tensor(a!)`.
- Comments edited.
- `sources.cmake` (new) replaces the fork's TurboMind source-list entry.
- Python side (`sm70/vllm_sm70/ops/qpn8.py`): wrappers, configs and
  admission from fork `vllm/_sm70/linear.py` and
  `model_executor/kernels/linear/qpn/fp8*.py`, without `native_policy` and
  the TP4 shape tables.
