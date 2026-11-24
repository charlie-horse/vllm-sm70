# csrc/custom_ar: provenance

Source: charlie-horse/vLLM at commit `cf9492a01` (`csrc/custom_all_reduce.cu`,
`csrc/custom_all_reduce.cuh`, `csrc/torch_bindings.cpp` and the Python
callers named below). License: Apache-2.0. The kernels build against
upstream v0.33.0's own `custom_all_reduce.cuh`,
`custom_collective_common.cuh` and `type_convert.cuh` (pinned in
`UPSTREAM_HEADERS.sha256`); no upstream file is patched.

## TP2 all-reduce + add + RMSNorm (`tp2_ar_fused_add_rms_norm.cu`)

Fork commits `45896b591`, `9f175fc6e`, `2eb8f3b0a`.

| here | fork (cf9492a) | status |
|---|---|---|
| `tp2_ar_fused_add_rms_norm.cu: tp2_cross_device_reduce_fused_add_rms_norm<Sum2>` | `csrc/custom_all_reduce.cu:505-651` `sm70_tp2_cross_device_reduce_fused_add_rms_norm` | adapted: `vllm::` qualifiers; `_f16Vec`/`_typeConvert<c10::Half>` -> `<torch::headeronly::Half>`; `barrier_at_start` -> `staged_barrier_at_start` (below) |
| `staged_barrier_at_start<ngpus>` (`custom_ar_common.cuh`) | `csrc/custom_all_reduce.cuh:617-631` `barrier_at_start` (+ `st_flag_sys_visible` :560, `membar_sys` :575) | adapted: upstream's `barrier_at_start` has no system fence; the fork's does, and the staged kernel needs it |
| `tp2_all_reduce_fused_add_rms_norm` (host) | `csrc/custom_all_reduce.cu:1011-1095` | adapted: ROCm branch dropped; `int64_t` handle |
| `TORCH_LIBRARY_FRAGMENT(_sm70_C)` def | `csrc/torch_bindings.cpp:1689-1695` (`_C_custom_ar`) | adapted: `Tensor!` -> `Tensor(a!)`/`Tensor(b!)` |
| `vllm_sm70/ar_rms_fusion.py: Tp2AllReduceFusedAddRMSNormPattern` | `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:757-848` | adapted: `torch.ops.vllm.all_reduce` called directly; replacement is `torch.ops.vllm_sm70.*` |
| `_match`, `_scaled_match`, `unfused_boundaries` | same file :725-754, :851-866 | verbatim logic |
| `rejection` | same file `tp2_add_rms_rejection` :869-890 | adapted: no communication policy, adds `disable_custom_all_reduce` |
| `apply`, `register` | same file `AllReduceFusionPass.__init__`/`__call__` TP2 branches | adapted: one entry of `Sm70CompositePass` |
| `operands_admitted`, `communicator_admitted`, `can_fuse` | `collective_provider.py:117-142` `fused_add_rms_norm_tp2`, `custom_all_reduce.py:675-684` `can_tp2_...` | adapted: upstream's `should_custom_ar` gate in front; multi-node repeated-buffer guard |
| `tp2_all_reduce_fused_add_rms_norm` (op impl) + fake | `parallel_state.py:186-214`, `GroupCoordinator._tp2_..._out_place` :837-858, `CudaCommunicator.tp2_...` :475-506, `CustomAllreduce.tp2_...` :638-672 | adapted: one function on upstream's `device_communicator.ca_comm` |

The staged-input design is the fork's (`9f175fc6e`).

Dropped: the fork's `FlintModelConfig._configure_sm70_allreduce_add_rms`
hook and `CommunicationPolicy.ar_add_rms_tp2`, the `vllm/_custom_ops.py`
wrapper, the benchmark, the fork's `sm70_tile_runtime_signal.cuh` and every
TP4 / other-model collective in the same fork files.

## Greedy top-1 exchange (`top1_custom_ar.cu`)

| here | fork (cf9492a) | status |
|---|---|---|
| `top1_custom_ar.cu: staged_cross_device_top1_argmax<ngpus>` | `csrc/custom_all_reduce.cuh:1720-1750` `cross_device_top1_argmax<ngpus>` | adapted: staged (below); tie rule (`top1_better`: NaN first, larger value, lowest id) and `llrintf` id decode verbatim |
| `custom_ar_top1_argmax` (host) | `csrc/custom_all_reduce.cu:1449-1475` `top1_argmax` + `custom_all_reduce.cuh:3072-3106` `CustomAllreduce::top1_argmax` | adapted: one function on upstream's handle; same checks (FP32 pair, 2-256 values, int64 output of numel/2), same 1x128 launch and 2/4/6/8 rank cases; `int64_t` handle |
| `TORCH_LIBRARY_FRAGMENT(_sm70_C)` def `custom_ar_top1_argmax` | `csrc/torch_bindings.cpp:1762-1765` (`_C_custom_ar` `top1_argmax`) | renamed in `_sm70_C`; `Tensor!` -> `Tensor(a!)` |
| `vllm_sm70/sample_greedy.py: exchange_custom_ar` | `vllm/_custom_ops.py:3267-3274`, `CustomAllreduce.top1_argmax`/`custom_top1_argmax` (custom_all_reduce.py:771-790, :977-998) | adapted: called with `ca._ptr`, `ca.buffer_ptrs[ca.rank]`, `ca.max_size` of upstream's `get_tp_group().device_communicator.ca_comm`, as `ar_rms_fusion` reaches it |

Changed: each rank stores its pair into its own registered staging buffer
inside the kernel (the fork copied it with `cudaMemcpyAsync` and pushed it
onto `graph_unreg_buffers_` under capture). No capture branch. Default off
(`VLLM_SM70_TOP1_CUSTOM_AR=1` opts in; the fork defaulted on). Not ported:
the fork `CudaCommunicator` creating a `CustomAllreduce` only for this
exchange when `disable_custom_all_reduce=True`.

## Shared (`custom_ar_common.cuh`)

The ABI `static_assert`s on upstream's `Signal`, `RankData`, `RankSignals`
and `CustomAllreduce`, and `staged_barrier_at_start` (from fork
`custom_all_reduce.cuh:617-631`, `barrier_at_start` with its system fences).
`sources.cmake` (new) refuses headers other than the pinned ones.
