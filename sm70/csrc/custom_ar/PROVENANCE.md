# TP2 all-reduce + add + RMSNorm: provenance (patch K17, carry-map C6/U5/I17)

Source: charlie-horse/vLLM at **cf9492a01**. Feature commits `45896b591`
(fused collective + pattern), `9f175fc6e` (R5 fix: staged inputs, no graph
buffers), `2eb8f3b0a` (default on). Fork measurement: decode step
6.00 -> 5.59 ms (166.4 -> 179.3 tok/s at 4K).

Design change from the carry map: **no upstream edit (U5 is dropped).** The
kernel is in `_sm70_C` and is built against upstream's own
`csrc/custom_all_reduce.cuh` + `custom_collective_common.cuh` +
`type_convert.cuh` from `$VLLM_SM70_VLLM_CSRC_DIR` (the patched tree's
`csrc/`, set by `nix/sm70-ext.nix`, PR #61). Python passes upstream's
`CustomAllreduce._ptr` to it.

## Function map

| here | fork (cf9492a) | status |
|---|---|---|
| `tp2_ar_fused_add_rms_norm.cu: tp2_cross_device_reduce_fused_add_rms_norm<Sum2>` | `csrc/custom_all_reduce.cu:505-651` `sm70_tp2_cross_device_reduce_fused_add_rms_norm` | adapted: `vllm::` qualifiers; `_f16Vec`/`_typeConvert<c10::Half>` -> `<torch::headeronly::Half>`; `barrier_at_start` -> `staged_barrier_at_start` (below) |
| `staged_barrier_at_start<ngpus>` | `csrc/custom_all_reduce.cuh:617-631` `barrier_at_start` (+ `st_flag_sys_visible` :560, `membar_sys` :575) | adapted: upstream's `barrier_at_start` has no system fence; the fork's does, and the staged kernel needs it |
| `tp2_all_reduce_fused_add_rms_norm` (host) | `csrc/custom_all_reduce.cu:1011-1095` | adapted: ROCm branch dropped; `int64_t` handle |
| `TORCH_LIBRARY_FRAGMENT(_sm70_C)` def | `csrc/torch_bindings.cpp:1689-1695` (`_C_custom_ar`) | adapted: `Tensor!` -> `Tensor(a!)`/`Tensor(b!)` |
| `vllm_sm70/ar_rms_fusion.py: Tp2AllReduceFusedAddRMSNormPattern` | `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:757-848` | adapted: `torch.ops.vllm.all_reduce` called directly; replacement is `torch.ops.vllm_sm70.*` |
| `_match`, `_scaled_match`, `unfused_boundaries` | same file :725-754, :851-866 | verbatim logic |
| `rejection` | same file `tp2_add_rms_rejection` :869-890 | adapted: no communication policy (kill switch moved to `register()`), adds `disable_custom_all_reduce` |
| `apply`, `register` | same file `AllReduceFusionPass.__init__`/`__call__` TP2 branches | adapted: one entry of `Sm70CompositePass` |
| `operands_admitted`, `communicator_admitted`, `can_fuse` | `collective_provider.py:117-142` `fused_add_rms_norm_tp2`, `custom_all_reduce.py:675-684` `can_tp2_...` | adapted: upstream's `should_custom_ar` gate in front; multi-node repeated-buffer guard |
| `tp2_all_reduce_fused_add_rms_norm` (op impl) + fake | `parallel_state.py:186-214`, `GroupCoordinator._tp2_..._out_place` :837-858, `CudaCommunicator.tp2_...` :475-506, `CustomAllreduce.tp2_...` :638-672 | adapted: one function on upstream's `device_communicator.ca_comm` |

Dropped: the fork's `FlintModelConfig._configure_sm70_allreduce_add_rms`
hook and `CommunicationPolicy.ar_add_rms_tp2` (fork-only config plumbing;
replaced by `register()` + `rejection()`), `vllm/_custom_ops.py` wrapper,
`benchmarks/kernels/benchmark_sm70_tp2_ar_fused_add_rms_norm.py`, the fork's
`sm70_tile_runtime_signal.cuh` (unused by this kernel), and every TP4 /
other-model collective in the same fork files.

## R5 (FULL capture: `cudaIpcGetMemHandle: invalid argument`)

`45896b591` pushed the op's inputs onto `graph_unreg_buffers_` under capture,
as the stock collective does. Its two-input form takes the shared- and
routed-expert outputs, which need not come from the capture pool; under
expandable segments (VMM) `get_graph_buffer_ipc_meta` then failed to export
them. `9f175fc6e` made each rank copy its (summed) input into its own
staging buffer, the communicator's eager buffer registered at init
(`buffers_`), publish the start flag behind a system fence, and read the
peer's staging copy. Nothing is added to the graph-buffer list, so the op
is identical eagerly, in warmup and in FULL/PIECEWISE capture wherever the
operands live. This port keeps that design. Upstream's own capture path for
the plain all-reduce is unaffected (it still registers, or copies when
`register_graph_buffers=False`, e.g. a cuMem pool).

## Admitted shape set

Flint only (`model_config.architecture == "FlintForCausalLM"`: the fork
requested the pass only from `FlintModelConfig`, `MODELS_CONFIG_MAP`
fork models/config.py:1052), TP2, PP1, FP16, hidden 2048, FP16 residual and
FP16 `[2048]` weight in the matched graph (an fp32 GemmaRMSNorm-style
weight + 1 never matches), rows 1-16, all operands contiguous and 16-byte
aligned on one device, `scale` 1.0 or a positive power of two, eps from the
model. The kernel hard-codes 256 threads x 8 halves per row, so hidden 2048
is the only safe width; it is not widened. Rows > 16 (prefill), a disabled
or multi-node custom all-reduce, or `should_custom_ar() == False` run the
unfused ops. Same set as the fork.

## ABI guards

* `static_assert`s on `sizeof`/`offsetof` of `vllm::Signal`,
  `RankData`, `RankSignals`, `CustomAllreduce` (v0.33.0, x86_64 libstdc++),
  checked with clang `-fsyntax-only`.
* `UPSTREAM_HEADERS.sha256`: CMake refuses other header bytes;
  `sm70/tests/compile/test_k17_custom_ar_headers.py` checks the same pins,
  `patches/BASE`, and that no patch in `patches/series` edits the headers.

## Numerics: fused_add_rms_norm provider

Upstream lowers `vllm.ir.ops.fused_add_rms_norm` to `["native"]` under
Inductor (UP platforms/cuda.py:712-734), which keeps the residual sum in FP32
(UP ir/ops/layernorm.py:44-62). The kernel follows `_C.fused_add_rms_norm`
(FP16 residual). `Sm70CudaPlatform.get_default_ir_op_priority` pins
`fused_add_rms_norm` to `["vllm_c", "native"]` through upstream's platform
hook (UP platforms/interface.py:1452, merged at config/kernel.py:461-478), so
the unfused graph (switch off, prefill rows) and the fused collective round
the same way. A user-set priority still comes first.
