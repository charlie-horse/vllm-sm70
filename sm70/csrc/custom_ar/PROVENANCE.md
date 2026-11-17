# csrc/custom_ar: provenance

Two kernels on upstream's `CustomAllreduce`, built into `_sm70_C`:

* `tp2_ar_fused_add_rms_norm.cu`: TP2 all-reduce + add + RMSNorm (K17), below.
* `top1_custom_ar.cu`: the F12 greedy top-1 pair exchange
  (`VLLM_SM70_TOP1_CUSTOM_AR`), section "F12 custom-AR top-1 exchange".

Both include `custom_ar_common.cuh`: the ABI pins (`static_assert`s) and
`staged_barrier_at_start`, moved there unchanged from the K17 file when the
second kernel arrived.

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
| `staged_barrier_at_start<ngpus>` (`custom_ar_common.cuh`) | `csrc/custom_all_reduce.cuh:617-631` `barrier_at_start` (+ `st_flag_sys_visible` :560, `membar_sys` :575) | adapted: upstream's `barrier_at_start` has no system fence; the fork's does, and the staged kernel needs it |
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
  in `custom_ar_common.cuh` (every `.cu` here includes it; the header test
  checks that), checked with clang `-fsyntax-only`.
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

# F12 custom-AR top-1 exchange: provenance (`VLLM_SM70_TOP1_CUSTOM_AR`)

Source: charlie-horse/vLLM at **cf9492a01**. Fork default on
(`VLLM_SM70_TOP1_CUSTOM_AR=1`, envs.py:8315; `CommunicationPolicy.
top1_custom_ar`, hashed pre-Ampere TP>1, policy_defaults.py:328); here
default off, opt-in with `=1` (see `csrc/sampler/PROVENANCE.md`, F12). The
Python side (switch, admission, fallback to the all-gather) is in
`vllm_sm70/sample_greedy.py`; see `csrc/sampler/PROVENANCE.md` (F12).

## Function map

| here | fork (cf9492a) | status |
|---|---|---|
| `top1_custom_ar.cu: staged_cross_device_top1_argmax<ngpus>` | `csrc/custom_all_reduce.cuh:1720-1750` `cross_device_top1_argmax<ngpus>` | adapted: staged (below); tie rule (`top1_better`: NaN first, larger value, lowest id) and `llrintf` id decode verbatim |
| `custom_ar_top1_argmax` (host) | `csrc/custom_all_reduce.cu:1449-1475` `top1_argmax` + `custom_all_reduce.cuh:3072-3106` `CustomAllreduce::top1_argmax` | adapted: one function on upstream's handle; same checks (FP32 pair, 2-256 values, int64 output of numel/2), same 1x128 launch and 2/4/6/8 rank cases; `int64_t` handle |
| `TORCH_LIBRARY_FRAGMENT(_sm70_C)` def `custom_ar_top1_argmax` | `csrc/torch_bindings.cpp:1762-1765` (`_C_custom_ar` `top1_argmax`) | renamed in `_sm70_C`; `Tensor!` -> `Tensor(a!)` |
| `vllm_sm70/sample_greedy.py: exchange_custom_ar` | `vllm/_custom_ops.py:3267-3274`, `CustomAllreduce.top1_argmax`/`custom_top1_argmax` (custom_all_reduce.py:771-790, :977-998) | adapted: called with `ca._ptr`, `ca.buffer_ptrs[ca.rank]`, `ca.max_size` of upstream's `get_tp_group().device_communicator.ca_comm`, as `ar_rms_fusion` reaches it |

## Staged exchange instead of graph buffers (R5 design)

The fork copied the pair into the registered eager buffer with
`cudaMemcpyAsync` out of capture and pushed the caller's pair tensor onto
`graph_unreg_buffers_` under capture (warmup took the all-gather). Here,
as in K17 after R5, each thread stores its row's pair into the rank's own
staging buffer (upstream's eager buffer, `buffers_`, registered at init)
inside the kernel, the block publishes the start flag behind
`membar.sys` (`staged_barrier_at_start`), reads every rank's staging copy
in rank order, and publishes/awaits the end flag after the reads, so no rank
rewrites its staging buffer while a peer reads it. The `__syncthreads`
before the start barrier orders every thread's staging store before the
flag thread's `membar.sys` (bar.sync synchronizes-with, and fence.sc is
cumulative over it in the PTX memory model), as in K17 and upstream's
two-stage kernel: no per-thread fence is needed. One launch, no memcpy,
nothing added to the graph-buffer list. F12 runs only eagerly (the U4 hook
rejects capture), so the capture branch is not needed.

Signal protocol: block 0's `start`/`end` counters and `_flag[0]`, advanced
exactly as upstream's kernels and K17 advance them (start +1, end +1), so the
exchange interleaves with upstream's custom all-reduce and K17 on the same
communicator. The end-flag value is read after a `__syncthreads` that
follows thread 0's `_flag` update in the start barrier.

## Admitted set

Kernel: FP32 contiguous pair `[rows, 2]`, 1-128 rows, int64 output, 2/4/6/8
ranks, the registered staging buffer. Python (`custom_ar_reject_reason`,
same on all ranks): switch on, TP > 1, CUDA, rows 1-128, upstream
`CustomAllreduce` present and enabled with world size = TP size and distinct
peer buffers (one node), `_sm70_C` built with this area. Otherwise the
all-gather in upstream `get_top_tokens`.

Not ported: the fork's `CudaCommunicator` creating a `CustomAllreduce` only
for this exchange when `disable_custom_all_reduce=True` (would need an
upstream patch); then the all-gather runs.

## Build and test status

Compiled for sm_70 (device: clang 18 CUDA + CUDA 12.6 headers, ptxas 12.6,
21-30 registers, no stack; host: clang 18 against torch 2.14 headers and the
pinned v0.33.0 csrc headers), together with the refactored K17 file. Not
yet built by nvcc in the Nix build and not yet run on a V100:
`pytest -m gpu sm70/tests/distributed/test_f12_top1_custom_ar_gpu.py`
compares it with the NCCL all-gather on two ranks.
