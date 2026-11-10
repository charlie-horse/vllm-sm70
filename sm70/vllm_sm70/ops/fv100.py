# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-V100 attention ops (K05): Python wrappers and fakes over torch.ops._sm70_C.

Ported from fork cf9492a flash-attention-v100/flash_attn_v100/
flash_attn_interface.py (the lean subset of Ivy's section B). Differences
from the fork:

* The pybind module `flash_attn_v100_cuda` became five `_sm70_C::fv100_*`
  out ops. Each takes `out` first and returns None; the wrappers below keep
  the fork signatures and return `out`.
* No policy object (`_runtime`, `legacy_policy`): the fork defaults are
  constants here. Decode workspaces are shared across batch sizes
  (fork VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE unset -> on) and the launch
  grid follows the runtime length (VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS
  unset -> on). VLLM_FLASH_V100_DECODE_PARTITION_SIZE is gone; pass
  `partition_size_hint` instead.
* No anchored SWA, FP8 KV, staged XQA or DFlash2 split.
* K14 (C4): Flint sink layers (GQA 8, D128, fp16 KV) decode in one launch
  through `fv100_decode_paged_sinks_fused` when the op is built and
  VLLM_SM70_FLINT_ATTN_FUSED_DECODE is not "0". Its int32 arrival counters
  are per device, allocated and zeroed only outside CUDA-graph capture (the
  `--max-num-seqs 20` gibberish fix, fork 7f12d21b8); a capture without them
  decodes in two launches. See "Fused one-launch sink decode" below.

KV cache views: `k_cache`/`v_cache` are [num_blocks, block_size, H_kv, D]
views with any strides in the first three dims and a contiguous last dim.
`check_kv_cache_views()` checks that on the CPU (metadata only), and
`split_lbnhc_kv_cache()` builds the views from an upstream v0.33.0 KV cache
(LBNHC: K and V interleaved per head, so stride(2) == 2 * D). Wiring the
backend to that layout is K06.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

try:
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover
    FakeTensor = None

NAMESPACE = "_sm70_C"

# Load _sm70_C before any has_op() check below (see ops/tm.py).
from vllm_sm70 import _ext  # noqa: E402

_ext.load()


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def _op(name: str):
    if not has_op(name):
        raise RuntimeError(
            f"SM70 Flash-V100 op {NAMESPACE}::{name} is not available. "
            "Build vllm_sm70._sm70_C with CUDA 12.x for sm_70."
        )
    return getattr(getattr(torch.ops, NAMESPACE), name)


def available() -> bool:
    """All K05 Flash-V100 ops are present in the loaded _sm70_C."""
    return all(has_op(name) for name in OPS)


# The two entry points that apply learned sinks (paged prefill and decode).
SINK_OPS = ("fv100_prefill_paged_sinks", "fv100_decode_paged_sinks")


def supports_sinks() -> bool:
    """The loaded _sm70_C applies learned attention sinks.

    True iff both K05 sink ops are registered. Replaces the fork's
    `flash_attn_v100.SUPPORTS_SINKS` (cf9492a flash_attn_v100/__init__.py,
    computed from `prefill_paged_sinks_fwd` and `decode_paged_sinks_fwd`;
    read by vllm/model_executor/models/config.py:218-229). Unlike
    FlashAttnV100Backend.supports_sink(), which is a constant class flag,
    this reflects the build. Sinks apply on head size 128 only (the D256
    kernels have no sink term).
    """
    return all(has_op(name) for name in SINK_OPS)


logger = logging.getLogger("vllm.sm70")


OPS = (
    "fv100_decode_paged",
    "fv100_decode_paged_sinks",
    "fv100_decode_paged_xqa",
    "fv100_prefill_paged",
    "fv100_prefill_paged_sinks",
)

# fork flash_attn_interface.py:20-21
DEFAULT_DECODE_PARTITION_SIZE = 256
VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)
# Head sizes the lean build instantiates (decode: 128/256; XQA: 256).
SUPPORTED_HEAD_SIZES = (128, 256)
XQA_GQA_RATIOS = (4, 6, 8)

# K14 (C4): optional sixth op, built unless VLLM_SM70_FV100_FUSED_DECODE=OFF.
FUSED_DECODE_OP = "fv100_decode_paged_sinks_fused"
FUSED_DECODE_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE"
FUSED_DECODE_ZERO_COUNTERS_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE_ZERO_COUNTERS"
# Shapes the fused kernel takes (flash_decode_paged.cu
# gqa8_d128_decode_eligible): D128, 8 query heads per KV head, with sinks.
FUSED_DECODE_HEAD_DIM = 128
FUSED_DECODE_GQA = 8
# fork flash_attn_interface.py:32-34: int32 arrival counters, one per
# (row, KV head), 1024 rows x 16 KV heads (64 KiB) per device.
FUSED_DECODE_COUNTER_CAPACITY = 1024 * 16

_decode_plan_cache: dict = {}
_decode_workspace_cache: dict = {}


# ---------------------------------------------------------------------------
# KV cache views (K05 decision 3: arbitrary strides, LBNHC interleave)
# ---------------------------------------------------------------------------


def check_kv_cache_views(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    head_size: int | None = None,
    num_kv_heads: int | None = None,
) -> None:
    """Raise ValueError unless k/v are views the kernels accept.

    Metadata only (shape, dtype, strides, alignment): no device sync, so the
    backend can call it once per KV cache at bind time. Requirements, from
    the kernels' TORCH_CHECKs and their aligned half8 loads:
      * 4-D [num_blocks, block_size, H_kv, D], same shape and device, fp16;
      * last dim contiguous, D in SUPPORTED_HEAD_SIZES;
      * strides of dims 0-2 multiples of 8 elements and 16-byte aligned base
        pointers, so LBNHC (stride(2) == 2 * D, K and V interleaved per
        head) and the fork's separate-K/V layout both pass.
    """
    problems = []
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        if t.dim() != 4:
            problems.append(f"{name} must be 4-D [blocks, block_size, H_kv, D]")
            continue
        if t.dtype != torch.float16:
            problems.append(f"{name} must be float16, got {t.dtype}")
        if t.stride(-1) != 1:
            problems.append(f"{name} last dim must be contiguous")
        bad = [s for s in t.stride()[:3] if s % 8 != 0]
        if bad:
            problems.append(f"{name} strides {tuple(t.stride())} not multiples of 8")
        if t.device.type != "meta" and t.data_ptr() % 16 != 0:
            problems.append(f"{name} data pointer is not 16-byte aligned")
    if not problems:
        if k_cache.shape != v_cache.shape:
            problems.append(
                f"k/v shapes differ: {tuple(k_cache.shape)} vs {tuple(v_cache.shape)}"
            )
        if k_cache.device != v_cache.device:
            problems.append("k/v caches are on different devices")
        d = k_cache.shape[-1]
        if d not in SUPPORTED_HEAD_SIZES:
            problems.append(f"head size {d} not in {SUPPORTED_HEAD_SIZES}")
        if head_size is not None and d != head_size:
            problems.append(f"head size {d} != expected {head_size}")
        if num_kv_heads is not None and k_cache.shape[2] != num_kv_heads:
            problems.append(f"H_kv {k_cache.shape[2]} != expected {num_kv_heads}")
    if problems:
        raise ValueError("Flash-V100 KV cache views: " + "; ".join(problems))


def split_lbnhc_kv_cache(
    kv_cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V views of an upstream v0.33.0 per-layer KV cache.

    `kv_cache` is the logical (num_blocks, H_kv, block_size, 2 * head_size)
    tensor upstream TritonAttention receives (UP triton_attn.py:655-657).
    Returns two [num_blocks, block_size, H_kv, head_size] views, no copy.
    """
    if kv_cache.dim() != 4 or kv_cache.shape[-1] != 2 * head_size:
        raise ValueError(
            f"expected (blocks, H_kv, block_size, {2 * head_size}), "
            f"got {tuple(kv_cache.shape)}"
        )
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Helpers: fork flash_attn_interface.py:94-135
# ---------------------------------------------------------------------------


def maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _copy_bhmd_to_bmhd_out(
    out_bhmd: torch.Tensor,
    out_bmhd: torch.Tensor | None,
) -> torch.Tensor:
    out = out_bhmd.permute(0, 2, 1, 3).contiguous()
    if out_bmhd is not None:
        out_bmhd.copy_(out)
        return out_bmhd
    return out


def _is_fake_tensor(x: torch.Tensor) -> bool:
    return FakeTensor is not None and isinstance(x, FakeTensor)


def _can_cache_workspace(x: torch.Tensor) -> bool:
    return (
        not torch.compiler.is_compiling() and not x.is_meta and not _is_fake_tensor(x)
    )


def _workspace_stream_id(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    if torch.compiler.is_compiling():
        return 0
    return int(torch.cuda.current_stream(device).cuda_stream)


def _round_decode_partition_capacity(required_num_partitions: int) -> int:
    if required_num_partitions <= 1:
        return 1
    return 1 << (required_num_partitions - 1).bit_length()


# fork flash_attn_interface.py:282-289
def _cuda_graph_capture_active() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Decode plan and workspace: fork flash_attn_interface.py:52-92, 292-590,
# 840-889 (policy reads replaced by the fork defaults)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DecodePlan:
    partition_size: int
    actual_num_partitions: int
    launch_num_partitions: int
    workspace_num_partitions: int


@dataclass
class _DecodeWorkspace:
    tmp_out: torch.Tensor
    max_logits: torch.Tensor
    exp_sums: torch.Tensor
    active_num_partitions: torch.Tensor
    max_num_partitions: int
    captured: bool = False
    previous: _DecodeWorkspace | None = None


def _allocate_decode_workspace(
    q: torch.Tensor,
    *,
    batch_capacity: int,
    num_heads: int,
    head_dim: int,
    max_num_partitions: int,
    partial_dtype: torch.dtype = torch.float16,
) -> _DecodeWorkspace:
    return _DecodeWorkspace(
        tmp_out=torch.empty(
            (batch_capacity, num_heads, max_num_partitions, head_dim),
            dtype=partial_dtype,
            device=q.device,
        ),
        max_logits=torch.empty(
            (batch_capacity, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=q.device,
        ),
        exp_sums=torch.empty(
            (batch_capacity, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=q.device,
        ),
        active_num_partitions=torch.empty(
            (1,),
            dtype=torch.int32,
            device=q.device,
        ),
        max_num_partitions=max_num_partitions,
    )


def _select_default_decode_partition_size(max_seq_len_hint: int | None) -> int:
    if max_seq_len_hint is None:
        return DEFAULT_DECODE_PARTITION_SIZE
    seq_len = max(1, int(max_seq_len_hint))
    if seq_len >= 32768:
        return 1024
    return DEFAULT_DECODE_PARTITION_SIZE


def _validate_decode_partition_size(
    value: int,
    name: str,
    valid_partition_sizes: tuple[int, ...] = VALID_DECODE_PARTITION_SIZES,
) -> int:
    if value not in valid_partition_sizes:
        raise ValueError(f"{name} must be one of {valid_partition_sizes}, got {value}")
    return value


def _get_decode_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    max_seq_len_hint: int | None = None,
    batch_size_hint: int | None = None,
    workspace_seq_capacity_hint: int | None = None,
    active_num_partitions: torch.Tensor | None = None,
    partition_size_hint: int | None = None,
) -> _DecodePlan:
    batch_capacity = batch_size_hint or block_table.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    max_seq_capacity = block_table.shape[1] * k_cache.shape[1]
    effective_max_seq_len = int(max_seq_len_hint or max_seq_capacity)
    effective_workspace_seq_capacity = int(
        workspace_seq_capacity_hint or effective_max_seq_len
    )
    effective_max_seq_len = max(1, effective_max_seq_len)
    effective_workspace_seq_capacity = max(
        1,
        effective_workspace_seq_capacity,
        effective_max_seq_len,
    )
    if (
        workspace_seq_capacity_hint is not None
        and active_num_partitions is None
        and _cuda_graph_capture_active()
    ):
        # Capture the full workspace envelope: replay re-runs the captured
        # fill_ of active_num_partitions, not this planning (fork :170-182).
        effective_max_seq_len = max(
            effective_max_seq_len,
            effective_workspace_seq_capacity,
        )
    if partition_size_hint is not None:
        partition_size = _validate_decode_partition_size(
            int(partition_size_hint), "partition_size_hint"
        )
    else:
        partition_size = _select_default_decode_partition_size(effective_max_seq_len)
    runtime_num_partitions = max(
        1,
        (effective_max_seq_len + partition_size - 1) // partition_size,
    )
    workspace_num_partitions = max(
        1,
        (effective_workspace_seq_capacity + partition_size - 1) // partition_size,
    )
    plan = _DecodePlan(
        partition_size=partition_size,
        actual_num_partitions=runtime_num_partitions,
        launch_num_partitions=(
            workspace_num_partitions
            if workspace_seq_capacity_hint is not None
            else runtime_num_partitions
        ),
        workspace_num_partitions=workspace_num_partitions,
    )
    device_index = q.device.index if q.device.index is not None else -1
    key = (
        device_index,
        batch_capacity,
        num_heads,
        head_dim,
        q.dtype,
        plan.partition_size,
        plan.actual_num_partitions,
        plan.launch_num_partitions,
        plan.workspace_num_partitions,
    )
    if _can_cache_workspace(q) and workspace_seq_capacity_hint is None:
        cached = _decode_plan_cache.get(key)
        if cached is not None:
            return cached
        _decode_plan_cache[key] = plan
    return plan


def _assert_decode_launch_covers_seq_lens(
    plan: _DecodePlan,
    seq_lens: torch.Tensor | None,
    *,
    workspace_seq_capacity_hint: int | None,
) -> None:
    """The launch grid must cover the batch's true longest sequence.

    Runs only with a workspace capacity hint, outside CUDA-graph capture
    (fork :452-500: one device sync, never on the plain q=1 path).
    """
    if workspace_seq_capacity_hint is None or _cuda_graph_capture_active():
        return
    if seq_lens is None or seq_lens.numel() == 0:
        return
    true_max_seq_len = int(seq_lens.max().item())
    required_num_partitions = max(
        1,
        (true_max_seq_len + plan.partition_size - 1) // plan.partition_size,
    )
    if plan.launch_num_partitions < required_num_partitions:
        raise RuntimeError(
            "decode launch grid under-covers the batch: launch_num_partitions="
            f"{plan.launch_num_partitions} < required={required_num_partitions} "
            f"to cover max seq_len={true_max_seq_len} "
            f"(partition_size={plan.partition_size}, "
            f"workspace_seq_capacity_hint={workspace_seq_capacity_hint})."
        )


def _get_decode_workspace_for_plan(
    q: torch.Tensor,
    *,
    batch_capacity: int,
    num_heads: int,
    head_dim: int,
    plan: _DecodePlan,
    active_num_partitions: torch.Tensor | None = None,
    partial_dtype: torch.dtype = torch.float16,
    round_partitions: bool = True,
    refresh_active: bool = True,
):
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = _workspace_stream_id(q.device)
    # B2's verify workspace keeps its exact partition count (round_partitions
    # False); the K05 decode rounds to a power of two (fork behaviour).
    capacity_of = _round_decode_partition_capacity if round_partitions else int
    # Rows are shared across batch sizes (fork default share_workspace=on).
    partition_capacity = capacity_of(plan.workspace_num_partitions)
    key = (
        device_index,
        stream_id,
        num_heads,
        head_dim,
        plan.partition_size,
        partial_dtype,
        partition_capacity,
    )
    workspace = _decode_workspace_cache.get(key) if _can_cache_workspace(q) else None
    if (
        workspace is None
        or workspace.tmp_out.size(0) < batch_capacity
        or workspace.max_num_partitions < plan.workspace_num_partitions
    ):
        # Growth keeps older buffers alive for graphs that captured them.
        previous = workspace
        workspace = _allocate_decode_workspace(
            q,
            batch_capacity=max(
                batch_capacity, previous.tmp_out.size(0) if previous else 0
            ),
            num_heads=num_heads,
            head_dim=head_dim,
            max_num_partitions=capacity_of(
                max(
                    plan.workspace_num_partitions,
                    previous.max_num_partitions if previous else 0,
                )
            ),
            partial_dtype=partial_dtype,
        )
        if previous is not None:
            workspace.previous = previous if previous.captured else previous.previous
        if _can_cache_workspace(q):
            _decode_workspace_cache[key] = workspace

    if _cuda_graph_capture_active():
        workspace.captured = True
    if active_num_partitions is None:
        if refresh_active:
            workspace.active_num_partitions.fill_(plan.actual_num_partitions)
        active_num_partitions = workspace.active_num_partitions
    return (
        workspace.tmp_out[:batch_capacity],
        workspace.max_logits[:batch_capacity],
        workspace.exp_sums[:batch_capacity],
        active_num_partitions,
    )


def clear_workspaces() -> None:
    """Drop cached decode plans, workspaces and fused-decode counters
    (tests, engine shutdown)."""
    _decode_plan_cache.clear()
    _decode_workspace_cache.clear()
    _fused_decode_counter_cache.clear()
    _fused_decode_warned.clear()
    _fused_decode_capture_counts.update(fused=0, two_launch=0)


# ---------------------------------------------------------------------------
# Fused one-launch sink decode (K14, C4): fork flash_attn_interface.py:30-36,
# 59-71, 141-275 and the counter fix 7f12d21b8 / 89f751847 / 7e1c8c089 /
# 7b2bacefd. The policy object is gone: the switches are read from the
# environment on each call, as the fork's legacy_policy.py:42-47 did.
# ---------------------------------------------------------------------------


@dataclass
class _FusedDecodeCounters:
    """Zero-between-launches arrival counters of the one-launch sink decode.

    Allocated (and zeroed) only outside CUDA-graph capture: a ``torch.zeros``
    under capture only records a memset into that graph, so graphs captured
    later would replay on whatever the buffer holds (the --max-num-seqs 20
    gibberish). ``retired`` keeps smaller buffers alive that captured graphs
    may still reference.
    """

    buffer: torch.Tensor
    retired: list[torch.Tensor]


# Keyed by device index (-1: no index), not by stream; see _fused_decode_counters.
_fused_decode_counter_cache: dict[int, _FusedDecodeCounters] = {}
# Grouped sink decodes recorded into CUDA graphs: one-launch vs two-launch.
_fused_decode_capture_counts = {"fused": 0, "two_launch": 0}
_fused_decode_warned: set[str] = set()
_FUSED_DECODE_OFF = (
    "Flash-V100 one-launch sink decode is off for CUDA graphs captured on "
    "this device ({}); they decode with the partition kernel plus reducer. "
    "To keep it, run an eager sink-layer forward or call "
    "flash_attn_decode_paged_sinks_prepare on the device before capture; "
    f"{FUSED_DECODE_ENV}=0 selects the two-launch route on purpose."
)
# Logging formats per fixed reason code; numbers stay lazy %-arguments.
_FUSED_DECODE_OFF_FORMATS = {
    "no_counters": _FUSED_DECODE_OFF.format(
        "no counters were allocated before capture"
    ),
    "capacity": _FUSED_DECODE_OFF.format(
        "capture needs %s counters, %s were allocated"
    ),
}


def fused_decode_enabled() -> bool:
    """Kill switch: VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0 turns it off."""
    return os.environ.get(FUSED_DECODE_ENV) != "0"


def _fused_decode_zero_counters_enabled() -> bool:
    """Opt-in (fork default off): one counter memset per fused launch."""
    return os.environ.get(FUSED_DECODE_ZERO_COUNTERS_ENV) == "1"


def fused_decode_available() -> bool:
    """The fused op is built and not switched off (fork :236-240)."""
    return has_op(FUSED_DECODE_OP) and fused_decode_enabled()


def fused_decode_eligible(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    sinks: torch.Tensor | None,
    kv_cache_dtype: str = "auto",
) -> bool:
    """Shapes the one-launch kernel takes (fork :1366-1377).

    Flint per rank: TP1 32q/4kv and TP2 16q/2kv (and TP4 8q/1kv), D128,
    sinks. The C++ side re-checks this and the 16-byte alignment of q, K, V,
    out and tmp_out (gqa8_d128_decode_eligible) and falls back to two
    launches for anything else, so a False here only saves the counters.
    """
    if sinks is None or kv_cache_dtype not in ("auto", "float16"):
        return False
    if q.dim() != 3 or k_cache.dim() != 4 or k_cache.shape[2] <= 0:
        return False
    return (
        q.shape[2] == FUSED_DECODE_HEAD_DIM
        and q.shape[1] == FUSED_DECODE_GQA * k_cache.shape[2]
    )


def _warn_fused_decode_once(code: str, *args) -> None:
    """Warn once per fixed reason code, with its numbers as lazy args."""
    if code in _fused_decode_warned:
        return
    _fused_decode_warned.add(code)
    logger.warning(_FUSED_DECODE_OFF_FORMATS[code], *args)


def _normalized_device(device: torch.device) -> torch.device:
    """``cuda`` without an index is the current device, as kernels see it."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.accelerator.current_device_index())
    return device


def _fused_decode_counters(device: torch.device, required: int) -> torch.Tensor | None:
    """The device's counters; None where they cannot be created.

    Keyed by device, not stream: a capture finds them whichever stream ran
    the eager call that created them (vLLM captures on a fresh stream). That
    is safe because each launch leaves them at zero and sink decodes on one
    device never run concurrently (one compute stream; captures run one
    after another). Never creates zero-dependent state under capture: a
    capture without a big-enough buffer gets None and takes two launches.
    """
    device = _normalized_device(device)
    key = device.index if device.index is not None else -1
    entry = _fused_decode_counter_cache.get(key)
    if entry is not None and entry.buffer.numel() >= required:
        return entry.buffer
    if _cuda_graph_capture_active():
        if entry is None:
            _warn_fused_decode_once("no_counters")
        else:
            _warn_fused_decode_once("capacity", required, entry.buffer.numel())
        return None
    if torch.compiler.is_compiling():
        return None
    buffer = torch.zeros(
        max(FUSED_DECODE_COUNTER_CAPACITY, required),
        dtype=torch.int32,
        device=device,
    )
    retired = [] if entry is None else [*entry.retired, entry.buffer]
    _fused_decode_counter_cache[key] = _FusedDecodeCounters(buffer, retired)
    return buffer


def flash_attn_decode_paged_sinks_prepare(device: torch.device | str | int) -> bool:
    """Allocate the one-launch sink decode's counters for a device.

    Counters are per device and must exist before capture; a graph captured
    on any stream then finds them. The Flash-V100 impl calls this when a
    sink layer is built and on its first eager forward. A no-op, returning
    False, when the fused decode is switched off or not built, for a
    non-CUDA device, or under capture.
    """
    if not fused_decode_available():
        return False
    device = torch.device(device)
    if device.type != "cuda" or _cuda_graph_capture_active():
        return False
    return _fused_decode_counters(device, 0) is not None


def flash_attn_decode_paged_sinks_capture_counts() -> tuple[int, int]:
    """Grouped sink decodes captured so far: (one-launch, two-launch)."""
    return (
        _fused_decode_capture_counts["fused"],
        _fused_decode_capture_counts["two_launch"],
    )


# ---------------------------------------------------------------------------
# Sink validation: fork flash_attn_interface.py:1437-1451, 2103-2121
# ---------------------------------------------------------------------------


def _validate_decode_sinks(sinks: torch.Tensor, q: torch.Tensor) -> None:
    """Metadata-only checks: no host sync, copy or allocation per call."""
    if not isinstance(sinks, torch.Tensor):
        raise TypeError(f"sinks must be a torch.Tensor, got {type(sinks)!r}")
    if sinks.dtype != torch.float32:
        raise TypeError(f"sinks must be float32, got {sinks.dtype}")
    if sinks.dim() != 1 or sinks.shape[0] != q.shape[1]:
        raise ValueError(
            f"sinks must have shape [{q.shape[1]}] (one per query head), "
            f"got {tuple(sinks.shape)}"
        )
    if sinks.device != q.device:
        raise ValueError(f"sinks must be on {q.device}, got {sinks.device}")
    if not sinks.is_contiguous():
        raise ValueError("sinks must be contiguous")


def _validate_prefill_sinks(q: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """`q` is [B, M, H, D], so `sinks` must be fp32 [H] on its device."""
    if not isinstance(sinks, torch.Tensor):
        raise TypeError(f"sinks must be a torch.Tensor, got {type(sinks).__name__}")
    if sinks.dtype != torch.float32:
        raise TypeError(f"sinks must be float32, got {sinks.dtype}")
    if q.ndim != 4 or tuple(sinks.shape) != (q.shape[2],):
        raise ValueError(
            f"sinks must have shape [num_q_heads] for q of shape "
            f"{tuple(q.shape)} ([B, M, H, D]), got {tuple(sinks.shape)}"
        )
    if sinks.device != q.device:
        raise ValueError(
            f"sinks must be on the query device {q.device}, got {sinks.device}"
        )
    return maybe_contiguous(sinks)


def _window(window_size: tuple) -> tuple[int, int]:
    left, right = window_size
    if left < -1 or right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")
    return int(left), int(right)


# ---------------------------------------------------------------------------
# Public wrappers (fork signatures minus anchored SWA and `_runtime`)
# ---------------------------------------------------------------------------


# fork flash_attn_interface.py:1273-1434
def flash_attn_decode_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    window_size: tuple = (-1, -1),
    max_seq_len_hint: int | None = None,
    workspace_seq_capacity_hint: int | None = None,
    active_num_partitions: torch.Tensor | None = None,
    partition_size_hint: int | None = None,
    *,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged decode, q [B, H, D] -> out [B, H, D].

    `sinks` (fp32 [H], natural-log units) adds exp(sink) to each softmax
    denominator with no value row; None keeps the sink-free kernels. With
    sinks, GQA-8 D128 layers (Flint) take the one-launch fused kernel when
    it is built, switched on and has counters (K14), else the grouped
    two-launch kernel.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if sinks is not None:
        _validate_decode_sinks(sinks, q)
    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    if out is None:
        out = torch.empty_like(q)
    window_size_left, window_size_right = _window(window_size)
    batch_capacity, num_heads, head_dim = q.shape
    plan = _get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len_hint,
        batch_size_hint=batch_capacity,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        active_num_partitions=active_num_partitions,
        partition_size_hint=partition_size_hint,
    )
    _assert_decode_launch_covers_seq_lens(
        plan, seq_lens, workspace_seq_capacity_hint=workspace_seq_capacity_hint
    )
    # K14 (fork :1366-1391): with the fused decode switched off or not built,
    # nothing here touches counters or capture state.
    fused_ok = (
        fused_decode_eligible(q, k_cache, sinks, kv_cache_dtype)
        and fused_decode_available()
    )
    counters = (
        _fused_decode_counters(q.device, batch_capacity * k_cache.shape[2])
        if fused_ok
        else None
    )
    if fused_ok and _cuda_graph_capture_active():
        _fused_decode_capture_counts["two_launch" if counters is None else "fused"] += 1
    tmp_out, max_logits, exp_sums, active_num_partitions = (
        _get_decode_workspace_for_plan(
            q,
            batch_capacity=batch_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            plan=plan,
            active_num_partitions=active_num_partitions,
        )
    )
    args = (
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        active_num_partitions,
        float(softmax_scale),
        plan.partition_size,
        plan.launch_num_partitions,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        window_size_left,
        window_size_right,
    )
    if sinks is None:
        _op("fv100_decode_paged")(*args)
    elif counters is not None:
        if _fused_decode_zero_counters_enabled():
            # Optional belt and braces: one memset per call (captured too).
            counters[: batch_capacity * k_cache.shape[2]].zero_()
        _op(FUSED_DECODE_OP)(*args, sinks, counters)
    else:
        _op("fv100_decode_paged_sinks")(*args, sinks)
    return out


# fork flash_attn_interface.py:1616-1811
def flash_attn_decode_paged_xqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    window_size: tuple = (-1, -1),
    max_seq_len_hint: int | None = None,
    workspace_seq_capacity_hint: int | None = None,
    active_num_partitions: torch.Tensor | None = None,
    partition_size_hint: int | None = None,
    batch_context_routing: bool = False,
) -> torch.Tensor:
    """Tensor-core XQA decode for D=256, GQA 4/6/8, fp16 KV, no window/sinks.

    `batch_context_routing` is accepted for the fork signature; the lean
    build has no batch-context route, so it only sets the (unused) hint.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    if out is None:
        out = torch.empty_like(q)
    window_size_left, window_size_right = _window(window_size)
    batch_capacity, num_heads, head_dim = q.shape
    plan = _get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len_hint,
        batch_size_hint=batch_capacity,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        active_num_partitions=active_num_partitions,
        partition_size_hint=partition_size_hint,
    )
    _assert_decode_launch_covers_seq_lens(
        plan, seq_lens, workspace_seq_capacity_hint=workspace_seq_capacity_hint
    )
    tmp_out, max_logits, exp_sums, active_num_partitions = (
        _get_decode_workspace_for_plan(
            q,
            batch_capacity=batch_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            plan=plan,
            active_num_partitions=active_num_partitions,
        )
    )
    batch_context_max_seq_len = 0
    if batch_context_routing:
        batch_context_max_seq_len = int(
            max_seq_len_hint
            or workspace_seq_capacity_hint
            or plan.launch_num_partitions * plan.partition_size
        )
    _op("fv100_decode_paged_xqa")(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        active_num_partitions,
        float(softmax_scale),
        plan.partition_size,
        plan.launch_num_partitions,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        window_size_left,
        window_size_right,
        batch_context_max_seq_len,
    )
    return out


def xqa_supported(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Shapes flash_attn_decode_paged_xqa accepts (fork routing.py:133-135)."""
    if q.dim() != 3 or k_cache.dim() != 4 or k_cache.shape[2] <= 0:
        return False
    if q.shape[2] != 256 or q.shape[1] % k_cache.shape[2] != 0:
        return False
    return q.shape[1] // k_cache.shape[2] in XQA_GQA_RATIOS


# ---------------------------------------------------------------------------
# B2: DFlash2 verify attention (verify_decode.cu, new code)
# ---------------------------------------------------------------------------
VERIFY_OP = "fv100_decode_verify"
VERIFY_QUERY_LEN = 8  # tokens per request (DFlash2 k = 7 plus the anchor)
VERIFY_PARTITION_SIZE = 128  # keys per CTA (verify_decode.cu kPart)
VERIFY_GQA_RATIOS = (6, 8)
VERIFY_MAX_PARTITIONS = 12000  # verify_decode.cu kMaxPartitions
# Bytes of verify workspace allowed per (device, stream). The workspace is
# rows x H_q x partitions x (256 fp16 partial + 2 fp32 stats) = 520 B per
# (row, head, partition), partitions = ceil(max_model_len / 128), exact (not
# rounded to a power of two). Qwen3.8 TP2 (12 q heads): 1.6 MB per row at a
# 32768 envelope (cap 168 rows = 21 requests), 7.3 MB per row at 150000
# (cap 32 rows = 4 requests). Larger verify batches keep K07's XQA path.
VERIFY_WORKSPACE_BUDGET = 256 * 2**20


def verify_max_rows(num_heads: int, head_dim: int, max_seq_len: int) -> int:
    """Most verify rows (whole requests) whose workspace fits the budget."""
    parts = max(1, -(-int(max_seq_len) // VERIFY_PARTITION_SIZE))
    per_row = num_heads * parts * (head_dim * 2 + 2 * 4)
    rows = VERIFY_WORKSPACE_BUDGET // max(per_row, 1)
    return rows - rows % VERIFY_QUERY_LEN


def verify_supported(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Shapes flash_attn_decode_verify accepts (besides the row layout)."""
    if not has_op(VERIFY_OP) or not xqa_supported(q, k_cache):
        return False
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16:
        return False
    if q.shape[0] % VERIFY_QUERY_LEN:
        return False
    return q.shape[1] // k_cache.shape[2] in VERIFY_GQA_RATIOS


def flash_attn_decode_verify(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    max_seq_len_hint: int | None = None,
    workspace_seq_capacity_hint: int | None = None,
) -> torch.Tensor:
    """DFlash2 verify rows, KV read once per request (B2).

    q / out [rows, H_q, 256] with rows = requests * 8, token-major (row =
    request * 8 + token); seq_lens [rows] and block_table [rows, blocks] are
    K07's expansion (visible length per token; the 8 rows of a request share
    one table row). The grid covers max_seq_len_hint keys, or the
    workspace_seq_capacity_hint envelope under CUDA-graph capture.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if out is None:
        out = torch.empty_like(q)
    seq_hint = int(max_seq_len_hint or block_table.shape[1] * k_cache.shape[1])
    capacity = int(workspace_seq_capacity_hint or seq_hint)
    part = VERIFY_PARTITION_SIZE
    launch = max(1, -(-max(seq_hint, capacity) // part))
    if launch > VERIFY_MAX_PARTITIONS:
        raise ValueError(
            f"{launch} verify partitions exceed {VERIFY_MAX_PARTITIONS} "
            f"(max_seq_len {max(seq_hint, capacity)})"
        )
    plan = _DecodePlan(
        partition_size=part,
        actual_num_partitions=max(1, -(-seq_hint // part)),
        launch_num_partitions=launch,
        workspace_num_partitions=launch,
    )
    # Exact partition capacity (the budget above assumes it) and no
    # active_num_partitions refresh (the verify kernel does not read it).
    tmp_out, max_logits, exp_sums, _ = _get_decode_workspace_for_plan(
        q,
        batch_capacity=q.shape[0],
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        plan=plan,
        round_partitions=False,
        refresh_active=False,
    )
    _op(VERIFY_OP)(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        float(softmax_scale),
        VERIFY_QUERY_LEN,
        launch,
    )
    return out


# fork flash_attn_interface.py:1950-2100 (DFlash2 split route dropped)
def flash_attn_prefill_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool = True,
    window_size: tuple = (-1, -1),
    *,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged prefill, q [B, M, H, D] -> out [B, M, H, D].

    The M queries of each row are the last M tokens of its sequence
    (seq_lens[b] includes them; causal masking is bottom-right aligned).
    `sinks` is fp32 [H]; with sinks every head size takes the generic kernel.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if sinks is not None:
        sinks = _validate_prefill_sinks(q, sinks)
    window_size_left, window_size_right = _window(window_size)
    out_original = out
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    q_bhmd = q.permute(0, 2, 1, 3).contiguous()
    out_bhmd = torch.empty_like(q_bhmd)
    args = (
        out_bhmd,
        q_bhmd,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        float(softmax_scale),
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        bool(causal),
        window_size_left,
        window_size_right,
    )
    if sinks is None:
        _op("fv100_prefill_paged")(*args)
    else:
        _op("fv100_prefill_paged_sinks")(*args, sinks)
    return _copy_bhmd_to_bmhd_out(out_bhmd, out_original)


# ---------------------------------------------------------------------------
# Fake impls: every fv100 op is an out op with no return value.
# ---------------------------------------------------------------------------


if has_op("fv100_decode_paged"):

    @register_fake("_sm70_C::fv100_decode_paged")
    def _fv100_decode_paged_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tmp_out: torch.Tensor,
        max_logits: torch.Tensor,
        exp_sums: torch.Tensor,
        active_num_partitions: torch.Tensor,
        softmax_scale: float,
        partition_size: int,
        launch_num_partitions: int,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size_left: int,
        window_size_right: int,
    ) -> None:
        return None


if has_op("fv100_decode_paged_sinks"):

    @register_fake("_sm70_C::fv100_decode_paged_sinks")
    def _fv100_decode_paged_sinks_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tmp_out: torch.Tensor,
        max_logits: torch.Tensor,
        exp_sums: torch.Tensor,
        active_num_partitions: torch.Tensor,
        softmax_scale: float,
        partition_size: int,
        launch_num_partitions: int,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size_left: int,
        window_size_right: int,
        sinks: torch.Tensor | None,
    ) -> None:
        return None


if has_op("fv100_decode_paged_sinks_fused"):

    @register_fake("_sm70_C::fv100_decode_paged_sinks_fused")
    def _fv100_decode_paged_sinks_fused_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tmp_out: torch.Tensor,
        max_logits: torch.Tensor,
        exp_sums: torch.Tensor,
        active_num_partitions: torch.Tensor,
        softmax_scale: float,
        partition_size: int,
        launch_num_partitions: int,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size_left: int,
        window_size_right: int,
        sinks: torch.Tensor,
        counters: torch.Tensor,
    ) -> None:
        return None


if has_op("fv100_decode_paged_xqa"):

    @register_fake("_sm70_C::fv100_decode_paged_xqa")
    def _fv100_decode_paged_xqa_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tmp_out: torch.Tensor,
        max_logits: torch.Tensor,
        exp_sums: torch.Tensor,
        active_num_partitions: torch.Tensor,
        softmax_scale: float,
        partition_size: int,
        launch_num_partitions: int,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        window_size_left: int,
        window_size_right: int,
        batch_context_max_seq_len: int,
    ) -> None:
        return None


if has_op("fv100_prefill_paged"):

    @register_fake("_sm70_C::fv100_prefill_paged")
    def _fv100_prefill_paged_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
    ) -> None:
        return None


if has_op("fv100_prefill_paged_sinks"):

    @register_fake("_sm70_C::fv100_prefill_paged_sinks")
    def _fv100_prefill_paged_sinks_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        sinks: torch.Tensor,
    ) -> None:
        return None


if has_op(VERIFY_OP):

    @register_fake("_sm70_C::fv100_decode_verify")
    def _fv100_decode_verify_fake(
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tmp_out: torch.Tensor,
        max_logits: torch.Tensor,
        exp_sums: torch.Tensor,
        softmax_scale: float,
        query_len: int,
        launch_num_partitions: int,
    ) -> None:
        return None


# Every fv100 op takes tensors, so none is exempt from a fake (see tm.py).
NO_FAKE_OPS: frozenset[str] = frozenset()
