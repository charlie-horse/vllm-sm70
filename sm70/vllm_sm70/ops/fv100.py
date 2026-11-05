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
* No anchored SWA, FP8 KV, staged XQA, DFlash2 split or fused one-launch sink
  decode (K14 adds the fused path).

KV cache views: `k_cache`/`v_cache` are [num_blocks, block_size, H_kv, D]
views with any strides in the first three dims and a contiguous last dim.
`check_kv_cache_views()` checks that on the CPU (metadata only), and
`split_lbnhc_kv_cache()` builds the views from an upstream v0.33.0 KV cache
(LBNHC: K and V interleaved per head, so stride(2) == 2 * D). Wiring the
backend to that layout is K06.
"""

from __future__ import annotations

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
):
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = _workspace_stream_id(q.device)
    # Rows are shared across batch sizes (fork default share_workspace=on).
    partition_capacity = _round_decode_partition_capacity(plan.workspace_num_partitions)
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
            max_num_partitions=_round_decode_partition_capacity(
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
        workspace.active_num_partitions.fill_(plan.actual_num_partitions)
        active_num_partitions = workspace.active_num_partitions
    return (
        workspace.tmp_out[:batch_capacity],
        workspace.max_logits[:batch_capacity],
        workspace.exp_sums[:batch_capacity],
        active_num_partitions,
    )


def clear_workspaces() -> None:
    """Drop cached decode plans and workspaces (tests, engine shutdown)."""
    _decode_plan_cache.clear()
    _decode_workspace_cache.clear()


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
    sinks, GQA-8 D128 layers (Flint) take the grouped two-launch kernel.
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


# Every fv100 op takes tensors, so none is exempt from a fake (see tm.py).
NO_FAKE_OPS: frozenset[str] = frozenset()
