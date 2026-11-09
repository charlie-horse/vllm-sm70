# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K18: DFlash2 draft FP16 M<=8 GEMM and compact FP16 top-64 (_sm70_C ops).

What: the DFlash2 draft's unquantized projections (qkv, o, gate_up, down and
  the two conv kernel projections per layer) run a split-K mma.m8n8k4 GEMM
  with FP32 accumulation when M (tokens) is 1..8; every other call keeps
  upstream's ``F.linear``. The weight is repacked once after load into a
  second, non-persistent buffer (the original stays for M>8 and prefill).
How it hooks in: patch U1. ``Sm70CudaPlatform.get_unquantized_gemm`` returns
  :func:`dflash2_unquantized_gemm` (via :func:`select_unquantized_gemm`); it
  serves only layers that :func:`mark_draft_linears` flagged and
  :func:`prepare_dflash2_fp16_m8` packed, and calls ``F.linear`` (UP
  ``default_unquantized_gemm``) for everything else. M7's DFlash2 draft
  subclass marks the layers at construction and packs them in load_weights.
Switch: ``VLLM_SM70_DFLASH2_FP16``: "0" off, "1" forced on, anything else = auto
  (:func:`k18_wanted`: on iff the DFlash2 verifier resolves on, i.e. the
  Qwen3.8 + DFlash2 k=7 contract; M9 decision 2026-11-09). Off:
  ``get_unquantized_gemm`` returns None (upstream dispatch, untouched),
  nothing is marked or packed, F.linear runs.
Selection is decided once: ``select_unquantized_gemm`` is cached per
  linear_backend (UP calls ``dispatch_unquantized_gemm()`` on every forward of
  lm_head and the MoE router gate), and each layer's route (N, K, tile, warps)
  is fixed when it is packed (``ROUTE_ATTR``); a call only checks ``x``.

Ported from fork cf9492a vllm/model_executor/layers/quantization/
sm70_dflash2_fp16.py and vllm/model_executor/layers/sm70_compact_topk.py;
see sm70/csrc/dflash2/PROVENANCE.md for the line map and what changed.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

from vllm_sm70 import envs

NAMESPACE = "_sm70_C"
KILL_SWITCH = "VLLM_SM70_DFLASH2_FP16"

# Load _sm70_C before any has_op() check below (see ops/tm.py).
from vllm_sm70 import _ext  # noqa: E402

_ext.load()

from vllm_sm70.ops import dflash2_fp8  # noqa: E402  (M9a, rides on this hook)

logger = logging.getLogger("vllm.sm70")

GEMM_OPS = ("sm70_dflash2_fp16_m8_out", "sm70_dflash2_fp16_dispatch_out")
TOPK_OPS = ("sm70_compact_half_topk_out",)
OPS = GEMM_OPS + TOPK_OPS

MARK_ATTR = "_sm70_dflash2_fp16_m8"
PACKED_ATTR = "_sm70_dflash2_fp16_packed"
GEOMETRY_ATTR = "_sm70_dflash2_fp16_geometry"
# (N, K, tile, warps), set only on a packed layer: the per-layer decision.
ROUTE_ATTR = "_sm70_dflash2_fp16_route"

# The kernel runs on exactly 8 rows; M 1..7 is zero-padded inside the op.
MAX_M = 8

# (N, K) of the local weight shard -> (tile, warps). Admission is by shape:
# a marked layer whose shard is not listed keeps F.linear.
#
# ported from FORK sm70_dflash2_fp16.py:14-20, which held the five Qwen3.8-27B
# TP4 shards measured at M8. Only (1280, 5120) (the conv kernel projection,
# replicated, so the same at every TP size) is kept from it; the four TP4-only
# entries are dropped (PROVENANCE.md). The TP2 entries are new and their
# geometry is not yet measured: chosen by the fork's rule of thumb (tile 16 x
# 16 warps when N/32 < 80 SMs, else tile 32; 16 warps for K >= 5120, else 8),
# which keeps 4-34 K16 groups per split. Bea's bench tunes them.
DRAFT_SHAPES_TP2: dict[str, dict[str, tuple[int, int]]] = {
    # hidden 5120, 32 q / 8 kv heads x 128, intermediate 17408, conv
    # kernel 2 x group 16 -> kernel_projection 2*2*(5120/16) = 1280.
    "qwen3.8-27b-dflash2": {
        "qkv_proj": (3072, 5120),
        "o_proj": (5120, 2048),
        "gate_up_proj": (17408, 5120),
        "down_proj": (5120, 8704),
        "kernel_projection": (1280, 5120),
    },
    # hidden 2048, 32 q / 8 kv heads x 128, intermediate 6144, conv
    # kernel 2 x group 16 -> kernel_projection 2*2*(2048/16) = 512.
    "qwen3.6-35b-a3b-dflash2": {
        "qkv_proj": (3072, 2048),
        "o_proj": (2048, 2048),
        "gate_up_proj": (6144, 2048),
        "down_proj": (2048, 3072),
        "kernel_projection": (512, 2048),
    },
}

GEOMETRY: dict[tuple[int, int], tuple[int, int]] = {
    # measured in the fork (Qwen3.8 draft, M8)
    (1280, 5120): (16, 16),
    # Qwen3.8 draft, TP2 (provisional)
    (3072, 5120): (32, 16),
    (5120, 2048): (32, 8),
    (17408, 5120): (32, 16),
    (5120, 8704): (32, 16),
    # Qwen3.6-35B-A3B draft, TP2 (provisional)
    (3072, 2048): (32, 8),
    (2048, 2048): (16, 16),
    (6144, 2048): (32, 8),
    (2048, 3072): (16, 16),
    (512, 2048): (16, 16),
}


def geometry_valid(n: int, k: int, tile: int, warps: int) -> bool:
    """The C++ shape checks of sm70_dflash2_fp16_m8_out, on the CPU."""
    if (tile, warps) not in ((16, 16), (32, 8), (32, 16)):
        return False
    split = warps * (2 if tile == 16 else 1)
    return n > 0 and k > 0 and n % tile == 0 and k % (16 * split) == 0


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def gemm_available() -> bool:
    return all(has_op(name) for name in GEMM_OPS)


# The auto decision from vLLM's current config, resolved once per process
# (UP calls the U1 hook on every lm_head / router-gate forward, where no config
# may be current). reset_selection_cache() clears it.
_auto_decision: bool | None = None


def k18_wanted(vllm_config: Any = None) -> bool:
    """The K18 switch. "0" -> False, "1" -> True, otherwise auto: True iff
    the DFlash2 verifier resolves on for ``vllm_config`` (default: vLLM's
    current config, decided once and cached; none current -> False, not
    cached). Auto reuses M9a's gate (dflash2_fp8.draft_fp8_wanted), so K18 and
    M9a turn on together and VLLM_SM70_DFLASH2_VERIFIER=0 turns both off."""
    global _auto_decision
    value = envs.VLLM_SM70_DFLASH2_FP16
    if value is not None:
        return value
    if vllm_config is not None:
        return _auto_from(vllm_config)
    if _auto_decision is not None:
        return _auto_decision
    try:
        from vllm.config import get_current_vllm_config_or_none
    except ImportError:  # pragma: no cover - vllm layout changed
        return False
    current = get_current_vllm_config_or_none()
    if current is None:
        return False
    _auto_decision = _auto_from(current)
    return _auto_decision


def _auto_from(vllm_config: Any) -> bool:
    try:
        return bool(dflash2_fp8.draft_fp8_wanted(vllm_config))
    except Exception:
        logger.warning(
            "vllm_sm70: K18 auto check failed; DFlash2 draft FP16 GEMM off",
            exc_info=True,
        )
        return False


def enabled(vllm_config: Any = None) -> bool:
    """K18 wanted (:func:`k18_wanted`) and the GEMM ops built."""
    return k18_wanted(vllm_config) and gemm_available()


def _is_sm70(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_capability(device) == (7, 0)
    except Exception:
        return False


# --------------------------------------------------------------------------
# GEMM: U1 hook
# --------------------------------------------------------------------------
def dflash2_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """U1 GEMM for DFlash2 draft layers: M9a's channel-FP8 route (M 8..32),
    else the M<=8 FP16 kernel, else F.linear (FORK linear.py:396-410 order)."""
    out = dflash2_fp8.apply_dflash2_fp8_m8(layer, x, bias)
    if out is not None:
        return out
    out = apply_dflash2_fp16_m8(layer, x, bias)
    if out is not None:
        return out
    # UP model_executor/layers/utils.py default_unquantized_gemm
    return F.linear(x, weight, bias)


def select_unquantized_gemm(linear_backend: str = "auto") -> Callable[..., Any] | None:
    """What Sm70CudaPlatform.get_unquantized_gemm returns (None -> upstream).

    Called when each UnquantizedLinearMethod is built (UP linear.py:176) and
    on every forward of lm_head (UP vocab_parallel_embedding.py:83) and the
    MoE router gate (UP moe_runner.py:910), so the answer is cached per
    linear_backend for the process. reset_selection_cache() clears it.
    None unless K18 is wanted (:func:`k18_wanted`: VLLM_SM70_DFLASH2_FP16=1,
    or auto with the Qwen3.8 DFlash2 verifier contract) and the ops are built,
    and for an explicit FlashInfer BF16 backend. Logs once when it selects.
    """
    return _select_cached(linear_backend, k18_wanted())


def reset_selection_cache() -> None:
    global _auto_decision
    _auto_decision = None
    _select_cached.cache_clear()


@functools.cache
def _select_cached(linear_backend: str, wanted: bool) -> Callable[..., Any] | None:
    if not wanted:
        return None
    if not gemm_available():
        return None
    # An explicit FlashInfer BF16 backend keeps upstream's choice (UP
    # layers/utils.py:695); "auto" there is default_unquantized_gemm, which
    # dflash2_unquantized_gemm falls back to.
    try:
        from vllm.model_executor.layers import utils as up_utils

        if linear_backend in getattr(up_utils, "_FLASHINFER_BF16_BACKENDS", {}):
            return None
    except Exception:  # pragma: no cover - vllm layout changed
        return None
    logger.info(
        "vllm_sm70: K18 DFlash2 draft GEMM dispatcher selected (%s)",
        "VLLM_SM70_DFLASH2_FP16=1"
        if envs.VLLM_SM70_DFLASH2_FP16
        else "auto: DFlash2 verifier contract",
    )
    return dflash2_unquantized_gemm


def mark_draft_linears(
    decoder_layer: torch.nn.Module, *, fp8: bool = False, vllm_config: Any = None
) -> list[torch.nn.Module]:
    """Flag one DFlash2 draft decoder layer's unquantized projections.

    ported from FORK qwen3_dflash2.py:280-298, without its TP==4 condition
    (admission is by shard shape, GEOMETRY). ``fp8`` also flags them for
    M9a's channel-FP8 route (the caller decides: verifier on and k == 7,
    FORK :295-297). Returns the flagged modules.
    """
    if not k18_wanted(vllm_config):
        return []
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    attn = getattr(decoder_layer, "self_attn", None)
    mlp = getattr(decoder_layer, "mlp", None)
    candidates = [
        getattr(attn, "qkv_proj", None),
        getattr(attn, "o_proj", None),
        getattr(mlp, "gate_up_proj", None),
        getattr(mlp, "down_proj", None),
        getattr(
            getattr(decoder_layer, "attention_conv", None), "kernel_projection", None
        ),
        getattr(getattr(decoder_layer, "mlp_conv", None), "kernel_projection", None),
    ]
    marked = []
    for projection in candidates:
        if projection is None:
            continue
        if isinstance(
            getattr(projection, "quant_method", None), UnquantizedLinearMethod
        ):
            setattr(projection, MARK_ATTR, True)
            if fp8:
                setattr(projection, dflash2_fp8.FP8_MARK_ATTR, True)
            marked.append(projection)
    return marked


@torch.no_grad()
def pack_weight(
    weight: torch.Tensor, tile: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    """[N, K] fp16 -> the kernel's [N/tile, K/16, tile, 16] slot order.

    With ``out`` (same shape/dtype/device, contiguous) the result is written
    into it in place, without allocating a second packed copy.
    """
    # ported from FORK sm70_dflash2_fp16.py:45-56
    n, k = weight.shape
    slots = torch.arange(tile, device=weight.device)
    columns = (
        (slots & 3) | ((slots & 12) << 1) | ((slots & 16) >> 2)
        if tile == 32
        else (slots & 3) | ((slots & 4) << 1) | ((slots & 8) >> 1)
    )
    source = weight.view(n // tile, tile, k // 16, 16).permute(0, 2, 1, 3)
    if out is None:
        out = torch.empty(
            (n // tile, k // 16, tile, 16), dtype=weight.dtype, device=weight.device
        )
    # index_select(out=) writes the gathered slots straight into `out`.
    torch.index_select(source, 2, columns, out=out)
    return out


@torch.no_grad()
def prepare_dflash2_fp16_m8(layer: torch.nn.Module) -> bool:
    """Pack a marked layer's weight after load. True if packed.

    A repeated call (weight reload, sleep/wake) repacks in place into the
    existing buffer: no second copy, and never stale. no_grad (not
    inference_mode), so the buffer stays an ordinary tensor that a later
    in-place repack may write to.
    """
    # ported from FORK sm70_dflash2_fp16.py:23-67
    if not getattr(layer, MARK_ATTR, False):
        return False
    if not enabled():
        return False
    weight = getattr(layer, "weight", None)
    if (
        weight is None
        or weight.ndim != 2
        or weight.dtype != torch.float16
        or not weight.is_contiguous()
        or getattr(layer, "bias", None) is not None
        or not _is_sm70(weight.device)
    ):
        return False
    geometry = GEOMETRY.get(tuple(weight.shape))
    if geometry is None:
        logger.info(
            "vllm_sm70: DFlash2 FP16 M8 not admitted for %s shard N%d/K%d "
            "(not in the shape table); it keeps F.linear.",
            getattr(layer, "prefix", "?"),
            *weight.shape,
        )
        return False
    tile, warps = geometry
    n, k = weight.shape
    if not geometry_valid(n, k, tile, warps):  # pragma: no cover - table bug
        return False
    existing = getattr(layer, PACKED_ATTR, None)
    if (
        existing is not None
        and existing.shape == (n // tile, k // 16, tile, 16)
        and existing.dtype == weight.dtype
        and existing.device == weight.device
        and existing.is_contiguous()
        and getattr(layer, GEOMETRY_ATTR, None) == geometry
    ):
        pack_weight(weight.detach(), tile, out=existing)
        return True
    packed = pack_weight(weight.detach(), tile)
    # Retain the original matrix for prefill, batched drafting, and context KV.
    layer.register_buffer(PACKED_ATTR, packed, persistent=False)
    setattr(layer, GEOMETRY_ATTR, geometry)
    setattr(layer, ROUTE_ATTR, (n, k, tile, warps))
    logger.info(
        "vllm_sm70: DFlash2 FP16 M8 layout prepared: N%d/K%d, tile=%d, warps=%d.",
        n,
        k,
        tile,
        warps,
    )
    return True


@torch.no_grad()
def prepare_marked(modules: Iterable[torch.nn.Module] | torch.nn.Module) -> int:
    """Pack every marked layer under ``modules``; returns how many were packed.

    A layer M9a admits gets the channel-FP8 copy instead of the FP16 M8
    packing (FORK linear.py:351-352). Safe to call again after a weight
    reload: existing buffers are refilled.
    """
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()
    return sum(
        1
        for m in modules
        if dflash2_fp8.prepare_dflash2_fp8_m8(m) or prepare_dflash2_fp16_m8(m)
    )


def apply_dflash2_fp16_m8(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    """The kernel's output, or None when this call keeps F.linear."""
    # ported from FORK sm70_dflash2_fp16.py:70-92
    route = getattr(layer, ROUTE_ATTR, None)  # decided once, at pack time
    if route is None:
        return None
    n, k, tile, warps = route
    if (
        bias is not None
        or x.ndim != 2
        or x.shape[1] != k
        or x.dtype != torch.float16
        or not x.is_contiguous()
    ):
        return None
    # Under torch.compile the op is traced for every M and dispatches at run
    # time (M 1..8: kernel, else mm), so a dynamic trace keeps the decode route.
    if not torch.compiler.is_compiling() and not 1 <= x.shape[0] <= MAX_M:
        return None
    output = x.new_empty((x.shape[0], n))
    torch.ops._sm70_C.sm70_dflash2_fp16_dispatch_out(
        output, x, getattr(layer, PACKED_ATTR), layer.weight, tile, warps
    )
    return output


def mark_and_prepare(layer: torch.nn.Module) -> bool:
    """Flag and pack one linear (tests, tools). True if packed."""
    setattr(layer, MARK_ATTR, True)
    return prepare_dflash2_fp16_m8(layer)


# --------------------------------------------------------------------------
# Compact FP16 top-64
# --------------------------------------------------------------------------
def compact_half_topk(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a sorted top-64 probe; the caller must retain the tie guard.

    This does not implement the draft selector's tie contract. Never cast
    FP32 logits to FP16 to qualify: ranking uses the original FP16 values.
    Returns None (caller uses torch.topk) when the input does not qualify.
    """
    # ported from FORK sm70_compact_topk.py:10-39
    if (
        not logits.is_cuda
        or logits.dtype != torch.float16
        or logits.ndim != 2
        or not 1 <= logits.shape[0] <= 32
        or not 64 <= logits.shape[1] <= 65535
        or logits.stride(1) != 1
        or logits.stride(0) > (2**31 - 1) // logits.shape[0]
        or not _is_sm70(logits.device)
        or not has_op("sm70_compact_half_topk_out")
    ):
        return None
    rows, width = logits.shape
    partial = torch.empty(
        (rows, ((width + 1023) // 1024) * 64),
        dtype=torch.uint32,
        device=logits.device,
    )
    values = torch.empty((rows, 64), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, 64), dtype=torch.int64, device=logits.device)
    torch.ops._sm70_C.sm70_compact_half_topk_out(logits, partial, values, ids, 64)
    return values, ids


# --------------------------------------------------------------------------
# Fakes (only for ops the loaded _sm70_C has)
# --------------------------------------------------------------------------
if has_op("sm70_dflash2_fp16_m8_out"):

    @register_fake("_sm70_C::sm70_dflash2_fp16_m8_out")
    def _fake_m8(output, input, packed, tile, warps) -> None:
        return None


if has_op("sm70_dflash2_fp16_dispatch_out"):

    @register_fake("_sm70_C::sm70_dflash2_fp16_dispatch_out")
    def _fake_dispatch(output, input, packed, weight, tile, warps) -> None:
        return None


if has_op("sm70_compact_half_topk_out"):

    @register_fake("_sm70_C::sm70_compact_half_topk_out")
    def _fake_topk(logits, partial, values, ids, k) -> None:
        return None
