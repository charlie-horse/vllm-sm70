# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint NVFP4 MoE M<=2 QPN decode (K12): wrappers, fakes and admission.

Two `_sm70_C` out ops (sm70/csrc/flint_moe/nvfp4_moe_decode_qpn_sm70.cu):

* ``nvfp4_moe_decode_w13_swiglu_sm70_out``: W13 + SwiGLU for every routed
  slot, straight from the TurboMind-packed bank and device expert IDs (no
  permute), split-K over 16 CTAs joined by integer arrival counters.
* ``nvfp4_moe_decode_w2_reduce_sm70_out``: W2 + the top-k weighted sum, one
  FP16 [M, H] row per token.

Ported from fork cf9492a: wrappers vllm/_sm70/moe.py:345-389, fakes
:450-481, admission and workspace model_executor/layers/quantization/
nvfp4_sm70_moe.py:120-130 and :180-290, both without the single-launch
(persistent) variant and without ``native_policy``.

The kernels are compiled for one shape only (C++ ``kFlint*``): Flint at TP2,
i.e. 256 experts, hidden 2048, local intermediate 256 (moe_intermediate 512
/ 2), top-6, M in {1, 2}. Everything else (Flint TP1, Qwen3.6 top-8, M>2)
must stay on the dense-stage route; ``contract_rejection`` says so at load.

CUDA graphs (risk R4): the only state is the split-K workspace, ``partials``
(FP32, scratch) and ``counters`` (int32, zero between launches; the last CTA
of each (route, tile) re-arms its counter, so a replay needs no reset).
``allocate_workspace`` must run at load, before any capture, and refuses
inside one. Nothing here allocates lazily; ``flint_moe_decode_out`` never
allocates.

Kill switch: ``VLLM_SM70_FLINT_MOE_QPN_DECODE=0`` (registered in
vllm_sm70/envs.py, mirrored by Sm70Config.flint_moe_qpn_decode). It is read
by ``admit_layer`` at load, where the decode path is chosen per layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import torch

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

try:
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover
    FakeTensor = None

logger = logging.getLogger(__name__)

NAMESPACE = "_sm70_C"

# Load _sm70_C before any has_op() check below (see ops/tm.py).
from vllm_sm70 import _ext  # noqa: E402

_ext.load()

W13_OP: Final = "nvfp4_moe_decode_w13_swiglu_sm70_out"
W2_OP: Final = "nvfp4_moe_decode_w2_reduce_sm70_out"
OPS: Final = (W13_OP, W2_OP)
KILL_SWITCH: Final = "VLLM_SM70_FLINT_MOE_QPN_DECODE"


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def _op(name: str):
    if not has_op(name):
        raise RuntimeError(
            f"SM70 Flint MoE decode op {NAMESPACE}::{name} is not available. "
            "Build vllm_sm70._sm70_C with CUDA 12.x for sm_70."
        )
    return getattr(getattr(torch.ops, NAMESPACE), name)


def available() -> bool:
    """Both K12 decode ops are present in the loaded _sm70_C."""
    return all(has_op(name) for name in OPS)


# ---------------------------------------------------------------------------
# The compiled contract (C++ kFlint* constants, nvfp4_moe_decode_qpn_sm70.cu)
# ---------------------------------------------------------------------------

# (experts, hidden, local intermediate, top-k); fork nvfp4_sm70_moe.py:124.
FLINT_TP2: Final = (256, 2048, 256, 6)
CONTRACTS: Final = frozenset({FLINT_TP2})
MAX_TOKENS: Final = 2  # kFlintMaxTokens
# Split-K workspace sizes in elements (kFlintPartials, kFlintCounters). The
# fork allocated 96 + 32 counters; the 32 were the single-launch flags.
DECODE_PARTIALS: Final = 98304
DECODE_COUNTERS: Final = 96
# The kernels form prepared_scale * 2^(14 - shift) in FP16.
MAX_SCALE_SHIFT: Final = 14
_FP16_MAX: Final = 65504.0


def enabled() -> bool:
    """False only when VLLM_SM70_FLINT_MOE_QPN_DECODE is "0"."""
    from vllm_sm70 import envs

    return bool(getattr(envs, KILL_SWITCH))


def contract_rejection(contract: tuple[int, int, int, int]) -> str | None:
    """Why (experts, hidden, local_intermediate, top_k) cannot use the kernels.

    The local intermediate is per TP rank: Flint is (256, 2048, 256, 6) at
    TP2 only; TP1 is (256, 2048, 512, 6) and is rejected, as is Qwen3.6
    (top-8). None means admitted.
    """
    contract = tuple(int(v) for v in contract)
    if contract in CONTRACTS:
        return None
    return (
        f"(experts, hidden, local_intermediate, top_k)={contract} is not the "
        f"compiled Flint TP2 contract {FLINT_TP2}"
    )


def scale_shift(scales: torch.Tensor) -> int | None:
    """Power-of-two pre-shift that keeps the decode scale product finite.

    fork nvfp4_sm70_moe.py:180-195 (_decode_qpn_scale_shift). The kernels
    expand E2M1 codes as code * 2^-14 and multiply by the prepared FP16 scale
    times 2^(14 - shift); the FP32 accumulator is then multiplied by 2^shift.
    Both factors are exact, so the weights stay e2m1 * e4m3 * global. Returns
    None when no shift fits (non-finite or out-of-range scales). Reads the
    scales back to the host: call it at load, never in forward.
    """
    peak = float(scales.detach().abs().amax().float())
    if not peak < float("inf"):
        return None
    for shift in range(MAX_SCALE_SHIFT + 1):
        if peak * 2.0 ** (14 - shift) <= _FP16_MAX:
            return shift
    return None


def bank_rejection(
    w13_weight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scales: torch.Tensor,
) -> str | None:
    """Host mirror of the C++ bank guards (check_bank/check_w13/check_w2).

    Banks are the stacked tm.nvfp4_sm70_prepare outputs: W13 int32
    [E, H, 2I/8] + FP16 [E, H/16, 2I] (gate columns first, not interleaved),
    W2 int32 [E, I, H/8] + FP16 [E, I/16, H]; contiguous, one device.
    """
    e, h, i, _ = FLINT_TP2
    want = (
        ("w13_weight", w13_weight, torch.int32, (e, h, 2 * i // 8)),
        ("w13_scales", w13_scales, torch.float16, (e, h // 16, 2 * i)),
        ("w2_weight", w2_weight, torch.int32, (e, i, h // 8)),
        ("w2_scales", w2_scales, torch.float16, (e, i // 16, h)),
    )
    for name, t, dtype, shape in want:
        if t.dtype != dtype or tuple(t.shape) != shape:
            return f"{name} is {t.dtype} {tuple(t.shape)}, expected {dtype} {shape}"
        if not t.is_contiguous():
            return f"{name} is not contiguous"
        if t.device != w13_weight.device:
            return f"{name} is on {t.device}, not {w13_weight.device}"
    return None


@dataclass(frozen=True)
class DecodePlan:
    """Per-layer admission result: the two load-time scale pre-shifts."""

    w13_scale_shift: int
    w2_scale_shift: int


def admit_layer(
    contract: tuple[int, int, int, int],
    w13_weight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scales: torch.Tensor,
    *,
    policy_enabled: bool | None = None,
) -> DecodePlan | None:
    """Load-time admission (fork nvfp4_sm70_moe.py:218-290, two-kernel form).

    Returns the plan, or None (logged) when the layer must keep the dense
    stage: kill switch off, ops absent, shape outside the compiled contract,
    or scales with no FP16 headroom. ``policy_enabled`` is the caller's
    resolved switch (Sm70Config.flint_moe_qpn_decode); None reads the env.
    """
    if not (enabled() if policy_enabled is None else policy_enabled):
        logger.debug("SM70 Flint MoE decode off (%s=0).", KILL_SWITCH)
        return None
    if not available():
        logger.debug("SM70 Flint MoE decode ops absent; dense stage serves M<=2.")
        return None
    reason = contract_rejection(contract) or bank_rejection(
        w13_weight, w13_scales, w2_weight, w2_scales
    )
    if reason is not None:
        logger.debug("SM70 Flint MoE decode not admitted: %s.", reason)
        return None
    w13_shift = scale_shift(w13_scales)
    w2_shift = scale_shift(w2_scales)
    if w13_shift is None or w2_shift is None:
        logger.warning(
            "SM70 Flint MoE decode: a prepared expert scale is not finite or "
            "exceeds the FP16 decode range even after a 2^%d pre-shift; this "
            "layer keeps the dense stage.",
            MAX_SCALE_SHIFT,
        )
        return None
    if w13_shift or w2_shift:
        logger.info(
            "SM70 Flint MoE decode: exact power-of-two scale pre-shifts "
            "W13 2^%d, W2 2^%d.",
            w13_shift,
            w2_shift,
        )
    return DecodePlan(w13_shift, w2_shift)


# ---------------------------------------------------------------------------
# Workspace (R4: allocated at load, never under capture)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeWorkspace:
    """Split-K scratch for one layer (or several layers on one stream).

    partials: FP32 [DECODE_PARTIALS], no initial value needed.
    counters: int32 [DECODE_COUNTERS], zero between launches.
    ids: int32 [MAX_TOKENS * top_k], scratch for int64 top-k IDs.
    Sharing one workspace between layers is safe only while they run one
    after another on one stream (the counters are re-armed per launch).
    """

    partials: torch.Tensor
    counters: torch.Tensor
    ids: torch.Tensor


def allocate_workspace(device: torch.device | str) -> DecodeWorkspace:
    """Allocate and zero the decode workspace. Call at load/warmup only.

    Refuses inside CUDA graph capture: a workspace created there would be
    zeroed only in the graph being captured, and every other graph would
    replay on whatever the counters held (the C4 `--max-num-seqs 20`
    gibberish, fork docs/design/sm70_flint_max_num_seqs_postmortem.md).
    """
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SM70 Flint MoE decode: allocate_workspace() called during CUDA "
            "graph capture; allocate it in process_weights_after_loading."
        )
    return DecodeWorkspace(
        partials=torch.empty(DECODE_PARTIALS, dtype=torch.float32, device=device),
        counters=torch.zeros(DECODE_COUNTERS, dtype=torch.int32, device=device),
        ids=torch.zeros(MAX_TOKENS * FLINT_TP2[3], dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Per-call admission (metadata only: no device sync, safe under capture)
# ---------------------------------------------------------------------------


def _is_fake(t: torch.Tensor) -> bool:
    return t.device.type == "meta" or (
        FakeTensor is not None and isinstance(t, FakeTensor)
    )


def call_rejection(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> str | None:
    """Why this forward call cannot take the decode ops (None: it can).

    The C++ guards, checked on the host: M in {1, 2}; x FP16 [M, 2048],
    contiguous and 16-byte aligned (uint4 row loads); topk_weights FP32
    [M, 6] contiguous; topk_ids int32 or int64 [M, 6] (int64 is copied into
    the workspace by flint_moe_decode_out).
    """
    _, hidden, _, top_k = FLINT_TP2
    if x.dim() != 2 or not 0 < x.shape[0] <= MAX_TOKENS:
        return f"x shape {tuple(x.shape)} is not [M<={MAX_TOKENS}, {hidden}]"
    tokens = x.shape[0]
    if x.shape[1] != hidden or x.dtype != torch.float16 or not x.is_contiguous():
        return f"x must be contiguous FP16 [M, {hidden}]"
    if not _is_fake(x) and x.data_ptr() % 16:
        return "x is not 16-byte aligned"
    if tuple(topk_weights.shape) != (tokens, top_k) or tuple(topk_ids.shape) != (
        tokens,
        top_k,
    ):
        return f"top-k tensors must be [{tokens}, {top_k}]"
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        return "topk_weights must be contiguous FP32"
    if topk_ids.dtype not in (torch.int32, torch.int64):
        return f"topk_ids dtype {topk_ids.dtype} is not int32/int64"
    return None


def can_decode(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> bool:
    return call_rejection(x, topk_weights, topk_ids) is None


# ---------------------------------------------------------------------------
# Ops: fork vllm/_sm70/moe.py
# ---------------------------------------------------------------------------


# fork vllm/_sm70/moe.py:345-367
def nvfp4_moe_decode_w13_swiglu_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    scale_shift: int,
) -> None:
    _op("nvfp4_moe_decode_w13_swiglu_sm70_out")(
        out,
        input,
        weights,
        scales,
        expert_ids,
        partials,
        counters,
        scale_shift,
    )


# fork vllm/_sm70/moe.py:370-389
def nvfp4_moe_decode_w2_reduce_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    scale_shift: int,
) -> None:
    _op("nvfp4_moe_decode_w2_reduce_sm70_out")(
        out,
        input,
        weights,
        scales,
        expert_ids,
        topk_weights,
        scale_shift,
    )


# fork vllm/_sm70/moe.py:450-464
if has_op(W13_OP):

    @register_fake("_sm70_C::nvfp4_moe_decode_w13_swiglu_sm70_out")
    def _nvfp4_moe_decode_w13_swiglu_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        partials: torch.Tensor,
        counters: torch.Tensor,
        scale_shift: int,
    ) -> None:
        return None


# fork vllm/_sm70/moe.py:467-481
if has_op(W2_OP):

    @register_fake("_sm70_C::nvfp4_moe_decode_w2_reduce_sm70_out")
    def _nvfp4_moe_decode_w2_reduce_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        scale_shift: int,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Composite (fork fused_moe/sm70/fp4_codec.py:158-167 + :252-260, the
# DECODE_QPN -> DECODE_REDUCE plan of nvfp4_sm70_moe.py:1443-1450)
# ---------------------------------------------------------------------------


def flint_moe_decode_out(
    out: torch.Tensor,
    intermediate: torch.Tensor,
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace: DecodeWorkspace,
    plan: DecodePlan,
) -> torch.Tensor:
    """Routed-expert output of an admitted layer for M <= 2 tokens.

    out: FP16 [M, 2048] contiguous, fully overwritten (no zero fill needed).
    intermediate: FP16 [M * 6, 256] contiguous scratch (SwiGLU activations).
    Banks: as bank_rejection. topk_weights FP32 [M, 6]; topk_ids int32 or
    int64 [M, 6]; IDs outside [0, 256) contribute nothing. Returns ``out``:
    the local, un-reduced sum over the top-k (the runner all-reduces).
    Callers check ``call_rejection`` first; this does no allocation, so it
    is safe inside CUDA graph capture.
    """
    routes = x.shape[0] * FLINT_TP2[3]
    if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
        ids = topk_ids.view(-1)
    else:
        ids = workspace.ids[:routes]
        ids.copy_(topk_ids.reshape(-1))
    nvfp4_moe_decode_w13_swiglu_sm70_out(
        intermediate,
        x,
        w13_weight,
        w13_scales,
        ids,
        workspace.partials,
        workspace.counters,
        plan.w13_scale_shift,
    )
    nvfp4_moe_decode_w2_reduce_sm70_out(
        out,
        intermediate,
        w2_weight,
        w2_scales,
        ids,
        topk_weights,
        plan.w2_scale_shift,
    )
    return out
