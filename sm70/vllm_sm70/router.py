# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SM70 decode router for the Flint E256/K6 sigmoid MoE (patch K13).

The generic route casts the FP16 hidden states to FP32, runs an FP32 GEMM
against the FP32 gate and selects with ``topk_sigmoid``: several kernels and
an FP32 copy of the activations per MoE layer. For one or two tokens,
``torch.ops._sm70_C.flint_route`` reads the FP16 hidden states and the FP32
gate as loaded, accumulates in FP32 and selects the renormalized sigmoid top
six in the same launch, with ``topk_sigmoid``'s scoring, tie-breaking and
normalization. Only the FP32 summation order of the logits differs.

Contract (U3, patches/0004 from PR #43): when a MoE runner holds the gate,
every forward calls ``current_platform.get_moe_router_provider()``, then
``provider(runner) -> route | None`` and ``route.try_route(hidden_states)
-> (topk_weights fp32, topk_ids int32) | None``. A pair skips the gate GEMM
and ``select_experts`` (UP moe_runner.py:907, :630); monolithic kernels
assert. U3 caches nothing, so this module does: :func:`flint_router_provider`
decides once per layer and stores the route on the runner.
``Sm70CudaPlatform.get_moe_router_provider()`` returns it, and the U1
``kernel_warmup`` hook calls :func:`prepare_model` before CUDA graph capture.

Capture safety (risk R4): the per-layer workspace (FP32 logits) and the
INT32 arrival counter are allocated only when a layer is decided, i.e. at
warmup or on the first *eager* forward. Under capture an undecided layer
gets no route and nothing is cached or allocated. The kernel's last CTA
re-arms the counter, so every graph and every eager call can share one
counter per layer in any order.

Ported from fork cf9492a vllm/model_executor/layers/fused_moe/router/
sm70_flint_router.py (205 lines, last touch 3629e29f6). See
csrc/router/PROVENANCE.md for the function map.

Kill switch: VLLM_SM70_FLINT_FUSED_ROUTER=0 (read once, when the platform
getter is first asked; then the getter returns None and U3 runs upstream).
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import torch

from vllm_sm70 import _ext, envs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

logger = logging.getLogger("vllm.sm70")

NAMESPACE = "_sm70_C"
OP = "flint_route"
KILL_SWITCH = "VLLM_SM70_FLINT_FUSED_ROUTER"

HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 6
MAX_TOKENS = 2
# FP32 logits for MAX_TOKENS rows; the counter is a separate INT32 tensor.
_LOGITS = MAX_TOKENS * NUM_EXPERTS

# Per-layer decision, stored on the runner: FlintFusedRouter or None.
_ROUTE_ATTR = "_sm70_flint_route"
_UNDECIDED = object()

_logged: set[str] = set()


def _log_once(level: int, msg: str, *args: Any) -> None:
    key = msg % args if args else msg
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def has_op() -> bool:
    _ext.load()
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), OP)
    except (AttributeError, RuntimeError):
        return False


def _flint_route_op() -> Any | None:
    """The fused operator, or None when _sm70_C lacks it."""
    return getattr(getattr(torch.ops, NAMESPACE), OP) if has_op() else None


def enabled() -> bool:
    """Kill switch: only VLLM_SM70_FLINT_FUSED_ROUTER=0 disables."""
    return bool(getattr(envs, KILL_SWITCH))


def _capturing() -> bool:
    """True while the current CUDA stream captures (False on CPU builds)."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (RuntimeError, AssertionError):
        return False


def _is_exact_sm70(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (7, 0)


def _register_fake() -> None:
    if not has_op():
        return
    try:
        from torch.library import register_fake
    except ImportError:  # pragma: no cover - torch < 2.4
        from torch.library import impl_abstract as register_fake  # type: ignore

    try:
        # Mutates its outputs only: nothing to compute on meta tensors.
        register_fake(f"{NAMESPACE}::{OP}")(lambda *args: None)
    except RuntimeError:  # already registered (re-import)
        pass


class FlintFusedRouter:
    """Per-layer state: the gate weight, the router and a private workspace.

    The workspace holds the FP32 logits and the arrival counter that the
    kernel's last CTA re-arms; it is per layer so no two launches share it.
    Created only at admission (eager), never during CUDA graph capture.
    """

    def __init__(self, op: Any, weight: torch.Tensor, runner: Any) -> None:
        self.op = op
        self.weight = weight
        self.runner = runner
        self.logits = torch.empty(_LOGITS, dtype=torch.float32, device=weight.device)
        # The kernel requires the counter to be zero before its first launch.
        self.counter = torch.zeros(1, dtype=torch.int32, device=weight.device)

    @staticmethod
    def accepts(hidden_states: torch.Tensor) -> bool:
        """Tensor-metadata gate; fixed for a captured graph."""
        return (
            hidden_states.dim() == 2
            and 1 <= hidden_states.shape[0] <= MAX_TOKENS
            and hidden_states.shape[1] == HIDDEN
            and hidden_states.dtype == torch.float16
            and hidden_states.stride(1) == 1
            and hidden_states.stride(0) % 4 == 0
            and hidden_states.data_ptr() % 8 == 0
        )

    def try_route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Top-k weights and ids, or None to take the generic route.

        Returns what ``select_experts`` would: INT32 logical ids (no EPLB).
        Defers when a routed-id capture or routing-replay hook is set, or
        when the experts want another id dtype.
        """
        if not self.accepts(hidden_states):
            return None
        router = self.runner.router
        if (
            getattr(router, "capture_fn", None) is not None
            or getattr(router, "_routing_replay_out", None) is not None
            or getattr(self.runner._quant_method, "topk_indices_dtype", None)
            not in (None, torch.int32)
        ):
            return None
        return self.route(hidden_states)

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = hidden_states.shape[0]
        topk_weights = torch.empty(
            (tokens, TOP_K), dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            (tokens, TOP_K), dtype=torch.int32, device=hidden_states.device
        )
        self.op(
            topk_weights,
            topk_ids,
            hidden_states,
            self.weight,
            self.logits,
            self.counter,
        )
        return topk_weights, topk_ids


def _rejection(runner: Any) -> str | None:
    """Why the layer cannot take the fused router, or None if it can."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = runner.gate
    router = runner.router
    if not isinstance(gate, GateLinear):
        return "the gate is not a GateLinear"
    weight = getattr(gate, "weight", None)
    if (
        not getattr(gate, "is_unquantized", True)
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.float32
        or tuple(weight.shape) != (NUM_EXPERTS, HIDDEN)
        or not weight.is_contiguous()
        or weight.data_ptr() % 16 != 0
        or getattr(gate, "bias", None) is not None
        or gate.out_dtype != torch.float32
    ):
        return "the gate is not an unbiased FP32 [256, 2048] GEMM"
    if not _is_exact_sm70(weight.device):
        return "the device is not SM70"
    if type(router) is not FusedTopKRouter:
        return "the router is not the plain fused top-k router"
    if (
        router.scoring_func != "sigmoid"
        or not router.renormalize
        or router.top_k != TOP_K
        or router.global_num_experts != NUM_EXPERTS
        or router.eplb_state is not None
    ):
        return "the router is not a renormalized sigmoid top-6 over 256 experts"
    if (
        runner._fse_fuse_gate
        or runner.routed_input_transform is not None
        or runner.enable_dbo
        or runner._quant_method.is_monolithic
    ):
        return "the layer fuses gates, transforms inputs, overlaps or is monolithic"
    moe = runner.moe_config
    if (
        moe.dp_size != 1
        or moe.pcp_size != 1
        or getattr(moe, "sp_size", 1) != 1
        or moe.use_ep
    ):
        return "the layer dispatches tokens (DP, PCP, SP or EP)"
    return None


def admit_flint_fused_router(runner: MoERunner) -> FlintFusedRouter | None:
    """U3 router provider: decide once whether this layer routes fused at M<=2."""
    if runner.gate is None or not enabled():
        return None
    reason = _rejection(runner)
    if reason is not None:
        logger.debug(
            "SM70 fused E256/K6 router not admitted for %s: %s.",
            runner.layer_name,
            reason,
        )
        return None
    if _capturing():
        # U3 never calls a provider under capture; refuse rather than take
        # the workspace from a graph pool if a caller does.
        _log_once(logging.WARNING, "vllm_sm70: router admission refused under capture")
        return None
    op = _flint_route_op()
    if op is None:
        _log_once(
            logging.WARNING,
            "vllm_sm70: _sm70_C lacks flint_route; the E256/K6 sigmoid router "
            "keeps the generic FP32 GEMM and topk_sigmoid.",
        )
        return None
    _log_once(
        logging.INFO,
        "vllm_sm70: fused E256/K6 sigmoid router enabled for one or two tokens "
        "(FP16 input, FP32 gate and accumulation; %s=0 disables it).",
        KILL_SWITCH,
    )
    return FlintFusedRouter(op, runner.gate.weight, runner)


def flint_router_provider(runner: MoERunner) -> FlintFusedRouter | None:
    """U3 provider, called on every forward: the layer's cached route.

    Decides once per layer (and caches the decision, including "no") in an
    eager pass. While a CUDA graph captures, an undecided layer gets None
    and nothing is cached or allocated.
    """
    route = runner.__dict__.get(_ROUTE_ATTR, _UNDECIDED)
    if route is not _UNDECIDED:
        return route
    if _capturing():
        return None
    route = admit_flint_fused_router(runner)
    runner.__dict__[_ROUTE_ATTR] = route
    return route


@functools.cache
def provider() -> Any | None:
    """For ``Platform.get_moe_router_provider()``; None when killed.

    Cached: U3 asks on every MoE forward, so the kill switch is read once.
    """
    return flint_router_provider if enabled() else None


def _is_moe_runner(module: Any) -> bool:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    return isinstance(module, MoERunner)


def _u3_present() -> bool:
    from vllm.platforms.interface import Platform

    return hasattr(Platform, "get_moe_router_provider")


def prepare_model(model: torch.nn.Module) -> int:
    """Decide every MoE runner's route before CUDA graph capture (R4).

    Called from the platform's U1 ``kernel_warmup`` hook (eager, before
    capture). Returns the number of admitted layers. A no-op without U3,
    with the kill switch engaged, or under capture.
    """
    if provider() is None or _capturing() or not _u3_present():
        return 0
    admitted = 0
    for module in model.modules():
        if _is_moe_runner(module) and module.gate is not None:
            admitted += flint_router_provider(module) is not None
    if admitted:
        _log_once(
            logging.INFO, "vllm_sm70: fused router admitted on %d layers", admitted
        )
    return admitted


def register() -> None:
    """Called from vllm_sm70.fused_moe.register(). Idempotent."""
    _register_fake()
