# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K17 (C6): TP2 all-reduce + fused_add_rms_norm as one collective.

Ported from fork cf9492a (commits 45896b591, 9f175fc6e, 2eb8f3b0a):

* ``Tp2AllReduceFusedAddRMSNormPattern`` <- fork compilation/passes/fusion/
  allreduce_rms_fusion.py (pattern, extra checks, unfused-boundary counter,
  pass admission ``tp2_add_rms_rejection``). Here it is one entry of the
  composite pass (vllm_sm70.passes), on upstream ``BasePattern`` and
  ``vllm.ir.ops.fused_add_rms_norm``.
* ``torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm`` <- fork
  parallel_state.py op + GroupCoordinator/CudaCommunicator/CustomAllreduce
  methods. The fork extended vLLM's communicator; here the op reads
  upstream's ``CustomAllreduce`` (``device_communicator.ca_comm``) and calls
  ``_sm70_C::tp2_all_reduce_fused_add_rms_norm`` with its handle. No vLLM
  source is patched.

Interface (fixed; tests pin it):
  torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm(
      tensor, tensor_b: Tensor | None, residual, weight,
      epsilon: float, scale: float, group_name: str) -> (normalized, residual)
  torch.ops._sm70_C.tp2_all_reduce_fused_add_rms_norm(
      fa: int, inp, inp_b: Tensor | None, residual, weight,
      normalized_out (a!), residual_out (b!), reg_buffer: int,
      reg_buffer_sz_bytes: int, epsilon: float, scale: float) -> ()

Admitted models: Flint only (FlintSm70ForCausalLM or FlintForCausalLM), as in the
fork, where only FlintModelConfig requested the pass (fork models/config.py:
341-372, MODELS_CONFIG_MAP["FlintForCausalLM"] :1052). Other hidden-2048
models (e.g. GemmaRMSNorm-style fp32 weight + 1) are never matched.

Admitted shapes (fused kernel): TP2, PP1, FP16, hidden 2048, 1-16 rows,
contiguous and 16-byte aligned operands, scale 1 or a positive power of two.
Anything else at runtime (prefill rows, a disabled or absent custom
all-reduce) runs the unfused ops, with the same result.

Kill switch: VLLM_SM70_FLINT_AR_RMSNORM=0 (read in register(), so the
composite pass's uuid changes with it).
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
from typing import Any

import torch
from torch import fx

logger = logging.getLogger("vllm.sm70")

PATTERN_NAME = "tp2_ar_fused_add_rms_norm"
# Bump when the pattern set or its checks change.
PATTERN_VERSION = 1

OP_NAMESPACE = "vllm_sm70"
OP_NAME = "tp2_all_reduce_fused_add_rms_norm"
KERNEL_NAMESPACE = "_sm70_C"
KERNEL_NAME = "tp2_all_reduce_fused_add_rms_norm"
KILL_SWITCH = "VLLM_SM70_FLINT_AR_RMSNORM"

# Shapes of the kernel (sm70/csrc/custom_ar): one 256-thread row per pack row.
HIDDEN_SIZE = 2048
MAX_ROWS = 16

# The fork keyed the request on the Flint architecture (MODELS_CONFIG_MAP).
# Production serves M5's FlintSm70ForCausalLM (via --hf-overrides), which
# model_config.architecture then reports; upstream's class is kept too.
FLINT_ARCHITECTURES = ("FlintSm70ForCausalLM", "FlintForCausalLM")

# Attributes of upstream's CustomAllreduce the fused path reads.
_CA_ATTRS = ("_ptr", "disabled", "rank", "world_size", "buffer_ptrs", "max_size")


# ---------------------------------------------------------------------------
# Kernel availability
# ---------------------------------------------------------------------------


def kernel_available() -> bool:
    """True if _sm70_C was built with the custom_ar area."""
    from vllm_sm70 import _ext

    if _ext.load() is None:
        return False
    try:
        return hasattr(getattr(torch.ops, KERNEL_NAMESPACE), KERNEL_NAME)
    except (AttributeError, RuntimeError):
        return False


@functools.cache
def _log_once(level: int, msg: str) -> None:
    logger.log(level, msg)


# ---------------------------------------------------------------------------
# Runtime: the op the pattern rewrites to
# ---------------------------------------------------------------------------


def _aligned(t: torch.Tensor) -> bool:
    return t.is_contiguous() and t.data_ptr() % 16 == 0


def operands_admitted(
    inp: torch.Tensor,
    inp_b: torch.Tensor | None,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """FP16 [1-16, 2048] operands, contiguous and 16-byte aligned, one device.

    fork CollectiveCapabilities.fused_add_rms_norm_tp2 (collective_provider.py,
    45896b591) minus the communicator part; the C++ host function checks the
    same again.
    """
    if inp.dim() != 2:
        return False
    if not (1 <= inp.shape[0] <= MAX_ROWS and inp.shape[1] == HIDDEN_SIZE):
        return False
    tensors = (inp, residual) if inp_b is None else (inp, inp_b, residual)
    for t in tensors:
        if t.dtype != torch.float16 or t.shape != inp.shape:
            return False
        if t.device != inp.device or not _aligned(t):
            return False
    return bool(
        weight.dtype == torch.float16
        and weight.dim() == 1
        and weight.numel() == HIDDEN_SIZE
        and weight.device == inp.device
        and _aligned(weight)
    )


def communicator_admitted(ca: Any) -> bool:
    """An enabled upstream CustomAllreduce over two same-node ranks."""
    if ca is None or any(not hasattr(ca, a) for a in _CA_ATTRS):
        return False
    if ca.disabled or ca.world_size != 2 or not ca._ptr:
        return False
    # Real IPC peers, not the multi-node fallback that repeats the local
    # buffer (UP custom_all_reduce.py:309-311).
    ptrs = ca.buffer_ptrs
    return len(ptrs) == 2 and ptrs[0] != ptrs[1]


def can_fuse(
    ca: Any,
    inp: torch.Tensor,
    inp_b: torch.Tensor | None,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether the fused kernel takes these operands on this communicator.

    Upstream's own gate (should_custom_ar) decides as for the plain
    all-reduce, so the fusion never changes which collective runs.
    """
    return bool(
        communicator_admitted(ca)
        and inp.is_cuda
        and operands_admitted(inp, inp_b, residual, weight)
        and inp.numel() * inp.element_size() <= ca.max_size
        and kernel_available()
        and ca.should_custom_ar(inp)
    )


def _kernel():
    return getattr(getattr(torch.ops, KERNEL_NAMESPACE), KERNEL_NAME)


def _group(group_name: str):
    from vllm.distributed import parallel_state

    assert group_name in parallel_state._groups, f"Group {group_name} is not found."
    group = parallel_state._groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group


def _unfused(group, tensor, tensor_b, residual, weight, epsilon, scale):
    import vllm.ir.ops

    total = tensor if tensor_b is None else tensor + tensor_b
    reduced = group._all_reduce_out_place(total)
    if scale != 1.0:
        reduced = reduced * scale
    return vllm.ir.ops.fused_add_rms_norm(reduced, residual, weight, epsilon)


def tp2_all_reduce_fused_add_rms_norm(
    tensor: torch.Tensor,
    tensor_b: torch.Tensor | None,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scale: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fused_add_rms_norm(all_reduce(tensor [+ tensor_b]) * scale, ...)``."""
    group = _group(group_name)
    communicator = group.device_communicator
    ca = getattr(communicator, "ca_comm", None)
    if not can_fuse(ca, tensor, tensor_b, residual, weight):
        return _unfused(group, tensor, tensor_b, residual, weight, epsilon, scale)
    normalized_out = torch.empty_like(tensor)
    residual_out = torch.empty_like(residual)
    _kernel()(
        ca._ptr,
        tensor,
        tensor_b,
        residual,
        weight,
        normalized_out,
        residual_out,
        ca.buffer_ptrs[ca.rank],
        ca.max_size,
        epsilon,
        scale,
    )
    return normalized_out, residual_out


def tp2_all_reduce_fused_add_rms_norm_fake(
    tensor: torch.Tensor,
    tensor_b: torch.Tensor | None,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scale: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(tensor), torch.empty_like(residual)


_LIB: Any = None


def register_op() -> None:
    """Define torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm. Idempotent.

    Pure Python (no _sm70_C needed); the pattern only targets it when the
    kernel is built.
    """
    global _LIB
    if _LIB is not None:
        return
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = Library(OP_NAMESPACE, "FRAGMENT")  # noqa: TOR901
    direct_register_custom_op(
        op_name=OP_NAME,
        op_func=tp2_all_reduce_fused_add_rms_norm,
        fake_impl=tp2_all_reduce_fused_add_rms_norm_fake,
        target_lib=lib,
    )
    _LIB = lib


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


def _all_reduce_op():
    return torch.ops.vllm.all_reduce.default


def _norm_op():
    return torch.ops.vllm_ir.fused_add_rms_norm.default


def _fp16_node(node: Any, dim: int, size: int) -> bool:
    if not isinstance(node, fx.Node):
        return False
    value = node.meta.get("val")
    return bool(
        value is not None
        and value.dtype == torch.float16
        and value.dim() == dim
        and value.shape[-1] == size
    )


def _match(match) -> bool:
    """FP16 [M, 2048] collective input, FP16 residual and [2048] FP16 weight.

    M is checked again at runtime. A non-FP16 weight (e.g. an fp32 weight + 1)
    never matches: the kernel multiplies by an FP16 weight.
    """
    collective = norm = None
    for node in match.nodes:
        if node.target == _all_reduce_op():
            collective = node
        elif node.target == _norm_op():
            norm = node
    if collective is None or norm is None:
        return False
    return (
        _fp16_node(collective.args[0], 2, HIDDEN_SIZE)
        and _fp16_node(norm.args[1], 2, HIDDEN_SIZE)
        and _fp16_node(norm.args[2], 1, HIDDEN_SIZE)
    )


def _scaled_match(match) -> bool:
    # The kernel rounds fp16(fp32(x) * scale), as the elementwise multiply
    # does; exact for a positive power-of-two scale.
    scales = [
        node.args[1]
        for node in match.nodes
        if node.target == torch.ops.aten.mul.Tensor and len(node.args) == 2
    ]
    if len(scales) != 1 or not isinstance(scales[0], float) or scales[0] <= 0.0:
        return False
    mantissa, _ = math.frexp(scales[0])
    return mantissa == 0.5 and _match(match)


def _pattern_class():
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import BasePattern

    class Tp2AllReduceFusedAddRMSNormPattern(BasePattern):
        """TP2: all-reduce + fused_add_rms_norm as one custom collective.

        Variants, each also with only the normalized output used:
          - ``fused_add_rms_norm(all_reduce(x), residual)``;
          - ``fused_add_rms_norm(all_reduce(x) * scale, residual)``, a
            positive power-of-two scale (residual-stream rescale);
          - ``fused_add_rms_norm(all_reduce(a + b), residual)``, two partial
            outputs (shared + routed experts) summed before the collective.
        """

        _SCALE_EXAMPLE = 0.125

        def __init__(self, epsilon: float, dtype: torch.dtype, device: str | None):
            super().__init__(dtype, device)
            self.epsilon = epsilon
            self.group_name = self.tp.unique_name

        def _all_reduce(self, x):
            return torch.ops.vllm.all_reduce(x, group_name=self.group_name)

        def _fused(self, input, input_b, residual, weight, scale):
            return getattr(torch.ops, OP_NAMESPACE).tp2_all_reduce_fused_add_rms_norm(
                input,
                input_b,
                residual,
                weight,
                self.epsilon,
                scale,
                group_name=self.group_name,
            )

        def _norm(self, reduced, residual, weight):
            import vllm.ir.ops

            return vllm.ir.ops.fused_add_rms_norm(
                reduced, residual, weight, self.epsilon
            )

        def register(self, pm_pass) -> None:
            import torch._inductor.pattern_matcher as pm

            h = HIDDEN_SIZE

            def plain(residual, input, weight):
                return self._norm(self._all_reduce(input), residual, weight)

            def plain_fused(residual, input, weight):
                return self._fused(input, None, residual, weight, 1.0)

            def scaled(residual, input, weight, scale):
                return self._norm(self._all_reduce(input) * scale, residual, weight)

            def scaled_fused(residual, input, weight, scale):
                return self._fused(input, None, residual, weight, scale)

            def sum2(residual, input, input_b, weight):
                reduced = self._all_reduce(input + input_b)
                return self._norm(reduced, residual, weight)

            def sum2_fused(residual, input, input_b, weight):
                return self._fused(input, input_b, residual, weight, 1.0)

            def normalized_only(fn):
                # Keep the parameter names: scalar_workaround binds by name.
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)[0]

                wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
                return wrapper

            two = [self.empty(5, h), self.empty(5, h), self.empty(h)]
            three = [
                self.empty(5, h),
                self.empty(5, h),
                self.empty(5, h),
                self.empty(h),
            ]
            variants = [
                # The sum2 pattern contains the plain one; try it first.
                (sum2, sum2_fused, three, _match, None),
                (
                    scaled,
                    scaled_fused,
                    two,
                    _scaled_match,
                    {"scale": self._SCALE_EXAMPLE},
                ),
                (plain, plain_fused, two, _match, None),
            ]
            for search, replace, inputs, check, scalars in variants:
                for wrap in (lambda fn: fn, normalized_only):
                    pm.register_replacement(
                        wrap(search),
                        wrap(replace),
                        inputs,
                        pm.fwd_only,
                        pm_pass,
                        extra_check=check,
                        scalar_workaround=scalars,
                    )

    return Tp2AllReduceFusedAddRMSNormPattern


def __getattr__(name: str) -> Any:
    # Lazy: importing upstream's fusion module pulls in the compile stack.
    if name == "Tp2AllReduceFusedAddRMSNormPattern":
        return _pattern_class()
    raise AttributeError(name)


def unfused_boundaries(graph: fx.Graph) -> int:
    """All-reduces still feeding a fused_add_rms_norm (missed fusions)."""
    count = 0
    for node in graph.nodes:
        if node.target != _all_reduce_op():
            continue
        users = list(node.users)
        users += [
            u
            for user in users
            if user.target == torch.ops.aten.mul.Tensor
            for u in user.users
        ]
        count += any(user.target == _norm_op() for user in users)
    return count


# ---------------------------------------------------------------------------
# Admission and the composite-pass entry
# ---------------------------------------------------------------------------


def rejection(config: Any, tp_size: int) -> str | None:
    """Why the fusion is off for this engine config, or None."""
    model_config = config.model_config
    architecture = getattr(model_config, "architecture", None)
    if architecture not in FLINT_ARCHITECTURES:
        return f"it is Flint-only ({'/'.join(FLINT_ARCHITECTURES)}), not {architecture}"
    parallel = config.parallel_config
    if tp_size != 2 or parallel.pipeline_parallel_size != 1:
        return "it needs TP2 without pipeline parallelism"
    if getattr(parallel, "disable_custom_all_reduce", False):
        return "the custom all-reduce is disabled"
    if model_config.dtype != torch.float16:
        return "it needs an FP16 model"
    if model_config.get_hidden_size() != HIDDEN_SIZE:
        return f"hidden size {model_config.get_hidden_size()} is not supported"
    if not kernel_available():
        return f"_sm70_C lacks {KERNEL_NAME} (built without VLLM_SM70_VLLM_CSRC_DIR)"
    return None


def _epsilon(config: Any) -> float:
    hf = getattr(config.model_config, "hf_text_config", None)
    return float(getattr(hf, "rms_norm_eps", 1e-6))


@functools.cache
def _pattern_pass(epsilon: float, dtype: torch.dtype, device: str, group: str):
    from torch._inductor.pattern_matcher import PatternMatcherPass
    from vllm.compilation.passes.inductor_pass import enable_fake_mode

    del group  # cache key only: the pattern binds the TP group's name
    import vllm.model_executor.layers.layernorm  # noqa: F401  (IR norm op)

    patterns = PatternMatcherPass(pass_name=PATTERN_NAME)

    @enable_fake_mode
    def build() -> None:
        _pattern_class()(epsilon, dtype, device).register(patterns)

    build()
    return patterns


_totals = {"fused": 0}


def apply(graph: fx.Graph) -> int:
    """Composite-pass entry: rewrite the graph if this engine admits it."""
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

    config = get_current_vllm_config()
    reason = rejection(config, get_tensor_model_parallel_world_size())
    if reason is not None:
        _log_once(
            logging.INFO,
            f"vllm_sm70: TP2 all-reduce + fused_add_rms_norm fusion is off: {reason}.",
        )
        return 0
    patterns = _pattern_pass(
        _epsilon(config),
        config.model_config.dtype,
        "cuda",
        get_tp_group().unique_name,
    )
    matched = patterns.apply(graph)
    _totals["fused"] += matched
    missed = unfused_boundaries(graph)
    logger.debug(
        "vllm_sm70: TP2 all-reduce + fused_add_rms_norm: %d fused in this graph "
        "(%d in total), %d unfused.",
        matched,
        _totals["fused"],
        missed,
    )
    if matched:
        # Compile-time only: absent when Inductor's compile cache is warm.
        _log_once(
            logging.INFO,
            "vllm_sm70: TP2 all-reduce + fused_add_rms_norm fused "
            f"{matched} pairs in the first compiled graph.",
        )
    if missed:
        _log_once(
            logging.WARNING,
            "vllm_sm70: TP2 all-reduce + fused_add_rms_norm fusion left "
            f"{missed} all-reduce -> fused_add_rms_norm pairs unfused in a graph.",
        )
    return matched


def register() -> bool:
    """Called by vllm_sm70.passes.register(). Returns True if registered.

    The pattern is registered only when the kill switch is on and _sm70_C has
    the kernel, so the composite pass's uuid (its pattern list) reflects both.
    """
    from vllm_sm70 import envs
    from vllm_sm70.passes import register_pattern

    if not getattr(envs, KILL_SWITCH):
        _log_once(
            logging.INFO,
            f"vllm_sm70: {KILL_SWITCH}=0; TP2 all-reduce + fused_add_rms_norm "
            "fusion not registered.",
        )
        return False
    if not kernel_available():
        _log_once(
            logging.INFO,
            f"vllm_sm70: _sm70_C has no {KERNEL_NAME} (custom_ar area not built: "
            "VLLM_SM70_VLLM_CSRC_DIR unset); TP2 all-reduce + fused_add_rms_norm "
            "fusion not registered.",
        )
        return False
    register_op()
    register_pattern(PATTERN_NAME, PATTERN_VERSION, apply)
    _log_once(
        logging.INFO,
        "vllm_sm70: TP2 all-reduce + fused_add_rms_norm fusion registered "
        f"({KILL_SWITCH}=0 disables it); it applies when the model is admitted "
        "and a graph is compiled.",
    )
    return True
