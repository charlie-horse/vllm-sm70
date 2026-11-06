# SPDX-License-Identifier: Apache-2.0
"""What the pre-capture warmup tunes: TurboMind layers, M values and limits.

Pure Python (no CUDA, no vLLM import), so the CPU tests cover it directly.

Three inputs decide the plan:
- the M values a decode step can run (``decode_m_values``): every CUDA graph
  capture size, the spec-decode verify M, every M that runs eagerly because it
  is above the largest capture size (or because graphs are off), and the
  fork's power-of-two set;
- the TurboMind layers the model holds (``discover``), one per distinct
  shape, found by the attributes our M3/M4/M6/M8 methods set at load;
- the C++ tuning limits (``limits``), mirrored from
  csrc/turbomind/ops/gemm_runtime.cpp. Above a limit the dispatch policy is
  kDefault in eager and in capture alike, so there is nothing to warm.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

# Kinds, in execution order.
NVFP4_DENSE = "nvfp4_dense"  # M3: ModelOpt W4A16 NVFP4 linear (TM GEMM, K11 GEMV)
FP8_DENSE = "fp8_dense"  # M6: block-FP8 linear on TurboMind
FP8_QPN8_TM = "fp8_qpn8_tm"  # M8: QPN8 linear's TurboMind copy (M > QPN8 max)
NVFP4_MOE = "nvfp4_moe"  # M4: NVFP4 routed experts (dense stage; K12 M<=2)
FP8_MOE = "fp8_moe"  # M6: FP8 routed experts (batched; compact M == 1)
KINDS = (NVFP4_DENSE, FP8_DENSE, FP8_QPN8_TM, NVFP4_MOE, FP8_MOE)
MOE_KINDS = frozenset({NVFP4_MOE, FP8_MOE})

# Fork set: {1, 2, 4, 8} and powers of two up to awq_warmup_max_m, which the
# fork's c4130 runtime defaults raised to 64 (FORK awq_sm70_warmup.py:313-333,
# platforms/runtime_defaults.py:36-44; recorded as kAwqWarmupMaxMDefault in
# csrc/turbomind/ops/tm_config.h).
FORK_M_VALUES = (1, 2, 4, 8, 16, 32, 64)
NVFP4_GROUP_SIZE = 16
FP8_BLOCK = 128
K12_MAX_TOKENS = 2  # vllm_sm70.ops.flint_moe.MAX_TOKENS


def _atoi(raw: str) -> int:
    """C atoi(): optional space and sign, then digits; anything else is 0."""
    s = raw.lstrip(" \t\n\r\f\v")
    sign = 1
    if s[:1] in ("+", "-"):
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    digits = ""
    for ch in s:
        if not ch.isdigit():
            break
        digits += ch
    return sign * int(digits) if digits else 0


# The C++ default of the three *_TUNE_SMALL_SHAPES knobs (gemm_runtime.cpp
# tune_small_shapes_enabled and friends). Main: 1. Keep in step with C++: if
# small-shape tuning defaults off there (#77), set this to 0. Then the warmup
# has nothing to tune unless a knob is set to 1, and every call is kDefault
# (deterministic, no LUT needed).
TUNE_SMALL_SHAPES_DEFAULT = 1


def knob(name: str, default: int) -> int:
    """tm_config.h policy_atoi(): set (even to "") -> atoi(value), else default."""
    raw = os.environ.get(name)
    return default if raw is None else _atoi(raw)


@dataclass(frozen=True)
class Limit:
    """C++ tuning gate of one kind: tunes iff enabled and M (or MoE slots) <= max."""

    enabled: bool
    max_m: int  # dense: rows; MoE: routed slots (tokens * top_k)
    source: str


def limits() -> dict[str, Limit]:
    """Per-kind tuning limits, mirroring gemm_runtime.cpp.

    - NVFP4 dense: select_nvfp4_dense_dispatch_policy (:525-535),
      nvfp4_tune_small_shapes (default 1), nvfp4_dense_tune_max_m (64).
    - FP8 dense (TM and the QPN8 TM copy): select_fp8_dense_dispatch_policy
      (:509-523), fp8_tune_small_shapes (1) and fp8_dense_tune_max_m (64);
      with VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR=1 the generic gate instead
      (awq_tune_small_shapes 1, f16_dense_tune_max_m 16).
    - NVFP4 MoE: select_nvfp4_moe_dispatch_policy (:592-599),
      nvfp4_tune_small_shapes (1), nvfp4_moe_tune_max_tokens (128 slots).
    - FP8 MoE: select_fp8_moe_dispatch_policy (:578-590),
      fp8_tune_small_shapes (1), awq_moe_tune_max_tokens (128 slots).
    """
    nvfp4_on = knob("VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES", TUNE_SMALL_SHAPES_DEFAULT) != 0
    fp8_on = knob("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", TUNE_SMALL_SHAPES_DEFAULT) != 0
    if knob("VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR", 0) != 0:
        fp8_dense = Limit(
            knob("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES", TUNE_SMALL_SHAPES_DEFAULT) != 0,
            max(knob("VLLM_SM70_F16_DENSE_TUNE_MAX_M", 16), 0),
            "generic (0dot3 selector)",
        )
    else:
        fp8_dense = Limit(
            fp8_on, max(knob("VLLM_SM70_FP8_DENSE_TUNE_MAX_M", 64), 0), "fp8"
        )
    return {
        NVFP4_DENSE: Limit(
            nvfp4_on, max(knob("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", 64), 0), "nvfp4"
        ),
        FP8_DENSE: fp8_dense,
        FP8_QPN8_TM: fp8_dense,
        NVFP4_MOE: Limit(
            nvfp4_on,
            max(knob("VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS", 128), 0),
            "nvfp4 moe",
        ),
        FP8_MOE: Limit(
            fp8_on, max(knob("VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS", 128), 0), "fp8 moe"
        ),
    }


# --------------------------------------------------------------------------
# M values
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MValues:
    """The decode-step M values and where each group comes from (for the log)."""

    all: tuple[int, ...]
    capture: tuple[int, ...]
    eager: tuple[int, ...]
    verify_m: int
    max_decode_m: int


def _mode_is_none(mode: Any) -> bool:
    if mode is None:
        return True
    name = getattr(mode, "name", None)
    return (name if name is not None else str(mode)).upper().endswith("NONE")


def decode_m_values(vllm_config: Any) -> MValues:
    """Every M a decode step can run a TurboMind GEMM at.

    - graphs: every size in compilation_config.cudagraph_capture_sizes
      (a padded decode batch only ever runs at one of these);
    - verify: 1 + num_speculative_tokens, one request's spec-decode verify;
    - eager: every M above the largest capture size up to
      max_num_seqs * (num_speculative_tokens + 1), since those batches run
      without a graph. With graphs off (enforce_eager, cudagraph_mode NONE)
      or a NONE mixed mode (mixed batches run eagerly), every M from 1 up;
    - FORK_M_VALUES, as the fork warmed.
    Everything is capped at max_num_batched_tokens. Each kind then drops the
    M values above its own tuning limit (``limits``).
    """
    sched = getattr(vllm_config, "scheduler_config", None)
    comp = getattr(vllm_config, "compilation_config", None)
    model = getattr(vllm_config, "model_config", None)
    spec = getattr(vllm_config, "speculative_config", None)

    num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
    verify_m = 1 + max(num_spec, 0)
    max_seqs = int(getattr(sched, "max_num_seqs", 0) or 0)
    max_tokens = int(getattr(sched, "max_num_batched_tokens", 0) or 0)

    mode = getattr(comp, "cudagraph_mode", None)
    enforce_eager = bool(getattr(model, "enforce_eager", False))
    graphs_on = not enforce_eager and not _mode_is_none(mode)
    sizes = getattr(comp, "cudagraph_capture_sizes", None) or []
    capture = sorted({int(s) for s in sizes if int(s) > 0}) if graphs_on else []

    max_decode_m = max_seqs * verify_m if max_seqs else max([verify_m, *capture])
    mixed_eager = False
    if capture:
        try:
            mixed_eager = _mode_is_none(mode.mixed_mode())
        except AttributeError:
            mixed_eager = False
    eager_lo = 0 if (not capture or mixed_eager) else capture[-1]
    eager = list(range(eager_lo + 1, max_decode_m + 1))

    values = {*capture, verify_m, *eager, *FORK_M_VALUES}
    if max_tokens:
        values = {m for m in values if m <= max_tokens}
    return MValues(
        all=tuple(sorted(values)),
        capture=tuple(capture),
        eager=tuple(eager),
        verify_m=verify_m,
        max_decode_m=max_decode_m,
    )


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """One distinct TurboMind shape and the M values (or MoE tokens) to run."""

    kind: str
    key: tuple
    ms: tuple[int, ...]
    layer: Any = field(compare=False, hash=False, repr=False)


@dataclass
class Plan:
    items: list[Item]
    m_values: MValues
    limits: dict[str, Limit]
    skipped: dict[str, str] = field(default_factory=dict)  # kind -> why

    def by_kind(self) -> dict[str, list[Item]]:
        out: dict[str, list[Item]] = {k: [] for k in KINDS}
        for item in self.items:
            out[item.kind].append(item)
        return out

    def calls(self) -> int:
        return sum(len(i.ms) for i in self.items)

    def fingerprint(self) -> str:
        """Same on every TP rank iff they will make the same TurboMind calls."""
        body = json.dumps(
            [[i.kind, list(i.key), list(i.ms)] for i in self.items], sort_keys=True
        )
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def _qpn8_max_m() -> int:
    """K09's block-scaled QPN8 GEMV bound (8; 32 with VLLM_SM70_FP8_QPN8_M32=1)."""
    try:
        from vllm_sm70.ops import qpn8

        return int(qpn8.qpn8_max_m(True))
    except Exception:
        return 32 if os.environ.get("VLLM_SM70_FP8_QPN8_M32") == "1" else 8


def _classify(layer: Any) -> tuple[str, tuple] | None:
    """(kind, shape key) of a prepared SM70 TurboMind layer, else None.

    Keys hold every value the dispatch key depends on, so two layers with
    one key make the same TurboMind calls.
    """
    state = getattr(layer, "_sm70_turbomind_linear", None)  # M3 STATE_ATTR
    if state is not None:
        k = int(state.weight.shape[0])
        n = int(state.weight.shape[1]) * 8  # tm_weight i32 [K, N/8]
        return NVFP4_DENSE, (k, n, int(state.group_size), bool(state.tm_gemv))
    if getattr(layer, "sm70_fp8_turbomind", False):  # M6 dense
        k = int(layer.weight.shape[0])  # tm_weight u8 [K, N]
        return FP8_DENSE, (k, int(layer.output_size_per_partition))
    if getattr(layer, "sm70_fp8_qpn8", False):  # M8
        tm_weight = getattr(layer, "sm70_qpn8_tm_weight", None)
        if tm_weight is None:
            return None  # no TurboMind copy: dequantize + mm above the QPN8 bound
        return FP8_QPN8_TM, (int(tm_weight.shape[0]), int(tm_weight.shape[1]))
    if hasattr(layer, "sm70_nvfp4_num_experts"):  # M4
        return NVFP4_MOE, (
            int(layer.sm70_nvfp4_num_experts),
            int(layer.sm70_nvfp4_hidden_size),
            int(layer.sm70_nvfp4_intermediate_size),
            int(layer.sm70_nvfp4_top_k),
            getattr(layer, "sm70_nvfp4_decode_plan", None) is not None,
        )
    if hasattr(layer, "sm70_fp8_moe_compact") and hasattr(layer, "sm70_num_experts"):
        return FP8_MOE, (  # M6 MoE
            int(layer.sm70_num_experts),
            int(layer.sm70_hidden_size),
            int(layer.sm70_intermediate_size),
            int(layer.sm70_top_k),
            bool(layer.sm70_fp8_moe_compact),
            bool(getattr(layer, "sm70_fp8_moe_decode", False)),
        )
    return None


def item_ms(kind: str, key: tuple, ms: Iterable[int], limit: Limit) -> tuple[int, ...]:
    """The M values (MoE: tokens) of one shape that reach a tunable TM GEMM."""
    if not limit.enabled:
        return ()
    qpn8_max = _qpn8_max_m() if kind == FP8_QPN8_TM else 0
    out = []
    for m in ms:
        if kind == NVFP4_DENSE:
            # K11 GEMV serves M <= 2 of admitted layers; TurboMind above.
            if key[3] and m <= 2:
                continue
            if m > limit.max_m:
                continue
        elif kind == FP8_DENSE:
            if m > limit.max_m:
                continue
        elif kind == FP8_QPN8_TM:
            if m <= qpn8_max or m > limit.max_m:
                continue
        elif kind in MOE_KINDS:
            top_k = key[3]
            if m * top_k > limit.max_m:  # the MoE gate counts routed slots
                continue
            if kind == NVFP4_MOE and key[4] and m <= K12_MAX_TOKENS:
                continue  # K12 two-kernel decode, no TurboMind
        out.append(m)
    return tuple(out)


def iter_layers(models: Iterable[Any]) -> Iterator[Any]:
    seen: set[int] = set()
    for model in models:
        if model is None:
            continue
        for layer in model.modules():
            if id(layer) not in seen:
                seen.add(id(layer))
                yield layer


def build(models: Iterable[Any], vllm_config: Any) -> Plan:
    """One Item per distinct (kind, shape), in KINDS then discovery order."""
    m_values = decode_m_values(vllm_config)
    lims = limits()
    found: dict[tuple[str, tuple], Any] = {}
    for layer in iter_layers(models):
        hit = _classify(layer)
        if hit is not None and hit not in found:
            found[hit] = layer
    items: list[Item] = []
    skipped: dict[str, str] = {}
    for kind in KINDS:
        for (k, key), layer in found.items():
            if k != kind:
                continue
            if not lims[kind].enabled:
                skipped[kind] = f"tuning off ({lims[kind].source} tune_small_shapes=0)"
                continue
            # MoE: every token count up to the slot limit, not just the
            # decode M set. Eager MoE dispatch has no imported-cache reuse
            # (select_moe_dispatch_policy_impl), so an unwarmed count would be
            # timed on first use, differently per instance. That is cheap:
            # <= 128 / top_k counts.
            pool = m_values.all
            if kind in MOE_KINDS:
                pool = range(1, lims[kind].max_m // max(key[3], 1) + 1)
            ms = item_ms(kind, key, pool, lims[kind])
            if ms:
                items.append(Item(kind, key, ms, layer))
    return Plan(items=items, m_values=m_values, limits=lims, skipped=skipped)
