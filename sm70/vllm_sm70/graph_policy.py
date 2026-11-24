# SPDX-License-Identifier: Apache-2.0
"""SM70 CUDA-graph capture sizes and graph-memory profiling.

On the Flash-V100 compile path, replace upstream's default CUDA-graph
capture sizes with the SM70 tables below (fewer sizes: less graph memory,
shorter capture), and (V1 runner only) set
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 when it is unset.
Not applied (see the *_rejection functions) under enforce_eager,
cudagraph_mode NONE, a disabled compile, a non-Flash-V100 attention
backend, or explicit user capture sizes.

Called from Sm70CudaPlatform.apply_config_platform_defaults, before
upstream's _set_cudagraph_sizes, so the sizes still go through upstream's
filtering (<= max_num_batched_tokens) and spec-decode rounding. A GEMM LUT
exported with other capture sizes does not cover these sizes.
This module imports nothing from vllm at import time."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger("vllm.sm70")

SIZES_SWITCH = "VLLM_SM70_GRAPH_CAPTURE_SIZES"
MEMORY_SWITCH = "VLLM_SM70_GRAPH_MEMORY_POLICY"
UPSTREAM_MEMORY_ENV = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"

NOMTP_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32)
SPEC_REQUEST_SIZES = (1, 2, 3, 4, 6, 8, 12, 16)
SPEC_AUX_CAPTURE_SIZES = (1, 2, 4, 8, 9, 18)
# Mixed steps that join a short prompt to the resident verifier rows.
SPEC_MIXED_CAPTURE_SIZES = (64, 96, 128, 160, 192, 256)
# Largest request count that gets SM70 graph sizes.
MAX_GRAPH_REQS = 32


def _graph_reqs(max_num_seqs: int) -> int:
    return min(max(int(max_num_seqs), 1), MAX_GRAPH_REQS)


def nomtp_capture_sizes(max_num_seqs: int) -> list[int]:
    """{1, 2, 4, 8, 16, 32} up to min(max_num_seqs, 32), plus 1, 2 and that cap."""
    cap = _graph_reqs(max_num_seqs)
    sizes = {s for s in NOMTP_CAPTURE_SIZES if s <= cap}
    sizes.update((1, 2, cap))
    return sorted(sizes)


def speculative_capture_sizes(max_num_seqs: int, decode_query_len: int) -> list[int]:
    """q x request sizes up to the cap, plus q x cap, aux and mixed sizes (q = k + 1)."""
    cap = _graph_reqs(max_num_seqs)
    reqs = {s for s in SPEC_REQUEST_SIZES if s <= cap}
    reqs.add(cap)
    verifier = {decode_query_len * r for r in reqs}
    return sorted(
        set(SPEC_AUX_CAPTURE_SIZES) | verifier | set(SPEC_MIXED_CAPTURE_SIZES)
    )


def capture_sizes(max_num_seqs: int, num_speculative_tokens: int = 0) -> list[int]:
    """The SM70 capture sizes for a scheduler capacity and draft width."""
    if num_speculative_tokens:
        return speculative_capture_sizes(max_num_seqs, num_speculative_tokens + 1)
    return nomtp_capture_sizes(max_num_seqs)


def max_capture_size(sizes: Iterable[int], max_num_batched_tokens: int) -> int:
    """The largest size <= max_num_batched_tokens.

    Upstream drops sizes above max_num_batched_tokens; a larger cap would
    then contradict the sizes and fail validation.
    """
    return max(s for s in sizes if s <= max_num_batched_tokens)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def baseline_rejection(vllm_config: Any) -> str | None:
    """Why the SM70 graph policy does not apply, or None (sizes and memory)."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return "no model config"
    if getattr(model_config, "enforce_eager", False):
        return "enforce_eager"
    comp = vllm_config.compilation_config
    mode = getattr(comp, "cudagraph_mode", None)
    if mode is not None and getattr(mode, "name", str(mode)) == "NONE":
        return "cudagraph_mode NONE"
    for name in ("TORCH_COMPILE_DISABLE", "VLLM_USE_BREAKABLE_CUDAGRAPH"):
        if _env_on(name):
            return f"{name}=1"
    from vllm_sm70 import attention

    if not attention.flash_v100_enabled():
        return f"{attention.KILL_SWITCH}=0"
    attn_cfg = getattr(vllm_config, "attention_config", None)
    backend = getattr(attn_cfg, "backend", None)
    if backend is not None and getattr(backend, "name", backend) != "CUSTOM":
        return f"attention backend {getattr(backend, 'name', backend)} selected"
    return None


def _num_speculative_tokens(vllm_config: Any) -> int:
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return 0
    return int(getattr(spec, "num_speculative_tokens", 0) or 0)


def sizes_rejection(vllm_config: Any) -> str | None:
    """Why the capture sizes stay upstream's, or None to apply them."""
    if os.environ.get(SIZES_SWITCH) == "0":
        return f"{SIZES_SWITCH}=0"
    reason = baseline_rejection(vllm_config)
    if reason is not None:
        return reason
    comp = vllm_config.compilation_config
    if comp.cudagraph_capture_sizes is not None:
        return "cudagraph_capture_sizes set by the user"
    if comp.max_cudagraph_capture_size is not None:
        return "max_cudagraph_capture_size set by the user"
    spec = getattr(vllm_config, "speculative_config", None)
    dynamic = getattr(spec, "uses_dynamic_speculative_decoding", None)
    if callable(dynamic) and dynamic():
        return "dynamic speculative decoding"
    return None


def _uses_v2_runner(vllm_config: Any) -> bool:
    try:
        return bool(vllm_config.use_v2_model_runner)
    except Exception:  # an unresolvable runner choice: leave upstream alone
        logger.debug("vllm_sm70: use_v2_model_runner failed", exc_info=True)
        return True


def memory_rejection(vllm_config: Any) -> str | None:
    """Why the graph-memory estimate stays upstream's, or None to turn it off."""
    if os.environ.get(MEMORY_SWITCH) == "0":
        return f"{MEMORY_SWITCH}=0"
    if UPSTREAM_MEMORY_ENV in os.environ:
        return f"{UPSTREAM_MEMORY_ENV} set by the user"
    reason = baseline_rejection(vllm_config)
    if reason is not None:
        return reason
    if _uses_v2_runner(vllm_config):
        return "V2 model runner"
    return None


_LOGGED: set[str] = set()
# True once apply() itself wrote UPSTREAM_MEMORY_ENV in this process.
_memory_env_written = False


def _info_once(fmt: str, *args: Any) -> None:
    msg = fmt % args
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        logger.info(msg)


def apply(vllm_config: Any) -> None:
    """Apply the capture sizes and memory policy before upstream sizes graphs."""
    reason = sizes_rejection(vllm_config)
    if reason is None:
        sched = vllm_config.scheduler_config
        sizes = capture_sizes(sched.max_num_seqs, _num_speculative_tokens(vllm_config))
        comp = vllm_config.compilation_config
        comp.cudagraph_capture_sizes = sizes
        comp.max_cudagraph_capture_size = max_capture_size(
            sizes, sched.max_num_batched_tokens
        )
        _info_once(
            "vllm_sm70: SM70 CUDA graph capture sizes %s (max %d); %s=0 "
            "restores upstream's",
            sizes,
            comp.max_cudagraph_capture_size,
            SIZES_SWITCH,
        )
    else:
        # Also seen when a VllmConfig is rebuilt from one that already holds
        # the SM70 sizes: they are then kept as they are.
        _info_once("vllm_sm70: SM70 CUDA graph capture sizes not applied (%s)", reason)

    global _memory_env_written
    if _memory_env_written and os.environ.get(UPSTREAM_MEMORY_ENV) == "0":
        return  # a rebuilt VllmConfig in this process: already applied
    reason = memory_rejection(vllm_config)
    if reason is None:
        # Read by the workers' memory profiler; they are spawned after this,
        # so they inherit it.
        os.environ[UPSTREAM_MEMORY_ENV] = "0"
        _memory_env_written = True
        _info_once(
            "vllm_sm70: %s=0 (V1 runner: CUDA graph memory is not taken from "
            "the KV budget, as on the fork); %s=0 restores upstream's estimate",
            UPSTREAM_MEMORY_ENV,
            MEMORY_SWITCH,
        )
    else:
        _info_once("vllm_sm70: upstream CUDA graph memory estimate (%s)", reason)
