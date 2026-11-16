# SPDX-License-Identifier: Apache-2.0
"""SM70 CUDA-graph capture sizes and graph-memory profiling (T8, T9).

What: on the Flash-V100 compile path, replace upstream's default CUDA-graph
  capture sizes with the fork's SM70 tables, and (V1 runner only) turn off
  upstream's CUDA-graph memory estimate in the KV budget.
  - No speculative decoding: {1, 2, 4, 8, 16, 32} up to min(max_num_seqs,
    32), plus 1, 2 and that cap. max_num_seqs=20 gives [1, 2, 4, 8, 16, 20];
    upstream gives [1, 2, 4, 8, 16, 20, 24, 32, 40] (2 x max_num_seqs, UP
    config/vllm.py:2398-2611).
  - Speculative decoding (k draft tokens, q = k + 1): q x {1, 2, 3, 4, 6, 8,
    12, 16} up to min(max_num_seqs, 32), plus q x that cap, plus the
    auxiliary sizes {1, 2, 4, 8, 9, 18} and the mixed prefill+verify sizes
    {64, 96, 128, 160, 192, 256}.
  - max_cudagraph_capture_size: the largest of those sizes that is
    <= max_num_batched_tokens.
  - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 when the V1 model runner is
    used and the variable is unset (upstream default 1, UP envs.py:2219).
Why: fork parity. Port of FORK vllm/platforms/runtime_defaults.py:22-33
  (tables), :200-251 (size functions) and :549-603, :636-642 (applied for
  cc 7.0 under the Flash-V100 0.0.3 compile graph). B32 is the largest
  concurrency the fork validated end to end on SM70; fewer sizes also mean
  less graph memory and a shorter capture.
  The fork's MTP split-draft sizes (VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS,
  default off there) and its Flash-V100 small-q auto-raise (only when
  k + 1 > 16; c4130's DFlash2 has k + 1 = 8) are not ported.
When it applies (all must hold; else upstream sizing is untouched):
  - VLLM_SM70_GRAPH_CAPTURE_SIZES is not "0" (sizes) /
    VLLM_SM70_GRAPH_MEMORY_POLICY is not "0" (memory estimate);
  - a model config, no enforce_eager, cudagraph_mode not NONE,
    TORCH_COMPILE_DISABLE and VLLM_USE_BREAKABLE_CUDAGRAPH not on (fork
    rule: compile not disabled by the user);
  - the Flash-V100 baseline: VLLM_SM70_FLASH_V100 is not "0" and no
    attention backend other than CUSTOM (Flash-V100) was chosen;
  - sizes only: the user set neither cudagraph_capture_sizes nor
    max_cudagraph_capture_size (an explicit --compilation-config always
    wins), and no dynamic speculative-decoding schedule is configured;
  - memory only: the V1 model runner, and the user did not set
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS.
Hook: Sm70CudaPlatform.apply_config_platform_defaults, which upstream calls
  in VllmConfig.__post_init__ (UP config/vllm.py:1825) before
  _set_cudagraph_sizes (:2013), so the sizes go through upstream's own
  filtering (<= max_num_batched_tokens) and the V1 spec-decode rounding.
Note: the TurboMind pre-capture warmup plans one call per capture size, so a
  GEMM LUT exported with other capture sizes does not cover these (re-export).

This module imports nothing from vllm at import time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger("vllm.sm70")

SIZES_SWITCH = "VLLM_SM70_GRAPH_CAPTURE_SIZES"
MEMORY_SWITCH = "VLLM_SM70_GRAPH_MEMORY_POLICY"
UPSTREAM_MEMORY_ENV = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"

# FORK runtime_defaults.py:22-33.
NOMTP_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32)
SPEC_REQUEST_SIZES = (1, 2, 3, 4, 6, 8, 12, 16)
SPEC_AUX_CAPTURE_SIZES = (1, 2, 4, 8, 9, 18)
# Mixed steps that join a short prompt to the resident verifier rows.
SPEC_MIXED_CAPTURE_SIZES = (64, 96, 128, 160, 192, 256)
# B32: largest concurrency with end-to-end SM70 graph validation (fork).
MAX_GRAPH_REQS = 32


def _graph_reqs(max_num_seqs: int) -> int:
    return min(max(int(max_num_seqs), 1), MAX_GRAPH_REQS)


def nomtp_capture_sizes(max_num_seqs: int) -> list[int]:
    """FORK _sm70_nomtp_cudagraph_capture_sizes (runtime_defaults.py:200)."""
    cap = _graph_reqs(max_num_seqs)
    sizes = {s for s in NOMTP_CAPTURE_SIZES if s <= cap}
    sizes.update((1, 2, cap))
    return sorted(sizes)


def speculative_capture_sizes(max_num_seqs: int, decode_query_len: int) -> list[int]:
    """FORK _sm70_speculative_cudagraph_capture_sizes (runtime_defaults.py:234)."""
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
    """FORK _sm70_max_cudagraph_capture_size (runtime_defaults.py:212).

    Upstream drops sizes above max_num_batched_tokens; a larger cap would
    then contradict the sizes and fail validation (UP config/vllm.py:2612).
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
    """Apply T8 and T9 to a VllmConfig before upstream sizes its graphs."""
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
        # Read by the workers' memory profiler (UP v1/worker/gpu_worker.py:
        # 719-723); they are spawned after this, so they inherit it.
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
