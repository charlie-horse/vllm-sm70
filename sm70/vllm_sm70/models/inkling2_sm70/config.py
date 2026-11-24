# SPDX-License-Identifier: Apache-2.0
"""Engine-config hook for Inkling2LoopSm70ForConditionalGeneration.

* Refuses anything but fp16 on cc 7.0, prompt embeddings, pipeline
  parallelism, an explicit ``--attention-backend`` (the model owns its
  attention) and a speculative config other than a staged dspark draft.
* Sets block size ``KV_BLOCK_SIZE`` and the KV cache in the model dtype.
* Async scheduling keeps two batches in flight, so the admission caps count
  2 x ``max_num_batched_tokens``; VLLM_SM70_INKLING_ASYNC_SCHED=0 turns it off.
* Sets the DSpark batch-size schedule (spec_policy.py).
* CUDA graphs: an unset mode becomes FULL_DECODE_ONLY; FULL mixed batches are
  refused; capture sizes are capped at q x ``DECODE_GRAPH_MAX_BATCH``.
* Copies ``max_image_encoder_tokens`` onto ``hf_config`` and, unless they are
  set or VLLM_SM70_INKLING_TM_TUNE=0, sets the TurboMind tuning knobs for
  the workers (``tm_tuning_env``). Logs the routes once per process.
"""

import os
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger("vllm.sm70.inkling2")

ARCH = "Inkling2LoopSm70ForConditionalGeneration"

# Attention page (block x 2 x head_dim) == conv page (kernel x conv head size).
KV_BLOCK_SIZE = 32
# Most requests of a decode batch captured in a FULL graph: a decode step
# gathers the whole block-table width of keys per global-layer row.
DECODE_GRAPH_MAX_BATCH = 8

TM_TUNE_KNOBS = (
    "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
    "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS",
    "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M",
)

_logged = False


def _log_routes_once(
    visit_dump_dir: str | None, cudagraph_mode: str, async_scheduling: str
) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    logger.info(
        "Inkling 2 SM70: fp16 on cc 7.0; model-owned attention and KV caches "
        "(neither FLASH_ATTN_V100 nor TRITON_ATTN); KV block size %d; "
        "async scheduling %s; cudagraph_mode %s; no "
        "prompt embeddings; per-visit state dump %s.",
        KV_BLOCK_SIZE,
        async_scheduling,
        cudagraph_mode,
        f"to {visit_dump_dir}" if visit_dump_dir else "off",
    )


def _schedule_capture_sizes(schedule: Any, query_len: int, max_reqs: int) -> list[int]:
    """Capture sizes under a dynamic speculative-decoding `schedule`: q x n
    for each q it gives and n up to the most requests (<= `max_reqs`) it
    gives q, and the draft's `query_len` x n up to the most requests of a
    drafting step (v0.33's managers round a size up to a multiple of q, UP
    v1/worker/gpu/cudagraph_utils.py)."""
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    lookup = build_dynamic_sd_schedule_lookup(schedule, max_reqs, query_len - 1)
    widest = {1 + n: reqs for reqs, n in enumerate(lookup) if reqs}
    drafting = max((reqs for reqs, n in enumerate(lookup) if reqs and n), default=0)
    sizes = {q * n for q, reqs in widest.items() for n in range(1, reqs + 1)}
    sizes.update(query_len * n for n in range(1, drafting + 1))
    return sorted(sizes)


def _cap_decode_graph_batch(
    compilation_config: Any,
    scheduler_config: Any,
    query_len: int = 1,
    schedule: Any = None,
) -> None:
    """max_cudagraph_capture_size = `query_len` tokens per request times
    min(DECODE_GRAPH_MAX_BATCH, max_num_seqs) requests unless the capture
    sizes or their maximum are set; set ones above `query_len` x
    DECODE_GRAPH_MAX_BATCH raise ValueError. With a dynamic
    speculative-decoding `schedule` the capture sizes are
    ``_schedule_capture_sizes`` up to that many requests instead."""
    sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    largest = getattr(compilation_config, "max_cudagraph_capture_size", None)
    if sizes:
        largest = max(largest or 0, *sizes)
    if largest is not None:
        if largest > DECODE_GRAPH_MAX_BATCH * query_len:
            raise ValueError(
                f"{ARCH} captures FULL decode graphs of at most "
                f"{DECODE_GRAPH_MAX_BATCH} requests of {query_len} tokens, got "
                f"a capture size of {largest}; lower "
                "-cc.max_cudagraph_capture_size or -cc.cudagraph_capture_sizes."
            )
        return
    cap = DECODE_GRAPH_MAX_BATCH
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    if isinstance(max_num_seqs, int) and max_num_seqs > 0:
        cap = min(cap, max_num_seqs)
    if not schedule:
        compilation_config.max_cudagraph_capture_size = cap * query_len
        return
    sizes = _schedule_capture_sizes(schedule, query_len, cap)
    compilation_config.cudagraph_capture_sizes = sizes
    compilation_config.max_cudagraph_capture_size = sizes[-1]


def tm_tuning_env(
    compilation_config: Any, hf_config: Any, query_len: int = 1
) -> dict[str, str]:
    """The ``TM_TUNE_KNOBS`` values to set for this engine ({} for none);
    `query_len` tokens per request of a decode step."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm_sm70 import envs as sm70_envs

    from . import moe_tm

    if not sm70_envs.VLLM_SM70_INKLING_TM_TUNE or not moe_tm.enabled():
        return {}
    if any(name in os.environ for name in TM_TUNE_KNOBS):
        return {}
    get_text = getattr(hf_config, "get_text_config", None)
    text = get_text() if callable(get_text) else hf_config
    top_k = int(getattr(text, "num_experts_per_tok", 0) or 0)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if top_k <= 0 or mode is None or mode.decode_mode() != CUDAGraphMode.FULL:
        return {}
    sizes = getattr(compilation_config, "cudagraph_capture_sizes", None) or ()
    largest = getattr(compilation_config, "max_cudagraph_capture_size", None)
    cap = max(largest or 0, *sizes, 0)
    if cap <= 0 or moe_tm.turbomind() is None:
        return {}
    slots = min(cap, DECODE_GRAPH_MAX_BATCH * query_len) * top_k
    return dict(zip(TM_TUNE_KNOBS, ("1", str(slots), "0"), strict=True))


class Inkling2Sm70ModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Raises ValueError for an engine this model can't serve."""
        import torch

        from vllm import envs
        from vllm.config.compilation import CUDAGraphMode
        from vllm.platforms import current_platform
        from vllm_sm70 import envs as sm70_envs
        from vllm_sm70.config import sm70_config

        from . import spec_policy
        from .backend import decode_graph_enabled, decode_query_len
        from .draft import speculative_rejection
        from .sm70_configs import IMAGE_ENCODER_TOKEN_CAP_ATTR

        capability = (
            current_platform.get_device_capability()
            if current_platform.is_cuda()
            else None
        )
        if capability is None or (capability.major, capability.minor) != (7, 0):
            raise ValueError(
                f"{ARCH} needs a compute capability 7.0 GPU (V100), got "
                f"{capability}. Drop --hf-overrides architectures to serve "
                "upstream's Inkling 2 model on other GPUs."
            )
        model_config = vllm_config.model_config
        if model_config.dtype != torch.float16:
            raise ValueError(
                f"{ARCH} needs --dtype half (V100 has no bf16), got "
                f"{model_config.dtype}."
            )
        spec_reason = speculative_rejection(
            vllm_config.speculative_config,
            getattr(model_config, "hf_config", None),
            v2_runner=bool(getattr(vllm_config, "use_v2_model_runner", False)),
        )
        if spec_reason is not None:
            raise ValueError(f"{ARCH} {spec_reason}.")
        if getattr(model_config, "enable_prompt_embeds", False):
            raise ValueError(
                f"{ARCH} does not support prompt embeddings; drop "
                "--enable-prompt-embeds."
            )
        backend = vllm_config.attention_config.backend
        if backend is not None:
            raise ValueError(
                f"{ARCH} owns its attention layers and KV caches and uses no "
                f"vLLM attention backend, got --attention-backend "
                f"{getattr(backend, 'name', backend)}; drop it."
            )
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype not in ("auto", "float16"):
            raise ValueError(
                f"{ARCH} keeps its KV caches in fp16, got --kv-cache-dtype "
                f"{cache_config.cache_dtype}; drop it."
            )
        if not cache_config.user_specified_block_size:
            cache_config.block_size = KV_BLOCK_SIZE
            cache_config.user_specified_block_size = True
        elif cache_config.block_size != KV_BLOCK_SIZE:
            raise ValueError(
                f"{ARCH} needs --block-size {KV_BLOCK_SIZE} (the attention "
                f"and conv-state pages are equal there), got "
                f"{cache_config.block_size}."
            )
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size != 1:
            raise ValueError(
                f"{ARCH} does not support pipeline parallelism, got "
                f"--pipeline-parallel-size {pp_size}."
            )
        scheduler_config = vllm_config.scheduler_config
        if not sm70_envs.VLLM_SM70_INKLING_ASYNC_SCHED:
            if scheduler_config.async_scheduling:
                raise ValueError(
                    f"{ARCH} with VLLM_SM70_INKLING_ASYNC_SCHED=0 needs "
                    "--no-async-scheduling."
                )
            scheduler_config.async_scheduling = False
        inkling2 = sm70_config(vllm_config).inkling2
        spec_schedule = spec_policy.apply(vllm_config, inkling2)
        compilation_config = vllm_config.compilation_config
        cudagraph_mode = compilation_config.cudagraph_mode
        decode_graph = decode_graph_enabled(vllm_config)
        if cudagraph_mode is None:
            if decode_graph:
                cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            elif envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
                cudagraph_mode = CUDAGraphMode.PIECEWISE
            else:
                cudagraph_mode = CUDAGraphMode.NONE
            compilation_config.cudagraph_mode = cudagraph_mode
        elif cudagraph_mode.has_full_cudagraphs() and not decode_graph:
            raise ValueError(
                f"{ARCH} captures FULL CUDA graphs only with "
                "VLLM_SM70_INKLING_DECODE_GRAPH on and no visit dump, got "
                f"cudagraph_mode {cudagraph_mode.name}; use "
                "-cc.cudagraph_mode=piecewise or none."
            )
        elif cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL:
            raise ValueError(
                f"{ARCH} captures FULL CUDA graphs of decode-only batches "
                f"only, got cudagraph_mode {cudagraph_mode.name}; use "
                "-cc.cudagraph_mode=full_decode_only, piecewise or none."
            )
        elif (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE
            and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        ):
            logger.warning(
                "%s: cudagraph_mode %s without VLLM_USE_BREAKABLE_CUDAGRAPH=1 "
                "captures no graph of mixed batches (the model's ops are "
                "eager breaks); set it to 1 or use "
                "-cc.cudagraph_mode=full_decode_only or none.",
                ARCH,
                cudagraph_mode.name,
            )
        query_len = decode_query_len(vllm_config)
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            _cap_decode_graph_batch(
                compilation_config, scheduler_config, query_len, spec_schedule
            )
        hf_config = getattr(model_config, "hf_config", None)
        tuning = tm_tuning_env(compilation_config, hf_config, query_len)
        if tuning:
            os.environ.update(tuning)
            logger.info(
                "Inkling 2 SM70: TurboMind NVFP4 expert stages tuned up to %s "
                "routed slots (%s); VLLM_SM70_INKLING_TM_TUNE=0 turns it off.",
                tuning[TM_TUNE_KNOBS[1]],
                ", ".join(f"{k}={v}" for k, v in tuning.items()),
            )
        if hf_config is not None:
            setattr(
                hf_config,
                IMAGE_ENCODER_TOKEN_CAP_ATTR,
                inkling2.max_image_encoder_tokens,
            )
        if spec_schedule is not None:
            logger.info(
                "Inkling 2 SM70: speculative-decoding schedule %s "
                "(first and last request count, drafts verified).",
                spec_schedule,
            )
        _log_routes_once(
            inkling2.visit_dump_dir,
            compilation_config.cudagraph_mode.name,
            {False: "off", True: "on"}.get(
                scheduler_config.async_scheduling, "upstream's default"
            ),
        )
