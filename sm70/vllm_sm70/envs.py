# SPDX-License-Identifier: Apache-2.0
"""All `VLLM_SM70_*` environment variables, in one registry.

- Kill switches: only "0" acts, and it turns a feature off. Defaults are
  set in code; the environment never turns a feature on.
- Opt-in switches (`opt_in`): only "1" acts. Most add a check or a memset.
  Three select a kernel: VLLM_SM70_FP8_QPN8_M32, VLLM_SM70_TOP1_CUSTOM_AR
  and VLLM_SM70_INKLING_MERGED_PROJ.
- Route switches (`route_switch`): "0" forces a route off, "1" forces it on,
  anything else keeps the default route. Each area has its own section.
- `merge_into_vllm()` adds every getter to `vllm.envs.environment_variables`
  (never overriding upstream). Then `Platform.validate_environ` accepts the
  names and `vllm.envs.compile_factors()` hashes them into the torch.compile
  cache key. `vllm.envs.<NAME>` lookups work after `enable_envs_cache()`.

This module imports nothing from vllm at import time."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def kill_switch(name: str) -> Callable[[], bool]:
    """Getter that is True (feature enabled) unless `name` == "0"."""

    def getter() -> bool:
        return os.environ.get(name) != "0"

    getter.__name__ = f"env_{name}"
    return getter


def _debug_flag(name: str) -> Callable[[], bool]:
    """Diagnostics only (extra logging); it never changes what runs."""

    def getter() -> bool:
        return os.environ.get(name, "") not in ("", "0")

    getter.__name__ = f"env_{name}"
    return getter


def opt_in(name: str) -> Callable[[], bool]:
    """Getter that is True only when `name` == "1".

    For the few switches that default off. Most add a check or a memset;
    VLLM_SM70_FP8_QPN8_M32, VLLM_SM70_TOP1_CUSTOM_AR and
    VLLM_SM70_INKLING_MERGED_PROJ select a kernel.
    """

    def getter() -> bool:
        return os.environ.get(name) == "1"

    getter.__name__ = f"env_{name}"
    return getter


def route_switch(name: str) -> Callable[[], bool | None]:
    """Getter that is False when `name` == "0", True when it is "1" and None
    (the code's default route) otherwise."""

    def getter() -> bool | None:
        return {"0": False, "1": True}.get(os.environ.get(name, ""))

    getter.__name__ = f"route_{name}"
    return getter


def _build_time(name: str) -> Callable[[], None]:
    """Getter for a variable read only by setup.py / CMakeLists.txt.

    Registered so that `validate_environ` accepts it when it is still set at
    runtime (normal in CI and dev shells). It always returns None, whatever
    the environment says, so it adds a constant to compile_factors() and can
    never split the compile cache. (Upstream's `ignored_factors` list is
    local to compile_factors(), so a plugin cannot add to it.)
    """

    def getter() -> None:
        return None

    getter.__name__ = f"build_time_{name}"
    return getter


def _tuning(name: str) -> Callable[[], str | None]:
    """Raw value of a C++ tuning knob (read by csrc/turbomind/ops/tm_config.h
    via one cached getenv). Registered so validate_environ accepts it; the
    Python side never acts on it."""

    def getter() -> str | None:
        return os.environ.get(name)

    getter.__name__ = f"tuning_{name}"
    return getter


def _path(name: str) -> Callable[[], str | None]:
    """Optional file path; unset or empty means no file."""

    def getter() -> str | None:
        return os.environ.get(name) or None

    getter.__name__ = f"path_{name}"
    return getter


def _ext_fingerprint() -> str:
    from vllm_sm70._ext import fingerprint

    return fingerprint()


# Per-area sections, in register() order. Values are zero-argument getters.
_SECTIONS: dict[str, dict[str, Callable[[], Any]]] = {
    # Read by setup.py / CMakeLists.txt only; inert at runtime.
    "build": {
        name: _build_time(name)
        for name in (
            "VLLM_SM70_SKIP_EXT",  # setup.py: build without _sm70_C
            "VLLM_SM70_ALLOW_NO_CUDA",  # setup.py: CPU-only binding (tests)
            "VLLM_SM70_CUTLASS_DIR",  # setup.py -> CMake: CUTLASS headers
            "VLLM_SM70_REQUIRE_CUDA",  # CMake option
            "VLLM_SM70_ABI_VERSION",  # CMake variable / C++ define
            "VLLM_SM70_FV100_FUSED_DECODE",  # CMake option / C++ define
            # CMake $ENV: vLLM's patched csrc/ for csrc/custom_ar
            "VLLM_SM70_VLLM_CSRC_DIR",
        )
    },
    "core": {
        # Debug logging only. Not a kill switch, so the only-"0"-acts rule
        # does not apply: any value other than "" or "0" turns it on.
        "VLLM_SM70_PLUGIN_DEBUG": _debug_flag("VLLM_SM70_PLUGIN_DEBUG"),
        # Not meant to be set: a derived getter so that compile_factors()
        # changes whenever _sm70_C is rebuilt (abi_version + .so sha256).
        # vLLM hashes .py files, not the .so.
        "VLLM_SM70_EXT_FINGERPRINT": _ext_fingerprint,
        # "0": a failed area logs a warning instead of failing startup
        # (Sm70CudaPlatform._enforce_strict).
        "VLLM_SM70_PLUGIN_STRICT": kill_switch("VLLM_SM70_PLUGIN_STRICT"),
    },
    # TurboMind tuning knobs, read only by C++ (tm_config.h). Values are
    # passed through; defaults live in tm_config.h.
    "ops": {
        name: _tuning(name)
        for name in (
            "VLLM_SM70_AWQ_DENSE_TUNE_MAX_M",
            "VLLM_SM70_AWQ_MOE_BATCHED_ACTIVE_EXACT_W2",
            "VLLM_SM70_AWQ_MOE_DISPATCH_POLICY",
            "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS",
            "VLLM_SM70_AWQ_MTP_M5_FAST_SELECTOR",
            "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS",
            "VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY",
            "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE",
            "VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE",
            "VLLM_SM70_AWQ_TP2_FAST_SELECTOR",
            "VLLM_SM70_AWQ_TP2_FAST_TARGETS",
            "VLLM_SM70_AWQ_TP4_QKV_CTA64",
            "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES",
            "VLLM_SM70_BATCH_GEMM_LAYOUTS",
            "VLLM_SM70_DFLASH2_QPN8_RERANK",
            "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW",
            "VLLM_SM70_DFLASH2_SHARDED_CONTEXT_FC",
            "VLLM_SM70_F16_DENSE_MAX_M",
            "VLLM_SM70_F16_DENSE_TUNE_MAX_M",
            "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR",
            "VLLM_SM70_FP8_DENSE_TUNE_MAX_M",
            "VLLM_SM70_FP8_GROUPED_BMM_DECODE",
            "VLLM_SM70_FP8_MOE_PREPARE_VEC",
            "VLLM_SM70_FP8_MOE_SINGLE_TOKEN_PER_EXPERT_DISPATCH",
            "VLLM_SM70_FP8_PREFILL_CUTLASS",
            "VLLM_SM70_FP8_PREFILL_FAST_SELECTOR",
            "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS",
            "VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY",
            "VLLM_SM70_FP8_REUSE_IMPORTED_CACHE",
            "VLLM_SM70_FP8_SAFE_FAST_SELECTOR",
            "VLLM_SM70_FP8_TUNE_SMALL_SHAPES",
            # "0": Inkling 2 NVFP4 expert stages above 64 routed slots take
            # the per-expert loop instead of one grouped launch
            # (ops/tm.py nvfp4_moe_dense_stage_route).
            "VLLM_SM70_INKLING_TM_GROUPED_PREFILL",
            "VLLM_SM70_MXFP4_MOE_GROUPED_M8",
            "VLLM_SM70_MXFP4_MOE_GROUPED_M8_FAST_SELECTOR",
            "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M",
            "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS",
            "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL",
            "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS",
            "VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL",
            "VLLM_SM70_NVFP4_QWEN38_TP4_M1_FAST_SELECTOR",
            "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
            "VLLM_SM70_PROFILE_TRACE",
        )
    }
    | {
        # "1": block-scaled QPN8 also serves M=9..32 (ops/qpn8.py
        # qpn8_max_m). Off, block-scaled M>8 takes TurboMind (if held) or
        # dequantize + mm. Slower than TurboMind; logs a warning when used.
        "VLLM_SM70_FP8_QPN8_M32": opt_in("VLLM_SM70_FP8_QPN8_M32"),
        # "0": no DFlash2 draft FP16 M<=8 GEMM, and so no channel-FP8 draft
        # route either (ops/dflash2.py). It applies only to the Qwen3.8
        # DFlash2 setup that dflash2.k18_wanted accepts. It keeps a second
        # copy of the draft weights. Read once per engine.
        "VLLM_SM70_DFLASH2_FP16": kill_switch("VLLM_SM70_DFLASH2_FP16"),
        # "0": the DFlash2 draft's candidate top-k uses upstream's dense FP16
        # LM head instead of the FP8 rerank (ops/dflash2_rerank.py).
        # VLLM_SM70_DFLASH2_VERIFIER=0 also turns it off. Read when the draft
        # model is built.
        "VLLM_SM70_DFLASH2_QPN8_RERANK": kill_switch("VLLM_SM70_DFLASH2_QPN8_RERANK"),
    },
    "attention": {
        # "0": upstream attention selection (TRITON_ATTN) instead of Flash-V100.
        "VLLM_SM70_FLASH_V100": kill_switch("VLLM_SM70_FLASH_V100"),
        # "0": a captured XQA decode of <= 16 rows keeps the default
        # partition size instead of 256 (attention/flash_v100/dispatch.py).
        # Read at capture.
        "VLLM_SM70_FLASH_V100_SMALL_BATCH_PARTITION": kill_switch(
            "VLLM_SM70_FLASH_V100_SMALL_BATCH_PARTITION"
        ),
        # "0": DFlash2 verify rows take the XQA expansion (each token reads
        # the KV) instead of the verify kernel that reads it once
        # (csrc/flash_v100/verify_decode.cu). Read on every decode call; a
        # captured graph keeps its route.
        "VLLM_SM70_FLASH_V100_VERIFY_KERNEL": kill_switch(
            "VLLM_SM70_FLASH_V100_VERIFY_KERNEL"
        ),
        # "0": Flint sink layers decode with the partition kernel plus a
        # reducer (two launches) instead of one fused kernel. Read on every
        # decode call (ops/fv100.py:fused_decode_enabled).
        "VLLM_SM70_FLINT_ATTN_FUSED_DECODE": kill_switch(
            "VLLM_SM70_FLINT_ATTN_FUSED_DECODE"
        ),
        # "1": adds a memset of the arrival counters before every fused
        # decode launch. A safety check; the kernel re-zeroes them itself.
        "VLLM_SM70_FLINT_ATTN_FUSED_DECODE_ZERO_COUNTERS": opt_in(
            "VLLM_SM70_FLINT_ATTN_FUSED_DECODE_ZERO_COUNTERS"
        ),
        # "0": verify rows take the causal paged prefill instead of the
        # smallq -> XQA expansion; graphs then cover only 1-token decode.
        "VLLM_SM70_FLASH_V100_SPEC": kill_switch("VLLM_SM70_FLASH_V100_SPEC"),
        # "0": the verify expansion metadata is built with torch ops instead
        # of one Triton launch (attention/flash_v100/spec/smallq.py). Read
        # when each metadata builder is built.
        "VLLM_SM70_FLASH_V100_SMALLQ_METADATA": kill_switch(
            "VLLM_SM70_FLASH_V100_SMALLQ_METADATA"
        ),
    },
    # FlashQLA GDN kernels and the FLA launch schedule.
    "gdn": {
        # "0": GDN chunked prefill stays on upstream Triton/FLA (read when
        # ChunkGatedDeltaRule is built, vllm_sm70/gdn/kernels.py).
        "VLLM_SM70_GDN_FLASHQLA_PREFILL": kill_switch("VLLM_SM70_GDN_FLASHQLA_PREFILL"),
        # "0": non-spec GDN decode stays on upstream's packed Triton decode
        # (read on each flashqla_decode() call).
        "VLLM_SM70_GDN_DECODE_FLASHQLA": kill_switch("VLLM_SM70_GDN_DECODE_FLASHQLA"),
        # "0": the FLA recurrent kernels keep upstream's BV block size on
        # sm_70 (read by the patched fused_recurrent.py).
        "VLLM_SM70_FLA_RECURRENT_SCHED": kill_switch("VLLM_SM70_FLA_RECURRENT_SCHED"),
    },
    "fused_moe": {
        # "0": the Flint decode router takes upstream's gate GEMM and
        # topk_sigmoid instead of the fused router kernel (router.py).
        "VLLM_SM70_FLINT_FUSED_ROUTER": kill_switch("VLLM_SM70_FLINT_FUSED_ROUTER"),
    },
    "quant": {
        # "0": the SM70 "fp8"/"modelopt_fp4" configs are not registered;
        # upstream's then refuse SM70 (min capability 75).
        "VLLM_SM70_QUANT": kill_switch("VLLM_SM70_QUANT"),
        # "0": the SM70 FP8 TurboMind path refuses with an error (upstream
        # FP8 refuses SM70).
        "VLLM_SM70_FP8_TURBOMIND": kill_switch("VLLM_SM70_FP8_TURBOMIND"),
        # "0": Sm70Config.nvfp4_gemv=False (Flint NVFP4 W4A16 dense M<=2 GEMV).
        "VLLM_SM70_MODELOPT_NVFP4_GEMV": kill_switch("VLLM_SM70_MODELOPT_NVFP4_GEMV"),
        # "0": Sm70Config.fp8_moe_decode=False (Qwen FP8 MoE single-token route).
        "VLLM_SM70_FP8_MOE_DECODE": kill_switch("VLLM_SM70_FP8_MOE_DECODE"),
        # "0": Sm70Config.fp8_moe_compact_decode=False (M==1 decomposed plan
        # instead of the fused legacy compact op).
        "VLLM_SM70_FP8_MOE_COMPACT_DECODE": kill_switch(
            "VLLM_SM70_FP8_MOE_COMPACT_DECODE"
        ),
        # "0": Sm70Config.fp8_qpn8=False (no QPN8 FP8 decode GEMV).
        "VLLM_SM70_FP8_QPN8": kill_switch("VLLM_SM70_FP8_QPN8"),
        # "0": Sm70Config.fp8_block_qpn8=False (no block-scaled QPN8 decode
        # GEMV; FP8 dense linears stay on TurboMind).
        "VLLM_SM70_FP8_BLOCK_QPN8": kill_switch("VLLM_SM70_FP8_BLOCK_QPN8"),
    },
    "models": {
        # "0": Sm70Config.flint_moe_qpn_decode=False (Flint NVFP4 M<=2 decode).
        "VLLM_SM70_FLINT_MOE_QPN_DECODE": kill_switch("VLLM_SM70_FLINT_MOE_QPN_DECODE"),
        # "0": Sm70Config.flint_moe_compact=False (Flint NVFP4 MoE dense stage
        # with M*top_k <= 80 slots uses per-expert offsets; also the
        # Inkling 2 TurboMind experts).
        "VLLM_SM70_FLINT_MOE_COMPACT": kill_switch("VLLM_SM70_FLINT_MOE_COMPACT"),
        # "0": Inkling 2 NVFP4 experts stay on the fp16 torch path instead of
        # the TurboMind dense stage (models/inkling2_sm70/moe_tm.py, where
        # moe_tm.rejection admits them). Env-only, no field.
        "VLLM_SM70_INKLING_NVFP4_TM": kill_switch("VLLM_SM70_INKLING_NVFP4_TM"),
        # "0": Sm70Config.flint_qk_rope_kv=False (fused q/k norm+RoPE+KV write).
        "VLLM_SM70_FLINT_QK_ROPE_KV": kill_switch("VLLM_SM70_FLINT_QK_ROPE_KV"),
        # "0": Sm70Config.dflash2_verifier="off".
        "VLLM_SM70_DFLASH2_VERIFIER": kill_switch("VLLM_SM70_DFLASH2_VERIFIER"),
        # "0": DFlash2 draft BF16 emulation off.
        "VLLM_SM70_DFLASH2_EMULATION": kill_switch("VLLM_SM70_DFLASH2_EMULATION"),
        # "0": Sm70Config.shared_moe_overlap=False.
        "VLLM_SM70_SHARED_MOE_OVERLAP": kill_switch("VLLM_SM70_SHARED_MOE_OVERLAP"),
        # Route of the Inkling 2 norms, AttnRes mixes, message gates and
        # residual adds (inkling2_sm70/norms.py). Default: Triton kernels on
        # decode-only batches, torch ops otherwise. "0": torch ops on every
        # batch; "1": the kernels on every batch. Read once per forward.
        "VLLM_SM70_INKLING_TRITON_NORMS": route_switch(
            "VLLM_SM70_INKLING_TRITON_NORMS"
        ),
        # "0": the Inkling 2 paged short convs and the prologue's K/V convs
        # take the torch ops instead of one Triton kernel per call. Env-only
        # (no Sm70Config field), read per eager call (inkling2_sm70/sconv.py)
        # and fixed at capture in a replayed CUDA graph; tensors off the GPU
        # always take the torch ops.
        "VLLM_SM70_INKLING_TRITON_SCONV": kill_switch("VLLM_SM70_INKLING_TRITON_SCONV"),
        # "0": the one-query rows of Inkling 2 attention take the torch ops
        # instead of one Triton kernel per layer visit. Env-only (no
        # Sm70Config field), read per eager call (inkling2_sm70/
        # attention_torch.py) and fixed at capture in a replayed CUDA graph;
        # tensors off the GPU always take the torch ops.
        "VLLM_SM70_INKLING_TRITON_DECODE_ATTN": kill_switch(
            "VLLM_SM70_INKLING_TRITON_DECODE_ATTN"
        ),
        # "0": the Inkling 2 decode-attention kernel walks each row's keys in
        # one program per row and head instead of splitting them over several
        # programs and merging the partials (split-KV). Env-only (no
        # Sm70Config field), read per eager call (inkling2_sm70/
        # attention_torch.py) and fixed at capture in a replayed CUDA graph.
        "VLLM_SM70_INKLING_SPLIT_KV_ATTN": kill_switch(
            "VLLM_SM70_INKLING_SPLIT_KV_ATTN"
        ),
        # "0": the query rows of an Inkling 2 request with 2 to
        # SHORT_QUERY_LEN query tokens in an eager batch take the torch
        # attention over shared key chunks instead of the decode-attention
        # kernel. Env-only (no Sm70Config field), read per eager call
        # (inkling2_sm70/attention_torch.py); only where that kernel runs.
        "VLLM_SM70_INKLING_SHORT_QUERY_ATTN": kill_switch(
            "VLLM_SM70_INKLING_SHORT_QUERY_ATTN"
        ),
        # "0": Inkling 2 decode-only batches take the ops of mixed batches,
        # the cache backends declare no full-graph support and an
        # unset cudagraph_mode is not FULL_DECODE_ONLY (inkling2_sm70/
        # config.py, backend.py). Env-only (no Sm70Config field).
        "VLLM_SM70_INKLING_DECODE_GRAPH": kill_switch("VLLM_SM70_INKLING_DECODE_GRAPH"),
        # "0": the Inkling 2 experts held in the model dtype take the torch
        # expert sum (a host sync per batch, row by row in decode-only
        # batches) instead of the sync-free grouped route. Env-only (no
        # Sm70Config field), read per eager call (inkling2_sm70/moe.py) and
        # fixed at capture in a replayed CUDA graph; tensors off the GPU
        # always take the torch expert sum.
        "VLLM_SM70_INKLING_GROUPED_MOE": kill_switch("VLLM_SM70_INKLING_GROUPED_MOE"),
        # "0": Inkling 2 batches of at most 64 routed slots on the TurboMind
        # or grouped expert route take the torch router and the torch slot
        # sort instead of the routing kernels (inkling2_sm70/ops/routing.py).
        # Env-only (no Sm70Config field), read per eager call (inkling2_sm70/
        # moe.py) and fixed at capture in a replayed CUDA graph.
        "VLLM_SM70_INKLING_FUSED_ROUTING": kill_switch(
            "VLLM_SM70_INKLING_FUSED_ROUTING"
        ),
        # "0": the Inkling 2 config hook leaves the TurboMind NVFP4 tuning
        # knobs alone. Otherwise, when none of VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES,
        # VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS and VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M
        # is set, it sets them (1, decode graph cap x top-k, 0) for the
        # workers it spawns, so the expert stages of decode batches up to the
        # capture cap are tuned in eager warmup and reused by capture
        # (inkling2_sm70/config.py). Env-only, read once per engine config.
        "VLLM_SM70_INKLING_TM_TUNE": kill_switch("VLLM_SM70_INKLING_TM_TUNE"),
        # "0": the Inkling 2 config hook refuses every --speculative-config
        # and Sm70CudaPlatform.get_speculator_cls returns None
        # (inkling2_sm70/draft.py). Env-only, read once per engine config.
        "VLLM_SM70_INKLING_DSPARK": kill_switch("VLLM_SM70_INKLING_DSPARK"),
        # "0": the Inkling 2 config hook sets no speculative-decoding
        # schedule, so every decode step of a DSpark engine verifies
        # num_speculative_tokens drafts. Otherwise steps of more than
        # sm70.inkling2.dspark_spec_max_reqs requests decode one token per
        # request (inkling2_sm70/spec_policy.py). Env-only, read once per
        # engine config.
        "VLLM_SM70_INKLING_DSPARK_AUTO": kill_switch("VLLM_SM70_INKLING_DSPARK_AUTO"),
        # "0": the Inkling 2 draft (inkling2_sm70/dspark.py) runs the torch
        # forms of its norm, RoPE, block attention, SwiGLU, Markov gate and
        # soft-cap ops instead of the Triton kernels (inkling2_sm70/ops/
        # dspark.py). Env-only, read per eager call and fixed at capture in a
        # replayed CUDA graph.
        "VLLM_SM70_INKLING_DSPARK_KERNELS": kill_switch(
            "VLLM_SM70_INKLING_DSPARK_KERNELS"
        ),
        # "0": the Inkling 2 config hook turns async scheduling off when it
        # is unset and refuses --async-scheduling. Otherwise it leaves the
        # setting to upstream, which turns it on when unset
        # (inkling2_sm70/config.py). Env-only, read once per engine config.
        "VLLM_SM70_INKLING_ASYNC_SCHED": kill_switch("VLLM_SM70_INKLING_ASYNC_SCHED"),
        # "0": every TP rank holds the whole Inkling 2 LM head and computes
        # the logits of the whole vocabulary. Otherwise, at TP > 1 with a
        # vocabulary that splits evenly, each rank holds and computes its own
        # vocabulary rows and the ranks all-gather the logits (inkling2_sm70/
        # weights.py, model.py). Env-only, read when the model is built.
        "VLLM_SM70_INKLING_SHARDED_HEAD": kill_switch(
            "VLLM_SM70_INKLING_SHARDED_HEAD"
        ),
        # "0": every TP rank holds the whole Inkling 2 token embedding.
        # Otherwise, at TP > 1 with a padded vocabulary that splits evenly,
        # each rank holds its own vocabulary rows, looks up the ids in them
        # and the ranks all-reduce the rows (inkling2_sm70/weights.py,
        # loop.py). Env-only, read when the model is built.
        "VLLM_SM70_INKLING_SHARDED_EMBED": kill_switch(
            "VLLM_SM70_INKLING_SHARDED_EMBED"
        ),
        # "0": the Inkling 2 MXFP8 expert layers are dequantized to the model
        # dtype at load. Otherwise, in an fp16 model, they stay e4m3 codes
        # with their per-32 exponent scales and the expert GEMMs decode them
        # (inkling2_sm70/moe.py, ops/grouped_gemm.py). Env-only, read when a
        # layer is loaded.
        "VLLM_SM70_INKLING_MXFP8_EXPERTS": kill_switch(
            "VLLM_SM70_INKLING_MXFP8_EXPERTS"
        ),
        # "0": the small torch chains of Inkling 2 decode-only batches (the
        # prologue's q/r clips and relative projection, the attention output
        # gate, the TurboMind SwiGLU, the short convs' padding masks) run on
        # torch instead of one Triton kernel each (inkling2_sm70/
        # fused_small.py). Env-only (no Sm70Config field), read per eager
        # call and fixed at capture in a replayed CUDA graph; other batches
        # and tensors off the GPU always take the torch ops.
        "VLLM_SM70_INKLING_FUSED_SMALL_OPS": kill_switch(
            "VLLM_SM70_INKLING_FUSED_SMALL_OPS"
        ),
        # "0": Inkling 2 decode-only batches run one kernel per op of a layer
        # visit instead of the merged kernels of inkling2_sm70/
        # visit_fusion.py (short conv with residual and AttnRes mix, message
        # gates, prologue, router with route). Env-only (no Sm70Config
        # field), read per eager call and fixed at capture in a replayed CUDA
        # graph; other batches and tensors off the GPU never merge.
        "VLLM_SM70_INKLING_VISIT_FUSION": kill_switch(
            "VLLM_SM70_INKLING_VISIT_FUSION"
        ),
        # "0": the Inkling 2 dense projections of 1 to 8 tokens (qkvr with
        # its k/v inputs and packed taps, the attention output projection,
        # the LM head of 1 to 4 tokens) run on cuBLAS instead of one Triton
        # GEMV launch each (inkling2_sm70/gemv.py). Env-only (no Sm70Config
        # field), read per eager call and fixed at capture in a replayed CUDA
        # graph; tensors off the GPU always take cuBLAS.
        "VLLM_SM70_INKLING_SMALL_M_GEMV": kill_switch("VLLM_SM70_INKLING_SMALL_M_GEMV"),
        # "0": the one-row expert slots of the Inkling 2 TurboMind experts
        # (at most 80 routed slots) run the two TurboMind dense stages and
        # the SwiGLU kernel instead of the Triton NVFP4 GEMV over the same
        # banks, W13 with the SwiGLU in its epilogue (inkling2_sm70/gemv.py,
        # moe_tm.py). Env-only (no Sm70Config field), read per eager call
        # and fixed at capture in a replayed CUDA graph.
        "VLLM_SM70_INKLING_NVFP4_GEMV": kill_switch("VLLM_SM70_INKLING_NVFP4_GEMV"),
        # "1": Inkling 2 projections that share an input are packed into one
        # buffer and run as one GEMM. Off, each projection keeps its own
        # buffer and GEMM. Env-only (no Sm70Config field), read when the
        # model is built (inkling2_sm70/weights.py, layers.py).
        "VLLM_SM70_INKLING_MERGED_PROJ": opt_in("VLLM_SM70_INKLING_MERGED_PROJ"),
        # "0": the Inkling 2 model does not opt in to the V1 hidden-state
        # sampler, so every batch takes compute_logits (fp16 logit
        # all-gather) and the upstream Sampler. Otherwise an all-greedy
        # batch the greedy fast path admits takes the per-shard soft-capped
        # top-1 kernels and a (value, id) pair all-gather (inkling2_sm70/
        # greedy.py). Env-only (no Sm70Config field), read on every step.
        "VLLM_SM70_INKLING_GREEDY_TOP1": kill_switch("VLLM_SM70_INKLING_GREEDY_TOP1"),
        # "0": an Inkling 2 image larger than max_image_encoder_tokens allows
        # is refused with an error. Otherwise it is scaled down, keeping its
        # shape, until it fits (inkling2_sm70/processing.py). Env-only (no
        # Sm70Config field), read for each image.
        "VLLM_SM70_INKLING_IMAGE_RESIZE": kill_switch(
            "VLLM_SM70_INKLING_IMAGE_RESIZE"
        ),
        # "0": Inkling 2 image pixels are used as stored, ignoring the EXIF
        # Orientation tag, so a phone photo can reach the model sideways.
        # Otherwise they are turned upright first (inkling2_sm70/
        # processing.py). Env-only (no Sm70Config field), read for each image.
        "VLLM_SM70_INKLING_IMAGE_EXIF": kill_switch("VLLM_SM70_INKLING_IMAGE_EXIF"),
    },
    # Flint top-k candidate sampler.
    "sampler": {
        # "0": Sm70CudaPlatform.get_hidden_state_sampler() returns None, so
        # the V1 runner keeps compute_logits + the upstream Sampler (read
        # once per runner, vllm_sm70/_cuda_platform.py).
        "VLLM_SM70_FLINT_SAMPLER": kill_switch("VLLM_SM70_FLINT_SAMPLER"),
        # "0": Sm70Config.greedy_token_fastpath=False, so all-greedy batches
        # take compute_logits (full-vocab all-gather) + the upstream Sampler
        # (vllm_sm70/sample_greedy.py).
        "VLLM_SM70_GREEDY_TOKEN_FASTPATH": kill_switch(
            "VLLM_SM70_GREEDY_TOKEN_FASTPATH"
        ),
        # "1": the greedy top-1 fast path exchanges its (value, id) pairs with
        # the custom-AR kernel _sm70_C::custom_ar_top1_argmax instead of the
        # NCCL all-gather (get_top_tokens). Read per greedy step by
        # vllm_sm70/sample_greedy.py; env-only (no Sm70Config field). It can
        # return a different token than the all-gather on a near-tie.
        "VLLM_SM70_TOP1_CUSTOM_AR": opt_in("VLLM_SM70_TOP1_CUSTOM_AR"),
        # Diagnostics: any value but "" / "0" logs, per greedy fast-path step,
        # each rank's local (max, global id) pairs and the exchanged tokens
        # (vllm_sm70/sample_greedy.py; read once per process; syncs the
        # device).
        "VLLM_SM70_TOP1_DEBUG": _debug_flag("VLLM_SM70_TOP1_DEBUG"),
    },
    # Read by vllm_sm70.ar_rms_fusion when the composite pass first runs on
    # a graph.
    "passes": {
        # "0": the TP2 all-reduce + fused_add_rms_norm pattern is not applied;
        # graphs keep upstream's all_reduce and fused_add_rms_norm.
        "VLLM_SM70_FLINT_AR_RMSNORM": kill_switch("VLLM_SM70_FLINT_AR_RMSNORM"),
    },
    # TurboMind pre-capture warmup (vllm_sm70/warmup). Read once per
    # worker by Sm70CudaPlatform.kernel_warmup, before CUDA graph capture.
    "warmup": {
        # "0": no pre-capture TurboMind tuning and no TP plan exchange. Then
        # the first eager call of a shape tunes it per rank, and capture of
        # an untuned shape takes the default spec.
        "VLLM_SM70_TM_PRECAPTURE_WARMUP": kill_switch("VLLM_SM70_TM_PRECAPTURE_WARMUP"),
        # Optional TurboMind dispatch-cache file to pin specs.
        # Imported on every TP rank; the warmup calls its manifest
        # (<path>.plan.json) covers are not measured, the others are tuned
        # (with a WARNING) before the import. A manifest written for another
        # _sm70_C build, GPU or "ops" knob setting refuses the file (WARNING,
        # not loaded, tuned as without it). "{device}", "{rank}" and
        # "{arch}" are substituted. Unset: no file.
        "VLLM_SM70_GEMM_LUT_PATH": _path("VLLM_SM70_GEMM_LUT_PATH"),
        # Optional: after the warmup the source rank writes its whole
        # dispatch cache here (atomically, plus <path>.plan.json), for a
        # later run's VLLM_SM70_GEMM_LUT_PATH. Same substitutions. Unset: none.
        "VLLM_SM70_GEMM_LUT_EXPORT_PATH": _path("VLLM_SM70_GEMM_LUT_EXPORT_PATH"),
        # "1": with VLLM_SM70_GEMM_LUT_PATH set, a missing or unreadable
        # LUT file, or a missing, old-version, unreadable or mismatched
        # manifest (sha256, _sm70_C build, GPU or "ops" knobs) raises
        # RuntimeError on every rank, listing the differences, instead of
        # the WARNING + re-tune. Path unset: no effect. Kept out of "ops",
        # so it is not part of the LUT identity.
        "VLLM_SM70_GEMM_LUT_STRICT": opt_in("VLLM_SM70_GEMM_LUT_STRICT"),
    },
    # CUDA graph policy (vllm_sm70/graph_policy.py). Read once per
    # VllmConfig in Sm70CudaPlatform.apply_config_platform_defaults.
    "graph": {
        # "0": upstream's CUDA graph capture sizes instead of the SM70
        # tables ({1, 2, 4, 8, 16, 32} <= max_num_seqs plus max_num_seqs;
        # speculative: verifier, auxiliary and mixed sizes).
        "VLLM_SM70_GRAPH_CAPTURE_SIZES": kill_switch("VLLM_SM70_GRAPH_CAPTURE_SIZES"),
        # "0": upstream's CUDA graph memory estimate stays in the KV budget
        # (V1 runner). Otherwise VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 is
        # set when unset.
        "VLLM_SM70_GRAPH_MEMORY_POLICY": kill_switch("VLLM_SM70_GRAPH_MEMORY_POLICY"),
    },
}


def environment_variables() -> dict[str, Callable[[], Any]]:
    """All getters, flattened in section order."""
    flat: dict[str, Callable[[], Any]] = {}
    for section in _SECTIONS.values():
        for name, getter in section.items():
            assert name.startswith("VLLM_SM70_"), name
            assert name not in flat, f"duplicate env var {name}"
            flat[name] = getter
    return flat


def merge_into_vllm() -> list[str]:
    """Add our getters to vllm.envs. Idempotent; returns names newly added."""
    import vllm.envs as vllm_envs

    added = []
    target = vllm_envs.environment_variables
    for name, getter in environment_variables().items():
        if name not in target:
            target[name] = getter
            added.append(name)
    return added


def __getattr__(name: str) -> Any:
    getters = environment_variables()
    if name in getters:
        return getters[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables())
