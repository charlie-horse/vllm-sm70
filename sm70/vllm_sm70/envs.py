# SPDX-License-Identifier: Apache-2.0
"""All `VLLM_SM70_*` environment variables, in one registry.

Rules:
- Kill switches: only the value "0" acts, and it turns a feature off. The
  environment can never turn a feature on; defaults are decided in code.
- Each area adds its variables to its own section in `_SECTIONS`.
- `merge_into_vllm()` adds every getter to `vllm.envs.environment_variables`
  (setdefault, never overriding upstream). This has two effects:
  `Platform.validate_environ` (UP platforms/interface.py:1458-1471) accepts
  the names, and `vllm.envs.compile_factors()` (UP envs.py:2353) hashes them
  into the torch.compile cache key. Lookups through `vllm.envs.<NAME>`
  still work after `enable_envs_cache()` (UP envs.py:2310), because the
  cached `__getattr__` (UP envs.py:2293) reads the same dict on a miss.

This module imports nothing from vllm at import time, so it is safe to
import during platform resolution and in tests without vllm.
"""

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


def _build_time(name: str) -> Callable[[], None]:
    """Getter for a variable read only by setup.py / CMakeLists.txt.

    Registered so that `validate_environ` accepts it when it is still set at
    runtime (normal in CI and dev shells). It always returns None, whatever
    the environment says, so it adds a constant to compile_factors() and can
    never split the compile cache. (Upstream's exclusion list, `ignored_factors`
    in UP envs.py:2358-2450, is a local set inside compile_factors(), so a
    plugin cannot add to it.)
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
            "VLLM_SM70_FV100_FUSED_DECODE",  # CMake option / C++ define (K05/K14)
        )
    },
    "core": {
        # Debug logging only. Not a kill switch, so the only-"0"-acts rule
        # does not apply: any value other than "" or "0" turns it on.
        "VLLM_SM70_PLUGIN_DEBUG": _debug_flag("VLLM_SM70_PLUGIN_DEBUG"),
        # Not meant to be set: a derived getter so that compile_factors()
        # changes whenever _sm70_C is rebuilt (abi_version + .so sha256).
        # vLLM hashes .py files, not the .so (UP compilation/backends.py:1064).
        "VLLM_SM70_EXT_FINGERPRINT": _ext_fingerprint,
        # Escape hatch: "0" turns the SM70 fail-fast on failed areas
        # (Sm70CudaPlatform._enforce_strict) into a warning.
        "VLLM_SM70_PLUGIN_STRICT": kill_switch("VLLM_SM70_PLUGIN_STRICT"),
    },
    # TurboMind tuning knobs, read only by C++ (tm_config.h, K02). Values are
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
    },
    "attention": {
        # K06: "0" -> upstream selection (TRITON_ATTN) instead of Flash-V100.
        "VLLM_SM70_FLASH_V100": kill_switch("VLLM_SM70_FLASH_V100"),
        # K07: "0" -> verify rows take the causal paged prefill (K06) instead
        # of the smallq -> XQA expansion; graphs then cover 1-token decode.
        "VLLM_SM70_FLASH_V100_SPEC": kill_switch("VLLM_SM70_FLASH_V100_SPEC"),
    },
    # Kai (K08 FlashQLA GDN, U2 FLA launch schedule). Kill switches only.
    "gdn": {
        # "0": GDN chunked prefill stays on upstream Triton/FLA (read when
        # ChunkGatedDeltaRule is built, vllm_sm70/gdn/kernels.py).
        "VLLM_SM70_GDN_FLASHQLA_PREFILL": kill_switch(
            "VLLM_SM70_GDN_FLASHQLA_PREFILL"
        ),
        # "0": non-spec GDN decode stays on upstream's packed Triton decode
        # (read on each flashqla_decode() call). Fork name kept.
        "VLLM_SM70_GDN_DECODE_FLASHQLA": kill_switch("VLLM_SM70_GDN_DECODE_FLASHQLA"),
        # "0": patch U2 off; the FLA recurrent kernels keep upstream's BV on
        # sm_70 (read by vllm/third_party/flash_linear_attention/ops/
        # fused_recurrent.py, which only checks for "0").
        "VLLM_SM70_FLA_RECURRENT_SCHED": kill_switch("VLLM_SM70_FLA_RECURRENT_SCHED"),
    },
    "fused_moe": {},
    # Mara (M1 sm70-config, M2 sm70-quant-gates). Kill switches only.
    "quant": {
        # M2: "0" skips registering the SM70 "fp8"/"modelopt_fp4" configs;
        # upstream's then refuse SM70 (min capability 75).
        "VLLM_SM70_QUANT": kill_switch("VLLM_SM70_QUANT"),
        # "0": the SM70 FP8 TurboMind path refuses with a clear error (no
        # Marlin route in v1; upstream FP8 refuses SM70).
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
        # "0": Sm70Config.fp8_qpn8=False.
        "VLLM_SM70_FP8_QPN8": kill_switch("VLLM_SM70_FP8_QPN8"),
        # "0": Sm70Config.fp8_block_qpn8=False (M8 block QPN8 decode GEMV;
        # the Qwen FP8 dense linears stay on M6 TurboMind).
        "VLLM_SM70_FP8_BLOCK_QPN8": kill_switch("VLLM_SM70_FP8_BLOCK_QPN8"),
    },
    "models": {
        # "0": Sm70Config.flint_moe_qpn_decode=False (Flint NVFP4 M<=2 decode).
        "VLLM_SM70_FLINT_MOE_QPN_DECODE": kill_switch(
            "VLLM_SM70_FLINT_MOE_QPN_DECODE"
        ),
        # "0": Sm70Config.flint_qk_rope_kv=False (fused q/k norm+RoPE+KV write).
        "VLLM_SM70_FLINT_QK_ROPE_KV": kill_switch("VLLM_SM70_FLINT_QK_ROPE_KV"),
        # "0": Sm70Config.dflash2_verifier="off".
        "VLLM_SM70_DFLASH2_VERIFIER": kill_switch("VLLM_SM70_DFLASH2_VERIFIER"),
        # "0": DFlash2 draft BF16 emulation off (read by M7).
        "VLLM_SM70_DFLASH2_EMULATION": kill_switch("VLLM_SM70_DFLASH2_EMULATION"),
        # "0": Sm70Config.shared_moe_overlap=False.
        "VLLM_SM70_SHARED_MOE_OVERLAP": kill_switch("VLLM_SM70_SHARED_MOE_OVERLAP"),
    },
    "passes": {},
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
