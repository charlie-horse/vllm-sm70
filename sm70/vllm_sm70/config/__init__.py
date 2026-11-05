# SPDX-License-Identifier: Apache-2.0
"""M1 sm70-config: the typed SM70 policy for the plugin.

What: ``Sm70Config``, a frozen pydantic model with 11 fields (extra keys are
  rejected), and ``sm70_config(vllm_config)``. The helper reads
  ``additional_config["sm70"]`` (or ``kernel_config.sm70``), applies the
  kill-only env switches from ``vllm_sm70.envs``, and caches the result per
  vllm_config. ``compute_hash()`` covers only the graph-affecting fields.
  ``CONSTANTS`` holds the tuned values that were env knobs in the fork.
Why: the 1Cat fork (charlie-horse/vLLM cf9492a) resolved about 300 env knobs
  on the Flint / Qwen FP8 / DFlash2 paths through LegacyInputs, PolicyDefaults
  and native ABI tuples. Defaults here are the *effective* values the c4130
  service ran with: fork defaults, plus the cc=7.0 platform overrides in
  vllm/platforms/runtime_defaults.py, with no VLLM_SM70_* env on the unit
  (FORK nix/module.nix:198-207, :329; HOST hosts/c4130/vllm/default.nix).
  The evidence chain is in docs/models/sm70-effective-defaults.md (repo root).
Kill-switch: VLLM_SM70_<THING>=0, registered in the "quant" and "models"
  sections of vllm_sm70/envs.py (K01 ``kill_switch``; only "0" acts) and read
  here only as ``vllm_sm70.envs.<NAME>``. It can only turn a field off.
  Configuration is the only way to turn something on.

This package imports nothing from vllm (only pydantic and vllm_sm70.envs).
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from vllm_sm70 import envs

QuantBackend = Literal["auto", "turbomind"]
VerifierMode = Literal["auto", "on", "off"]


class Sm70Config(BaseModel):
    """SM70 (Volta) kernel policy. Defaults are the effective cf9492a values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quant_backend: QuantBackend = "auto"
    """Provider for SM70 quantized linear and MoE layers. "auto" resolves to
    TurboMind on every c4130 path. Evidence: fork SM70_QUANT_BACKEND default
    'auto' (FORK envs.py:3276); fork SM70_FP8_TURBOMIND=1 (envs.py:8485) and
    fork SM70_NVFP4_TURBOMIND=1 (envs.py:8546); use_turbomind returns
    turbomind for "auto" when enabled (config/kernel.py:232-234). No override
    applies. Route-M (Marlin) is dropped for v1, so "marlin" is not accepted.
    Kill: VLLM_SM70_FP8_TURBOMIND=0 makes the SM70 FP8 path refuse with a
    clear error (see require_fp8_turbomind). There is no SM70 fallback:
    upstream FP8 refuses capability 70."""

    nvfp4_gemv: bool = True
    """Flint NVFP4 W4A16 dense GEMV for M<=2 rows, limited to the six TP2
    (K, N) shapes in sm70_turbomind.py:569-579. Larger M uses the TurboMind
    GEMM. Evidence: fork SM70_MODELOPT_NVFP4_GEMV=1 (envs.py:8669), loader
    alias kernel.py:62. No override applies.
    Kill: VLLM_SM70_MODELOPT_NVFP4_GEMV=0."""

    flint_moe_qpn_decode: bool = True
    """Flint NVFP4 routed-expert decode for M<=2 tokens at TP2: two kernels,
    W13 with fused SwiGLU, then W2 with the reduce. Gates: the layer contract
    must be in _DECODE_QPN_CONTRACTS, the native ops must be present, and the
    scales must be finite (nvfp4_sm70_moe.py:230-290). M>2 uses the TurboMind
    grouped route; with --kernel-config.sm70_skinny_moe false, TurboMind also
    handles M>2. The single-launch persistent variant stays off
    (fork SM70_FLINT_MOE_PERSISTENT=0, envs.py:5100).
    Evidence: fork SM70_FLINT_MOE_QPN_DECODE=1 (envs.py:5079;
    sm70_moe.py:472). No override applies.
    Kill: VLLM_SM70_FLINT_MOE_QPN_DECODE=0."""

    fp8_moe_decode: bool = True
    """Qwen FP8 MoE single-token decode route: the legacy compact
    active-expert decode (fork SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT=1),
    the indexed W2 (fork SM70_FP8_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH=1),
    and the weighted native reduce (fork SM70_MOE_SINGLE_TOKEN_UNPERMUTE_
    FASTPATH=1, envs.py:9262, resolves single_token_reduce to "weighted",
    sm70_moe.py:393-397; consumer fp8_sm70_moe.py:54). W13 stays dense and permute stays generic.
    Applies when M==1. Larger M uses the batched FP8 MoE GEMM
    (fork SM70_FP8_MOE_BATCHED_GEMM=1, envs.py:8857). Evidence: alias tables
    in sm70_moe.py:16-71. No override applies.
    Kill: VLLM_SM70_FP8_MOE_DECODE=0."""

    fp8_moe_compact_decode: bool = True
    """Within the fp8_moe_decode route (M==1): the fork's legacy compact op,
    one fused ``fp8_moe_single_token_sm70_out`` launch with exact_layout
    (FORK fp8_sm70_moe.py:547-554; frozen as CONSTANTS FP8_MOE_LEGACY_COMPACT,
    FP8_MOE_COMPACT_EXACT_LAYOUT, FP8_MOE_COMPACT_NATIVE_UNPERMUTE and
    FP8_MOE_COMPACT_DECOMPOSED below). Off: the decomposed single-token plan
    (active-expert W13, SiLU*up, indexed W2, unpermute). It has no effect
    when fp8_moe_decode is off (batched only).
    Kill: VLLM_SM70_FP8_MOE_COMPACT_DECODE=0."""

    flint_qk_rope_kv: bool = True
    """Flint fused q/k RMSNorm + RoPE + FP16 paged KV-cache write (one kernel).
    Gates: SM70, FP16 default dtype, FP16 KV (kv_cache_dtype auto), and the
    fused_route_rejection shape checks (FORK models/flint/model.py:312-330).
    Evidence: fork SM70_FLINT_QK_ROPE_KV=1 (envs.py:8622) ->
    KernelConfig.sm70_flint_qk_rope_kv (kernel.py:766-781). No override
    applies. v1 gates only this fusion; the Flint router, sampler, TP2
    AR+RMSNorm and sink-fused decode (all on at cf9492a) come later with
    their own switches. Kill: VLLM_SM70_FLINT_QK_ROPE_KV=0."""

    fp8_qpn8: bool | None = None
    """The fork's sm70_native "qpn8" selector (kernel_config.sm70_fp8.qpn8,
    FORK config/kernel.py:398), NOT block QPN8 (see fp8_block_qpn8). Reserved
    for the sm70_native qpn8 selector (M9/verifier); M8 does not read it.
    None (auto) means on only when the DFlash2 verifier contract qualifies
    and quantization == "fp8" (platforms/runtime_defaults.py:497-513;
    SM70_DFLASH2_VERIFIER_DEFAULTS sm70_dflash2.py:27). Effective values:
    Qwen3.8-27B True, Qwen3.6-35B-A3B False. Plain default 0 (envs.py:3880).
    Kill: VLLM_SM70_FP8_QPN8=0."""

    fp8_block_qpn8: bool = True
    """M8 block QPN8: weight-only FP8 decode GEMV for block-128 dense
    linears (K09). Evidence: FORK cf9492a kernel_config.sm70_fp8.block_qpn8
    = True (config/kernel.py:386); QPN8Fp8BlockScaledMMLinearKernel is first
    in _POSSIBLE_FP8_BLOCK_KERNELS for CUDA (kernels/linear/__init__.py:344)
    and its can_implement (kernels/linear/qpn/fp8_block.py:160-195) gates
    only on this flag, fp16 in/out, static 128x128 block weights, rank 2,
    N,K % 128 and the native ops. So it is on for both Qwen3.6-35B-A3B and
    Qwen3.8-27B (docs/models/sm70-effective-defaults.md). No override
    applies. Kill: VLLM_SM70_FP8_BLOCK_QPN8=0."""

    fp8_block_qpn8_turbomind_copy: bool | None = None
    """Block QPN8 vs a second, TurboMind-packed weight copy. rows =
    max_num_seqs * (num_speculative_tokens + 1); max_m = K09's
    qpn8_max_m(block_scaled=True): 8, or 32 with VLLM_SM70_FP8_QPN8_M32=1.
    None (default, safe): admit block QPN8 with no copy iff rows <= max_m;
    otherwise do NOT admit it (plain M6, one copy). True: admit and keep the
    TurboMind copy for rows above max_m (two copies; bench/explicit only).
    False: admit with no copy; rows above max_m use K09's dense fallback.
    None deliberately differs from the fork's auto, which kept the copy
    when rows > 8 (FORK config/kernel.py:388 block_qpn8_volta_turbomind_prefill,
    kernels/linear/qpn/fp8_block.py:204-213): the default must not double
    the dense FP8 weights on 32 GB V100s (~+12 GiB per GPU on Qwen3.8-27B
    at TP2). True reproduces the fork's two-copy behaviour. No kill switch
    (a policy, not a feature)."""

    dflash2_verifier: VerifierMode = "auto"
    """The DFlash2 verifier bundle: 17 fork SM70_DFLASH2_* flags plus FP8_QPN8
    (sm70_dflash2.py:20-39). "auto" turns it on iff
    sm70_dflash2_verifier_qualified holds (models/config.py:1087-1134): arch
    Qwen3_5*ForCausalLM/ConditionalGeneration, fp16, hidden 5120, heads 24/4,
    head_dim 256, method dflash, k=7, draft selector_top_k 16, PP1.
    Effective values: Qwen3.8 on, Qwen3.6 off (MoE arch), Flint n/a.
    Kill: VLLM_SM70_DFLASH2_VERIFIER=0."""

    shared_moe_overlap: bool = False
    """Qwen3-Next/3.5-MoE shared-expert stream overlap. Effectively OFF:
    VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP default 0 (envs.py:13347), and
    qwen3_next.py:245-249 disables the stream unless it is requested. Only
    the Qwen4Exp FP16 lane forces it on (model_executor/models/
    runtime_defaults.py:110-111), which does not apply on c4130.
    Kill: VLLM_SM70_SHARED_MOE_OVERLAP=0."""


GRAPH_FIELDS: tuple[str, ...] = (
    "quant_backend",
    "flint_qk_rope_kv",
    "fp8_qpn8",
    "fp8_block_qpn8",
    "fp8_block_qpn8_turbomind_copy",
    "dflash2_verifier",
    "flint_moe_qpn_decode",
    "fp8_moe_decode",
    "fp8_moe_compact_decode",
    "nvfp4_gemv",
    "shared_moe_overlap",
)
"""Fields that change the compiled graph. shared_moe_overlap is included
because the fork always hashes it: all layer aliases pre-Ampere, this one
alone elsewhere (policy_defaults.py:275-277).
Keep nvfp4_gemv in until it is tested."""

# Kill switch name -> (field, value forced when engaged). VLLM_SM70_FP8_TURBOMIND,
# VLLM_SM70_QUANT and VLLM_SM70_DFLASH2_EMULATION are env-only kills (no field).
KILL_SWITCHES: dict[str, tuple[str, Any]] = {
    "VLLM_SM70_MODELOPT_NVFP4_GEMV": ("nvfp4_gemv", False),
    "VLLM_SM70_FLINT_MOE_QPN_DECODE": ("flint_moe_qpn_decode", False),
    "VLLM_SM70_FP8_MOE_DECODE": ("fp8_moe_decode", False),
    "VLLM_SM70_FP8_MOE_COMPACT_DECODE": ("fp8_moe_compact_decode", False),
    "VLLM_SM70_FLINT_QK_ROPE_KV": ("flint_qk_rope_kv", False),
    "VLLM_SM70_FP8_QPN8": ("fp8_qpn8", False),
    "VLLM_SM70_FP8_BLOCK_QPN8": ("fp8_block_qpn8", False),
    "VLLM_SM70_DFLASH2_VERIFIER": ("dflash2_verifier", "off"),
    "VLLM_SM70_SHARED_MOE_OVERLAP": ("shared_moe_overlap", False),
}


def apply_kill_switches(cfg: Sm70Config) -> Sm70Config:
    """Return cfg with every engaged kill switch applied (it can only disable).

    Reads each switch as ``vllm_sm70.envs.<NAME>`` (K01 getter: True means
    enabled, so only "0" acts).
    """
    updates = {
        field: value
        for name, (field, value) in KILL_SWITCHES.items()
        if not getattr(envs, name)
    }
    return cfg.model_copy(update=updates) if updates else cfg


def fp8_turbomind_enabled() -> bool:
    """False when VLLM_SM70_FP8_TURBOMIND=0 (kill-only; config can't override)."""
    return envs.VLLM_SM70_FP8_TURBOMIND


def require_fp8_turbomind() -> None:
    """Raise when the SM70 FP8 path is killed. Without Marlin, nothing serves FP8 on SM70."""
    if not fp8_turbomind_enabled():
        raise RuntimeError(
            "VLLM_SM70_FP8_TURBOMIND=0: the SM70 FP8 TurboMind path is disabled, "
            "and v1 has no other FP8 kernel for compute capability 7.0 (upstream "
            "FP8 needs >= 7.5). Unset it, or serve a non-FP8 checkpoint."
        )


def _raw_sm70(vllm_config: Any) -> Any:
    additional = getattr(vllm_config, "additional_config", None)
    if isinstance(additional, dict) and "sm70" in additional:
        return additional["sm70"]
    kernel = getattr(vllm_config, "kernel_config", None)
    return getattr(kernel, "sm70", None) if kernel is not None else None


def parse_sm70_config(raw: Any) -> Sm70Config:
    """Validate a dict, an Sm70Config or None (defaults). Unknown keys raise."""
    if raw is None:
        return Sm70Config()
    if isinstance(raw, Sm70Config):
        return raw
    if isinstance(raw, dict):
        return Sm70Config.model_validate(raw)
    raise TypeError(f"additional_config['sm70'] must be a dict, got {type(raw)!r}")


_CACHE: dict[int, tuple[Any, Sm70Config]] = {}


def sm70_config(vllm_config: Any = None) -> Sm70Config:
    """Effective Sm70Config for this engine (config + kill switches), cached.

    ``vllm_config`` may be None (defaults, uncached) or any object with
    ``additional_config`` and/or ``kernel_config.sm70``.
    """
    if vllm_config is None:
        return apply_kill_switches(Sm70Config())  # defaults + kill switches
    key = id(vllm_config)
    hit = _CACHE.get(key)
    if hit is not None:
        ref, cfg = hit
        if (ref() if isinstance(ref, weakref.ref) else ref) is vllm_config:
            return cfg
    cfg = apply_kill_switches(parse_sm70_config(_raw_sm70(vllm_config)))
    try:
        ref: Any = weakref.ref(vllm_config, lambda _r, k=key: _CACHE.pop(k, None))
    except TypeError:
        ref = vllm_config  # not weak-referenceable: keep a strong reference
    _CACHE[key] = (ref, cfg)
    return cfg


def clear_cache() -> None:
    _CACHE.clear()


def compute_hash(config: Any) -> str:
    """sha256 over GRAPH_FIELDS of the *effective* config.

    ``config`` is required: an Sm70Config (hashed as given, so pass the
    result of sm70_config()), or a vllm_config, which goes through
    sm70_config() and so includes the kill switches.
    """
    cfg = config if isinstance(config, Sm70Config) else sm70_config(config)
    factors = {name: getattr(cfg, name) for name in GRAPH_FIELDS}
    return hashlib.sha256(
        json.dumps(factors, sort_keys=True).encode()
    ).hexdigest()


def fp8_qpn8_enabled(cfg: Sm70Config, *, verifier_qualified: bool) -> bool:
    """Resolve fp8_qpn8=None (auto) the way cf9492a did (runtime_defaults.py:508-513)."""
    if cfg.fp8_qpn8 is not None:
        return cfg.fp8_qpn8
    return dflash2_verifier_enabled(cfg, verifier_qualified=verifier_qualified)


def dflash2_verifier_enabled(cfg: Sm70Config, *, verifier_qualified: bool) -> bool:
    if cfg.dflash2_verifier == "auto":
        return verifier_qualified
    return cfg.dflash2_verifier == "on"


def dflash2_verifier_qualified(
    model_config: Any, speculative_config: Any, parallel_config: Any
) -> bool:
    """The fork's DFlash2 verifier contract (FORK models/config.py:1087-1134,
    sm70_dflash2_verifier_qualified): Qwen3_5 arch, fp16, hidden 5120, 24/4
    heads x 256, method dflash with k=7, draft selector_top_k 16, PP1, no
    DBO/ubatching. Target quantization and TP are not part of it (fork
    docstring). Only getattr on the three configs, so no vllm import.
    Reserved for the sm70_native qpn8 selector / DFlash2 verifier (M9);
    M8's block QPN8 does not use it."""
    if model_config is None or speculative_config is None or parallel_config is None:
        return False
    import torch

    draft = getattr(speculative_config, "draft_model_config", None)
    dflash_config = getattr(getattr(draft, "hf_config", None), "dflash_config", None)
    selector_top_k = (
        int(dflash_config.get("selector_top_k", 0) or 0)
        if isinstance(dflash_config, Mapping)
        else 0
    )
    text = getattr(model_config, "hf_text_config", None)
    archs = set(getattr(model_config, "architectures", ()) or ())
    return bool(
        archs & {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"}
        and getattr(model_config, "dtype", None) == torch.float16
        and getattr(text, "hidden_size", None) == 5120
        and getattr(text, "num_attention_heads", None) == 24
        and getattr(text, "num_key_value_heads", None) == 4
        and getattr(text, "head_dim", None) == 256
        and getattr(speculative_config, "method", None) == "dflash"
        and int(getattr(speculative_config, "num_speculative_tokens", 0) or 0) == 7
        and selector_top_k == 16
        and getattr(parallel_config, "pipeline_parallel_size", 0) == 1
        and not getattr(parallel_config, "enable_dbo", False)
        and int(getattr(parallel_config, "ubatch_size", 0) or 0) <= 1
    )


# --------------------------------------------------------------------------
# Frozen constants: the effective cf9492a values of the knobs that are no
# longer configurable. "override" marks values the cc=7.0 platform defaults
# changed from the plain envs.py default.
# --------------------------------------------------------------------------
CONSTANTS: dict[str, Any] = {
    # Dense GEMM / native TurboMind tuning (F16_DENSE_MAX_M, FP8/NVFP4_DENSE_
    # TUNE_MAX_M=64, BATCH_GEMM_LAYOUTS=1, QPN8 tile choices, workspace bytes,
    # ...) is owned by the ops: csrc/turbomind/ops/tm_config.h (K02/K03).
    # Effective values: docs/models/sm70-effective-defaults.md.
    # FP8 MoE stage modes (sm70_moe.py:16-71)
    "FP8_MOE_BATCHED_GEMM": True,  # envs.py:8857
    "FP8_MOE_W13_PER_EXPERT": False,
    "FP8_MOE_W2_PER_EXPERT": False,
    "FP8_MOE_PERMUTE_WITH_SCRATCH": True,
    # Resolved as in Sm70MoEFormatConfig.resolve (sm70_moe.py:319-452).
    "FP8_MOE_LEGACY_COMPACT": True,  # envs.py:9136 (default "1")
    "FP8_MOE_COMPACT_EXACT_LAYOUT": True,  # unset -> "1", sm70_moe.py:344-349
    "FP8_MOE_COMPACT_NATIVE_UNPERMUTE": False,  # unset -> "0", same lines
    "FP8_MOE_COMPACT_DECOMPOSED": False,  # unset -> "0", same lines
    "MOE_SINGLE_TOKEN_W13": ("dense",),  # compact/indexed W13 aliases all 0, :370-379
    "MOE_SINGLE_TOKEN_W2": "indexed",  # fork FP8 INDEXED_W2 alias =1, :380-391
    "MOE_SINGLE_TOKEN_REDUCE": "weighted",  # UNPERMUTE_FASTPATH=1, :393-397
    "MOE_SINGLE_TOKEN_PERMUTE_FASTPATH": False,  # envs.py:9247
    # NVFP4 MoE routes (sm70_moe.py:454); the Qwen3.8 ones are exact-TP4 contracts
    "NVFP4_MOE_GROUPED_PREFILL": True,
    "NVFP4_MOE_FAST_PREFILL": True,
    "NVFP4_MOE_FUSED_SWIGLU_PREFILL": True,
    "NVFP4_MOE_INDEXED_PREFILL": True,
    "NVFP4_MOE_FUSED_BATCH_W13": True,
    "NVFP4_MOE_FUSED_BATCH_W2": True,
    "NVFP4_MOE_W2_DIRECT_REDUCE": True,
    "NVFP4_MOE_GROUPED_MTP5": True,
    "NVFP4_MOE_GROUPED_DECODE": False,
    "NVFP4_MOE_RAW_SCALE": False,
    "NVFP4_MOE_GROUPED_EXPERT_ROWS": False,
    "FLINT_MOE_PERSISTENT": False,  # envs.py:5100
    # Effective at cf9492a but wired later (router and sampler: M10).
    # Attention sink decode and AR+RMSNorm are Kai's (K14/K17).
    "CF9492A_FLINT_FUSED_ROUTER": True,  # envs.py:5400; M<=2, E256/K6, no DP/EP
    "CF9492A_FLINT_SAMPLER": True,  # envs.py:8749; <=2 rows, random, top_k<=20
    # DFlash2 (sm70_dflash2.py:93, :144)
    "DFLASH2_PROPOSAL_TEMPERATURE_SCALE": 1.0,  # GLM5-only overrides do not apply
    "DFLASH2_PROPOSAL_TOP_P": 1.0,
    "DFLASH2_LOOKUP": {
        "adaptive": True,
        "agree": 0,
        "cheap_context": 0,
        "entry_streak": 2,
        "long_min": 6,
        "nmin_tail": 4,
        "nstrong": 6,
        "search": 1 << 30,
        "sticky": 3,
    },
    # The DFlash2 verifier bundle (used when dflash2_verifier resolves on)
    "DFLASH2_VERIFIER_FLAGS": (
        "fused_gdn_verify",
        "fused_gdn_combined_split",
        "context_pipeline",
        "context_kv_graph",
        "quant_lm_head",
        "fp32_logits",
        "qpn8_rerank",
        "verify_fastpath",
        "fused_gdn_metadata",
        "fused_gdn_norm",
        "fused_gdn_split",
        "fused_gemma_rms",
        "fixed_gemma_rms",
        "fused_smallq_metadata",
        "grouped_smallq_metadata",
        "sparse_target_rejection",
        "sharded_context_fc",
    ),
    "DFLASH2_ALWAYS_OFF": ("tp2_gdn_bv2", "fused_qkv_pack", "qpn8_rerank_shadow"),
}

__all__ = [
    "CONSTANTS",
    "GRAPH_FIELDS",
    "KILL_SWITCHES",
    "Sm70Config",
    "apply_kill_switches",
    "clear_cache",
    "compute_hash",
    "dflash2_verifier_enabled",
    "dflash2_verifier_qualified",
    "fp8_qpn8_enabled",
    "fp8_turbomind_enabled",
    "require_fp8_turbomind",
    "parse_sm70_config",
    "sm70_config",
]
