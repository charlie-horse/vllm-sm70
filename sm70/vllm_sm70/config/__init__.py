# SPDX-License-Identifier: Apache-2.0
"""The typed SM70 kernel policy for the plugin.

``Sm70Config`` is a frozen pydantic model (extra keys are rejected).
``sm70_config(vllm_config)`` reads ``additional_config["sm70"]`` (or
``kernel_config.sm70``), applies the kill switches from ``vllm_sm70.envs``
and caches the result per vllm_config. A kill switch can only turn a field
off; configuration is the only way to turn something on.
``compute_hash()`` covers only the graph-affecting fields. ``CONSTANTS``
holds fixed tuning values that are not configurable.

This package imports nothing from vllm (only pydantic and vllm_sm70.envs).
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vllm_sm70 import envs

QuantBackend = Literal["auto", "turbomind"]
VerifierMode = Literal["auto", "on", "off"]


class Inkling2Config(BaseModel):
    """Inkling 2 (inkling2_sm70) settings: ``additional_config["sm70"]["inkling2"]``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    visit_dump_dir: str | None = None
    """Debug: directory the model writes its per-visit state dump to. None
    (default) writes nothing. Set only through --additional-config; there is
    no environment variable, because the environment can't turn a feature
    on. The dump runs eagerly, so this is not a graph field."""

    max_image_encoder_tokens: int = Field(default=10817, ge=2)
    """Largest vision-encoder sequence of one image: merge * merge patches
    per image token plus the class token (10817 is 2704 image tokens, e.g.
    2080 x 2080 pixels, for Inkling 2 Mini). A larger image is rejected, and
    the profiling image has this size. The encoder runs eagerly: not a
    graph field."""

    expert_act_fake_quant: bool = False
    """Numerics emulation: fake-quantize (fp32) each expert GEMM's input and
    post-activation value as upstream's NVFP4 (layers 0-19) and MXFP8
    (layers 20-23) activation GEMMs quantize them, on every expert path
    (models/inkling2_sm70/act_quant.py). Off (default): outputs are
    unchanged. Set only through --additional-config. Not a graph field: the
    model is not compiled, and its decode graphs are captured with it."""

    dspark_spec_max_reqs: int = Field(default=1, ge=1)
    """With a DSpark --speculative-config: a step of at most this many
    scheduled requests verifies drafts; a larger one decodes one token per
    request (models/inkling2_sm70/spec_policy.py). Not a graph field: it
    sets which decode graphs are captured, not what they compute.
    Kill: VLLM_SM70_INKLING_DSPARK_AUTO=0 (every step verifies)."""

    dspark_spec_tokens: int = Field(default=7, ge=1)
    """Draft tokens a speculating step verifies, at most
    num_speculative_tokens; the draft still drafts its whole block. Not a
    graph field (as dspark_spec_max_reqs)."""


class Sm70Config(BaseModel):
    """SM70 (Volta) kernel policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quant_backend: QuantBackend = "auto"
    """Provider for SM70 quantized linear and MoE layers. "auto" resolves to
    TurboMind; there is no Marlin route. VLLM_SM70_FP8_TURBOMIND=0 makes the
    SM70 FP8 path refuse with an error (see require_fp8_turbomind); there is
    no SM70 fallback, since upstream FP8 refuses capability 70."""

    nvfp4_gemv: bool = True
    """Flint NVFP4 W4A16 dense GEMV for M<=2 rows, limited to a fixed set
    of TP2 (K, N) shapes. Larger M uses the TurboMind GEMM.
    Kill: VLLM_SM70_MODELOPT_NVFP4_GEMV=0."""

    flint_moe_qpn_decode: bool = True
    """Flint NVFP4 routed-expert decode for M<=2 tokens at TP2: two kernels,
    W13 with fused SwiGLU, then W2 with the reduce. Gates: the layer contract
    must be in _DECODE_QPN_CONTRACTS, the native ops must be present, and the
    scales must be finite. M>2 uses the TurboMind grouped route.
    Kill: VLLM_SM70_FLINT_MOE_QPN_DECODE=0."""

    flint_moe_compact: bool = True
    """Flint NVFP4 routed experts when the dense-stage route serves a call
    with M*top_k <= 80 routed slots (M 3..13 at top-6, plus M<=2 when the
    QPN decode does not take it): each slot is its own one-row group, so each
    stage is one compact grouped TurboMind launch. Without it, M*top_k in
    18..64 runs TurboMind's per-expert loop.
    Kill: VLLM_SM70_FLINT_MOE_COMPACT=0."""

    fp8_moe_decode: bool = True
    """Qwen FP8 MoE single-token decode route: the compact active-expert
    decode, the indexed W2 and the weighted native reduce. W13 stays dense
    and permute stays generic. Applies when M==1. Larger M uses the batched
    FP8 MoE GEMM.
    Kill: VLLM_SM70_FP8_MOE_DECODE=0."""

    fp8_moe_compact_decode: bool = True
    """Within the fp8_moe_decode route (M==1): the compact op, one fused
    ``fp8_moe_single_token_sm70_out`` launch with exact_layout (see the
    FP8_MOE_*COMPACT* CONSTANTS below). Off: the decomposed single-token plan
    (active-expert W13, SiLU*up, indexed W2, unpermute). It has no effect
    when fp8_moe_decode is off (batched only).
    Kill: VLLM_SM70_FP8_MOE_COMPACT_DECODE=0."""

    flint_qk_rope_kv: bool = True
    """Flint fused q/k RMSNorm + RoPE + FP16 paged KV-cache write (one kernel).
    Gates: SM70, FP16 default dtype, FP16 KV (kv_cache_dtype auto), and the
    fused_route_rejection shape checks. The Flint router, sampler, TP2
    AR+RMSNorm and sink-fused decode have their own switches.
    Kill: VLLM_SM70_FLINT_QK_ROPE_KV=0."""

    fp8_qpn8: bool | None = None
    """The sm70_native "qpn8" selector for the DFlash2 verifier, not block
    QPN8 (see fp8_block_qpn8, which does not read it). None (auto) means on
    only when the DFlash2 verifier contract qualifies and quantization ==
    "fp8" (fp8_qpn8_enabled).
    Kill: VLLM_SM70_FP8_QPN8=0."""

    fp8_block_qpn8: bool = True
    """Block QPN8: weight-only FP8 decode GEMV for block-128 dense linears
    (fp16 in/out, static 128x128 block weights, N and K multiples of 128).
    The flag alone does not admit a layer; fp8_block_qpn8_turbomind_copy
    decides. Kill: VLLM_SM70_FP8_BLOCK_QPN8=0."""

    fp8_block_qpn8_turbomind_copy: bool | None = None
    """Block QPN8 vs a second, TurboMind-packed weight copy. rows =
    max_num_seqs * (num_speculative_tokens + 1); max_m =
    qpn8_max_m(block_scaled=True): 8, or 32 with VLLM_SM70_FP8_QPN8_M32=1.
    None (default): admit block QPN8 with no copy iff rows <= max_m;
    otherwise do not admit it (TurboMind only, one copy). True: admit and
    keep the TurboMind copy for rows above max_m (two copies of the dense
    FP8 weights). False: admit with no copy; rows above max_m use the
    QPN8 dense fallback. No kill switch (a policy, not a feature)."""

    dflash2_verifier: VerifierMode = "auto"
    """The DFlash2 verifier bundle (CONSTANTS["DFLASH2_VERIFIER_FLAGS"]
    plus fp8_qpn8). "auto" turns it on iff dflash2_verifier_qualified holds.
    Kill: VLLM_SM70_DFLASH2_VERIFIER=0."""

    shared_moe_overlap: bool = False
    """Qwen3-Next/3.5-MoE shared-expert stream overlap. Off by default.
    Kill: VLLM_SM70_SHARED_MOE_OVERLAP=0."""

    greedy_token_fastpath: bool = True
    """Vocab-parallel top-1 for all-greedy batches (temperature 0): each TP
    rank takes the argmax of its own LM-head logit shard and the ranks
    all-gather only (value, token id) pairs, instead of all-gathering the
    full-vocabulary logits and running the upstream Sampler. Same token as
    upstream's argmax (first index on ties). Admitted only for models that
    opt in (``supports_sm70_topk_sampler``), on SM70, without spec decode,
    structured output, logprobs, penalties, allowed/bad words, thinking
    budget or active non-argmax-invariant logits processors
    (vllm_sm70/sample_greedy.py). Not a graph field: sampling runs outside
    the compiled graph.
    Kill: VLLM_SM70_GREEDY_TOKEN_FASTPATH=0."""

    inkling2: Inkling2Config = Inkling2Config()
    """The Inkling 2 section (debug dump, image size cap, activation
    fake-quant, speculative-decoding policy). No field kill switch;
    not a graph field."""


GRAPH_FIELDS: tuple[str, ...] = (
    "quant_backend",
    "flint_qk_rope_kv",
    "fp8_qpn8",
    "fp8_block_qpn8",
    "fp8_block_qpn8_turbomind_copy",
    "dflash2_verifier",
    "flint_moe_qpn_decode",
    "flint_moe_compact",
    "fp8_moe_decode",
    "fp8_moe_compact_decode",
    "nvfp4_gemv",
    "shared_moe_overlap",
)
"""Fields that change the compiled graph. greedy_token_fastpath is not one:
sampling runs eagerly, outside the compiled graph. Nor is inkling2 (read eagerly)."""

# Kill switch name -> (field, value forced when engaged). VLLM_SM70_FP8_TURBOMIND,
# VLLM_SM70_QUANT and VLLM_SM70_DFLASH2_EMULATION are env-only kills (no field).
KILL_SWITCHES: dict[str, tuple[str, Any]] = {
    "VLLM_SM70_MODELOPT_NVFP4_GEMV": ("nvfp4_gemv", False),
    "VLLM_SM70_FLINT_MOE_QPN_DECODE": ("flint_moe_qpn_decode", False),
    "VLLM_SM70_FLINT_MOE_COMPACT": ("flint_moe_compact", False),
    "VLLM_SM70_FP8_MOE_DECODE": ("fp8_moe_decode", False),
    "VLLM_SM70_FP8_MOE_COMPACT_DECODE": ("fp8_moe_compact_decode", False),
    "VLLM_SM70_FLINT_QK_ROPE_KV": ("flint_qk_rope_kv", False),
    "VLLM_SM70_FP8_QPN8": ("fp8_qpn8", False),
    "VLLM_SM70_FP8_BLOCK_QPN8": ("fp8_block_qpn8", False),
    "VLLM_SM70_DFLASH2_VERIFIER": ("dflash2_verifier", "off"),
    "VLLM_SM70_SHARED_MOE_OVERLAP": ("shared_moe_overlap", False),
    "VLLM_SM70_GREEDY_TOKEN_FASTPATH": ("greedy_token_fastpath", False),
}


def apply_kill_switches(cfg: Sm70Config) -> Sm70Config:
    """Return cfg with every engaged kill switch applied (it can only disable).

    Reads each switch as ``vllm_sm70.envs.<NAME>`` (True means enabled, so
    only "0" acts).
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
    """Resolve fp8_qpn8=None (auto): on iff the DFlash2 verifier is on."""
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
    """The DFlash2 verifier contract: Qwen3_5 arch, fp16, hidden 5120, 24/4
    heads x 256, method dflash with k=7, draft selector_top_k 16, PP1, no
    DBO/ubatching. Target quantization and TP are not part of it. Only
    getattr on the three configs, so no vllm import. Block QPN8 does not
    use it."""
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
# Fixed values of knobs that are not configurable.
# --------------------------------------------------------------------------
CONSTANTS: dict[str, Any] = {
    # Dense GEMM / native TurboMind tuning is owned by the ops:
    # csrc/turbomind/ops/tm_config.h.
    # FP8 MoE stage modes
    "FP8_MOE_BATCHED_GEMM": True,
    "FP8_MOE_W13_PER_EXPERT": False,
    "FP8_MOE_W2_PER_EXPERT": False,
    "FP8_MOE_PERMUTE_WITH_SCRATCH": True,
    # FP8 MoE single-token compact route
    "FP8_MOE_LEGACY_COMPACT": True,
    "FP8_MOE_COMPACT_EXACT_LAYOUT": True,
    "FP8_MOE_COMPACT_NATIVE_UNPERMUTE": False,
    "FP8_MOE_COMPACT_DECOMPOSED": False,
    "MOE_SINGLE_TOKEN_W13": ("dense",),
    "MOE_SINGLE_TOKEN_W2": "indexed",
    "MOE_SINGLE_TOKEN_REDUCE": "weighted",
    "MOE_SINGLE_TOKEN_PERMUTE_FASTPATH": False,
    # NVFP4 MoE routes; the Qwen3.8 ones are exact-TP4 contracts
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
    "FLINT_MOE_PERSISTENT": False,
    "CF9492A_FLINT_FUSED_ROUTER": True,  # M<=2, E256/K6, no DP/EP
    "CF9492A_FLINT_SAMPLER": True,  # <=2 rows, random, top_k<=20
    # DFlash2 draft proposal
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
