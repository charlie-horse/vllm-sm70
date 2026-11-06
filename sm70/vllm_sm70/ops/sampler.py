# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k candidate sampler ops (K16): wrappers and fakes over torch.ops._sm70_C.

The fork registered `sm70_local_top32_half` and
`sm70_sample_topk_candidates` in upstream's `_C` namespace from its
`_sm70_sampler_C` sidecar (fork vllm/v1/sample/ops/sm70_topk_candidates.py:63-69, :267,
:336-345). Here the two ops are part of `_sm70_C`, with the fork schemas
unchanged (csrc/sampler/sampler_ops.cpp). The sampling logic that uses them
is vllm_sm70/sample.py.
"""

from __future__ import annotations

import torch

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

NAMESPACE = "_sm70_C"

# Load _sm70_C before any has_op() check below (see ops/tm.py).
from vllm_sm70 import _ext

_ext.load()

OPS = ("sm70_local_top32_half", "sm70_sample_topk_candidates")

# Kernel constants (csrc/sampler/topk_candidates_sampler.cu:51-60).
LOCAL_K = 32  # kLocalK: candidates per TP rank
WIRE_WIDTH = 2 * LOCAL_K  # kWireWidth: 32 values, then 32 token ids
TILE = 2048  # kTile: logits per selection block
MAX_TILES = 128  # kMaxTiles
MAX_LOCAL_VOCAB = MAX_TILES * TILE  # 262144 logits per rank
MAX_CANDIDATES = 128  # kMaxCandidates: tp_size * LOCAL_K <= 128, so TP <= 4
MAX_TP = MAX_CANDIDATES // LOCAL_K
MAX_KERNEL_ROWS = 8  # kMaxRows
MAX_TOKEN_ID = 1 << 24  # token ids travel as exact FP32


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def available() -> bool:
    """Both K16 ops are present in the loaded _sm70_C."""
    return all(has_op(name) for name in OPS)


def _op(name: str):
    if not has_op(name):
        raise RuntimeError(
            f"SM70 sampler op {NAMESPACE}::{name} is not available. "
            "Build vllm_sm70._sm70_C with CUDA 12.x for sm_70."
        )
    return getattr(getattr(torch.ops, NAMESPACE), name)


def local_top32_half(
    logits: torch.Tensor, vocab_start: int, num_valid: int
) -> torch.Tensor:
    """[rows, 64] FP32 wire rows: the 32 largest of `logits[:, :num_valid]`
    (descending), then their global token ids (`vocab_start + column`)."""
    return _op("sm70_local_top32_half")(logits, vocab_start, num_valid)


def sample_topk_candidates(
    wire: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor | None,
    generator: torch.Generator | None,
    vocab_size: int,
    tp_size: int,
    overflow: torch.Tensor | None,
) -> torch.Tensor:
    """int32 [rows] token ids sampled from rank-major gathered wire rows."""
    return _op("sm70_sample_topk_candidates")(
        wire, temperature, top_k, top_p, generator, vocab_size, tp_size, overflow
    )


# ---------------------------------------------------------------------------
# Fake impls (shape/dtype only), registered only when the op exists.
# ---------------------------------------------------------------------------

if has_op("sm70_local_top32_half"):

    @register_fake("_sm70_C::sm70_local_top32_half")
    def _sm70_local_top32_half_fake(
        logits: torch.Tensor, vocab_start: int, num_valid: int
    ) -> torch.Tensor:
        return logits.new_empty((logits.shape[0], WIRE_WIDTH), dtype=torch.float32)


if has_op("sm70_sample_topk_candidates"):

    @register_fake("_sm70_C::sm70_sample_topk_candidates")
    def _sm70_sample_topk_candidates_fake(
        wire: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor | None,
        generator: torch.Generator | None,
        vocab_size: int,
        tp_size: int,
        overflow: torch.Tensor | None,
    ) -> torch.Tensor:
        return wire.new_empty((wire.shape[0] // tp_size,), dtype=torch.int32)
