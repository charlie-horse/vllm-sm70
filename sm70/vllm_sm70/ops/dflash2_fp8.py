# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M9a: channel-FP8 QPN8 route for the DFlash2 draft's M8 projections.

What: the DFlash2 draft decoder runs 8 query rows per drafting request
  (k=7 plus the anchor), so a batch of 1-4 requests is M = 8, 16, 24 or 32.
  For those M this route quantizes each flagged draft projection once after
  load to E4M3 with one FP32 scale per output channel, packs it with K09's
  ``fp8_qpn8_prepare_sm70`` and runs K09's ``fp8_qpn8_gemm_sm70_out``
  (channel scales, M <= 32). Any other M uses the original FP16 weight
  (``torch.mm``). The FP8 copy replaces K18's FP16 M8 packing for that layer,
  as in the fork ("The FP8 layout replaces the FP16 M8 packing").
How it hooks in: patch U1, through K18. ``ops.dflash2.dflash2_unquantized_gemm``
  (what ``Sm70CudaPlatform.get_unquantized_gemm`` returns) tries
  :func:`apply_dflash2_fp8_m8` first, then K18's FP16 M8 kernel, then
  ``F.linear``. M7's DFlash2 decoder-layer subclass flags the projections
  (``FP8_MARK_ATTR``) only when the DFlash2 verifier resolves on
  (``Sm70Config.dflash2_verifier``) and k == 7. No upstream line.
Switch: ``VLLM_SM70_DFLASH2_VERIFIER=0`` (M1) -> nothing is flagged, and K18's
  FP16 M8 path (or F.linear) runs. The route also needs K18's opt-in
  ``VLLM_SM70_DFLASH2_FP16=1``, because only then does U1 select the K18
  dispatcher; like K18 it was never run at TP2 in the fork.

Ported from fork cf9492a
vllm/model_executor/layers/quantization/sm70_dflash2_fp8.py (133 lines)
and its call sites vllm/model_executor/layers/linear.py:344-352, :396-403,
vllm/model_executor/models/qwen3_dflash2.py:278-297. Changes: the fork's
TP4-only (N, K) -> split table and its TP==4 condition are dropped; split-K is
derived from K so the same split serves M = 8..32 (K09's M=17..32 channel
kernels need split 12 or 16 with two accumulator chains).
"""

from __future__ import annotations

import logging

import torch

from vllm_sm70.ops import qpn8

logger = logging.getLogger("vllm.sm70.dflash2")

FP8_MARK_ATTR = "_sm70_dflash2_fp8_m8"
CODES_ATTR = "_sm70_dflash2_fp8_codes"
SCALES_ATTR = "_sm70_dflash2_fp8_scales"
# (N, K, split_k), set only on a prepared layer: the per-layer decision.
ROUTE_ATTR = "_sm70_dflash2_fp8_route"

ROWS_PER_REQUEST = 8
MAX_ROWS = 32
E4M3_MAX = 448.0
# Smallest scale kept: the fork's clamp (2**-24, the FP16 subnormal floor).
MIN_SCALE = 2.0**-24
# M=17..32 channel kernels admit only these (fp8_qpn8_sm70.cu TORCH_CHECKs).
ACCUMULATOR_CHAINS = 2
_SPLITS = (16, 12)

OPS = ("fp8_qpn8_prepare_sm70", "fp8_qpn8_gemm_sm70_out")


def draft_fp8_wanted(vllm_config) -> bool:
    """Should M7's draft decoder layer flag its projections for this route?

    Yes when the DFlash2 verifier resolves on (M1 ``dflash2_verifier``:
    "auto" = the fork contract, which already requires k == 7; "on" forces
    it, so k == 7 is checked again here as in FORK qwen3_dflash2.py:295-297)
    and ``VLLM_SM70_DFLASH2_VERIFIER`` is not 0. Only getattr on vllm_config.
    """
    from vllm_sm70.config import (
        dflash2_verifier_enabled,
        dflash2_verifier_qualified,
        sm70_config,
    )

    spec = getattr(vllm_config, "speculative_config", None)
    if int(getattr(spec, "num_speculative_tokens", 0) or 0) != 7:
        return False
    qualified = dflash2_verifier_qualified(
        getattr(vllm_config, "model_config", None),
        spec,
        getattr(vllm_config, "parallel_config", None),
    )
    return dflash2_verifier_enabled(
        sm70_config(vllm_config), verifier_qualified=qualified
    )


def fp8_rows(rows: int) -> bool:
    """M this route serves: 8, 16, 24, 32 (FORK sm70_dflash2_fp8.py:31-32)."""
    return 0 < rows <= MAX_ROWS and rows % ROWS_PER_REQUEST == 0


def ops_available() -> bool:
    return all(qpn8.has_op(name) for name in OPS)


def split_for(k: int) -> int | None:
    """The split-K serving every M in 8..32 for a channel-scaled [N, K], or None.

    K09's M=17..32 channel kernels require split 12 or 16 with two
    accumulator chains, the fast decoder and no prefetch; split 16 is
    preferred (fork: "The M=17..32 kernels take split 12 or 16").
    """
    if k <= 0 or k % 16:
        return None
    for split in _SPLITS:
        if (k // 16) % split == 0:
            return split
    return None


@torch.no_grad()
def channel_fp8_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] fp16 -> (E4M3 [N, K], FP32 scales [N, 1]).

    FORK sm70_dflash2_fp8.py:94-100: scale = amax/448 per output channel; an
    all-zero channel takes scale 1 (exact zero codes); scales are clamped to
    2**-24, the smallest the QPN8 channel-scale storage keeps.
    """
    dense = weight.float()
    scales = dense.abs().amax(1, keepdim=True) / E4M3_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scales = scales.clamp_min(MIN_SCALE)
    quantized = (dense / scales).to(torch.float8_e4m3fn)
    return quantized, scales


def _is_sm70(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_capability(device) == (7, 0)
    except Exception:
        return False


def admission(layer: torch.nn.Module) -> str | None:
    """Why a flagged layer cannot take the FP8 route, or None if it can."""
    if not getattr(layer, FP8_MARK_ATTR, False):
        return "not flagged"
    from vllm_sm70 import envs

    if not envs.VLLM_SM70_DFLASH2_VERIFIER:
        return "VLLM_SM70_DFLASH2_VERIFIER=0"
    if _batch_invariant():
        return "VLLM_BATCH_INVARIANT"
    if not ops_available():
        return "K09 QPN8 ops not built"
    weight = getattr(layer, "weight", None)
    if weight is None or weight.ndim != 2:
        return "no 2-D weight"
    if weight.dtype != torch.float16 or not weight.is_contiguous():
        return "weight is not contiguous fp16"
    if getattr(layer, "bias", None) is not None:
        return "has bias"
    n, k = weight.shape
    if n % 32:
        return f"N={n} is not a multiple of 32"
    if split_for(k) is None:
        return f"no split 12/16 divides K/16 for K={k}"
    return None


def _batch_invariant() -> bool:
    try:
        import vllm.envs as up_envs

        return bool(getattr(up_envs, "VLLM_BATCH_INVARIANT", False))
    except Exception:  # pragma: no cover - vllm not importable
        return False


@torch.no_grad()
def prepare_dflash2_fp8_m8(layer: torch.nn.Module) -> bool:
    """Quantize and pack a flagged layer after load. True if prepared.

    A repeated call (weight reload) re-quantizes from the current weight.
    """
    reason = admission(layer)
    if reason is not None:
        if getattr(layer, FP8_MARK_ATTR, False):
            logger.info(
                "vllm_sm70: DFlash2 FP8 M8 not admitted for %s: %s.",
                getattr(layer, "prefix", "?"),
                reason,
            )
        return False
    weight = layer.weight
    if not _is_sm70(weight.device):
        return False
    n, k = weight.shape
    quantized, scales = channel_fp8_quantize(weight.detach())
    codes, packed_scales = qpn8.fp8_qpn8_prepare_sm70(quantized, scales)
    layer.register_buffer(CODES_ATTR, codes, persistent=False)
    layer.register_buffer(SCALES_ATTR, packed_scales, persistent=False)
    setattr(layer, ROUTE_ATTR, (n, k, split_for(k)))
    logger.info(
        "vllm_sm70: DFlash2 channel-FP8 M8 prepared: N%d/K%d, split=%d.",
        n,
        k,
        split_for(k),
    )
    return True


# ------------------------------------------------------------------ the op
_OP = "vllm_sm70::dflash2_fp8_m8"
_SCHEMA = (
    "(Tensor x, Tensor weight, Tensor codes, Tensor scales, int split_k) -> Tensor"
)


def _dispatch_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    split_k: int,
) -> torch.Tensor:
    # FORK sm70_dflash2_fp8.py:35-52. The M decision stays inside the opaque
    # op, so a compiled graph traced for a dynamic M keeps the decode route.
    output = x.new_empty((x.shape[0], weight.shape[0]))
    if fp8_rows(x.shape[0]):
        qpn8.fp8_qpn8_gemm_sm70_out(
            output, x, codes, scales, split_k, ACCUMULATOR_CHAINS, True, False
        )
    else:
        torch.mm(x, weight.t(), out=output)
    return output


def _dispatch_fake(x, weight, codes, scales, split_k):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _register_op():
    ns, name = _OP.split("::")
    if hasattr(getattr(torch.ops, ns), name):  # module reloaded
        return getattr(getattr(torch.ops, ns), name)
    op = torch.library.custom_op(_OP, mutates_args=(), schema=_SCHEMA)(_dispatch_impl)
    op.register_fake(_dispatch_fake)
    return getattr(getattr(torch.ops, ns), name)


dflash2_fp8_m8 = _register_op()


def apply_dflash2_fp8_m8(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    """The route's output, or None when this call goes to the next route."""
    # FORK sm70_dflash2_fp8.py:111-133
    route = getattr(layer, ROUTE_ATTR, None)  # decided once, at prepare time
    if route is None:
        return None
    _n, k, split_k = route
    if (
        bias is not None
        or x.ndim != 2
        or x.shape[1] != k
        or x.dtype != torch.float16
        or not x.is_contiguous()
    ):
        return None
    if not torch.compiler.is_compiling() and not fp8_rows(x.shape[0]):
        return None
    codes = getattr(layer, CODES_ATTR)
    scales = getattr(layer, SCALES_ATTR)
    return dflash2_fp8_m8(x, layer.weight, codes, scales, split_k)
