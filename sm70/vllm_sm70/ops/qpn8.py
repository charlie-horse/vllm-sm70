# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN8 block-FP8 decode GEMV ops (K09): wrappers, fakes and `qpn8_linear`.

QPN8 is a weight-only FP8 layout for block-quantized checkpoints (E4M3
weights with 128 x 128 fp32 block scales, e.g. Qwen3.6-35B-A3B-FP8 and
Qwen3.8-27B-FP8). Each [N, K] weight is repacked once into uint8 codes
[K, N] (in QPN8 tile order) plus fp16 group scales [K/128, N/32] (scale *
256). The M<=32 GEMV decodes FP8 to half in registers and runs split-K
mma.m8n8k4 tiles; activations and output are fp16. M<=8 is one 8-row
tile, M=9..16 two, M=17..32 two 16-row passes (block scales; see
sm70/csrc/qpn8/PROVENANCE.md).

Ported from fork cf9492a:
- csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu ops, now sm70/csrc/qpn8/ (see
  PROVENANCE.md there); the fork's Python wrappers were vllm/_sm70/linear.py
  (fp8_qpn8_*), with a `native_policy` argument that is gone here.
- vllm/model_executor/kernels/linear/qpn/fp8_block.py:50-134, the
  `sm70_fp8::qpn8_native_linear` custom op, now
  `torch.ops.vllm_sm70.qpn8_native_linear` behind `qpn8_linear()`.
- vllm/model_executor/kernels/linear/qpn/fp8.py:55-63, 208-211, the
  split-K table `_SM70_FP8_QPN8_CONFIGS` and its generic fallback.

Contract for the FP8 linear method (Mara, M8). This module has no on/off
switch of its own: M8 decides whether to call it from its Sm70Config field
`fp8_block_qpn8` (FORK config/kernel.py:386 `kernel_config.sm70_fp8.block_qpn8`),
and keeps the TurboMind copy per `fp8_block_qpn8_turbomind_copy`
(FORK :388). VLLM_SM70_FP8_QPN8_M32 (below) is this module's only switch.

M cap: `qpn8_max_m(block_scaled=True)` is the M up to which the QPN8 GEMV
runs. It is 8 for block-scaled weights (the block-FP8 checkpoints) unless
VLLM_SM70_FP8_QPN8_M32=1 opts in to the K09 block-scaled M=9..32 routing
(not recommended: 27-33% slower than TurboMind at M=24/32 in the M8 A/B,
2026-11-08), and QPN8_MAX_M = 32 (the compiled kernel bound)
for channel-scaled weights (the fork's own M9..32 kernels, on by default).
Dispatch and needs_turbomind_fallback() use it; use it for any M gate too.

    prepared = qpn8_prepare(weight, weight_scale_inv,
                            turbomind_fallback=needs_turbomind_fallback(
                                max_num_seqs, num_speculative_tokens))
                            # True when rows > qpn8_max_m()
        weight: [N, K] float8_e4m3fn, CUDA, contiguous (checkpoint layout,
          after process_fp8_weight_block_strategy). weight_scale_inv:
          [N/128, K/128], cast to float32. N and K must be multiples of 128
          (`qpn8_rejection(n, k)` says why not). With
          turbomind_fallback=True the TurboMind FP8 operands (K03
          fp8_sm70_prepare) are kept too, for M > qpn8_max_m(); otherwise those run a
          dequantize into a [K, N] fp16 scratch and torch.mm. The caller
          frees the checkpoint weight afterwards.
    y = qpn8_linear(x, prepared, bias=None, dense_scratch=None)
        x: [..., K] fp16 (flattened to [M, K]). Returns [..., N] fp16.
          M == 0: empty output. 1 <= M <= qpn8_max_m(): the QPN8 GEMV (8 by
          default for block scales, 32 with VLLM_SM70_FP8_QPN8_M32=1). Above it: TurboMind
          FP8 if prepared has the fallback operands, else dequantize +
          mm, into `dense_scratch` (fp16, >= K*N elements, mutated) when
          given, else a per-call allocation. The M decision is inside the
          opaque custom op, so a graph traced at small M stays correct for
          prefill (fork fp8_block.py:71-73).

`prepared` is a Qpn8Prepared; store its tensors as layer buffers
(register_buffer(..., persistent=False)) so they follow the module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake  # type: ignore

NAMESPACE = "_sm70_C"
BLOCK = 128
# Compiled kernel bound (fp8_qpn8_gemm_sm70_out admits M in [1, 32]). The
# effective cap is qpn8_max_m(): fork fp8_block.py:45 used 8 for block
# scales, which stays the default; VLLM_SM70_FP8_QPN8_M32=1 raises it to 32
# (DFlash2 k=7 verify for up to 4 requests, 8 rows each).
QPN8_MAX_M = 32
QPN8_BLOCK_DEFAULT_MAX_M = 8
M32_SWITCH = "VLLM_SM70_FP8_QPN8_M32"


def qpn8_max_m(block_scaled: bool = True) -> int:
    """Largest M the QPN8 GEMV serves: QPN8_MAX_M for channel scales; for
    block scales QPN8_MAX_M only with VLLM_SM70_FP8_QPN8_M32=1, else 8."""
    if not block_scaled:
        return QPN8_MAX_M
    from vllm_sm70 import envs

    return QPN8_MAX_M if getattr(envs, M32_SWITCH) else QPN8_BLOCK_DEFAULT_MAX_M


# Load _sm70_C before any has_op() check below (see ops/tm.py).
from vllm_sm70 import _ext  # noqa: E402

_ext.load()


def has_op(name: str) -> bool:
    """True if torch.ops._sm70_C.<name> is registered."""
    try:
        return hasattr(getattr(torch.ops, NAMESPACE), name)
    except (AttributeError, RuntimeError):
        return False


def _op(name: str):
    if not has_op(name):
        raise RuntimeError(
            f"SM70 QPN8 op {NAMESPACE}::{name} is not available. "
            "Build vllm_sm70._sm70_C with CUDA 12.x for sm_70."
        )
    return getattr(getattr(torch.ops, NAMESPACE), name)


# ---------------------------------------------------------------------------
# Op wrappers and fakes
# ---------------------------------------------------------------------------


# fork vllm/_sm70/linear.py fp8_qpn8_prepare_sm70 (minus native_policy)
def fp8_qpn8_prepare_sm70(
    qweight: torch.Tensor, scales: torch.Tensor
) -> list[torch.Tensor]:
    """[N, K] e4m3 + [N/128, K/128] (or [N, 1]) fp32 -> [codes, group_scales]."""
    return _op("fp8_qpn8_prepare_sm70")(qweight, scales)


if has_op("fp8_qpn8_prepare_sm70"):

    @register_fake("_sm70_C::fp8_qpn8_prepare_sm70")
    def _fp8_qpn8_prepare_sm70_fake(
        qweight: torch.Tensor, scales: torch.Tensor
    ) -> list[torch.Tensor]:
        n, k = qweight.shape
        channel = scales.shape[0] == n and scales.shape[1] == 1
        codes = qweight.new_empty((k, n), dtype=torch.uint8)
        group_scales = qweight.new_empty(
            (1, n) if channel else (k // BLOCK, n // 32), dtype=torch.float16
        )
        return [codes, group_scales]


def fp8_qpn8_dequantize_sm70_out(
    out: torch.Tensor, codes: torch.Tensor, group_scales: torch.Tensor
) -> None:
    """Write the fp16 [K, N] weight (out) from QPN8 codes and scales."""
    _op("fp8_qpn8_dequantize_sm70_out")(out, codes, group_scales)


if has_op("fp8_qpn8_dequantize_sm70_out"):

    @register_fake("_sm70_C::fp8_qpn8_dequantize_sm70_out")
    def _fp8_qpn8_dequantize_sm70_out_fake(
        out: torch.Tensor, codes: torch.Tensor, group_scales: torch.Tensor
    ) -> None:
        return None


def fp8_qpn8_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    fast_decoder: bool = True,
    prefetch_codes: bool = False,
) -> None:
    """out[M, N] = input[M, K] @ W, 1 <= M <= 32, fp16 (split_k <= 16 for M > 8)."""
    _op("fp8_qpn8_gemm_sm70_out")(
        out,
        input,
        codes,
        group_scales,
        split_k,
        accumulator_chains,
        fast_decoder,
        prefetch_codes,
    )


if has_op("fp8_qpn8_gemm_sm70_out"):

    @register_fake("_sm70_C::fp8_qpn8_gemm_sm70_out")
    def _fp8_qpn8_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        codes: torch.Tensor,
        group_scales: torch.Tensor,
        split_k: int,
        accumulator_chains: int,
        fast_decoder: bool,
        prefetch_codes: bool,
    ) -> None:
        return None


# Every QPN8 op takes a Tensor, so all have fakes (rule in ops/tm.py).
NO_FAKE_OPS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Split-K configuration (fork qpn/fp8.py:55-63, 208-211)
# ---------------------------------------------------------------------------

# (K, N, fused gated-SiLU): (split-K, accumulator chains, prefetch codes).
# Measured on Qwen3.8-27B-FP8 at TP4 (fork docs/design/
# sm70_qwen38_qpn8_decode.md). No TP2 Qwen3.6/3.8 shape is in it, so TP2
# uses the generic rule below. Gated entries are kept for fidelity; this
# port has no fused gated-SiLU op.
QPN8_CONFIGS: dict[tuple[int, int, bool], tuple[int, int, bool]] = {
    (4352, 5120, False): (16, 1, False),
    (1536, 5120, False): (16, 1, False),
    (5120, 4096, False): (16, 2, False),
    (5120, 3584, False): (16, 2, False),
    (5120, 8704, False): (16, 1, True),
    (5120, 8704, True): (8, 2, True),
}
_SPLITS = (4, 8, 12, 16, 32)


def qpn8_config(k: int, n: int, gated: bool = False) -> tuple[int, int, bool]:
    """(split_k, accumulator_chains, prefetch_codes) for a [K, N] weight.

    Fork rule: the table, else split 8 if gated or K % 256 else 16, two
    chains, no prefetch. K % 128 == 0 (block FP8) makes the generic split
    always divide K/16; a table entry that would not is replaced by the
    generic rule, and that by the largest of 16/8/4 that divides K/16.
    """
    cfg = QPN8_CONFIGS.get((k, n, gated))
    if cfg is not None and _valid_config(k, *cfg):
        return cfg
    split = 8 if gated or k % 256 else 16
    if _valid_config(k, split, 2, False):
        return (split, 2, False)
    for split in (16, 8, 4):
        if (k // 16) % split == 0:
            return (split, 2, False)
    raise ValueError(f"QPN8: no split-K divides K/16 for K={k}")


def _valid_config(k: int, split_k: int, chains: int, prefetch: bool) -> bool:
    # Mirrors the TORCH_CHECKs of fp8_qpn8_gemm_sm70_out.
    # split 32 is valid only up to M=8; configs here must serve M<=32.
    if split_k not in _SPLITS or split_k > 16 or chains not in (1, 2) or k % 16:
        return False
    if (k // 16) % split_k:
        return False
    return not (split_k == 12 and (chains != 2 or prefetch))


def qpn8_rejection(n: int, k: int) -> str | None:
    """Why a block-FP8 [N, K] weight cannot use QPN8, or None (fork
    fp8_block.py can_implement: positive N, K divisible by 128)."""
    if n <= 0 or k <= 0 or n % BLOCK or k % BLOCK:
        return f"requires positive N,K divisible by {BLOCK}, got ({n},{k})"
    return None


def needs_turbomind_fallback(
    max_num_seqs: int, num_speculative_tokens: int = 0, block_scaled: bool = True
) -> bool:
    """Keep TurboMind FP8 operands when decode/verify rows can exceed
    qpn8_max_m() (fork fp8_block.py:204-213 _keep_volta_turbomind_copy)."""
    return max_num_seqs * (1 + num_speculative_tokens) > qpn8_max_m(block_scaled)


# ---------------------------------------------------------------------------
# Prepare + linear
# ---------------------------------------------------------------------------


@dataclass
class Qpn8Prepared:
    """Operands of one QPN8 linear. Tensors live on the weight's device."""

    codes: torch.Tensor  # [K, N] uint8
    group_scales: torch.Tensor  # [K/128, N/32] fp16
    n: int
    k: int
    split_k: int
    accumulator_chains: int
    prefetch_codes: bool
    # TurboMind FP8 operands for M > 32 (K03 fp8_sm70_prepare), or None.
    tm_weight: torch.Tensor | None = None
    tm_scales: torch.Tensor | None = None
    tm_k_ld: int = 0
    tm_q_ld: int = 0


def qpn8_prepare(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    turbomind_fallback: bool = False,
) -> Qpn8Prepared:
    """Repack a block-FP8 checkpoint weight (fork fp8_block.py:215-277)."""
    n, k = weight.shape
    reason = qpn8_rejection(n, k)
    if reason is not None:
        raise ValueError(f"QPN8: {reason}")
    if weight.dtype != torch.float8_e4m3fn:
        weight = weight.view(torch.float8_e4m3fn)
    scales = weight_scale_inv.detach().float().contiguous()
    if tuple(scales.shape) != (n // BLOCK, k // BLOCK):
        raise ValueError(
            f"QPN8: scale raster {tuple(scales.shape)} does not match "
            f"weights {n}x{k} at block [{BLOCK},{BLOCK}]"
        )
    weight = weight.contiguous()
    codes, group_scales = fp8_qpn8_prepare_sm70(weight, scales)
    split_k, chains, prefetch = qpn8_config(k, n, False)
    prepared = Qpn8Prepared(codes, group_scales, n, k, split_k, chains, prefetch)
    if turbomind_fallback:
        from vllm_sm70.ops import tm

        tm_weight, tm_scales, meta = tm.fp8_sm70_prepare(weight, scales, BLOCK, False)
        prepared.tm_weight = tm_weight
        prepared.tm_scales = tm_scales
        prepared.tm_k_ld, prepared.tm_q_ld = (int(v) for v in meta.tolist())
    return prepared


_LINEAR_OP = "vllm_sm70::qpn8_native_linear"
_LINEAR_SCHEMA = (
    "(Tensor x, Tensor codes, Tensor group_scales, int split_k, "
    "int accumulator_chains, bool prefetch_codes, "
    "Tensor? fallback_weight=None, Tensor? fallback_scales=None, "
    "int fallback_k_ld=0, int fallback_q_ld=0, "
    "Tensor(a!)? dense_scratch=None) -> Tensor"
)


def _qpn8_native_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    fallback_weight: torch.Tensor | None = None,
    fallback_scales: torch.Tensor | None = None,
    fallback_k_ld: int = 0,
    fallback_q_ld: int = 0,
    dense_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    # fork fp8_block.py:58-118. The M decision stays inside the opaque op: a
    # compiled graph covers a dynamic M range, so a Python branch traced at
    # small M would be reused for prefill.
    k_dim, n_dim = codes.shape
    out = x.new_empty((x.shape[0], n_dim))
    m = x.shape[0]
    if m == 0:
        return out
    # Channel scales are [1, N]; block scales [K/128, N/32].
    block_scaled = not (group_scales.shape[0] == 1 and group_scales.shape[1] == n_dim)
    if m <= qpn8_max_m(block_scaled):
        fp8_qpn8_gemm_sm70_out(
            out,
            x,
            codes,
            group_scales,
            split_k,
            accumulator_chains,
            True,
            prefetch_codes,
        )
        return out
    if fallback_weight is not None:
        from vllm_sm70.ops import tm

        tm.fp8_gemm_sm70_out(
            out,
            x,
            fallback_weight,
            fallback_scales,
            BLOCK,
            fallback_k_ld,
            fallback_q_ld,
            False,
        )
        return out
    # Replaces the fork's fp8_qpn8_prefill_sm70_out(int dense_weight_ptr):
    # dequantize into a caller tensor (or a per-call one), then mm.
    numel = k_dim * n_dim
    if (
        dense_scratch is not None
        and dense_scratch.dtype == torch.float16
        and dense_scratch.is_contiguous()
        and dense_scratch.numel() >= numel
    ):
        dense = dense_scratch.view(-1)[:numel].view(k_dim, n_dim)
    else:
        dense = torch.empty((k_dim, n_dim), dtype=torch.float16, device=x.device)
    fp8_qpn8_dequantize_sm70_out(dense, codes, group_scales)
    torch.mm(x, dense, out=out)
    return out


def _qpn8_native_linear_fake(
    x,
    codes,
    group_scales,
    split_k,
    accumulator_chains,
    prefetch_codes,
    fallback_weight=None,
    fallback_scales=None,
    fallback_k_ld=0,
    fallback_q_ld=0,
    dense_scratch=None,
):
    return x.new_empty((x.shape[0], codes.shape[1]))


def _register_linear_op():
    ns, name = _LINEAR_OP.split("::")
    if hasattr(getattr(torch.ops, ns), name):  # module reloaded
        return getattr(getattr(torch.ops, ns), name)
    op = torch.library.custom_op(
        _LINEAR_OP, mutates_args=("dense_scratch",), schema=_LINEAR_SCHEMA
    )(_qpn8_native_linear_impl)
    op.register_fake(_qpn8_native_linear_fake)
    return getattr(getattr(torch.ops, ns), name)


qpn8_native_linear = _register_linear_op()


def qpn8_linear(
    x: torch.Tensor,
    prepared: Qpn8Prepared,
    bias: torch.Tensor | None = None,
    dense_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """y[..., N] = x[..., K] @ W^T (+ bias), fp16; see the module docstring."""
    if x.dtype != torch.float16:
        raise TypeError(f"QPN8 linear needs FP16 activations, got {x.dtype}.")
    x_2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    y = qpn8_native_linear(
        x_2d,
        prepared.codes,
        prepared.group_scales,
        prepared.split_k,
        prepared.accumulator_chains,
        prepared.prefetch_codes,
        prepared.tm_weight,
        prepared.tm_scales,
        prepared.tm_k_ld,
        prepared.tm_q_ld,
        dense_scratch,
    )
    if bias is not None:
        y = y + bias
    return y if x.ndim == 2 else y.reshape(*x.shape[:-1], prepared.n)
