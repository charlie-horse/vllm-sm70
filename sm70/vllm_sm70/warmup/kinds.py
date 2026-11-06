# SPDX-License-Identifier: Apache-2.0
"""Run one plan Item at one M: the same TurboMind call the layer makes in forward.

Dense kinds call the op with the operands the layer's apply passes; MoE
kinds call the layer's own ``quant_method.apply`` with synthetic routing,
so the route taken at each M (batched, compact M == 1, decomposed) and
therefore the dispatch keys are exactly the runtime's.

ported from FORK cf9492a vllm/model_executor/warmup/awq_sm70_warmup.py:
- nvfp4_dense: ``_warmup_fp4_dense_layers`` :721-793, nvfp4 non-prescaled,
  non-scale-code branch; through M3's ``apply_prepared_linear`` so the K11
  GEMV/TurboMind operator choice and output padding match forward.
- fp8_dense / fp8_qpn8_tm: ``_warmup_fp8_dense_layers`` :620-718, the plain
  TurboMind branch (:666-677, :680-696) and the block-QPN8 TurboMind-copy
  branch (:635-640, :683-684).
- nvfp4_moe / fp8_moe: replaces ``_warmup_nvfp4_moe_decode_layers``
  :952-1051 and ``_warmup_moe_dense_stage_layers`` :796-878, which built the
  stage operands by hand; ``_build_balanced_offsets`` :349-363 becomes the
  routing in ``moe_routing`` (balanced expert use, distinct experts per token).
"""

from __future__ import annotations

from typing import Any

from vllm_sm70.warmup import plan as P

FP8_BLOCK = P.FP8_BLOCK


def moe_routing(num_tokens: int, top_k: int, num_experts: int, device: Any):
    """topk ids/weights: slot j of token t -> expert (t * top_k + j) % E.

    Every expert is used about equally (the fork's balanced offsets) and a
    token never picks one expert twice (top_k <= E).
    """
    import torch

    slots = torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
    ids = torch.remainder(slots, num_experts).view(num_tokens, top_k).contiguous()
    weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return weights, ids


def _device_of(item: P.Item) -> Any:
    layer = item.layer
    if item.kind == P.NVFP4_DENSE:
        return layer._sm70_turbomind_linear.weight.device
    if item.kind == P.FP8_DENSE:
        return layer.weight.device
    if item.kind == P.FP8_QPN8_TM:
        return layer.sm70_qpn8_tm_weight.device
    return layer.w13_tm_weight.device


def run(item: P.Item, m: int) -> None:
    """One forward-equivalent TurboMind call of ``item`` at M (MoE: tokens) m."""
    import torch

    layer = item.layer
    device = _device_of(item)
    if item.kind == P.NVFP4_DENSE:
        from vllm_sm70.quant.nvfp4_linear import apply_prepared_linear

        k = item.key[0]
        x = torch.zeros((m, k), dtype=torch.float16, device=device)
        apply_prepared_linear(layer, x, None)
        return
    if item.kind in (P.FP8_DENSE, P.FP8_QPN8_TM):
        from vllm_sm70.ops import tm

        k, n = item.key[0], item.key[1]
        if item.kind == P.FP8_DENSE:
            weight, scales = layer.weight, layer.weight_scale_inv
            k_ld, q_ld = int(layer.sm70_fp8_k_ld), int(layer.sm70_fp8_q_ld)
        else:
            weight, scales = layer.sm70_qpn8_tm_weight, layer.sm70_qpn8_tm_scales
            k_ld, q_ld = (int(v) for v in layer.sm70_qpn8_meta[5:7])
        x = torch.zeros((m, k), dtype=torch.float16, device=device)
        out = torch.empty((m, n), dtype=torch.float16, device=device)
        tm.fp8_gemm_sm70_out(out, x, weight, scales, FP8_BLOCK, k_ld, q_ld, False)
        return
    if item.kind in P.MOE_KINDS:
        experts, hidden, _inter, top_k = item.key[:4]
        x = torch.zeros((m, hidden), dtype=torch.float16, device=device)
        weights, ids = moe_routing(m, top_k, experts, device)
        layer.quant_method.apply(layer, x, weights, ids, None, None)
        return
    raise ValueError(f"unknown warmup kind {item.kind!r}")
