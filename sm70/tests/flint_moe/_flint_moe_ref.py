# SPDX-License-Identifier: Apache-2.0
"""Flint TP2 expert banks and references for the K12 GPU tests.

Ported from fork cf9492a tests/kernels/moe/test_sm70_nvfp4_contract_flint.py
(`_dequantize` :1038, `_flint_expert_weights` :1048, `_reference_moe` :1073,
`_flint_routing` :1157, `_relative_error` :1173, and `_dirty_graph_pool` from
tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py:738). The banks
are prepared here with tm.nvfp4_sm70_prepare exactly as
vllm_sm70/quant/nvfp4_moe.py (M4) does at load, instead of through the
fork's ModelOptNvFp4SM70MoEMethod.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

HIDDEN, INTER, EXPERTS, TOP_K = 2048, 256, 256, 6  # Flint TP2 local
GROUP = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize(codes, block_scales, row_scales):
    """ModelOpt NVFP4 [N, K / 2] (low nibble first) -> FP32 [N, K]."""
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=codes.device)
    values = torch.stack(
        (lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1
    ).flatten(1)
    scales = block_scales.float().repeat_interleave(GROUP, dim=1)
    return values * scales * row_scales


def flint_expert_weights(device, seed: int, w2_global: float = 0.04):
    """Random ModelOpt NVFP4 Flint TP2 experts (fork :1048-1070)."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def block_scales(rows, groups):
        values = (
            torch.rand(EXPERTS, rows, groups, generator=gen, device=device) * 1.5 + 0.5
        )
        return values.to(torch.float8_e4m3fn)

    w13_global = torch.full((EXPERTS, 2), 0.0075, device=device)
    w13_global[:, 1] *= 1.25  # distinct gate/up globals catch a swapped split
    return {
        "w13_weight": torch.randint(
            0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), generator=gen, device=device
        ).to(torch.uint8),
        "w13_weight_scale": block_scales(2 * INTER, HIDDEN // GROUP),
        "w13_weight_scale_2": w13_global,
        "w2_weight": torch.randint(
            0, 256, (EXPERTS, HIDDEN, INTER // 2), generator=gen, device=device
        ).to(torch.uint8),
        "w2_weight_scale": block_scales(HIDDEN, INTER // GROUP),
        "w2_weight_scale_2": torch.full((EXPERTS,), w2_global, device=device),
    }


def _codes_kn(packed):
    """[N, K/2] u8, low nibble = even k -> [K, N] u8 E2M1 codes (as
    vllm_sm70.quant.nvfp4_linear.unpack_fp4_weight)."""
    n = packed.shape[0]
    return torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, -1).t()


def _scales_kn(block, glob, split):
    """As vllm_sm70.quant.nvfp4_moe.expert_scales: [N, K/16] x global ->
    [K/16, N] FP16 (W13: gate rows use glob[0], up rows glob[-1])."""
    s = block.float().clone()
    g = glob.float().reshape(-1)
    if split is None:
        s.mul_(g[0])
    else:
        s[:split].mul_(g[0])
        s[split:].mul_(g[-1])
    return s.half().t().contiguous()


@dataclass
class Banks:
    w13_weight: torch.Tensor
    w13_scales: torch.Tensor
    w2_weight: torch.Tensor
    w2_scales: torch.Tensor
    w13_meta: tuple[int, int]
    w2_meta: tuple[int, int]


def prepare_banks(tm, raw) -> Banks:
    """Stacked TurboMind banks, as M4 process_weights_after_loading builds them."""
    out = {}
    for stage, split in (("w13", INTER), ("w2", None)):
        ws, ss, meta = [], [], None
        for e in range(EXPERTS):
            codes = _codes_kn(raw[f"{stage}_weight"][e]).contiguous()
            scales = _scales_kn(
                raw[f"{stage}_weight_scale"][e],
                raw[f"{stage}_weight_scale_2"][e],
                split,
            )
            w, s, m = tm.nvfp4_sm70_prepare(codes, scales, GROUP, False)
            ws.append(w)
            ss.append(s)
            meta = meta or tuple(int(v) for v in m.tolist())
        out[stage] = (torch.stack(ws), torch.stack(ss), meta)
    return Banks(
        out["w13"][0],
        out["w13"][1],
        out["w2"][0],
        out["w2"][1],
        out["w13"][2],
        out["w2"][2],
    )


def routing(tokens: int, gen: torch.Generator, device):
    """Flint sigmoid top-6 with renormalize (fork :1157-1162)."""
    x = torch.randn(tokens, HIDDEN, generator=gen).half().to(device)
    logits = torch.randn(tokens, EXPERTS, generator=gen).to(device)
    scores, ids = torch.sigmoid(logits).topk(TOP_K, dim=-1)
    weights = (scores / scores.sum(dim=-1, keepdim=True)).contiguous()
    return x, weights, ids.to(torch.int32).contiguous()


def reference_moe(x, weights, ids, raw):
    """FP32 oracle on the dequantized checkpoint (fork :1073-1091)."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in ids.unique().tolist():
        if not 0 <= expert < EXPERTS:
            continue
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        g = raw["w13_weight_scale_2"][expert]
        row_scales = g.repeat_interleave(INTER)[:, None]
        w13 = dequantize(
            raw["w13_weight"][expert], raw["w13_weight_scale"][expert], row_scales
        )
        gate_up = x[rows].float() @ w13.t()
        act = F.silu(gate_up[:, :INTER]) * gate_up[:, INTER:]
        w2 = dequantize(
            raw["w2_weight"][expert],
            raw["w2_weight_scale"][expert],
            raw["w2_weight_scale_2"][expert],
        )
        out.index_add_(0, rows, (act @ w2.t()) * weights[rows, slots, None])
    return out


def turbomind_moe(tm, banks: Banks, x, weights, ids):
    """Per-slot TurboMind dense GEMMs over the same banks, with the decode's
    FP16 boundaries (gate/up, SiLU*up, each routed output). Replaces the
    fork's TurboMind-route comparison (:1215-1220): same rounding points,
    another K order, and an independent reader of the bank layout."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    gate_up = torch.empty(1, 2 * INTER, dtype=torch.float16, device=x.device)
    routed = torch.empty(1, HIDDEN, dtype=torch.float16, device=x.device)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(ids[t, k])
            if not 0 <= e < EXPERTS:
                continue
            tm.nvfp4_gemm_sm70_out(
                gate_up,
                x[t : t + 1],
                banks.w13_weight[e],
                banks.w13_scales[e],
                GROUP,
                *banks.w13_meta,
            )
            gate = gate_up[:, :INTER].float()
            silu = (gate / (1.0 + torch.exp(-gate))).half()
            act = (silu * gate_up[:, INTER:]).contiguous()
            tm.nvfp4_gemm_sm70_out(
                routed,
                act,
                banks.w2_weight[e],
                banks.w2_scales[e],
                GROUP,
                *banks.w2_meta,
            )
            out[t] += routed[0].float() * float(weights[t, k])
    return out


def relative_error(actual, expected) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def dirty_graph_pool(device, pool, stream):
    """Leave freed memory of a shared graph pool full of 0x7fffffff.

    Tensors allocated in a throwaway capture on the pool are filled by one
    replay and then freed, so later captures on the pool reuse poisoned
    memory instead of whatever cudaMalloc returned (fork sinks test :738).
    """
    numel = 64 << 20  # 256 MiB of int32
    poison = torch.cuda.CUDAGraph()
    with torch.cuda.graph(poison, pool=pool, stream=stream):
        blocks = [torch.empty(numel, dtype=torch.int32, device=device)]
        blocks += [
            torch.empty(1 << 17, dtype=torch.int32, device=device) for _ in range(64)
        ]
        for block in blocks:
            block.fill_(0x7FFFFFFF)
    poison.replay()
    torch.cuda.synchronize(device)
    del blocks, block
    return poison
