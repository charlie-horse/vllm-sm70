# SPDX-License-Identifier: Apache-2.0
"""Reference dequantizers and random weights for the K03 GPU tests."""

from __future__ import annotations

import torch

# e2m1 code -> value; bit 3 is the sign (as fork awq_sm70_gemm.cu:565-568).
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_table(device) -> torch.Tensor:
    pos = torch.tensor(E2M1, dtype=torch.float32, device=device)
    return torch.cat([pos, -pos])


def nvfp4_weight(k: int, n: int, device="cuda", seed: int = 0):
    """codes uint8 [K, N] (one e2m1 code per byte), scales fp16 [K/16, N]."""
    g = torch.Generator(device=device).manual_seed(seed)
    codes = torch.randint(16, (k, n), device=device, dtype=torch.uint8, generator=g)
    scales = (torch.rand(k // 16, n, device=device, generator=g) * 0.02 + 0.002).half()
    return codes, scales


def nvfp4_dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 [K, N]."""
    values = e2m1_table(codes.device)[codes.long()]
    return values * scales.float().repeat_interleave(16, dim=0)


def fp8_block_weight(n: int, k: int, device="cuda", seed: int = 0):
    """qweight e4m3 [N, K], block scales fp32 [N/128, K/128] (Qwen FP8 layout)."""
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(n, k, device=device, generator=g).clamp_(-8, 8)
    q = w.to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device=device, generator=g) * 0.01 + 0.001
    return q, scales


def fp8_block_dequant(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 [N, K]. fp8_sm70_prepare stores the scales as fp16; do the same."""
    s = scales.half().float()
    s = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return q.float() * s[: q.shape[0], : q.shape[1]]


def max_abs_rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    """max |out - ref| / max |ref|."""
    return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
