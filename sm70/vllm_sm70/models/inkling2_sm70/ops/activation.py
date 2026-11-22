# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 soft-capped SwiGLU of the expert gate/up buffer as one Triton
kernel (sm_70: libdevice tanh, no tanh.approx).

    softcapped_swiglu_out   act = gate_cap * tanh(g / gate_cap) * sigmoid(g)
                                  * up_cap * tanh(u / up_cap)
                            torch form: torch_ref.softcapped_swiglu

`gate_up` [R, 2 * I] holds a row's gate units then its up units, in the
model dtype. The math runs in fp32 and rounds once to the dtype of `out`
[R, I], which the caller provides: the op allocates nothing. |act| stays
below gate_cap * up_cap.
"""

from __future__ import annotations

import torch

from . import torch_ref
from ._compat import exp, tanh, tl, triton

__all__ = ["softcapped_swiglu_out"]

BLOCK_MAX = 1024


@triton.jit
def _softcapped_swiglu_kernel(
    gate_up_ptr,
    out_ptr,
    stride_g,
    stride_o,
    units,
    gate_cap,
    up_cap,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < units
    src = gate_up_ptr + row * stride_g + cols
    g = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(src + units, mask=mask, other=0.0).to(tl.float32)
    gate = tanh(tl.div_rn(g, gate_cap)) * gate_cap
    sig = tl.div_rn(1.0, 1.0 + exp(-g))
    up = tanh(tl.div_rn(u, up_cap)) * up_cap
    act = gate * sig * up
    tl.store(
        out_ptr + row * stride_o + cols, act.to(out_ptr.dtype.element_ty), mask=mask
    )


def softcapped_swiglu_out(
    gate_up: torch.Tensor,
    out: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    *,
    kernel: bool = True,
) -> torch.Tensor:
    """Write the soft-capped SwiGLU of `gate_up` [R, 2 * I] into `out` [R, I].

    kernel=False runs the torch form (torch_ref.softcapped_swiglu) instead.
    """
    rows, units = out.shape
    assert gate_up.shape == (rows, 2 * units), (gate_up.shape, out.shape)
    if not kernel:
        gate, up = gate_up.chunk(2, dim=-1)
        return out.copy_(torch_ref.softcapped_swiglu(gate, up, gate_cap, up_cap))
    if rows == 0:
        return out
    assert gate_up.stride(-1) == 1 and out.stride(-1) == 1
    block = min(BLOCK_MAX, triton.next_power_of_2(units))
    grid = (rows, triton.cdiv(units, block))
    _softcapped_swiglu_kernel[grid](
        gate_up, out, gate_up.stride(0), out.stride(0), units,
        float(gate_cap), float(up_cap),
        BLOCK=block, num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out
