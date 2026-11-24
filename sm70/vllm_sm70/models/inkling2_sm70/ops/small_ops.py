# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 decode kernels for the small torch chains of a layer visit
(sm_70: no tl.dot, libdevice exp and log, no tanh.approx).

    query_rel   the q and r clips of the attention prologue, tau of a global
                layer, r @ rel_proj and zeroed padding rows, in one launch
                (prologue.query_states and attention_prologue)
    gate        attn_output * sigmoid(gate) with padding rows zeroed, in one
                launch (attention_torch.gate_and_project)

Math and rounding as the torch ops, the reference: products, norms and sums
in fp32; a clip rounds to its input dtype before tau multiplies it, and the
clipped r rounds before the projection. A row is padding when it is at or
after `num_tokens` or its slot is negative; its outputs are zero.
"""

from __future__ import annotations

import torch

from ._compat import INTERPRET, exp, tl, triton

if INTERPRET:

    @triton.jit
    def _log(x):
        return tl.log(x)

else:
    from vllm.triton_utils import tldevice

    @triton.jit
    def _log(x):
        return tldevice.log(x)


__all__ = ["gate", "query_rel"]

BLOCK_E = 256
BLOCK_GATE = 2048


@triton.jit
def _clip(src, scale, cols, mask, clip):
    """fp32 clip of one head: `src` times `scale`, shrunk to length `clip`."""
    y = tl.load(src + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * tl.load(scale + cols, mask=mask, other=0.0).to(tl.float32)
    norm = tl.sqrt_rn(tl.sum(y * y, axis=0))
    # clip * (1 / (norm + eps)), as torch evaluates the scalar division.
    return y * tl.minimum(tl.div_rn(1.0, norm + 1e-6) * clip, 1.0)


@triton.jit
def _query_rel_head(
    row,
    head,
    q_ptr,
    r_ptr,
    q_scale_ptr,
    r_scale_ptr,
    proj_ptr,
    slot_ptr,
    pos_ptr,
    q_out_ptr,
    rel_out_ptr,
    num_tokens,
    stride_q,
    stride_r,
    stride_qo0,
    stride_qo1,
    stride_ro0,
    stride_ro1,
    stride_p,
    extent,
    q_clip,
    r_clip,
    alpha,
    n_floor,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    LOG_SCALE: tl.constexpr,
):
    """q_out and rel_out of one (row, query head)."""
    real = row < num_tokens
    slot = tl.load(slot_ptr + row, mask=real, other=-1)
    keep = real & (slot >= 0)
    row64 = row.to(tl.int64)

    q_cols = tl.arange(0, BLOCK_D)
    q_mask = q_cols < D
    q = _clip(q_ptr + row64 * stride_q + head * D, q_scale_ptr + head * D,
              q_cols, q_mask, q_clip)  # fmt: skip
    q = q.to(q_out_ptr.dtype.element_ty)
    r_cols = tl.arange(0, BLOCK_R)
    r_mask = r_cols < DR
    r = _clip(r_ptr + row64 * stride_r + head * DR, r_scale_ptr + head * DR,
              r_cols, r_mask, r_clip)  # fmt: skip
    r = r.to(q_out_ptr.dtype.element_ty)
    if LOG_SCALE:
        pos = tl.load(pos_ptr + row, mask=real, other=0).to(tl.float32)
        tau = 1.0 + alpha * _log(tl.maximum(tl.div_rn(pos + 1.0, n_floor), 1.0))
        q = (q.to(tl.float32) * tau).to(q_out_ptr.dtype.element_ty)
        r = (r.to(tl.float32) * tau).to(q_out_ptr.dtype.element_ty)
    q = tl.where(keep, q.to(tl.float32), 0.0)
    tl.store(
        q_out_ptr + row64 * stride_qo0 + head * stride_qo1 + q_cols,
        q.to(q_out_ptr.dtype.element_ty),
        mask=q_mask,
    )

    r = tl.where(keep, r.to(tl.float32), 0.0)
    out = rel_out_ptr + row64 * stride_ro0 + head * stride_ro1
    for e0 in tl.range(0, extent, BLOCK_E):
        e = e0 + tl.arange(0, BLOCK_E)
        e_mask = e < extent
        p = tl.load(
            proj_ptr + r_cols[:, None] * stride_p + e[None, :],
            mask=r_mask[:, None] & e_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rel = tl.sum(r[:, None] * p, axis=0)
        tl.store(out + e, rel.to(rel_out_ptr.dtype.element_ty), mask=e_mask)


@triton.jit(do_not_specialize=["num_tokens"])
def _query_rel_kernel(
    q_ptr,
    r_ptr,
    q_scale_ptr,
    r_scale_ptr,
    proj_ptr,
    slot_ptr,
    pos_ptr,
    q_out_ptr,
    rel_out_ptr,
    num_tokens,
    stride_q,
    stride_r,
    stride_qo0,
    stride_qo1,
    stride_ro0,
    stride_ro1,
    stride_p,
    extent,
    q_clip,
    r_clip,
    alpha,
    n_floor,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    LOG_SCALE: tl.constexpr,
):
    _query_rel_head(
        tl.program_id(0), tl.program_id(1),
        q_ptr, r_ptr, q_scale_ptr, r_scale_ptr, proj_ptr, slot_ptr, pos_ptr,
        q_out_ptr, rel_out_ptr, num_tokens, stride_q, stride_r, stride_qo0,
        stride_qo1, stride_ro0, stride_ro1, stride_p, extent, q_clip, r_clip,
        alpha, n_floor,
        D=D, DR=DR, BLOCK_D=BLOCK_D, BLOCK_R=BLOCK_R, BLOCK_E=BLOCK_E,
        LOG_SCALE=LOG_SCALE,
    )  # fmt: skip


def query_rel(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    rel_proj: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor,
    rel_out: torch.Tensor,
    num_tokens: int,
    *,
    q_clip: float,
    r_clip: float,
    log_scaling: tuple[float, float] | None = None,
) -> None:
    """q [T, heads * D] and r [T, heads * d_rel] (rows with unit column
    stride) -> `q_out` [T, heads, D] and `rel_out` [T, heads, extent]:
    clip(q * q_scale), clip(r * r_scale) @ rel_proj [d_rel, extent], each
    clip times tau of `positions` when `log_scaling` (alpha, n_floor) is
    given. Padding rows (module docstring) are zero."""
    total = q_out.shape[0]
    heads, head_dim = q_scale.shape
    d_rel = r_scale.shape[1]
    extent = rel_proj.shape[1]
    assert q.stride(-1) == r.stride(-1) == rel_proj.stride(-1) == 1
    assert q_out.stride(-1) == rel_out.stride(-1) == 1
    assert q_scale.is_contiguous() and r_scale.is_contiguous()
    assert q_out.dtype == rel_out.dtype
    if total == 0:
        return
    alpha, n_floor = log_scaling if log_scaling is not None else (0.0, 1.0)
    _query_rel_kernel[(total, heads)](
        q, r, q_scale, r_scale, rel_proj, slot_mapping, positions, q_out, rel_out,
        num_tokens, q.stride(0), r.stride(0), q_out.stride(0), q_out.stride(1),
        rel_out.stride(0), rel_out.stride(1), rel_proj.stride(0), extent,
        float(q_clip), float(r_clip), float(alpha), float(n_floor),
        D=head_dim, DR=d_rel,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=max(2, triton.next_power_of_2(d_rel)),
        BLOCK_E=min(BLOCK_E, max(16, triton.next_power_of_2(extent))),
        LOG_SCALE=log_scaling is not None,
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip


@triton.jit(do_not_specialize=["num_tokens"])
def _gate_kernel(
    attn_ptr,
    gate_ptr,
    slot_ptr,
    out_ptr,
    num_tokens,
    stride_a,
    stride_g,
    stride_o,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < C
    real = row < num_tokens
    slot = tl.load(slot_ptr + row, mask=real, other=-1)
    row64 = row.to(tl.int64)
    a = tl.load(attn_ptr + row64 * stride_a + cols, mask=mask, other=0.0)
    g = tl.load(gate_ptr + row64 * stride_g + cols, mask=mask, other=0.0)
    y = a.to(tl.float32) * tl.div_rn(1.0, 1.0 + exp(-g.to(tl.float32)))
    y = tl.where(real & (slot >= 0), y, 0.0)
    tl.store(
        out_ptr + row64 * stride_o + cols, y.to(out_ptr.dtype.element_ty), mask=mask
    )


def gate(
    attn_output: torch.Tensor,
    gate_logits: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """`attn_output` [P, heads, D] times sigmoid(`gate_logits` [P, heads * D])
    in fp32, rounded to the dtype of `out` [P, heads * D]; padding rows
    (module docstring) are zero."""
    total, channels = out.shape
    attn = attn_output.reshape(total, channels)
    assert attn.stride(-1) == gate_logits.stride(-1) == out.stride(-1) == 1
    assert gate_logits.shape == out.shape
    if total == 0:
        return out
    block = min(BLOCK_GATE, triton.next_power_of_2(channels))
    _gate_kernel[(total, triton.cdiv(channels, block))](
        attn, gate_logits, slot_mapping, out, num_tokens,
        attn.stride(0), gate_logits.stride(0), out.stride(0),
        C=channels, BLOCK=block, num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return out
