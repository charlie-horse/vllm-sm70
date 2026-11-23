# SPDX-License-Identifier: Apache-2.0
"""Kernels of the Inkling 2 DSpark draft (dspark.py) on sm_70: Triton, no
tl.dot, no bf16, libdevice tanh/exp. Each op has a torch form, taken with
``kernel=False``; the torch forms are the reference of the kernels.

    mix_rmsnorm      RMSNorm of an alpha-weighted sum of target features
    qk_norm_rope_    in place: per-head Q/K RMSNorm, then neox RoPE
    rope_            in place: neox RoPE
    silu_and_mul     silu(gate) * up
    add_softcap      cap * tanh(round(a + b) / cap), fp32 out
    markov_gate      the Markov head's GRU-style gate
    block_attention  a block of queries per request over its paged keys

Math runs in fp32. Where upstream (vLLM c32251b models/inkling2/nvidia/
ops/dspark.py) rounds to bf16, these round to the dtype of the tensors they
store, the model dtype (fp16). block_attention: every query of a request's
block sees the keys [seq_len - window, seq_len) (all keys with window 0);
a request with seq_len 0 gets zeros.
"""

from __future__ import annotations

import torch

from ._compat import exp, tanh, tl, triton

__all__ = [
    "add_softcap",
    "block_attention",
    "markov_gate",
    "mix_rmsnorm",
    "qk_norm_rope_",
    "rope_",
    "rope_cache",
    "silu_and_mul",
]


def rope_cache(head_dim: int, max_position: int, theta: float) -> torch.Tensor:
    """fp32 [max_position, head_dim]: cos of the half frequencies, then sin."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )
    freqs = torch.arange(max_position, dtype=torch.float64)[:, None] * inv_freq
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).float()


# --- mix_rmsnorm ----------------------------------------------------------------


@triton.jit
def _mix_rmsnorm_kernel(
    x_ptr,
    alpha_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    stride_tap,
    stride_h,
    eps,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mask = h < HIDDEN
    mixed = tl.zeros((BLOCK,), tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            x_ptr + row * stride_row + tap * stride_tap + h * stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mixed += x * tl.load(alpha_ptr + tap).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(mixed * mixed, 0) / HIDDEN + eps)
    weight = tl.load(weight_ptr + h, mask=mask, other=0.0).to(tl.float32)
    out = mixed * inv_rms * weight
    tl.store(out_ptr + row * HIDDEN + h, out.to(out_ptr.dtype.element_ty), mask=mask)


def mix_rmsnorm(
    targets: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    kernel: bool = True,
) -> torch.Tensor:
    """RMSNorm of sum_i alpha[i] * targets[:, i] for targets [R, taps, H];
    one rounding, to targets.dtype."""
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and weight.shape == (hidden,)
    if not kernel:
        mixed = (targets.float() * alpha.float()[:, None]).sum(1)
        out = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
        return (out * weight.float()).to(targets.dtype)
    out = targets.new_empty((rows, hidden))
    if rows:
        _mix_rmsnorm_kernel[(rows,)](
            targets,
            alpha.contiguous(),
            weight.contiguous(),
            out,
            *targets.stride(),
            float(eps),
            TAPS=taps,
            HIDDEN=hidden,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=8,
            enable_fp_fusion=False,
        )
    return out


# --- qk_norm_rope_ / rope_ ---------------------------------------------------------


@triton.jit
def _qk_norm_rope_kernel(
    x_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    pos_ptr,
    stride_row,
    q_size,
    eps,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    HALF: tl.constexpr = D // 2
    half = tl.arange(0, HALF)
    cos = tl.load(cos_sin_ptr + pos * D + half)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half)
    is_q = h < NHQ
    col = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    row = x_ptr + t * stride_row + col
    x1 = tl.load(row + half).to(tl.float32)
    x2 = tl.load(row + HALF + half).to(tl.float32)
    if NORM:
        inv = 1.0 / tl.sqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D + eps)
        w1 = tl.where(
            is_q, tl.load(q_weight_ptr + half), tl.load(k_weight_ptr + half)
        ).to(tl.float32)
        w2 = tl.where(
            is_q,
            tl.load(q_weight_ptr + HALF + half),
            tl.load(k_weight_ptr + HALF + half),
        ).to(tl.float32)
        x1 = (x1 * inv * w1).to(x_ptr.dtype.element_ty).to(tl.float32)
        x2 = (x2 * inv * w2).to(x_ptr.dtype.element_ty).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(row + half, o1.to(x_ptr.dtype.element_ty))
    tl.store(row + HALF + half, o2.to(x_ptr.dtype.element_ty))


def _rope_torch(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """neox RoPE of x [T, heads, D] (fp32 in, fp32 out)."""
    half = x.shape[-1] // 2
    table = cos_sin[positions.long()][:, None, :]
    cos, sin = table[..., :half], table[..., half:]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
    *,
    kernel: bool = True,
) -> None:
    """In place on qkv [T, (q + 2 k heads) * D]: per-head RMSNorm of the Q
    and K heads (rounded to qkv.dtype), then neox RoPE."""
    head_dim = cos_sin_cache.shape[-1]
    tokens = qkv.shape[0]
    if not kernel:
        width = (num_q_heads + num_k_heads) * head_dim
        heads = qkv[:, :width].float().view(tokens, -1, head_dim)
        weight = torch.cat(
            [
                q_weight.float().expand(num_q_heads, -1),
                k_weight.float().expand(num_k_heads, -1),
            ]
        )
        normed = heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + eps)
        normed = (normed * weight).to(qkv.dtype).float()
        roped = _rope_torch(normed, positions, cos_sin_cache)
        qkv[:, :width] = roped.flatten(1).to(qkv.dtype)
        return
    assert qkv.stride(1) == 1
    if tokens:
        _qk_norm_rope_kernel[(tokens, num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            qkv.stride(0),
            num_q_heads * head_dim,
            float(eps),
            NHQ=num_q_heads,
            D=head_dim,
            NORM=True,
            num_warps=1,
            enable_fp_fusion=False,
        )


def rope_(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    kernel: bool = True,
) -> None:
    """In place neox RoPE of x [T, heads, D] (fp32 math)."""
    tokens, heads, head_dim = x.shape
    assert x.stride(1) == head_dim and x.stride(2) == 1
    if not kernel:
        x.copy_(_rope_torch(x.float(), positions, cos_sin_cache).to(x.dtype))
        return
    if tokens:
        _qk_norm_rope_kernel[(tokens, heads)](
            x,
            x,
            x,
            cos_sin_cache,
            positions,
            x.stride(0),
            0,
            0.0,
            NHQ=0,
            D=head_dim,
            NORM=False,
            num_warps=1,
            enable_fp_fusion=False,
        )


# --- silu_and_mul ---------------------------------------------------------------------


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, stride_x, n, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n
    gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + n + cols, mask=mask, other=0.0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + exp(-gate))
    tl.store(out_ptr + row * n + cols, (silu * up).to(out_ptr.dtype.element_ty), mask)


def silu_and_mul(x: torch.Tensor, *, kernel: bool = True) -> torch.Tensor:
    """silu(gate) * up of x [T, 2 * n] in fp32, rounded once."""
    tokens, width = x.shape
    n = width // 2
    if not kernel:
        gate, up = x.float().chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)
    out = x.new_empty((tokens, n))
    if tokens:
        _silu_and_mul_kernel[(tokens, triton.cdiv(n, 1024))](
            x, out, x.stride(0), n, BLOCK=1024, num_warps=4, enable_fp_fusion=False
        )
    return out


# --- add_softcap ----------------------------------------------------------------------


@triton.jit
def _add_softcap_kernel(
    a_ptr, b_ptr, out_ptr, stride_a, stride_b, stride_o, cap, n, BLOCK: tl.constexpr
):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n
    a = tl.load(a_ptr + row * stride_a + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    x = (a + b).to(a_ptr.dtype.element_ty).to(tl.float32)
    x = tanh(tl.div_rn(x, cap)) * cap
    tl.store(out_ptr + row * stride_o + cols, x, mask)


def add_softcap(
    a: torch.Tensor, b: torch.Tensor, cap: float, *, kernel: bool = True
) -> torch.Tensor:
    """fp32 cap * tanh(round(a + b) / cap) of two [R, n] tensors; the sum
    rounds to a.dtype."""
    rows, n = a.shape
    if not kernel:
        x = (a.float() + b.float()).to(a.dtype).float()
        return torch.tanh(x / cap) * cap
    assert a.stride(1) == 1 and b.stride(1) == 1
    out = torch.empty((rows, n), dtype=torch.float32, device=a.device)
    if rows:
        _add_softcap_kernel[(triton.cdiv(n, 1024), rows)](
            a,
            b,
            out,
            a.stride(0),
            b.stride(0),
            out.stride(0),
            float(cap),
            n,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


# --- markov_gate ----------------------------------------------------------------------


@triton.jit
def _markov_gate_kernel(raw_ptr, state_ptr, out_ptr, latent_ptr, RANK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    dtype = state_ptr.dtype.element_ty
    raw = raw_ptr + row * 3 * RANK + c
    gate = tl.load(raw).to(dtype).to(tl.float32)
    cand = tl.load(raw + RANK).to(dtype).to(tl.float32)
    out = tl.load(raw + 2 * RANK).to(dtype).to(tl.float32)
    state = tl.load(state_ptr + row * RANK + c).to(tl.float32)
    gate = tl.div_rn(1.0, 1.0 + exp(-gate)).to(dtype).to(tl.float32)
    cand = tanh(cand).to(dtype).to(tl.float32)
    keep = (gate * state).to(dtype).to(tl.float32)
    update = ((1.0 - gate).to(dtype).to(tl.float32) * cand).to(dtype).to(tl.float32)
    tl.store(out_ptr + row * RANK + c, (keep + update).to(dtype))
    tl.store(latent_ptr + row * RANK + c, tanh(out).to(dtype))


def markov_gate(
    raw: torch.Tensor, state: torch.Tensor, *, kernel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate the fp32 rnn_proj output raw [R, 3 r] (rounded to state.dtype
    first) with state [R, r]; every op rounds to state.dtype. Returns
    (gate * state + (1 - gate) * tanh(cand), tanh(out))."""
    rows, rank = state.shape
    assert raw.shape == (rows, 3 * rank)
    if not kernel:
        gate, cand, out = raw.to(state.dtype).chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        return gate * state + (1.0 - gate) * torch.tanh(cand), torch.tanh(out)
    assert raw.is_contiguous() and state.is_contiguous()
    assert rank & (rank - 1) == 0
    new_state, latent = torch.empty_like(state), torch.empty_like(state)
    if rows:
        _markov_gate_kernel[(rows,)](
            raw,
            state,
            new_state,
            latent,
            RANK=rank,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return new_state, latent


# --- block_attention ----------------------------------------------------------------


@triton.jit
def _block_attention_kernel(
    q_ptr,
    key_ptr,
    value_ptr,
    out_ptr,
    table_ptr,
    seq_ptr,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_ot,
    stride_bt,
    window,
    scale,
    BLOCK_Q: tl.constexpr,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    req = t // BLOCK_Q
    kv_head = (h // GROUPS).to(tl.int64)
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    seq_len = tl.load(seq_ptr + req)
    start = tl.where(window > 0, tl.maximum(seq_len - window, 0), 0)
    q = tl.load(q_ptr + t * stride_qt + h * stride_qh + d).to(tl.float32)
    m = tl.full((), float("-inf"), tl.float32)
    total = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    first = (start // BLOCK_N) * BLOCK_N
    for base in range(first, seq_len, BLOCK_N):
        block = tl.load(table_ptr + req * stride_bt + base // BLOCK_N).to(tl.int64)
        pos = base + n
        valid = (pos >= start) & (pos < seq_len)
        offs = block * stride_kb + n[:, None] * stride_kn + kv_head * stride_kh
        k = tl.load(key_ptr + offs + d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(value_ptr + offs + d[None, :], mask=valid[:, None], other=0.0)
        s = tl.sum(k.to(tl.float32) * q[None, :], 1) * scale
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        p = tl.exp(s - m_new)
        p = tl.where(valid, p, 0.0)
        corr = tl.exp(m - m_new)
        total = total * corr + tl.sum(p, 0)
        acc = acc * corr + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m = m_new
    out = tl.where(total > 0, acc / tl.maximum(total, 1e-30), 0.0)
    tl.store(out_ptr + t * stride_ot + h * D + d, out.to(out_ptr.dtype.element_ty))


def _block_attention_torch(
    q, key_cache, value_cache, out, block_table, seq_lens, block, window
):
    tokens, heads, head_dim = q.shape
    page = key_cache.shape[1]
    groups = heads // key_cache.shape[2]
    view = out.view(tokens, heads, head_dim)
    for req in range(tokens // block):
        seq_len = int(seq_lens[req])
        rows = slice(req * block, (req + 1) * block)
        if seq_len == 0:
            view[rows] = 0
            continue
        start = max(0, seq_len - window) if window else 0
        pos = torch.arange(start, seq_len, device=q.device)
        blocks = block_table[req, pos // page].long()
        keys = key_cache[blocks, pos % page].float().repeat_interleave(groups, 1)
        vals = value_cache[blocks, pos % page].float().repeat_interleave(groups, 1)
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), keys) * head_dim**-0.5
        view[rows] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals).to(
            out.dtype
        )


def block_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block: int,
    window: int,
    kernel: bool = True,
) -> torch.Tensor:
    """Attention of q [R * block, heads, D] into out [R * block, heads * D]
    over the paged keys of request r: key_cache / value_cache [blocks,
    page, kv_heads, D], block_table [R, max_blocks], seq_lens [R]."""
    tokens, heads, head_dim = q.shape
    if not kernel:
        _block_attention_torch(
            q, key_cache, value_cache, out, block_table, seq_lens, block, window
        )
        return out
    page = key_cache.shape[1]
    assert page == triton.next_power_of_2(page) and head_dim == triton.next_power_of_2(
        head_dim
    )
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(-1) == 1
    assert q.stride(-1) == 1 and out.stride(-1) == 1
    if tokens:
        _block_attention_kernel[(tokens, heads)](
            q,
            key_cache,
            value_cache,
            out,
            block_table,
            seq_lens,
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            out.stride(0),
            block_table.stride(0),
            int(window),
            head_dim**-0.5,
            BLOCK_Q=block,
            GROUPS=heads // key_cache.shape[2],
            D=head_dim,
            BLOCK_N=page,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
