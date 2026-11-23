# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 DSpark draft ops (inkling2_sm70/ops/dspark.py), on CPU.

- the Triton kernels through Triton's interpreter against the torch forms
  of the same ops, in fp32 within 1e-5 and in fp16 within 2e-3 of the
  largest reference value: ragged widths, GQA, a window that starts inside
  a page, a key range that crosses pages, an empty request;
- the torch form of block_attention against dense attention over the
  unpaged keys.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite); in the parent they show as skipped. The sm_70
compile check is in test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
DTYPES = [
    pytest.param(torch.float32, 1e-5, id="fp32"),
    pytest.param(torch.float16, 2e-3, id="fp16"),
]


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 and require the
    interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu",
            "-k",
            "not test_interpreted_suite",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 20, summary
    assert "skipped" not in summary, summary


def _ops():
    from vllm_sm70.models.inkling2_sm70.ops import dspark

    return dspark


def _err(got: torch.Tensor, want: torch.Tensor) -> float:
    assert torch.isfinite(got).all()
    scale = max(1.0, want.float().abs().max().item())
    return (got.float() - want.float()).abs().max().item() / scale


@pytest.fixture
def ops():
    if not INTERPRETED:
        pytest.skip("runs in the TRITON_INTERPRET=1 child (test_interpreted_suite)")
    return _ops()


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(*shape, dtype, seed=0, scale=1.0):
    return (torch.randn(*shape, generator=_gen(seed)) * scale).to(dtype)


# --- interpreted kernels ----------------------------------------------------------


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize(("rows", "taps", "hidden"), [(1, 3, 64), (5, 3, 200)])
def test_mix_rmsnorm(ops, dtype, tol, rows, taps, hidden):
    x = _randn(rows, taps, hidden, dtype=dtype, scale=3.0)
    alpha = torch.softmax(torch.randn(taps, generator=_gen(1)), -1)
    weight = _randn(hidden, dtype=dtype, seed=2)
    got = ops.mix_rmsnorm(x, alpha, weight, 1e-6, kernel=True)
    want = ops.mix_rmsnorm(x, alpha, weight, 1e-6, kernel=False)
    assert _err(got, want) <= tol


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize(("nq", "nk"), [(4, 2), (0, 2)], ids=["qk", "k_only"])
def test_qk_norm_rope(ops, dtype, tol, nq, nk):
    d, tokens = 16, 6
    cos_sin = ops.rope_cache(d, 64, 10000.0)
    positions = torch.tensor([0, 1, 7, 30, 31, 63])
    # A strided view, like the K half of the packed context K/V.
    packed = _randn(tokens, 2 * (nq + 2 * nk) * d, dtype=dtype, scale=2.0)
    q_w = _randn(d, dtype=dtype, seed=3)
    k_w = _randn(d, dtype=dtype, seed=4)
    got, want = packed.clone(), packed.clone()
    view = slice(0, (nq + 2 * nk) * d)
    ops.qk_norm_rope_(got[:, view], positions, q_w, k_w, cos_sin, nq, nk, 1e-6)
    ops.qk_norm_rope_(
        want[:, view], positions, q_w, k_w, cos_sin, nq, nk, 1e-6, kernel=False
    )
    assert _err(got, want) <= tol
    rest = slice((nq + nk) * d, None)
    assert torch.equal(got[:, rest], packed[:, rest])


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
def test_rope(ops, dtype, tol):
    d = 32
    cos_sin = ops.rope_cache(d, 128, 10000.0)
    positions = torch.tensor([3, 0, 127])
    x = _randn(3, 5, d, dtype=dtype)
    got, want = x.clone(), x.clone()
    ops.rope_(got, positions, cos_sin)
    ops.rope_(want, positions, cos_sin, kernel=False)
    assert _err(got, want) <= tol


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("n", [64, 1100])
def test_silu_and_mul(ops, dtype, tol, n):
    x = _randn(3, 2 * n, dtype=dtype, scale=4.0)
    assert _err(ops.silu_and_mul(x), ops.silu_and_mul(x, kernel=False)) <= tol


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
def test_add_softcap(ops, dtype, tol):
    a = _randn(3, 1500, dtype=dtype, scale=20.0)
    b = _randn(3, 1500, dtype=dtype, seed=1, scale=5.0)
    got = ops.add_softcap(a, b, 30.0)
    assert got.dtype == torch.float32
    assert _err(got, ops.add_softcap(a, b, 30.0, kernel=False)) <= tol
    assert got.abs().max() < 30.0


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
def test_markov_gate(ops, dtype, tol):
    raw = torch.randn(4, 3 * 16, generator=_gen(0)) * 3
    state = _randn(4, 16, dtype=dtype, seed=1)
    got = ops.markov_gate(raw, state)
    want = ops.markov_gate(raw, state, kernel=False)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == dtype and _err(g, w) <= tol


def _paged(dtype, *, seq_lens, page, kv_heads, d, seed=0):
    """Paged key/value caches with shuffled pages, and their block table."""
    blocks_per = [max(1, -(-s // page)) for s in seq_lens]
    total = sum(blocks_per) + 2
    order = torch.randperm(total, generator=_gen(seed))
    table = torch.zeros(len(seq_lens), max(blocks_per), dtype=torch.int32)
    used = 0
    for r, n in enumerate(blocks_per):
        table[r, :n] = order[used : used + n].int()
        used += n
    keys = _randn(total, page, kv_heads, d, dtype=dtype, seed=seed + 1)
    values = _randn(total, page, kv_heads, d, dtype=dtype, seed=seed + 2)
    return keys, values, table, torch.tensor(seq_lens, dtype=torch.int32)


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@pytest.mark.parametrize("window", [0, 13])
def test_block_attention(ops, dtype, tol, window):
    block, heads, kv_heads, d, page = 4, 4, 2, 16, 8
    seq_lens = [21, 0, 4, 40]
    keys, values, table, lens = _paged(
        dtype, seq_lens=seq_lens, page=page, kv_heads=kv_heads, d=d
    )
    q = _randn(len(seq_lens) * block, heads, d, dtype=dtype, seed=5)
    got = torch.full((q.shape[0], heads * d), float("nan"), dtype=dtype)
    want = got.clone()
    kw = {"block_table": table, "seq_lens": lens, "block": block, "window": window}
    ops.block_attention(q, keys, values, got, **kw)
    ops.block_attention(q, keys, values, want, **kw, kernel=False)
    assert _err(got, want) <= tol
    assert torch.equal(got[block : 2 * block], torch.zeros_like(got[:block]))


# --- the torch forms ----------------------------------------------------------------


@pytest.mark.parametrize("window", [0, 13])
def test_block_attention_torch_form_is_dense_attention(window):
    ops = _ops()
    block, heads, kv_heads, d, page = 4, 4, 2, 16, 8
    seq_lens = [21, 4, 40]
    keys, values, table, lens = _paged(
        torch.float32, seq_lens=seq_lens, page=page, kv_heads=kv_heads, d=d, seed=7
    )
    q = _randn(len(seq_lens) * block, heads, d, dtype=torch.float32, seed=9)
    out = torch.empty(q.shape[0], heads * d)
    ops.block_attention(
        q,
        keys,
        values,
        out,
        block_table=table,
        seq_lens=lens,
        block=block,
        window=window,
        kernel=False,
    )
    for r, s in enumerate(seq_lens):
        pos = torch.arange(max(0, s - window) if window else 0, s)
        rows = table[r, pos // page].long()
        k = keys[rows, pos % page].repeat_interleave(heads // kv_heads, 1)
        v = values[rows, pos % page].repeat_interleave(heads // kv_heads, 1)
        qr = q[r * block : (r + 1) * block]
        want = torch.nn.functional.scaled_dot_product_attention(
            qr.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)
        got = out[r * block : (r + 1) * block].view(block, heads, d)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
