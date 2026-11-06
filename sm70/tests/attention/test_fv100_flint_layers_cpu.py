# SPDX-License-Identifier: Apache-2.0
"""Flint's two layer types on Flash-V100 (CPU, cpu_ops stand-ins).

Flint per TP2 rank: 16 query / 2 KV heads, head size 128, fp32 sinks [16]
on every layer; `sliding_attention` layers have sliding_window=513 (the
query token plus 512 keys, so the impl's window is (512, 0)) and
`full_attention` layers none (FORK models/flint/configs.py:40-74; M5
model.py:304-319). For each layer type this checks the route an engine
with AttentionBackendEnum.CUSTOM relies on:

* admission: no Triton fallback reason, K14 fused decode marked;
* eager mixed batch: decode rows -> fv100_decode_paged_sinks_fused (with
  the device counters), prefill groups -> fv100_prefill_paged_sinks, the
  window and sinks passed through, output equal to the SDPA reference;
* CUDA-graph capture: decode launches the max_model_len envelope on the
  fused op with counters prepared before capture, never allocates under
  capture, and without counters takes the two-launch op (no fail-closed);
* sliding layers: pages wholly before the window may be upstream's
  null block (all-zero KV) after the sliding-window manager frees them;
  the result must not change.

Also the capability flag `fv100.supports_sinks()` on fake op sets.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# K06 helpers/fixtures: builder, LBNHC cache, CommonAttentionMetadata,
# reference check, stable-libtorch stub (also gates on vLLM being importable).
from test_fv100_backend_cpu import (  # noqa: E402,F401
    _common_metadata,
    _forward_case,
    _have_vllm,
    _lbnhc_cache,
    _stub_stable_libtorch,
    builder,
)

FUSED = "fv100_decode_paged_sinks_fused"
TWO = "fv100_decode_paged_sinks"
PREFILL = "fv100_prefill_paged_sinks"
TP2_HEADS = (16, 2)
HEAD = 128
LAYERS = pytest.mark.parametrize(
    ("window", "impl_window"),
    [(513, (512, 0)), (None, (-1, -1))],
    ids=["sliding_attention", "full_attention"],
)


# ------------------------------------------------------- capability flag


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        (("fv100_prefill_paged_sinks", "fv100_decode_paged_sinks"), True),
        (
            (
                "fv100_prefill_paged_sinks",
                "fv100_decode_paged_sinks",
                "fv100_decode_paged",
            ),
            True,
        ),
        (("fv100_prefill_paged_sinks",), False),  # prefill sinks only
        (("fv100_decode_paged_sinks",), False),  # decode sinks only
        (("fv100_decode_paged", "fv100_prefill_paged"), False),  # sink-free build
        ((), False),  # no _sm70_C
    ],
    ids=["both", "both+more", "prefill-only", "decode-only", "sink-free", "none"],
)
def test_supports_sinks_needs_both_sink_ops(monkeypatch, ops, expected):
    from vllm_sm70.ops import fv100

    monkeypatch.setattr(fv100, "has_op", lambda name: name in ops)
    assert fv100.supports_sinks() is expected


def test_supports_sinks_with_stub_schemas():
    """Against the real registry: the K05 schemas defined in a Python
    torch.library (as test_fv100_ops_cpu does) make the flag true."""
    sys.path.insert(0, str(HERE.parent / "ops"))
    from _schemas import sm70_schemas

    from vllm_sm70.ops import fv100

    lib = None
    if not fv100.supports_sinks():
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name in fv100.SINK_OPS:
            if not fv100.has_op(name):
                lib.define(sm70_schemas(area="flash_v100")[name])
    try:
        assert fv100.supports_sinks()
    finally:
        if lib is not None:
            lib._destroy()


# ---------------------------------------------------------- layer routes

pytestmark_vllm = pytest.mark.skipif(
    not _have_vllm(), reason="vLLM source not importable"
)


@pytest.fixture
def flint_ops(monkeypatch):
    """CpuOps stand-ins, every K05 op plus the K14 fused op 'built'."""
    from _fv100_cpu_ops import CpuOps

    from vllm_sm70.attention.flash_v100 import dispatch
    from vllm_sm70.ops import fv100

    monkeypatch.delenv("VLLM_SM70_FLINT_ATTN_FUSED_DECODE", raising=False)
    built = set(fv100.OPS) | {fv100.FUSED_DECODE_OP}
    monkeypatch.setattr(fv100, "has_op", lambda name: name in built)
    ops = CpuOps()
    monkeypatch.setattr(fv100, "_op", ops)
    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: False)
    monkeypatch.setattr(dispatch, "_capturing", lambda: False)
    fv100.clear_workspaces()
    yield ops
    fv100.clear_workspaces()


def _flint_impl(window):
    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

    sinks = torch.linspace(-2.0, 6.0, TP2_HEADS[0], dtype=torch.float32)
    return FlashAttnV100Impl(
        TP2_HEADS[0], HEAD, HEAD**-0.5, TP2_HEADS[1], None, window, "auto", sinks=sinks
    )


def _capture(monkeypatch, active: bool):
    from vllm_sm70.attention.flash_v100 import dispatch
    from vllm_sm70.ops import fv100

    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: active)
    monkeypatch.setattr(dispatch, "_capturing", lambda: active)


@pytestmark_vllm
@LAYERS
def test_flint_layer_is_admitted(flint_ops, window, impl_window):
    impl = _flint_impl(window)
    assert impl._fv100_reason is None
    assert impl._fv100_fused_sinks is True
    assert tuple(impl.sliding_window) == impl_window
    assert impl.sinks.dtype == torch.float32 and impl.sinks.shape == (16,)


@pytestmark_vllm
@LAYERS
def test_flint_layer_eager_mixed_batch(flint_ops, builder, window, impl_window):
    """Decode rows on the fused op, prefill groups (fresh and chunked with a
    prefix) on the sink prefill op; output equals the reference."""
    impl = _flint_impl(window)
    _forward_case(
        builder,
        impl,
        query_lens=[1, 1, 1, 40, 7],
        seq_lens=[1500, 33, 700, 40, 900],
        heads=TP2_HEADS,
        head_size=HEAD,
    )
    names = [name for name, _ in flint_ops.calls]
    assert names == [FUSED, PREFILL, PREFILL]
    decode = flint_ops.calls[0][1]
    assert decode["rows"] == 3 and decode["window"] == impl_window
    assert decode["sinks"] is impl.sinks
    counters = decode["counters"]
    assert counters.dtype == torch.int32 and counters.numel() >= 3 * TP2_HEADS[1]
    for _, prefill in flint_ops.calls[1:]:
        assert prefill["window"] == impl_window and prefill["sinks"] is impl.sinks
        assert prefill["causal"] is True


def _decode_step(builder, impl, seq_lens, *, capture, seed=3):
    from _fv100_ref import decode_reference

    max_blocks = -(-max(seq_lens) // 16)
    kv = _lbnhc_cache(len(seq_lens) * max_blocks, 16, TP2_HEADS[1], HEAD, seed)
    cm = _common_metadata([1] * len(seq_lens), seq_lens, 16, max_blocks)
    md = builder.build(0, cm)
    md.cudagraph_capture = capture
    gen = torch.Generator().manual_seed(seed + 1)
    query = torch.randn(len(seq_lens), TP2_HEADS[0], HEAD, generator=gen).half()
    output = torch.empty_like(query)
    impl.forward(None, query, None, None, kv, md, output)
    k, v = kv.transpose(1, 2).split(HEAD, dim=-1)
    ref = decode_reference(
        query,
        k,
        v,
        cm.block_table_tensor,
        cm.seq_lens,
        scale=impl.scale,
        window=tuple(impl.sliding_window),
        sinks=impl.sinks,
    )
    torch.testing.assert_close(output.float(), ref, atol=2e-3, rtol=2e-3)


@pytestmark_vllm
@LAYERS
@pytest.mark.parametrize("prepared", [True, False], ids=["prepared", "no-counters"])
def test_flint_layer_decode_under_capture(
    flint_ops, builder, monkeypatch, window, impl_window, prepared
):
    """Captured decode: the launch covers the max_model_len envelope (graphs
    replay at any length); with counters prepared before capture (as
    init_fused_decode does on CUDA) it is the fused op, otherwise the
    two-launch op. Nothing is allocated under capture either way."""
    from vllm_sm70.ops import fv100

    impl = _flint_impl(window)
    if prepared:
        assert fv100._fused_decode_counters(torch.device("cpu"), 0) is not None
    before = {k: e.buffer for k, e in fv100._fused_decode_counter_cache.items()}
    warned: list[str] = []
    monkeypatch.setattr(fv100.logger, "warning", lambda m, *a: warned.append(m % a))
    _capture(monkeypatch, True)
    _decode_step(builder, impl, [1500, 33, 700, 4000], capture=True)
    ((name, decode),) = flint_ops.calls
    assert name == (FUSED if prepared else TWO)
    assert decode["window"] == impl_window
    assert decode["launch"] * decode["partition"] >= builder.max_model_len
    # No counters created or replaced under capture.
    after = {k: e.buffer for k, e in fv100._fused_decode_counter_cache.items()}
    assert after.keys() == before.keys()
    assert all(after[k] is before[k] for k in before)
    assert bool(warned) is not prepared  # two-launch fallback is warned once
    assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (
        (1, 0) if prepared else (0, 1)
    )


@pytestmark_vllm
def test_sliding_layer_ignores_freed_pages(flint_ops, builder):
    """Sliding layers: block-table entries for pages wholly before the
    513-token window point at a zero ("null") block, as upstream's
    sliding-window manager leaves them; decode and chunked prefill give
    the same output as with the original pages."""
    impl = _flint_impl(513)
    block, seq_lens, query_lens = 16, [2000, 1200], [1, 24]
    max_blocks = -(-max(seq_lens) // block)
    kv = _lbnhc_cache(len(seq_lens) * max_blocks + 1, block, 2, HEAD, seed=9)
    null = kv.shape[0] - 1
    kv[null].zero_()
    cm = _common_metadata(query_lens, seq_lens, block, max_blocks)
    gen = torch.Generator().manual_seed(10)
    query = torch.randn(sum(query_lens), 16, HEAD, generator=gen).half()

    def run(table):
        cm.block_table_tensor.copy_(table)
        md = builder.build(0, cm)
        out = torch.empty_like(query)
        impl.forward(None, query, None, None, kv, md, out)
        return out

    original = cm.block_table_tensor.clone()
    full = run(original)
    freed = original.clone()
    for row, (q_len, s_len) in enumerate(zip(query_lens, seq_lens)):
        first_query = s_len - q_len  # earliest query position of the row
        first_key = max(0, first_query - 512)
        freed[row, : first_key // block] = null  # pages wholly before window
        assert first_key // block > 0
    assert torch.equal(run(freed), full)
    assert [n for n, _ in flint_ops.calls] == [FUSED, PREFILL] * 2
