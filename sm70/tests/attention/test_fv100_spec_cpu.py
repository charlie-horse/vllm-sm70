# SPDX-License-Identifier: Apache-2.0
"""K07 CPU checks: Flash-V100 spec-decode verify (smallq -> XQA).

Needs upstream v0.33.0's Python importable (skipped otherwise). Covers the
token-wise verify expansion and its persistent buffers (restating fork
tests/v1/attention/test_flash_v100_spec_metadata.py
:test_persistent_draft_buffers_keep_addresses_across_replays and
:test_capture_hook_preserves_small_query_lengths for v0.33), the builder
with a speculative config (reorder threshold, decode/verify split, the
non-causal DFlash draft staying on paged prefill: fork test_dflash2.py
:test_noncausal_draft_cannot_enter_flash_v100_small_query_fast_path and
:test_noncausal_dflash_capture_binds_paged_prefix_attention), the kill
switch, the graph-support level and the impl forward on CPU stand-ins of
the K05 ops against the SDPA reference.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPEC_TOKENS = 7
L = SPEC_TOKENS + 1


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


requires_vllm = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


# ----------------------------------------------- expansion (no vLLM needed)


def test_expansion_matches_reference_and_pads_with_zero():
    from vllm_sm70.attention.flash_v100.spec.verify import (
        VerifyExpansion,
        expand_reference,
    )

    exp = VerifyExpansion(64, torch.device("cpu"))
    query_lens, seq_lens = [8, 8, 1, 5], [100, 9, 40, 2000]
    qsl = torch.tensor([0, 8, 16, 17, 22], dtype=torch.int32)
    table = torch.arange(4 * 6, dtype=torch.int32).view(4, 6)
    s, t = exp.expand(
        qsl,
        torch.tensor(seq_lens, dtype=torch.int32),
        table,
        num_rows=4,
        num_mapped_tokens=22,
        num_tokens=24,
    )
    rows, lens = expand_reference(query_lens, seq_lens)
    assert s[:22].tolist() == lens and s[22:].tolist() == [0, 0]
    assert torch.equal(t[:22], table[rows])
    assert s.data_ptr() == exp.seq_lens.data_ptr()
    assert lens[:8] == list(range(93, 101))  # 8 queries ending at 100 keys


def test_expansion_buffers_keep_addresses_across_steps():
    from vllm_sm70.attention.flash_v100.spec.verify import VerifyExpansion

    exp = VerifyExpansion(32, torch.device("cpu"))
    table = torch.zeros((2, 4), dtype=torch.int32)
    ptrs = set()
    for step in range(3):
        qsl = torch.tensor([0, 8, 16], dtype=torch.int32)
        lens = torch.tensor([50 + step, 70], dtype=torch.int32)
        s, t = exp.expand(qsl, lens, table, 2, 16, 16)
        ptrs.add((s.data_ptr(), t.data_ptr()))
        assert s[7].item() == 50 + step
    assert len(ptrs) == 1


def test_expansion_rejects_more_rows_than_the_buffer():
    from vllm_sm70.attention.flash_v100.spec.verify import VerifyExpansion

    exp = VerifyExpansion(8, torch.device("cpu"))
    with pytest.raises(ValueError, match="exceed"):
        exp.expand(
            torch.tensor([0, 16], dtype=torch.int32),
            torch.tensor([30], dtype=torch.int32),
            torch.zeros((1, 2), dtype=torch.int32),
            1,
            16,
            16,
        )


def test_kill_switch_registered_and_reads_only_zero(monkeypatch):
    from vllm_sm70 import envs
    from vllm_sm70.attention.flash_v100 import spec

    assert "VLLM_SM70_FLASH_V100_SPEC" in envs._SECTIONS["attention"]
    monkeypatch.delenv(spec.KILL_SWITCH, raising=False)
    assert spec.spec_verify_enabled()
    monkeypatch.setenv(spec.KILL_SWITCH, "1")
    assert spec.spec_verify_enabled()
    monkeypatch.setenv(spec.KILL_SWITCH, "0")
    assert not spec.spec_verify_enabled()


# ------------------------------------------------------------- builder


def _cfg(heads, head_size, spec_tokens, max_num_seqs=8, max_model_len=4096):
    from vllm.config import CUDAGraphMode

    num_heads, num_kv_heads = heads
    model = SimpleNamespace(
        get_num_attention_heads=lambda p: num_heads,
        get_num_kv_heads=lambda p: num_kv_heads,
        get_head_size=lambda: head_size,
        rswa_window=None,
        max_model_len=max_model_len,
    )
    spec = (
        SimpleNamespace(num_speculative_tokens=spec_tokens, parallel_drafting=False)
        if spec_tokens
        else None
    )
    return SimpleNamespace(
        model_config=model,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=[]
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=256
        ),
        speculative_config=spec,
        use_v2_model_runner=False,
    )


def _builder(monkeypatch, heads=(12, 2), head_size=256, spec_tokens=SPEC_TOKENS):
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.kv_cache_interface import KVQuantMode

    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100MetadataBuilder

    monkeypatch.setattr(
        triton_attn, "get_num_attention_heads_from_layers", lambda *a: heads[0]
    )
    spec = SimpleNamespace(block_size=16, kv_quant_mode=KVQuantMode.NONE)
    return FlashAttnV100MetadataBuilder(
        spec, ["l0"], _cfg(heads, head_size, spec_tokens), torch.device("cpu")
    )


def _cm(query_lens, seq_lens, causal=True, block_size=16, width=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    width = width or -(-max(seq_lens) // block_size)
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens).cumsum(0)
    n = int(qsl[-1])
    table = torch.arange(len(query_lens) * width, dtype=torch.int32).view(-1, width)
    return CommonAttentionMetadata(
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=n,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=table,
        slot_mapping=torch.zeros(n, dtype=torch.int64),
        causal=causal,
    )


@requires_vllm
def test_builder_reorders_verify_rows_into_decodes(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SPEC", raising=False)
    b = _builder(monkeypatch)
    assert b.reorder_batch_threshold == L and b.spec_verify
    cm = _cm([L, L, 1, 20], [300, 9, 40, 500])
    md = b.build(0, cm)
    assert md.spec_verify and md.num_decodes == 3 and md.decode_rows == 2 * L + 1
    from vllm_sm70.attention.flash_v100.spec.verify import expand_reference

    rows, lens = expand_reference([L, L, 1], [300, 9, 40])
    assert md.decode_seq_lens.tolist() == lens
    assert torch.equal(md.decode_block_table, cm.block_table_tensor[rows])
    (group,) = md.prefill_groups
    assert (group.query_len, group.rows.tolist()) == (20, [3])
    # Persistent: the forward of a captured step reads these same buffers.
    md2 = b.build(0, _cm([L, L, 1, 20], [301, 10, 41, 520]))
    assert md2.decode_seq_lens.data_ptr() == md.decode_seq_lens.data_ptr()
    assert md2.fv_seq_lens.data_ptr() == md.fv_seq_lens.data_ptr()


@requires_vllm
def test_capture_metadata_keeps_verify_lengths(monkeypatch):
    b = _builder(monkeypatch)
    md = b.build_for_cudagraph_capture(_cm([L] * 3, [L] * 3))
    assert md.cudagraph_capture and md.decode_rows == 3 * L
    assert md.decode_seq_lens.tolist() == list(range(1, L + 1)) * 3


@requires_vllm
def test_kill_switch_restores_k06_routing(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_SPEC", "0")
    b = _builder(monkeypatch)
    assert b.reorder_batch_threshold == 1 and b.verify is None
    md = b.build(0, _cm([1, L, L], [40, 300, 9]))
    assert not md.spec_verify and md.decode_rows == 1
    (group,) = md.prefill_groups
    assert group.query_len == L and group.rows.tolist() == [1, 2]


@requires_vllm
@pytest.mark.parametrize(
    "env,expected", [(None, "UNIFORM_BATCH"), ("0", "UNIFORM_SINGLE_TOKEN_DECODE")]
)
def test_graph_support_level(monkeypatch, env, expected):
    from vllm.v1.attention.backend import AttentionCGSupport

    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100MetadataBuilder

    if env is None:
        monkeypatch.delenv("VLLM_SM70_FLASH_V100_SPEC", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLASH_V100_SPEC", env)
    cfg = _cfg((12, 2), 256, SPEC_TOKENS)
    got = FlashAttnV100MetadataBuilder.get_cudagraph_support(cfg, None)
    assert got is AttentionCGSupport[expected]
    plain = _cfg((12, 2), 256, 0)
    assert FlashAttnV100MetadataBuilder.get_cudagraph_support(plain, None) is (
        AttentionCGSupport.UNIFORM_BATCH
    )


@requires_vllm
def test_noncausal_draft_stays_on_paged_prefill(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SPEC", raising=False)
    b = _builder(monkeypatch, heads=(8, 2), head_size=128)
    md = b.build(0, _cm([L] * 4, [2000, 300, 40, 9], causal=False))
    assert not md.spec_verify and md.decode_rows == 0 and md.num_prefills == 4
    (group,) = md.prefill_groups
    # A uniform draft batch binds persistent arange slices (capture-safe).
    assert group.rows.data_ptr() == b._arange.data_ptr()
    md2 = b.build(0, _cm([L] * 4, [2001, 301, 41, 10], causal=False))
    (group2,) = md2.prefill_groups
    assert group2.token_index.data_ptr() == group.token_index.data_ptr()


# ------------------------------------------------------- impl forward (CPU)


@pytest.fixture
def cpu_ops(monkeypatch):
    from _fv100_cpu_ops import CpuOps

    from vllm_sm70.ops import fv100

    ops = CpuOps()
    monkeypatch.setattr(fv100, "_op", ops)
    fv100.clear_workspaces()
    return ops


def _lbnhc(num_blocks, block_size, h_kv, d, seed):
    gen = torch.Generator().manual_seed(seed)
    storage = torch.randn(num_blocks, block_size, h_kv, 2 * d, generator=gen).half()
    return storage.permute(0, 2, 1, 3)


def _run_case(
    monkeypatch,
    cpu_ops,
    *,
    heads,
    head_size,
    window,
    sinks,
    query_lens,
    seq_lens,
    causal=True,
):
    from _fv100_ref import prefill_reference

    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl, symmetric_window

    b = _builder(monkeypatch, heads=heads, head_size=head_size)
    cm = _cm(query_lens, seq_lens, causal=causal)
    md = b.build(0, cm)
    kv = _lbnhc(cm.block_table_tensor.numel(), 16, heads[1], head_size, 1)
    impl = FlashAttnV100Impl(
        heads[0],
        head_size,
        head_size**-0.5,
        heads[1],
        None,
        window,
        "auto",
        sinks=sinks,
    )
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(cm.num_actual_tokens, heads[0], head_size, generator=gen).half()
    out = torch.empty_like(q)
    impl.forward(None, q, None, None, kv, md, out)
    k, v = kv.transpose(1, 2).split(head_size, dim=-1)
    starts = cm.query_start_loc_cpu.tolist()
    for r in range(len(query_lens)):
        a, z = starts[r], starts[r + 1]
        ref = prefill_reference(
            q[None, a:z],
            k,
            v,
            cm.block_table_tensor[r : r + 1],
            cm.seq_lens[r : r + 1],
            scale=impl.scale,
            causal=causal,
            window=symmetric_window(tuple(impl.sliding_window), causal),
            sinks=sinks,
        )[0]
        torch.testing.assert_close(out[a:z].float(), ref, atol=2e-3, rtol=2e-3)
    return [c[0] for c in cpu_ops.calls]


@requires_vllm
@pytest.mark.parametrize("heads", [(12, 2), (8, 1)], ids=["gqa6", "gqa8"])
def test_qwen_verify_takes_xqa(monkeypatch, cpu_ops, heads):
    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SPEC", raising=False)
    names = _run_case(
        monkeypatch,
        cpu_ops,
        heads=heads,
        head_size=256,
        window=None,
        sinks=None,
        query_lens=[L, L, L],
        seq_lens=[300, 9, 1000],
    )
    assert names == ["fv100_decode_paged_xqa"]
    assert cpu_ops.calls[0][1]["rows"] == 3 * L


@requires_vllm
@pytest.mark.parametrize("fused", ["1", "0"])
def test_flint_verify_takes_sink_decode(monkeypatch, cpu_ops, fused):
    # K14: with its op built and its switch on, Flint's sink decode is the
    # one-launch fused kernel; otherwise the two-launch one. Which ops exist
    # depends on the build (a host with the real _sm70_C has the fused op).
    from vllm_sm70.ops import fv100

    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SPEC", raising=False)
    monkeypatch.setenv("VLLM_SM70_FLINT_ATTN_FUSED_DECODE", fused)
    decode_op = (
        "fv100_decode_paged_sinks_fused"
        if fv100.fused_decode_available()
        else "fv100_decode_paged_sinks"
    )
    if fused == "0":
        assert decode_op == "fv100_decode_paged_sinks"
    sinks = torch.linspace(-2, 6, 16)
    names = _run_case(
        monkeypatch,
        cpu_ops,
        heads=(16, 2),
        head_size=128,
        window=513,
        sinks=sinks,
        query_lens=[L, L, 30],
        seq_lens=[900, 600, 700],
    )
    assert names == [decode_op, "fv100_prefill_paged_sinks"]


@requires_vllm
def test_kill_switch_verifies_with_paged_prefill(monkeypatch, cpu_ops):
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_SPEC", "0")
    names = _run_case(
        monkeypatch,
        cpu_ops,
        heads=(8, 1),
        head_size=256,
        window=None,
        sinks=None,
        query_lens=[L, L],
        seq_lens=[300, 9],
    )
    assert names == ["fv100_prefill_paged"]


@requires_vllm
def test_noncausal_draft_forward(monkeypatch, cpu_ops):
    names = _run_case(
        monkeypatch,
        cpu_ops,
        heads=(8, 2),
        head_size=128,
        window=None,
        sinks=None,
        query_lens=[L, L],
        seq_lens=[300, 40],
        causal=False,
    )
    assert names == ["fv100_prefill_paged"]
    assert cpu_ops.calls[0][1]["causal"] is False


@requires_vllm
def test_noncausal_draft_window_is_two_sided(monkeypatch, cpu_ops):
    # DFlash drafts are non-causal with a sliding window: every draft token
    # sees w - 1 keys on each side, like Triton (|dist| < w) and upstream
    # FlashAttention's _maybe_symmetrize_window. A left-only (w - 1, 0)
    # window would drop the draft's future keys (Bea's c4130 GPU run).
    from vllm_sm70.attention.flash_v100.impl import symmetric_window

    assert symmetric_window((2047, 0), False) == (2047, 2047)
    assert symmetric_window((2047, 0), True) == (2047, 0)
    assert symmetric_window((-1, -1), False) == (-1, -1)
    names = _run_case(
        monkeypatch,
        cpu_ops,
        heads=(8, 2),
        head_size=128,
        window=64,
        sinks=None,
        query_lens=[L, L],
        seq_lens=[300, 40],
        causal=False,
    )
    assert names == ["fv100_prefill_paged"]
    assert cpu_ops.calls[0][1]["window"] == (63, 63)


@requires_vllm
@pytest.mark.parametrize("spec_tokens", [SPEC_TOKENS, 0], ids=["verify", "decode"])
def test_rebuild_refreshes_captured_buffers_in_place(monkeypatch, spec_tokens):
    # A captured graph keeps reading the tensors bound at capture. The V1
    # runner rebuilds metadata every step before replay, so each rebuild must
    # write the new lengths/tables into those same buffers (Bea's c4130 run:
    # replay without a rebuild read the capture-time seq_lens of 1).
    monkeypatch.delenv("VLLM_SM70_FLASH_V100_SPEC", raising=False)
    from vllm_sm70.attention.flash_v100.spec.verify import expand_reference

    q = spec_tokens + 1
    b = _builder(monkeypatch, spec_tokens=spec_tokens)
    # The runner's block table has a fixed width (max_model_len / block).
    width = 4096 // 16
    cap = b.build_for_cudagraph_capture(_cm([q] * 4, [q] * 4, width=width))
    assert cap.cudagraph_capture and cap.decode_rows == 4 * q
    for lens in ([600, 50, 4096, 9], [3000, 8, 2049, 700]):
        cm = _cm([q] * 4, lens, width=width)
        md = b.build(0, cm)
        assert md.decode_rows == cap.decode_rows
        assert md.fv_seq_lens.data_ptr() == cap.fv_seq_lens.data_ptr()
        assert md.fv_seq_lens.tolist() == lens
        assert md.decode_seq_lens.data_ptr() == cap.decode_seq_lens.data_ptr()
        rows, want = expand_reference([q] * 4, lens)
        assert md.decode_seq_lens.tolist() == want
        assert cap.decode_seq_lens.tolist() == want  # the captured view sees it
        if spec_tokens:
            assert md.spec_verify
            assert md.decode_block_table.data_ptr() == cap.decode_block_table.data_ptr()
            assert torch.equal(cap.decode_block_table, cm.block_table_tensor[rows])
