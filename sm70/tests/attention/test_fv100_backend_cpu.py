# SPDX-License-Identifier: Apache-2.0
"""K06 CPU checks for the Flash-V100 attention backend (no GPU).

Needs upstream v0.33.0's Python source importable (installed vllm, or the
source tree on PYTHONPATH); skipped otherwise. No compiled vLLM extension:
`vllm._C_stable_libtorch` is stubbed when absent, as in tests/plugin.

Covers: CUSTOM registration, backend class flags and validate_configuration,
get_supported_kernel_block_sizes(spec), the platform selection rules (a)-(e)
and the kill switch, the LBNHC split, the metadata builder on fake
CommonAttentionMetadata, and the impl forward routed onto CPU stand-ins of
the K05 ops (_fv100_cpu_ops.py) against the SDPA reference.
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


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM source not importable")


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


@pytest.fixture
def backend_cls():
    from vllm_sm70.attention.flash_v100.backend import FlashAttnV100Backend

    return FlashAttnV100Backend


# ------------------------------------------------------------- registration


def test_register_points_custom_at_flash_v100(monkeypatch, backend_cls):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70 import attention

    monkeypatch.setattr(attention, "_registered", False)
    attention.register()
    attention.register()  # idempotent
    try:
        assert AttentionBackendEnum.CUSTOM.get_class() is backend_cls
        assert attention.custom_backend_path() == attention.BACKEND_PATH
    finally:
        AttentionBackendEnum.CUSTOM.clear_override()


def test_kill_switch_is_registered():
    from vllm_sm70 import envs

    assert "VLLM_SM70_FLASH_V100" in envs._SECTIONS["attention"]


# --------------------------------------------------------------- class flags


def test_backend_flags(backend_cls):
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.kv_cache_layout import KVCacheLayout

    assert backend_cls.get_name() == "FLASH_ATTN_V100"
    assert backend_cls.supported_dtypes == [torch.float16]
    assert backend_cls.supported_kv_cache_dtypes == ["auto", "float16"]
    assert backend_cls.get_supported_head_sizes() == [128, 256]
    assert [backend_cls.supports_head_size(h) for h in (64, 128, 256)] == [
        False,
        True,
        True,
    ]
    assert backend_cls.supports_sink() and backend_cls.supports_sliding_window()
    assert backend_cls.supports_non_causal()
    assert not backend_cls.supports_rswa()
    assert not backend_cls.supports_mm_prefix()
    assert not backend_cls.supports_device_cpu_query_lens_mismatch()
    assert backend_cls.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)
    assert backend_cls.forward_includes_kv_cache_update is False
    assert backend_cls.supports_attn_type(AttentionType.DECODER)
    assert not backend_cls.supports_attn_type(AttentionType.ENCODER_ONLY)
    spec = object()
    assert backend_cls.customize_spec(spec) is spec


def test_kernel_block_sizes_take_a_spec(backend_cls):
    from vllm.v1.attention.backend import MultipleOf

    spec = SimpleNamespace(block_size=784)
    (size,) = backend_cls.get_supported_kernel_block_sizes(spec)
    assert isinstance(size, MultipleOf) and size.base == 16
    assert backend_cls.get_supported_kernel_block_sizes() == [MultipleOf(16)]
    assert backend_cls.supports_block_size(784)
    assert not backend_cls.supports_block_size(8)


def test_builder_flags():
    from vllm.v1.attention.backend import AttentionCGSupport

    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100MetadataBuilder

    assert FlashAttnV100MetadataBuilder._cudagraph_support is (
        AttentionCGSupport.UNIFORM_BATCH
    )
    assert FlashAttnV100MetadataBuilder.supports_draft_decode_metadata_update is False
    spec_cfg = SimpleNamespace(speculative_config=object())
    assert FlashAttnV100MetadataBuilder.get_cudagraph_support(spec_cfg, None) is (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    plain = SimpleNamespace(speculative_config=None)
    assert FlashAttnV100MetadataBuilder.get_cudagraph_support(plain, None) is (
        AttentionCGSupport.UNIFORM_BATCH
    )


def _selector(**kw):
    from vllm.v1.attention.selector import AttentionSelectorConfig

    base = dict(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        has_sink=True,
        has_sliding_window=True,
    )
    base.update(kw)
    return AttentionSelectorConfig(**base)


def _cap(major, minor):
    from vllm.platforms.interface import DeviceCapability

    return DeviceCapability(major, minor)


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({}, None),
        ({"head_size": 256, "has_sink": False, "has_sliding_window": False}, None),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
        ({"dtype": torch.bfloat16}, "dtype"),
        ({"head_size": 64}, "head_size"),
        ({"block_size": 8}, "block_size"),
        ({"use_mla": True}, "MLA"),
        ({"use_rswa": True}, "R-SWA"),
        ({"use_mm_prefix": True}, "multimodal"),
    ],
)
def test_validate_configuration(backend_cls, kw, reason):
    invalid = backend_cls.validate_configuration(
        device_capability=_cap(7, 0), **_selector(**kw)._asdict()
    )
    if reason is None:
        assert invalid == []
    else:
        assert any(reason in r for r in invalid), invalid


def test_validate_rejects_other_capabilities(backend_cls):
    invalid = backend_cls.validate_configuration(
        device_capability=_cap(7, 5), **_selector()._asdict()
    )
    assert "compute capability not supported" in invalid


# ------------------------------------------------------------------ selection


@pytest.fixture
def platform(monkeypatch):
    from vllm.platforms.cuda import CudaPlatform

    from vllm_sm70 import attention
    from vllm_sm70.platform import Sm70CudaPlatform

    seen = []
    monkeypatch.setattr(
        CudaPlatform,
        "get_attn_backend_cls",
        classmethod(
            lambda cls, *a: seen.append(a) or "upstream.TritonAttentionBackend"
        ),
    )
    monkeypatch.setattr(
        Sm70CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls, i=0: _cap(7, 0)),
    )
    monkeypatch.setattr(attention, "_ops_available", lambda: True)
    monkeypatch.delenv("VLLM_SM70_FLASH_V100", raising=False)
    return SimpleNamespace(cls=Sm70CudaPlatform, seen=seen, attention=attention)


def test_selects_flash_v100_for_flint_and_qwen(platform):
    for cfg in (_selector(), _selector(head_size=256, has_sink=False)):
        path = platform.cls.get_attn_backend_cls(None, cfg, 16)
        assert path == platform.attention.custom_backend_path()
    assert platform.seen == []


@pytest.mark.parametrize(
    "case",
    ["selected", "kill_switch", "head_size", "dtype", "no_ops", "fp8_kv"],
)
def test_falls_through_to_upstream(platform, monkeypatch, case):
    selected, cfg = None, _selector()
    if case == "selected":  # (a) an explicit --attention-backend wins
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        selected = AttentionBackendEnum.TRITON_ATTN
    elif case == "kill_switch":  # (b)
        monkeypatch.setenv("VLLM_SM70_FLASH_V100", "0")
    elif case == "head_size":  # (c)
        cfg = _selector(head_size=64)
    elif case == "dtype":  # (d)
        cfg = _selector(dtype=torch.bfloat16)
    elif case == "no_ops":  # (e)
        monkeypatch.setattr(platform.attention, "_ops_available", lambda: False)
    elif case == "fp8_kv":  # backend validation
        cfg = _selector(kv_cache_dtype="fp8")
    path = platform.cls.get_attn_backend_cls(selected, cfg, 16)
    assert path == "upstream.TritonAttentionBackend"
    assert platform.seen == [(selected, cfg, 16)]


def test_kill_switch_only_acts_on_zero(platform, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASH_V100", "1")
    assert platform.cls.get_attn_backend_cls(None, _selector(), 16) != (
        "upstream.TritonAttentionBackend"
    )


def test_rejection_reasons():
    from vllm_sm70 import attention

    assert "selected" in attention.flash_v100_rejection("X", _selector())
    assert "head size" in attention.flash_v100_rejection(None, "not-a-config")


# --------------------------------------------------------------- KV layout


def _lbnhc_cache(num_blocks, block_size, h_kv, d, seed=0):
    gen = torch.Generator().manual_seed(seed)
    storage = torch.randn(num_blocks, block_size, h_kv, 2 * d, generator=gen).half()
    return storage.permute(0, 2, 1, 3)  # logical (blocks, H_kv, block_size, 2D)


def test_split_kv_cache_is_a_strided_view():
    from vllm_sm70.attention.flash_v100 import dispatch

    kv = _lbnhc_cache(4, 16, 2, 128)
    k, v = dispatch.split_kv_cache(kv, 128)
    assert k.shape == (4, 16, 2, 128) and k.stride(2) == 256
    assert k.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()
    # Upstream's own split of the same cache gives the same views.
    uk, uv = kv.transpose(1, 2).split(128, dim=-1)
    assert torch.equal(k, uk) and torch.equal(v, uv)


# ------------------------------------------------------------------ metadata


def _common_metadata(query_lens, seq_lens, block_size=16, max_blocks=8):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens).cumsum(0)
    n = int(qsl[-1])
    table = torch.arange(len(query_lens) * max_blocks, dtype=torch.int32).view(
        len(query_lens), max_blocks
    )
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
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )


@pytest.fixture
def builder(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backends import triton_attn

    from vllm_sm70.attention.flash_v100.metadata import FlashAttnV100MetadataBuilder

    monkeypatch.setattr(
        triton_attn, "get_num_attention_heads_from_layers", lambda *a: 16
    )
    model = SimpleNamespace(
        get_num_attention_heads=lambda p: 16,
        get_num_kv_heads=lambda p: 2,
        get_head_size=lambda: 128,
        rswa_window=None,
        max_model_len=4096,
    )
    vllm_config = SimpleNamespace(
        model_config=model,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=[]
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=None,
    )
    spec = SimpleNamespace(block_size=16)
    return FlashAttnV100MetadataBuilder(spec, ["l0"], vllm_config, torch.device("cpu"))


def test_builder_init(builder):
    assert builder.reorder_batch_threshold == 1
    assert builder.supports_draft_decode_metadata_update is False
    assert builder.max_model_len == 4096


def test_build_splits_decodes_and_groups_prefills(builder):
    # Reordered batch: decodes first, then prefill rows of lengths 5, 3, 5.
    cm = _common_metadata([1, 1, 5, 3, 5], [40, 7, 20, 3, 5])
    md = builder.build(0, cm)
    assert (md.num_decodes, md.num_decode_tokens, md.num_prefills) == (2, 2, 3)
    groups = {g.query_len: g for g in md.prefill_groups}
    assert sorted(groups) == [3, 5]
    assert groups[5].rows.tolist() == [2, 4]
    assert groups[5].token_index.tolist() == [2, 3, 4, 5, 6, 10, 11, 12, 13, 14]
    assert groups[3].rows.tolist() == [3]
    assert groups[3].token_index.tolist() == [7, 8, 9]
    # The Triton fields are passed through (fallback stays possible).
    assert md.seq_lens is cm.seq_lens and md.block_table is cm.block_table_tensor
    assert md.max_model_len == 4096 and not md.cudagraph_capture


def test_build_pure_decode_has_no_groups(builder):
    md = builder.build(0, _common_metadata([1, 1, 1], [9, 300, 2]))
    assert (md.num_decodes, md.num_prefills, md.prefill_groups) == (3, 0, ())


def test_build_for_cudagraph_capture_marks_metadata(builder):
    cm = _common_metadata([1, 1], [9, 300])
    md = builder.build_for_cudagraph_capture(cm)
    assert md.cudagraph_capture
    assert md.seq_lens.tolist() == [1, 1]  # upstream's capture fill


# ------------------------------------------------- impl forward on CPU ops


@pytest.fixture
def cpu_ops(monkeypatch):
    from _fv100_cpu_ops import CpuOps

    from vllm_sm70.ops import fv100

    ops = CpuOps()
    monkeypatch.setattr(fv100, "_op", ops)
    fv100.clear_workspaces()
    return ops


def _impl(num_heads, head_size, num_kv_heads, window, sinks):
    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

    return FlashAttnV100Impl(
        num_heads,
        head_size,
        head_size**-0.5,
        num_kv_heads,
        None,
        window,
        "auto",
        sinks=sinks,
    )


def _forward_case(
    builder, impl, *, query_lens, seq_lens, heads, head_size, block_size=16, seed=0
):
    from _fv100_ref import decode_reference, prefill_reference

    num_heads, num_kv_heads = heads
    max_blocks = -(-max(seq_lens) // block_size)
    num_blocks = len(seq_lens) * max_blocks
    kv = _lbnhc_cache(num_blocks, block_size, num_kv_heads, head_size, seed)
    cm = _common_metadata(query_lens, seq_lens, block_size, max_blocks)
    md = builder.build(0, cm)
    gen = torch.Generator().manual_seed(seed + 1)
    n = cm.num_actual_tokens
    query = torch.randn(n, num_heads, head_size, generator=gen).half()
    output = torch.empty_like(query)
    impl.forward(None, query, None, None, kv, md, output)
    k, v = kv.transpose(1, 2).split(head_size, dim=-1)
    window = tuple(impl.sliding_window)
    sinks = impl.sinks
    starts = cm.query_start_loc_cpu.tolist()
    for row, q_len in enumerate(query_lens):
        a, b = starts[row], starts[row + 1]
        lens = cm.seq_lens[row : row + 1]
        table = cm.block_table_tensor[row : row + 1]
        if q_len == 1:
            ref = decode_reference(
                query[a:b],
                k,
                v,
                table,
                lens,
                scale=impl.scale,
                window=window,
                sinks=sinks,
            )
        else:
            ref = prefill_reference(
                query[None, a:b],
                k,
                v,
                table,
                lens,
                scale=impl.scale,
                causal=True,
                window=window,
                sinks=sinks,
            )[0]
        torch.testing.assert_close(output[a:b].float(), ref, atol=2e-3, rtol=2e-3)


def test_forward_flint_mixed_batch(builder, cpu_ops):
    sinks = torch.linspace(-2, 6, 16)
    impl = _impl(16, 128, 2, 513, sinks)
    assert impl._fv100_reason is None
    _forward_case(
        builder,
        impl,
        query_lens=[1, 1, 7, 3, 7],
        seq_lens=[900, 33, 40, 600, 7],
        heads=(16, 2),
        head_size=128,
    )
    names = [c[0] for c in cpu_ops.calls]
    assert names == ["fv100_decode_paged_sinks"] + ["fv100_prefill_paged_sinks"] * 2
    decode = cpu_ops.calls[0][1]
    assert decode["rows"] == 2 and decode["window"] == (512, 0)
    assert decode["sinks"] is sinks
    assert {c[1]["query_len"] for c in cpu_ops.calls[1:]} == {3, 7}


def test_forward_qwen_d256_uses_xqa(builder, cpu_ops):
    impl = _impl(12, 256, 2, None, None)
    _forward_case(
        builder,
        impl,
        query_lens=[1, 1, 1, 9],
        seq_lens=[1000, 5, 129, 64],
        heads=(12, 2),
        head_size=256,
    )
    names = [c[0] for c in cpu_ops.calls]
    assert names == ["fv100_decode_paged_xqa", "fv100_prefill_paged"]


def test_forward_gqa_outside_xqa_takes_scalar_decode(builder, cpu_ops):
    impl = _impl(10, 256, 2, None, None)  # GQA 5: no XQA instantiation
    _forward_case(
        builder,
        impl,
        query_lens=[1, 1],
        seq_lens=[300, 17],
        heads=(10, 2),
        head_size=256,
    )
    assert [c[0] for c in cpu_ops.calls] == ["fv100_decode_paged"]


def test_forward_falls_back_to_triton(builder, cpu_ops, monkeypatch):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    seen = []
    monkeypatch.setattr(
        TritonAttentionImpl, "forward", lambda self, *a, **k: seen.append(a) or "tri"
    )
    impl = _impl(16, 128, 2, None, None)
    md = builder.build(0, _common_metadata([1, 1], [9, 30]))
    md.causal = torch.tensor([True, True])
    q = torch.zeros(2, 16, 128).half()
    kv = _lbnhc_cache(16, 16, 2, 128)
    assert impl.forward(None, q, None, None, kv, md, torch.empty_like(q)) == "tri"
    assert len(seen) == 1 and cpu_ops.calls == []
    softcap = _impl(16, 128, 2, None, None)
    softcap.logits_soft_cap = 30.0
    assert softcap._unsupported_reason() == "logits soft cap"


def test_forward_profiling_run_zeroes_output(builder):
    impl = _impl(16, 128, 2, None, None)
    out = torch.ones(3, 16, 128).half()
    impl.forward(None, out, None, None, None, None, out)
    assert out.abs().sum() == 0


def test_forward_checks_kv_views_once(builder, cpu_ops, monkeypatch):
    from vllm_sm70.ops import fv100

    calls = []
    real = fv100.check_kv_cache_views
    monkeypatch.setattr(
        fv100, "check_kv_cache_views", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    impl = _impl(16, 128, 2, None, None)
    md = builder.build(0, _common_metadata([1], [20]))
    q = torch.zeros(1, 16, 128).half()
    kv = _lbnhc_cache(8, 16, 2, 128)
    for _ in range(3):
        impl.forward(None, q, None, None, kv, md, torch.empty_like(q))
    assert calls == [1]
