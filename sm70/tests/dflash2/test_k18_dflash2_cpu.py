# SPDX-License-Identifier: Apache-2.0
"""K18 CPU tests: schemas, fakes, shape admission, U1 selection, kill switch.

No GPU or built _sm70_C needed: the op schemas are read from
sm70/csrc/dflash2/*.cu and their fakes run on meta tensors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from _dflash2_helpers import dflash2_schemas  # noqa: E402
from torch import nn  # noqa: E402
from vllm_sm70 import envs  # noqa: E402
from vllm_sm70.ops import dflash2  # noqa: E402

KILL = "VLLM_SM70_DFLASH2_FP16"


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.delenv(KILL, raising=False)


# ------------------------------------------------------------------ schemas
def test_schemas_from_sources():
    schemas = dflash2_schemas()
    assert set(schemas) == set(dflash2.OPS)
    assert schemas["sm70_dflash2_fp16_m8_out"] == (
        "sm70_dflash2_fp16_m8_out(Tensor(a!) output, Tensor input, Tensor "
        "packed, int tile, int warps) -> ()"
    )
    assert schemas["sm70_dflash2_fp16_dispatch_out"] == (
        "sm70_dflash2_fp16_dispatch_out(Tensor(a!) output, Tensor input, Tensor "
        "packed, Tensor weight, int tile, int warps) -> ()"
    )
    assert schemas["sm70_compact_half_topk_out"] == (
        "sm70_compact_half_topk_out(Tensor x, Tensor(a!) partial, Tensor(b!) "
        "values, Tensor(c!) ids, int k) -> ()"
    )


def test_sources_cmake_lists_both_files():
    import pathlib

    area = pathlib.Path(__file__).resolve().parents[2] / "csrc" / "dflash2"
    text = (area / "sources.cmake").read_text()
    for name in ("dflash2_fp16_m8.cu", "compact_half_topk.cu"):
        assert name in text and (area / name).exists()
    top = (area.parents[1] / "CMakeLists.txt").read_text()
    assert "csrc/dflash2/sources.cmake" in top


# ---------------------------------------------------------------- env / kill
def test_kill_switch_registered_default_on(monkeypatch):
    assert KILL in envs.environment_variables()
    assert KILL in envs._SECTIONS["ops"]
    assert envs.VLLM_SM70_DFLASH2_FP16 is True
    monkeypatch.setenv(KILL, "0")
    assert envs.VLLM_SM70_DFLASH2_FP16 is False
    monkeypatch.setenv(KILL, "1")
    assert envs.VLLM_SM70_DFLASH2_FP16 is True


# ------------------------------------------------------------ shape guards
def _draft_shards(cfg: dict, tp: int) -> dict[str, tuple[int, int]]:
    """(N, K) of each projection's local shard, from a draft's HF config."""
    h, d = cfg["hidden_size"], cfg["head_dim"]
    q, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    inter = cfg["intermediate_size"]
    groups = h // cfg["conv_group_size"]
    return {
        "qkv_proj": ((q // tp + 2 * (kv // tp)) * d, h),
        "o_proj": (h, q // tp * d),
        "gate_up_proj": (2 * inter // tp, h),
        "down_proj": (h, inter // tp),
        # ReplicatedLinear(hidden, 2 * taps * groups)
        "kernel_projection": (2 * cfg["conv_kernel_size"] * groups, h),
    }


# Public config.json values of the incoai/*-DFlash2 drafts the host serves.
DRAFT_CONFIGS = {
    "qwen3.8-27b-dflash2": dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        conv_kernel_size=2,
        conv_group_size=16,
    ),
    "qwen3.6-35b-a3b-dflash2": dict(
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        conv_kernel_size=2,
        conv_group_size=16,
    ),
}


@pytest.mark.parametrize("draft", sorted(DRAFT_CONFIGS))
def test_tp2_shards_admitted(draft):
    shards = _draft_shards(DRAFT_CONFIGS[draft], tp=2)
    assert shards == dflash2.DRAFT_SHAPES_TP2[draft]
    for name, (n, k) in shards.items():
        assert (n, k) in dflash2.GEOMETRY, f"{draft} {name} N{n}/K{k} not admitted"
        tile, warps = dflash2.GEOMETRY[(n, k)]
        assert dflash2.geometry_valid(n, k, tile, warps), (draft, name, n, k)


def test_admitted_set_is_exactly_the_tp2_drafts():
    wanted = {s for d in dflash2.DRAFT_SHAPES_TP2.values() for s in d.values()}
    assert set(dflash2.GEOMETRY) == wanted


def test_fork_tp4_only_shapes_dropped():
    # FORK sm70_dflash2_fp16.py:14-20; only the replicated (1280, 5120) stays.
    for shape in [(1536, 5120), (5120, 1024), (8704, 5120), (5120, 4352)]:
        assert shape not in dflash2.GEOMETRY
    assert dflash2.GEOMETRY[(1280, 5120)] == (16, 16)


@pytest.mark.parametrize(
    "n,k,tile,warps,ok",
    [
        (1280, 5120, 16, 16, True),
        (1280, 5120, 16, 8, False),  # no <8, 16> instantiation
        (1296, 5120, 32, 8, False),  # N % tile
        (5120, 1000, 32, 8, False),  # K % (16 * split)
        (2048, 2048, 16, 16, True),  # K = 4 * 512
        (2048, 2304, 16, 16, False),
    ],
)
def test_geometry_valid(n, k, tile, warps, ok):
    assert dflash2.geometry_valid(n, k, tile, warps) is ok


@pytest.mark.parametrize("tile", [16, 32])
def test_pack_weight_layout(tile):
    # packed[b, g, s, :] = W[b*tile + col(s), 16g:16g+16]: a slot permutation
    # inside each tile; every row appears exactly once (FORK :45-56).
    n, k = 2 * tile, 64
    weight = torch.arange(n * k, dtype=torch.float32).reshape(n, k)
    packed = dflash2.pack_weight(weight, tile)
    assert packed.shape == (n // tile, k // 16, tile, 16)
    slots = torch.arange(tile)
    if tile == 32:
        cols = (slots & 3) | ((slots & 12) << 1) | ((slots & 16) >> 2)
    else:
        cols = (slots & 3) | ((slots & 4) << 1) | ((slots & 8) >> 1)
    assert sorted(cols.tolist()) == list(range(tile))
    for b in range(n // tile):
        for g in range(k // 16):
            for s in range(tile):
                row = b * tile + int(cols[s])
                assert torch.equal(packed[b, g, s], weight[row, 16 * g : 16 * g + 16])


# ------------------------------------------------------------ U1 selection
def test_select_off_without_ops(monkeypatch):
    monkeypatch.setattr(dflash2, "gemm_available", lambda: False)
    assert dflash2.select_unquantized_gemm("auto") is None


def test_select_on_with_ops(monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    assert dflash2.select_unquantized_gemm("auto") is dflash2.dflash2_unquantized_gemm
    # An explicit FlashInfer BF16 backend keeps upstream's dispatch.
    assert dflash2.select_unquantized_gemm("flashinfer_cutedsl") is None


def test_select_kill_switch(monkeypatch):
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.setenv(KILL, "0")
    assert dflash2.select_unquantized_gemm("auto") is None


def test_unmarked_layer_falls_through_to_f_linear():
    layer = nn.Linear(64, 32, bias=True)
    x = torch.randn(8, 64)
    out = dflash2.dflash2_unquantized_gemm(layer, x, layer.weight, layer.bias)
    assert torch.equal(out, torch.nn.functional.linear(x, layer.weight, layer.bias))


def test_prepare_refuses_cpu_and_unmarked(monkeypatch):
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    layer = nn.Linear(5120, 1280, bias=False).half()
    assert not dflash2.prepare_dflash2_fp16_m8(layer)  # not marked
    assert not dflash2.mark_and_prepare(layer)  # CPU weight
    assert not hasattr(layer, dflash2.PACKED_ATTR)


# --------------------------------------------- apply / fakes on meta tensors
def _meta_layer(mod, n=1280, k=5120, tile=16, warps=16):
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.empty(n, k, dtype=torch.float16, device="meta"), requires_grad=False
    )
    layer.register_buffer(
        mod.PACKED_ATTR,
        torch.empty(n // tile, k // 16, tile, 16, dtype=torch.float16, device="meta"),
        persistent=False,
    )
    setattr(layer, mod.GEOMETRY_ATTR, (tile, warps))
    return layer


@pytest.mark.parametrize("m", [1, 3, 7, 8])
def test_apply_routes_m_le_8(dflash2_stub, m):
    mod = dflash2_stub
    assert all(mod.has_op(name) for name in mod.OPS)
    layer = _meta_layer(mod)
    x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
    out = mod.apply_dflash2_fp16_m8(layer, x, None)
    assert out is not None and out.shape == (m, 1280) and out.dtype == torch.float16
    via_gemm = mod.dflash2_unquantized_gemm(layer, x, layer.weight, None)
    assert via_gemm.shape == (m, 1280)


def test_apply_rejects(dflash2_stub):
    mod = dflash2_stub
    layer = _meta_layer(mod)

    def x(*shape, dtype=torch.float16):
        return torch.empty(*shape, dtype=dtype, device="meta")

    assert mod.apply_dflash2_fp16_m8(layer, x(0, 5120), None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x(9, 5120), None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x(32, 5120), None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x(1, 8, 5120), None) is None
    assert mod.apply_dflash2_fp16_m8(layer, x(8, 4096), None) is None
    assert (
        mod.apply_dflash2_fp16_m8(layer, x(8, 5120, dtype=torch.float32), None) is None
    )
    assert mod.apply_dflash2_fp16_m8(layer, x(5120, 8).t(), None) is None
    bias = torch.empty(1280, dtype=torch.float16, device="meta")
    assert mod.apply_dflash2_fp16_m8(layer, x(8, 5120), bias) is None
    unpacked = nn.Linear(5120, 1280, bias=False, device="meta", dtype=torch.float16)
    assert mod.apply_dflash2_fp16_m8(unpacked, x(8, 5120), None) is None


def test_topk_fake_runs_on_meta(dflash2_stub):
    mod = dflash2_stub
    logits = torch.empty(8, 62080, dtype=torch.float16, device="meta")
    partial = torch.empty(8, 61 * 64, dtype=torch.uint32, device="meta")
    values = torch.empty(8, 64, dtype=torch.float32, device="meta")
    ids = torch.empty(8, 64, dtype=torch.int64, device="meta")
    torch.ops._sm70_C.sm70_compact_half_topk_out(logits, partial, values, ids, 64)
    # The Python guard never qualifies non-CUDA or FP32 inputs.
    assert mod.compact_half_topk(torch.randn(8, 1024).half()) is None
    assert mod.compact_half_topk(torch.randn(8, 1024)) is None


# ---------------------------------------------------- marking (M7 layers)
def _linear_method():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    return object.__new__(UnquantizedLinearMethod)


def _fake_decoder_layer(quant_method):
    def proj():
        m = nn.Module()
        m.quant_method = quant_method
        return m

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = proj()
    layer.self_attn.o_proj = proj()
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = proj()
    layer.mlp.down_proj = proj()
    layer.attention_conv = nn.Module()
    layer.attention_conv.kernel_projection = proj()
    layer.mlp_conv = nn.Module()
    layer.mlp_conv.kernel_projection = proj()
    return layer


def test_mark_draft_linears_flags_six_projections():
    layer = _fake_decoder_layer(_linear_method())
    marked = dflash2.mark_draft_linears(layer)
    assert len(marked) == 6
    assert all(getattr(m, dflash2.MARK_ATTR) for m in marked)


def test_mark_skips_quantized_and_kill_switch(monkeypatch):
    _linear_method()
    assert dflash2.mark_draft_linears(_fake_decoder_layer(object())) == []
    monkeypatch.setenv(KILL, "0")
    assert dflash2.mark_draft_linears(_fake_decoder_layer(_linear_method())) == []


def test_m7_marks_and_packs():
    """M7's draft classes call into K18 (constructor marks, load_weights packs)."""
    import inspect

    from vllm_sm70.models.qwen import dflash_sm70

    src = inspect.getsource(dflash_sm70._build_classes)
    assert "dflash2.mark_draft_linears(self)" in src
    assert "dflash2.prepare_marked(self)" in src
