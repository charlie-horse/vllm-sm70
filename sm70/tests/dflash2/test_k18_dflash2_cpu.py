# SPDX-License-Identifier: Apache-2.0
"""K18 CPU tests: schemas, fakes, shape admission, U1 selection, kill switch.

No GPU or built _sm70_C needed: the op schemas are read from
sm70/csrc/dflash2/*.cu and their fakes run on meta tensors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from _dflash2_helpers import dflash2_schemas, qwen_vllm_config  # noqa: E402
from torch import nn  # noqa: E402
from vllm_sm70 import config as sm70_cfg  # noqa: E402
from vllm_sm70 import envs  # noqa: E402
from vllm_sm70.ops import dflash2  # noqa: E402

KILL = "VLLM_SM70_DFLASH2_FP16"
QWEN38 = qwen_vllm_config()
QWEN36 = qwen_vllm_config(arch="Qwen3_5MoeForConditionalGeneration")


@pytest.fixture(autouse=True)
def _qwen38_engine(monkeypatch):
    """Default switch (unset = on), and the current engine is the qualified
    Qwen3.8 DFlash2 contract. Tests that need another engine patch
    dflash2._current_vllm_config themselves."""
    monkeypatch.delenv(KILL, raising=False)
    monkeypatch.delenv("VLLM_SM70_DFLASH2_VERIFIER", raising=False)
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN38)
    sm70_cfg.clear_cache()
    dflash2.reset_selection_cache()
    yield
    dflash2.reset_selection_cache()
    sm70_cfg.clear_cache()


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
def test_switch_registered_kill_switch(monkeypatch):
    """K18 is a kill switch since E3: unset or anything but "0" is on."""
    assert KILL in envs.environment_variables()
    assert KILL in envs._SECTIONS["ops"]
    assert envs.VLLM_SM70_DFLASH2_FP16 is True  # default ON (unset)
    for value in ("", "1", "true", "on", " 0", "00", "false"):
        monkeypatch.setenv(KILL, value)
        assert envs.VLLM_SM70_DFLASH2_FP16 is True, value
    monkeypatch.setenv(KILL, "0")
    assert envs.VLLM_SM70_DFLASH2_FP16 is False


def test_k18_wanted_only_for_the_qwen38_contract(monkeypatch):
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    assert dflash2.k18_wanted(QWEN38)
    assert not dflash2.k18_wanted(QWEN36)  # MoE arch: not qualified
    assert not dflash2.k18_wanted(qwen_vllm_config(k=5))
    assert not dflash2.k18_wanted(None)
    # Independent of the verifier switch: VERIFIER=0 is "K18 alone" (E3 B).
    monkeypatch.setenv("VLLM_SM70_DFLASH2_VERIFIER", "0")
    sm70_cfg.clear_cache()
    assert dflash2.k18_wanted(QWEN38)
    monkeypatch.setenv(KILL, "0")
    assert not dflash2.k18_wanted(QWEN38)
    monkeypatch.delenv(KILL)
    monkeypatch.setattr(dflash2, "gemm_available", lambda: False)
    assert not dflash2.k18_wanted(QWEN38)  # ops not built


def test_unqualified_engine_selects_and_marks_nothing(monkeypatch):
    """Qwen3.6 (and any engine outside the contract) keeps upstream dispatch
    exactly, with K18's switch at its default."""
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN36)
    assert dflash2.enabled()
    assert dflash2.select_unquantized_gemm("auto") is None
    layer = _fake_decoder_layer(_linear_method())
    assert dflash2.mark_draft_linears(layer, vllm_config=QWEN36) == []


def test_selection_waits_for_an_engine_config(monkeypatch):
    """A call with no current VllmConfig returns None and caches nothing; the
    first call under a config decides."""
    pytest.importorskip("vllm")
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: None)
    assert dflash2.select_unquantized_gemm("auto") is None
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN38)
    assert dflash2.select_unquantized_gemm("auto") is dflash2.dflash2_unquantized_gemm
    # Decided: later calls without a config (per-forward lm_head) keep it.
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: None)
    assert dflash2.select_unquantized_gemm("auto") is dflash2.dflash2_unquantized_gemm


def test_each_engine_decides_for_itself(monkeypatch):
    """Cached per engine config, not per process: a second engine in the same
    process (in-process TP1 tests) re-decides; config-less calls follow the
    most recent engine."""
    pytest.importorskip("vllm")
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    dispatcher = dflash2.dflash2_unquantized_gemm
    assert dflash2.select_unquantized_gemm("auto") is dispatcher  # Qwen3.8
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN36)
    assert dflash2.select_unquantized_gemm("auto") is None
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: None)
    assert dflash2.select_unquantized_gemm("auto") is None  # latest: Qwen3.6
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN38)
    assert dflash2.select_unquantized_gemm("auto") is dispatcher


def test_selection_logged_once_per_switch(monkeypatch):
    """The "prepared" lines come from marking/packing; only this line shows
    the U1 dispatcher was picked (Quinn, #108). Logged once per switch to the
    dispatcher, not per config object: on c4130 the draft is built under its
    own qualified VllmConfig, and per-key logging printed it twice per rank
    (Bea's #108 smoke)."""
    pytest.importorskip("vllm")
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    lines = []
    monkeypatch.setattr(
        dflash2.logger, "info", lambda msg, *a, **k: lines.append(msg % a)
    )
    for _ in range(5):
        dflash2.select_unquantized_gemm("auto")
    draft_cfg = qwen_vllm_config()  # a second, equally qualified config object
    assert draft_cfg is not QWEN38
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: draft_cfg)
    for _ in range(5):
        dflash2.select_unquantized_gemm("auto")
    assert len(lines) == 1
    assert "K18 DFlash2 draft GEMM dispatcher selected" in lines[0]
    assert "linear_backend=auto" in lines[0]
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN36)
    dflash2.select_unquantized_gemm("auto")
    assert len(lines) == 1  # not selected: no line
    # A later qualified engine switches back to the dispatcher: logged again.
    monkeypatch.setattr(dflash2, "_current_vllm_config", lambda: QWEN38)
    dflash2.select_unquantized_gemm("auto")
    assert len(lines) == 2


def test_killed_packs_nothing_even_when_forced(monkeypatch):
    monkeypatch.setenv(KILL, "0")
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    assert not dflash2.enabled()
    layer = nn.Linear(2048, 512, bias=False).half()
    monkeypatch.setattr(dflash2, "_is_sm70", lambda device: True)
    assert not dflash2.mark_and_prepare(layer)  # even a forced mark packs nothing
    assert not hasattr(layer, dflash2.PACKED_ATTR)


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


def test_selection_is_cached(monkeypatch):
    """UP calls dispatch_unquantized_gemm() per forward (lm_head, router gate):
    the platform answer is computed once per linear_backend."""
    pytest.importorskip("vllm")
    calls = []

    def counting():
        calls.append(1)
        return True

    monkeypatch.setattr(dflash2, "gemm_available", counting)
    for _ in range(100):
        assert (
            dflash2.select_unquantized_gemm("auto") is dflash2.dflash2_unquantized_gemm
        )
    assert len(calls) == 1
    for _ in range(10):
        dflash2.select_unquantized_gemm("flashinfer_cutedsl")
    assert len(calls) == 2
    dflash2.reset_selection_cache()
    dflash2.select_unquantized_gemm("auto")
    assert len(calls) == 3


def test_per_layer_route_decided_once(monkeypatch):
    """After packing, a GEMM call reads one cached route; no admission work."""
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.setattr(dflash2, "_is_sm70", lambda device: True)
    layer = nn.Linear(2048, 512, bias=False).half()
    assert dflash2.mark_and_prepare(layer)
    assert getattr(layer, dflash2.ROUTE_ATTR) == (512, 2048, 16, 16)
    counts = {"enabled": 0, "sm70": 0, "valid": 0}

    def count(key, ret):
        def fn(*a, **k):
            counts[key] += 1
            return ret

        return fn

    monkeypatch.setattr(dflash2, "enabled", count("enabled", True))
    monkeypatch.setattr(dflash2, "_is_sm70", count("sm70", True))
    monkeypatch.setattr(dflash2, "geometry_valid", count("valid", True))
    plain = nn.Linear(2048, 512, bias=False).half()
    x = torch.randn(16, 2048).half()  # M > 8: F.linear on the CPU
    for _ in range(20):
        dflash2.dflash2_unquantized_gemm(layer, x, layer.weight, None)
        dflash2.dflash2_unquantized_gemm(plain, x, plain.weight, None)
    assert counts == {"enabled": 0, "sm70": 0, "valid": 0}


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
    setattr(layer, mod.ROUTE_ATTR, (n, k, tile, warps))
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


def test_mark_draft_linears_flags_six_projections(monkeypatch):
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    layer = _fake_decoder_layer(_linear_method())
    marked = dflash2.mark_draft_linears(layer, vllm_config=QWEN38)  # default on
    assert len(marked) == 6
    assert all(getattr(m, dflash2.MARK_ATTR) for m in marked)


def test_mark_skips_quantized_unqualified_and_kill_switch(monkeypatch):
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    _linear_method()
    layer = _fake_decoder_layer(object())
    assert dflash2.mark_draft_linears(layer, vllm_config=QWEN38) == []
    layer = _fake_decoder_layer(_linear_method())
    assert dflash2.mark_draft_linears(layer, vllm_config=QWEN36) == []
    monkeypatch.setenv(KILL, "0")
    layer = _fake_decoder_layer(_linear_method())
    assert dflash2.mark_draft_linears(layer, vllm_config=QWEN38) == []
    monkeypatch.delenv(KILL)
    monkeypatch.setattr(dflash2, "gemm_available", lambda: False)
    layer = _fake_decoder_layer(_linear_method())
    assert dflash2.mark_draft_linears(layer, vllm_config=QWEN38) == []


# ------------------------------------------------- reload / no autograd
def test_reload_repacks_in_place_without_autograd(monkeypatch):
    """A second prepare refills the same buffer (no 2nd copy, never stale)."""
    monkeypatch.setattr(dflash2, "enabled", lambda: True)
    monkeypatch.setattr(dflash2, "_is_sm70", lambda device: True)
    calls = []
    real_pack = dflash2.pack_weight

    def spy_pack(weight, tile, out=None):
        calls.append(out)
        return real_pack(weight, tile, out=out)

    monkeypatch.setattr(dflash2, "pack_weight", spy_pack)
    torch.manual_seed(0)
    layer = nn.Linear(2048, 512, bias=False).half()  # (512, 2048): admitted
    assert layer.weight.requires_grad
    assert dflash2.mark_and_prepare(layer)
    packed = getattr(layer, dflash2.PACKED_ATTR)
    ptr = packed.data_ptr()
    assert calls == [None]  # first call allocates
    assert torch.equal(packed, real_pack(layer.weight.detach(), 16))
    # no_grad, not inference_mode: no graph, and still writable in place.
    assert not packed.requires_grad and packed.grad_fn is None
    assert not packed.is_inference()
    old = packed.clone()

    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))  # "reload"
    assert dflash2.prepare_marked(layer) == 1
    again = getattr(layer, dflash2.PACKED_ATTR)
    assert again is packed and again.data_ptr() == ptr
    assert len(calls) == 2 and calls[1] is packed  # repacked into it
    assert torch.equal(again, real_pack(layer.weight.detach(), 16))
    assert not torch.equal(again, old)
    assert not again.requires_grad and again.grad_fn is None
    assert list(layer.state_dict()) == ["weight"]


# ------------------------------------------- M7 hook (regression guard)
def _m7_classes(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models import qwen3_dflash2

    from vllm_sm70.models.qwen import dflash_sm70 as m7

    method = object.__new__(UnquantizedLinearMethod)

    shapes = dflash2.DRAFT_SHAPES_TP2["qwen3.6-35b-a3b-dflash2"]

    def proj(kind=None):
        m = nn.Module()
        m.quant_method = method
        if kind is not None and WEIGHTS["on"]:
            n, k = shapes[kind]
            m.weight = nn.Parameter(
                torch.randn(n, k, dtype=torch.float16) * 0.01, requires_grad=False
            )
        return m

    def fake_base_init(
        self,
        vllm_config,
        *,
        config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = proj("qkv_proj")
        self.self_attn.o_proj = proj("o_proj")
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = proj("gate_up_proj")
        self.mlp.down_proj = proj("down_proj")
        self.attention_conv = nn.Module()
        self.attention_conv.kernel_projection = proj("kernel_projection")
        self.mlp_conv = nn.Module()
        self.mlp_conv.kernel_projection = proj("kernel_projection")
        # an unrelated linear must stay unmarked
        self.other = proj()

    monkeypatch.setattr(
        qwen3_dflash2.DFlash2Qwen3DecoderLayer, "__init__", fake_base_init
    )
    monkeypatch.setattr(m7, "emulation_enabled", lambda config, dtype: True)
    monkeypatch.setattr(m7, "_CLASSES", None)
    return m7


WEIGHTS = {"on": False}


def _build_m7_layer(m7, weights=False, vllm_config=QWEN38):
    """M7's draft decoder layer under ``vllm_config`` (default: the qualified
    Qwen3.8 contract, so K18 marks at its default). The fake weights use the
    Qwen3.6 TP2 shard shapes only to keep CPU tensors small; K18 admits by
    shard shape, independently of which contract enabled it."""
    from types import SimpleNamespace

    WEIGHTS["on"] = weights

    cls = m7.DFlash2SM70Qwen3DecoderLayer
    config = SimpleNamespace(hidden_size=64, rms_norm_eps=1e-6)
    return cls(vllm_config, config=config, layer_idx=0)


def _marked(layer):
    return {
        name
        for name, m in layer.named_modules()
        if getattr(m, dflash2.MARK_ATTR, False)
    }


def test_m7_layer_marks_exactly_six_projections(monkeypatch):
    m7 = _m7_classes(monkeypatch)
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    layer = _build_m7_layer(m7)
    assert layer.use_sm70_bf16_emulation
    assert _marked(layer) == {
        "self_attn.qkv_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
        "attention_conv.kernel_projection",
        "mlp_conv.kernel_projection",
    }


@pytest.mark.parametrize(
    "engine,value",
    [("qwen3.6", None), ("qwen3.8", "0")],
    ids=["unqualified_default", "qwen38_killed"],
)
def test_m7_unqualified_or_killed_marks_and_packs_nothing(monkeypatch, engine, value):
    m7 = _m7_classes(monkeypatch)
    if value is not None:
        monkeypatch.setenv(KILL, value)
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.setattr(dflash2, "_is_sm70", lambda device: True)
    cfg = QWEN36 if engine == "qwen3.6" else QWEN38
    layer = _build_m7_layer(m7, weights=True, vllm_config=cfg)
    assert _marked(layer) == set()
    assert dflash2.prepare_marked(layer) == 0
    assert not any(hasattr(m, dflash2.PACKED_ATTR) for m in layer.modules())


def test_m7_qwen38_default_marks_and_packs_six(monkeypatch):
    m7 = _m7_classes(monkeypatch)  # autouse fixture: switch unset (on)
    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.setattr(dflash2, "_is_sm70", lambda device: True)
    layer = _build_m7_layer(m7, weights=True)
    assert len(_marked(layer)) == 6
    assert dflash2.prepare_marked(layer) == 6
    packed = {n for n, m in layer.named_modules() if hasattr(m, dflash2.PACKED_ATTR)}
    assert packed == _marked(layer)


def test_m7_load_weights_calls_prepare_marked(monkeypatch):
    m7 = _m7_classes(monkeypatch)
    from vllm.model_executor.models import qwen3_dflash

    order = []

    def fake_super_load(self, weights):
        order.append("load")
        return {"w"}

    monkeypatch.setattr(
        qwen3_dflash.DFlashQwen3ForCausalLM, "load_weights", fake_super_load
    )
    monkeypatch.setattr(
        dflash2, "prepare_marked", lambda model: order.append(("prepare", model))
    )
    cls = m7.DFlash2SM70Qwen3ForCausalLM
    model = object.__new__(cls)
    assert cls.load_weights(model, iter(())) == {"w"}
    assert order == ["load", ("prepare", model)]
