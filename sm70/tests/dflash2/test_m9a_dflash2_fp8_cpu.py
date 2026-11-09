# SPDX-License-Identifier: Apache-2.0
"""M9a CPU tests: DFlash2 draft channel-FP8 M8 route behind U1/K18.

No GPU or built _sm70_C: K09's two ops are replaced by CPU references
(prepare keeps the E4M3 weight and scales; the GEMM dequantizes and
multiplies in FP32), and the SM70 device check is patched. What this
covers: row admission, split choice, the quantizer, layer admission and the
kill switch, route order in the U1 dispatcher, FP8 replacing the FP16
packing, the verifier gate M7 uses, and the op's fake under FakeTensorMode.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from vllm_sm70.ops import dflash2, dflash2_fp8  # noqa: E402

VERIFIER = "VLLM_SM70_DFLASH2_VERIFIER"
K18 = "VLLM_SM70_DFLASH2_FP16"

# DFlash2 draft shard shapes (N, K) at TP2 (ops/dflash2.py DRAFT_SHAPES_TP2).
TP2_SHAPES = [
    shape for model in dflash2.DRAFT_SHAPES_TP2.values() for shape in model.values()
]


# --------------------------------------------------------------- references
def _ref_prepare(quantized: torch.Tensor, scales: torch.Tensor):
    # Real op: [N, K] e4m3 + [N, 1] fp32 -> codes [K, N] u8, scales [1, N] fp16.
    return [quantized.clone(), scales.clone()]


def _ref_gemm(out, x, codes, scales, split_k, chains, fast, prefetch):
    assert chains == dflash2_fp8.ACCUMULATOR_CHAINS and fast and not prefetch
    assert split_k in (12, 16) and (x.shape[1] // 16) % split_k == 0
    weight = codes.float() * scales.float()
    out.copy_((x.float() @ weight.t()).to(out.dtype))


@pytest.fixture
def cpu_qpn8(monkeypatch):
    """K09 ops as CPU references; the CPU counts as SM70; switches on."""
    calls = {"gemm": 0}

    def gemm(*args):
        calls["gemm"] += 1
        _ref_gemm(*args)

    monkeypatch.setattr(dflash2_fp8.qpn8, "fp8_qpn8_prepare_sm70", _ref_prepare)
    monkeypatch.setattr(dflash2_fp8.qpn8, "fp8_qpn8_gemm_sm70_out", gemm)
    monkeypatch.setattr(dflash2_fp8, "ops_available", lambda: True)
    monkeypatch.setattr(dflash2_fp8, "_is_sm70", lambda device: True)
    monkeypatch.delenv(VERIFIER, raising=False)
    monkeypatch.setenv(K18, "1")
    return calls


def _linear(n: int, k: int, *, seed: int = 0, bias: bool = False) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    layer = nn.Module()
    weight = (torch.randn(n, k, generator=g) / k**0.5).half()
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.bias = (
        nn.Parameter(torch.zeros(n, dtype=torch.float16), requires_grad=False)
        if bias
        else None
    )
    layer.prefix = f"draft.N{n}K{k}"
    setattr(layer, dflash2_fp8.FP8_MARK_ATTR, True)
    return layer


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


# ------------------------------------------------------------- admission
@pytest.mark.parametrize("rows", [8, 16, 24, 32])
def test_fp8_rows_admits_whole_requests(rows):
    assert dflash2_fp8.fp8_rows(rows)


@pytest.mark.parametrize("rows", [0, 1, 7, 9, 15, 33, 40, 64])
def test_fp8_rows_rejects_other_m(rows):
    assert not dflash2_fp8.fp8_rows(rows)


@pytest.mark.parametrize("shape", TP2_SHAPES, ids=lambda s: f"N{s[0]}K{s[1]}")
def test_every_tp2_draft_shard_gets_split_16(shape):
    n, k = shape
    assert n % 32 == 0
    assert dflash2_fp8.split_for(k) == 16


def test_split_falls_back_to_12_then_none():
    assert dflash2_fp8.split_for(192) == 12  # K/16 = 12
    assert dflash2_fp8.split_for(16 * 20) is None  # 20: neither 12 nor 16
    assert dflash2_fp8.split_for(100) is None  # K % 16
    assert dflash2_fp8.split_for(0) is None


# ------------------------------------------------------------- quantizer
def test_quantizer_scales_and_zero_rows():
    w = torch.tensor(
        [[1.0, -2.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [1e-12, 0.0, 0.0, 0.0]],
        dtype=torch.float16,
    )
    q, s = dflash2_fp8.channel_fp8_quantize(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
    assert s.dtype == torch.float32 and s.shape == (3, 1)
    assert s[0, 0] == pytest.approx(2.0 / 448.0)
    assert float(q[0].float().abs().max()) == 448.0  # amax -> E4M3 max
    assert s[1, 0] == 1.0 and torch.all(q[1].float() == 0)  # zero channel
    # fp16(1e-12) underflows to 0 -> zero channel; the clamp floor still holds
    assert float(s.min()) >= dflash2_fp8.MIN_SCALE


def test_quantizer_round_trip_error_is_e4m3_sized():
    w = (torch.randn(256, 512, generator=torch.Generator().manual_seed(1))).half()
    q, s = dflash2_fp8.channel_fp8_quantize(w)
    deq = q.float() * s
    # E4M3 keeps 3 mantissa bits: per-element relative error <= 2**-4.
    assert _rel_l2(deq, w) < 0.04


# ------------------------------------------------------- layer admission
def test_admission_reasons(cpu_qpn8, monkeypatch):
    layer = _linear(64, 256)
    assert dflash2_fp8.admission(layer) is None

    unflagged = _linear(64, 256)
    setattr(unflagged, dflash2_fp8.FP8_MARK_ATTR, False)
    assert dflash2_fp8.admission(unflagged) == "not flagged"
    assert "bias" in dflash2_fp8.admission(_linear(64, 256, bias=True))
    assert "multiple of 32" in dflash2_fp8.admission(_linear(48, 256))
    assert "split" in dflash2_fp8.admission(_linear(64, 16 * 20))

    monkeypatch.setattr(dflash2_fp8, "ops_available", lambda: False)
    assert "K09" in dflash2_fp8.admission(layer)


def test_kill_switch_blocks_prepare(cpu_qpn8, monkeypatch):
    monkeypatch.setenv(VERIFIER, "0")
    layer = _linear(64, 256)
    assert dflash2_fp8.admission(layer) == f"{VERIFIER}=0"
    assert not dflash2_fp8.prepare_dflash2_fp8_m8(layer)
    assert getattr(layer, dflash2_fp8.ROUTE_ATTR, None) is None
    assert dflash2_fp8.apply_dflash2_fp8_m8(layer, _x(8, 256), None) is None


@pytest.mark.parametrize("value", ["1", "yes", ""])
def test_only_zero_kills(cpu_qpn8, monkeypatch, value):
    monkeypatch.setenv(VERIFIER, value)
    assert dflash2_fp8.prepare_dflash2_fp8_m8(_linear(64, 256))


# ------------------------------------------------------ route + dispatcher
def _x(rows: int, k: int, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, k, generator=g).half()


@pytest.mark.parametrize("rows", [8, 16, 24, 32])
def test_prepared_layer_runs_fp8_for_whole_requests(cpu_qpn8, rows):
    layer = _linear(128, 512)
    assert dflash2_fp8.prepare_dflash2_fp8_m8(layer)
    assert getattr(layer, dflash2_fp8.ROUTE_ATTR) == (128, 512, 16)
    x = _x(rows, 512)
    out = dflash2.dflash2_unquantized_gemm(layer, x, layer.weight, None)
    assert cpu_qpn8["gemm"] == 1
    assert out.shape == (rows, 128) and out.dtype == torch.float16
    ref = torch.nn.functional.linear(x.float(), layer.weight.float())
    assert _rel_l2(out, ref) < 0.05


@pytest.mark.parametrize("rows", [1, 4, 7, 9, 64])
def test_other_m_falls_through_to_f_linear(cpu_qpn8, rows):
    layer = _linear(128, 512)
    assert dflash2_fp8.prepare_dflash2_fp8_m8(layer)
    x = _x(rows, 512)
    assert dflash2_fp8.apply_dflash2_fp8_m8(layer, x, None) is None
    out = dflash2.dflash2_unquantized_gemm(layer, x, layer.weight, None)
    assert cpu_qpn8["gemm"] == 0
    torch.testing.assert_close(out, torch.nn.functional.linear(x, layer.weight))


def test_bias_or_bad_input_falls_through(cpu_qpn8):
    layer = _linear(128, 512)
    assert dflash2_fp8.prepare_dflash2_fp8_m8(layer)
    bias = torch.zeros(128, dtype=torch.float16)
    assert dflash2_fp8.apply_dflash2_fp8_m8(layer, _x(8, 512), bias) is None
    assert dflash2_fp8.apply_dflash2_fp8_m8(layer, _x(8, 512).float(), None) is None
    assert dflash2_fp8.apply_dflash2_fp8_m8(layer, _x(16, 512)[::2], None) is None


def test_unprepared_layer_is_untouched(cpu_qpn8):
    layer = nn.Linear(512, 128, bias=False).half()
    x = _x(8, 512)
    out = dflash2.dflash2_unquantized_gemm(layer, x, layer.weight, None)
    assert cpu_qpn8["gemm"] == 0
    torch.testing.assert_close(out, torch.nn.functional.linear(x, layer.weight))


def test_reprepare_follows_new_weights(cpu_qpn8):
    layer = _linear(64, 256)
    assert dflash2_fp8.prepare_dflash2_fp8_m8(layer)
    layer.weight.data.mul_(2)
    assert dflash2_fp8.prepare_dflash2_fp8_m8(layer)
    x = _x(8, 256)
    out = dflash2_fp8.apply_dflash2_fp8_m8(layer, x, None)
    ref = torch.nn.functional.linear(x.float(), layer.weight.float())
    assert _rel_l2(out, ref) < 0.05


def test_fp8_replaces_fp16_packing(cpu_qpn8, monkeypatch):
    fp16_calls = []
    monkeypatch.setattr(
        dflash2, "prepare_dflash2_fp16_m8", lambda m: fp16_calls.append(m) or True
    )
    root = nn.Module()
    root.a = _linear(64, 256)
    root.b = _linear(64, 256)
    setattr(root.b, dflash2_fp8.FP8_MARK_ATTR, False)  # FP16-only layer
    assert dflash2.prepare_marked([root.a, root.b]) == 2
    assert fp16_calls == [root.b]
    assert getattr(root.a, dflash2_fp8.ROUTE_ATTR, None) is not None

    fp16_calls.clear()
    monkeypatch.setenv(VERIFIER, "0")  # killed: back to K18 FP16 packing
    c = _linear(64, 256)
    assert dflash2.prepare_marked([c]) == 1
    assert fp16_calls == [c]
    assert getattr(c, dflash2_fp8.ROUTE_ATTR, None) is None


def test_op_fake_under_fake_tensor_mode(cpu_qpn8):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(24, 512, dtype=torch.float16)
        w = torch.empty(128, 512, dtype=torch.float16)
        codes = torch.empty(512, 128, dtype=torch.uint8)
        scales = torch.empty(1, 128, dtype=torch.float16)
        out = dflash2_fp8.dflash2_fp8_m8(x, w, codes, scales, 16)
    assert tuple(out.shape) == (24, 128) and out.dtype == torch.float16


# --------------------------------------------------------- verifier gate
def _vllm_config(*, arch="Qwen3_5ForConditionalGeneration", k=7, sm70=None):
    text = SimpleNamespace(
        hidden_size=5120, num_attention_heads=24, num_key_value_heads=4, head_dim=256
    )
    model = SimpleNamespace(
        architectures=[arch], dtype=torch.float16, hf_text_config=text
    )
    draft = SimpleNamespace(
        hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
    )
    spec = SimpleNamespace(
        method="dflash", num_speculative_tokens=k, draft_model_config=draft
    )
    parallel = SimpleNamespace(
        pipeline_parallel_size=1, enable_dbo=False, ubatch_size=0
    )
    additional = {"sm70": sm70} if sm70 is not None else {}
    return SimpleNamespace(
        model_config=model,
        speculative_config=spec,
        parallel_config=parallel,
        additional_config=additional,
    )


@pytest.fixture
def _fresh_cfg(monkeypatch):
    from vllm_sm70 import config

    monkeypatch.delenv(VERIFIER, raising=False)
    config.clear_cache()
    yield
    config.clear_cache()


def test_gate_on_for_qwen38_contract(_fresh_cfg):
    assert dflash2_fp8.draft_fp8_wanted(_vllm_config())


def test_gate_off_for_qwen36_moe_arch(_fresh_cfg):
    assert not dflash2_fp8.draft_fp8_wanted(
        _vllm_config(arch="Qwen3_5MoeForConditionalGeneration")
    )


def test_gate_needs_k7_even_when_forced_on(_fresh_cfg):
    assert not dflash2_fp8.draft_fp8_wanted(
        _vllm_config(k=5, sm70={"dflash2_verifier": "on"})
    )
    assert dflash2_fp8.draft_fp8_wanted(
        _vllm_config(
            arch="Qwen3_5MoeForConditionalGeneration", sm70={"dflash2_verifier": "on"}
        )
    )


def test_gate_config_off_and_kill_switch(_fresh_cfg, monkeypatch):
    from vllm_sm70 import config

    assert not dflash2_fp8.draft_fp8_wanted(
        _vllm_config(sm70={"dflash2_verifier": "off"})
    )
    monkeypatch.setenv(VERIFIER, "0")
    config.clear_cache()
    assert not dflash2_fp8.draft_fp8_wanted(_vllm_config())


def test_gate_tolerates_missing_configs(_fresh_cfg):
    assert not dflash2_fp8.draft_fp8_wanted(SimpleNamespace())


# ----------------------------------------------------------- marking (vllm)
def test_mark_draft_linears_sets_fp8_flag(monkeypatch):
    linear = pytest.importorskip("vllm.model_executor.layers.linear")
    monkeypatch.setenv(K18, "1")

    def proj():
        m = nn.Module()
        m.quant_method = linear.UnquantizedLinearMethod()
        return m

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = proj()
    layer.mlp = nn.Module()
    layer.mlp.down_proj = proj()
    marked = dflash2.mark_draft_linears(layer, fp8=True)
    assert len(marked) == 2
    for m in marked:
        assert getattr(m, dflash2.MARK_ATTR) and getattr(m, dflash2_fp8.FP8_MARK_ATTR)
    other = nn.Module()
    other.mlp = nn.Module()
    other.mlp.down_proj = proj()
    (m,) = dflash2.mark_draft_linears(other)
    assert not getattr(m, dflash2_fp8.FP8_MARK_ATTR, False)
