# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 NVFP4 TurboMind expert route (moe_tm.py).

CPU: ``vllm_sm70.ops.tm`` is replaced by pure-torch ops with the same
signatures (prepare packs 8 codes per int32, the dense stage dequantizes a
bank and runs one GEMM per group). They check the bank layout against a torch
dequant, the route against the torch path of moe.py and the reference
recording on the tiny checkpoint (models/fixtures/inkling2_tiny/), the two
stage call forms, the slot chunks, padding, the scale audit, the admission
rules, the kill switch and the memory accounting. The real ops run in
test_m12_inkling2_nvfp4_tm_gpu.py.
Kill-switch: VLLM_SM70_INKLING_NVFP4_TM covered below.
"""

import json
import pathlib
import sys
import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch

from vllm_sm70 import envs
from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import moe, moe_tm, quant_config, sm70_configs
from vllm_sm70.models.inkling2_sm70.ops import torch_ref

TINY = pathlib.Path(__file__).resolve().parents[1] / "models/fixtures/inkling2_tiny"
NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
OPS = safetensors_torch.load_file(TINY / "ops.safetensors")
FORMATS = quant_config.expert_formats(
    json.loads((TINY / "hf_quant_config.json").read_text())
)
E2M1 = torch.tensor(ew.E2M1_VALUES, dtype=torch.float32)
SWITCH = "VLLM_SM70_INKLING_NVFP4_TM"
REAL = (2304, 1152, 128, 8)
# The tiny checkpoint: (hidden, global intermediate, experts, top-k).
TINY_CONTRACT = (64, 32, 8, 2)
ADMITTED_TP = moe_tm.INKLING_TP_SIZES
NVFP4_LAYER, MXFP8_LAYER = 1, 23
SIZES = [1, 2, 8, 40, 41, 64]  # 41 tokens x top-2 is the first dense-form size
TOL = 1e-2  # max-abs-rel, TurboMind NVFP4 against the fp16 dequant


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


# --------------------------------------------------------------------------
# Pure-torch vllm_sm70.ops.tm
# --------------------------------------------------------------------------
def _pack(codes):
    k, n = codes.shape
    c = codes.to(torch.int32).reshape(k, n // 8, 8)
    out = torch.zeros((k, n // 8), dtype=torch.int32)
    for i in range(8):
        out |= c[:, :, i] << (4 * i)
    return out


def _unpack(weight):
    parts = [(weight >> (4 * i)) & 15 for i in range(8)]
    return torch.stack(parts, dim=-1).reshape(weight.shape[0], -1).long()


def bank_dequant(weight, scales):
    """fp32 [K, N] of one prepared expert."""
    return E2M1[_unpack(weight)] * scales.float().repeat_interleave(16, dim=0)


def make_mock_tm():
    tm = types.ModuleType("vllm_sm70.ops.tm")
    tm.prepared, tm.stages, tm.banks = [], [], {}

    def nvfp4_sm70_prepare(qweight, scales, group_size=16, interleave_gated_silu=False):
        assert qweight.dtype == torch.uint8 and scales.dtype == torch.float16
        assert qweight.is_contiguous() and scales.is_contiguous()
        k, n = qweight.shape
        assert scales.shape == (k // 16, n)
        tm.prepared.append((qweight.clone(), group_size, interleave_gated_silu))
        return [_pack(qweight), scales.clone(), torch.tensor([32 * k, n])]

    def awq_moe_build_strided_ptrs(tm_w, tm_s, k_ld, q_ld, num_experts):
        assert tm_w.shape[0] == tm_s.shape[0] == num_experts
        assert (k_ld, q_ld) == (32 * tm_w.shape[1], tm_s.shape[2])
        ptrs = torch.arange(num_experts, dtype=torch.int64) + 1000 * len(tm.banks)
        tm.banks[int(ptrs[0])] = (tm_w, tm_s)
        return [ptrs, ptrs.clone()]

    def nvfp4_moe_dense_stage_sm70_out(out, inp, offsets, ids, pw, ps, ne, k, n, g):
        assert out.dtype == inp.dtype == torch.float16
        assert out.is_contiguous() and inp.is_contiguous()
        assert offsets.dtype == ids.dtype == torch.int32
        assert len(ids) == ne and len(offsets) == ne + 1 and g == 16
        assert inp.shape == (int(offsets[-1]), k) and out.shape == (inp.shape[0], n)
        assert torch.equal(pw, ps)
        tm.stages.append((inp.shape[0], ne, k, n, offsets.clone(), ids.clone()))
        tm_w, tm_s = tm.banks[int(pw[0])]
        for group in range(ne):
            a, b = int(offsets[group]), int(offsets[group + 1])
            if b > a:
                expert = int(ids[group])
                assert 0 <= expert < tm_w.shape[0]
                weight = bank_dequant(tm_w[expert], tm_s[expert])
                out[a:b] = (inp[a:b].float() @ weight).to(out.dtype)

    tm.nvfp4_sm70_prepare = nvfp4_sm70_prepare
    tm.awq_moe_build_strided_ptrs = awq_moe_build_strided_ptrs
    tm.nvfp4_moe_dense_stage_sm70_out = nvfp4_moe_dense_stage_sm70_out
    tm.has_op = lambda name: callable(getattr(tm, name, None))
    return tm


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    moe_tm.Workspace.clear()
    yield
    moe_tm.Workspace.clear()


@pytest.fixture
def tm(monkeypatch):
    """The pure-torch tm, installed, with the tiny shape admitted at TP1 and
    TP2 and the route switched on."""
    monkeypatch.setenv(SWITCH, "1")
    mock = make_mock_tm()
    ops = types.ModuleType("vllm_sm70.ops")
    ops.tm = mock
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", mock)
    monkeypatch.setattr(
        moe_tm, "INKLING_CONTRACTS", moe_tm.INKLING_CONTRACTS | {TINY_CONTRACT}
    )
    monkeypatch.setattr(moe_tm, "INKLING_TP_SIZES", (1, 2))
    return mock


@pytest.fixture(scope="module")
def config():
    return sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )


def expert_layer(config, layer, tp_size=1, tp_rank=0):
    return ew.load_expert_layers(
        NATIVE, config, tp_size=tp_size, tp_rank=tp_rank, formats=FORMATS
    )[layer]


def build(config, layer=NVFP4_LAYER, *, tp_size=1, tp_rank=0, dtype=torch.float16):
    module = moe.Inkling2Sm70Experts(config, tp_size=tp_size, dtype=dtype)
    module.load_expert_layer(expert_layer(config, layer, tp_size, tp_rank))
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    module.gate.weight.data.copy_(NATIVE[prefix + "weight"])
    module.gate.bias.data.copy_(NATIVE[prefix + "bias"])
    return module


def build_torch(config, monkeypatch, **kwargs):
    """The same layer on the torch path (the oracle)."""
    with monkeypatch.context() as patch:
        patch.setenv(SWITCH, "0")
        module = build(config, **kwargs)
    assert module.tm_route is None
    return module


def hidden(m):
    x = OPS["router.0.in.hidden_states"]
    return x[torch.arange(m) % x.shape[0]].half()


def run(module, x, is_padding=None):
    out = torch.full_like(x, float("nan"))
    module(x, out, is_padding)
    return out


# --------------------------------------------------------------------------
# Switch and admission
# --------------------------------------------------------------------------
def test_switch_is_registered_in_models():
    assert SWITCH == moe_tm.SWITCH and SWITCH in envs._SECTIONS["models"]


@pytest.mark.parametrize("value", [None, "", "1", "true", "off", " 0"])
def test_only_zero_disables(monkeypatch, tm, value):
    if value is None:
        monkeypatch.delenv(SWITCH)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert moe_tm.enabled()
    assert moe_tm.rejection(REAL, 2, torch.float16) is None
    monkeypatch.setenv(SWITCH, "0")
    assert not moe_tm.enabled()
    assert moe_tm.rejection(REAL, 2, torch.float16) == moe_tm.SWITCHED_OFF


@pytest.mark.parametrize("tp", ADMITTED_TP)
def test_real_shape_is_admitted(tm, tp, monkeypatch):
    monkeypatch.setattr(moe_tm, "INKLING_TP_SIZES", ADMITTED_TP)
    assert moe_tm.INKLING_CONTRACTS >= {REAL}
    assert moe_tm.rejection(REAL, tp, torch.float16) is None
    assert moe_tm.rejection(REAL, tp, torch.float16, torch.device("cpu")) is None


def test_only_tp2_shapes_are_admitted(tm, config, monkeypatch):
    assert ADMITTED_TP == (2,)
    monkeypatch.setattr(moe_tm, "INKLING_TP_SIZES", ADMITTED_TP)
    reason = moe_tm.rejection(REAL, 1, torch.float16)
    assert reason == "TP size 1 is not in (2,), the GPU-checked bank shapes"
    module = build(config, tp_size=1)
    assert module.tm_route is None and module.tm_rejection == reason
    assert module.w13.shape == (8, 64, 64) and not tm.prepared
    assert build(config, tp_size=2).tm_route is not None


def test_rejections(monkeypatch, tm):
    reject = moe_tm.rejection
    assert "not float16" in reject(REAL, 2, torch.float32)
    assert "TP size 4" in reject(REAL, 4, torch.float16)
    for contract in ((2048, 512, 256, 6), (2304, 1152, 128, 6), (2304, 1024, 128, 8)):
        assert "is not in" in reject(contract, 2, torch.float16)
    monkeypatch.setattr(moe_tm, "INKLING_CONTRACTS", frozenset({(2304, 1160, 128, 8)}))
    assert "multiples of 16" in reject((2304, 1160, 128, 8), 2, torch.float16)
    monkeypatch.setattr(moe_tm, "INKLING_CONTRACTS", frozenset({REAL}))
    monkeypatch.setattr(moe_tm, "_real_extension", lambda: True)
    assert "not CUDA" in reject(REAL, 2, torch.float16, torch.device("cpu"))
    assert reject(REAL, 2, torch.float16) is None
    monkeypatch.setenv(SWITCH, "0")
    assert reject(REAL, 2, torch.float16) == moe_tm.SWITCHED_OFF


def test_missing_ops_reject_and_build_refuses(monkeypatch, tm, config):
    del tm.nvfp4_moe_dense_stage_sm70_out
    assert moe_tm.turbomind() is None
    assert "lacks" in moe_tm.rejection(REAL, 2, torch.float16)
    w13, w2 = expert_layer(config, NVFP4_LAYER).nvfp4_codes()
    with pytest.raises(RuntimeError, match="nvfp4_moe_dense_stage_sm70_out"):
        moe_tm.build(
            w13, w2, device=torch.device("cpu"), gate_cap=4.0, up_cap=25.0, top_k=2
        )
    module = build(config)
    assert module.tm_route is None and "lacks" in module.tm_rejection


def test_tiny_shape_is_not_admitted_by_default(config, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    mock = make_mock_tm()
    ops = types.ModuleType("vllm_sm70.ops")
    ops.tm = mock
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", mock)
    module = build(config)
    assert module.tm_route is None and "is not in" in module.tm_rejection
    assert not mock.prepared and module.w13.shape[0] == 8


def test_route_is_on_by_default(config, monkeypatch):
    mock = make_mock_tm()
    ops = types.ModuleType("vllm_sm70.ops")
    ops.tm = mock
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", mock)
    monkeypatch.setattr(
        moe_tm, "INKLING_CONTRACTS", moe_tm.INKLING_CONTRACTS | {TINY_CONTRACT}
    )
    module = build(config, tp_size=2)
    assert module.tm_route is not None and module.tm_rejection is None
    assert mock.prepared and module.w13.numel() == 0
    monkeypatch.setenv(SWITCH, "0")
    module = build(config, tp_size=2)
    assert module.tm_route is None and module.tm_rejection == moe_tm.SWITCHED_OFF
    assert module.w13.shape == (8, 32, 64)


def test_default_falls_back_to_torch_without_the_ops(config, monkeypatch):
    """CPU without the extension: the shape is admitted, the ops are not."""
    if moe_tm.turbomind() is not None:
        pytest.skip("the TurboMind ops are loaded")
    monkeypatch.setattr(
        moe_tm, "INKLING_CONTRACTS", moe_tm.INKLING_CONTRACTS | {TINY_CONTRACT}
    )
    assert moe_tm.enabled()
    module = build(config, tp_size=2)
    assert module.tm_route is None and "lacks" in module.tm_rejection
    assert module.w13.shape == (8, 32, 64)
    reason = moe_tm.rejection(REAL, 2, torch.float16, torch.device("cpu"))
    assert "lacks" in reason


def test_range_check_sees_the_casts_of_both_routes(tm, config, monkeypatch):
    layer = expert_layer(config, NVFP4_LAYER)
    seen = []

    def check(name, values, cast):
        assert values.dtype == torch.float32 and cast.dtype == torch.float16
        seen.append(name)

    module = moe.Inkling2Sm70Experts(config)
    module.load_expert_layer(layer, range_check=check)
    assert module.tm_route is not None
    assert len(seen) == 2 and all("NVFP4 scales" in name for name in seen)
    seen.clear()
    monkeypatch.setenv(SWITCH, "0")
    module = moe.Inkling2Sm70Experts(config)
    module.load_expert_layer(layer, range_check=check)
    assert module.tm_route is None and seen
    assert not any("NVFP4 scales" in name for name in seen)


# --------------------------------------------------------------------------
# Banks
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_banks_match_the_torch_dequant(tm, config, tp_size, tp_rank):
    module = build(config, tp_size=tp_size, tp_rank=tp_rank)
    route = module.tm_route
    assert route is not None and module.tm_rejection is None
    layer = expert_layer(config, NVFP4_LAYER, tp_size, tp_rank)
    e, h, i = 8, 64, 32 // tp_size
    assert len(tm.prepared) == 2 * e
    assert all(call[1:] == (16, False) for call in tm.prepared)
    for bank, codes, (k, n) in zip(
        (route.w13, route.w2), layer.nvfp4_codes(), ((h, 2 * i), (i, h))
    ):
        assert (bank.k, bank.n, bank.num_experts) == (k, n, e)
        assert bank.weight.shape == (e, k, n // 8) and bank.weight.dtype == torch.int32
        assert bank.scales.shape == (e, k // 16, n)
        assert bank.scales.dtype == torch.float16
        assert bool(
            ((bank.scales.float().abs() >= 2.0**-14) | (bank.scales == 0)).all()
        )
        assert codes.codes.shape == (e, n, k)
    # prepare saw [K, N] codes: the transposed gate-first Q1 codes
    w13_codes, w2_codes = layer.nvfp4_codes()
    assert torch.equal(tm.prepared[0][0], w13_codes.codes[0].t())
    assert torch.equal(tm.prepared[e][0], w2_codes.codes[0].t())
    # [K, N] bank values == the fp32 dequant [N, K], up to the fp16 scale
    w13, w2 = layer.dequantize(torch.float32)
    for bank, ref in ((route.w13, w13), (route.w2, w2)):
        for expert in (0, e - 1):
            got = bank_dequant(bank.weight[expert], bank.scales[expert]).t()
            assert got.shape == ref[expert].shape
            assert torch.allclose(got, ref[expert], rtol=2.0**-10, atol=0)


def test_prepared_scales_equal_the_m4_helper(config):
    from vllm_sm70.quant.nvfp4_moe import expert_scales

    for codes in expert_layer(config, NVFP4_LAYER).nvfp4_codes():
        scales, shift = moe_tm.prepared_scales(codes)
        assert shift == 0 and scales.dtype == torch.float16
        for expert in range(codes.codes.shape[0]):
            ref = expert_scales(
                codes.block_scale[expert], codes.global_scale[expert], None
            )
            assert torch.equal(scales[expert], ref)


def test_admitted_module_holds_no_fp16_weights(tm, config):
    empty = moe.Inkling2Sm70Experts(config)
    assert empty.tm_rejection is None and empty.tm_route is None
    assert empty.w13.shape == (0, 64, 64) and empty.w2.shape == (0, 64, 32)
    module = build(config)
    assert module.w13.numel() == module.w2.numel() == 0
    assert module.state_dict().keys() == {"gate.weight", "gate.bias"}


def test_mxfp8_layer_of_an_admitted_module_uses_torch(tm, config):
    module = build(config, layer=MXFP8_LAYER, dtype=torch.float16)
    assert module.tm_route is None and module.tm_rejection is None
    assert module.w13.shape == (8, 64, 64) and module.w2.shape == (8, 64, 32)
    assert not tm.prepared
    x = OPS["experts_mxfp8.0.in.hidden_states"].half()
    out = torch.full_like(x, float("nan"))
    module.expert_sum(
        x,
        OPS["experts_mxfp8.0.in.top_k_index"],
        OPS["experts_mxfp8.0.in.top_k_weights"],
        out,
    )
    assert max_abs_rel(out, OPS["experts_mxfp8.0.out.0"]) < TOL


def test_reload_as_torch_drops_the_route(tm, config, monkeypatch):
    module = build(config)
    assert module.tm_route is not None
    monkeypatch.setenv(SWITCH, "0")
    module.tm_rejection = moe_tm.rejection(module.contract, 1, torch.float16)
    module.load_expert_layer(expert_layer(config, NVFP4_LAYER))
    assert module.tm_route is None and module.w13.shape == (8, 64, 64)


# --------------------------------------------------------------------------
# The route against the torch path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", SIZES)
def test_route_matches_torch_path_tp1(tm, config, monkeypatch, m):
    module = build(config)
    oracle = build_torch(config, monkeypatch)
    x = hidden(m)
    out, ref = run(module, x), run(oracle, x)
    assert out.dtype == torch.float16 and bool(torch.isfinite(out).all())
    assert max_abs_rel(out, ref) < TOL
    assert len(tm.stages) == 2


@pytest.mark.parametrize("m", SIZES + [20])
def test_tp2_partials_sum_to_the_recording(tm, config, monkeypatch, m):
    x = hidden(m)
    total = torch.zeros(x.shape, dtype=torch.float32)
    for rank in (0, 1):
        module = build(config, tp_size=2, tp_rank=rank)
        oracle = build_torch(config, monkeypatch, tp_size=2, tp_rank=rank)
        out, ref = run(module, x), run(oracle, x)
        assert module.tm_route.local_intermediate == 16
        assert max_abs_rel(out, ref) < TOL
        total += out.float()
    recorded = OPS["experts_nvfp4.0.out.0"]
    recorded = recorded[torch.arange(m) % recorded.shape[0]]
    assert max_abs_rel(total, recorded) < TOL


def test_fp32_reference_from_codes(tm, config):
    """Independent of ExpertLayer.dequantize: the Q1 value formula."""
    module = build(config)
    w13, w2 = expert_layer(config, NVFP4_LAYER).nvfp4_codes()
    x = hidden(20)
    _, weights, indices = module.route(x)
    ref = torch_ref.experts(
        x.float(), indices, weights, codes_dequant(w13), codes_dequant(w2), 4.0, 25.0
    )
    assert max_abs_rel(run(module, x), ref) < TOL


# --------------------------------------------------------------------------
# Stage call forms
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 8, 40])
def test_compact_form_up_to_80_slots(tm, config, m):
    module = build(config)
    x = hidden(m)
    _, _, indices = module.route(x)
    run(module, x)
    slots = 2 * m
    sorted_ids = torch.sort(indices.reshape(-1), stable=True).values
    (rows13, ne13, k13, n13, off13, ids13), (rows2, ne2, k2, n2, off2, ids2) = tm.stages
    assert (rows13, ne13, k13, n13) == (slots, slots, 64, 64)
    assert (rows2, ne2, k2, n2) == (slots, slots, 32, 64)
    for offsets, ids in ((off13, ids13), (off2, ids2)):
        assert torch.equal(offsets, torch.arange(slots + 1, dtype=torch.int32))
        assert torch.equal(ids.long(), sorted_ids)


@pytest.mark.parametrize("m", [41, 64])
def test_dense_form_above_80_slots(tm, config, m):
    module = build(config)
    x = hidden(m)
    _, _, indices = module.route(x)
    run(module, x)
    counts = torch.bincount(indices.reshape(-1), minlength=8)
    for rows, ne, _, _, offsets, ids in tm.stages:
        assert (rows, ne) == (2 * m, 8)
        assert torch.equal(ids, torch.arange(8, dtype=torch.int32))
        assert offsets[0] == 0
        assert torch.equal(offsets[1:].long(), torch.cumsum(counts, 0))


def test_compact_bound_is_the_turbomind_limit():
    from vllm_sm70.fused_moe import method as moe_rt

    assert moe_tm.COMPACT_MAX_SLOTS == moe_rt.COMPACT_MAX_SLOTS == 80


@pytest.mark.parametrize("m,pad", [(8, [0, 5]), (48, [1, 2, 47]), (64, [0, 63])])
def test_padding_rows_are_zero(tm, config, monkeypatch, m, pad):
    module = build(config)
    oracle = build_torch(config, monkeypatch)
    x = hidden(m)
    is_padding = torch.zeros(m, dtype=torch.bool)
    is_padding[pad] = True
    out, ref = run(module, x, is_padding), run(oracle, x, is_padding)
    assert bool((out[is_padding] == 0).all()) and bool(torch.isfinite(out).all())
    assert max_abs_rel(out, ref) < TOL
    # Padding slots run on expert E - 1 and add zero: no host count.
    for rows, _, _, _, offsets, ids in tm.stages:
        assert rows == int(offsets[-1]) == 2 * m
        assert int(ids.max()) < 8 and int(ids.min()) >= 0


@pytest.mark.parametrize("m", [1, 64])
def test_all_padding_is_zero(tm, config, m):
    module = build(config)
    x = hidden(m)
    x[0] = float("nan")
    out = run(module, x, torch.ones(m, dtype=torch.bool))
    assert len(tm.stages) == 2 and bool((out == 0).all())


@pytest.mark.parametrize("m", [1, 8, 41])
def test_expert_sum_has_no_host_sync(tm, config, m):
    """The mock stage stands in for a kernel; its own reads are exempt."""
    sys.path.insert(0, str(TINY.parents[1]))
    from inkling2_sync_guard import allowed, no_host_sync

    module = build(config)
    route = module.tm_route
    route.tm = types.SimpleNamespace(
        nvfp4_moe_dense_stage_sm70_out=allowed(tm.nvfp4_moe_dense_stage_sm70_out)
    )
    x = hidden(m)
    is_padding = torch.arange(m) == m - 1
    _, weights, indices = module.route(x, is_padding)
    want = torch.full_like(x, float("nan"))
    route.expert_sum(x, indices, weights, want)
    got = torch.full_like(x, float("nan"))
    with no_host_sync():
        route.expert_sum(x, indices, weights, got)
    assert torch.equal(got, want) and bool((got[-1] == 0).all())


def test_input_contract(tm, config):
    module = build(config)
    with pytest.raises(TypeError, match="fp16"):
        run(module, hidden(2).float())


# --------------------------------------------------------------------------
# Kill switch
# --------------------------------------------------------------------------
def test_kill_switch_restores_the_torch_path(tm, config, monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    module = build(config)
    assert module.tm_route is None and module.tm_rejection == moe_tm.SWITCHED_OFF
    assert module.w13.shape == (8, 64, 64) and module.w2.shape == (8, 64, 32)
    x = hidden(20)
    out = run(module, x)
    assert not tm.prepared and not tm.stages
    _, weights, indices = module.route(x)
    ref = torch_ref.experts(
        x, indices, weights, module.w13, module.w2, module.gate_cap, module.up_cap
    )
    assert torch.equal(out, ref)


def test_switch_is_read_at_construction_and_load(tm, config, monkeypatch):
    module = moe.Inkling2Sm70Experts(config)
    assert module.tm_rejection is None
    monkeypatch.setenv(SWITCH, "0")
    module.load_expert_layer(expert_layer(config, NVFP4_LAYER))
    assert module.tm_route is None and module.w13.shape == (8, 64, 64)
    assert not tm.prepared


# --------------------------------------------------------------------------
# Scale audit
# --------------------------------------------------------------------------
def random_codes(name, e, n, k, seed, *, low=1.0, high=4.0, global_scale=0.02):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (e, n, k), generator=g, dtype=torch.uint8)
    block = (torch.rand(e, n, k // 16, generator=g) * (high - low) + low).to(
        torch.float8_e4m3fn
    )
    return ew.Nvfp4Codes(name, codes, block, torch.full((e,), global_scale))


def codes_dequant(codes):
    """fp32 [E, N, K] by the Q1 formula."""
    block = codes.block_scale.float().repeat_interleave(16, dim=-1)
    return E2M1[codes.codes.long()] * block * codes.global_scale.reshape(-1, 1, 1)


def with_block(codes, value, where):
    block = codes.block_scale.float().clone()
    block[where] = value
    return ew.Nvfp4Codes(
        codes.name, codes.codes, block.to(torch.float8_e4m3fn), codes.global_scale
    )


def test_normal_scales_take_no_shift():
    scales, shift = moe_tm.prepared_scales(random_codes("w", 4, 16, 32, 0))
    assert shift == 0 and scales.shape == (4, 2, 16)
    zero = with_block(random_codes("w", 4, 16, 32, 0), 0.0, (0, 0, 0))
    assert moe_tm.prepared_scales(zero)[1] == 0


def test_non_finite_scales_raise():
    codes = random_codes("layer.w13", 4, 16, 32, 0)
    bad = ew.Nvfp4Codes(
        codes.name,
        codes.codes,
        codes.block_scale,
        torch.tensor([0.02, float("inf"), 1, 1]),
    )
    with pytest.raises(ValueError, match="layer.w13: NVFP4 scales are not finite"):
        moe_tm.prepared_scales(bad)


def test_unfittable_scale_range_raises():
    codes = random_codes("layer.w2", 2, 16, 32, 0)
    wide = ew.Nvfp4Codes(
        codes.name, codes.codes, codes.block_scale, torch.tensor([1e-9, 100.0])
    )
    with pytest.raises(ValueError, match="no power-of-two shift"):
        moe_tm.prepared_scales(wide)


def test_subnormal_scales_take_a_shift_that_the_route_refuses(tm):
    e, h, i = 8, 64, 32
    # 2**-9 * 0.02 = 3.9e-5 is below the smallest normal fp16 (6.1e-5).
    w13 = with_block(random_codes("w13", e, 2 * i, h, 1), 2.0**-9, (0, 0, 0))
    w2 = random_codes("w2", e, h, i, 2)
    scales, shift = moe_tm.prepared_scales(w13)
    assert shift == 1
    assert bool((scales.float().abs() >= 2.0**-14).all())
    exact = w13.block_scale.float() * 0.02 * 2.0
    assert torch.equal(scales, exact.half().transpose(1, 2))
    with pytest.raises(moe_tm.Rejected, match="w13: .* 2\\*\\*1 pre-shift"):
        moe_tm.Nvfp4TmExperts(
            tm, w13, w2, device=torch.device("cpu"), gate_cap=4.0, up_cap=25.0
        )
    assert not tm.prepared


@pytest.mark.parametrize("bank", [0, 1])  # W13, W2: the order Bank sees them
def test_a_shifted_layer_falls_back_to_torch(tm, config, monkeypatch, bank):
    unshifted = moe_tm.prepared_scales
    calls = []

    def shifted(codes, range_check=None):
        scales, shift = unshifted(codes, range_check)
        calls.append(codes.name)
        return scales, shift + (len(calls) - 1 == bank)

    monkeypatch.setattr(moe_tm, "prepared_scales", shifted)
    module = build(config)
    assert module.tm_route is None and "2**1 pre-shift" in module.tm_rejection
    assert module.w13.shape == (8, 64, 64) and module.w2.shape == (8, 64, 32)
    oracle = build_torch(config, monkeypatch)
    x = hidden(20)
    assert torch.equal(run(module, x), run(oracle, x))
    assert not tm.stages


def test_mismatched_pair_raises(tm):
    w13 = random_codes("w13", 8, 64, 64, 1)
    with pytest.raises(ValueError, match="not one layer's pair"):
        moe_tm.Nvfp4TmExperts(
            tm, w13, random_codes("w2", 8, 64, 16, 2),
            device=torch.device("cpu"), gate_cap=4.0, up_cap=25.0,
        )  # fmt: skip
    with pytest.raises(ValueError, match="K % 16"):
        moe_tm.Bank(tm, random_codes("w", 2, 12, 32, 0), torch.device("cpu"))


# --------------------------------------------------------------------------
# Slot chunks and the compact switch
# --------------------------------------------------------------------------
def random_route(tm, **kwargs):
    return moe_tm.Nvfp4TmExperts(
        tm, random_codes("w13", 8, 64, 64, 1), random_codes("w2", 8, 64, 32, 2),
        device=torch.device("cpu"), gate_cap=4.0, up_cap=25.0, **kwargs,
    )  # fmt: skip


def random_batch(m, seed=3, e=8, top_k=2):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, 64, generator=g) * 0.5).half()
    indices = torch.stack([torch.randperm(e, generator=g)[:top_k] for _ in range(m)])
    return x, indices, torch.rand(m, top_k, generator=g)


def test_workspace_is_bounded_independent_of_the_batch(tm, monkeypatch):
    assert moe_tm.CHUNK_ROWS == 4096
    # TP2 real shape: 4096 x (2304 + 1152 + 576 + 2304) fp16 per device.
    assert moe_tm.CHUNK_ROWS * (2304 + 1152 + 576 + 2304) * 2 == 51_904_512
    route = random_route(tm, max_slots=8192 * 8)
    ws = route.workspace
    assert ws.rows == moe_tm.CHUNK_ROWS and ws.nbytes() == 4096 * 224 * 2
    monkeypatch.setattr(moe_tm, "CHUNK_ROWS", 24)
    moe_tm.Workspace.clear()
    route = random_route(tm, max_slots=10**6)
    for m in (5, 40, 300):
        route.expert_sum(*random_batch(m), torch.empty(m, 64, dtype=torch.float16))
        assert route.workspace.rows == 24
    assert max(stage[0] for stage in tm.stages) == 24


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("m,chunk", [(40, 16), (64, 48), (300, 80), (7, 14)])
def test_chunked_output_equals_unchunked(tm, monkeypatch, compact, m, chunk):
    x, indices, weights = random_batch(m)
    whole = torch.full_like(x, float("nan"))
    random_route(tm, max_slots=2 * m, compact=compact).expert_sum(
        x, indices, weights, whole
    )
    assert [stage[0] for stage in tm.stages] == [2 * m, 2 * m]
    tm.stages.clear()
    moe_tm.Workspace.clear()
    monkeypatch.setattr(moe_tm, "CHUNK_ROWS", chunk)
    route = random_route(tm, max_slots=2 * m, compact=compact)
    chunked = torch.full_like(x, float("nan"))
    route.expert_sum(x, indices, weights, chunked)
    sizes = [min(chunk, 2 * m - start) for start in range(0, 2 * m, chunk)]
    assert [stage[0] for stage in tm.stages[::2]] == sizes
    assert route.workspace.rows == min(chunk, 2 * m)
    for rows, ne, *_ in tm.stages:
        assert ne == (rows if compact and rows <= 80 else 8)
    assert torch.allclose(chunked, whole, rtol=0, atol=2e-3)


def test_compact_switch_turns_the_compact_form_off(tm, config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_COMPACT", "0")
    module = build(config)
    assert module.tm_route.compact is False
    run(module, hidden(8))
    assert all(ne == 8 for _, ne, *_ in tm.stages)
    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_COMPACT")
    assert build(config).tm_route.compact is True


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def test_real_shape_bank_bytes():
    e, h = 128, 2304
    # TP2 rank: W13 int32 [128, 2304, 144] + fp16 [128, 144, 1152];
    # W2 int32 [128, 576, 288] + fp16 [128, 36, 2304].
    assert moe_tm.bank_nbytes(e, h, 1152) == 212_336_640
    assert moe_tm.bank_nbytes(e, 576, h) == 106_168_320
    per_layer = moe_tm.layer_nbytes(e, h, 576)
    assert per_layer == 318_504_960
    assert round(20 * per_layer / 2**30, 2) == 5.93
    assert moe_tm.layer_nbytes(e, h, 1152) == 2 * per_layer  # TP1
    # the fp16 torch path holds 3 * E * H * I * 2 bytes per layer
    assert per_layer * 16 == 5 * (3 * e * h * 576 * 2)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_route_bytes_match_the_formula(tm, config, tp_size):
    route = build(config, tp_size=tp_size).tm_route
    pointer_tables = 4 * 8 * 8
    assert route.nbytes() == moe_tm.layer_nbytes(8, 64, 32 // tp_size) + pointer_tables


def test_workspace_is_shared_and_grows(tm, config):
    first, second = build(config).tm_route, build(config).tm_route
    ws = first.workspace
    assert ws is second.workspace
    assert ws is not build(config, tp_size=2).tm_route.workspace
    rows = ws.rows
    assert rows >= moe_tm.COMPACT_MAX_SLOTS
    assert ws.nbytes() == rows * (64 + 64 + 32 + 64) * 2
    buffers = dict(ws.buffers)
    run(build(config), hidden(8))
    assert all(ws.buffers[name] is buffers[name] for name in buffers)
    ws.reserve(rows + 1)
    assert ws.rows == rows == moe_tm.CHUNK_ROWS  # max_batched_tokens * top-k
    fresh = moe_tm.Workspace(torch.device("cpu"), 8, 64, 32)
    assert fresh.rows == moe_tm.COMPACT_MAX_SLOTS
    fresh.reserve(100)
    assert fresh.rows == 100 and fresh.nbytes() == 100 * 224 * 2
