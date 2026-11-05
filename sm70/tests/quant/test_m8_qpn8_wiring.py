# SPDX-License-Identifier: Apache-2.0
"""Tests for M8: K09 QPN8 FP8 decode wired into the Qwen dense FP8 path.

What: the CPU tests run Kai's real vllm_sm70.ops.qpn8 Python layer
  (qpn8_prepare, qpn8_linear, the opaque qpn8_native_linear op and its M
  choice) over pure-torch fakes of the three _sm70_C QPN8 ops, and a mock of
  the K03 TurboMind FP8 ops for the M6 route. They check:
  - routing per resolved Sm70Config.fp8_qpn8 (True / False / auto through the
    DFlash2 verifier contract), the VLLM_SM70_FP8_QPN8=0 kill (and "1" never
    enabling), the TP gate (1 and 2 only);
  - the M boundary inside K09's op (QPN8_MAX_M read from the module): up to
    it the QPN8 GEMV, above it the dequantize into the layer's load-time
    scratch, never the GEMV;
  - the max-rows gate: max_num_seqs * (k + 1) <= QPN8_MAX_M;
  - missing K09 module / ops / rejected shape -> the plain M6 path;
  - prepare runs once at load, apply allocates no weight-sized tensor and
    changes no buffer; the scratch is shared per device and grown at load;
  - compute_hash differs with fp8_qpn8 on/off.
  GPU (``-m gpu``, one V100 with _sm70_C built with K03 and K09), for Bea:
    python -m pytest -q -p no:cacheprovider sm70/tests/quant/test_m8_qpn8_wiring.py -m gpu
  QPN8 route vs M6 route at the Qwen3.8-27B TP2 dense shapes, M in {1, 2, 8}
  (K09 tolerances: max-abs-rel < 1e-2, rel-L2 < 1e-3 vs the dequantized
  reference), M=QPN8_MAX_M+1 falls back off the GEMV, and an M=1 CUDA
  graph replay.
Why: a wrong route or scale layout is silent garbage, not a crash.
Kill-switch: VLLM_SM70_FP8_QPN8=0.
"""

import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch
from vllm.model_executor.layers.quantization import fp8 as up_fp8

from vllm_sm70 import config as sm70_cfg

B = 128
QWEN_FP8_CFG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [B, B],
}
HAS_SM70_GPU = bool(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
def _fake_dequant(codes, group_scales):
    """Fake QPN8 layout: codes = plain [K, N] e4m3 bytes, group scales
    [K/128, N/32] = fp16(scale * 256), as K09's docstring describes."""
    scale = group_scales.float().repeat_interleave(B, 0).repeat_interleave(32, 1)
    return codes.view(torch.float8_e4m3fn).float() * scale / 256.0


def install_fake_qpn8(monkeypatch, qpn8):
    calls = []

    def prepare(qweight, scales):
        assert qweight.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
        n, k = qweight.shape
        assert scales.shape == (n // B, k // B)
        calls.append(("prepare", (n, k)))
        codes = qweight.t().contiguous().view(torch.uint8)
        gs = (scales.t().repeat_interleave(4, dim=1) * 256).half().contiguous()
        return [codes, gs]

    def gemm(out, x, codes, gs, split_k, chains, fast=True, prefetch=False):
        assert 1 <= x.shape[0] <= qpn8.QPN8_MAX_M
        calls.append(("qpn8", x.shape[0], split_k, chains))
        out.copy_((x.float() @ _fake_dequant(codes, gs)).to(out.dtype))

    def dequant(out, codes, gs):
        calls.append(("dequant", tuple(out.shape), out.data_ptr()))
        out.copy_(_fake_dequant(codes, gs).to(out.dtype))

    monkeypatch.setattr(qpn8, "has_op", lambda name: True)
    monkeypatch.setattr(qpn8, "fp8_qpn8_prepare_sm70", prepare)
    monkeypatch.setattr(qpn8, "fp8_qpn8_gemm_sm70_out", gemm)
    monkeypatch.setattr(qpn8, "fp8_qpn8_dequantize_sm70_out", dequant)
    return calls


def make_mock_tm(calls):
    tm = types.ModuleType("vllm_sm70.ops.tm")

    def fp8_sm70_prepare(qweight, scales, group_size, interleave_gated_silu=False):
        n, k = qweight.shape
        calls.append(("tm_prepare", (n, k)))
        tm_w = qweight.t().contiguous().view(torch.uint8)
        tm_s = scales.t().repeat_interleave(B, dim=1).half().contiguous()
        return [tm_w, tm_s, torch.tensor([k, n])]

    def fp8_gemm_sm70_out(out, x, w, s, group, k_ld, q_ld, gated_silu=False):
        calls.append(("tm_gemm", x.shape[0]))
        deq = w.view(torch.float8_e4m3fn).float() * s.float().repeat_interleave(B, 0)
        out.copy_((x.float() @ deq).to(out.dtype))

    tm.fp8_sm70_prepare = fp8_sm70_prepare
    tm.fp8_gemm_sm70_out = fp8_gemm_sm70_out
    return tm


def _vllm_config(sm70=None, tp=2, qualified=False, max_num_seqs=1, spec_k=7):
    """A vllm_config stand-in; qualified=True matches the fork's Qwen3.8
    DFlash2 verifier contract, False is Qwen3.6-35B-A3B (MoE arch)."""
    arch = "Qwen3_5ForCausalLM" if qualified else "Qwen3_5MoeForCausalLM"
    return SimpleNamespace(
        additional_config={"sm70": dict(sm70 or {})},
        model_config=SimpleNamespace(
            dtype=torch.float16,
            architectures=[arch],
            hf_text_config=SimpleNamespace(
                hidden_size=5120,
                num_attention_heads=24,
                num_key_value_heads=4,
                head_dim=256,
            ),
        ),
        speculative_config=None
        if spec_k is None
        else SimpleNamespace(
            method="dflash",
            num_speculative_tokens=spec_k,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=1
        ),
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("VLLM_SM70_FP8_TURBOMIND", "VLLM_SM70_FP8_QPN8"):
        monkeypatch.delenv(name, raising=False)
    import vllm.distributed as dist
    import vllm.model_executor.parameter as param_mod

    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_tensor_model_parallel_world_size", lambda: 1)
    sm70_cfg.clear_cache()
    yield
    sm70_cfg.clear_cache()
    import vllm_sm70.quant.fp8_linear as lin

    lin.clear_qpn8_scratch()


@pytest.fixture
def engine(monkeypatch):
    """Set the current vllm_config seen by M6/M8."""
    import vllm_sm70.quant.fp8_linear as lin

    def _set(vc):
        monkeypatch.setattr(lin, "get_current_vllm_config_or_none", lambda: vc)
        monkeypatch.setattr(up_fp8, "get_current_vllm_config", lambda: vc)
        return vc

    return _set


@pytest.fixture
def mods(monkeypatch, engine):
    """(lin, qpn8, calls): M6 TurboMind mocked, K09 Python real, its _sm70_C
    ops faked. with_qpn8=False installs a vllm_sm70.ops without qpn8."""
    import vllm_sm70.ops.qpn8 as qpn8

    def _install(with_qpn8=True):
        calls = []
        tm = make_mock_tm(calls)
        ops = types.ModuleType("vllm_sm70.ops")
        ops.tm = tm
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", tm)
        if with_qpn8:
            calls_q = install_fake_qpn8(monkeypatch, qpn8)
            ops.qpn8 = qpn8
            monkeypatch.setitem(sys.modules, "vllm_sm70.ops.qpn8", qpn8)
        else:
            calls_q = []
            monkeypatch.delitem(sys.modules, "vllm_sm70.ops.qpn8", raising=False)
        import vllm_sm70.quant.fp8_linear as lin

        return lin, qpn8, SimpleNamespace(tm=calls, qpn8=calls_q)

    return _install


def _cfg():
    from vllm_sm70.quant.configs import Sm70Fp8Config

    return Sm70Fp8Config.from_config(dict(QWEN_FP8_CFG))


def _loader(param, loaded, *a, **k):
    param.data.copy_(loaded)


def _random_block_fp8(n, k, seed, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(n // B, k // B, generator=g) * 2e-3 + 1e-4
    return w.to(device), s.to(device)


def _ref_dequant(w, s):
    scale = s.float().repeat_interleave(B, 0).repeat_interleave(B, 1)
    return w.float() * scale


def _close(out, ref):
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


def _layer(lin, k, parts, seed=0, device="cpu"):
    method = lin.Fp8Sm70LinearMethod(_cfg())
    layer = torch.nn.Module()
    layer.tp_size = 2
    method.create_weights(
        layer, k, list(parts), k, sum(parts), torch.float16, weight_loader=_loader
    )
    w, s = _random_block_fp8(sum(parts), k, seed)
    layer.weight.data.copy_(w)
    layer.weight_scale_inv.data.copy_(s)
    if device != "cpu":
        layer.to(device)
    method.process_weights_after_loading(layer)
    return method, layer, w, s


def _x(m, k, seed=1, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, k, generator=g) * 0.5).half().to(device)


# --------------------------------------------------------------------------
# CPU: routing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sm70, qualified, routed",
    [
        ({"fp8_qpn8": True}, False, True),
        ({"fp8_qpn8": False}, True, False),
        ({}, True, True),  # auto, Qwen3.8 DFlash2 contract
        ({}, False, False),  # auto, Qwen3.6-35B-A3B
        ({"dflash2_verifier": "off"}, True, False),  # auto follows the verifier
    ],
    ids=["on", "off", "auto-3.8", "auto-3.6", "auto-verifier-off"],
)
def test_route_follows_resolved_config(mods, engine, sm70, qualified, routed):
    lin, _, calls = mods()
    engine(_vllm_config(sm70, qualified=qualified))
    method, layer, w, s = _layer(lin, 256, [256, 128])
    assert layer.sm70_fp8_qpn8 is routed
    assert [c[0] for c in calls.tm] == ([] if routed else ["tm_prepare"])
    x = _x(1, 256)
    bias = torch.randn(384).half()
    out = method.apply(layer, x, bias)
    _close(out, x.float() @ _ref_dequant(w, s).t() + bias.float())
    if routed:
        assert [c[0] for c in calls.qpn8] == ["prepare", "qpn8"]
        assert layer.weight.numel() == 0 and layer.weight_scale_inv.numel() == 0
    else:
        assert calls.tm[-1] == ("tm_gemm", 1)
        assert not any(c[0] == "qpn8" for c in calls.qpn8)


def test_env_kill_and_never_enable(mods, engine, monkeypatch):
    lin, _, calls = mods()
    engine(_vllm_config({"fp8_qpn8": True}))
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "0")
    _, layer, _, _ = _layer(lin, 256, [256])
    assert layer.sm70_fp8_qpn8 is False and calls.tm[0][0] == "tm_prepare"
    sm70_cfg.clear_cache()
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "1")
    engine(_vllm_config({"fp8_qpn8": False}))
    _, layer, _, _ = _layer(lin, 256, [256])
    assert layer.sm70_fp8_qpn8 is False


@pytest.mark.parametrize("tp, routed", [(1, True), (2, True), (4, False), (8, False)])
def test_tp_gate(mods, engine, tp, routed):
    lin, _, _ = mods()
    engine(_vllm_config({"fp8_qpn8": True}, tp=tp))
    _, layer, _, _ = _layer(lin, 256, [256])
    assert layer.sm70_fp8_qpn8 is routed


@pytest.mark.parametrize(
    "max_num_seqs, spec_k, max_m",
    [
        (1, 7, 8),  # M = 8: admitted
        (2, 7, 8),  # M = 16: M6
        (8, None, 8),  # no spec, M = 8: admitted
        (9, None, 8),  # M6
        (4, 7, 32),  # K09 with M<=32: M = 32, admitted
        (5, 7, 32),  # M = 40: M6
        (4, 7, None),  # the real K09 constant
        (5, 7, None),
        (1, 7, None),
    ],
    ids=[
        "1x8-max8",
        "2x8-max8",
        "8x1-max8",
        "9x1-max8",
        "4x8-max32",
        "5x8-max32",
        "4x8-real",
        "5x8-real",
        "1x8-real",
    ],
)
def test_max_rows_gate(mods, engine, monkeypatch, max_num_seqs, spec_k, max_m):
    lin, qpn8, calls = mods()
    if max_m is not None:
        monkeypatch.setattr(qpn8, "QPN8_MAX_M", max_m)
    routed = max_num_seqs * ((spec_k or 0) + 1) <= qpn8.QPN8_MAX_M
    engine(_vllm_config({"fp8_qpn8": True}, max_num_seqs=max_num_seqs, spec_k=spec_k))
    method, layer, w, s = _layer(lin, 256, [256])
    assert layer.sm70_fp8_qpn8 is routed
    assert [c[0] for c in calls.tm] == ([] if routed else ["tm_prepare"])
    x = _x(2, 256)
    _close(method.apply(layer, x), x.float() @ _ref_dequant(w, s).t())


def test_missing_max_m_constant_is_8(mods, engine, monkeypatch):
    lin, qpn8, _ = mods()
    monkeypatch.delattr(qpn8, "QPN8_MAX_M")
    assert lin.qpn8_max_m(qpn8) == 8
    engine(_vllm_config({"fp8_qpn8": True}, max_num_seqs=8, spec_k=None))
    assert _layer(lin, 256, [256])[1].sm70_fp8_qpn8 is True
    lin.clear_qpn8_scratch()
    sm70_cfg.clear_cache()
    engine(_vllm_config({"fp8_qpn8": True}, max_num_seqs=9, spec_k=None))
    assert _layer(lin, 256, [256])[1].sm70_fp8_qpn8 is False


def test_no_vllm_config_is_m6(mods, engine):
    lin, _, _ = mods()
    engine(None)
    assert lin.qpn8_route_enabled() is False


# --------------------------------------------------------------------------
# CPU: M range (inside K09's opaque op)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", [0, 1, 2, 8, 9, 32, 33, 64])
def test_m_boundary(mods, engine, m):
    lin, qpn8, calls = mods()
    engine(_vllm_config({"fp8_qpn8": True}))
    method, layer, w, s = _layer(lin, 512, [384])
    calls.qpn8.clear()
    x = _x(m, 512)
    out = method.apply(layer, x)
    assert out.shape == (m, 384) and out.dtype == torch.float16
    kinds = [c[0] for c in calls.qpn8]
    if m == 0:
        assert kinds == []
        return
    _close(out, x.float() @ _ref_dequant(w, s).t())
    if m <= qpn8.QPN8_MAX_M:
        assert kinds == ["qpn8"]
    else:
        scratch = getattr(layer, lin.QPN8_SCRATCH)
        assert calls.qpn8 == [("dequant", (512, 384), scratch.data_ptr())]
    assert calls.tm == []  # no TurboMind copy, no M6 GEMM on a QPN8 layer
    out3 = method.apply(layer, x.view(1, m, 512))
    assert out3.shape == (1, m, 384)


# --------------------------------------------------------------------------
# CPU: admission / feature detection -> M6
# --------------------------------------------------------------------------
def test_missing_qpn8_module_is_m6(mods, engine):
    lin, _, calls = mods(with_qpn8=False)
    engine(_vllm_config({"fp8_qpn8": True}))
    assert lin.qpn8_module() is None
    method, layer, w, s = _layer(lin, 256, [256])
    assert layer.sm70_fp8_qpn8 is False
    x = _x(2, 256)
    _close(method.apply(layer, x), x.float() @ _ref_dequant(w, s).t())
    assert [c[0] for c in calls.tm] == ["tm_prepare", "tm_gemm"]


@pytest.mark.parametrize("missing", ["has_op", "qpn8_prepare", "qpn8_rejection"])
def test_missing_op_or_api_is_m6(mods, engine, monkeypatch, missing):
    lin, qpn8, calls = mods()
    engine(_vllm_config({"fp8_qpn8": True}))
    if missing == "has_op":
        monkeypatch.setattr(qpn8, "has_op", lambda n: n != "fp8_qpn8_gemm_sm70_out")
    else:
        monkeypatch.setattr(qpn8, missing, None)
    _, layer, _, _ = _layer(lin, 256, [256])
    assert layer.sm70_fp8_qpn8 is False and calls.tm[0][0] == "tm_prepare"


def test_rejected_shape_is_m6(mods, engine, monkeypatch):
    lin, qpn8, calls = mods()
    engine(_vllm_config({"fp8_qpn8": True}))
    seen = []

    def reject(n, k):
        seen.append((n, k))
        return "no"

    monkeypatch.setattr(qpn8, "qpn8_rejection", reject)
    _, layer, _, _ = _layer(lin, 256, [256, 128])
    assert seen == [(384, 256)]
    assert layer.sm70_fp8_qpn8 is False and calls.qpn8 == []


# --------------------------------------------------------------------------
# CPU: load-time prep, no allocation in apply, shared scratch
# --------------------------------------------------------------------------
def test_prepare_once_at_load_no_alloc_in_apply(mods, engine, monkeypatch):
    lin, _, calls = mods()
    engine(_vllm_config({"fp8_qpn8": True}))
    method, layer, _, _ = _layer(lin, 512, [256])
    assert [c[0] for c in calls.qpn8] == ["prepare"]
    before = {n: (b.data_ptr(), tuple(b.shape)) for n, b in layer.named_buffers()}
    assert set(before) == {
        "sm70_qpn8_codes",
        "sm70_qpn8_group_scales",
        lin.QPN8_SCRATCH,
    }
    assert before["sm70_qpn8_codes"][1] == (512, 256)
    assert before["sm70_qpn8_group_scales"][1] == (4, 8)

    big = []
    real_empty = torch.empty

    def spy_empty(*shape, **kw):
        dims = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        if int(torch.Size(dims).numel()) >= 512 * 256:
            big.append(dims)
        return real_empty(*shape, **kw)

    monkeypatch.setattr(torch, "empty", spy_empty)
    for m in (1, 8, 9, 32, 33, 64):
        method.apply(layer, _x(m, 512))
    monkeypatch.setattr(torch, "empty", real_empty)
    assert big == []  # M>8 used the load-time scratch
    assert [c[0] for c in calls.qpn8].count("prepare") == 1
    after = {n: (b.data_ptr(), tuple(b.shape)) for n, b in layer.named_buffers()}
    assert after == before


def test_scratch_shared_and_grown_at_load(mods, engine):
    lin, _, _ = mods()
    engine(_vllm_config({"fp8_qpn8": True}))
    _, small, _, _ = _layer(lin, 256, [128])
    _, big, _, _ = _layer(lin, 512, [384], seed=1)
    _, mid, _, _ = _layer(lin, 256, [256], seed=2)
    scratches = [getattr(layer, lin.QPN8_SCRATCH) for layer in (small, big, mid)]
    assert all(t is scratches[0] for t in scratches)
    assert scratches[0].numel() == 512 * 384 and scratches[0].dtype == torch.float16


# --------------------------------------------------------------------------
# CPU: config plumbing
# --------------------------------------------------------------------------
def test_compute_hash_differs_with_fp8_qpn8(monkeypatch):
    assert "fp8_qpn8" in sm70_cfg.GRAPH_FIELDS
    on = sm70_cfg.compute_hash(sm70_cfg.Sm70Config(fp8_qpn8=True))
    off = sm70_cfg.compute_hash(sm70_cfg.Sm70Config(fp8_qpn8=False))
    auto = sm70_cfg.compute_hash(sm70_cfg.Sm70Config())
    assert len({on, off, auto}) == 3
    vc = _vllm_config({"fp8_qpn8": True})
    assert sm70_cfg.compute_hash(vc) == on
    sm70_cfg.clear_cache()
    monkeypatch.setenv("VLLM_SM70_FP8_QPN8", "0")
    assert sm70_cfg.compute_hash(_vllm_config({"fp8_qpn8": True})) == off


def test_verifier_contract_fields():
    q = sm70_cfg.dflash2_verifier_qualified
    vc = _vllm_config(qualified=True)
    args = (vc.model_config, vc.speculative_config, vc.parallel_config)
    assert q(*args)
    assert not q(vc.model_config, None, vc.parallel_config)
    vc.speculative_config.num_speculative_tokens = 5
    assert not q(*args)
    vc.speculative_config.num_speculative_tokens = 7
    vc.speculative_config.draft_model_config.hf_config.dflash_config = {}
    assert not q(*args)
    vc.speculative_config.draft_model_config.hf_config.dflash_config = {
        "selector_top_k": 16
    }
    assert q(*args)
    vc.parallel_config.pipeline_parallel_size = 2
    assert not q(*args)
    vc.parallel_config.pipeline_parallel_size = 1
    vc.model_config.dtype = torch.bfloat16
    assert not q(*args)


# --------------------------------------------------------------------------
# GPU (Bea, c4130): real K03 + K09 ops
# --------------------------------------------------------------------------
# Qwen3.8-27B dense FP8 linears per TP2 rank: (output partitions, K).
QWEN38_TP2 = {
    "qkv_proj": ([6144, 512, 512], 5120),
    "o_proj": ([5120], 3072),
    "gdn.in_proj_qkvz": ([1024, 1024, 3072, 3072], 5120),
    "gdn.out_proj": ([5120], 3072),
    "mlp.gate_up_proj": ([8704, 8704], 5120),
    "mlp.down_proj": ([5120], 8704),
}


@pytest.fixture
def gpu_ops():
    if not HAS_SM70_GPU:
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail("SM70 GPU present but vllm_sm70._sm70_C is not loaded")
    import vllm_sm70.quant.fp8_linear as lin
    from vllm_sm70.ops import qpn8

    if lin.qpn8_module() is None:
        pytest.fail("_sm70_C is loaded but has no K09 QPN8 ops")
    return lin, qpn8


def _gpu_pair(lin, engine, parts, k):
    engine(_vllm_config({"fp8_qpn8": False}))
    m6, l6, w, s = _layer(lin, k, parts, device="cuda")
    sm70_cfg.clear_cache()
    engine(_vllm_config({"fp8_qpn8": True}))
    q8, lq, _, _ = _layer(lin, k, parts, device="cuda")
    assert l6.sm70_fp8_qpn8 is False and lq.sm70_fp8_qpn8 is True
    ref = _ref_dequant(w, s).cuda()  # fp32 [N, K]
    return (m6, l6), (q8, lq), ref


def _errors(y, ref):
    return {
        "max_abs_rel": ((y.float() - ref).abs().max() / ref.abs().max()).item(),
        "rel_l2": ((y.float() - ref).norm() / ref.norm()).item(),
    }


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_SM70_GPU, reason="requires an SM70 GPU")
@pytest.mark.parametrize("name", sorted(QWEN38_TP2))
@pytest.mark.parametrize("m", [1, 2, 8])
def test_gpu_qpn8_matches_m6(gpu_ops, engine, name, m):
    lin, _ = gpu_ops
    parts, k = QWEN38_TP2[name]
    (m6, l6), (q8, lq), ref_w = _gpu_pair(lin, engine, parts, k)
    x = _x(m, k, device="cuda")
    ref = x.float() @ ref_w.t()
    y6, yq = m6.apply(l6, x), q8.apply(lq, x)
    for y in (y6, yq):
        err = _errors(y, ref)
        assert err["max_abs_rel"] < 1e-2 and err["rel_l2"] < 1e-3, err
    assert _errors(yq, y6.float())["max_abs_rel"] < 1e-2


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_SM70_GPU, reason="requires an SM70 GPU")
@pytest.mark.parametrize("name", ["mlp.gate_up_proj", "mlp.down_proj"])
def test_gpu_above_max_m_falls_back_off_the_gemv(gpu_ops, engine, monkeypatch, name):
    lin, qpn8 = gpu_ops
    parts, k = QWEN38_TP2[name]
    (m6, l6), (q8, lq), ref_w = _gpu_pair(lin, engine, parts, k)
    gemv = []
    real = qpn8.fp8_qpn8_gemm_sm70_out
    monkeypatch.setattr(
        qpn8,
        "fp8_qpn8_gemm_sm70_out",
        lambda *a, **kw: gemv.append(1) or real(*a, **kw),
    )
    m_max = qpn8.QPN8_MAX_M  # 32 with K09 M9..32
    x = _x(m_max + 1, k, device="cuda")
    yq = q8.apply(lq, x)
    assert gemv == []
    ref = x.float() @ ref_w.t()
    err = _errors(yq, ref)
    assert err["max_abs_rel"] < 1e-2 and err["rel_l2"] < 1e-3, err
    assert _errors(yq, m6.apply(l6, x).float())["max_abs_rel"] < 1e-2
    q8.apply(lq, _x(m_max, k, device="cuda"))
    assert gemv == [1]


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_SM70_GPU, reason="requires an SM70 GPU")
def test_gpu_qpn8_m1_cuda_graph(gpu_ops, engine):
    lin, _ = gpu_ops
    parts, k = QWEN38_TP2["mlp.down_proj"]
    _, (q8, lq), _ = _gpu_pair(lin, engine, parts, k)
    x = _x(1, k, device="cuda")
    eager = q8.apply(lq, x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q8.apply(lq, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = q8.apply(lq, x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
