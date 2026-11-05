# SPDX-License-Identifier: Apache-2.0
"""Tests for M6 qwen-fp8-turbomind (vllm_sm70/quant/fp8_linear.py, fp8_moe.py).

What: the CPU tests mock Kai's ``vllm_sm70.ops.tm`` and UP's ``_moe_C`` /
  ``_C`` with pure-torch versions of the same signatures. They check:
  - parameter parity with UP Fp8LinearMethod / Fp8MoEMethod create_weights
    for block [128, 128];
  - fp8_sm70_prepare arguments;
  - apply vs an independent block-FP8 dequant reference for M in {1, 2, 64}
    (dense, and MoE at top-k 8, on both the batched and single-token routes);
  - the VLLM_SM70_FP8_TURBOMIND and VLLM_SM70_FP8_MOE_DECODE kill switches;
  - routing at cc 7.0 vs 8.0, skipped layers and the TP gate;
  - a weight-memory budget for both Qwen models at TP2.
  The single-token op mocks assume the semantics in FORK cf9492a
  single_token.py:19-151 (the W13 op builds the slot order itself); they are
  unverified against the CUDA source.
  GPU (``-m gpu``, one V100, _sm70_C and vLLM's _C/_moe_C), for Bea on c4130:
    python -m pytest -q -p no:cacheprovider sm70/tests/quant/test_m6_qwen_fp8.py -m gpu
Why: M6 serves both Qwen FP8 checkpoints. A wrong scale layout or route is
  silent garbage, not a crash.
Kill-switch: VLLM_SM70_FP8_TURBOMIND, VLLM_SM70_FP8_MOE_DECODE (below and in M1).
"""

import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch
import torch.nn.functional as F
import vllm.platforms
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization import fp8 as up_fp8
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.fused_moe import method as moe_rt

QWEN_FP8_CFG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
B = 128
# Small CPU MoE contract: hidden, local intermediate, experts, top-k (TP1).
H, I, E, K = 256, 128, 16, 8


# --------------------------------------------------------------------------
# Mocks
# --------------------------------------------------------------------------
def _tm_dequant(w, s):
    """TurboMind operands -> fp32 [K, N]: e4m3 [K, N] x fp16 scales [K/128, N]."""
    return w.view(
        torch.float8_e4m3fn
    ).float().cpu() * s.float().cpu().repeat_interleave(B, dim=0)


def make_mock_tm(single_token=True, compact=True):
    tm = types.ModuleType("vllm_sm70.ops.tm")
    tm.calls = []
    tm.banks = {}

    def fp8_sm70_prepare(qweight, scales, group_size, interleave_gated_silu=False):
        assert qweight.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
        n, k = qweight.shape
        assert scales.shape == (n // B, k // B) and group_size == B
        assert interleave_gated_silu is False
        tm.calls.append(("prepare", (n, k)))
        tm_w = qweight.t().contiguous().view(torch.uint8)
        tm_s = scales.t().repeat_interleave(B, dim=1).half().contiguous()  # [K/128, N]
        return [tm_w, tm_s, torch.tensor([k, n])]

    def fp8_gemm_sm70_out(out, x, w, s, group, k_ld, q_ld, gated_silu=False):
        assert group == B and gated_silu is False and x.dtype == torch.float16
        tm.calls.append(("gemm", x.shape[0]))
        out.copy_((x.float().cpu() @ _tm_dequant(w, s)).to(out.dtype))

    def awq_moe_build_strided_ptrs(tm_w, tm_s, k_ld, q_ld, num_experts):
        ptrs = torch.arange(num_experts, dtype=torch.int64) + 1000 * len(tm.banks)
        tm.banks[int(ptrs[0])] = (tm_w, tm_s)
        return [ptrs, ptrs.clone()]

    def _w(ptrs, e):
        tm_w, tm_s = tm.banks[int(ptrs[0])]
        return _tm_dequant(tm_w[e], tm_s[e])

    def fp8_moe_gemm_sm70_out(out, inp, offsets, pw, ps, ne, k, n, group, gated=False):
        assert offsets.dtype == torch.int32 and group == B and not gated
        tm.calls.append(("moe_gemm", inp.shape[0], k, n))
        for e in range(ne):
            a, b = int(offsets[e]), int(offsets[e + 1])
            if b > a:
                out[a:b] = (inp[a:b].float() @ _w(pw, e)).to(out.dtype)

    def single_w13(gu, compact, x, ids, pw, ps, off, off64, inv, sids, k, n, g, hid):
        tm.calls.append(("st_w13", k, n))
        order = torch.sort(ids[0].long(), stable=True).indices
        sids.copy_(ids[0][order])
        inv[0, order] = torch.arange(ids.shape[1], dtype=torch.int32)
        off.view(-1)[: ids.shape[1] + 1] = torch.arange(ids.shape[1] + 1)
        off64.view(-1)[: ids.shape[1] + 1] = torch.arange(ids.shape[1] + 1)
        for j, e in enumerate(sids.tolist()):
            compact[j] = x[0]
            gu[j] = (x[0].float() @ _w(pw, e)).to(gu.dtype)

    def single_w2(out, inp, off, sids, pw, ps, top_k, k, n, g):
        tm.calls.append(("st_w2", k, n))
        for j, e in enumerate(sids.tolist()):
            out[j] = (inp[j].float() @ _w(pw, e)).to(out.dtype)

    tm.fp8_sm70_prepare = fp8_sm70_prepare
    tm.fp8_gemm_sm70_out = fp8_gemm_sm70_out
    tm.awq_moe_build_strided_ptrs = awq_moe_build_strided_ptrs
    tm.fp8_moe_gemm_sm70_out = fp8_moe_gemm_sm70_out
    if single_token:
        tm.fp8_moe_single_token_dense_w13_sm70_out = single_w13
        tm.fp8_moe_single_token_indexed_dense_stage_sm70_out = single_w2
    if compact:

        def single_compact(out, x, weights, ids, w13w, w13s, w2w, w2s, *rest):
            # rest: compact_input, gate_up, intermediate, sorted_output,
            # sorted_weights, 4 dst ptr rows, expert_offsets, inv, sorted ids,
            # broadcast idx, w2 raw weight/scale, w13_k, w13_n, w2_k, w2_n,
            # group, hidden, 5 flags, exact_per_route (K03 tm.py signature).
            assert len(rest) == 27
            dims, flags = rest[15:21], rest[21:]
            tm.calls.append(("compact", dims, flags))
            assert ids.dtype == torch.int32 and weights.dtype == torch.float32
            assert w13w.dim() == 2 and w13w.shape[0] == w2w.shape[0]  # ptr rows
            inter = dims[2]
            acc = torch.zeros(out.shape[1])
            for kk, e in enumerate(ids[0].tolist()):
                gu = x[0].float() @ _w(w13w.view(-1), e)
                act = (F.silu(gu[:inter]) * gu[inter:]).half().float()
                y = (act @ _w(w2w.view(-1), e)).half().float()
                acc += float(weights[0, kk]) * y
            out[0] = acc.to(out.dtype)

        tm.fp8_moe_single_token_sm70_out = single_compact
    tm.has_op = lambda name: callable(getattr(tm, name, None))
    return tm


class TorchMoeOps:
    """Pure-torch UP _moe_C / _C (csrc/moe/torch_bindings.cpp:105-136)."""

    @staticmethod
    def available():
        return True

    @staticmethod
    def sort_workspace_size(rows, n_expert):
        return 16

    @staticmethod
    def permute_with_scratch(
        x, ids, tei, emap, n, nl, topk, pin, off, inv, pidx, ws, pe, sr, tfs
    ):
        flat = ids.reshape(-1).long()
        order = torch.sort(flat, stable=True).indices
        inv.view(-1)[order] = torch.arange(flat.numel(), dtype=torch.int32)
        counts = torch.bincount(flat, minlength=n)
        off.copy_(torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]))
        pin.copy_(x[order // topk])
        pidx.copy_(order)

    @staticmethod
    def unpermute(sorted_out, weights, inv, off, topk, out):
        rows = sorted_out[inv.long()].float()
        out.copy_((rows * weights.unsqueeze(-1)).sum(1).to(out.dtype))

    @staticmethod
    def silu_and_mul(out, x):
        d = x.shape[-1] // 2
        out.copy_((F.silu(x[:, :d].float()) * x[:, d:].float()).to(out.dtype))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (
        "VLLM_SM70_FP8_TURBOMIND",
        "VLLM_SM70_FP8_MOE_DECODE",
        "VLLM_SM70_FP8_MOE_COMPACT_DECODE",
    ):
        monkeypatch.delenv(name, raising=False)
    import vllm.model_executor.parameter as param_mod

    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    import vllm.distributed as dist

    monkeypatch.setattr(dist, "get_tensor_model_parallel_world_size", lambda: 1)
    vc = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16))
    monkeypatch.setattr(up_fp8, "get_current_vllm_config", lambda: vc)
    moe_rt.Sm70MoEWorkspace.clear()
    yield
    moe_rt.Sm70MoEWorkspace.clear()


@pytest.fixture
def mods(monkeypatch):
    def _install(single_token=True, compact=True):
        tm = make_mock_tm(single_token, compact)
        ops = types.ModuleType("vllm_sm70.ops")
        ops.tm = tm
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", tm)
        monkeypatch.setattr(moe_rt, "OPS", TorchMoeOps)
        monkeypatch.setattr(moe_rt, "max_batched_tokens", lambda: 64)
        import vllm_sm70.quant.fp8_linear as lin
        import vllm_sm70.quant.fp8_moe as moe

        return lin, moe, tm

    return _install


class _FakeCuda(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"
    capability: DeviceCapability | None = None

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return cls.capability


@pytest.fixture
def device(monkeypatch):
    def _set(major, minor, ops=True):
        cls = type("Dev", (_FakeCuda,), {})
        cls.capability = DeviceCapability(major, minor)
        monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
        monkeypatch.setattr(
            sm70_platform, "all_visible_gpus_are_sm70", lambda: (major, minor) == (7, 0)
        )
        monkeypatch.setattr(_ext, "is_loaded", lambda: ops)
        sm70_platform.is_sm70_ready.cache_clear()

    yield _set
    sm70_platform.is_sm70_ready.cache_clear()


def _cfg(**extra):
    from vllm_sm70.quant.configs import Sm70Fp8Config

    return Sm70Fp8Config.from_config({**QWEN_FP8_CFG, **extra})


def _loader(param, loaded, *a, **k):
    param.data.copy_(loaded)


def _random_block_fp8(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(n // B, k // B, generator=g) * 2e-3 + 1e-4
    return w, s


def _ref_dequant(w, s):
    """Block FP8: W[n, k] = e4m3[n, k] * s[n//128, k//128] (scale rounded to fp16
    as TurboMind stores it)."""
    scale = s.half().float().repeat_interleave(B, 0).repeat_interleave(B, 1)
    return w.float() * scale


def _close(out, ref):
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------
def _dense_layer(method, k, parts):
    layer = torch.nn.Module()
    layer.tp_size = 1
    method.create_weights(
        layer, k, list(parts), k, sum(parts), torch.float16, weight_loader=_loader
    )
    return layer


def test_dense_create_weights_match_upstream(mods, monkeypatch):
    lin, _, _ = mods()
    monkeypatch.setattr(up_fp8, "init_fp8_linear_kernel", lambda **k: object())
    parts = [512, 128, 128]  # qkv-like
    ours = _dense_layer(lin.Fp8Sm70LinearMethod(_cfg()), 256, parts)
    theirs = _dense_layer(up_fp8.Fp8LinearMethod(_cfg()), 256, parts)

    def sig(layer):
        return {
            n: (type(p), p.dtype, tuple(p.shape)) for n, p in layer.named_parameters()
        }

    assert sig(ours) == sig(theirs)
    assert sig(ours)["weight"][1:] == (torch.float8_e4m3fn, (768, 256))
    assert sig(ours)["weight_scale_inv"][1:] == (torch.float32, (6, 2))
    from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

    assert "Fp8Sm70LinearMethod" in WEIGHT_LOADER_V2_SUPPORTED


@pytest.mark.parametrize("m", [1, 2, 64])
def test_dense_apply_matches_reference(mods, m):
    lin, _, tm = mods()
    method = lin.Fp8Sm70LinearMethod(_cfg())
    layer = _dense_layer(method, 256, [256, 128])
    w, s = _random_block_fp8(384, 256, seed=m)
    layer.weight.data.copy_(w)
    layer.weight_scale_inv.data.copy_(s)
    method.process_weights_after_loading(layer)
    assert tm.calls[0] == ("prepare", (384, 256))
    assert (layer.sm70_fp8_k_ld, layer.sm70_fp8_q_ld) == (256, 384)
    x = (torch.randn(m, 256, generator=torch.Generator().manual_seed(m)) * 0.5).half()
    bias = torch.randn(384).half()
    out = method.apply(layer, x, bias)
    assert out.shape == (m, 384) and out.dtype == torch.float16
    _close(out, x.float() @ _ref_dequant(w, s).t() + bias.float())
    out3 = method.apply(layer, x.view(1, m, 256))
    assert out3.shape == (1, m, 384)


def test_dense_rejects_non_block_and_bf16(mods):
    lin, _, _ = mods()
    with pytest.raises(NotImplementedError):
        lin.Fp8Sm70LinearMethod(_cfg(weight_block_size=[64, 64]))
    method = lin.Fp8Sm70LinearMethod(_cfg())
    layer = _dense_layer(method, 256, [128])
    layer.orig_dtype = torch.bfloat16
    with pytest.raises(RuntimeError, match="--dtype half"):
        method.process_weights_after_loading(layer)


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------
def _moe_config(hidden=None, inter=None, experts=None, top_k=None, tp=1):
    # Resolve defaults at call time. Default arguments bound to the module
    # constants at definition time (H, I, E, K) are what made the GPU test
    # build a 256-hidden layer while generating 2048-hidden weights.
    hidden = H if hidden is None else hidden
    inter = I if inter is None else inter
    experts = E if experts is None else experts
    top_k = K if top_k is None else top_k
    par = SimpleNamespace(use_all2all_kernels=False, use_ep=False)
    return SimpleNamespace(
        hidden_dim=hidden,
        intermediate_size_per_partition=inter,
        tp_size=tp,
        num_experts=experts,
        experts_per_token=top_k,
        has_bias=False,
        is_act_and_mul=True,
        w13_num_shards=2,
        moe_parallel_config=par,
        dp_size=1,
        is_sequence_parallel=False,
    )


def _routed(cfg):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    class _Routed(RoutedExperts):
        expert_map = None  # UP's property needs the full layer; TP only here

    layer = _Routed.__new__(_Routed)
    torch.nn.Module.__init__(layer)
    layer.moe_config = cfg
    layer.activation = MoEActivation.SILU
    layer.swiglu_limit = None
    layer.apply_router_weight_on_input = False
    layer.local_num_experts = layer.global_num_experts = cfg.num_experts
    return layer


def _moe_layer(method, cfg, seed=0):
    layer = _routed(cfg)
    method.create_weights(
        layer,
        cfg.num_experts,
        cfg.hidden_dim,
        cfg.intermediate_size_per_partition,
        torch.float16,
        weight_loader=_loader,
    )
    # Shapes come from the layer's own config, the same values UP's
    # Fp8MoEMethod.create_weights uses: w13 [E, 2*I_rank, H], w2 [E, H, I_rank].
    h, i, e_n = cfg.hidden_dim, cfg.intermediate_size_per_partition, cfg.num_experts
    w = {}
    for stage, n, k in (("w13", 2 * i, h), ("w2", h, i)):
        ws, ss = zip(
            *[
                _random_block_fp8(n, k, seed * 100 + e + (stage == "w2") * 50)
                for e in range(e_n)
            ]
        )
        w[stage] = (torch.stack(ws), torch.stack(ss))
        getattr(layer, stage + "_weight").data.copy_(w[stage][0])
        getattr(layer, stage + "_weight_scale_inv").data.copy_(w[stage][1])
    return layer, w


def _moe_reference(x, w, ids, weights):
    inter = w["w2"][0].shape[2]  # w2 [E, H, I_rank]
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for kk in range(ids.shape[1]):
            e = int(ids[t, kk])
            gu = x[t].float() @ _ref_dequant(w["w13"][0][e], w["w13"][1][e]).t()
            act = (F.silu(gu[:inter]) * gu[inter:]).half().float()
            y = (act @ _ref_dequant(w["w2"][0][e], w["w2"][1][e]).t()).half().float()
            out[t] += float(weights[t, kk]) * y
    return out


def _routing(m, seed=1, experts=None, top_k=None):
    experts = E if experts is None else experts
    top_k = K if top_k is None else top_k
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(m, experts, generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k)
    return (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)


def _build_moe(moe, cfg=None):
    cfg = cfg or _moe_config()
    method = moe.Fp8Sm70MoEMethod(_cfg(), _routed(cfg))
    layer, w = _moe_layer(method, cfg)
    method.process_weights_after_loading(layer)
    return method, layer, w


def test_moe_create_weights_match_upstream(mods):
    _, moe, _ = mods()
    cfg = _moe_config()
    ours = _routed(cfg)
    moe.Fp8Sm70MoEMethod(_cfg(), ours).create_weights(ours, E, H, I, torch.float16)
    up = up_fp8.Fp8MoEMethod.__new__(up_fp8.Fp8MoEMethod)
    up.moe, up.quant_config = cfg, _cfg()
    up.weight_block_size = up.moe_block_shape = [B, B]
    up.block_quant, up.weight_scale_refine = True, None
    up.weight_scale_name = "weight_scale_inv"
    theirs = _routed(cfg)
    up.create_weights(theirs, E, H, I, torch.float16)

    def sig(layer):
        return {
            n: (type(p), p.dtype, tuple(p.shape), getattr(p, "quant_method", None))
            for n, p in layer.named_parameters()
        }

    assert sig(ours) == sig(theirs)
    assert sig(ours)["w13_weight_scale_inv"][1:3] == (torch.float32, (E, 2, 2))
    assert ours.w13_input_scale is None and ours.w2_input_scale is None


def test_moe_prepare_args(mods):
    _, moe, tm = mods()
    _, layer, _ = _build_moe(moe)
    prep = [c for c in tm.calls if c[0] == "prepare"]
    assert prep[:1] == [("prepare", (2 * I, H))] and prep[-1] == ("prepare", (H, I))
    assert len(prep) == 2 * E
    assert not hasattr(layer, "w13_weight") and layer.w13_tm_weight.shape == (
        E,
        H,
        2 * I,
    )


@pytest.mark.parametrize("route", ["compact", "decomposed", "batched"])
@pytest.mark.parametrize("m", [1, 2, 64])
def test_moe_apply_matches_reference(mods, monkeypatch, m, route):
    if route == "decomposed":
        monkeypatch.setenv("VLLM_SM70_FP8_MOE_COMPACT_DECODE", "0")
    if route == "batched":
        monkeypatch.setenv("VLLM_SM70_FP8_MOE_DECODE", "0")
        monkeypatch.setenv("VLLM_SM70_FP8_MOE_COMPACT_DECODE", "1")  # decode=0 wins
    _, moe, tm = mods()
    method, layer, w = _build_moe(moe)
    x = (torch.randn(m, H, generator=torch.Generator().manual_seed(m)) * 0.5).half()
    weights, ids = _routing(m)
    tm.calls.clear()
    out = method.apply(layer, x, weights, ids, None, None)
    _close(out, _moe_reference(x, w, ids, weights))
    kinds = [c[0] for c in tm.calls]
    if m == 1 and route == "compact":
        assert kinds == ["compact"]  # exactly one launch
        _, dims, flags = tm.calls[0]
        assert dims == (H, 2 * I, I, H, B, H)
        assert flags == (False, False, False, False, False, True)  # exact_layout
    elif m == 1 and route == "decomposed":
        assert kinds == ["st_w13", "st_w2"]
    else:
        assert kinds == ["moe_gemm", "moe_gemm"]


def test_moe_decode_off_without_ops(mods):
    _, moe, _ = mods(single_token=False)
    _, layer, _ = _build_moe(moe)
    assert not layer.sm70_fp8_moe_decode


def test_moe_compact_off_without_op(mods):
    _, moe, tm = mods(compact=False)
    method, layer, w = _build_moe(moe)
    assert not layer.sm70_fp8_moe_compact and layer.sm70_fp8_moe_decode
    weights, ids = _routing(1)
    x = torch.randn(1, H).half()
    tm.calls.clear()
    method.apply(layer, x, weights, ids, None, None)
    assert [c[0] for c in tm.calls] == ["st_w13", "st_w2"]


def test_moe_compact_kill_switch(mods, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_MOE_COMPACT_DECODE", "0")
    _, moe, _ = mods()
    _, layer, _ = _build_moe(moe)
    assert not layer.sm70_fp8_moe_compact and layer.sm70_fp8_moe_decode
    assert not moe.compact_policy_enabled() and moe.decode_policy_enabled()


def test_moe_decode_kill_switch(mods, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FP8_MOE_DECODE", "0")
    _, moe, _ = mods()
    _, layer, _ = _build_moe(moe)
    assert not layer.sm70_fp8_moe_decode and not moe.decode_policy_enabled()
    assert not layer.sm70_fp8_moe_compact  # decode=0 also disables compact


# --------------------------------------------------------------------------
# Routing and kill switches
# --------------------------------------------------------------------------
def _linear_base():
    from vllm.model_executor.layers.linear import LinearBase

    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return layer


@pytest.mark.parametrize(
    "cap, ready, kill, routed",
    [
        ((7, 0), True, False, True),
        ((7, 0), True, True, False),  # VLLM_SM70_FP8_TURBOMIND=0
        ((7, 0), False, False, False),  # no _sm70_C
        ((8, 0), True, False, False),
    ],
)
def test_get_quant_method_routing(mods, device, monkeypatch, cap, ready, kill, routed):
    lin, moe, _ = mods()
    device(*cap, ops=ready)
    if kill:
        monkeypatch.setenv("VLLM_SM70_FP8_TURBOMIND", "0")

    class UpMoE:
        def __init__(self, quant_config, layer):
            pass

    monkeypatch.setattr(up_fp8, "Fp8MoEMethod", UpMoE)
    cfg = _cfg()
    lm = cfg.get_quant_method(_linear_base(), "model.layers.0.mlp.down_proj")
    mm = cfg.get_quant_method(_routed(_moe_config()), "model.layers.0.mlp.experts")
    if routed:
        assert type(lm) is lin.Fp8Sm70LinearMethod
        assert isinstance(mm, moe.Fp8Sm70MoEMethod)
    else:
        assert type(lm) is up_fp8.Fp8LinearMethod
        assert isinstance(mm, UpMoE)
    from vllm_sm70.quant.configs import Sm70Fp8Config

    expected_cap = 70 if (cap == (7, 0) and ready and not kill) else 75
    assert Sm70Fp8Config.get_min_capability() == expected_cap


def test_skipped_layers_stay_upstream(mods, device):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    mods()
    device(7, 0)
    cfg = _cfg(ignored_layers=["model.layers.0.linear_attn.in_proj_ba"])
    method = cfg.get_quant_method(
        _linear_base(), "model.layers.0.linear_attn.in_proj_ba"
    )
    assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_moe_tp_gate(mods, device, monkeypatch, tp):
    _, moe, _ = mods()
    device(7, 0)
    monkeypatch.setattr(up_fp8, "Fp8MoEMethod", None)  # never fall back
    layer = _routed(_moe_config(tp=tp))
    if tp in (1, 2):
        assert isinstance(
            _cfg().get_quant_method(layer, "x.experts"), moe.Fp8Sm70MoEMethod
        )
    else:
        with pytest.raises(NotImplementedError, match="tensor_parallel_size"):
            _cfg().get_quant_method(layer, "x.experts")


# --------------------------------------------------------------------------
# Memory budget (TP2, per GPU)
# --------------------------------------------------------------------------
# Dims verified against config.json at the pinned revisions (2026-11-05):
# Qwen/Qwen3.6-35B-A3B-FP8 @95a723d0 (text_config: hidden 2048, 40 layers,
# 256 experts x moe_inter 512, shared 512, heads 16/2, head_dim 256, linear
# heads k16/v32 x 128, vocab 248320) and Qwen/Qwen3.8-27B-FP8 @017b9c7a
# (hidden 5120, inter 17408, 64 layers, heads 24/4, linear k16/v48).
# Both pin in_proj_a/in_proj_b as not-converted, so the 48-row TP2 shards
# never hit validate_fp8_block_shape.
QWEN36_35B_A3B = dict(
    hidden=2048,
    layers=40,
    full_attn_every=4,
    vocab=248320,
    experts=256,
    moe_inter=512,
    shared_inter=512,
    q_heads=16,
    kv_heads=2,
    head_dim=256,
    lin_k_heads=16,
    lin_v_heads=32,
    lin_head_dim=128,
    dense_inter=0,
)
QWEN38_27B = dict(
    hidden=5120,
    layers=64,
    full_attn_every=4,
    vocab=248320,
    experts=0,
    moe_inter=0,
    shared_inter=0,
    q_heads=24,
    kv_heads=4,
    head_dim=256,
    lin_k_heads=16,
    lin_v_heads=48,
    lin_head_dim=128,
    dense_inter=17408,
)


def _per_gpu_weight_bytes(c, tp=2):
    """FP8 (1 B/param) for quantized linears and experts, FP16 (2 B) for
    embeddings, lm_head, norms and the router; everything is split over TP."""
    h, layers = c["hidden"], c["layers"]
    full = layers // c["full_attn_every"]
    gdn = layers - full
    attn = (
        full
        * h
        * (c["q_heads"] * c["head_dim"] * 2 + 2 * c["kv_heads"] * c["head_dim"])
    )
    attn += full * c["q_heads"] * c["head_dim"] * h  # o_proj
    kd = c["lin_k_heads"] * c["lin_head_dim"]
    vd = c["lin_v_heads"] * c["lin_head_dim"]
    lin = gdn * (h * (2 * kd + 2 * vd) + vd * h)  # in_proj_qkvz + out_proj
    mlp = layers * 3 * h * (c["dense_inter"] + c["shared_inter"])
    experts = layers * c["experts"] * 3 * h * c["moe_inter"]
    fp8 = attn + lin + mlp + experts
    fp16 = 2 * c["vocab"] * h + layers * (c["experts"] * h + 4 * h)
    return (fp8 * 1 + fp16 * 2) / tp


@pytest.mark.parametrize(
    "cfg", [QWEN36_35B_A3B, QWEN38_27B], ids=["3.6-35B-A3B", "3.8-27B"]
)
def test_fp8_weights_fit_v100_at_tp2(cfg):
    per_gpu = _per_gpu_weight_bytes(cfg)
    total_params_b = per_gpu * 2 / 1e9
    assert 20 < total_params_b < 40  # sanity: ~27B/~35B params, mostly 1 B each
    # 32 GiB V100, 0.91 utilization, minus ~2.2 GiB LMCache context: weights
    # must leave room for KV; require < 70% of the 32 GiB.
    assert per_gpu < 0.7 * 32 * 2**30, f"{per_gpu / 2**30:.1f} GiB per GPU"


QWEN36_TOP_K = 8  # num_experts_per_tok in the pinned config.json


def _qwen36_tp2_moe_config():
    """Qwen3.6-35B-A3B routed experts on one rank at TP2 (pinned config)."""
    c = QWEN36_35B_A3B
    return _moe_config(
        hidden=c["hidden"],
        inter=c["moe_inter"] // 2,
        experts=c["experts"],
        top_k=QWEN36_TOP_K,
        tp=2,
    )


@pytest.mark.parametrize("route", ["compact", "batched"])
def test_moe_qwen36_tp2_real_shapes_cpu(mods, monkeypatch, route):
    """The GPU test's exact layer build at real Qwen3.6 TP2 shapes, with ops
    mocked: catches a helper/create_weights layout mismatch on CPU."""
    if route == "batched":
        monkeypatch.setenv("VLLM_SM70_FP8_MOE_DECODE", "0")
    _, moe, tm = mods()
    cfg = _qwen36_tp2_moe_config()
    assert (cfg.hidden_dim, cfg.intermediate_size_per_partition) == (2048, 256)
    method = moe.Fp8Sm70MoEMethod(_cfg(), _routed(cfg))
    layer, w = _moe_layer(method, cfg)
    assert layer.w13_weight.shape == (256, 512, 2048)
    assert layer.w2_weight.shape == (256, 2048, 256)
    assert layer.w13_weight_scale_inv.shape == (256, 4, 16)
    assert layer.w2_weight_scale_inv.shape == (256, 16, 2)
    method.process_weights_after_loading(layer)
    assert layer.w13_tm_weight.shape == (256, 2048, 512)
    assert layer.w2_tm_weight.shape == (256, 256, 2048)
    m = 1 if route == "compact" else 2
    x = (torch.randn(m, 2048, generator=torch.Generator().manual_seed(3)) * 0.5).half()
    weights, ids = _routing(m, experts=256, top_k=QWEN36_TOP_K)
    tm.calls.clear()
    out = method.apply(layer, x, weights, ids, None, None)
    _close(out, _moe_reference(x, w, ids, weights))
    assert [c[0] for c in tm.calls] == (
        ["compact"] if route == "compact" else ["moe_gemm", "moe_gemm"]
    )


# --------------------------------------------------------------------------
# GPU (Bea, c4130)
# --------------------------------------------------------------------------
def _gpu_dense(n_parts, k):
    import vllm_sm70.quant.fp8_linear as lin

    method = lin.Fp8Sm70LinearMethod(_cfg())
    layer = _dense_layer(method, k, n_parts)
    w, s = _random_block_fp8(sum(n_parts), k, seed=0)
    layer.weight.data.copy_(w)
    layer.weight_scale_inv.data.copy_(s)
    layer.cuda()
    method.process_weights_after_loading(layer)
    return method, layer, w, s


# Boolean condition only (string skipif conditions break under inference_mode).
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize(
    "parts, k",
    [([8704, 8704], 5120), ([5120], 8704), ([256], 2048), ([2048], 4096)],
    ids=["3.8-gate_up", "3.8-down", "3.6-shared", "3.6-o_proj"],
)
@pytest.mark.parametrize("m", [1, 2, 64, 512])
def test_gpu_dense_parity(parts, k, m):
    method, layer, w, s = _gpu_dense(parts, k)
    x = (torch.randn(m, k) * 0.5).half().cuda()
    out = method.apply(layer, x)
    _close(out.cpu(), x.float().cpu() @ _ref_dequant(w, s).t())


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("m", [1, 2, 8, 128])
def test_gpu_moe_qwen36_tp2_parity_and_graph(m):
    import vllm_sm70.quant.fp8_moe as moe

    cfg = _qwen36_tp2_moe_config()
    method = moe.Fp8Sm70MoEMethod(_cfg(), _routed(cfg))
    layer, w = _moe_layer(method, cfg)
    layer.cuda()
    method.process_weights_after_loading(layer)
    x = (torch.randn(m, cfg.hidden_dim) * 0.5).half().cuda()
    weights, ids = _routing(m, experts=cfg.num_experts, top_k=cfg.experts_per_token)
    weights, ids = weights.cuda(), ids.cuda()
    eager = method.apply(layer, x, weights, ids, None, None)
    _close(eager.cpu(), _moe_reference(x.cpu(), w, ids.cpu(), weights.cpu()))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        method.apply(layer, x, weights, ids, None, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, weights, ids, None, None)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
