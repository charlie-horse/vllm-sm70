# SPDX-License-Identifier: Apache-2.0
"""Tests for M4 nvfp4-moe-turbomind (vllm_sm70/quant/nvfp4_moe.py,
vllm_sm70/fused_moe/method.py).

What: the CPU tests mock Kai's ``vllm_sm70.ops.tm`` and UP's ``_moe_C`` / ``_C``
  ops with pure-torch versions that have the same signatures and semantics.
  They check: parameter parity with UP's ModelOptNvFp4FusedMoE.create_weights;
  that process_weights_after_loading calls prepare per expert with UP-format
  [K, N] codes and [K/16, N] fp16 scales with the global folded in; apply
  against an independent ModelOpt-NVFP4 reference (dequant + top-k + SwiGLU
  + weighted sum) for M in {1, 2, 64}, top-k 8, always on the materialized W13 route
  (never the Qwen3.8-only indexed op, even when present); the K12 decode
  wiring against a fake ``vllm_sm70.ops.flint_moe`` (admit_layer args and
  policy, kill switch, one shared workspace per device allocated at load,
  can_decode gating with int32/int64 IDs, exact flint_moe_decode_out args,
  dense route otherwise); that workspaces are persistent (no allocation in
  apply); and get_quant_method routing at cc 7.0 vs 8.0.
  GPU (``-m gpu``, Bea on c4130, one V100, needs _sm70_C and vLLM's
  _C/_moe_C): parity on the real Flint TP2 expert shapes (E256, H2048,
  I_local 256, top-6), K12 decode == dense route for M in {1, 2} (int32
  and int64 IDs), and CUDA-graph capture == eager:
    python -m pytest -q -p no:cacheprovider sm70/tests/quant/test_m4_nvfp4_moe.py -m gpu
Why: Flint's routed experts are most of its FLOPs; M4 is their only SM70 path.
Kill-switch: VLLM_SM70_FLINT_MOE_QPN_DECODE covered below; VLLM_SM70_QUANT in M2.
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
from vllm.model_executor.layers.quantization import modelopt as up_modelopt
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

from vllm_sm70 import _ext
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.fused_moe import method as moe_rt

E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
FLINT_QUANT_CFG = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["model.embed_tokens", "*.mlp.gate", "lm_head"],
    },
}
# Small CPU contract: (hidden, global intermediate, experts, top-k), TP1.
SMALL = (64, 64, 16, 8)


# --------------------------------------------------------------------------
# Mocks: Kai's tm and UP's _moe_C/_C
# --------------------------------------------------------------------------
def _pack_n8(codes):
    k, n = codes.shape
    c = codes.to(torch.int32).reshape(k, n // 8, 8)
    out = torch.zeros((k, n // 8), dtype=torch.int32)
    for i in range(8):
        out |= c[:, :, i] << (4 * i)
    return out


def _unpack_n8(w):
    parts = [(w >> (4 * i)) & 15 for i in range(8)]
    return torch.stack(parts, dim=-1).reshape(w.shape[0], -1).long()


def _dequant_tm(w, s, group=16):
    return E2M1[_unpack_n8(w).cpu()] * s.float().cpu().repeat_interleave(group, dim=0)


def make_mock_tm(indexed=True):
    tm = types.ModuleType("vllm_sm70.ops.tm")
    tm.calls = []
    tm.banks = {}

    def nvfp4_sm70_prepare(qweight, scales, group_size=16, interleave_gated_silu=False):
        assert qweight.dtype == torch.uint8 and scales.dtype == torch.float16
        assert scales.shape == (qweight.shape[0] // 16, qweight.shape[1])
        tm.calls.append(
            ("prepare", tuple(qweight.shape), group_size, interleave_gated_silu)
        )
        k, n = qweight.shape
        return [_pack_n8(qweight), scales.clone(), torch.tensor([32 * k, n])]

    def awq_moe_build_strided_ptrs(tm_w, tm_s, k_ld, q_ld, num_experts):
        ptrs = torch.arange(num_experts, dtype=torch.int64) + 1000 * len(tm.banks)
        tm.banks[int(ptrs[0])] = (tm_w, tm_s)
        return [ptrs, ptrs.clone()]

    def _stage(out, rows, offsets, expert_ids, ptrs_w, k, n, group):
        tm_w, tm_s = tm.banks[int(ptrs_w[0])]
        assert out.shape[1] == n and rows.shape[1] == k
        for e in range(len(expert_ids)):
            a, b = int(offsets[e]), int(offsets[e + 1])
            if b > a:
                w = _dequant_tm(
                    tm_w[int(expert_ids[e])], tm_s[int(expert_ids[e])], group
                )
                out[a:b] = (rows[a:b].float() @ w).to(out.dtype)

    def nvfp4_moe_dense_stage_sm70_out(out, inp, offsets, ids, pw, ps, ne, k, n, g):
        assert offsets.dtype == torch.int32 and ne == len(ids)
        tm.calls.append(("dense", inp.shape[0], k, n))
        _stage(out, inp, offsets, ids, pw, k, n, g)

    def nvfp4_moe_indexed_dense_stage_sm70_out(
        out, inp, rows, offsets, ids, pw, ps, ne, k, n, g
    ):
        tm.calls.append(("indexed", inp.shape[0], k, n))
        _stage(out, inp[rows.long()], offsets, ids, pw, k, n, g)

    tm.nvfp4_sm70_prepare = nvfp4_sm70_prepare
    tm.awq_moe_build_strided_ptrs = awq_moe_build_strided_ptrs
    tm.nvfp4_moe_dense_stage_sm70_out = nvfp4_moe_dense_stage_sm70_out
    if indexed:
        tm.nvfp4_moe_indexed_dense_stage_sm70_out = (
            nvfp4_moe_indexed_dense_stage_sm70_out
        )
    tm.has_op = lambda name: callable(getattr(tm, name, None))
    return tm


def make_fake_flint_moe(admit=True):
    """Fake vllm_sm70.ops.flint_moe (K12): records every call.

    admit_layer admits iff the policy is on and ``admit``; can_decode takes
    M in {1, 2}; flint_moe_decode_out computes the routed sum in torch (any
    top-k), so the decode route is checked for numbers too.
    """
    fm = types.ModuleType("vllm_sm70.ops.flint_moe")
    fm.calls = []
    fm.allocations = []

    class DecodePlan(SimpleNamespace):
        pass

    def enabled():
        from vllm_sm70 import envs

        return bool(envs.VLLM_SM70_FLINT_MOE_QPN_DECODE)

    def admit_layer(contract, w13_w, w13_s, w2_w, w2_s, *, policy_enabled=None):
        fm.calls.append(
            ("admit", tuple(contract), policy_enabled, w13_w, w13_s, w2_w, w2_s)
        )
        if not (enabled() if policy_enabled is None else policy_enabled):
            return None
        return DecodePlan(w13_scale_shift=3, w2_scale_shift=5) if admit else None

    def allocate_workspace(device):
        fm.allocations.append(torch.device(device))
        return SimpleNamespace(device=torch.device(device))

    def can_decode(x, topk_weights, topk_ids):
        fm.calls.append(("can_decode", x.shape[0], topk_ids.dtype))
        return 0 < x.shape[0] <= 2

    def flint_moe_decode_out(
        out, inter, x, w13_w, w13_s, w2_w, w2_s, weights, ids, ws, plan
    ):
        fm.calls.append(
            ("decode", out, inter, x, w13_w, w13_s, w2_w, w2_s, weights, ids, ws, plan)
        )
        acc = x.new_zeros(out.shape, dtype=torch.float32)  # not torch.zeros
        for t in range(x.shape[0]):
            for kk in range(ids.shape[1]):
                e = int(ids[t, kk])
                gu = x[t].float() @ _dequant_tm(w13_w[e], w13_s[e])
                g, u = gu.chunk(2)
                act = (F.silu(g) * u).half().float()
                acc[t] += float(weights[t, kk]) * (act @ _dequant_tm(w2_w[e], w2_s[e]))
        out.copy_(acc.to(out.dtype))  # fully overwrite, like the kernel
        return out

    for fn in (
        enabled,
        admit_layer,
        allocate_workspace,
        can_decode,
        flint_moe_decode_out,
    ):
        setattr(fm, fn.__name__, fn)
    fm.DecodePlan = DecodePlan
    return fm


class TorchMoeOps:
    """Pure-torch UP _moe_C / _C semantics (csrc/moe/torch_bindings.cpp:105-136)."""

    allocations = 0

    @staticmethod
    def available():
        return True

    @staticmethod
    def sort_workspace_size(rows, n_expert):
        TorchMoeOps.allocations += 1
        return 16

    @staticmethod
    def _sort(ids, n_expert, offsets64, scatter, sorted_experts, sorted_rows):
        flat = ids.reshape(-1).long()
        order = torch.sort(flat, stable=True).indices
        sorted_rows.view(-1).copy_(order)
        sorted_experts.view(-1).copy_(flat[order])
        scatter.view(-1)[order] = torch.arange(flat.numel(), dtype=torch.int32)
        counts = torch.bincount(flat, minlength=n_expert)
        offsets64.copy_(torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]))
        return order

    @staticmethod
    def prepare_scatter(ids, tei, emap, n, nl, off, scatter, ws, sexp, srows, tfs):
        assert ids.dtype == torch.int32 and emap is None
        TorchMoeOps._sort(ids, n, off, scatter, sexp, srows)

    @staticmethod
    def permute_with_scratch(
        x, ids, tei, emap, n, nl, topk, pin, off, inv, pidx, ws, pe, srows, tfs
    ):
        assert ids.dtype == torch.int32
        order = TorchMoeOps._sort(ids, n, off, inv, pe, srows)
        pin.copy_(x[order // topk])
        pidx.copy_(order)

    @staticmethod
    def unpermute(sorted_out, weights, inv, off, topk, out):
        rows = sorted_out[inv.long()].float()  # [M, topk, H]
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
    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", raising=False)
    from vllm_sm70.quant import nvfp4_moe

    nvfp4_moe.clear_decode_workspaces()
    import vllm.model_executor.parameter as param_mod

    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(param_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    moe_rt.Sm70MoEWorkspace.clear()
    yield
    moe_rt.Sm70MoEWorkspace.clear()
    nvfp4_moe.clear_decode_workspaces()


@pytest.fixture
def mods(monkeypatch):
    """mods(indexed=True, flint=None) -> (nvfp4_moe module, mock tm).

    ``flint`` is the vllm_sm70.ops.flint_moe to install (None: absent, the
    import fails); the mock tm is ``mods.tm``.
    """

    def _install(indexed=True, small=True, flint=None):
        tm = make_mock_tm(indexed)
        ops = types.ModuleType("vllm_sm70.ops")
        ops.tm = tm
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops", ops)
        monkeypatch.setitem(sys.modules, "vllm_sm70.ops.tm", tm)
        if flint is not None:
            ops.flint_moe = flint
            monkeypatch.setitem(sys.modules, "vllm_sm70.ops.flint_moe", flint)
        monkeypatch.setattr(moe_rt, "OPS", TorchMoeOps)
        monkeypatch.setattr(moe_rt, "max_batched_tokens", lambda: 64)
        import vllm_sm70.quant.nvfp4_moe as m4

        if small:
            monkeypatch.setattr(m4, "SUPPORTED_CONTRACTS", frozenset({SMALL}))
        return m4, tm

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


def _quant_config():
    from vllm_sm70.quant.configs import Sm70ModelOptNvFp4Config

    return Sm70ModelOptNvFp4Config.from_config(
        {k: (dict(v) if isinstance(v, dict) else v) for k, v in FLINT_QUANT_CFG.items()}
    )


def _moe_config(hidden, inter_global, experts, top_k, tp=1):
    par = SimpleNamespace(use_all2all_kernels=False, use_ep=False)
    return SimpleNamespace(
        hidden_dim=hidden,
        intermediate_size_per_partition=inter_global // tp,
        tp_size=tp,
        num_experts=experts,
        experts_per_token=top_k,
        has_bias=False,
        is_act_and_mul=True,
        moe_parallel_config=par,
        dp_size=1,
        is_sequence_parallel=False,
    )


def _loader(param, loaded, *a, **k):
    param.data.copy_(loaded)


def _random_experts(e, h, i, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)

    def blk(n, k):
        return (torch.rand((e, n, k // 16), generator=g) * 400 + 1).to(
            torch.float8_e4m3fn
        )

    return {
        "w13_weight": torch.randint(
            0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, generator=g
        ),
        "w2_weight": torch.randint(
            0, 256, (e, h, i // 2), dtype=torch.uint8, generator=g
        ),
        "w13_weight_scale": blk(2 * i, h),
        "w2_weight_scale": blk(h, i),
        # gate and up globals differ, to catch a swapped split
        "w13_weight_scale_2": (torch.rand((e, 2), generator=g) * 2e-4 + 1e-4),
        "w2_weight_scale_2": (torch.rand((e,), generator=g) * 2e-4 + 1e-4),
        "w13_input_scale": torch.ones(e, 2),
        "w2_input_scale": torch.ones(e),
    }


def _make_layer(method, cfg, weights, device="cpu"):
    h, e = cfg.hidden_dim, cfg.num_experts
    i = cfg.intermediate_size_per_partition
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        e,
        h,
        i,
        torch.float16,
        weight_loader=_loader,
        global_num_experts=e,
        unpadded_hidden_size=h,
    )
    for name, value in weights.items():
        getattr(layer, name).data.copy_(value)
    if device != "cpu":
        layer.to(device)
    layer.moe_config = cfg
    layer.activation = MoEActivation.SILU
    layer.swiglu_limit = None
    layer.apply_router_weight_on_input = False
    layer.expert_map = None
    layer.local_num_experts = layer.global_num_experts = e
    return layer


def _dequant_ref(packed, block, glob, split=None):
    """ModelOpt NVFP4: W[n, k] = e2m1(nibble k; low = even) * block[n, k//16] * global;
    the global is folded into an FP16 scale, as TurboMind consumes it."""
    n = packed.shape[0]
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(n, -1).long().cpu()
    scale = block.float().cpu()
    glob = glob.float().cpu().reshape(-1)
    if split is None:
        scale = scale * glob[0]
    else:
        scale = torch.cat([scale[:split] * glob[0], scale[split:] * glob[-1]])
    return E2M1[codes] * scale.half().float().repeat_interleave(16, dim=1)


def _moe_reference(x, w, ids, weights, i):
    out = torch.zeros(x.shape, dtype=torch.float32)
    xf = x.float().cpu()
    for t in range(x.shape[0]):
        for kk in range(ids.shape[1]):
            e = int(ids[t, kk])
            w13 = _dequant_ref(
                w["w13_weight"][e],
                w["w13_weight_scale"][e],
                w["w13_weight_scale_2"][e],
                i,
            )
            w2 = _dequant_ref(
                w["w2_weight"][e], w["w2_weight_scale"][e], w["w2_weight_scale_2"][e]
            )
            gu = xf[t] @ w13.t()
            act = (F.silu(gu[:i]) * gu[i:]).half().float()  # fp16 boundary
            out[t] += float(weights[t, kk]) * (act @ w2.t()).half().float()
    return out


def _routing(m, e, k, seed=1):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(m, e, generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), k)
    return (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)


def _build(m4, seed=0, cfg=None):
    cfg = cfg or _moe_config(*SMALL)
    method = m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), cfg)
    weights = _random_experts(
        cfg.num_experts, cfg.hidden_dim, cfg.intermediate_size_per_partition, seed
    )
    layer = _make_layer(method, cfg, weights)
    method.process_weights_after_loading(layer)
    return method, layer, weights


# --------------------------------------------------------------------------
# CPU tests
# --------------------------------------------------------------------------
def test_create_weights_match_upstream(mods, monkeypatch):
    m4, _ = mods(small=False)  # real Flint TP2 contract
    cfg = _moe_config(2048, 512, 256, 6, tp=2)
    ours = torch.nn.Module()
    m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), cfg).create_weights(
        ours, 4, 2048, 256, torch.float16, weight_loader=_loader, global_num_experts=4
    )
    up = up_modelopt.ModelOptNvFp4FusedMoE.__new__(up_modelopt.ModelOptNvFp4FusedMoE)
    up.moe, up.quant_config, up.use_global_sf = cfg, _quant_config(), False
    theirs = torch.nn.Module()
    up.create_weights(
        theirs, 4, 2048, 256, torch.float16, weight_loader=_loader, global_num_experts=4
    )

    def sig(layer):
        return {
            n: (type(p), p.dtype, tuple(p.shape)) for n, p in layer.named_parameters()
        }

    assert sig(ours) == sig(theirs)
    assert sig(ours)["w13_weight"][1:] == (torch.uint8, (4, 512, 1024))
    assert sig(ours)["w2_weight_scale"][1:] == (torch.float8_e4m3fn, (4, 2048, 16))
    assert sig(ours)["w13_weight_scale_2"][1:] == (torch.float32, (4, 2))


def test_process_weights_prepare_args(mods):
    m4, tm = mods()
    _method, layer, w = _build(m4)
    h, gi, e, _k = SMALL
    prep = [c for c in tm.calls if c[0] == "prepare"]
    assert len(prep) == 2 * e
    assert prep[0] == ("prepare", (h, 2 * gi), 16, False)  # W13: [K=H, N=2I]
    assert prep[1] == ("prepare", (gi, h), 16, False)  # W2: [K=I, N=H]
    # TM bank equals the UP-format dequant (codes transposed, globals folded).
    for ex in (0, e - 1):
        got = _dequant_tm(layer.w13_tm_weight[ex], layer.w13_tm_scales[ex]).t()
        ref = _dequant_ref(
            w["w13_weight"][ex],
            w["w13_weight_scale"][ex],
            w["w13_weight_scale_2"][ex],
            gi,
        )
        torch.testing.assert_close(got, ref)
        got2 = _dequant_tm(layer.w2_tm_weight[ex], layer.w2_tm_scales[ex]).t()
        ref2 = _dequant_ref(
            w["w2_weight"][ex], w["w2_weight_scale"][ex], w["w2_weight_scale_2"][ex]
        )
        torch.testing.assert_close(got2, ref2)
    for raw in ("w13_weight", "w2_weight_scale_2", "w13_input_scale"):
        assert not hasattr(layer, raw)
    assert layer.sm70_moe_workspace.key.max_tokens == 64


@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("m", [1, 2, 64])
def test_apply_matches_reference(mods, m, indexed):
    m4, tm = mods(indexed=indexed)
    method, layer, w = _build(m4)
    h, gi, e, k = SMALL
    x = (torch.randn(m, h, generator=torch.Generator().manual_seed(m)) * 0.5).half()
    weights, ids = _routing(m, e, k)
    tm.calls.clear()
    TorchMoeOps.allocations = 0
    out = method.apply(layer, x, weights, ids, None, None)
    assert out.dtype == torch.float16 and out.shape == (m, h)
    ref = _moe_reference(x, w, ids, weights, gi)
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )
    kinds = [c[0] for c in tm.calls]
    assert kinds == ["dense", "dense"]  # indexed op never used for Flint
    assert TorchMoeOps.allocations == 0  # no workspace allocation in forward


def test_int64_ids_are_cast_into_workspace(mods):
    m4, _ = mods()
    method, layer, _w = _build(m4)
    weights, ids = _routing(3, SMALL[2], SMALL[3])
    a = method.apply(
        layer, torch.randn(3, SMALL[0]).half(), weights, ids.long(), None, None
    )
    assert torch.isfinite(a).all()


def test_workspace_shared_and_bounded(mods):
    m4, _ = mods()
    _, l1, _ = _build(m4, seed=0)
    _, l2, _ = _build(m4, seed=1)
    assert l1.sm70_moe_workspace is l2.sm70_moe_workspace
    with pytest.raises(RuntimeError, match="exceed the persistent workspace"):
        l1.sm70_moe_workspace.views(65)


def _decode_calls(fm, kind):
    return [c for c in fm.calls if c[0] == kind]


def test_decode_off_when_flint_moe_absent(mods):
    m4, tm = mods(flint=None)
    method, layer, _w = _build(m4)
    assert method.decode_enabled and layer.sm70_nvfp4_decode_plan is None
    assert layer.sm70_nvfp4_decode_ws is None
    x = torch.randn(1, SMALL[0]).half()
    weights, ids = _routing(1, SMALL[2], SMALL[3])
    tm.calls.clear()
    method.apply(layer, x, weights, ids, None, None)
    assert [c[0] for c in tm.calls] == ["dense", "dense"]


def test_admit_layer_gets_contract_banks_and_policy(mods):
    fm = make_fake_flint_moe()
    m4, _tm = mods(flint=fm)
    _method, layer, _ = _build(m4)
    (call,) = _decode_calls(fm, "admit")
    h, gi, e, k = SMALL
    assert call[1] == (e, h, gi, k)  # (experts, hidden, local I, top_k)
    assert call[2] is True  # resolved Sm70Config policy, explicit
    assert call[3] is layer.w13_tm_weight and call[4] is layer.w13_tm_scales
    assert call[5] is layer.w2_tm_weight and call[6] is layer.w2_tm_scales
    assert isinstance(layer.sm70_nvfp4_decode_plan, fm.DecodePlan)
    assert layer.sm70_nvfp4_decode_ws.device == torch.device("cpu")
    # the fork pre-wiring's per-layer partials/counters are gone
    assert not hasattr(layer, "sm70_nvfp4_decode_counters")
    assert not hasattr(layer, "sm70_nvfp4_decode_partials")


def test_plan_none_keeps_dense_route(mods):
    fm = make_fake_flint_moe(admit=False)
    m4, tm = mods(flint=fm)
    method, layer, w = _build(m4)
    assert layer.sm70_nvfp4_decode_plan is None and not fm.allocations
    for m in (1, 2):
        x = torch.randn(m, SMALL[0], generator=torch.Generator().manual_seed(m)).half()
        weights, ids = _routing(m, SMALL[2], SMALL[3])
        tm.calls.clear()
        fm.calls.clear()
        out = method.apply(layer, x, weights, ids, None, None)
        assert [c[0] for c in tm.calls] == ["dense", "dense"]
        assert fm.calls == []  # can_decode is not even asked
        ref = _moe_reference(x, w, ids, weights, SMALL[1])
        torch.testing.assert_close(
            out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
        )


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("m", [1, 2])
def test_decode_route_exact_args_and_no_dense(mods, monkeypatch, m, ids_dtype):
    fm = make_fake_flint_moe()
    m4, tm = mods(flint=fm)
    method, layer, w = _build(m4)
    x = torch.randn(m, SMALL[0], generator=torch.Generator().manual_seed(m)).half()
    weights, ids = _routing(m, SMALL[2], SMALL[3])
    ids = ids.to(ids_dtype)
    zeros = []
    real_zeros = torch.zeros
    tm.calls.clear()
    fm.calls.clear()
    TorchMoeOps.allocations = 0
    with monkeypatch.context() as mp:
        mp.setattr(
            torch, "zeros", lambda *a, **k: zeros.append(a) or real_zeros(*a, **k)
        )
        out = method.apply(layer, x, weights, ids, None, None)
    assert tm.calls == [] and TorchMoeOps.allocations == 0  # no dense stage
    assert zeros == []  # output is torch.empty, fully overwritten by K12
    assert _decode_calls(fm, "can_decode") == [("can_decode", m, ids_dtype)]
    (call,) = _decode_calls(fm, "decode")
    _, d_out, inter, d_x, w13w, w13s, w2w, w2s, d_w, d_ids, ws, plan = call
    assert d_out is out and d_out.shape == (m, SMALL[0])
    assert d_out.dtype == torch.float16
    buf = layer.sm70_moe_workspace.views(m)["intermediate"]
    assert inter.data_ptr() == buf.data_ptr() and inter.shape == buf.shape
    assert torch.equal(d_x, x)
    assert w13w is layer.w13_tm_weight and w13s is layer.w13_tm_scales
    assert w2w is layer.w2_tm_weight and w2s is layer.w2_tm_scales
    assert d_w is weights and d_ids is ids  # int64 passed through, not cast
    assert ws is layer.sm70_nvfp4_decode_ws and plan is layer.sm70_nvfp4_decode_plan
    ref = _moe_reference(x, w, ids, weights, SMALL[1])
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


def test_can_decode_false_takes_dense(mods):
    fm = make_fake_flint_moe()
    m4, tm = mods(flint=fm)
    method, layer, w = _build(m4)
    x = torch.randn(3, SMALL[0]).half()
    weights, ids = _routing(3, SMALL[2], SMALL[3])
    tm.calls.clear()
    fm.calls.clear()
    out = method.apply(layer, x, weights, ids.long(), None, None)
    assert _decode_calls(fm, "can_decode") == [("can_decode", 3, torch.int64)]
    assert _decode_calls(fm, "decode") == []
    assert [c[0] for c in tm.calls] == ["dense", "dense"]
    ref = _moe_reference(x, w, ids, weights, SMALL[1])
    torch.testing.assert_close(
        out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


def test_decode_kill_switch(mods, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    fm = make_fake_flint_moe()
    m4, _tm = mods(flint=fm)
    method, layer, _ = _build(m4)
    assert not method.decode_enabled and not m4.decode_policy_enabled()
    ((_, _, policy, *_),) = _decode_calls(fm, "admit")
    assert policy is False
    assert layer.sm70_nvfp4_decode_plan is None and not fm.allocations


def test_kill_switch_applies_even_if_policy_resolved_on(mods, monkeypatch):
    """admit_layer ignores the env when given a policy; M4 must apply it."""
    fm = make_fake_flint_moe()
    m4, _tm = mods(flint=fm)
    method = m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), _moe_config(*SMALL))
    assert method.decode_enabled
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    cfg = _moe_config(*SMALL)
    layer = _make_layer(method, cfg, _random_experts(SMALL[2], SMALL[0], SMALL[1]))
    method.process_weights_after_loading(layer)
    ((_, _, policy, *_),) = _decode_calls(fm, "admit")
    assert policy is False and layer.sm70_nvfp4_decode_plan is None


def test_config_policy_off_is_not_reenabled_by_env(mods, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "1")
    fm = make_fake_flint_moe()
    m4, _tm = mods(flint=fm)
    method = m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), _moe_config(*SMALL))
    method.decode_enabled = False  # Sm70Config.flint_moe_qpn_decode=False
    layer = _make_layer(
        method, _moe_config(*SMALL), _random_experts(SMALL[2], SMALL[0], SMALL[1])
    )
    method.process_weights_after_loading(layer)
    ((_, _, policy, *_),) = _decode_calls(fm, "admit")
    assert policy is False and layer.sm70_nvfp4_decode_plan is None


def test_decode_workspace_shared_per_device_and_not_in_apply(mods):
    fm = make_fake_flint_moe()
    m4, _tm = mods(flint=fm)
    m1, l1, _ = _build(m4, seed=0)
    _m2, l2, _ = _build(m4, seed=1)
    assert fm.allocations == [torch.device("cpu")]  # once, at load
    assert l1.sm70_nvfp4_decode_ws is l2.sm70_nvfp4_decode_ws
    assert l1.sm70_nvfp4_decode_plan is not None
    for m in (1, 2, 3):
        weights, ids = _routing(m, SMALL[2], SMALL[3])
        m1.apply(l1, torch.randn(m, SMALL[0]).half(), weights, ids, None, None)
    assert len(fm.allocations) == 1  # never lazily in forward


def test_decode_workspace_keyed_by_device(mods):
    fm = make_fake_flint_moe()
    m4, _tm = mods(flint=fm)
    a = m4.decode_workspace(fm, torch.device("cpu"))
    b = m4.decode_workspace(fm, torch.device("meta"))
    assert a is not b and a is m4.decode_workspace(fm, torch.device("cpu"))
    assert fm.allocations == [torch.device("cpu"), torch.device("meta")]


def test_real_flint_moe_refuses_small_contract(mods):
    """With Kai's real module, a non-Flint-TP2 layer is not admitted."""
    from vllm_sm70.ops import flint_moe as real

    m4, _tm = mods(flint=real)
    _method, layer, _ = _build(m4)
    assert layer.sm70_nvfp4_decode_plan is None
    assert layer.sm70_nvfp4_decode_ws is None


def test_no_private_k12_contract_copy():
    """admit_layer owns the contract; M4 keeps no copy of the fork constants."""
    import vllm_sm70.quant.nvfp4_moe as m4

    for name in (
        "DECODE_QPN_CONTRACTS",
        "DECODE_QPN_COUNTERS",
        "DECODE_QPN_PARTIALS",
        "DECODE_W13_OP",
        "decode_scale_shift",
        "decode_ops_available",
    ):
        assert not hasattr(m4, name), name
    assert not hasattr(m4.ModelOptNvFp4Sm70MoEMethod, "_use_decode")


def test_missing_ops_fail_closed(mods, monkeypatch):
    m4, tm = mods()
    del tm.nvfp4_moe_dense_stage_sm70_out
    with pytest.raises(RuntimeError, match="nvfp4_moe_dense_stage_sm70_out"):
        _build(m4)


@pytest.mark.parametrize(
    "field, value",
    [("tp_size", 4), ("tp_size", 8), ("num_experts", 32)],
)
def test_contract_rejects(mods, field, value):
    m4, _ = mods()
    cfg = _moe_config(*SMALL)
    setattr(cfg, field, value)
    with pytest.raises(NotImplementedError):
        m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), cfg)


def test_runner_surface(mods):
    m4, _ = mods()
    method = m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), _moe_config(*SMALL))
    assert not method.is_monolithic and method.moe_kernel is None
    assert not method.supports_internal_mk and not method.supports_eplb
    assert method.topk_indices_dtype is None
    assert method.uses_weight_scale_2_pattern()
    assert method.get_fused_moe_quant_config(None) is None
    assert method.maybe_roundup_sizes(64, 64, torch.float16, None) == (64, 64)


def _routed_layer(cfg):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = cfg
    return layer


@pytest.mark.parametrize(
    "cap, ready", [((7, 0), True), ((7, 0), False), ((8, 0), False)]
)
def test_get_quant_method_routing(mods, device, monkeypatch, cap, ready):
    m4, _ = mods()
    device(*cap, ops=ready)

    class UpstreamSentinel:
        def __init__(self, quant_config, moe_config):
            self.moe_config = moe_config

    monkeypatch.setattr(
        up_modelopt.ModelOptNvFp4Config, "FusedMoEMethodCls", UpstreamSentinel
    )
    cfg = _quant_config()
    method = cfg.get_quant_method(
        _routed_layer(_moe_config(*SMALL)), "model.layers.0.mlp.experts"
    )
    if cap == (7, 0) and ready:
        assert isinstance(method, m4.ModelOptNvFp4Sm70MoEMethod)
    else:
        assert isinstance(method, UpstreamSentinel)
    # excluded layers keep upstream's behaviour (None) even on SM70
    excluded = cfg.get_quant_method(
        _routed_layer(_moe_config(*SMALL)), "model.layers.0.mlp.gate"
    )
    assert excluded is None


def test_register_checks_m4_hook():
    from vllm_sm70.quant import configs

    assert "get_quant_method" in configs.Sm70ModelOptNvFp4Config.__dict__
    configs.register()  # idempotent, raises on a lost hook


# --------------------------------------------------------------------------
# GPU (Bea, c4130)
# --------------------------------------------------------------------------
def _gpu_layer(m4):
    from vllm_sm70.ops import tm

    for name in m4.REQUIRED_OPS:
        assert tm.has_op(name), name
    cfg = _moe_config(2048, 512, 256, 6, tp=2)  # Flint TP2
    method = m4.ModelOptNvFp4Sm70MoEMethod(_quant_config(), cfg)
    w = _random_experts(256, 2048, 256, seed=0)
    layer = _make_layer(method, cfg, w, device="cuda")
    method.process_weights_after_loading(layer)
    return method, layer, w


@pytest.mark.gpu
@pytest.mark.parametrize("m", [1, 2, 7, 64, 512])
def test_gpu_flint_tp2_parity(m):
    import vllm_sm70.quant.nvfp4_moe as m4

    torch.cuda.set_device(0)
    method, layer, w = _gpu_layer(m4)
    g = torch.Generator().manual_seed(m)
    x = (torch.randn(m, 2048, generator=g) * 0.5).half().cuda()
    weights, ids = _routing(m, 256, 6)
    out = method.apply(layer, x, weights.cuda(), ids.cuda(), None, None)
    ref = _moe_reference(x.cpu(), w, ids, weights, 256)
    torch.testing.assert_close(
        out.float().cpu(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )


@pytest.mark.gpu
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("m", [1, 2])
def test_gpu_flint_tp2_decode_matches_dense(monkeypatch, m, ids_dtype):
    """K12 decode route vs the dense-stage route on the same Flint TP2 layer."""
    import vllm_sm70.quant.nvfp4_moe as m4
    from vllm_sm70.ops import flint_moe

    torch.cuda.set_device(0)
    assert flint_moe.available(), "_sm70_C lacks the K12 decode ops"
    method, layer, w = _gpu_layer(m4)
    plan = layer.sm70_nvfp4_decode_plan
    assert plan is not None, "Flint TP2 layer not admitted to the K12 decode"
    used = []
    real = flint_moe.flint_moe_decode_out
    monkeypatch.setattr(
        flint_moe, "flint_moe_decode_out", lambda *a: used.append(1) or real(*a)
    )
    g = torch.Generator().manual_seed(100 + m)
    x = (torch.randn(m, 2048, generator=g) * 0.5).half().cuda()
    weights, ids = _routing(m, 256, 6, seed=m)
    dw, di = weights.cuda(), ids.to(ids_dtype).cuda()
    decode = method.apply(layer, x, dw, di, None, None).clone()
    assert used == [1]
    layer.sm70_nvfp4_decode_plan = None
    dense = method.apply(layer, x, dw, di, None, None)
    layer.sm70_nvfp4_decode_plan = plan
    assert used == [1]
    ref = _moe_reference(x.cpu(), w, ids, weights, 256)
    tol = 2e-2 * ref.abs().max().item()
    torch.testing.assert_close(decode.float().cpu(), ref, rtol=2e-2, atol=tol)
    torch.testing.assert_close(decode.float(), dense.float(), rtol=2e-2, atol=tol)


@pytest.mark.gpu
@pytest.mark.parametrize("m", [1, 2, 8])
def test_gpu_cuda_graph_matches_eager(m):
    import vllm_sm70.quant.nvfp4_moe as m4

    torch.cuda.set_device(0)
    method, layer, _ = _gpu_layer(m4)
    x = (torch.randn(m, 2048) * 0.5).half().cuda()
    weights, ids = _routing(m, 256, 6)
    weights, ids = weights.cuda(), ids.cuda()
    eager = method.apply(layer, x, weights, ids, None, None).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        method.apply(layer, x, weights, ids, None, None)  # warm up off-graph
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, weights, ids, None, None)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_get_quant_method_tp_gate(mods, device, monkeypatch, tp):
    """SM70 MoE is validated at TP1/TP2 only; TP4+ refuses, never falls back."""
    m4, _ = mods()
    device(7, 0, ops=True)

    class UpstreamSentinel:
        def __init__(self, quant_config, moe_config):
            raise AssertionError("must not fall back to upstream")

    monkeypatch.setattr(
        up_modelopt.ModelOptNvFp4Config, "FusedMoEMethodCls", UpstreamSentinel
    )
    layer = _routed_layer(_moe_config(*SMALL, tp=tp))
    if tp in (1, 2):
        method = _quant_config().get_quant_method(layer, "model.layers.0.mlp.experts")
        assert isinstance(method, m4.ModelOptNvFp4Sm70MoEMethod)
    else:
        with pytest.raises(NotImplementedError, match="tensor_parallel_size"):
            _quant_config().get_quant_method(layer, "model.layers.0.mlp.experts")
