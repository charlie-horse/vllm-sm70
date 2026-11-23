# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 expert activation fake-quant (act_quant.py).

CPU, on the tiny checkpoint (models/fixtures/inkling2_tiny/). The oracle is
the S5 spike's fake-quant (``s5_*`` below, the functions of its script
verbatim), applied as the spike applied it: to every expert GEMM input and
post-activation value, NVFP4 in the NVFP4 layers and MXFP8 in the MXFP8
layers. Checked: the quantizers against it and against the float8 cast, the
expert module on its grouped, one-token and row-wise paths (fp32, 1e-5), the
TurboMind route with the pure-torch ``tm`` of test_m12_inkling2_nvfp4_tm_cpu
(both rewrite points reached), the TP2 rank partials, no host sync, the
switch (off: no op added, outputs bit-identical) and the whole model against
the dense oracle with the same fake-quant.
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
import torch.nn.functional as F

from vllm_sm70.config import Inkling2Config, parse_sm70_config
from vllm_sm70.models.inkling2_sm70 import act_quant as aq
from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import moe, moe_tm, quant_config, sm70_configs
from vllm_sm70.models.inkling2_sm70.ops import torch_ref

HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
TINY = MODELS / "fixtures/inkling2_tiny"
sys.path.insert(0, str(MODELS))
sys.path.insert(0, str(HERE))
from inkling2_ref import Inkling2Weights
from inkling2_sync_guard import allowed, no_host_sync
from test_m12_inkling2_nvfp4_tm_cpu import TINY_CONTRACT, make_mock_tm

NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
OPS = safetensors_torch.load_file(TINY / "ops.safetensors")
FORMATS = quant_config.expert_formats(
    json.loads((TINY / "hf_quant_config.json").read_text())
)
NVFP4_LAYER, MXFP8_LAYER = 1, 23
LAYERS = {"nvfp4": NVFP4_LAYER, "mxfp8": MXFP8_LAYER}
SIZES = [1, 2, 8, 20, 64]
TOL = 1e-5  # fp32, max-abs-rel
TM_TOL = 1e-2  # TurboMind NVFP4 against the fp16 torch path
# The same with fake-quant: an fp16 rounding difference before it can move a
# value by a whole NVFP4 step (6-12% of it).
TM_FQ_TOL = 3e-2
MODEL_TOL = 1e-5  # every visit, the final state and the logits
FP32_MAX = 3.4028234663852886e38


# --------------------------------------------------------------------------
# The S5 fake-quant (goldens/scripts/s5.py), verbatim
# --------------------------------------------------------------------------
def s5_e4m3_round(v):
    _, e = torch.frexp(v)
    step = torch.exp2(((e - 1).clamp(min=-6) - 3).float())
    return (torch.round(v / step) * step).clamp(max=448.0)


def s5_e2m1_round(a):
    out = torch.full_like(a, 6.0)
    for bound, val, incl in ((5.0, 4.0, True), (3.5, 3.0, False), (2.5, 2.0, True), (1.75, 1.5, False), (1.25, 1.0, True), (0.75, 0.5, False), (0.25, 0.0, True)):  # fmt: skip
        out = torch.where((a <= bound) if incl else (a < bound), torch.full_like(a, val), out)  # fmt: skip
    return out


def s5_nvfp4_fq(x, return_errors=False):
    m, k = x.shape
    amax = x.abs().amax(1, keepdim=True)
    enc = (1536.0 / amax).clamp(max=FP32_MAX)
    dec = 1.0 / enc
    outer = amax * (1.0 / 1536.0)
    xb = x.view(m, k // 16, 16)
    bmax = xb.abs().amax(-1)
    recs, errs = [], []
    for div in (6.0, 4.0):
        sc = s5_e4m3_round((bmax * (enc * (1.0 / div))).clamp(max=448.0))
        inv = (1.0 / (sc * dec)).clamp(max=FP32_MAX)
        q = s5_e2m1_round((xb * inv[..., None]).abs()) * torch.sign(xb)
        rec = (q * sc[..., None]) * outer[..., None]
        recs.append(rec)
        errs.append((rec - xb).abs().sum(-1))
    choose4 = errs[1] < errs[0]
    out = torch.where(choose4[..., None], recs[1], recs[0]).view(m, k)
    return (out, errs) if return_errors else out


def s5_mxfp8_fq(x):
    m, k = x.shape
    xb = x.view(m, k // 32, 32)
    raw = xb.abs().amax(-1) * (1.0 / 448.0)
    mant, e = torch.frexp(raw)
    e = torch.where(mant > 0.5, e, e - 1)
    scale = torch.exp2(e.float())
    scale = torch.where(raw > 0, scale, torch.ones_like(raw))[..., None]
    v = (xb / scale).clamp(-448.0, 448.0)
    return (s5_e4m3_round(v.abs()) * torch.sign(v) * scale).view(m, k)


def s5_fq(fmt):
    """S5's ``fq``: fp32 in, the input dtype out."""
    fn = s5_nvfp4_fq if fmt == ew.NVFP4 else s5_mxfp8_fq
    return lambda x: fn(x.float()).to(x.dtype)


def s5_experts(x, index, weights, gate_up, down, gate_cap, up_cap, fq):
    """torch_ref.experts with S5's two rewrite points."""
    x = fq(x)
    total = torch.zeros_like(x, dtype=torch.float32)
    for expert in index.unique().tolist():
        token, slot = torch.where(index == expert)
        gate, up = F.linear(x[token], gate_up[expert]).chunk(2, dim=-1)
        act = fq(torch_ref.softcapped_swiglu(gate, up, gate_cap, up_cap))
        out = F.linear(act, down[expert]) * weights[token, slot, None]
        total.index_add_(0, token, out.float())
    return total.to(x.dtype)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(moe_tm.SWITCH, raising=False)
    moe_tm.Workspace.clear()
    yield
    moe_tm.Workspace.clear()


@pytest.fixture
def tm(monkeypatch):
    """The pure-torch tm of the TurboMind route's tests, installed, with the
    tiny shape admitted at TP1 and TP2."""
    monkeypatch.setenv(moe_tm.SWITCH, "1")
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


@pytest.fixture(scope="module")
def oracle_weights():
    return Inkling2Weights(NATIVE)


def build(config, layer, *, tp_size=1, tp_rank=0, dtype=torch.float32, quant=True):
    module = moe.Inkling2Sm70Experts(config, tp_size=tp_size, dtype=dtype)
    layers = ew.load_expert_layers(
        NATIVE, config, tp_size=tp_size, tp_rank=tp_rank, formats=FORMATS
    )
    module.load_expert_layer(layers[layer])
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    module.gate.weight.data.copy_(NATIVE[prefix + "weight"])
    module.gate.bias.data.copy_(NATIVE[prefix + "bias"])
    if quant:
        module.act_quant = aq.ActQuant(FORMATS[layer])
    return module


def hidden(m, dtype=torch.float32):
    x = OPS["router.0.in.hidden_states"]
    return x[torch.arange(m) % x.shape[0]].to(dtype)


def run(module, x, rowwise=False):
    out = torch.full_like(x, float("nan"))
    module(x, out, None, rowwise)
    return out


def reference(module, oracle_weights, layer, x, fq):
    """The S5 expert sum of the module's own routing, on the oracle's
    independent dequant of the checkpoint."""
    _, weights, index = module.route(x)
    gate_up, down = oracle_weights.experts(layer)
    return s5_experts(
        x, index, weights, gate_up.to(x.dtype), down.to(x.dtype),
        module.gate_cap, module.up_cap, fq,
    )  # fmt: skip


def random_rows(seed, m=64, k=256):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) * torch.exp(3 * torch.randn(m, 1, generator=g))
    x[0] = 0.0
    x[1, :32] = 0.0
    x[2, 5] = 1e4
    x[3] = 1e-30
    return x


# --------------------------------------------------------------------------
# The quantizers
# --------------------------------------------------------------------------
def test_e4m3_round_equals_the_float8_cast():
    v = torch.cat(
        [torch.rand(200_000) * 448, torch.rand(20_000) * 2**-6, torch.zeros(1)]
    )
    assert torch.equal(aq.e4m3_round(v), v.to(torch.float8_e4m3fn).float())
    assert torch.equal(aq.e4m3_round(v), s5_e4m3_round(v))


def test_e2m1_round_equals_s5():
    a = torch.cat([torch.rand(100_000) * 8, torch.tensor(ew.E2M1_VALUES)])
    a = torch.cat([a, a.nextafter(torch.tensor(0.0)), a.nextafter(torch.tensor(9.0))])
    assert torch.equal(aq.e2m1_round(a), s5_e2m1_round(a))


@pytest.mark.parametrize("seed", range(4))
def test_mxfp8_equals_s5(seed):
    x = random_rows(seed)
    assert torch.equal(aq.mxfp8_fake_quant(x), s5_mxfp8_fq(x))


@pytest.mark.parametrize("seed", range(4))
def test_nvfp4_equals_s5_but_for_error_ties(seed):
    """Upstream sums a block's error pairwise, S5 with ``sum``: the two pick
    another block scale only where both candidates' errors agree to rounding."""
    x = random_rows(seed)
    got = aq.nvfp4_fake_quant(x)
    want, (err6, err4) = s5_nvfp4_fq(x, return_errors=True)
    differs = (got != want).view(err6.shape + (16,)).any(-1)
    gap = (err4 - err6).abs()[differs]
    assert bool((gap <= 1e-6 * err6[differs]).all())
    assert int(differs.sum()) <= 0.001 * differs.numel()


@pytest.mark.parametrize("fmt", [ew.NVFP4, ew.MXFP8])
def test_fixture_activations_match_s5(fmt):
    x = OPS["router.0.in.hidden_states"]
    assert torch.equal(aq.ActQuant(fmt)(x), s5_fq(fmt)(x))


@pytest.mark.parametrize("fmt", [ew.NVFP4, ew.MXFP8])
def test_fp16_is_quantized_in_fp32_and_cast_back(fmt):
    x = (random_rows(7) * 1e-3).half()
    got = aq.ActQuant(fmt)(x)
    assert got.dtype == torch.float16 and torch.isfinite(got).all()
    assert torch.equal(got, s5_fq(fmt)(x))


@pytest.mark.parametrize("fmt", [ew.NVFP4, ew.MXFP8])
def test_zero_and_tiny_rows_stay_finite(fmt):
    x = random_rows(1)
    got = aq.ActQuant(fmt)(x)
    assert torch.isfinite(got).all()
    assert bool((got[0] == 0).all()) and bool((got[1, :32] == 0).all())


def test_quantization_error_is_the_s5_size():
    """S5 measured a relative RMS error of 8.6-8.8% (NVFP4), 2.7% (MXFP8)."""
    x = torch.randn(512, 2304, generator=torch.Generator().manual_seed(0))
    for fn, low, high in ((aq.nvfp4_fake_quant, 0.07, 0.11), (aq.mxfp8_fake_quant, 0.02, 0.035)):  # fmt: skip
        err = float((fn(x) - x).norm() / x.norm())
        assert low < err < high, (fn.__name__, err)


def test_widths_must_be_whole_blocks():
    aq.ActQuant(ew.NVFP4).check(2304, 576, 16)
    aq.ActQuant(ew.MXFP8).check(2304, 576, 32)
    with pytest.raises(ValueError, match="multiples of 32"):
        aq.ActQuant(ew.MXFP8).check(64, 16)  # the tiny MXFP8 layer at TP2
    with pytest.raises(ValueError, match="no activation fake-quant"):
        aq.ActQuant("bf16")


# --------------------------------------------------------------------------
# The torch expert paths (moe.py)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(LAYERS))
@pytest.mark.parametrize("m", SIZES)
@pytest.mark.parametrize("rowwise", [False, True])
def test_torch_paths_match_s5_reference(config, oracle_weights, case, m, rowwise):
    layer = LAYERS[case]
    module = build(config, layer)
    x = hidden(m)
    want = reference(module, oracle_weights, layer, x, s5_fq(FORMATS[layer]))
    got = run(module, x, rowwise)
    assert max_abs_rel(got, want) <= TOL
    # The switch reaches the path: without it the sum is another one.
    module.act_quant = None
    assert max_abs_rel(run(module, x, rowwise), want) > 1e-3


@pytest.mark.parametrize("case", list(LAYERS))
def test_fp16_torch_path_matches_s5_reference(config, oracle_weights, case):
    layer = LAYERS[case]
    module = build(config, layer, dtype=torch.float16)
    x = hidden(20, torch.float16)
    want = reference(module, oracle_weights, layer, x, s5_fq(FORMATS[layer]))
    for rowwise in (False, True):
        assert max_abs_rel(run(module, x, rowwise), want) <= TM_FQ_TOL


@pytest.mark.parametrize("m", [1, 20])
def test_tp2_partials_quantize_the_rank_slice(config, oracle_weights, m):
    """Each rank's post-activation amax spans its own intermediate slice."""
    x = hidden(m)
    gate_up, down = oracle_weights.experts(NVFP4_LAYER)
    local = gate_up.shape[1] // 4
    for rank in range(2):
        module = build(config, NVFP4_LAYER, tp_size=2, tp_rank=rank)
        units = slice(rank * local, (rank + 1) * local)
        rank_gate_up = torch.cat(
            [gate_up[:, units], gate_up[:, 2 * local :][:, units]], dim=1
        )
        _, weights, index = module.route(x)
        want = s5_experts(
            x, index, weights, rank_gate_up, down[:, :, units],
            module.gate_cap, module.up_cap, s5_fq(ew.NVFP4),
        )  # fmt: skip
        assert max_abs_rel(run(module, x), want) <= TOL


@pytest.mark.parametrize("case", list(LAYERS))
@pytest.mark.parametrize("m,rowwise", [(1, False), (8, True)])
def test_torch_decode_paths_have_no_host_sync(config, case, m, rowwise):
    module = build(config, LAYERS[case])
    _, weights, index = module.route(hidden(m))
    traces = []
    for seed in range(2):
        x = hidden(m) * (1 + seed)
        want = torch.full_like(x, float("nan"))
        module.expert_sum(x, index, weights, want, rowwise)
        got = torch.full_like(x, float("nan"))
        trace = []
        with no_host_sync(trace):
            module.expert_sum(x, index, weights, got, rowwise)
        assert torch.equal(got, want)
        traces.append(trace)
    # Equal ops on equal allocation sizes for other values: graph-replayable.
    assert traces[0] == traces[1]


# --------------------------------------------------------------------------
# Off: bit-identical
# --------------------------------------------------------------------------
def test_off_by_default():
    assert Inkling2Config().expert_act_fake_quant is False
    assert parse_sm70_config(None).inkling2.expert_act_fake_quant is False
    on = parse_sm70_config({"inkling2": {"expert_act_fake_quant": True}})
    assert on.inkling2.expert_act_fake_quant is True


def test_apply_off_returns_the_tensor_itself():
    x = hidden(4)
    assert aq.apply(None, x) is x


@pytest.mark.parametrize("case", list(LAYERS))
@pytest.mark.parametrize("m,rowwise", [(1, False), (8, False), (8, True)])
def test_off_adds_no_op_and_is_bit_identical(config, monkeypatch, case, m, rowwise):
    """Off, the expert sum runs exactly the ops of a module without the hook
    points, so its output is the same bits."""
    module = build(config, LAYERS[case], quant=False)
    assert module.act_quant is None
    x = hidden(m)
    _, weights, index = module.route(x)

    def traced():
        out, trace = torch.full_like(x, float("nan")), []
        with no_host_sync(trace) if (m == 1 or rowwise) else _trace_only(trace):
            module.expert_sum(x, index, weights, out, rowwise)
        return out, trace

    got, trace = traced()
    with monkeypatch.context() as patch:
        patch.setattr(moe.act_quant, "apply", lambda quant, x: x)
        want, bare = traced()
    assert torch.equal(got, want) and trace == bare
    assert not any("frexp" in op or "exp2" in op for op, _ in trace)


class _trace_only:
    """``no_host_sync`` trace for a batch path that reads counts on the host."""

    def __init__(self, trace):
        self.cm = no_host_sync(trace)

    def __enter__(self):
        self.cm.__enter__()
        from inkling2_sync_guard import _state

        _state["exempt"] += 1

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc)


def test_install(config):
    layers = [
        types.SimpleNamespace(
            layer_idx=i,
            mlp=types.SimpleNamespace(
                hidden_size=64, local_intermediate=32, act_quant=0
            ),
        )
        for i in range(24)
    ]
    aq.install(layers, None, enabled=False)
    assert all(lay.mlp.act_quant is None for lay in layers)
    aq.install(layers, FORMATS, enabled=True)
    assert [lay.mlp.act_quant.format for lay in layers] == [
        FORMATS[i] for i in range(24)
    ]
    aq.install(layers, None, enabled=True)
    assert [lay.mlp.act_quant.format for lay in layers] == [ew.NVFP4] * 20 + [ew.MXFP8] * 4  # fmt: skip
    assert FORMATS == {i: ew.NVFP4 if i < 20 else ew.MXFP8 for i in range(24)}


# --------------------------------------------------------------------------
# The TurboMind route (moe_tm.py)
# --------------------------------------------------------------------------
def build_tm(config, *, tp_size=1, tp_rank=0):
    module = build(
        config, NVFP4_LAYER, tp_size=tp_size, tp_rank=tp_rank, dtype=torch.float16
    )
    assert module.tm_route is not None
    return module


@pytest.mark.parametrize("m", [1, 8, 41, 64])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_tm_route_matches_the_torch_path(tm, config, monkeypatch, m, tp_size, tp_rank):
    module = build_tm(config, tp_size=tp_size, tp_rank=tp_rank)
    with monkeypatch.context() as patch:
        patch.setenv(moe_tm.SWITCH, "0")
        oracle = build(
            config, NVFP4_LAYER, tp_size=tp_size, tp_rank=tp_rank, dtype=torch.float16
        )
    x = hidden(m, torch.float16)
    want = run(oracle, x)
    assert max_abs_rel(run(module, x), want) <= TM_FQ_TOL
    oracle.act_quant = module.act_quant = None
    assert max_abs_rel(run(module, x), run(oracle, x)) <= TM_TOL
    assert max_abs_rel(run(oracle, x), want) > TM_FQ_TOL


@pytest.mark.parametrize("m", [1, 41])
def test_tm_route_rewrites_both_stage_inputs(tm, config, m):
    """W13 reads the fake-quantized input rows, W2 the fake-quantized SwiGLU
    output (the stage pair is not fused: the activation runs in torch)."""
    module = build_tm(config)
    quant = module.act_quant
    seen = []

    def spy(x):
        out = quant(x)
        seen.append((x.clone(), out.clone()))
        return out

    module.act_quant = spy
    stage = tm.nvfp4_moe_dense_stage_sm70_out
    inputs = []

    def recording_stage(out, inp, *args):
        inputs.append(inp.clone())
        stage(out, inp, *args)

    module.tm_route.tm = types.SimpleNamespace(
        nvfp4_moe_dense_stage_sm70_out=recording_stage
    )
    x = hidden(m, torch.float16)
    _, weights, index = module.route(x)
    module.expert_sum(x, index, weights, torch.empty_like(x))
    assert len(seen) == 2 and len(inputs) == 2
    (x_in, x_q), (act_in, act_q) = seen
    assert torch.equal(x_in, x) and torch.equal(x_q, quant(x))
    tokens = torch.sort(index.reshape(-1), stable=True).indices // index.shape[1]
    assert torch.equal(inputs[0], x_q[tokens])
    assert torch.equal(inputs[1], act_q) and torch.equal(act_q, quant(act_in))
    assert not torch.equal(act_in, act_q)


@pytest.mark.parametrize("m", [1, 8, 41])
def test_tm_route_has_no_host_sync(tm, config, m):
    module = build_tm(config)
    route = module.tm_route
    route.tm = types.SimpleNamespace(
        nvfp4_moe_dense_stage_sm70_out=allowed(tm.nvfp4_moe_dense_stage_sm70_out)
    )
    x = hidden(m, torch.float16)
    _, weights, index = module.route(x)
    want = torch.full_like(x, float("nan"))
    module.expert_sum(x, index, weights, want)
    got = torch.full_like(x, float("nan"))
    with no_host_sync():
        module.expert_sum(x, index, weights, got)
    assert torch.equal(got, want)


# --------------------------------------------------------------------------
# The whole model
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engines():
    from test_m12_inkling2_model_cpu import Engine

    on = {"inkling2": {"expert_act_fake_quant": True}}
    return {"off": Engine(), "off_explicit": Engine(sm70={"inkling2": {}}), "on": Engine(sm70=on)}  # fmt: skip


@pytest.fixture(scope="module")
def ids():
    from test_m12_inkling2_model_cpu import _raw_config

    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    return torch.randint(0, vocab, (17,), generator=torch.Generator().manual_seed(5))


def _oracle_rows(engine, ids, fake_quant):
    """The dense oracle's [T, visits * H + H + vocab] rows, with S5's
    fake-quant in every expert GEMM when `fake_quant`."""
    import inkling2_ref
    from inkling2_ref import Inkling2Oracle

    oracle = Inkling2Oracle.from_checkpoint(TINY)
    current = {}

    def experts(x, index, weights, gate_up, down, gate_cap, up_cap):
        fq = s5_fq(FORMATS[current["layer"]]) if fake_quant else (lambda t: t)
        return s5_experts(x, index, weights, gate_up, down, gate_cap, up_cap, fq)

    original = Inkling2Weights.experts

    def tracking(self, layer):
        current["layer"] = layer
        return original(self, layer)

    saved = inkling2_ref.ref.experts
    Inkling2Weights.experts = tracking
    inkling2_ref.ref.experts = experts
    try:
        out = oracle.forward(ids[None])
    finally:
        Inkling2Weights.experts = original
        inkling2_ref.ref.experts = saved
    return torch.cat(
        [
            out.visit_states[:, 0].transpose(0, 1).flatten(1),
            out.final_hidden_state[0],
            out.logits[0],
        ],
        dim=1,
    )


def _engine_rows(engine, ids, plan="prefill"):
    import inkling2_harness as hx

    plans = {
        "prefill": hx.prefill_plan("a", len(ids)),
        "decode": hx.decode_plan("a", len(ids), 9),
    }
    return engine.run(plans[plan], {"a": ids})["a"]


def test_model_installs_the_switch(engines):
    for name, engine in engines.items():
        quants = [layer.mlp.act_quant for layer in engine.model.model.layers]
        if name == "on":
            assert [q.format for q in quants] == [FORMATS[i] for i in range(24)]
        else:
            assert quants == [None] * 24


@pytest.mark.parametrize("plan", ["prefill", "decode"])
def test_model_off_is_bit_identical(engines, ids, plan):
    got = _engine_rows(engines["off_explicit"], ids, plan)
    assert torch.equal(got, _engine_rows(engines["off"], ids, plan))


@pytest.mark.parametrize("plan", ["prefill", "decode"])
def test_model_on_matches_the_oracle_with_s5_fake_quant(engines, ids, plan):
    engine = engines["on"]
    got = _engine_rows(engine, ids, plan)
    want = _oracle_rows(engine, ids, fake_quant=True)
    split = engine.num_visits * engine.hidden
    per_visit = (got - want)[:, :split].abs().unflatten(1, (engine.num_visits, -1))
    assert float(per_visit.amax()) <= MODEL_TOL
    assert float((got - want)[:, split:].abs().max()) <= MODEL_TOL
    plain = _oracle_rows(engine, ids, fake_quant=False)
    assert float((got - plain)[:, split + engine.hidden :].abs().max()) > 1e-3
