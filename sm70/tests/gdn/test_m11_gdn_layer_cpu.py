# SPDX-License-Identifier: Apache-2.0
"""M11 CPU checks for the Qwen GDN layer subclass (vllm_sm70/gdn/layer.py).

What: the subclass is installed over upstream QwenGatedDeltaNetAttention via
  PluggableLayer.register_oot and is what `QwenGatedDeltaNetAttention(...)`
  builds; non-spec decode runs upstream's conv1d update then K08's
  flashqla_decode, falling back to upstream's packed Triton decode with the
  exact arguments upstream passes; VLLM_SM70_GDN_DECODE_FLASHQLA=0 (or a
  build without the op) runs the upstream method untouched; the warmup runs
  upstream's warmup once and then the FlashQLA decode at 1 and 2 tokens.
Needs vLLM's Python source; no GPU, no built _sm70_C.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
up = pytest.importorskip("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")

T, H, HV, D, W = 3, 8, 16, 128, 4  # Qwen3.6-35B-A3B at TP2, conv width 4
QKV = 2 * H * D + HV * D
SWITCH = "VLLM_SM70_GDN_DECODE_FLASHQLA"


@pytest.fixture
def m11(monkeypatch, gdn_stub):
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_sm70.gdn import layer

    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(layer, "_layer_class", None)
    monkeypatch.setattr(layer, "kernels", gdn_stub)
    monkeypatch.delitem(op_registry_oot, "QwenGatedDeltaNetAttention", raising=False)
    cls = layer.register_qwen_gdn_layer()
    yield layer, cls
    op_registry_oot.pop("QwenGatedDeltaNetAttention", None)


def _bare(cls, slots=5):
    """A layer instance with just what the decode/warmup overrides read."""
    obj = up.QwenGatedDeltaNetAttention.__new__(up.QwenGatedDeltaNetAttention)
    assert type(obj) is cls
    torch.nn.Module.__init__(obj)
    obj.conv1d = types.SimpleNamespace(
        weight=torch.randn(QKV, 1, W, dtype=torch.float16), bias=None
    )
    obj.activation = "silu"
    obj.A_log = torch.randn(HV)
    obj.dt_bias = torch.randn(HV, dtype=torch.float16)
    obj.head_k_dim = obj.head_v_dim = D
    obj.num_v_heads, obj.tp_size = HV * 2, 2
    obj.prefix = "model.layers.0.linear_attn"
    obj.kv_cache = (
        torch.zeros(slots, W - 1, QKV, dtype=torch.float16),  # SD layout
        torch.zeros(slots, HV, D, D, dtype=torch.float32),
    )
    obj._prefill_kernels_warmed_up = False
    obj.get_state_dtype = lambda: (torch.float16, torch.float32)
    return obj


def _decode_inputs(tokens=T + 2):
    qkvz = torch.randn(tokens, QKV + HV * D, dtype=torch.float16)
    ba = torch.randn(tokens, 2 * HV, dtype=torch.float16)
    b, a = ba.chunk(2, dim=-1)  # column views, as split_ba() gives
    core = torch.zeros(tokens, HV, D, dtype=torch.float16)
    md = types.SimpleNamespace(
        num_actual_tokens=T,
        non_spec_state_indices_tensor=torch.tensor([1, 2, 3, 0, 0], dtype=torch.int32),
    )
    return qkvz[:, :QKV], b, a, core, md


def _record(monkeypatch, *, admitted):
    calls: dict[str, list] = {"conv": [], "qla": [], "packed": []}

    def conv(x, *args, **kw):
        calls["conv"].append((x, args, kw))
        return x

    def qla(*args, **kw):
        calls["qla"].append((args, kw))
        return admitted

    monkeypatch.setattr(up, "causal_conv1d_update", conv)
    monkeypatch.setattr(
        up,
        "fused_recurrent_gated_delta_rule_packed_decode",
        lambda **kw: calls["packed"].append(kw),
    )
    return calls, qla


# ------------------------------------------------------------ registration


def test_installed_over_upstream_and_idempotent(m11):
    from vllm.model_executor.custom_op import op_registry_oot

    layer, cls = m11
    assert op_registry_oot["QwenGatedDeltaNetAttention"] is cls
    assert issubclass(cls, up.QwenGatedDeltaNetAttention)
    assert layer.register_qwen_gdn_layer() is cls
    obj = up.QwenGatedDeltaNetAttention.__new__(up.QwenGatedDeltaNetAttention)
    assert type(obj) is cls


def test_overrides_only_the_two_methods(m11):
    _, cls = m11
    own = {k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")}
    assert own == {
        "_forward_core_decode_non_spec",
        "_warmup_prefill_kernels",
        "_sm70_warmup_flashqla_decode",
    }


def test_leaves_a_foreign_oot_class_alone(monkeypatch, gdn_stub):
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_sm70.gdn import layer

    monkeypatch.setattr(layer, "_layer_class", None)
    monkeypatch.setitem(op_registry_oot, "QwenGatedDeltaNetAttention", object)
    assert layer.register_qwen_gdn_layer() is None
    assert op_registry_oot["QwenGatedDeltaNetAttention"] is object


def test_gdn_area_registers_the_layer(monkeypatch, gdn_stub):
    import vllm_sm70.gdn as gdn_area
    from vllm_sm70 import _ext
    from vllm_sm70.gdn import layer

    seen = []
    monkeypatch.setattr(gdn_area, "_registered", False)
    monkeypatch.setattr(_ext, "is_loaded", lambda: True)
    monkeypatch.setattr(gdn_stub, "has_op", lambda name: True)
    monkeypatch.setattr(gdn_stub, "register_chunk_gated_delta_rule", lambda: None)
    monkeypatch.setattr(layer, "register_qwen_gdn_layer", lambda: seen.append(1))
    gdn_area.register()
    assert seen == [1]


# ----------------------------------------------------------------- decode


@pytest.mark.parametrize("dim_first", [False, True])
def test_decode_takes_flashqla(monkeypatch, m11, gdn_stub, dim_first):
    _, cls = m11
    obj = _bare(cls)
    calls, qla = _record(monkeypatch, admitted=True)
    monkeypatch.setattr(gdn_stub, "has_op", lambda name: True)
    monkeypatch.setattr(gdn_stub, "flashqla_decode", qla)
    monkeypatch.setattr(up, "is_conv_state_dim_first", lambda: dim_first)
    mixed, b, a, core, md = _decode_inputs()

    assert obj._forward_core_decode_non_spec(mixed, b, a, core, md) is None

    ((x, (conv_state, w, bias, act), kw),) = calls["conv"]
    assert x.shape == (T, QKV) and x.data_ptr() == mixed.data_ptr()
    cs = obj.kv_cache[0]
    assert conv_state.shape == (cs if dim_first else cs.transpose(-1, -2)).shape
    assert w.shape == (QKV, W) and bias is None and act == "silu"
    assert kw["conv_state_indices"].tolist() == [1, 2, 3]
    assert kw["validate_data"] is False

    ((args, kw),) = calls["qla"]
    qx, qa, qb, A_log, dt_bias, ssm, idx, out = args
    assert qx is x
    assert qa.is_contiguous() and qb.is_contiguous()
    assert torch.equal(qa, a[:T]) and torch.equal(qb, b[:T])
    assert A_log is obj.A_log and dt_bias is obj.dt_bias
    assert ssm is obj.kv_cache[1]
    assert idx.tolist() == [1, 2, 3]
    assert out.shape == (T, HV, D) and out.data_ptr() == core.data_ptr()
    assert kw == {"scale": D**-0.5}
    assert calls["packed"] == []


def test_decode_refused_runs_upstream_packed_decode(monkeypatch, m11, gdn_stub):
    """Same calls, same arguments as upstream's own method."""
    _, cls = m11
    obj = _bare(cls)
    monkeypatch.setattr(gdn_stub, "has_op", lambda name: True)
    mixed, b, a, core, md = _decode_inputs()

    calls, qla = _record(monkeypatch, admitted=False)
    monkeypatch.setattr(gdn_stub, "flashqla_decode", qla)
    obj._forward_core_decode_non_spec(mixed, b, a, core, md)
    ours = calls
    assert len(ours["qla"]) == 1

    calls, _ = _record(monkeypatch, admitted=False)
    up.QwenGatedDeltaNetAttention._forward_core_decode_non_spec(
        obj, mixed, b, a, core, md
    )
    ref = calls

    assert len(ours["conv"]) == len(ref["conv"]) == 1
    (x1, a1, k1), (x2, a2, k2) = ours["conv"][0], ref["conv"][0]
    assert torch.equal(x1, x2) and x1.data_ptr() == x2.data_ptr()
    assert all(
        torch.equal(p, q) if torch.is_tensor(p) else p == q for p, q in zip(a1, a2)
    )
    assert k1.keys() == k2.keys()
    assert torch.equal(k1["conv_state_indices"], k2["conv_state_indices"])

    (p1,), (p2,) = ours["packed"], ref["packed"]
    assert p1.keys() == p2.keys()
    for key in p1:
        v1, v2 = p1[key], p2[key]
        if torch.is_tensor(v1):
            assert v1.shape == v2.shape and torch.equal(v1, v2), key
            if key in ("mixed_qkv", "initial_state", "out"):
                assert v1.data_ptr() == v2.data_ptr(), key
        else:
            assert v1 == v2, key


@pytest.mark.parametrize("why", ["switch", "no_op"])
def test_decode_off_is_upstream_method(monkeypatch, m11, gdn_stub, why):
    _, cls = m11
    obj = _bare(cls)
    if why == "switch":
        monkeypatch.setenv(SWITCH, "0")
        monkeypatch.setattr(gdn_stub, "has_op", lambda name: True)
    else:
        monkeypatch.setattr(gdn_stub, "has_op", lambda name: False)
    seen = []
    monkeypatch.setattr(
        up.QwenGatedDeltaNetAttention,
        "_forward_core_decode_non_spec",
        lambda self, *args: seen.append(args),
    )
    calls, qla = _record(monkeypatch, admitted=True)
    monkeypatch.setattr(gdn_stub, "flashqla_decode", qla)
    args = _decode_inputs()
    obj._forward_core_decode_non_spec(*args)
    assert len(seen) == 1 and seen[0][3] is args[3]
    assert calls["conv"] == calls["qla"] == []


# ----------------------------------------------------------------- warmup


def _warmup_setup(monkeypatch, gdn_stub, cls):
    obj = _bare(cls)
    ups = []

    def upstream_warmup(self, qkv_or_qkvz, v_dim):
        ups.append(v_dim)
        self._prefill_kernels_warmed_up = True

    monkeypatch.setattr(
        up.QwenGatedDeltaNetAttention, "_warmup_prefill_kernels", upstream_warmup
    )
    monkeypatch.setattr(gdn_stub, "has_op", lambda name: True)
    qla = []
    monkeypatch.setattr(
        gdn_stub, "flashqla_decode", lambda *a, **k: qla.append((a, k)) or True
    )
    return obj, ups, qla


def test_warmup_runs_upstream_then_flashqla_decode_once(monkeypatch, m11, gdn_stub):
    _, cls = m11
    obj, ups, qla = _warmup_setup(monkeypatch, gdn_stub, cls)
    qkvz = torch.zeros(4, QKV + HV * D, dtype=torch.float16)
    obj._warmup_prefill_kernels(qkvz[:, :QKV], 0)  # fp16 path: strided view
    obj._warmup_prefill_kernels(qkvz[:, :QKV], 0)
    assert ups == [0, 0]  # upstream keeps its own once-flag
    shapes = [(a[0].shape, a[0].stride(0), a[6].tolist()) for a, _ in qla]
    assert shapes == [
        ((1, QKV), QKV, [1]),
        ((1, QKV), QKV + HV * D, [1]),
        ((2, QKV), QKV, [1, 2]),
        ((2, QKV), QKV + HV * D, [1, 2]),
    ]
    for a, k in qla:
        state, out = a[5], a[7]
        assert state.shape == (3, HV, D, D) and state.dtype == torch.float32
        assert out.shape == (a[0].shape[0], HV, D)
        assert k == {"scale": D**-0.5}


def test_warmup_dense_input_warms_one_layout(monkeypatch, m11, gdn_stub):
    _, cls = m11
    obj, _, qla = _warmup_setup(monkeypatch, gdn_stub, cls)
    obj._warmup_prefill_kernels(torch.zeros(4, QKV, dtype=torch.float16), 0)
    assert [(a[0].shape[0], a[0].stride(0)) for a, _ in qla] == [(1, QKV), (2, QKV)]


def test_warmup_kill_switch_skips_decode(monkeypatch, m11, gdn_stub):
    _, cls = m11
    obj, ups, qla = _warmup_setup(monkeypatch, gdn_stub, cls)
    monkeypatch.setenv(SWITCH, "0")
    obj._warmup_prefill_kernels(torch.zeros(4, QKV, dtype=torch.float16), 0)
    assert ups == [0] and qla == []


def test_warmup_failure_is_logged_not_raised(monkeypatch, m11, gdn_stub, caplog):
    _, cls = m11
    obj, _, _ = _warmup_setup(monkeypatch, gdn_stub, cls)

    def boom(*a, **k):
        raise RuntimeError("no kernel image")

    monkeypatch.setattr(gdn_stub, "flashqla_decode", boom)
    with caplog.at_level("WARNING", logger="vllm.sm70"):
        obj._warmup_prefill_kernels(torch.zeros(4, QKV, dtype=torch.float16), 0)
    assert "FlashQLA GDN decode warmup failed" in caplog.text
