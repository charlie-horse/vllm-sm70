# SPDX-License-Identifier: Apache-2.0
"""M11 GPU exit test: Qwen GDN layer decode output, K08 vs the upstream path.

What: on SM70, the M11 subclass's `_forward_core_decode_non_spec` (upstream
  conv1d update + K08 FlashQLA decode) against the same method with
  VLLM_SM70_GDN_DECODE_FLASHQLA=0 (upstream conv1d update + packed Triton
  decode), from identical conv/SSM state, at the Qwen3.6 and Qwen3.8 TP2
  shapes with a strided mixed_qkv (the fp16 forward_cuda layout) and padded
  rows/slots: core_attn_out and the touched SSM slots within rel-L2 < 1e-3,
  conv state bit-identical, untouched slots and padding rows unchanged; and
  the FlashQLA route was actually taken. Then the warmup override runs the
  FlashQLA decode without error.

Run from the repo root: pytest -m gpu sm70/tests/gdn/test_m11_gdn_layer_gpu.py
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from _gdn_ref import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

TOL = 1e-3
D, W, T, PAD = 128, 4, 4, 2
SWITCH = "VLLM_SM70_GDN_DECODE_FLASHQLA"


def _layer(cls, h, hv, state_dtype, slots=6):
    up = pytest.importorskip(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    qkv = 2 * h * D + hv * D
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def rnd(*shape, dtype=torch.float16, s=0.1):
        return (torch.randn(*shape, generator=g, device=dev) * s).to(dtype)

    obj = up.QwenGatedDeltaNetAttention.__new__(up.QwenGatedDeltaNetAttention)
    assert type(obj) is cls
    torch.nn.Module.__init__(obj)
    obj.conv1d = types.SimpleNamespace(weight=rnd(qkv, 1, W, s=0.5), bias=None)
    obj.activation = "silu"
    obj.A_log = rnd(hv, dtype=torch.float32, s=1.0)
    obj.dt_bias = rnd(hv, s=1.0)
    obj.head_k_dim = obj.head_v_dim = D
    obj.num_v_heads, obj.tp_size = hv * 2, 2
    obj.prefix = "model.layers.0.linear_attn"
    obj._prefill_kernels_warmed_up = False
    obj.get_state_dtype = lambda: (torch.float16, state_dtype)
    conv = rnd(slots, W - 1, qkv)  # SD layout
    ssm = rnd(slots, hv, D, D, dtype=state_dtype)
    n = T + PAD
    qkvz = rnd(n, qkv + hv * D, s=1.0)
    ba = rnd(n, 2 * hv, s=1.0)
    b, a = ba.chunk(2, dim=-1)
    md = types.SimpleNamespace(
        num_actual_tokens=T,
        non_spec_state_indices_tensor=torch.tensor(
            [1, 2, 3, 5, 0, 0], device=dev, dtype=torch.int32
        ),
    )
    return obj, (conv, ssm), (qkvz, qkv, b.contiguous(), a.contiguous(), md)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "h, hv", [(8, 16), (8, 24)], ids=["qwen3.6-tp2", "qwen3.8-tp2"]
)
def test_decode_matches_upstream_path(monkeypatch, gdn_gpu, h, hv, state_dtype):
    from vllm_sm70.gdn import layer

    monkeypatch.setattr(layer, "kernels", gdn_gpu)
    monkeypatch.delenv(SWITCH, raising=False)
    cls = layer.register_qwen_gdn_layer()
    obj, (conv0, ssm0), (qkvz, qkv, b, a, md) = _layer(cls, h, hv, state_dtype)

    def run(switch_off):
        if switch_off:
            monkeypatch.setenv(SWITCH, "0")
        else:
            monkeypatch.delenv(SWITCH, raising=False)
        obj.kv_cache = (conv0.clone(), ssm0.clone())
        out = torch.zeros(T + PAD, hv, D, device="cuda", dtype=torch.float16)
        # conv1d update writes in place; keep the strided fp16 layout each run.
        obj._forward_core_decode_non_spec(qkvz.clone()[:, :qkv], b, a, out, md)
        torch.cuda.synchronize()
        return out, obj.kv_cache

    taken = []
    real = gdn_gpu.flashqla_decode
    monkeypatch.setattr(
        gdn_gpu,
        "flashqla_decode",
        lambda *x, **k: taken.append(real(*x, **k)) or taken[-1],
    )
    out, (conv, ssm) = run(switch_off=False)
    assert taken == [True], "K08 refused the decode tensors"
    ref_out, (ref_conv, ref_ssm) = run(switch_off=True)
    assert taken == [True]

    used = [1, 2, 3, 5]
    errs = {
        "out": rel_l2(out[:T], ref_out[:T]),
        "ssm": rel_l2(ssm[used], ref_ssm[used]),
    }
    assert all(e < TOL for e in errs.values()), errs
    assert torch.equal(conv, ref_conv)
    assert torch.count_nonzero(out[T:]) == 0
    for s in (0, 4):
        assert torch.equal(ssm[s], ssm0[s]) and torch.equal(ref_ssm[s], ssm0[s])


def test_warmup_runs_flashqla_decode(monkeypatch, gdn_gpu, caplog):
    from vllm_sm70.gdn import layer

    up = pytest.importorskip(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    monkeypatch.setattr(layer, "kernels", gdn_gpu)
    monkeypatch.delenv(SWITCH, raising=False)
    cls = layer.register_qwen_gdn_layer()
    obj, _, (qkvz, qkv, _, _, _) = _layer(cls, 8, 16, torch.float32)
    monkeypatch.setattr(
        up.QwenGatedDeltaNetAttention, "_warmup_prefill_kernels", lambda *a: None
    )
    taken = []
    real = gdn_gpu.flashqla_decode
    monkeypatch.setattr(
        gdn_gpu,
        "flashqla_decode",
        lambda *x, **k: taken.append(real(*x, **k)) or taken[-1],
    )
    with caplog.at_level("WARNING", logger="vllm.sm70"):
        obj._warmup_prefill_kernels(qkvz[:, :qkv], 0)
    torch.cuda.synchronize()
    assert taken == [True] * 4, taken
    assert "warmup failed" not in caplog.text
