# SPDX-License-Identifier: Apache-2.0
"""Tests for M7 dflash2-fp16-range (vllm_sm70/models/qwen/dflash_sm70.py).

What: on CPU, checks that the torch-reference BF16-range emulation matches an
  fp32 reference within BF16 tolerance on inputs where plain FP16 overflows
  to inf. Also covers the bit-exact RNE helper, the o_proj input scale, the
  module swap (parameter names unchanged), the gate (cc / draft dtype /
  runtime dtype), the VLLM_SM70_DFLASH2_EMULATION kill switch, and the
  ModelRegistry wiring with the GPU / _sm70_C probes mocked. The classes are checked
  against UP 4ba787e.
Why: the DFlash2 drafts are BF16-trained but run with --dtype half on V100;
  without the emulation their residual and SwiGLU products leave FP16 range.
Kill-switch: VLLM_SM70_DFLASH2_EMULATION is covered below.

GPU tests (marker ``gpu``, see sm70/pyproject.toml) for Bea on c4130, with
the vllm service stopped
(`systemctl kill --signal=SIGKILL vllm && systemctl stop vllm`), from the
repo root:

  # Triton vs torch-reference parity: one GPU, no _sm70_C ops needed.
  CUDA_VISIBLE_DEVICES=0 python -m pytest -q -p no:cacheprovider \\
      sm70/tests/models/test_m7_dflash_sm70.py -m gpu -k triton

  # Real target + draft end to end: both GPUs (TP2), the built _sm70_C
  # extension, and the SM70 FP8 path for the target (M6).
  DFLASH2_E2E_TARGET=/var/lib/vllm/models/Qwen3.8-27B-FP8 \\
  DFLASH2_E2E_DRAFT=/var/lib/vllm/models/Qwen3.8-27B-DFlash2 \\
      python -m pytest -q -p no:cacheprovider \\
      sm70/tests/models/test_m7_dflash_sm70.py -m gpu -k e2e
"""

import os
from types import SimpleNamespace

import pytest

# Collected without vllm/torch installed (e.g. plain `pytest sm70/tests`): skip.
pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch
import torch.nn.functional as F
from torch import nn

import vllm_sm70.platform as sm70_platform
from vllm_sm70.models.qwen import dflash_sm70 as m7

FP16_MAX = torch.finfo(torch.float16).max
BF16_RTOL = 2e-2  # a few BF16 ulps (2^-8 = 3.9e-3) accumulated over 3-4 roundings


def _close(actual: torch.Tensor, expected: torch.Tensor, rtol: float = BF16_RTOL):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    err = (actual - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert err <= rtol * scale, f"max err {err} vs scale {scale}"


@pytest.fixture(autouse=True)
def _no_kill(monkeypatch):
    monkeypatch.delenv(m7.KILL_SWITCH, raising=False)


# --------------------------------------------------------------------------
# Numerics (CPU, torch reference path)
# --------------------------------------------------------------------------
def test_round_rne_matches_torch_bf16_cast():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, generator=g) * torch.logspace(-20, 30, 4096)
    x = torch.cat([x, torch.tensor([0.0, -0.0, 1.0, 65504.0, 3.0e38])])
    assert torch.equal(m7.round_fp32_to_bf16_rne(x), x.to(torch.bfloat16).float())


class _FakeLinear(nn.Module):
    """Returns (y, None) like vllm's parallel linears; fp16 weights."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.half(), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight), None


def _mlp(hidden=16, inter=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(2 * inter, hidden, generator=g) * 25.0
    w2 = torch.randn(hidden, inter, generator=g) * 0.05
    mlp = nn.Module()
    mlp.gate_up_proj = _FakeLinear(w13)
    mlp.down_proj = _FakeLinear(w2)
    return mlp, w13, w2


def test_swiglu_overflows_plain_fp16_but_not_emulation():
    mlp, w13, w2 = _mlp()
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(4, 16, generator=g) * 2.0).half()

    # Plain FP16 (what upstream does with --dtype half): inf.
    gate_up = F.linear(x, w13.half())
    gate, up = gate_up.chunk(2, dim=-1)
    plain = F.linear(F.silu(gate) * up, w2.half())
    assert not torch.isfinite(plain).all()

    # fp32 reference of the true value, then the /256 transport.
    g32, u32 = F.linear(x.float(), w13.float()).chunk(2, dim=-1)
    ref = F.linear(F.silu(g32) * u32, w2.float()) / m7.DFLASH_SM70_WIDE_OUTPUT_SCALE
    assert (F.silu(g32) * u32).abs().max() > FP16_MAX  # the overflow is real

    out = m7.DFlashSM70MLP(mlp)(x)
    assert out.dtype == torch.float16
    _close(out, ref)


def test_silu_and_mul_reference_row_scales_are_powers_of_two():
    gate_up = torch.tensor([[300.0, 1.0, 300.0, 1.0], [0.1, 0.2, 0.3, 0.4]]).half()
    out, scales = m7.dflash_silu_and_mul_sm70(gate_up / 32)
    assert out.dtype == torch.float16 and torch.isfinite(out).all()
    assert torch.equal(torch.log2(scales), torch.round(torch.log2(scales)))
    assert scales[0] > 1 and scales[1] == 1
    assert out.float().abs().max() <= 32752.0


def test_rmsnorm_residual_beyond_fp16_range():
    g = torch.Generator().manual_seed(2)
    hidden = 64
    norm = m7.DFlashSM70RMSNorm(hidden, 1e-6, torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(hidden, generator=g) + 0.5)
    residual = torch.randn(3, hidden, generator=g) * 1.5e5  # FP32 stream
    sub_true = torch.randn(3, hidden, generator=g) * 2.0e4
    x = (sub_true / m7.DFLASH_SM70_WIDE_OUTPUT_SCALE).half()

    # Plain FP16 residual add overflows.
    assert not torch.isfinite(residual.half() + sub_true.half()).all()

    out, res_out = norm(x, residual)
    value = x.float() * m7.DFLASH_SM70_WIDE_OUTPUT_SCALE + residual
    ref = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    ref = ref * norm.weight.float()
    assert out.dtype == torch.float16 and res_out.dtype == torch.float32
    _close(out, ref)
    _close(res_out, value, rtol=1e-2)
    # The residual stream is exactly BF16-representable.
    assert torch.equal(res_out, res_out.to(torch.bfloat16).float())


def test_rmsnorm_no_residual_and_hidden_norm_scale():
    x = torch.randn(2, 32).half()
    norm = m7.DFlashSM70RMSNorm(32, 1e-6, torch.float16, residual_scale=1.0)
    out = norm(x)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float16
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    _close(out, ref)


def test_layer_chain_stays_finite_where_plain_fp16_overflows():
    """Toy post-attention norm -> MLP -> final norm, true values > 65504."""
    mlp, w13, w2 = _mlp(hidden=16, inter=8, seed=3)
    post = m7.DFlashSM70RMSNorm(16, 1e-6, torch.float16)
    final = m7.DFlashSM70RMSNorm(16, 1e-6, torch.float16)
    g = torch.Generator().manual_seed(4)
    residual = (torch.randn(2, 16, generator=g) * 1.0e4).clamp(-6e4, 6e4).half()
    attn_true = torch.randn(2, 16, generator=g) * 1.5e5  # beyond FP16 range
    h, res = post((attn_true / 256).half(), residual)
    h = m7.DFlashSM70MLP(mlp)(h)
    out, res = final(h, res)
    assert torch.isfinite(out).all() and torch.isfinite(res).all()
    assert res.abs().max() > FP16_MAX  # the stream really left FP16 range
    assert not torch.isfinite(attn_true.half()).all()  # plain FP16 transport: inf


# --------------------------------------------------------------------------
# Attention o_proj scale and module swap
# --------------------------------------------------------------------------
class _StubAttention(nn.Module):
    """Has the attributes UP DFlashQwen3Attention.forward uses (qwen3_dflash.py:268-295)."""

    def __init__(self, heads=2, kv_heads=1, head_dim=4):
        super().__init__()
        self.head_dim, self.q_size, self.kv_size = head_dim, heads * head_dim, kv_heads * head_dim
        self.v_scale = None
        self.qkv_proj = _FakeLinear(torch.eye(self.q_size + 2 * self.kv_size, 8))
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.rotary_emb = lambda pos, q, k: (q, k)
        self.attn = lambda q, k, v: torch.full_like(q, 512.0)
        self.seen = None
        outer = self

        class _O(nn.Module):
            def forward(self, x):
                outer.seen = x
                return x, None

        self.o_proj = _O()

    def forward(self, positions, hidden_states):  # upstream would be unscaled
        raise AssertionError("SM70 forward must be used")


def test_attention_cls_scales_o_proj_input():
    attn = _StubAttention()
    attn.output_input_scale = m7.DFLASH_SM70_WIDE_OUTPUT_SCALE
    attn.__class__ = m7.sm70_attention_cls(_StubAttention)
    assert isinstance(attn, _StubAttention)
    assert m7.sm70_attention_cls(type(attn)) is type(attn)  # idempotent
    attn(torch.zeros(3), torch.randn(3, 8).half())
    assert torch.all(attn.seen == 2.0)  # 512 / 256


def test_apply_layer_emulation_keeps_param_names():
    layer = nn.Module()
    layer.self_attn = _StubAttention()
    layer.mlp, _, _ = _mlp()
    layer.input_layernorm = nn.LayerNorm(16)
    layer.post_attention_layernorm = nn.LayerNorm(16)
    names_before = {
        n for n, _ in layer.named_parameters() if "layernorm" not in n
    }
    cfg = SimpleNamespace(hidden_size=16, rms_norm_eps=1e-6)
    m7.apply_layer_emulation(layer, cfg, torch.float16)
    names_after = {n for n, _ in layer.named_parameters()}
    assert names_before <= names_after
    assert {"input_layernorm.weight", "post_attention_layernorm.weight"} <= names_after
    assert isinstance(layer.mlp, m7.DFlashSM70MLP)
    assert isinstance(layer.input_layernorm, m7.DFlashSM70RMSNorm)
    assert layer.self_attn._sm70_bf16_attention
    assert layer.self_attn.output_input_scale == 256.0


def test_apply_model_emulation_norms():
    model = nn.Module()
    m7.apply_model_emulation(model, SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6), torch.float16)
    assert model.hidden_norm.residual_scale == 1.0
    assert model.norm.residual_scale == 256.0


# --------------------------------------------------------------------------
# Gate and kill switch
# --------------------------------------------------------------------------
BF16_CFG = SimpleNamespace(dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "config, runtime, cap, expected",
    [
        (BF16_CFG, torch.float16, 70, True),
        (BF16_CFG, torch.float16, 75, True),  # Turing has no BF16 either
        (BF16_CFG, torch.float16, 80, False),
        (BF16_CFG, torch.bfloat16, 70, False),
        (SimpleNamespace(dtype=torch.float16), torch.float16, 70, False),
        (SimpleNamespace(dtype=None, torch_dtype="bfloat16"), torch.float16, 70, True),
        (SimpleNamespace(dtype="bfloat16"), torch.float16, 70, True),
        (BF16_CFG, torch.float16, None, False),  # no CUDA device
    ],
)
def test_gate(config, runtime, cap, expected):
    assert m7.emulation_enabled(config, runtime, capability=cap) is expected


def test_gate_real_qwen3_config():
    from transformers import Qwen3Config

    assert m7.draft_declared_bf16(Qwen3Config(dtype="bfloat16"))
    assert not m7.draft_declared_bf16(Qwen3Config(dtype="float16"))


@pytest.mark.parametrize(
    "value, expected", [("0", False), ("1", True), ("yes", True), (" 0", True)]
)
def test_kill_switch(monkeypatch, value, expected):
    monkeypatch.setenv(m7.KILL_SWITCH, value)
    assert m7.emulation_enabled(BF16_CFG, torch.float16, capability=70) is expected


# --------------------------------------------------------------------------
# Registry wiring and upstream class hooks (UP 4ba787e)
# --------------------------------------------------------------------------
@pytest.fixture
def registry():
    from vllm.model_executor.models import ModelRegistry

    saved = ModelRegistry.models[m7.DRAFT_ARCH]
    yield ModelRegistry
    ModelRegistry.models[m7.DRAFT_ARCH] = saved


def _ready(monkeypatch, sm70_gpus: bool, ext_loaded: bool):
    from vllm_sm70 import _ext

    monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: sm70_gpus)
    monkeypatch.setattr(_ext, "is_loaded", lambda: ext_loaded)


def test_register_raises_on_sm70_without_ext(monkeypatch, registry, caplog):
    before = registry.models[m7.DRAFT_ARCH]
    _ready(monkeypatch, True, False)
    with caplog.at_level("WARNING", logger=m7.logger.name):
        with pytest.raises(RuntimeError, match="_sm70_C is not loaded"):
            m7.register()
    assert "_sm70_C is not loaded" in caplog.text
    assert registry.models[m7.DRAFT_ARCH] is before


@pytest.mark.parametrize("ext_loaded", [False, True])
def test_register_skips_quietly_off_sm70(monkeypatch, registry, ext_loaded, caplog):
    before = registry.models[m7.DRAFT_ARCH]
    _ready(monkeypatch, False, ext_loaded)
    with caplog.at_level("DEBUG", logger=m7.logger.name):
        assert m7.register() is False
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]
    assert registry.models[m7.DRAFT_ARCH] is before


def test_register_skips_on_kill_even_without_ext(monkeypatch, registry, caplog):
    before = registry.models[m7.DRAFT_ARCH]
    _ready(monkeypatch, True, False)
    monkeypatch.setenv(m7.KILL_SWITCH, "0")
    with caplog.at_level("INFO", logger=m7.logger.name):
        assert m7.register() is False
    assert "not registered" in caplog.text
    assert registry.models[m7.DRAFT_ARCH] is before


def test_register_when_ready(monkeypatch, registry):
    _ready(monkeypatch, True, True)
    assert m7.register() is True
    entry = registry.models[m7.DRAFT_ARCH]
    assert (entry.module_name, entry.class_name) == (
        "vllm_sm70.models.qwen.dflash_sm70",
        "DFlash2SM70Qwen3ForCausalLM",
    )
    cls = registry._try_load_model_cls(m7.DRAFT_ARCH)
    assert cls is m7.DFlash2SM70Qwen3ForCausalLM


def test_not_registered_when_not_ready(monkeypatch, registry):
    before = registry.models[m7.DRAFT_ARCH]
    _ready(monkeypatch, False, False)
    assert m7.register() is False
    assert registry.models[m7.DRAFT_ARCH] is before
    assert before.class_name == "DFlash2Qwen3ForCausalLM"  # UP registry.py:666


def test_not_registered_when_killed(monkeypatch, registry):
    before = registry.models[m7.DRAFT_ARCH]
    _ready(monkeypatch, True, True)
    monkeypatch.setenv(m7.KILL_SWITCH, "0")
    assert m7.register() is False
    assert registry.models[m7.DRAFT_ARCH] is before


def test_area_register_idempotent_and_retries_after_failure(monkeypatch):
    import vllm_sm70.models as models_area
    from vllm_sm70.models import qwen

    calls = []

    def boom():
        calls.append("boom")
        raise RuntimeError("registry changed")

    monkeypatch.setattr(models_area, "_registered", False)
    monkeypatch.setattr(qwen, "_registered", False)
    monkeypatch.setattr(m7, "register", boom)
    with pytest.raises(RuntimeError):
        models_area.register()
    assert not models_area._registered and not qwen._registered

    monkeypatch.setattr(m7, "register", lambda: calls.append("ok") or True)
    models_area.register()
    models_area.register()  # idempotent
    assert calls == ["boom", "ok"]
    assert models_area._registered and qwen._registered


def test_register_raises_if_registry_ignores_override(monkeypatch, registry):
    _ready(monkeypatch, True, True)
    monkeypatch.setattr(registry, "register_model", lambda arch, cls: None)
    with pytest.raises(RuntimeError, match="DFlash2DraftModel"):
        m7.register()


def test_subclasses_use_upstream_factories():
    from vllm.model_executor.models import qwen3_dflash2 as up

    lm, model, layer = (
        m7.DFlash2SM70Qwen3ForCausalLM,
        m7.DFlash2SM70Qwen3Model,
        m7.DFlash2SM70Qwen3DecoderLayer,
    )
    assert issubclass(lm, up.DFlash2Qwen3ForCausalLM)
    assert issubclass(model, up.DFlash2Qwen3Model)
    assert issubclass(layer, up.DFlash2Qwen3DecoderLayer)
    assert lm.model_cls is model and model.decoder_layer_cls is layer
    # The upstream hooks we rely on still exist.
    assert up.DFlash2Qwen3Model.decoder_layer_cls is up.DFlash2Qwen3DecoderLayer
    assert up.DFlash2Qwen3ForCausalLM.model_cls is up.DFlash2Qwen3Model
    assert hasattr(up.DFlash2Qwen3Model, "_project_context_kv")


def test_attention_forward_copy_matches_upstream_source():
    """Guards the 28-line copy of UP DFlashQwen3Attention.forward on rebase."""
    import inspect

    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention

    up_src = inspect.getsource(DFlashQwen3Attention.forward)
    for needle in (
        "self.qkv_proj(hidden_states)",
        "self.rotary_emb(positions, q, k)",
        "v = v * self.v_scale",
        "attn_output = self.attn(q, k, v)",
        "output, _ = self.o_proj(attn_output)",
    ):
        assert needle in up_src
    assert up_src.count("\n") < 35, "upstream forward grew; re-sync the SM70 copy"


# --------------------------------------------------------------------------
# GPU (Bea, c4130)
# --------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("rows, cols", [(1, 8704), (8, 8704), (32, 4352)])
def test_triton_matches_reference_gpu(rows, cols):
    # On a GPU host Triton must be there: fail, don't skip.
    assert m7.HAS_TRITON
    g = torch.Generator(device="cuda").manual_seed(0)
    gate_up = (torch.randn(rows, 2 * cols, device="cuda", generator=g) * 20).half()
    out, s = m7.dflash_silu_and_mul_sm70(gate_up)
    ref_out, ref_s = m7.silu_and_mul_reference(gate_up.cpu())
    torch.testing.assert_close(s.cpu(), ref_s.float(), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), ref_out, rtol=1e-3, atol=1e-3)

    y = (torch.randn(rows, 5120, device="cuda", generator=g) * 100).half()
    rs = torch.pow(2.0, torch.randint(0, 6, (rows,), device="cuda").float())
    ref_y = m7.scale_output_reference(y.cpu(), rs.cpu())
    torch.testing.assert_close(m7.dflash_scale_output_sm70(y.clone(), rs).cpu(), ref_y)

    norm = m7.DFlashSM70RMSNorm(5120, 1e-6, torch.float16).cuda()
    x = (torch.randn(rows, 5120, device="cuda", generator=g) * 50).half()
    res = torch.randn(rows, 5120, device="cuda", generator=g) * 1e5
    o, r = norm(x, res)
    ro, rr = m7.rmsnorm_reference(x.cpu(), norm.weight.cpu(), 1e-6, res.cpu())
    torch.testing.assert_close(r.cpu(), rr, rtol=0, atol=0)
    torch.testing.assert_close(o.cpu(), ro, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(
    not (os.environ.get("DFLASH2_E2E_TARGET") and os.environ.get("DFLASH2_E2E_DRAFT")),
    reason="set DFLASH2_E2E_TARGET and DFLASH2_E2E_DRAFT",
)
def test_e2e_real_draft_gpu():
    """Serve the real target + DFlash2 draft (TP2, fp16) and check acceptance."""
    import json

    from vllm import LLM, SamplingParams

    assert m7.register(), "plugin must register on SM70 (is_sm70_ready)"
    llm = LLM(
        model=os.environ["DFLASH2_E2E_TARGET"],
        tensor_parallel_size=2,
        dtype="half",
        max_model_len=8192,
        gpu_memory_utilization=0.88,
        # LLM() defaults disable_log_stats=True; get_metrics() below needs stats.
        disable_log_stats=False,
        speculative_config=json.loads(
            json.dumps(
                {
                    "method": "dflash",
                    "model": os.environ["DFLASH2_E2E_DRAFT"],
                    "num_speculative_tokens": 7,
                }
            )
        ),
    )
    out = llm.generate(
        ["Write a Python function that merges two sorted lists."],
        SamplingParams(temperature=0.0, max_tokens=256),
    )
    text = out[0].outputs[0].text
    assert len(text) > 50 and "�" not in text
    metrics = {m.name: m for m in llm.get_metrics()}
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens")
    assert accepted is not None and accepted.value > 0
