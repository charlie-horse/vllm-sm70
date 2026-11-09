# SPDX-License-Identifier: Apache-2.0
"""A2 CPU tests: DFlash2 draft candidate top-k by QPN8 support search + rerank.

No GPU or built _sm70_C: K09's prepare and GEMM are CPU references (prepare
keeps the E4M3 weight and scales; the GEMM dequantizes and multiplies in
FP32, rounding to FP16 like the kernel's output), the SM70 device check is
patched, and the exact dots take the torch reference. Covers the chunking,
admission and the kill switch, the chunked quantizer, prepare-once
semantics (never inside a graph or a compile), the selection against an
FP32 dense top-k, the scale and soft cap, the M limit and the gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from _dflash2_helpers import qwen_vllm_config as _vllm_config  # noqa: E402
from torch import nn  # noqa: E402
from vllm_sm70.ops import dflash2_fp8, dflash2_rerank  # noqa: E402

VERIFIER = "VLLM_SM70_DFLASH2_VERIFIER"
SWITCH = dflash2_rerank.KILL_SWITCH


# --------------------------------------------------------------- references
def _ref_prepare(quantized: torch.Tensor, scales: torch.Tensor):
    assert quantized.dtype == torch.float8_e4m3fn and scales.shape[1] == 1
    return [quantized.clone(), scales.clone()]


def _ref_gemm(out, x, codes, scales, split_k, chains, fast, prefetch):
    assert (split_k, chains, fast, prefetch) == (
        dflash2_rerank.SPLIT_K,
        dflash2_rerank.ACCUMULATOR_CHAINS,
        False,
        False,
    )
    weight = codes.float() * scales.float()
    out.copy_((x.float() @ weight.t()).to(out.dtype))


@pytest.fixture
def cpu_qpn8(monkeypatch):
    calls = {"prepare": 0, "gemm": 0}

    def prepare(*a):
        calls["prepare"] += 1
        return _ref_prepare(*a)

    def gemm(*a):
        calls["gemm"] += 1
        _ref_gemm(*a)

    monkeypatch.setattr(dflash2_rerank.qpn8, "fp8_qpn8_prepare_sm70", prepare)
    monkeypatch.setattr(dflash2_rerank.qpn8, "fp8_qpn8_gemm_sm70_out", gemm)
    monkeypatch.setattr(dflash2_rerank, "ops_available", lambda: True)
    monkeypatch.setattr(dflash2_fp8, "_is_sm70", lambda device: True)
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.delenv(VERIFIER, raising=False)
    return calls


def _head(n: int, k: int, *, seed: int = 0, start: int = 0) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    head = nn.Module()
    head.weight = nn.Parameter(
        (torch.randn(n, k, generator=g) / k**0.5).half(), requires_grad=False
    )
    head.bias = None
    head.tp_size = 1
    head.shard_indices = SimpleNamespace(
        num_org_vocab_padding=0, num_added_elements=0, org_vocab_start_index=start
    )
    return head


def _processor(scale: float = 1.0, soft_cap: float | None = None):
    return SimpleNamespace(
        scale=scale, soft_cap=soft_cap, logits_as_input=False, head_dtype=None
    )


def _hidden(m: int, k: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=g).half()


def _dense_topk(x, head, k):
    logits = x.float() @ head.weight.float().t()
    return torch.topk(logits, k, dim=-1)


# ----------------------------------------------------------------- chunking
@pytest.mark.parametrize(
    "rows,groups,candidates",
    [
        (124160, ((0, 62080), (62080, 124160)), 128),  # Qwen3.8 TP2 shard
        (62080, ((0, 62080),), 64),  # TP4 shard
        (100, ((0, 100),), 64),
        (64, ((0, 64),), 64),
    ],
)
def test_vocab_chunks(rows, groups, candidates):
    assert dflash2_rerank.vocab_groups(rows) == groups
    assert dflash2_rerank.num_candidates(rows) == candidates


def test_wide_shard_keeps_64_per_chunk():
    assert dflash2_rerank.num_candidates(248320) == 256


# ------------------------------------------------------------ admission
def test_admission_reasons(cpu_qpn8, monkeypatch):
    lp = _processor()
    assert dflash2_rerank.admission(_head(128, 256), lp, 16) is None
    bias = _head(128, 256)
    bias.bias = torch.zeros(128, dtype=torch.float16)
    assert dflash2_rerank.admission(bias, lp, 16) == "has bias"
    fp32 = _head(128, 256)
    fp32.weight = nn.Parameter(fp32.weight.float(), requires_grad=False)
    assert "fp16" in dflash2_rerank.admission(fp32, lp, 16)
    assert "shape" in dflash2_rerank.admission(_head(100, 256), lp, 16)
    assert "shape" in dflash2_rerank.admission(_head(128, 200), lp, 16)
    assert "shape" in dflash2_rerank.admission(_head(32, 256), lp, 16)
    padded = _head(128, 256)
    padded.shard_indices.num_org_vocab_padding = 8
    assert "padded" in dflash2_rerank.admission(padded, lp, 16)
    assert dflash2_rerank.admission(
        _head(128, 256), SimpleNamespace(**{**vars(lp), "logits_as_input": True}), 16
    )
    assert "head_dtype" in dflash2_rerank.admission(
        _head(128, 256),
        SimpleNamespace(**{**vars(lp), "head_dtype": torch.float32}),
        16,
    )
    assert "scale" in dflash2_rerank.admission(_head(128, 256), _processor(-1.0), 16)
    assert "top_k" in dflash2_rerank.admission(_head(128, 256), lp, 65)
    monkeypatch.setattr(dflash2_fp8, "_is_sm70", lambda device: False)
    assert "SM70" in dflash2_rerank.admission(_head(128, 256), lp, 16)
    monkeypatch.setattr(dflash2_rerank, "ops_available", lambda: False)
    assert "QPN8" in dflash2_rerank.admission(_head(128, 256), lp, 16)


def test_switch_registered_and_only_zero_kills(monkeypatch):
    from vllm_sm70 import envs

    assert SWITCH == "VLLM_SM70_DFLASH2_QPN8_RERANK"
    assert SWITCH in envs._SECTIONS["ops"]
    for value, on in ((None, True), ("1", True), ("false", True), ("0", False)):
        if value is None:
            monkeypatch.delenv(SWITCH, raising=False)
        else:
            monkeypatch.setenv(SWITCH, value)
        assert dflash2_rerank.enabled() is on


# ------------------------------------------------------------- quantizer
def test_chunked_quantizer_matches_one_shot(monkeypatch):
    monkeypatch.setattr(dflash2_rerank, "QUANT_CHUNK_ROWS", 48)
    w = _head(160, 128, seed=3).weight
    q, s = dflash2_rerank.channel_fp8_quantize_chunked(w)
    q_ref, s_ref = dflash2_fp8.channel_fp8_quantize(w)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)


# ---------------------------------------------------------------- prepare
def test_prepares_once_on_first_call_whatever_m(cpu_qpn8):
    head, lp = _head(256, 128), _processor()
    # The profile run drafts many requests: M > 8 prepares but stays dense.
    assert dflash2_rerank.maybe_candidates(head, lp, _hidden(70, 128), 16) is None
    state = getattr(head, dflash2_rerank.STATE_ATTR)
    assert cpu_qpn8["prepare"] == 1 and cpu_qpn8["gemm"] == 0
    assert state.approx.shape == (8, 256) and state.cand_ids.shape == (8, 64)
    assert dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 128), 16) is not None
    assert cpu_qpn8["prepare"] == 1 and cpu_qpn8["gemm"] == 1


def test_never_prepares_inside_capture_or_compile(cpu_qpn8, monkeypatch):
    head, lp = _head(256, 128), _processor()
    monkeypatch.setattr(dflash2_rerank, "_capturing", lambda: True)
    assert dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 128), 16) is None
    monkeypatch.setattr(dflash2_rerank, "_capturing", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 128), 16) is None
    assert not hasattr(head, dflash2_rerank.STATE_ATTR)
    assert cpu_qpn8["prepare"] == 0


def test_declined_head_is_not_retried(cpu_qpn8, caplog):
    head, lp = _head(100, 128), _processor()  # N % 32 != 0
    with caplog.at_level("INFO", logger="vllm.sm70.dflash2"):
        for _ in range(3):
            assert (
                dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 128), 16) is None
            )
    assert sum("not admitted" in r.message for r in caplog.records) == 1
    assert cpu_qpn8["prepare"] == 0


# -------------------------------------------------------------- selection
@pytest.mark.parametrize("m", [1, 7, 8])
def test_selection_matches_fp32_dense_topk(cpu_qpn8, monkeypatch, m):
    # Small chunks so several vocabulary groups are merged, as at TP2.
    monkeypatch.setattr(dflash2_rerank, "VOCAB_CHUNK", 320)
    head, lp = _head(1024, 256, seed=m), _processor()
    x = _hidden(m, 256, seed=10 + m)
    ids, values = dflash2_rerank.maybe_candidates(head, lp, x, 16)
    ref_values, ref_ids = _dense_topk(x, head, 16)
    state = getattr(head, dflash2_rerank.STATE_ATTR)
    assert len(state.groups) == 4 and state.cand_ids.shape[1] == 4 * 64
    assert ids.dtype == torch.int64 and values.dtype == torch.float32
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(values, ref_values, rtol=1e-5, atol=1e-5)


def test_global_ids_add_the_shard_start(cpu_qpn8):
    head, lp = _head(256, 128, start=124160), _processor()
    x = _hidden(7, 128)
    ids, _ = dflash2_rerank.maybe_candidates(head, lp, x, 16)
    assert torch.equal(ids - 124160, _dense_topk(x, head, 16)[1])


def test_scale_and_soft_cap_follow_upstream(cpu_qpn8):
    head, x = _head(256, 128), _hidden(7, 128)
    ids, values = dflash2_rerank.maybe_candidates(head, _processor(0.5, 3.0), x, 16)
    ref_values, ref_ids = _dense_topk(x, head, 16)
    assert torch.equal(ids, ref_ids)
    expected = torch.tanh(ref_values * 0.5 / 3.0) * 3.0
    torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-5)


def test_rows_outside_one_request_fall_back(cpu_qpn8):
    head, lp = _head(256, 128), _processor()
    assert dflash2_rerank.maybe_candidates(head, lp, _hidden(9, 128), 16) is None
    assert (
        dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 128).float(), 16) is None
    )
    assert dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 64), 16) is None


def test_buffers_are_reused_across_calls(cpu_qpn8):
    head, lp = _head(256, 128), _processor()
    dflash2_rerank.maybe_candidates(head, lp, _hidden(7, 128, seed=1), 16)
    state = getattr(head, dflash2_rerank.STATE_ATTR)
    ptrs = (state.approx.data_ptr(), state.cand_ids.data_ptr(), state.exact.data_ptr())
    for seed in range(2, 5):
        x = _hidden(7, 128, seed=seed)
        ids, _ = dflash2_rerank.maybe_candidates(head, lp, x, 16)
        assert torch.equal(ids, _dense_topk(x, head, 16)[1])
    state = getattr(head, dflash2_rerank.STATE_ATTR)
    assert ptrs == (
        state.approx.data_ptr(),
        state.cand_ids.data_ptr(),
        state.exact.data_ptr(),
    )


# --------------------------------------------------------------------- gate
@pytest.fixture
def _fresh_cfg(monkeypatch):
    from vllm_sm70 import config

    monkeypatch.delenv(VERIFIER, raising=False)
    monkeypatch.delenv(SWITCH, raising=False)
    config.clear_cache()
    yield
    config.clear_cache()


def test_gate_follows_m9a_and_the_switch(_fresh_cfg, monkeypatch):
    assert dflash2_rerank.rerank_wanted(_vllm_config())
    assert not dflash2_rerank.rerank_wanted(
        _vllm_config(arch="Qwen3_5MoeForConditionalGeneration")
    )
    assert not dflash2_rerank.rerank_wanted(_vllm_config(k=5))
    monkeypatch.setenv(SWITCH, "0")
    assert not dflash2_rerank.rerank_wanted(_vllm_config())


def test_verifier_kill_switch_turns_it_off(_fresh_cfg, monkeypatch):
    from vllm_sm70 import config

    monkeypatch.setenv(VERIFIER, "0")
    config.clear_cache()
    assert not dflash2_rerank.rerank_wanted(_vllm_config())
