# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 MXFP8 experts held 8-bit (inkling2_sm70/moe.py,
expert_weights.py, ops/grouped_gemm.py), on CPU.

- the kill switch VLLM_SM70_INKLING_MXFP8_EXPERTS: an fp16 layer is held as
  codes and scales (half the bytes), "0", an fp32 model, a value not exact
  in fp16 or a scale byte above 246 dequantize it;
- the codes decode to the dequantized values exactly, also for the column
  slice of a TP2 rank that starts inside a block, and for the MXFP8 layer of
  the tiny checkpoint;
- the held layer equals the dequantized layer bit for bit on the torch form
  of the grouped route (slot and capacity ways, chunks) and the torch
  expert sum (batch and row-wise);
- through Triton's interpreter: the decode of all 254 finite codes, the
  dequant kernel, the slot GEMV kernel against the fp16 one, the whole
  route and the module's fused-routing forward, all bit for bit.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). The AOT compile check is
tests/ops/test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
from inkling2_sync_guard import no_host_sync

TINY = HERE.parent / "models" / "fixtures" / "inkling2_tiny"
SWITCH = "VLLM_SM70_INKLING_MXFP8_EXPERTS"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 64, 32, 2
GATE_CAP, UP_CAP = 4.0, 25.0
# Real layers hold exponent bytes 113..118 (spikes S1).
SCALES = (113, 119)
FINITE = [c for c in range(256) if c & 0x7F != 0x7F]


def _moe():
    from vllm_sm70.models.inkling2_sm70 import moe

    return moe


def _gg():
    from vllm_sm70.models.inkling2_sm70.ops import grouped_gemm

    return grouped_gemm


def _ew():
    from vllm_sm70.models.inkling2_sm70 import expert_weights

    return expert_weights


def _config():
    return SimpleNamespace(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_per_tok=TOP_K,
        route_scale=2.0,
        mlp_gate_softcap=GATE_CAP,
        mlp_up_softcap=UP_CAP,
    )


def _codes(shape, g, scales=SCALES):
    pick = torch.tensor(FINITE, dtype=torch.uint8)
    codes = pick[torch.randint(len(pick), shape, generator=g)]
    blocks = (*shape[:-1], -(-shape[-1] // 32))
    scale = torch.randint(*scales, blocks, generator=g, dtype=torch.uint8)
    return codes.view(torch.float8_e4m3fn), scale


def layer(seed=0, *, tp_size=1, tp_rank=0, scales=SCALES):
    """A random MXFP8 ExpertLayer of this rank, sliced as the loader does."""
    ew = _ew()
    g = torch.Generator().manual_seed(seed)
    w13, s13 = _codes((EXPERTS, 2 * INTERMEDIATE, HIDDEN), g, scales)
    w2, s2 = _codes((EXPERTS, HIDDEN, INTERMEDIATE), g, scales)
    local = INTERMEDIATE // tp_size
    first = local * tp_rank
    rows = slice(2 * first, 2 * (first + local))
    return ew.ExpertLayer(
        23,
        ew.MXFP8,
        ew.PackedExpertWeight(
            "w13", ew.MXFP8, w13[:, rows], s13[:, rows], (0, HIDDEN),
            interleaved=True,
        ),
        ew.PackedExpertWeight(
            "w2", ew.MXFP8, w2[:, :, first : first + local], s2,
            (first, first + local),
        ),
    )  # fmt: skip


def module(layer_, *, tp_size=1, dtype=torch.float16, seed=0, held=True):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        patch.setenv(SWITCH, "1" if held else "0")
        m = _moe().Inkling2Sm70Experts(_config(), tp_size=tp_size, dtype=dtype)
        g = torch.Generator().manual_seed(seed)
        m.gate.weight.data.copy_(torch.randn(EXPERTS, HIDDEN, generator=g))
        m.load_expert_layer(layer_)
    assert (m.mx13 is not None) is held
    return m


def pair(seed=0, **kwargs):
    """(held, dequantized) modules of one layer."""
    lay = layer(seed, **kwargs)
    tp = kwargs.get("tp_size", 1)
    return module(lay, tp_size=tp), module(lay, tp_size=tp, held=False)


def batch(m, rows, seed, pad=()):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g).half()
    is_padding = torch.zeros(rows, dtype=torch.bool)
    is_padding[list(pad)] = True
    _, weights, indices = m.route(x, is_padding)
    return x, indices, weights


def grouped(m, x, indices, weights, kernels):
    out = torch.full_like(x, float("nan"))
    _gg().grouped_expert_sum(
        x, indices, weights, *m.expert_weights(), GATE_CAP, UP_CAP, out,
        kernels=kernels,
    )  # fmt: skip
    return out


# --- switch and load -------------------------------------------------------------


@pytest.mark.parametrize("value, held", [(None, True), ("1", True), ("0", False)])
def test_kill_switch_only_zero_dequantizes(monkeypatch, value, held):
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    moe = _moe()
    assert moe.MXFP8_SWITCH == SWITCH and moe.mxfp8_enabled() is held
    monkeypatch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
    m = moe.Inkling2Sm70Experts(_config(), dtype=torch.float16)
    m.load_expert_layer(layer())
    assert (m.mx13 is not None) is held


def test_held_layer_has_half_the_bytes_and_no_fp16_weights():
    held, plain = pair()
    w13, w2 = held.expert_weights()
    assert held.w13.numel() == held.w2.numel() == 0
    assert w13.codes.dtype == w2.codes.dtype == torch.uint8
    assert w13.shape == plain.w13.shape and w2.shape == plain.w2.shape
    held_bytes = sum(t.numel() for w in (w13, w2) for t in (w.codes, w.scale))
    plain_bytes = 2 * (plain.w13.numel() + plain.w2.numel())
    assert held_bytes == plain_bytes // 2 + plain_bytes // 2 // 32
    for got, want in zip(held.expert_weights(), plain.expert_weights()):
        assert torch.equal(_gg().dequantize(got, kernels=False), want)


def test_fp32_model_dequantizes():
    lay = layer()
    m = module(lay, dtype=torch.float32, held=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        patch.setenv(SWITCH, "1")
        m = _moe().Inkling2Sm70Experts(_config(), dtype=torch.float32)
        m.load_expert_layer(lay)
    assert m.mx13 is None and m.w13.dtype == torch.float32


@pytest.mark.parametrize("scales", [(90, 100), (247, 248)], ids=["inexact", "huge"])
def test_values_off_fp16_or_huge_scales_dequantize(scales):
    lay = layer(scales=scales)
    assert lay.w13.mxfp8_codes(torch.float16, range_check=None) is None
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        patch.setenv(SWITCH, "1")
        m = _moe().Inkling2Sm70Experts(_config(), dtype=torch.float16)
        if scales[0] > 200:  # the values overflow fp16: refused as before
            with pytest.raises(ValueError, match="does not fit"):
                m.load_expert_layer(lay)
            return
        m.load_expert_layer(lay, range_check=None)
    assert m.mx13 is None and m.w13.shape[0] == EXPERTS


def test_range_check_sees_every_chunk_once():
    seen = []
    m = _moe().Inkling2Sm70Experts(_config(), dtype=torch.float16)
    m.load_expert_layer(layer(), range_check=lambda name, v, c: seen.append(name))
    assert m.mx13 is not None
    assert seen == ["w13 (experts 0-7)", "w2 (experts 0-7)"]


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_codes_decode_to_the_dequantized_values(tp_rank):
    """At TP2 rank 1's w2 columns start at 16, inside the first block."""
    lay = layer(3, tp_size=2, tp_rank=tp_rank)
    for packed in (lay.w13, lay.w2):
        codes = packed.mxfp8_codes(torch.float16)
        assert codes.offset == (16 * tp_rank if packed is lay.w2 else 0)
        weight = _gg().Mxfp8Weight(codes.codes, codes.scale, codes.offset)
        got = _gg().dequantize(weight, kernels=False)
        assert torch.equal(got, packed.dequantize(torch.float16))


def test_tiny_checkpoint_layer_is_held_and_exact():
    from safetensors.torch import load_file

    from vllm_sm70.models.inkling2_sm70 import quant_config, sm70_configs

    native = load_file(TINY / "model.safetensors")
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    formats = quant_config.expert_formats(
        json.loads((TINY / "hf_quant_config.json").read_text())
    )
    for tp_rank in (0, 1):
        layers = _ew().load_expert_layers(
            native, config, tp_size=2, tp_rank=tp_rank, formats=formats
        )
        for index in (20, 23):
            lay = layers[index]
            m = _moe().Inkling2Sm70Experts(config, tp_size=2, dtype=torch.float16)
            m.load_expert_layer(lay)
            assert m.mx13 is not None
            want = lay.dequantize(torch.float16)
            for got, w in zip(m.expert_weights(), want):
                assert torch.equal(_gg().dequantize(got, kernels=False), w)


# --- held == dequantized, torch forms ----------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("m", [1, 8, 40, 512, 600])
def test_torch_form_route_is_bit_equal(tp_size, m):
    held, plain = pair(m, tp_size=tp_size)
    args = batch(plain, m, seed=m, pad=[m // 2] if m > 1 else ())
    assert torch.equal(grouped(held, *args, False), grouped(plain, *args, False))


def test_chunks_and_capacity_way_are_bit_equal(monkeypatch):
    held, plain = pair(5)
    args = batch(plain, 100, seed=4)
    monkeypatch.setattr(_gg(), "CHUNK_TOKENS", 48)
    monkeypatch.setattr(_gg(), "DEQUANT_EXPERTS", 3)
    assert _gg().capacity(48 * TOP_K, EXPERTS) > 0
    assert torch.equal(grouped(held, *args, False), grouped(plain, *args, False))


@pytest.mark.parametrize("rowwise, m", [(True, 1), (True, 5), (False, 40)])
def test_torch_expert_sum_is_bit_equal(rowwise, m):
    held, plain = pair(7)
    x, indices, weights = batch(plain, m, seed=1)
    outs = []
    for mod in (held, plain):
        out = torch.full_like(x, float("nan"))
        mod.torch_expert_sum(x, indices, weights, out, rowwise)
        outs.append(out)
    assert torch.equal(*outs)


def test_no_host_sync_on_the_held_route():
    held, _ = pair()
    trace: list = []
    with no_host_sync(trace):
        grouped(held, *batch(held, 40, seed=2), False)
    assert trace


# --- interpreted kernels ----------------------------------------------------------------


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 and require the
    interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    for name in (SWITCH, "VLLM_SM70_INKLING_GROUPED_MOE"):
        env.pop(name, None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", __file__, "-q", "-p",
            "no:cacheprovider", "-m", "not gpu", "-k",
            "interpreted and not test_interpreted_suite",
        ],
        env=env, capture_output=True, text=True, timeout=3600, check=False,
    )  # fmt: skip
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    passed = re.search(r"(\d+) passed", summary)
    assert passed and int(passed.group(1)) >= 8, summary
    assert "skipped" not in summary, summary


@interpreted
@pytest.mark.parametrize("offset", [0, 16])
def test_interpreted_dequant_kernel_decodes_every_code(offset):
    """All finite codes under every exponent byte from 100 to 246."""
    gg = _gg()
    scales = torch.arange(100, 247, dtype=torch.uint8)
    codes = torch.tensor(FINITE, dtype=torch.uint8).repeat(len(scales), 1)
    k = codes.shape[1] - offset
    codes = codes[:, offset:].reshape(1, len(scales), k).contiguous()
    blocks = -(-(k + offset) // 32)
    scale = scales[:, None].expand(-1, blocks).reshape(1, len(scales), blocks)
    weight = gg.Mxfp8Weight(codes, scale.contiguous(), offset)
    want = (
        codes.view(torch.float8_e4m3fn).float()
        * torch.exp2(scale.float() - 127)[..., :1]
    )
    keep = want.abs() <= 65504  # larger values never load
    got = gg.dequantize(weight, kernels=True)
    assert torch.equal(got, gg.dequantize(weight, kernels=False))
    exact = keep & (want.half().float() == want)
    assert torch.equal(got.float()[exact], want[exact])


@interpreted
@pytest.mark.parametrize("row_div", [1, TOP_K])
def test_interpreted_slot_gemv_equals_fp16_kernel(row_div):
    gg = _gg()
    g = torch.Generator().manual_seed(row_div)
    codes, scale = _codes((EXPERTS, 40, 150), g)
    weight = gg.Mxfp8Weight(codes.view(torch.uint8), scale[..., :5].contiguous())
    plain = gg.dequantize(weight, kernels=False)
    slot_experts = torch.tensor([3, -1, 0, 7, 7, -1], dtype=torch.int32)
    inputs = torch.randn(6 // row_div, 150, generator=g).half()
    got = gg._slot_gemv(inputs, weight, slot_experts, row_div, kernels=True)
    want = gg._slot_gemv(inputs, plain, slot_experts, row_div, kernels=True)
    assert torch.equal(got, want) and bool((got[[1, 5]] == 0).all())
    torch_form = gg._slot_gemv(inputs, weight, slot_experts, row_div, kernels=False)
    assert (got.float() - torch_form.float()).abs().max() <= 2e-3


@interpreted
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("m", [1, 8, 40])
def test_interpreted_route_is_bit_equal(tp_size, m):
    held, plain = pair(m, tp_size=tp_size)
    args = batch(plain, m, seed=m, pad=[m - 1] if m > 1 else ())
    assert torch.equal(grouped(held, *args, True), grouped(plain, *args, True))


@interpreted
@pytest.mark.parametrize("m", [1, 4])
def test_interpreted_module_forward_is_bit_equal(monkeypatch, m):
    """Batches of at most 64 slots take ops/routing.py and the slot way."""
    monkeypatch.delenv("VLLM_SM70_INKLING_GROUPED_MOE", raising=False)
    held, plain = pair(11)
    x = batch(plain, m, seed=9)[0]
    assert _moe().on_fused_routing(x, TOP_K) and held._fused_routing(x)
    outs = []
    for mod in (held, plain):
        out = torch.full_like(x, float("nan"))
        with no_host_sync():
            mod(x, out, torch.zeros(m, dtype=torch.bool), True)
        outs.append(out)
    assert torch.equal(*outs)
