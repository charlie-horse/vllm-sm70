# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 NVFP4 expert GEMV (inkling2_sm70/ops/nvfp4_gemv.py) and its
route in moe_tm.py, on CPU.

``vllm_sm70.ops.tm`` is a pure-torch stand-in whose prepare packs the codes
in the TurboMind layout the kernel reads (words [N / 32, K / 8, 32], nibble
p at K row NIBBLE_K[p]) and whose dense stage dequantizes that layout.

- the kill switch VLLM_SM70_INKLING_NVFP4_GEMV and the route choice: the
  real extension's banks with k_ld = 32 K and q_ld = N, compact one-row
  slots only, tensors off the GPU on TurboMind outside the interpreter;
- through Triton's interpreter:
  - ``nvfp4_slot_gemv`` for W2 and for W13 with the soft-capped SwiGLU,
    against an fp32 dequant, at small sizes and the Inkling 2 Mini TP2 bank
    shapes, out-of-range expert ids clamped;
  - ``Nvfp4TmExperts.expert_sum`` and ``routed_expert_sum`` on the GEMV
    route against the TurboMind stage route (no stage call on the GEMV
    route), with an activation hook, and the stage route above the compact
    bound and with the switch off.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). The AOT compile check is
tests/ops/test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import gemv, moe_tm
from vllm_sm70.models.inkling2_sm70.ops import nvfp4_gemv, torch_ref

OPS = pathlib.Path(nvfp4_gemv.__file__).resolve().parent
SWITCH = "VLLM_SM70_INKLING_NVFP4_GEMV"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
E2M1 = torch.tensor(ew.E2M1_VALUES, dtype=torch.float32)
GATE_CAP, UP_CAP = 4.0, 25.0
TOL = 2e-3


def max_abs_rel(got, want):
    scale = max(float(want.float().abs().max()), 1e-6)
    return float((got.float() - want.float()).abs().max() / scale)


# --------------------------------------------------------------------------
# The TurboMind code layout, in torch
# --------------------------------------------------------------------------
def tm_pack(codes):
    """int32 [K, N / 8] of uint8 codes [K, N] in the TurboMind layout."""
    k, n = codes.shape
    tiles = codes.to(torch.int32).reshape(k // 8, 8, n // 32, 32).permute(2, 0, 3, 1)
    words = torch.zeros(n // 32, k // 8, 32, dtype=torch.int32)
    for nibble, row in enumerate(nvfp4_gemv.NIBBLE_K):
        words |= tiles[..., row] << (4 * nibble)
    return words.reshape(k, n // 8)


def tm_unpack(words, k, n):
    """uint8 codes [K, N] of `tm_pack` words."""
    words = words.reshape(n // 32, k // 8, 32)
    codes = torch.empty(n // 32, k // 8, 32, 8, dtype=torch.long)
    for nibble, row in enumerate(nvfp4_gemv.NIBBLE_K):
        codes[..., row] = (words >> (4 * nibble)) & 15
    return codes.permute(1, 3, 0, 2).reshape(k, n)


def dequant(words, scales):
    """fp32 [K, N] of one prepared expert."""
    k, n = scales.shape[0] * 16, scales.shape[1]
    return E2M1[tm_unpack(words, k, n)] * scales.float().repeat_interleave(16, dim=0)


def make_tm():
    tm = types.ModuleType("vllm_sm70.ops.tm")
    tm.stages, tm.banks = [], {}

    def nvfp4_sm70_prepare(qweight, scales, group_size=16, interleave_gated_silu=False):
        k, n = qweight.shape
        return [tm_pack(qweight), scales.clone(), torch.tensor([32 * k, n])]

    def awq_moe_build_strided_ptrs(tm_w, tm_s, k_ld, q_ld, num_experts):
        ptrs = torch.arange(num_experts, dtype=torch.int64) + 1000 * len(tm.banks)
        tm.banks[int(ptrs[0])] = (tm_w, tm_s)
        return [ptrs, ptrs.clone()]

    def nvfp4_moe_dense_stage_sm70_out(out, inp, offsets, ids, pw, ps, ne, k, n, g):
        tm.stages.append((inp.shape[0], ne, k, n))
        tm_w, tm_s = tm.banks[int(pw[0])]
        for group in range(ne):
            a, b = int(offsets[group]), int(offsets[group + 1])
            if b > a:
                weight = dequant(tm_w[int(ids[group])], tm_s[int(ids[group])])
                out[a:b] = (inp[a:b].float() @ weight).to(out.dtype)

    tm.nvfp4_sm70_prepare = nvfp4_sm70_prepare
    tm.awq_moe_build_strided_ptrs = awq_moe_build_strided_ptrs
    tm.nvfp4_moe_dense_stage_sm70_out = nvfp4_moe_dense_stage_sm70_out
    return tm


def random_codes(name, e, n, k, seed, global_scale=0.02):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (e, n, k), generator=g, dtype=torch.uint8)
    block = (torch.rand(e, n, k // 16, generator=g) * 3 + 1).to(torch.float8_e4m3fn)
    return ew.Nvfp4Codes(name, codes, block, torch.full((e,), global_scale))


def experts(tm, *, real=True, monkeypatch=None, compact=True):
    """Eight experts of hidden 64 and local intermediate 32."""
    if monkeypatch is not None:
        monkeypatch.setattr(moe_tm, "_real_extension", lambda: real)
    return moe_tm.Nvfp4TmExperts(
        tm, random_codes("w13", 8, 64, 64, 1), random_codes("w2", 8, 64, 32, 2),
        device=torch.device("cpu"), gate_cap=GATE_CAP, up_cap=UP_CAP,
        compact=compact,
    )  # fmt: skip


def bank(e, k, n, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (e, k, n), generator=g, dtype=torch.uint8)
    scales = (torch.rand(e, k // 16, n, generator=g) * 0.01 + 1e-3).half()
    words = torch.stack([tm_pack(codes[i]) for i in range(e)])
    return words, scales


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    moe_tm.Workspace.clear()
    yield
    moe_tm.Workspace.clear()


# --------------------------------------------------------------------------
# Source, layout, switch and route choice
# --------------------------------------------------------------------------
def test_kernel_source_holds_nothing_sm70_rejects():
    text = re.sub(r'"""(?:.|\n)*?"""', "", (OPS / "nvfp4_gemv.py").read_text())
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    for banned in ("tl.dot", "tanh.approx", "inline_asm", "gdc_", "bfloat16"):
        assert banned not in code, banned


def test_switch_is_registered_in_models():
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["models"]
    assert gemv.NVFP4_SWITCH == SWITCH


def test_layout_round_trips():
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (48, 64), generator=g, dtype=torch.uint8)
    words = tm_pack(codes)
    assert words.shape == (48, 8)
    assert torch.equal(tm_unpack(words, 48, 64), codes.long())
    # Word (tile 1, K row block 2, column 5) holds K row 16 + 0 in its low
    # nibble and K row 16 + 1 in nibble 4.
    word = int(words.reshape(2, 6, 32)[1, 2, 5])
    assert word & 15 == codes[16, 37] and (word >> 16) & 15 == codes[17, 37]


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("1", True), ("0", False), ("off", True)]
)
def test_kill_switch_only_zero_turns_the_route_off(monkeypatch, value, enabled):
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    monkeypatch.setattr(gemv._compat, "AVAILABLE", True)
    monkeypatch.setattr(gemv._compat, "INTERPRET", False)
    assert gemv.on_nvfp4(types.SimpleNamespace(is_cuda=True)) is enabled
    assert gemv.on_nvfp4(types.SimpleNamespace(is_cuda=False)) is False


def test_only_real_banks_in_the_gemv_layout_take_the_route(monkeypatch):
    tm = make_tm()
    assert experts(tm, real=True, monkeypatch=monkeypatch).slot_gemv is True
    assert experts(tm, real=False, monkeypatch=monkeypatch).slot_gemv is False
    prepare = tm.nvfp4_sm70_prepare
    tm.nvfp4_sm70_prepare = lambda w, s, *a: [*prepare(w, s)[:2], torch.tensor([1, 2])]
    assert experts(tm, real=True, monkeypatch=monkeypatch).slot_gemv is False


def test_tensors_off_the_gpu_take_the_stages_outside_the_interpreter(monkeypatch):
    monkeypatch.setattr(gemv._compat, "INTERPRET", False)
    tm = make_tm()
    route = experts(tm, monkeypatch=monkeypatch)
    x = torch.randn(2, 64).half()
    route.expert_sum(x, torch.tensor([[1, 5], [0, 7]]), torch.rand(2, 2), x.clone())
    assert len(tm.stages) == 2


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------
def _gemv_reference(rows, words, scales, ids, swiglu):
    e = scales.shape[0]
    out = torch.stack(
        [
            rows[s].float() @ dequant(words[min(max(int(i), 0), e - 1)],
                                      scales[min(max(int(i), 0), e - 1)])
            for s, i in enumerate(ids)
        ]
    )  # fmt: skip
    if not swiglu:
        return out
    gate, up = out.half().chunk(2, dim=-1)
    return torch_ref.softcapped_swiglu(gate, up, GATE_CAP, UP_CAP)


@interpreted
@pytest.mark.parametrize("swiglu", [False, True], ids=["w2", "w13_swiglu"])
@pytest.mark.parametrize("k,n", [(48, 64), (160, 128)])
def test_interpreted_slot_gemv_matches_dequant(swiglu, k, n):
    words, scales = bank(4, k, n, seed=k)
    ids = torch.tensor([3, 0, 0, 2, 9, -1], dtype=torch.int32)
    wide = (torch.randn(len(ids), k + 8) * 0.5).half()
    rows = wide[:, 4 : 4 + k]
    width = n // 2 if swiglu else n
    out = torch.full((len(ids), width + 32), float("nan")).half()
    caps = (GATE_CAP, UP_CAP) if swiglu else None
    nvfp4_gemv.nvfp4_slot_gemv(rows, words, scales, ids, out[:, :width], caps)
    want = _gemv_reference(rows, words, scales, ids, swiglu)
    assert max_abs_rel(out[:, :width], want) <= TOL
    assert bool(out[:, width:].isnan().all())


@interpreted
@pytest.mark.parametrize(
    "k,n,swiglu", [(2304, 1152, True), (576, 2304, False)], ids=["w13", "w2"]
)
def test_interpreted_mini_tp2_banks(k, n, swiglu):
    words, scales = bank(3, k, n, seed=1)
    ids = torch.tensor([2, 0, 1], dtype=torch.int32)
    rows = (torch.randn(3, k) * 0.5).half()
    out = torch.empty(3, n // 2 if swiglu else n).half()
    caps = (GATE_CAP, UP_CAP) if swiglu else None
    nvfp4_gemv.nvfp4_slot_gemv(rows, words, scales, ids, out, caps)
    want = _gemv_reference(rows, words, scales, ids, swiglu)
    assert max_abs_rel(out, want) <= TOL


# --------------------------------------------------------------------------
# moe_tm.py on the GEMV route
# --------------------------------------------------------------------------
def _batch(m, seed=3, e=8, top_k=2):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, 64, generator=g) * 0.5).half()
    indices = torch.stack([torch.randperm(e, generator=g)[:top_k] for _ in range(m)])
    return x, indices, torch.rand(m, top_k, generator=g)


def _both(monkeypatch, run):
    """(GEMV route output, stage route output, stage calls on the GEMV route)."""
    outs, calls = [], []
    for real in (True, False):
        moe_tm.Workspace.clear()
        tm = make_tm()
        route = experts(tm, real=real, monkeypatch=monkeypatch)
        assert route.slot_gemv is real
        outs.append(run(route))
        calls.append(len(tm.stages))
    return outs[0], outs[1], calls[0]


@interpreted
@pytest.mark.parametrize("m", [1, 2, 8, 40, 41])
def test_interpreted_expert_sum_matches_the_stages(monkeypatch, m):
    x, indices, weights = _batch(m)
    indices[0, 0] = -1  # a padding slot

    def run(route):
        out = torch.full_like(x, float("nan"))
        route.expert_sum(x, indices, weights, out)
        return out

    got, want, calls = _both(monkeypatch, run)
    # Up to 640 slots run as compact chunks of at most 80 one-row slots.
    assert calls == 0 and max_abs_rel(got, want) <= TOL


@interpreted
def test_interpreted_per_expert_groups_take_the_stages(monkeypatch):
    tm = make_tm()
    route = experts(tm, monkeypatch=monkeypatch, compact=False)
    x, indices, weights = _batch(3)
    route.expert_sum(x, indices, weights, torch.empty_like(x))
    assert [(rows, groups) for rows, groups, *_ in tm.stages] == [(6, 8), (6, 8)]


@interpreted
def test_interpreted_activation_hook_runs_between_the_projections(monkeypatch):
    x, indices, weights = _batch(4)
    seen = []

    def hook(act):
        seen.append(act.shape)
        return act * 0.5

    def run(route):
        out = torch.empty_like(x)
        route.expert_sum(x, indices, weights, out, act_quant=hook)
        return out

    got, want, calls = _both(monkeypatch, run)
    assert calls == 0 and seen == [(8, 32), (8, 32)]
    assert max_abs_rel(got, want) <= TOL


@interpreted
@pytest.mark.parametrize("m", [1, 3])
def test_interpreted_routed_expert_sum_matches_the_stages(monkeypatch, m):
    g = torch.Generator().manual_seed(m)
    x = (torch.randn(m, 64, generator=g) * 0.5).half()
    logits = torch.randn(m, 8, generator=g)
    bias = torch.randn(8, generator=g) * 0.1
    padding = torch.zeros(m, dtype=torch.bool)
    padding[-1] = m > 1

    def run(route):
        out = torch.full_like(x, float("nan"))
        route.routed_expert_sum(x, logits, bias, 2, 2.5, padding, out)
        return out

    got, want, calls = _both(monkeypatch, run)
    assert calls == 0 and max_abs_rel(got, want) <= TOL
    if m > 1:
        assert bool((got[-1] == 0).all())


@interpreted
def test_interpreted_switch_off_takes_the_stages(monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    tm = make_tm()
    route = experts(tm, monkeypatch=monkeypatch)
    x, indices, weights = _batch(2)
    route.expert_sum(x, indices, weights, torch.empty_like(x))
    assert len(tm.stages) == 2


# --- interpreted child run ------------------------------------------------------------


def test_interpreted_suite():
    """Run this module again with TRITON_INTERPRET=1 (Triton reads it at
    import) and require the interpreted cases to pass, none skipped."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    env.pop(SWITCH, None)
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
    assert passed and int(passed.group(1)) >= 16, summary
    assert "skipped" not in summary, summary
