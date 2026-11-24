# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 NVFP4 expert prefill on TurboMind's grouped stage, on CPU.

- ``tm.nvfp4_moe_dense_stage_route`` (ops/tm.py) mirrors the launch form
  that ``nvfp4_moe_dense_stage_sm70_out`` picks: the shape sets, the row
  cut-offs and the knobs are parsed from tm_moe_sm70.cu and compared;
- every Inkling 2 bank shape that moe_tm.py admits takes the grouped form;
- a prefill batch on moe_tm.py's dense form issues one grouped call per
  stage per slot chunk (512 tokens: 2 launches per layer visit instead of
  256); smaller batches keep the compact form;
- the decode-graph tuning limit never reaches the dense form, so no shape
  of the grouped stage is measured, in capture or anywhere else.
The real op runs in test_m12_inkling2_grouped_prefill_gpu.py.
Kill-switch: VLLM_SM70_INKLING_TM_GROUPED_PREFILL covered below.
"""

from __future__ import annotations

import pathlib
import re
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch

from vllm_sm70 import envs
from vllm_sm70.models.inkling2_sm70 import config as icfg
from vllm_sm70.models.inkling2_sm70 import moe_tm
from vllm_sm70.ops import tm

SM70 = pathlib.Path(__file__).resolve().parents[2]
OPS = SM70 / "csrc/turbomind/ops"
SWITCH = tm.INKLING_GROUPED_PREFILL_SWITCH
PARENT = "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL"
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 2304, 1152, 128, 8
LOCAL = INTERMEDIATE // 2
W13, W2 = (HIDDEN, 2 * LOCAL), (LOCAL, HIDDEN)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (SWITCH, PARENT, "VLLM_SM70_FLINT_MOE_COMPACT"):
        monkeypatch.delenv(name, raising=False)
    moe_tm.Workspace.clear()
    yield
    moe_tm.Workspace.clear()


# --------------------------------------------------------------------------
# The mirror against the C++ source
# --------------------------------------------------------------------------
def _stage_body() -> str:
    text = (OPS / "tm_moe_sm70.cu").read_text()
    start = text.index("void nvfp4_moe_dense_stage_sm70_out(")
    end = text.index("\nvoid ", start + 1)
    return re.sub(r"\s+", " ", text[start:end])


def _pairs(expr: str) -> set[tuple[int, int]]:
    """(k, n) pairs of one `exact_*_prefill_shape` expression."""
    out = {
        (int(k), int(n)) for k, n in re.findall(r"\(k == (\d+) && n == (\d+)\)", expr)
    }
    for fixed, alts in re.findall(
        r"\(k == (\d+) && \(((?:n == \d+(?: \|\| )?)+)\)\)", expr
    ):
        out |= {(int(fixed), int(n)) for n in re.findall(r"n == (\d+)", alts)}
    for fixed, alts in re.findall(
        r"\(n == (\d+) && \(((?:k == \d+(?: \|\| )?)+)\)\)", expr
    ):
        out |= {(int(k), int(fixed)) for k in re.findall(r"k == (\d+)", alts)}
    return out


def test_mirror_matches_the_cpp_gate():
    body = _stage_body()
    shapes: dict[int, set] = {}
    for name, expr in re.findall(r"const bool (exact_\w+_prefill_shape) =(.*?);", body):
        assert "input.size(0) > kNvfp4LegacyCompactGroups" in expr, name
        experts = int(re.search(r"num_experts == (\d+)", expr).group(1))
        pairs = _pairs(expr)
        assert pairs, name
        shapes.setdefault(experts, set()).update(pairs)
        assert name in body.split(expr, 1)[1], f"{name} is never used"
    assert shapes == {e: set(s) for e, s in tm.NVFP4_MOE_GROUPED_PREFILL_SHAPES.items()}
    legacy = re.search(r"kNvfp4LegacyCompactGroups = (\d+) \* (\d+);", body)
    assert int(legacy[1]) * int(legacy[2]) + 1 == tm.NVFP4_MOE_GROUPED_MIN_ROWS
    compact = re.search(r"kNvfp4MaxCompactGroups = (\d+) \* (\d+);", body)
    assert int(compact[1]) * int(compact[2]) == tm.NVFP4_MOE_COMPACT_MAX_GROUPS
    # The Inkling branch alone carries the Inkling switch.
    branch = body[body.index("exact_inkling2_prefill_shape &&") :]
    assert branch.index("sm70_inkling_tm_grouped_prefill_enabled()") < branch.index(
        "return;"
    )


def test_switch_registered_in_cpp_and_python():
    assert SWITCH in envs._SECTIONS["ops"]
    config = (OPS / "tm_config.h").read_text()
    assert f'X(inkling_tm_grouped_prefill, "{SWITCH}")' in config


def test_every_admitted_inkling_bank_takes_the_grouped_form():
    for hidden, intermediate, experts, _ in moe_tm.INKLING_CONTRACTS:
        for tp in moe_tm.INKLING_TP_SIZES:
            local = intermediate // tp
            for k, n in ((hidden, 2 * local), (local, hidden)):
                for rows in (65, 4096, moe_tm.CHUNK_ROWS):
                    assert tm.nvfp4_moe_dense_stage_route(rows, experts, k, n) == (
                        "grouped"
                    ), (rows, experts, k, n)


@pytest.mark.parametrize(
    "rows, experts, k, n, route",
    [
        (4096, 128, *W13, "grouped"),
        (4096, 128, *W2, "grouped"),
        (65, 128, *W2, "grouped"),
        (64, 128, *W13, "per_expert"),  # the dense form at <= 64 rows
        (80, 80, *W13, "compact"),
        (4096, 128, 2304, 2304, "per_expert"),  # not a stage shape
        (4096, 64, *W13, "per_expert"),
        (4096, 256, 2048, 512, "grouped"),  # Flint / Qwen3.6
        (4096, 512, 160, 2560, "grouped"),  # Qwen4
    ],
)
def test_route(rows, experts, k, n, route):
    assert tm.nvfp4_moe_dense_stage_route(rows, experts, k, n) == route


@pytest.mark.parametrize(
    "inkling, parent, route",
    [
        ("0", None, "per_expert"),
        ("1", None, "grouped"),
        ("false", None, "grouped"),  # only "0" acts
        ("", None, "grouped"),
        (None, "0", "per_expert"),  # the shared grouped-prefill knob
        (None, "", "per_expert"),  # atoi("") == 0 for that knob
    ],
)
def test_switches(monkeypatch, inkling, parent, route):
    for name, value in ((SWITCH, inkling), (PARENT, parent)):
        if value is not None:
            monkeypatch.setenv(name, value)
    assert tm.nvfp4_moe_dense_stage_route(4096, EXPERTS, *W13) == route


def test_inkling_switch_leaves_other_models_alone(monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    assert tm.nvfp4_moe_dense_stage_route(4096, 256, 2048, 512) == "grouped"
    assert tm.nvfp4_moe_dense_stage_route(4096, 512, 2560, 320) == "grouped"


# --------------------------------------------------------------------------
# moe_tm.py's stage calls at prefill sizes, real per-rank shapes
# --------------------------------------------------------------------------
class FakeBank:
    """A bank of the given [E, N, K] shape without weights."""

    def __init__(self, tm_ops, codes, device, range_check=None):
        self.num_experts, self.n, self.k = codes.shape
        self.ptrs_w = self.ptrs_s = torch.zeros(self.num_experts, dtype=torch.int64)


def recording_tm():
    ops = types.ModuleType("fake_tm")
    ops.calls = []

    def stage(out, inp, offsets, ids, pw, ps, ne, k, n, g):
        rows = inp.shape[0]
        assert out.shape == (rows, n) and inp.shape[1] == k and g == 16
        assert len(offsets) >= ne + 1 and len(ids) >= ne
        assert int(offsets[ne]) == rows
        ops.calls.append(
            (rows, ne, k, n, tm.nvfp4_moe_dense_stage_route(rows, ne, k, n))
        )
        out.zero_()

    ops.nvfp4_moe_dense_stage_sm70_out = stage
    return ops


def experts(monkeypatch, compact=True):
    monkeypatch.setattr(moe_tm, "Bank", FakeBank)
    ops = recording_tm()
    route = moe_tm.Nvfp4TmExperts(
        ops,
        SimpleNamespace(shape=(EXPERTS, 2 * LOCAL, HIDDEN)),
        SimpleNamespace(shape=(EXPERTS, HIDDEN, LOCAL)),
        device=torch.device("cpu"),
        gate_cap=4.0,
        up_cap=25.0,
        max_slots=moe_tm.CHUNK_ROWS,
        compact=compact,
    )
    return route, ops


def run(route, tokens, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=g).half()
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)]
    )
    weights = torch.rand(tokens, TOP_K, generator=g)
    out = torch.empty_like(x)
    route.expert_sum(x, ids, weights, out)
    return out


@pytest.mark.parametrize("tokens", [512, 2048, 3867])
def test_prefill_runs_one_grouped_launch_per_stage_and_chunk(monkeypatch, tokens):
    route, ops = experts(monkeypatch)
    run(route, tokens)
    slots = tokens * TOP_K
    chunks = -(-slots // moe_tm.CHUNK_ROWS)
    assert len(ops.calls) == 2 * chunks
    assert [c[0] for c in ops.calls[::2]] == [
        min(moe_tm.CHUNK_ROWS, slots - i * moe_tm.CHUNK_ROWS) for i in range(chunks)
    ]
    assert {c[1:] for c in ops.calls} == {
        (EXPERTS, *W13, "grouped"),
        (EXPERTS, *W2, "grouped"),
    }


def test_switch_off_restores_the_per_expert_loop(monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    route, ops = experts(monkeypatch)
    run(route, 512)
    assert [c[4] for c in ops.calls] == ["per_expert", "per_expert"]
    launches = sum(EXPERTS if c[4] == "per_expert" else 1 for c in ops.calls)
    assert launches == 2 * EXPERTS  # per layer visit of a 512-token chunk


@pytest.mark.parametrize("tokens", [1, 8, 64, 80])
def test_decode_and_verify_sizes_keep_the_compact_form(monkeypatch, tokens):
    route, ops = experts(monkeypatch)
    run(route, tokens)
    assert {c[4] for c in ops.calls} == {"compact"}


def test_first_dense_size_is_grouped(monkeypatch):
    route, ops = experts(monkeypatch)
    tokens = moe_tm.COMPACT_BATCH_MAX_SLOTS // TOP_K + 1
    run(route, tokens)
    assert [c[:1] + c[4:] for c in ops.calls] == [(tokens * TOP_K, "grouped")] * 2


def test_compact_off_sends_65_slots_and_up_to_grouped(monkeypatch):
    route, ops = experts(monkeypatch, compact=False)
    run(route, 9)  # 72 slots
    assert [c[4] for c in ops.calls] == ["grouped", "grouped"]


# --------------------------------------------------------------------------
# Tuning never reaches the grouped stage
# --------------------------------------------------------------------------
@pytest.mark.parametrize("query_len", [1, 8])
def test_decode_tuning_limit_stays_below_the_dense_form(monkeypatch, query_len):
    """The engine hook's NVFP4 MoE tuning limit (routed slots) is within the
    compact chunks, so a measured shape is never a grouped-stage shape and
    the grouped stage always runs kDefault (gemm_runtime.cpp
    select_moe_dispatch_policy_impl)."""
    from vllm.config.compilation import CUDAGraphMode

    for name in icfg.TM_TUNE_KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_SM70_INKLING_TM_TUNE", raising=False)
    monkeypatch.delenv(moe_tm.SWITCH, raising=False)
    monkeypatch.setattr(moe_tm, "turbomind", lambda: object())
    cap = icfg.DECODE_GRAPH_MAX_BATCH * query_len
    comp = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        max_cudagraph_capture_size=cap,
        cudagraph_capture_sizes=list(range(1, cap + 1)),
    )
    text = SimpleNamespace(num_experts_per_tok=TOP_K)
    hf = SimpleNamespace(get_text_config=lambda: text)
    knobs = icfg.tm_tuning_env(comp, hf, query_len)
    limit = int(knobs["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"])
    assert limit <= moe_tm.COMPACT_BATCH_MAX_SLOTS
