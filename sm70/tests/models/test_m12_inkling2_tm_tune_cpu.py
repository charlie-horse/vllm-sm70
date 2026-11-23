# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 TurboMind tuning defaults of the engine-config hook
(inkling2_sm70/config.py ``tm_tuning_env``), on CPU.

- with FULL decode graphs, the TurboMind expert route on and its ops
  present, and no tuning knob set, the knobs are small-shape tuning on, the
  NVFP4 MoE limit at the graph cap times top-k (8 x 8 = 64 routed slots at
  most), the NVFP4 dense limit 0;
- nothing is set with VLLM_SM70_INKLING_TM_TUNE=0, VLLM_SM70_INKLING_NVFP4_TM
  =0, a knob already set (the user's), no FULL decode graph, or no
  TurboMind ops;
- the pre-capture warmup's mirror of the C++ limits (warmup/plan.py) reads
  them as the C++ will. The hook writing them into os.environ is
  test_m12_inkling2_config_cpu.py.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config.compilation import CUDAGraphMode
from vllm_sm70.models.inkling2_sm70 import config as icfg
from vllm_sm70.models.inkling2_sm70 import moe_tm
from vllm_sm70.warmup import plan

SWITCH = "VLLM_SM70_INKLING_TM_TUNE"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """No knob set; whatever a test writes is undone."""
    for name in (*icfg.TM_TUNE_KNOBS, SWITCH, moe_tm.SWITCH):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setattr(moe_tm, "turbomind", lambda: object())


def comp(mode=CUDAGraphMode.FULL_DECODE_ONLY, largest=None, sizes=None):
    return SimpleNamespace(
        cudagraph_mode=mode,
        max_cudagraph_capture_size=largest,
        cudagraph_capture_sizes=sizes,
    )


def hf(top_k=8):
    text = SimpleNamespace(num_experts_per_tok=top_k)
    return SimpleNamespace(get_text_config=lambda: text)


@pytest.mark.parametrize(
    "largest, sizes, slots",
    [(8, None, 64), (4, [1, 2, 4], 32), (None, [1, 2], 16), (1, None, 8)],
)
def test_knobs_for_the_decode_graph_cap(largest, sizes, slots):
    got = icfg.tm_tuning_env(comp(largest=largest, sizes=sizes), hf())
    assert got == {
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": "1",
        "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": str(slots),
        "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M": "0",
    }
    assert int(got["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"]) <= (
        icfg.DECODE_GRAPH_MAX_BATCH * 8
    )


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY]
)
def test_full_decode_modes_tune(mode):
    assert icfg.tm_tuning_env(comp(mode=mode, largest=4), hf())


@pytest.mark.parametrize(
    "case",
    [
        "switch_off",
        "tm_off",
        "user_knob",
        "user_empty_knob",
        "piecewise",
        "none",
        "no_mode",
        "no_cap",
        "no_top_k",
        "no_ops",
    ],
)
def test_nothing_is_set(monkeypatch, case):
    cfg, hf_config = comp(largest=8), hf()
    if case == "switch_off":
        monkeypatch.setenv(SWITCH, "0")
    elif case == "tm_off":
        monkeypatch.setenv(moe_tm.SWITCH, "0")
    elif case == "user_knob":
        monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS", "128")
    elif case == "user_empty_knob":
        monkeypatch.setenv("VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES", "")
    elif case == "piecewise":
        cfg = comp(mode=CUDAGraphMode.PIECEWISE, largest=8)
    elif case == "none":
        cfg = comp(mode=CUDAGraphMode.NONE, largest=8)
    elif case == "no_mode":
        cfg = comp(mode=None, largest=8)
    elif case == "no_cap":
        cfg = comp(largest=None, sizes=[])
    elif case == "no_top_k":
        hf_config = None
    elif case == "no_ops":
        monkeypatch.setattr(moe_tm, "turbomind", lambda: None)
    assert icfg.tm_tuning_env(cfg, hf_config) == {}


def test_the_switch_is_registered():
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["models"]


def test_the_warmup_plan_reads_the_knobs():
    before = plan.limits()[plan.NVFP4_MOE]
    assert not before.enabled
    tuning = icfg.tm_tuning_env(comp(largest=4), hf())
    os.environ.update(tuning)  # as verify_and_update_config does
    limits = plan.limits()
    assert limits[plan.NVFP4_MOE].enabled and limits[plan.NVFP4_MOE].max_m == 32
    assert limits[plan.NVFP4_DENSE].enabled and limits[plan.NVFP4_DENSE].max_m == 0
    # A second engine config in this process keeps them.
    assert icfg.tm_tuning_env(comp(largest=8), hf()) == {}
