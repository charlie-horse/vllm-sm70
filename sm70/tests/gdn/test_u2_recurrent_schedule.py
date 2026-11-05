# SPDX-License-Identifier: Apache-2.0
"""U2 (patches/0003): the FLA recurrent decode block size on sm_70. CPU only.

What: sm70_recurrent_bv() in vllm/third_party/flash_linear_attention/ops/
  fused_recurrent.py returns the fork's Volta BV (the largest of 32/16/8
  that gives >= 2 waves of CTAs on 80 SMs) on compute capability 7.0, and
  the caller's (upstream) BV on any other GPU or with
  VLLM_SM70_FLA_RECURRENT_SCHED=0. Both launch sites call it.
Needs vLLM's Python source with the patch series applied; skips on an
unpatched vLLM (CI installs the released wheel).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
fr = pytest.importorskip("vllm.third_party.flash_linear_attention.ops.fused_recurrent")

V100_SMS = 80


@pytest.fixture
def schedule(monkeypatch):
    if not hasattr(fr, "sm70_recurrent_bv"):
        # CI tests against the released vllm wheel (unpatched), like U1.
        pytest.skip("vLLM without patches/0003 (U2)")
    monkeypatch.delenv("VLLM_SM70_FLA_RECURRENT_SCHED", raising=False)
    if not hasattr(fr.triton, "next_power_of_2"):
        # No usable Triton on this host: vllm.triton_utils hands out a
        # placeholder module. The schedule only needs these two helpers.
        monkeypatch.setattr(
            fr,
            "triton",
            SimpleNamespace(
                next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length(),
                cdiv=lambda a, b: -(-a // b),
            ),
        )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: SimpleNamespace(multi_processor_count=V100_SMS),
    )

    def on(capability):
        monkeypatch.setattr(
            fr.current_platform,
            "is_device_capability",
            lambda cap, device_id=0: cap == capability,
        )
        return fr.sm70_recurrent_bv

    return on


DEV = torch.device("cuda", 0)


@pytest.mark.parametrize(
    "nhv, expected",
    [
        (1 * 16, 8),  # Qwen3.6 TP2, 1 request: 4*16=64, 8*16=128 < 160 -> 8
        (1 * 24, 16),  # Qwen3.8 TP2, 1 request: 8*24=192 >= 160
        (2 * 16, 16),  # 2 requests x Hv16: 8*32=256
        (4 * 16, 32),  # 4*64=256 >= 160 at BV=32
        (2, 8),  # nothing reaches 2 waves: smallest candidate
    ],
)
def test_volta_values(schedule, nhv, expected):
    assert schedule(70)(32, 128, nhv, DEV) == expected


@pytest.mark.parametrize("cap", [75, 80, 90, 100])
@pytest.mark.parametrize("bv", [32, 8, 4])
def test_other_gpus_keep_upstream(schedule, cap, bv):
    assert schedule(cap)(bv, 128, 16, DEV) == bv


def test_kill_switch(schedule, monkeypatch):
    fn = schedule(70)
    monkeypatch.setenv("VLLM_SM70_FLA_RECURRENT_SCHED", "0")
    assert fn(32, 128, 16, DEV) == 32
    monkeypatch.setenv("VLLM_SM70_FLA_RECURRENT_SCHED", "1")
    assert fn(32, 128, 16, DEV) == 8


def test_small_v_is_capped(schedule):
    assert schedule(70)(4, 4, 1, DEV) == 4  # candidates are min(bv, next_pow2(V))


def test_both_launch_sites_use_it(schedule):
    src = inspect.getsource(fr)
    assert "BV = sm70_recurrent_bv(BV, V, N * HV, q.device)" in src
    assert "BV = sm70_recurrent_bv(BV, V, B * HV, dev)" in src
