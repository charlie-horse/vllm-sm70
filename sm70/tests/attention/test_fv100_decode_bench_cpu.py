# SPDX-License-Identifier: Apache-2.0
"""CPU checks for sm70/tools/bench_fv100_decode.py's launch model.

The bench prints Z / active CTAs / tokens per CTA from a Python model of
the K14 fused decode launch. These tests pin that model to the kernel
source (constants, launch bounds) and to the fv100 planner (captured
envelope), and check the chunk split covers the window exactly.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import re
import sys

import pytest

SM70 = pathlib.Path(__file__).resolve().parents[2]
TOOL = SM70 / "tools" / "bench_fv100_decode.py"
CU = SM70 / "csrc" / "flash_v100" / "flash_decode_paged.cu"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_fv100_decode", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through it
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def _cu_int(name: str) -> int:
    m = re.search(rf"constexpr int {name} = (\d+);", CU.read_text())
    assert m, name
    return int(m.group(1))


def test_constants_match_the_kernel(bench):
    assert bench.FUSED_MAX_CHUNK == _cu_int("kGqaFusedMaxChunk")
    assert bench.FUSED_MIN_CHUNK == _cu_int("kGqaFusedMinChunk")
    assert bench.FUSED_MAX_CTAS_PER_HEAD == _cu_int("kGqaFusedMaxCtasPerHead")
    text = CU.read_text()
    m = re.search(
        r"__launch_bounds__\(kThreadsPerBlock, (\d+)\)\s*"
        r"flash_attention_decode_gqa8_d128_fused_kernel",
        text,
    )
    assert m and int(m.group(1)) == bench.FUSED_BLOCKS_PER_SM
    assert "<<<fused_grid, block, 0, stream>>>" in text


@pytest.mark.parametrize("mml", [4096, 32767, 32768, 131072, 262144])
def test_envelope_matches_the_captured_plan(bench, mml, monkeypatch):
    torch = pytest.importorskip("torch")
    from vllm_sm70.ops import fv100

    fv100.clear_workspaces()
    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: True)
    q = torch.empty((1, 16, 128), dtype=torch.float16)
    k = torch.empty((1, 16, 2, 128), dtype=torch.float16)
    table = torch.zeros((1, -(-mml // 16)), dtype=torch.int32)
    plan = fv100._get_decode_plan(
        q, k, table, max_seq_len_hint=mml, workspace_seq_capacity_hint=mml
    )
    psize, slots = bench.envelope_partition_slots(mml)
    assert psize == plan.partition_size
    assert slots == fv100._round_decode_partition_capacity(
        plan.workspace_num_partitions
    )


@pytest.mark.parametrize(
    "batch,heads_kv,slots,sms,expected",
    [
        (1, 2, 256, 80, 80),
        (4, 2, 256, 80, 20),
        (1, 2, 32, 80, 32),
        (64, 2, 256, 80, 2),
        (200, 2, 256, 80, 1),
    ],
)
def test_ctas_per_head(bench, batch, heads_kv, slots, sms, expected):
    assert bench.fused_ctas_per_head(batch, heads_kv, slots, sms) == expected


def _kernel_chunks(seq_len, wleft, z, bench):
    """The kernel's per-CTA chunk loop (flash_decode_paged.cu:2131-2190)."""
    work = bench.fused_work(seq_len, wleft, z)
    per_cta = []
    for cta in range(work.active):
        spans = []
        for chunk in range(cta, work.num_chunks, work.active):
            a = work.lo + chunk * work.length // work.num_chunks
            b = work.lo + (chunk + 1) * work.length // work.num_chunks
            spans.append((a, b))
        per_cta.append(spans)
    return work, per_cta


@pytest.mark.parametrize(
    "seq_len", [1, 63, 64, 65, 513, 514, 2048, 4096, 32768, 262144]
)
@pytest.mark.parametrize("wleft", [-1, 512])
@pytest.mark.parametrize("z", [1, 9, 20, 80, 256])
def test_chunks_cover_the_window(bench, seq_len, wleft, z):
    work, per_cta = _kernel_chunks(seq_len, wleft, z, bench)
    lo = max(0, seq_len - 1 - wleft) if wleft >= 0 else 0
    assert work.lo == lo and work.length == seq_len - lo
    assert 1 <= work.active <= z
    spans = sorted(s for cta in per_cta for s in cta)
    assert spans[0][0] == lo and spans[-1][1] == seq_len
    for (_, b), (c, _) in itertools.pairwise(spans):
        assert b == c
    for a, b in spans:
        assert 1 <= b - a <= bench.FUSED_MAX_CHUNK
    assert work.max_tokens_per_cta >= max(sum(b - a for a, b in s) for s in per_cta)


def test_flint_tp2_launch_shapes(bench):
    _, slots = bench.envelope_partition_slots(bench.MAX_MODEL_LEN)
    z1 = bench.fused_ctas_per_head(1, bench.HEADS_KV, slots, 80)
    # 4K full layer at batch 1: 64 CTAs of 64 keys, 64 partials to combine.
    full = bench.fused_work(4096, bench.window_left("full"), z1)
    assert (z1, full.active, full.max_tokens_per_cta) == (80, 64, 64)
    # SWA reads only the 513-key window whatever the context.
    for ctx in (2048, 4096, 32768):
        swa = bench.fused_work(ctx, bench.window_left("swa"), z1)
        assert (swa.length, swa.active) == (513, 9)


def test_two_launch_live_partitions(bench):
    assert bench.live_partitions(4096, -1, 1024) == 4
    assert bench.live_partitions(4096, 512, 1024) == 1
    assert bench.live_partitions(2048, 512, 1024) == 1
    assert bench.live_partitions(1100, 512, 1024) == 2


def test_layer_split_from_config(bench):
    cfg = {
        "num_hidden_layers": 4,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "sliding_window": 513,
    }
    assert bench.layer_split_from_config(cfg) == (2, 2, 513)
    assert bench.layer_split_from_config({"text_config": cfg}) == (2, 2, 513)
    assert bench.layer_split_from_config({**cfg, "use_sliding_window": False}) == (
        0,
        4,
        None,
    )
    with pytest.raises(ValueError):
        bench.layer_split_from_config({**cfg, "num_hidden_layers": 5})
    assert bench.parse_layer_split("36,12") == (36, 12)
    with pytest.raises(ValueError):
        bench.parse_layer_split("-1,3")


def test_step_estimate(bench):
    Row = bench.Row
    rows = [
        Row(1, 4096, "swa", "raw", 80, 9, 57, 9.0, 8.0, 0.0),
        Row(1, 4096, "full", "raw", 80, 64, 64, 15.0, 14.0, 0.0),
        Row(1, 4096, "swa", "dispatch", 80, 9, 57, None, None, None, "not built"),
    ]
    assert bench.step_estimate_us(rows, 3, 1) == {(1, 4096, "raw"): 3 * 8.0 + 14.0}
