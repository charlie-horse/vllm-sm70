# SPDX-License-Identifier: Apache-2.0
"""K06: decode planner (trimmed port) and decode dispatch wiring.

Planner: port of fork cf9492a tests/kernels/attention/
test_sm70_flash_v100_decode_planner.py, trimmed to the lean subset: the
partition-size defaults, the static (CUDA-graph) plan with its workspace
envelope, the fp16 XQA workspace dtype, and the stale active-partition GPU
case (D128 here; D64 is not built). Dropped: XQA batch-context routing and
the E4M3/E5M2 cases (not compiled in K05).

Dispatch: the non-fused CPU cases of fork test_sm70_flash_v100_sinks_decode.py
(:1444-1591) restated for vllm_sm70.attention.flash_v100.dispatch: a sink
layer passes its sinks and window to the paged decode, a sink-free layer
does not, and XQA never takes a sink layer.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@pytest.fixture
def fv100():
    from vllm_sm70.ops import fv100 as mod

    mod.clear_workspaces()
    yield mod
    mod.clear_workspaces()


def _plan(fv100, q, k_cache, block_table, **kw):
    return fv100._get_decode_plan(q, k_cache, block_table, **kw)


@pytest.mark.parametrize(
    "capacity,hint,expected",
    [(4096, 1, 256), (8192, 4097, 256), (65536, 32769, 1024)],
)
def test_default_partition_size(fv100, capacity, hint, expected):
    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((capacity // 16, 16, 1, 256), dtype=torch.float16)
    table = torch.zeros((1, capacity // 16), dtype=torch.int32)
    assert _plan(fv100, q, k_cache, table, max_seq_len_hint=hint).partition_size == (
        expected
    )


def test_static_plan_uses_fixed_launch_and_active_runtime(fv100):
    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    table = torch.zeros((1, 512), dtype=torch.int32)
    plan = _plan(
        fv100,
        q,
        k_cache,
        table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
    )
    assert (plan.partition_size, plan.actual_num_partitions) == (256, 17)
    assert plan.launch_num_partitions == plan.workspace_num_partitions == 32
    tmp, mx, es, active = fv100._get_decode_workspace_for_plan(
        q, batch_capacity=1, num_heads=8, head_dim=256, plan=plan
    )
    assert tmp.shape[2] >= plan.launch_num_partitions
    assert mx.shape[:3] == tmp.shape[:3] == es.shape[:3]
    assert active.dtype == torch.int32 and int(active) == 17


def test_static_plan_under_capture_uses_the_envelope(fv100, monkeypatch):
    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: True)
    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    table = torch.zeros((1, 512), dtype=torch.int32)
    plan = _plan(
        fv100,
        q,
        k_cache,
        table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
    )
    assert plan.actual_num_partitions == plan.launch_num_partitions == 32
    # A caller-owned active tensor keeps the short runtime plan.
    plan = _plan(
        fv100,
        q,
        k_cache,
        table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
        active_num_partitions=torch.empty((1,), dtype=torch.int32),
    )
    assert (plan.actual_num_partitions, plan.launch_num_partitions) == (17, 32)


def test_static_short_workspace_keeps_partition_hint(fv100, monkeypatch):
    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: True)
    q = torch.empty((5, 6, 256), dtype=torch.float16)
    k_cache = torch.empty((16448, 16, 1, 256), dtype=torch.float16)
    table = torch.zeros((5, 16448), dtype=torch.int32)
    plan = _plan(
        fv100,
        q,
        k_cache,
        table,
        max_seq_len_hint=4096,
        workspace_seq_capacity_hint=4096,
        partition_size_hint=1024,
    )
    assert (plan.partition_size, plan.actual_num_partitions) == (1024, 4)
    assert plan.launch_num_partitions == plan.workspace_num_partitions == 4


def test_xqa_workspace_keeps_fp16_partials(fv100, monkeypatch):
    seen = []
    monkeypatch.setattr(
        fv100, "_op", lambda name: lambda *a: seen.append((name, a[6].dtype))
    )
    q = torch.empty((2, 6, 256), dtype=torch.float16)
    k = torch.empty((16, 16, 1, 256), dtype=torch.float16)
    fv100.flash_attn_decode_paged_xqa(
        q,
        k,
        k,
        torch.zeros((2, 16), dtype=torch.int32),
        torch.full((2,), 100, dtype=torch.int32),
        max_seq_len_hint=100,
    )
    assert seen == [("fv100_decode_paged_xqa", torch.float16)]


# ------------------------------------------------------------ dispatch wiring


def _metadata(num_decodes, seq_lens, max_seq_len):
    return SimpleNamespace(
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        decode_rows=num_decodes,
        decode_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        decode_block_table=torch.zeros((num_decodes, 8), dtype=torch.int32),
        max_seq_len=max_seq_len,
        max_model_len=4096,
        cudagraph_capture=False,
    )


class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call


@pytest.fixture
def rec(fv100, monkeypatch):
    r = _Rec()
    monkeypatch.setattr(fv100, "_op", r)
    return r


def _run(rec, *, heads, head_size, window, sinks, capture=False):
    from vllm_sm70.attention.flash_v100 import dispatch

    num_heads, num_kv_heads = heads
    q = torch.zeros((2, num_heads, head_size), dtype=torch.float16)
    k = torch.zeros((8, 16, num_kv_heads, head_size), dtype=torch.float16)
    md = _metadata(2, [100, 30], 100)
    md.cudagraph_capture = capture
    out = torch.empty_like(q)
    dispatch.run_decode(q, k, k, out, md, scale=0.1, window=window, sinks=sinks)
    return rec.calls[-1]


def test_sink_layer_passes_sinks_and_window(rec):
    sinks = torch.zeros(16, dtype=torch.float32)
    name, args = _run(rec, heads=(16, 2), head_size=128, window=(512, 0), sinks=sinks)
    assert name == "fv100_decode_paged_sinks"
    assert args[-1] is sinks and args[-3:-1] == (512, 0)


def test_sink_free_layer_takes_the_sink_free_op(rec):
    name, args = _run(rec, heads=(16, 2), head_size=128, window=(-1, -1), sinks=None)
    assert name == "fv100_decode_paged" and len(args) == 18


def test_xqa_never_takes_a_sink_layer(rec):
    from vllm_sm70.attention.flash_v100 import dispatch

    sinks = torch.zeros(8, dtype=torch.float32)
    name, _ = _run(rec, heads=(8, 1), head_size=256, window=(-1, -1), sinks=sinks)
    assert name == "fv100_decode_paged_sinks"
    q = torch.zeros((1, 8, 256), dtype=torch.float16)
    k = torch.zeros((1, 16, 1, 256), dtype=torch.float16)
    assert not dispatch.use_xqa(q, k, (-1, -1), sinks)
    assert not dispatch.use_xqa(q, k, (127, 0), None)
    assert dispatch.use_xqa(q, k, (-1, -1), None)


def test_capture_launches_the_max_model_len_envelope(rec):
    name, args = _run(
        rec, heads=(8, 1), head_size=256, window=(-1, -1), sinks=None, capture=True
    )
    assert name == "fv100_decode_paged_xqa"
    partition, launch = args[11], args[12]
    assert launch * partition >= 4096
    _, eager = _run(rec, heads=(8, 1), head_size=256, window=(-1, -1), sinks=None)
    assert eager[12] * eager[11] < 4096


# --------------------------------------------------------------------- GPU


@pytest.mark.gpu
@torch.inference_mode()
def test_stale_active_num_partitions_does_not_truncate_decode(fv100_gpu):
    device, fv100_mod = fv100_gpu
    torch.manual_seed(0)
    seq_len, block_size = 513, 16
    num_blocks = -(-seq_len // block_size)
    q = torch.randn((1, 8, 128), dtype=torch.float16, device=device)
    k = torch.randn(
        (num_blocks, block_size, 1, 128), dtype=torch.float16, device=device
    )
    v = torch.randn_like(k)
    table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    def run(active):
        return fv100_mod.flash_attn_decode_paged(
            q,
            k,
            v,
            table,
            lens,
            max_seq_len_hint=seq_len,
            workspace_seq_capacity_hint=seq_len,
            active_num_partitions=torch.tensor(
                [active], dtype=torch.int32, device=device
            ),
        ).clone()

    expected, actual = run(3), run(1)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)
