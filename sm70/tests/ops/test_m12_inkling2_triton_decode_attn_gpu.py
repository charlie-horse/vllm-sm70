# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 paged decode-attention kernel
(inkling2_sm70/ops/paged_attention.py) on an SM70 GPU.

The cases of test_m12_inkling2_triton_decode_attn_cpu.py
(_inkling2_decode_attn.py), compiled for sm_70, against the torch route
(attention_torch.py ``gather_attention``) on the same GPU: fp16 within 2e-3
and fp32 within 1e-5 of the largest reference value. Plus Inkling 2 Mini
shapes per TP2 rank: 12 heads, 4 KV heads, head dim 128, 32-token blocks,
contexts up to 33K tokens. Split-KV (the default) against the unsplit
kernel (VLLM_SM70_INKLING_SPLIT_KV_ATTN=0), and inside a captured CUDA
graph replayed at other positions.

    pytest -m gpu sm70/tests/ops/test_m12_inkling2_triton_decode_attn_gpu.py
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

DTYPES = [
    pytest.param(torch.float32, 1e-5, id="fp32"),
    pytest.param(torch.float16, 2e-3, id="fp16"),
]
LAYERS = pytest.mark.parametrize(
    "window, extent",
    [(None, 8), (None, 48), (40, 8), (64, 48)],
    ids=["global", "global-long-bias", "sliding", "sliding-long-bias"],
)


@pytest.fixture
def cases(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    import _inkling2_decode_attn

    attention = _inkling2_decode_attn.attention_module()
    monkeypatch.delenv(_inkling2_decode_attn.SWITCH, raising=False)
    assert attention.on_triton(torch.zeros(1, device="cuda"))
    return _inkling2_decode_attn


def _cuda(case):
    for name in ("cache", "query", "rel", "sinks"):
        setattr(case, name, getattr(case, name).cuda())
    md = case.md
    for name in (
        "positions",
        "seq_idx",
        "slot_mapping",
        "block_table",
        "query_start_loc",
    ):
        setattr(md, name, getattr(md, name).cuda())
    return case


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@LAYERS
@pytest.mark.parametrize(
    "seq_lens, pad",
    [([1], 0), ([33], 3), ([7, 70, 32, 160], 0), ([300, 2, 64], 5)],
    ids=["first-token", "one-padded", "four", "long-padded"],
)
def test_decode_only_batch_matches_the_torch_route(
    cases, monkeypatch, dtype, tol, window, extent, seq_lens, pad
):
    case = _cuda(cases.Case(dtype, window, extent, seq_lens, pad=pad))
    got, want = cases.both_routes(case, monkeypatch)
    assert cases.rel_err(got, want) <= tol
    assert bool((got[len(seq_lens) :] == 0).all())


@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
@LAYERS
@pytest.mark.parametrize(
    "seq_lens, query_lens",
    [([90, 40, 50], [1, 1, 20]), ([90, 50, 40], [1, 20, 1]), ([129, 5], [1, 1])],
    ids=["decode-first", "decode-after-prefill", "all-decode"],
)
def test_eager_one_query_rows_match_the_torch_route(
    cases, monkeypatch, dtype, tol, window, extent, seq_lens, query_lens
):
    case = _cuda(
        cases.Case(
            dtype, window, extent, seq_lens, pad=2, decode_only=False,
            query_lens=query_lens, seed=1,
        )
    )  # fmt: skip
    got, want = cases.both_routes(case, monkeypatch)
    assert cases.rel_err(got, want) <= tol
    assert bool((got[sum(query_lens) :] == 0).all())


@pytest.mark.parametrize(
    "window, extent, seq_lens",
    [
        (None, 1024, [4000]),
        (1024, 1024, [4000]),
        (None, 1024, [33000]),
        (1024, 1024, [3000, 17, 1500, 900]),
    ],
    ids=["mini-global-4k", "mini-window-4k", "mini-global-33k", "mini-window-4req"],
)
def test_mini_shapes_match_the_torch_route(
    cases, monkeypatch, window, extent, seq_lens
):
    monkeypatch.setattr(cases, "HEADS", 12)
    monkeypatch.setattr(cases, "KV_HEADS", 4)
    monkeypatch.setattr(cases, "HEAD_DIM", 128)
    blocks = sum(-(-n // cases.BLOCK) for n in seq_lens) + 8
    monkeypatch.setattr(cases, "NUM_BLOCKS", blocks)
    case = _cuda(cases.Case(torch.float16, window, extent, seq_lens))
    got, want = cases.both_routes(case, monkeypatch)
    assert cases.rel_err(got, want) <= 2e-3


SPLIT_SWITCH = "VLLM_SM70_INKLING_SPLIT_KV_ATTN"


def _mini(cases, monkeypatch, seq_lens):
    monkeypatch.setattr(cases, "HEADS", 12)
    monkeypatch.setattr(cases, "KV_HEADS", 4)
    monkeypatch.setattr(cases, "HEAD_DIM", 128)
    blocks = sum(-(-n // cases.BLOCK) for n in seq_lens) + 8
    monkeypatch.setattr(cases, "NUM_BLOCKS", blocks)


@pytest.mark.parametrize(
    "window, seq_lens",
    [(None, [60]), (None, [4000]), (None, [33000]), (1024, [4000]),
     (1024, [3000, 17, 1500, 900]), (None, [33000, 5, 900, 4000])],
    ids=["short", "global-4k", "global-33k", "window-4k", "window-4req",
         "global-4req"],
)  # fmt: skip
def test_split_matches_the_unsplit_kernel(cases, monkeypatch, window, seq_lens):
    _mini(cases, monkeypatch, seq_lens)
    attention = cases.attention_module()
    case = _cuda(cases.Case(torch.float16, window, 1024, seq_lens, pad=3))
    monkeypatch.delenv(SPLIT_SWITCH, raising=False)
    got = case.run(attention)
    monkeypatch.setenv(SPLIT_SWITCH, "0")
    want = case.run(attention)
    assert cases.rel_err(got, want) <= 2e-3
    assert bool((got[len(seq_lens) :] == 0).all())


@pytest.mark.parametrize("window", [None, 1024], ids=["global", "sliding"])
def test_split_replays_in_a_cuda_graph(cases, monkeypatch, window):
    """Captured once at the longest context, replayed at shorter positions:
    each replay matches the eager torch route at those positions."""
    _mini(cases, monkeypatch, [33000])
    attention = cases.attention_module()
    monkeypatch.delenv(SPLIT_SWITCH, raising=False)
    case = _cuda(cases.Case(torch.float16, window, 1024, [33000]))
    # The case writes only the last window of keys; earlier replays read more.
    case.cache = torch.where(
        case.cache.isnan(), torch.randn_like(case.cache), case.cache
    )
    out = torch.zeros_like(case.query)

    def run():
        attention.gather_attention(
            case.query, case.rel, case.sinks, case.cache, case.md, out
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for position in (0, 31, 100, 1500, 4000, 32999):
        case.md.positions.fill_(position)
        graph.replay()
        torch.cuda.synchronize()
        got = out.clone()
        monkeypatch.setenv(cases.SWITCH, "0")
        want = case.run(attention)
        monkeypatch.delenv(cases.SWITCH)
        assert cases.rel_err(got, want) <= 2e-3, position
