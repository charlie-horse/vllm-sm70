# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 greedy top-1 (inkling2_sm70/greedy.py, ops/top1.py), on CPU.

- the kill switch VLLM_SM70_INKLING_GREEDY_TOP1 and the model's opt-in to
  the V1 sampler hook; the top-k candidate route refuses the soft cap;
- tensors off the GPU take upstream's ``get_top_tokens``;
- through Triton's interpreter, the model's ``get_top_tokens`` on one and on
  two vocabulary shards (a stand-in pair all-gather) against upstream's
  greedy token, the argmax of the gathered soft-capped fp32 logits: random
  logits, exact ties inside a block, across blocks, splits and shards,
  ties made by the cap's saturation, no cap, and batches of 1 to 8 rows.

The interpreted cases run in a child pytest with TRITON_INTERPRET=1
(test_interpreted_suite). The AOT compile check is
tests/ops/test_m12_inkling2_triton_sm70_compile_cpu.py.
"""

from __future__ import annotations

import importlib.util
import os
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

SWITCH = "VLLM_SM70_INKLING_GREEDY_TOP1"
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(not INTERPRETED, reason="interpreted child run only")
CAP = 30.0
HIDDEN = 8


@pytest.fixture
def env(monkeypatch):
    """What the logits processor reads without an engine, and a stand-in
    pair all-gather that records each rank's pairs, then exchanges them."""
    import vllm.config.vllm as vc
    import vllm.distributed as dist
    import vllm.model_executor.layers.logits_processor as lp_mod

    monkeypatch.setattr(
        vc,
        "_current_vllm_config",
        SimpleNamespace(model_config=SimpleNamespace(head_dtype=None)),
    )
    state = SimpleNamespace(pairs=[], exchange=False, calls=0)

    def all_gather(t, dim=-1):
        state.calls += 1
        if state.exchange:
            return torch.cat(state.pairs, dim=dim)
        state.pairs.append(t.clone())
        return torch.cat([t] * 2, dim=dim)

    for module in (dist, lp_mod):
        monkeypatch.setattr(module, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.delenv(SWITCH, raising=False)
    return state


def _model():
    from vllm_sm70.models.inkling2_sm70 import model

    return model


def _processor(soft_cap=CAP):
    return _model().Inkling2Sm70LogitsProcessor(
        VOCAB_OF[soft_cap], org_vocab_size=VOCAB_OF[soft_cap], soft_cap=soft_cap
    )


VOCAB_OF: dict = {}


def _heads(values: torch.Tensor, tp: int):
    """LM-head shards whose logits for hidden ``e_0`` are `values` [rows=1
    per hidden row is chosen by `_hidden`]: column 0 of the weight holds a
    row's logits, so the fp16 GEMM returns them exactly."""
    rows, vocab = values.shape
    heads = []
    for rank in range(tp):
        head = _model().Inkling2Sm70LMHead(
            vocab, rows, torch.float16, tp_size=tp, tp_rank=rank
        )
        share = vocab // tp
        head.weight.data.copy_(values[:, rank * share : (rank + 1) * share].t())
        heads.append(head)
    return heads


def _hidden(rows: int) -> torch.Tensor:
    return torch.eye(rows, dtype=torch.float16)


def _reference(values: torch.Tensor, soft_cap) -> torch.Tensor:
    """Upstream's greedy token: argmax over the soft-capped fp32 logits."""
    logits = values.float()
    if soft_cap is not None:
        logits = torch.tanh(logits / soft_cap) * soft_cap
    return logits.argmax(dim=-1)


def _tokens(env, values: torch.Tensor, tp: int, soft_cap=CAP) -> list[torch.Tensor]:
    VOCAB_OF[soft_cap] = values.shape[1]
    lp = _processor(soft_cap)
    heads = _heads(values, tp)
    hidden = _hidden(values.shape[0])
    env.pairs.clear()
    env.exchange = False
    if tp > 1:
        for head in heads:
            lp.get_top_tokens(head, hidden)
        env.exchange = True
    return [lp.get_top_tokens(head, hidden) for head in heads]


# --- switch, opt-in and route -------------------------------------------------------


def test_switch_and_opt_in(env, monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import greedy
    from vllm_sm70.sample_v1 import MODEL_OPT_IN

    cls = _model().Inkling2LoopSm70ForConditionalGeneration
    model = object.__new__(cls)
    assert greedy.SWITCH == SWITCH
    assert greedy.enabled() and getattr(model, MODEL_OPT_IN) is True
    for value in ("1", "", "off"):
        monkeypatch.setenv(SWITCH, value)
        assert getattr(model, MODEL_OPT_IN) is True
    monkeypatch.setenv(SWITCH, "0")
    assert getattr(model, MODEL_OPT_IN) is False


def test_topk_route_refuses_the_soft_cap(env):
    from vllm_sm70 import sample, sample_greedy

    VOCAB_OF[CAP] = 64
    lp = _processor()
    (head,) = _heads(torch.zeros(HIDDEN, 64), 1)
    assert sample.lm_head_reject_reason(lp, head, 2) == "logits_processor_transform"
    assert sample_greedy.lm_head_reject_reason(lp, head) is None


@pytest.mark.skipif(INTERPRETED, reason="the interpreter runs the kernels on CPU")
@pytest.mark.parametrize("tp", [1, 2])
def test_cpu_tensors_take_upstream(env, tp):
    from vllm_sm70.models.inkling2_sm70 import greedy

    torch.manual_seed(0)
    values = (torch.randn(3, 512) * 8).half()
    VOCAB_OF[CAP] = 512
    (head, *_) = _heads(values, tp)
    assert not greedy.on_kernels(_processor(), head, _hidden(3))
    for got in _tokens(env, values, tp):
        assert torch.equal(got, _reference(values, CAP))
    # Upstream gathers one pair per row and rank.
    if tp > 1:
        assert [p.shape for p in env.pairs] == [(3, 2)] * 2


# --- interpreted: kernels against upstream's token ------------------------------------


def _cases():
    torch.manual_seed(1)
    vocab = 2 * 9000  # 3 splits per shard at TP2, 5 at TP1
    random = (torch.randn(8, vocab) * 10).half()
    ties = torch.full((6, vocab), -5.0, dtype=torch.float16)
    ties[0, [7, 900, 1023]] = 12.0  # inside one block
    ties[1, [1500, 3000]] = 12.0  # across blocks of one split
    ties[2, [4095, 4096, 8000]] = 12.0  # across splits
    ties[3, [8999, 9000, 17999]] = 12.0  # across shards at TP2
    ties[4, 9001] = 310.0  # the cap saturates: these are equal after it
    ties[4, 17000] = 330.0
    ties[5, -1] = 1.0  # the last column
    return {"random": random, "ties": ties}


@interpreted
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("case", ["random", "ties"])
@pytest.mark.parametrize("soft_cap", [CAP, None], ids=["cap", "no_cap"])
def test_interpreted_tokens(env, tp, case, soft_cap):
    from vllm_sm70.models.inkling2_sm70 import greedy

    values = _cases()[case]
    VOCAB_OF[soft_cap] = values.shape[1]
    (head, *_) = _heads(values, tp)
    assert greedy.on_kernels(_processor(soft_cap), head, _hidden(values.shape[0]))
    want = _reference(values, soft_cap)
    for got in _tokens(env, values, tp, soft_cap):
        assert got.dtype == torch.int64
        assert torch.equal(got, want), (got, want)
    if tp > 1:
        # One (value, id) pair per split and rank, not the logits.
        splits = -(-values.shape[1] // tp // 4096)
        assert [p.shape for p in env.pairs] == [(values.shape[0], 2 * splits)] * 2


@interpreted
def test_interpreted_saturation_tie_takes_the_lower_id(env):
    values = _cases()["ties"]
    assert int(_reference(values, CAP)[4]) == 9001
    assert int(_reference(values, None)[4]) == 17000
    for got in _tokens(env, values, 2):
        assert int(got[4]) == 9001


@interpreted
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_interpreted_batch_rows(env, rows):
    torch.manual_seed(rows)
    values = (torch.randn(rows, 4 * 1031) * 6).half()
    for got in _tokens(env, values, 2):
        assert torch.equal(got, _reference(values, CAP))


@interpreted
def test_interpreted_pick_order():
    from vllm_sm70.models.inkling2_sm70.ops import top1

    pairs = torch.tensor(
        [[[1.0, 5.0], [3.0, 9.0], [3.0, 2.0], [-1.0, 0.0]]], dtype=torch.float32
    )
    assert top1.pick(pairs).tolist() == [9]


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
    assert passed and int(passed.group(1)) >= 14, summary
    assert "skipped" not in summary, summary
