# SPDX-License-Identifier: Apache-2.0
"""Custom-AR pair exchange (VLLM_SM70_TOP1_CUSTOM_AR): CPU checks.

* Switch: in the "sampler" section next to VLLM_SM70_GREEDY_TOKEN_FASTPATH,
  opt-in: default off, only "1" acts.
* Admission (``custom_ar_reject_reason``): every reason, rank-uniform inputs.
* Dispatch (``greedy_tokens``): admitted -> the kernel, never the
  all-gather; switch off, TP1, CPU, no or disabled custom all-reduce,
  missing kernel, too many rows -> upstream ``get_top_tokens``; a failing
  kernel -> the all-gather with a warning.
* Identity: ``local_pair`` builds the pair upstream all-gathers, and the
  kernel's tie rule (mirrored from csrc/custom_ar/top1_custom_ar.cu) picks
  upstream's token on ties across ranks, NaN and -inf.
* Source pins: launch shape and world sizes match the Python limits.
No GPU and no _sm70_C: the kernel is a Python stand-in.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_sm70 import sample_greedy

SM70 = Path(__file__).resolve().parents[2]
CU = SM70 / "csrc" / "custom_ar" / "top1_custom_ar.cu"
SWITCH = "VLLM_SM70_TOP1_CUSTOM_AR"
CUDA = torch.device("cuda", 0)


def _have(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(autouse=True)
def _clear_log_cache():
    sample_greedy._log_once.cache_clear()
    yield
    sample_greedy._log_once.cache_clear()


# ------------------------------------------------------------------ switch


def test_switch_registered_next_to_greedy_fastpath():
    from vllm_sm70 import envs

    assert SWITCH in envs._SECTIONS["sampler"]
    assert "VLLM_SM70_GREEDY_TOKEN_FASTPATH" in envs._SECTIONS["sampler"]
    assert sample_greedy.TOP1_CUSTOM_AR_SWITCH == SWITCH


@pytest.mark.parametrize(
    "value, on",
    [(None, False), ("1", True), ("", False), ("true", False), ("0", False)],
)
def test_switch_default_off_only_one_acts(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert sample_greedy.custom_ar_switch_on() is on


# --------------------------------------------------------------- admission


def _ca(world_size=2, rank=0, **kw):
    ns = SimpleNamespace(
        _ptr=0x1000,
        disabled=False,
        rank=rank,
        world_size=world_size,
        buffer_ptrs=[0x10000 * (i + 1) for i in range(world_size)],
        max_size=8 * 1024 * 1024,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def kernel_built(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(sample_greedy, "custom_ar_kernel_available", lambda: True)


def _missing(attr):
    ca = _ca()
    delattr(ca, attr)
    return ca


@pytest.mark.parametrize(
    "want, ca, rows, tp, device",
    [
        (None, _ca(), 1, 2, CUDA),
        (None, _ca(), 128, 2, CUDA),
        (None, _ca(4), 7, 4, CUDA),
        ("tp1", _ca(), 1, 1, CUDA),
        ("non_cuda", _ca(), 1, 2, torch.device("cpu")),
        ("non_cuda", _ca(), 1, 2, None),
        ("rows", _ca(), 0, 2, CUDA),
        ("rows", _ca(), 129, 2, CUDA),
        ("rows", _ca(max_size=8), 2, 2, CUDA),
        ("no_custom_ar", None, 1, 2, CUDA),
        ("no_custom_ar", _missing("buffer_ptrs"), 1, 2, CUDA),
        ("no_custom_ar", _missing("_ptr"), 1, 2, CUDA),
        ("custom_ar_disabled", _ca(disabled=True), 1, 2, CUDA),
        ("custom_ar_disabled", _ca(_ptr=0), 1, 2, CUDA),
        ("world_size", _ca(2), 1, 4, CUDA),
        ("world_size", _ca(3), 1, 3, CUDA),
        ("multi_node", _ca(buffer_ptrs=[0x10, 0x10]), 1, 2, CUDA),
        ("multi_node", _ca(buffer_ptrs=[0x10]), 1, 2, CUDA),
    ],
)
def test_reject_reason(kernel_built, want, ca, rows, tp, device):
    assert sample_greedy.custom_ar_reject_reason(ca, rows, tp, device) == want


def test_switch_off_and_missing_kernel(kernel_built, monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    assert sample_greedy.custom_ar_reject_reason(_ca(), 1, 2, CUDA) == "switch_off"
    monkeypatch.delenv(SWITCH)
    assert sample_greedy.custom_ar_reject_reason(_ca(), 1, 2, CUDA) == "switch_off"
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(sample_greedy, "custom_ar_kernel_available", lambda: False)
    assert sample_greedy.custom_ar_reject_reason(_ca(), 1, 2, CUDA) == "kernel"


def test_kernel_unavailable_without_sm70_c():
    # The CPU suite has no _sm70_C, so the real probe says no.
    from vllm_sm70 import _ext

    if _ext.is_loaded():
        pytest.skip("_sm70_C is loaded here")
    assert sample_greedy.custom_ar_kernel_available() is False


def test_no_tp_group_means_no_custom_ar():
    # TP1 / CPU / tests: no initialised TP group -> None, never an exception.
    if _have("vllm.distributed.parallel_state"):
        import vllm.distributed.parallel_state as ps

        if getattr(ps, "_TP", None) is not None:
            pytest.skip("a TP group is initialised in this process")
    assert sample_greedy.tp_custom_allreduce() is None


# ------------------------------------------------- the kernel's tie rule


def _kernel_rule(gathered: torch.Tensor) -> torch.Tensor:
    """Python mirror of staged_cross_device_top1_argmax: ranks in order,
    NaN first, then larger value, lowest id on ties."""
    out = []
    for row in gathered.tolist():
        best_v, best_i = -math.inf, 2**63 - 1
        for v, i in row:
            i = round(i)
            vn, bn = math.isnan(v), math.isnan(best_v)
            if (vn and (not bn or i < best_i)) or (
                not vn and not bn and (v > best_v or (v == best_v and i < best_i))
            ):
                best_v, best_i = v, i
        out.append(best_i)
    return torch.tensor(out, dtype=torch.int64)


def _upstream_rule(gathered: torch.Tensor) -> torch.Tensor:
    """UP get_top_tokens after the all-gather (logits_processor.py:241-247)."""
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
    return top.squeeze(-1).to(torch.int64)


def _pairs(values, shard=1000):
    """[rows, ranks, 2] with ids increasing by rank (vocab-parallel order)."""
    rows = []
    for row in values:
        rows.append([[v, r * shard + (7 * r + 3) % shard] for r, v in enumerate(row)])
    return torch.tensor(rows, dtype=torch.float32)


NAN, INF = math.nan, math.inf


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [-INF, -INF], [-INF, 0.0]],
        [[NAN, 1.0], [1.0, NAN], [NAN, NAN]],
        [[0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 2.0, 1.0], [-1.0, -2.0, -1.0, NAN]],
        [[float(v) for v in range(8)], [7.0 - v for v in range(8)]],
    ],
)
def test_kernel_tie_rule_equals_upstream_reduction(values):
    gathered = _pairs(values)
    assert torch.equal(_kernel_rule(gathered), _upstream_rule(gathered))


def test_kernel_tie_rule_random():
    g = torch.Generator().manual_seed(0)
    for ranks in (2, 4, 8):
        vals = torch.randint(-3, 3, (256, ranks), generator=g).float()
        vals[torch.rand(vals.shape, generator=g) < 0.05] = NAN
        vals[torch.rand(vals.shape, generator=g) < 0.05] = -INF
        gathered = _pairs(vals.tolist(), shard=50000)
        assert torch.equal(_kernel_rule(gathered), _upstream_rule(gathered))


# -------------------------------------------- dispatch, with a fake kernel


@pytest.fixture
def vllm_env(monkeypatch):
    if not _have("vllm.model_executor.layers.logits_processor"):
        pytest.skip("vLLM Python source not importable")
    import vllm.config.vllm as vc
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(
        vc,
        "_current_vllm_config",
        SimpleNamespace(model_config=SimpleNamespace(head_dtype=None)),
    )

    def set_tp(world_size, rank):
        monkeypatch.setattr(
            ps, "_TP", SimpleNamespace(world_size=world_size, rank_in_group=rank)
        )

    set_tp(1, 0)
    return set_tp


def _shards(vllm_env, weight, tp_size, *, poison_padding=False):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    vocab, hidden = weight.shape
    heads = []
    for rank in range(tp_size):
        vllm_env(tp_size, rank)
        head = ParallelLMHead(vocab, hidden, params_dtype=weight.dtype)
        head.weight_loader(head.weight, weight)
        pad = head.shard_indices.num_org_vocab_padding
        if poison_padding and pad:
            head.weight.data[-pad:] = 64.0
        heads.append(head)
    return heads


def _reference(lp, heads, hidden):
    parts = [lp._apply_head(h, hidden, None) for h in heads]
    logits = torch.cat(parts, dim=-1)[..., : lp.org_vocab_size]
    return logits.to(torch.float32).argmax(dim=-1).to(torch.int32)


class _Exchange:
    """Fake TP world: records every rank's pair in a first pass, then serves
    the all-gather and the custom-AR kernel from them in a second pass."""

    def __init__(self, monkeypatch, tp_size):
        import vllm.model_executor.layers.logits_processor as lp_mod

        self.tp_size = tp_size
        self.pairs: list[torch.Tensor] = []
        self.kernel_pairs: list[torch.Tensor] = []
        self.gathers = 0
        self.kernel_calls = []
        self.fail = False
        self.phase = "record"
        self.rank = 0
        monkeypatch.setattr(lp_mod, "tensor_model_parallel_all_gather", self.gather)
        monkeypatch.setattr(sample_greedy, "_kernel", lambda: self.kernel)
        monkeypatch.setattr(sample_greedy, "custom_ar_kernel_available", lambda: True)

    def gather(self, t, dim=-1):
        self.gathers += 1
        if self.phase == "record":
            self.pairs.append(t.clone())
            return torch.cat([t] * self.tp_size, dim=dim)
        return torch.cat(self.pairs, dim=dim)

    def kernel(self, fa, pair, out, reg_buffer, reg_buffer_sz_bytes):
        assert pair.dtype == torch.float32 and pair.is_contiguous()
        assert out.dtype == torch.int64 and out.shape == (pair.shape[0],)
        self.kernel_calls.append((fa, reg_buffer, reg_buffer_sz_bytes))
        if self.fail:
            raise RuntimeError("custom-AR top-1: boom")
        if self.phase == "record":
            self.kernel_pairs.append(pair.clone())
            out.zero_()
            return
        gathered = torch.stack(self.kernel_pairs, dim=1)
        out.copy_(_kernel_rule(gathered))


def _route_to_cuda(monkeypatch, ca_for_rank):
    """CPU tensors stand in for CUDA ones: admission sees a CUDA device,
    and the TP group's communicator is the given namespace."""
    real = sample_greedy.custom_ar_reject_reason

    def reject(ca, rows, tp, device):
        return real(ca, rows, tp, CUDA if device is not None else None)

    monkeypatch.setattr(sample_greedy, "custom_ar_reject_reason", reject)
    monkeypatch.setattr(sample_greedy, "tp_custom_allreduce", ca_for_rank)


def _run(vllm_env, ex, lp, heads, hidden):
    out = []
    for phase in ("record", "exchange"):
        ex.phase = phase
        out = []
        for rank, head in enumerate(heads):
            vllm_env(len(heads), rank)
            ex.rank = rank
            tokens = sample_greedy.greedy_tokens(lp, head, hidden)
            assert tokens.dtype == torch.int32
            out.append(tokens)
    return out


def _setup(monkeypatch, vllm_env, tp_size=2, vocab=509, rows=3, seed=0, **kw):
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(vocab, 32, generator=g).to(torch.float16)
    hidden = torch.randn(rows, 32, generator=g).to(torch.float16)
    heads = _shards(vllm_env, weight, tp_size, **kw)
    lp = LogitsProcessor(vocab)
    ex = _Exchange(monkeypatch, tp_size)
    return lp, heads, hidden, ex


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("rows", [1, 5, 128])
def test_admitted_takes_the_kernel_and_equals_argmax(
    monkeypatch, vllm_env, caplog, tp_size, rows
):
    monkeypatch.setenv(SWITCH, "1")
    lp, heads, hidden, ex = _setup(
        monkeypatch, vllm_env, tp_size, rows=rows, poison_padding=True
    )
    cas = [_ca(tp_size, rank=r) for r in range(tp_size)]
    _route_to_cuda(monkeypatch, lambda: cas[ex.rank])
    with caplog.at_level(logging.INFO, logger="vllm.sm70"):
        got = _run(vllm_env, ex, lp, heads, hidden)
    want = _reference(lp, heads, hidden)
    for tokens in got:
        assert torch.equal(tokens, want)
    assert ex.gathers == 0
    assert len(ex.kernel_calls) == 2 * tp_size
    # The rank's handle, its own registered buffer and the buffer size.
    for (fa, buf, size), r in zip(ex.kernel_calls, list(range(tp_size)) * 2):
        assert (fa, buf, size) == (cas[r]._ptr, cas[r].buffer_ptrs[r], cas[r].max_size)
    assert caplog.text.count("SM70 custom top1 argmax path enabled.") == 1


def test_local_pair_is_the_pair_get_top_tokens_gathers(monkeypatch, vllm_env):
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env, 2, poison_padding=True)
    for rank, head in enumerate(heads):
        vllm_env(2, rank)
        lp.get_top_tokens(head, hidden)
        assert torch.equal(sample_greedy.local_pair(lp, head, hidden), ex.pairs[rank])


def _fallback_case(monkeypatch, vllm_env, ca_for_rank, **kw):
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env, **kw)
    _route_to_cuda(monkeypatch, ca_for_rank)
    got = _run(vllm_env, ex, lp, heads, hidden)
    want = _reference(lp, heads, hidden)
    for tokens in got:
        assert torch.equal(tokens, want)
    return ex


@pytest.mark.parametrize("value", [None, "0"])
def test_switch_unset_or_zero_uses_the_all_gather(monkeypatch, vllm_env, value):
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    ex = _fallback_case(monkeypatch, vllm_env, lambda: _ca())
    assert ex.kernel_calls == [] and ex.gathers == 4


@pytest.mark.parametrize(
    "ca, reason",
    [
        (None, "no_custom_ar"),
        (_ca(disabled=True), "custom_ar_disabled"),
        (_ca(buffer_ptrs=[0x10, 0x10]), "multi_node"),
    ],
)
def test_uninitialised_or_disabled_custom_ar_uses_the_all_gather(
    monkeypatch, vllm_env, caplog, ca, reason
):
    monkeypatch.setenv(SWITCH, "1")
    with caplog.at_level(logging.INFO, logger="vllm.sm70"):
        ex = _fallback_case(monkeypatch, vllm_env, lambda: ca)
    assert ex.kernel_calls == [] and ex.gathers == 4
    assert caplog.text.count(f"SM70 custom top1 argmax unavailable ({reason})") == 1


def test_missing_kernel_uses_the_all_gather(monkeypatch, vllm_env):
    monkeypatch.setenv(SWITCH, "1")
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env)
    monkeypatch.setattr(sample_greedy, "custom_ar_kernel_available", lambda: False)
    _route_to_cuda(monkeypatch, lambda: _ca())
    _run(vllm_env, ex, lp, heads, hidden)
    assert ex.kernel_calls == [] and ex.gathers == 4


def test_too_many_rows_use_the_all_gather(monkeypatch, vllm_env):
    monkeypatch.setenv(SWITCH, "1")
    ex = _fallback_case(monkeypatch, vllm_env, lambda: _ca(), rows=129)
    assert ex.kernel_calls == [] and ex.gathers == 4


def test_cpu_tensors_never_look_up_the_communicator(monkeypatch, vllm_env):
    # The real admission on CPU tensors: non_cuda before any TP-group lookup.
    monkeypatch.setenv(SWITCH, "1")
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env)

    def boom():
        raise AssertionError("communicator looked up on CPU")

    monkeypatch.setattr(sample_greedy, "tp_custom_allreduce", boom)
    _run(vllm_env, ex, lp, heads, hidden)
    assert ex.kernel_calls == [] and ex.gathers == 4


def test_tp1_has_no_exchange(monkeypatch, vllm_env):
    monkeypatch.setenv(SWITCH, "1")
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env, tp_size=1)
    _route_to_cuda(monkeypatch, lambda: _ca())
    got = _run(vllm_env, ex, lp, heads, hidden)
    assert torch.equal(got[0], _reference(lp, heads, hidden))
    assert ex.kernel_calls == [] and ex.gathers == 0


def test_kernel_failure_falls_back_with_a_warning(monkeypatch, vllm_env, caplog):
    monkeypatch.setenv(SWITCH, "1")
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env)
    ex.fail = True
    _route_to_cuda(monkeypatch, lambda: _ca(rank=ex.rank))
    with caplog.at_level(logging.INFO, logger="vllm.sm70"):
        got = _run(vllm_env, ex, lp, heads, hidden)
    want = _reference(lp, heads, hidden)
    for tokens in got:
        assert torch.equal(tokens, want)
    assert len(ex.kernel_calls) == 4 and ex.gathers == 4
    assert "falling back to all_gather" in caplog.text
    assert "path enabled" not in caplog.text


# -------------------------------------------------------------- debug log

DEBUG = "VLLM_SM70_TOP1_DEBUG"


@pytest.fixture
def debug_env(monkeypatch):
    def set_debug(value):
        if value is None:
            monkeypatch.delenv(DEBUG, raising=False)
        else:
            monkeypatch.setenv(DEBUG, value)
        sample_greedy.debug_on.cache_clear()

    yield set_debug
    sample_greedy.debug_on.cache_clear()


@pytest.mark.parametrize("switch, path", [("1", "custom_ar"), (None, "all_gather")])
def test_debug_logs_pairs_and_tokens(
    monkeypatch, vllm_env, caplog, debug_env, switch, path
):
    debug_env("1")
    if switch is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, switch)
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env)
    _route_to_cuda(monkeypatch, lambda: _ca(rank=ex.rank))
    with caplog.at_level(logging.INFO, logger="vllm.sm70"):
        got = _run(vllm_env, ex, lp, heads, hidden)
    lines = [r.getMessage() for r in caplog.records if "top1 debug" in r.getMessage()]
    # Two passes x two ranks; the last two are the exchange pass.
    assert len(lines) == 4
    for rank, line in enumerate(lines[2:]):
        assert f"rank={rank} " in line and "rows=3 " in line
        assert f"path={path} " in line
        pair = sample_greedy.local_pair(lp, heads[rank], hidden)
        for (value, index), token in zip(pair.tolist(), got[rank].tolist()):
            assert f"({value!r}, {int(index)}) -> {token}" in line


def test_debug_off_logs_nothing_and_builds_no_extra_pair(
    monkeypatch, vllm_env, caplog, debug_env
):
    debug_env(None)
    monkeypatch.delenv(SWITCH, raising=False)
    lp, heads, hidden, ex = _setup(monkeypatch, vllm_env)
    _route_to_cuda(monkeypatch, lambda: _ca(rank=ex.rank))

    def boom(*a, **k):
        raise AssertionError("local_pair built with the debug log off")

    monkeypatch.setattr(sample_greedy, "local_pair", boom)
    with caplog.at_level(logging.DEBUG, logger="vllm.sm70"):
        _run(vllm_env, ex, lp, heads, hidden)
    assert "top1 debug" not in caplog.text


def test_debug_skipped_under_capture(monkeypatch, debug_env):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    hidden = SimpleNamespace(is_cuda=True)
    # Would raise on the namespaces if it went past the capture check.
    sample_greedy._debug_log("custom_ar", None, None, hidden, None, None)


# ------------------------------------------------------------ source pins


def test_kernel_source_matches_python_limits():
    src = CU.read_text()
    threads = int(re.search(r"constexpr int kTop1Threads = (\d+);", src).group(1))
    assert threads == sample_greedy.MAX_CUSTOM_AR_ROWS
    assert "constexpr int kTop1MaxRows = kTop1Threads;" in src
    cases = tuple(int(n) for n in re.findall(r"VLLM_SM70_TOP1_CASE\((\d+)\)\n", src))
    assert cases == sample_greedy.CUSTOM_AR_WORLD_SIZES
    assert (
        f'"{sample_greedy.KERNEL_NAME}(int fa, Tensor pair, Tensor(a!) output, "' in src
    )
    assert sample_greedy.KERNEL_NAMESPACE == "_sm70_C"
    assert "TORCH_LIBRARY_FRAGMENT(_sm70_C, m)" in src


def test_cmake_builds_the_kernel():
    cmake = (SM70 / "csrc" / "custom_ar" / "sources.cmake").read_text()
    assert '"${_VLLM_SM70_CUSTOM_AR_ROOT}/top1_custom_ar.cu"' in cmake
