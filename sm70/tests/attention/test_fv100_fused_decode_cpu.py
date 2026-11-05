# SPDX-License-Identifier: Apache-2.0
"""K14 (C4) CPU checks: one-launch sink decode routing, counters, capture.

Port of the CPU cases of fork cf9492a
tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py :967-1316
(fused entry point, counter allocation and capture rules, the prepare hook,
the optional per-launch zero, the impl hook, capture counts, the kill switch
and the grouped-shape gate), rewritten against vllm_sm70.ops.fv100 and
vllm_sm70.attention.flash_v100.dispatch:

* The fork injected a pybind module; here `fv100._op` is a recorder and
  `fv100.has_op` reports the fused op as built (or not).
* Argument positions follow the `_sm70_C` schemas: out first, no anchored
  arguments, so sinks is args[18] and counters args[19] (fork 20/21).
* Dropped: the `_runtime` policy cases (fork :1304-1316 and the runtime
  half of :1240-1266), the anchored-window shape (no anchored kernels in
  the lean build), the routing-probe internals of :1199-1237.
* New: the admitted shape set for Flint TP1/TP2/TP4 on 2xV100, a CPU model
  of vLLM's descending capture order (R4), and the impl wiring (counters at
  layer construction, before any capture).
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# K06's builder fixture, LBNHC forward helper and stable-libtorch stub.
from test_fv100_backend_cpu import (  # noqa: E402,F401
    _forward_case,
    _stub_stable_libtorch,
    builder,
)

FUSED_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE"
ZERO_ENV = "VLLM_SM70_FLINT_ATTN_FUSED_DECODE_ZERO_COUNTERS"
SWA_WINDOW = (512, 0)
HEAD_DIM = 128
# Schema positions (fv100_decode_paged_sinks_fused).
TMP_OUT, SINKS, COUNTERS = 6, 18, 19
FUSED = "fv100_decode_paged_sinks_fused"
TWO = "fv100_decode_paged_sinks"


class _Rec:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def __call__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call

    @property
    def names(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def load(monkeypatch):
    """load(with_fused=True) -> (fv100 module, recorder)."""
    from vllm_sm70.ops import fv100

    monkeypatch.delenv(FUSED_ENV, raising=False)
    monkeypatch.delenv(ZERO_ENV, raising=False)
    real_has_op = fv100.has_op

    def _load(with_fused: bool = True):
        monkeypatch.setattr(
            fv100,
            "has_op",
            lambda name: (
                with_fused if name == fv100.FUSED_DECODE_OP else real_has_op(name)
            ),
        )
        rec = _Rec()
        monkeypatch.setattr(fv100, "_op", rec)
        monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: False)
        fv100.clear_workspaces()
        return fv100, rec

    yield _load
    fv100.clear_workspaces()


def _inputs(num_heads=16, num_kv_heads=2, head_dim=HEAD_DIM, batch=2):
    q = torch.zeros((batch, num_heads, head_dim), dtype=torch.float16)
    k_cache = torch.zeros((4, 16, num_kv_heads, head_dim), dtype=torch.float16)
    block_table = torch.zeros((batch, 2), dtype=torch.int32)
    seq_lens = torch.tensor([20, 7, 5, 3][:batch], dtype=torch.int32)
    return q, k_cache, k_cache.clone(), block_table, seq_lens


def _sinks(n=16):
    return torch.zeros(n, dtype=torch.float32)


def _capturing(monkeypatch, fv100, active: bool) -> None:
    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: active)


def _warnings(calls):
    return [m for m in calls if "one-launch sink decode" in m]


@pytest.fixture
def warned(load, monkeypatch):
    from vllm_sm70.ops import fv100

    seen: list[str] = []
    monkeypatch.setattr(
        fv100.logger, "warning", lambda msg, *args: seen.append(msg % args)
    )
    return seen


# ------------------------------------------------------------ routing (fork)


def test_grouped_sinks_take_the_fused_op(load):
    """fork :967-985."""
    fv100, rec = load()
    q, k, v, table, lens = _inputs()
    sinks = _sinks()
    for _ in range(2):
        fv100.flash_attn_decode_paged(
            q, k, v, table, lens, window_size=SWA_WINDOW, sinks=sinks
        )
    assert rec.names == [FUSED] * 2
    (_, first), (_, second) = rec.calls
    assert len(first) == 20 and first[SINKS] is sinks
    counters = first[COUNTERS]
    # One zeroed int32 counter per (row, KV head) at least, reused across
    # calls (graph replays keep its address) with the shared workspace.
    assert counters.dtype == torch.int32 and counters.is_contiguous()
    assert counters.numel() >= q.shape[0] * k.shape[2]
    assert torch.count_nonzero(counters) == 0
    assert second[COUNTERS] is counters
    assert first[TMP_OUT].data_ptr() == second[TMP_OUT].data_ptr()
    assert first[16:18] == SWA_WINDOW


def test_sink_free_call_never_takes_the_fused_op(load):
    fv100, rec = load()
    q, k, v, table, lens = _inputs()
    fv100.flash_attn_decode_paged(q, k, v, table, lens)
    assert rec.names == ["fv100_decode_paged"]
    assert fv100._fused_decode_counter_cache == {}


def test_counters_are_never_created_under_capture(load, monkeypatch, warned):
    """fork :992-1012."""
    fv100, rec = load()
    q, k, v, table, lens = _inputs()
    _capturing(monkeypatch, fv100, True)
    for _ in range(2):
        fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=_sinks())
    assert rec.names == [TWO] * 2
    assert all(len(args) == 19 for _, args in rec.calls)
    assert fv100._fused_decode_counter_cache == {}
    (message,) = _warnings(warned)
    assert "no counters were allocated before capture" in message
    assert "flash_attn_decode_paged_sinks_prepare" in message


def test_counter_capacity_overflow_falls_back(load, monkeypatch, warned):
    """fork :1015-1052: a capture needing more counters takes two launches,
    allocating nothing."""
    fv100, rec = load()
    monkeypatch.setattr(fv100, "FUSED_DECODE_COUNTER_CAPACITY", 2)
    q, k, v, table, lens = _inputs()
    sinks = _sinks()
    fv100.flash_attn_decode_paged(q[:1], k, v, table[:1], lens[:1], sinks=sinks)
    ((key, entry),) = fv100._fused_decode_counter_cache.items()
    buffer = entry.buffer
    assert buffer.numel() == 2  # one row x two KV heads
    zeros: list[tuple] = []
    real_zeros = torch.zeros

    def counting_zeros(*args, **kwargs):
        zeros.append(args)
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(fv100.torch, "zeros", counting_zeros)
    _capturing(monkeypatch, fv100, True)
    for _ in range(2):
        fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert rec.names == [FUSED, TWO, TWO]
    assert fv100._fused_decode_counter_cache == {key: entry}
    assert entry.buffer is buffer and entry.retired == []
    assert zeros == []
    (message,) = _warnings(warned)
    assert "capture needs 4 counters, 2 were allocated" in message


def test_counters_normalise_the_cuda_device(load, monkeypatch):
    """fork :1055-1062."""
    fv100, _ = load()
    monkeypatch.setattr(fv100.torch.accelerator, "current_device_index", lambda: 3)
    assert fv100._normalized_device(torch.device("cuda")) == torch.device("cuda", 3)
    assert fv100._normalized_device(torch.device("cuda:1")) == torch.device("cuda", 1)
    assert fv100._normalized_device(torch.device("cpu")) == torch.device("cpu")


def test_counters_allocated_eagerly_serve_later_captures(load, monkeypatch):
    """fork :1065-1081."""
    fv100, rec = load()
    q, k, v, table, lens = _inputs()
    sinks = _sinks()
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    ((_, entry),) = fv100._fused_decode_counter_cache.items()
    assert entry.buffer.numel() == fv100.FUSED_DECODE_COUNTER_CAPACITY
    assert entry.buffer.dtype == torch.int32
    assert torch.count_nonzero(entry.buffer) == 0
    _capturing(monkeypatch, fv100, True)
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert rec.names == [FUSED] * 2
    assert all(args[COUNTERS] is entry.buffer for _, args in rec.calls)


def test_counter_capacity_refusal_and_eager_growth(load, monkeypatch):
    """fork :1084-1100."""
    fv100, _ = load()
    device = torch.device("cpu")
    capacity = fv100.FUSED_DECODE_COUNTER_CAPACITY
    first = fv100._fused_decode_counters(device, 8)
    assert first is not None and first.numel() == capacity
    _capturing(monkeypatch, fv100, True)
    assert fv100._fused_decode_counters(device, capacity) is first
    assert fv100._fused_decode_counters(device, capacity + 1) is None
    assert fv100._fused_decode_counter_cache[-1].buffer is first
    _capturing(monkeypatch, fv100, False)
    grown = fv100._fused_decode_counters(device, capacity + 1)
    assert grown is not None and grown.numel() == capacity + 1
    entry = fv100._fused_decode_counter_cache[-1]
    # Graphs captured with the old buffer keep it alive.
    assert entry.buffer is grown and entry.retired == [first]


def test_prepare_hook(load, monkeypatch):
    """fork :1103-1121."""
    fv100, _ = load()
    allocated: list[tuple[torch.device, int]] = []

    def fake_counters(device, required):
        allocated.append((device, required))
        return torch.zeros(1, dtype=torch.int32)

    monkeypatch.setattr(fv100, "_fused_decode_counters", fake_counters)
    assert not fv100.flash_attn_decode_paged_sinks_prepare("cpu")
    assert fv100.flash_attn_decode_paged_sinks_prepare("cuda:1")
    assert allocated == [(torch.device("cuda:1"), 0)]
    _capturing(monkeypatch, fv100, True)
    assert not fv100.flash_attn_decode_paged_sinks_prepare("cuda:1")
    monkeypatch.setenv(FUSED_ENV, "0")
    _capturing(monkeypatch, fv100, False)
    assert not fv100.flash_attn_decode_paged_sinks_prepare("cuda:1")
    assert allocated == [(torch.device("cuda:1"), 0)]


def test_prepare_is_a_noop_without_the_fused_op(load):
    fv100, rec = load(with_fused=False)
    assert not fv100.flash_attn_decode_paged_sinks_prepare("cuda:0")
    q, k, v, table, lens = _inputs()
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=_sinks())
    assert rec.names == [TWO]
    assert fv100._fused_decode_counter_cache == {}


def test_optional_per_launch_zero(load, monkeypatch):
    """fork :1124-1137: VLLM_SM70_FLINT_ATTN_FUSED_DECODE_ZERO_COUNTERS=1."""
    fv100, rec = load()
    q, k, v, table, lens = _inputs()
    sinks = _sinks()
    rows = q.shape[0] * k.shape[2]
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    counters = rec.calls[0][1][COUNTERS]
    counters.fill_(7)
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert torch.all(counters == 7)
    monkeypatch.setenv(ZERO_ENV, "0")  # opt-in: only "1" acts
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert torch.all(counters == 7)
    monkeypatch.setenv(ZERO_ENV, "1")
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert torch.count_nonzero(counters[:rows]) == 0
    assert torch.all(counters[rows:] == 7)


def test_capture_counts(load, monkeypatch, warned):
    """fork :1240-1262 (minus the _runtime policy)."""
    fv100, _ = load()
    q, k, v, table, lens = _inputs(batch=4)
    sinks = _sinks()
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (0, 0)
    _capturing(monkeypatch, fv100, True)
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    fv100.flash_attn_decode_paged(q, k, v, table, lens)  # sink-free: uncounted
    fv100.flash_attn_decode_paged(q[:2], k, v, table[:2], lens[:2], sinks=sinks)
    monkeypatch.setattr(fv100, "FUSED_DECODE_COUNTER_CAPACITY", 1)
    fv100._fused_decode_counter_cache.clear()
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (2, 1)
    # Switched off on purpose: no counters, no capture state, no warning.
    monkeypatch.setenv(FUSED_ENV, "0")
    before = list(warned)
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks)
    assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (2, 1)
    assert fv100._fused_decode_counter_cache == {}
    assert warned == before
    fv100.clear_workspaces()
    assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (0, 0)


def test_kill_switch(load, monkeypatch):
    """fork :1269-1279."""
    fv100, rec = load()
    monkeypatch.setenv(FUSED_ENV, "0")
    q, k, v, table, lens = _inputs()
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=_sinks())
    ((name, args),) = rec.calls
    assert name == TWO and len(args) == 19
    assert fv100._fused_decode_counter_cache == {}
    assert not fv100.flash_attn_decode_paged_sinks_prepare("cuda:0")
    assert fv100._fused_decode_counter_cache == {}
    # Only "0" acts.
    for value in ("", "1", "off", "false"):
        monkeypatch.setenv(FUSED_ENV, value)
        assert fv100.fused_decode_enabled()


def test_switches_are_registered():
    from vllm_sm70 import envs

    names = envs.environment_variables()
    assert FUSED_ENV in envs._SECTIONS["attention"]
    assert ZERO_ENV in envs._SECTIONS["attention"]
    assert names[FUSED_ENV].__name__ == f"env_{FUSED_ENV}"
    assert names[ZERO_ENV].__name__ == f"env_{ZERO_ENV}"


@pytest.mark.parametrize(("value", "on"), [(None, False), ("0", False), ("1", True)])
def test_zero_counters_switch_semantics(monkeypatch, value, on):
    from vllm_sm70 import envs

    if value is None:
        monkeypatch.delenv(ZERO_ENV, raising=False)
    else:
        monkeypatch.setenv(ZERO_ENV, value)
    assert getattr(envs, ZERO_ENV) is on


# ------------------------------------------------- shape set (Flint, 2xV100)


@pytest.mark.parametrize(
    "heads",
    [(32, 4), (16, 2), (8, 1)],
    ids=["tp1-32q4kv", "tp2-16q2kv", "tp4-8q1kv"],
)
def test_flint_shapes_are_admitted(load, heads):
    """Flint 32q/4kv per TP size (FORK models/flint/configs.py:27-41): the
    TP2 shape that runs on 2xV100 is 16q/2kv, D128, GQA 8."""
    fv100, rec = load()
    q, k, v, table, lens = _inputs(*heads)
    assert fv100.fused_decode_eligible(q, k, _sinks(heads[0]))
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=_sinks(heads[0]))
    assert rec.names == [FUSED]
    assert rec.calls[0][1][COUNTERS].numel() >= q.shape[0] * heads[1]


@pytest.mark.parametrize(
    ("heads", "head_dim", "kw"),
    [
        ((16, 4), 128, {}),
        ((12, 2), 128, {}),
        ((16, 2), 256, {}),
        ((16, 2), 128, {"kv_cache_dtype": "fp8"}),
    ],
    ids=["gqa4", "gqa6", "d256", "fp8-kv"],
)
def test_other_shapes_keep_two_launches(load, heads, head_dim, kw):
    """fork :1282-1301 (anchored case dropped: not in the lean build)."""
    fv100, rec = load()
    q, k, v, table, lens = _inputs(*heads, head_dim=head_dim)
    sinks = _sinks(heads[0])
    assert not fv100.fused_decode_eligible(
        q, k, sinks, kw.get("kv_cache_dtype", "auto")
    )
    fv100.flash_attn_decode_paged(q, k, v, table, lens, sinks=sinks, **kw)
    fv100.flash_attn_decode_paged(q, k, v, table, lens)
    assert rec.names == [TWO, "fv100_decode_paged"]
    assert fv100._fused_decode_counter_cache == {}


def test_eligibility_needs_sinks(load):
    fv100, _ = load()
    q, k, *_ = _inputs()
    assert not fv100.fused_decode_eligible(q, k, None)


# ------------------------------------------- R4: vLLM's descending capture


@pytest.mark.parametrize("eager_first", [False, True], ids=["capture-only", "eager"])
def test_descending_capture_never_allocates_counters_in_a_graph(
    load, monkeypatch, eager_first
):
    """CPU model of fork :534-660 / :764 (the --max-num-seqs 20 bug).

    vLLM captures decode graphs largest batch first on persistent buffers.
    With counters prepared eagerly, every capture gets the same buffer; with
    none, every capture takes two launches and nothing is allocated under
    capture (a torch.zeros there would only be memset inside the first
    graph, the original gibberish).
    """
    fv100, rec = load()
    rows = 20
    q, k, v, table, lens = _inputs(batch=4)
    q = torch.zeros((rows, 16, HEAD_DIM), dtype=torch.float16)
    table = torch.zeros((rows, 2), dtype=torch.int32)
    lens = torch.ones(rows, dtype=torch.int32)
    sinks = _sinks()
    if eager_first:
        monkeypatch.setattr(
            fv100, "_normalized_device", lambda device: torch.device("cpu")
        )
        assert fv100._fused_decode_counters(torch.device("cpu"), 0) is not None
    zeros: list[tuple] = []
    real_zeros = torch.zeros
    monkeypatch.setattr(
        fv100.torch,
        "zeros",
        lambda *a, **kw: zeros.append(a) or real_zeros(*a, **kw),
    )
    _capturing(monkeypatch, fv100, True)
    for batch in (20, 16, 8, 4, 2, 1):
        fv100.flash_attn_decode_paged(
            q[:batch],
            k,
            v,
            table[:batch],
            lens[:batch],
            max_seq_len_hint=1,
            workspace_seq_capacity_hint=8192,
            partition_size_hint=256,
            sinks=sinks,
        )
    assert zeros == []
    if eager_first:
        assert rec.names == [FUSED] * 6
        assert len({id(args[COUNTERS]) for _, args in rec.calls}) == 1
        assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (6, 0)
    else:
        assert rec.names == [TWO] * 6
        assert fv100._fused_decode_counter_cache == {}
        assert fv100.flash_attn_decode_paged_sinks_capture_counts() == (0, 6)


# ------------------------------------------------------------- impl wiring


def test_hook_prepares_once_per_layer_and_reports_captures(monkeypatch):
    """fork :1140-1196, on dispatch.fused_decode_hook."""
    from vllm_sm70.attention.flash_v100 import dispatch
    from vllm_sm70.ops import fv100

    calls: list[torch.device] = []
    counts = [(0, 0)]
    capturing = [False]
    logged: list[str] = []
    debug: list[str] = []
    monkeypatch.setattr(dispatch, "_SINK_CAPTURE_PENDING", [False])
    monkeypatch.setattr(dispatch, "_SINK_CAPTURE_REPORTED", [None])
    monkeypatch.setattr(dispatch, "_capturing", lambda: capturing[0])
    monkeypatch.setattr(fv100, "flash_attn_decode_paged_sinks_prepare", calls.append)
    monkeypatch.setattr(
        fv100, "flash_attn_decode_paged_sinks_capture_counts", lambda: counts[0]
    )
    monkeypatch.setattr(fv100.logger, "info", lambda m, *a: logged.append(m % a))
    monkeypatch.setattr(fv100.logger, "debug", lambda m, *a: debug.append(m % a))

    def layer(sinks=True):
        return SimpleNamespace(_fv100_fused_sinks=sinks, _fv100_fused_ready=False)

    cuda = SimpleNamespace(seq_lens=SimpleNamespace(device=torch.device("cuda", 0)))

    def hook(impl, *, capture):
        capturing[0] = capture
        dispatch.fused_decode_hook(impl, cuda)

    first, second, plain = layer(), layer(), layer(sinks=False)
    hook(first, capture=True)
    assert calls == []  # under capture: nothing allocated
    hook(first, capture=False)
    hook(first, capture=False)
    hook(second, capture=False)
    hook(plain, capture=False)
    assert calls == [torch.device("cuda", 0)] * 2
    assert first._fv100_fused_ready and not plain._fv100_fused_ready
    assert logged == []  # nothing captured yet: (0, 0) is not reported
    report = (
        "Flash-V100 grouped sink decode in captured CUDA graphs so far: "
        "{} one-launch, {} two-launch (partition kernel plus reducer)."
    )
    for totals in ((14, 0), (28, 0), (28, 0), (42, 0), (42, 3)):
        counts[0] = totals
        hook(first, capture=True)
        hook(second, capture=False)
        hook(first, capture=False)
    assert logged == [report.format(14, 0), report.format(42, 3)]
    assert debug == [report.format(28, 0), report.format(42, 0)]
    cpu = layer()
    dispatch.fused_decode_hook(
        cpu, SimpleNamespace(seq_lens=SimpleNamespace(device=torch.device("cpu")))
    )
    assert calls == [torch.device("cuda", 0)] * 2


def _have_vllm() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


needs_vllm = pytest.mark.skipif(not _have_vllm(), reason="vLLM source not importable")


@pytest.fixture
def impl_env(monkeypatch):
    from vllm_sm70.ops import fv100

    prepared: list[torch.device] = []
    monkeypatch.setattr(
        fv100,
        "flash_attn_decode_paged_sinks_prepare",
        lambda device: prepared.append(torch.device(device)) or True,
    )
    return prepared


def _impl(num_heads, head_size, num_kv_heads, sinks):
    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

    return FlashAttnV100Impl(
        num_heads,
        head_size,
        head_size**-0.5,
        num_kv_heads,
        None,
        513,
        "auto",
        sinks=sinks,
    )


@needs_vllm
@pytest.mark.parametrize(
    ("heads", "head_size", "with_sinks", "fused"),
    [
        ((16, 2), 128, True, True),
        ((32, 4), 128, True, True),
        ((16, 2), 128, False, False),
        ((16, 4), 128, True, False),
        ((16, 2), 256, True, False),
    ],
    ids=["tp2", "tp1", "no-sinks", "gqa4", "d256"],
)
def test_impl_marks_fused_sink_layers(impl_env, heads, head_size, with_sinks, fused):
    sinks = torch.linspace(-2, 6, heads[0]) if with_sinks else None
    impl = _impl(heads[0], head_size, heads[1], sinks)
    assert impl._fv100_fused_sinks is fused
    assert impl._fv100_fused_ready is False  # CPU: nothing to prepare yet
    assert impl_env == []


@needs_vllm
def test_impl_prepares_counters_at_construction(impl_env, monkeypatch):
    """R4: the counters exist before vLLM's first capture, whatever runs
    eagerly before it (v0.33.0's profile run has no attention metadata)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    impl = _impl(16, 128, 2, torch.linspace(-2, 6, 16))
    assert impl._fv100_fused_sinks and impl._fv100_fused_ready
    assert impl_env == [torch.device("cuda", 1)]
    _impl(16, 128, 2, None)  # sink-free layers never ask
    assert impl_env == [torch.device("cuda", 1)]


@needs_vllm
def test_impl_forward_fused(load, monkeypatch, builder):
    """Impl forward on CPU stand-ins with the fused op built: Flint decode
    rows take one launch with the device counters, prefill keeps the sink
    prefill op, and the result matches the reference."""
    from _fv100_cpu_ops import CpuOps

    fv100, _ = load()
    ops = CpuOps()
    monkeypatch.setattr(fv100, "_op", ops)
    sinks = torch.linspace(-2, 6, 16)
    impl = _impl(16, 128, 2, sinks)
    _forward_case(
        builder,
        impl,
        query_lens=[1, 1, 7, 3, 7],
        seq_lens=[900, 33, 40, 600, 7],
        heads=(16, 2),
        head_size=128,
    )
    names = [c[0] for c in ops.calls]
    assert names == [FUSED] + ["fv100_prefill_paged_sinks"] * 2
    decode = ops.calls[0][1]
    assert decode["rows"] == 2 and decode["sinks"] is sinks
    counters = decode["counters"]
    assert counters.dtype == torch.int32 and counters.numel() >= 2 * 2
    assert impl._fv100_fused_ready
