# SPDX-License-Identifier: Apache-2.0
"""TurboMind pre-capture warmup (vllm_sm70/warmup): CPU tests.

Shape enumeration for fake Flint TP2 and Qwen configs, the TP coordination
protocol on a threaded 2-rank group with a fake per-rank dispatch cache, the
kill switch, the capture guard, and the cache-file format.
"""

from __future__ import annotations

import base64
import json
import random
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_sm70 import (
    envs,
    warmup,
)
from vllm_sm70.warmup import cache, coord, kinds
from vllm_sm70.warmup import plan as P

META = "meta"
TUNE_VARS = (
    "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
    "VLLM_SM70_FP8_TUNE_SMALL_SHAPES",
    "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES",
    "VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR",
    "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M",
    "VLLM_SM70_FP8_DENSE_TUNE_MAX_M",
    "VLLM_SM70_F16_DENSE_TUNE_MAX_M",
    "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS",
    "VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS",
    "VLLM_SM70_FP8_QPN8_M32",
    "VLLM_SM70_TM_PRECAPTURE_WARMUP",
    "VLLM_SM70_GEMM_LUT_PATH",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in TUNE_VARS:
        monkeypatch.delenv(name, raising=False)
    # K09's bound without importing vllm_sm70.ops.qpn8 (default 8).
    monkeypatch.setattr(P, "_qpn8_max_m", lambda: 8)


# --------------------------------------------------------------------------
# Fake layers and configs
# --------------------------------------------------------------------------
class Layer(torch.nn.Module):
    def __init__(self, **attrs):
        super().__init__()
        for k, v in attrs.items():
            object.__setattr__(self, k, v)


def nvfp4_dense(k, n, gemv):
    state = SimpleNamespace(
        weight=torch.empty(k, n // 8, dtype=torch.int32, device=META),
        group_size=16,
        tm_gemv=gemv,
    )
    return Layer(_sm70_turbomind_linear=state)


def nvfp4_moe(experts=256, hidden=2048, inter=256, top_k=6, k12=True):
    return Layer(
        sm70_nvfp4_num_experts=experts,
        sm70_nvfp4_hidden_size=hidden,
        sm70_nvfp4_intermediate_size=inter,
        sm70_nvfp4_top_k=top_k,
        sm70_nvfp4_decode_plan=object() if k12 else None,
    )


def fp8_dense(k, n):
    return Layer(
        sm70_fp8_turbomind=True,
        weight=torch.empty(k, n, dtype=torch.uint8, device=META),
        output_size_per_partition=n,
    )


def fp8_qpn8(k, n, copy=True):
    attrs = {"sm70_fp8_qpn8": True, "sm70_fp8_turbomind": False}
    if copy:
        attrs["sm70_qpn8_tm_weight"] = torch.empty(k, n, dtype=torch.uint8, device=META)
    return Layer(**attrs)


def fp8_moe(experts=256, hidden=2048, inter=256, top_k=8):
    return Layer(
        sm70_num_experts=experts,
        sm70_hidden_size=hidden,
        sm70_intermediate_size=inter,
        sm70_top_k=top_k,
        sm70_fp8_moe_compact=True,
        sm70_fp8_moe_decode=True,
    )


class Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class Mode:
    def __init__(self, name, mixed="PIECEWISE"):
        self.name = name
        self._mixed = mixed

    def mixed_mode(self):
        return Mode(self._mixed, self._mixed)


def vcfg(
    capture,
    max_seqs,
    spec=0,
    *,
    eager=False,
    mode="FULL_AND_PIECEWISE",
    mixed="PIECEWISE",
    max_tokens=8192,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_seqs, max_num_batched_tokens=max_tokens
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=list(capture),
            cudagraph_mode=Mode(mode, mixed),
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        speculative_config=SimpleNamespace(num_speculative_tokens=spec)
        if spec
        else None,
        parallel_config=SimpleNamespace(enable_elastic_ep=False),
    )


# Flint TP2 per-rank NVFP4 dense (K, N): the K11 GEMV set (nvfp4_linear.py).
FLINT_DENSE = [
    (2048, 2560),
    (2048, 2048),
    (2048, 1024),
    (512, 2048),
    (2048, 6144),
    (3072, 2048),
]


def flint_model():
    layers = [nvfp4_dense(k, n, True) for k, n in FLINT_DENSE]
    layers += [nvfp4_dense(k, n, True) for k, n in FLINT_DENSE]  # 2nd decoder layer
    layers += [nvfp4_moe(), nvfp4_moe()]
    return Model(layers)


# Qwen3.8-27B FP8 TP2-like dense shapes: QPN8 serves M <= 8, the TurboMind
# copy above; one projection stays on plain M6 TurboMind.
QWEN38_QPN8 = [(5120, 4096), (2048, 5120), (5120, 8704), (8704, 5120)]


def qwen38_model(copy=True):
    layers = [fp8_qpn8(k, n, copy) for k, n in QWEN38_QPN8]
    layers += [fp8_dense(5120, 1024)]
    return Model(layers)


# --------------------------------------------------------------------------
# Shape enumeration
# --------------------------------------------------------------------------
FLINT_CAPTURE = [1, 2, 4, 8, 16, 24, 32, 40]


def test_flint_tp2_m_values_cover_capture_verify_and_fork_set():
    mv = P.decode_m_values(vcfg(FLINT_CAPTURE, max_seqs=8, spec=4))
    assert mv.verify_m == 5
    assert mv.max_decode_m == 40
    assert mv.capture == tuple(FLINT_CAPTURE)
    assert mv.eager == ()  # max capture == max_num_seqs * (spec + 1)
    assert mv.all == (1, 2, 4, 5, 8, 16, 24, 32, 40, 64)


def test_flint_tp2_plan():
    plan = P.build([flint_model()], vcfg(FLINT_CAPTURE, max_seqs=8, spec=4))
    by_kind = plan.by_kind()
    dense = by_kind[P.NVFP4_DENSE]
    assert [i.key[:2] for i in dense] == FLINT_DENSE  # deduplicated, in order
    for item in dense:
        # K11 GEMV owns M <= 2; NVFP4 dense tunes up to M 64.
        assert item.ms == (4, 5, 8, 16, 24, 32, 40, 64)
    (moe,) = by_kind[P.NVFP4_MOE]
    assert moe.key == (256, 2048, 256, 6, True)
    # K12 owns M <= 2; the MoE gate counts slots: 21 * 6 = 126 <= 128.
    assert moe.ms == (4, 5, 8, 16)
    assert plan.calls() == 6 * 8 + 4
    # Every capture size within the limits is warmed for every dense shape.
    for size in FLINT_CAPTURE:
        if size > 2:
            assert all(size in i.ms for i in dense)


def test_flint_without_k12_and_gemv_warms_small_m():
    model = Model([nvfp4_dense(2048, 2048, False), nvfp4_moe(k12=False)])
    plan = P.build([model], vcfg(FLINT_CAPTURE, max_seqs=8, spec=4))
    dense, moe = plan.items
    assert dense.ms[:3] == (1, 2, 4)
    assert moe.ms == (1, 2, 4, 5, 8, 16)


def test_qwen38_plan_capture_and_eager_sizes():
    capture = [1, 2, 4, 8, 16, 24, 32]
    cfg = vcfg(capture, max_seqs=12, spec=3)  # verify 4, max decode 48
    mv = P.decode_m_values(cfg)
    assert mv.verify_m == 4 and mv.max_decode_m == 48
    assert mv.eager == tuple(range(33, 49))  # above the largest capture size
    plan = P.build([qwen38_model()], cfg)
    qpn8 = plan.by_kind()[P.FP8_QPN8_TM]
    assert [i.key for i in qpn8] == QWEN38_QPN8
    expected = (16, 24, 32, *range(33, 49), 64)
    for item in qpn8:
        assert item.ms == expected  # QPN8 GEMV owns M <= 8
    (tm_dense,) = plan.by_kind()[P.FP8_DENSE]
    assert tm_dense.ms == (1, 2, 4, 8, 16, 24, 32, *range(33, 49), 64)


def test_qwen38_qpn8_without_copy_has_nothing_to_warm():
    plan = P.build([qwen38_model(copy=False)], vcfg([1, 2, 4, 8], max_seqs=8))
    assert plan.by_kind()[P.FP8_QPN8_TM] == []
    assert len(plan.by_kind()[P.FP8_DENSE]) == 1


def test_qwen36_fp8_moe_slots_limit():
    plan = P.build([Model([fp8_moe(top_k=8)])], vcfg([1, 2, 4, 8, 16, 32], 16))
    (moe,) = plan.items
    assert moe.ms == (1, 2, 4, 8, 16)  # 16 * 8 = 128 slots; 32 * 8 > 128


def test_eager_and_mixed_none_cover_every_decode_m():
    for cfg in (
        vcfg([1, 2, 4, 8], max_seqs=6, spec=1, eager=True),
        vcfg([1, 2, 4, 8], max_seqs=6, spec=1, mode="NONE"),
        vcfg([1, 2, 4, 8], max_seqs=6, spec=1, mode="FULL_DECODE_ONLY", mixed="NONE"),
    ):
        mv = P.decode_m_values(cfg)
        assert mv.eager == tuple(range(1, 13))
        assert set(range(1, 13)) <= set(mv.all)
    assert P.decode_m_values(vcfg([1, 2, 4, 8], 6, 1, eager=True)).capture == ()


def test_max_num_batched_tokens_caps_m():
    mv = P.decode_m_values(vcfg([1, 2, 4, 8, 16, 32], max_seqs=4, max_tokens=16))
    assert max(mv.all) == 16


def test_tuning_limits_follow_cpp_knobs(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", "16")
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "0")
    model = Model([nvfp4_dense(2048, 2048, False), fp8_dense(5120, 1024)])
    plan = P.build([model], vcfg(FLINT_CAPTURE, max_seqs=8, spec=4))
    (dense,) = plan.items
    assert dense.kind == P.NVFP4_DENSE and max(dense.ms) == 16
    assert P.FP8_DENSE in plan.skipped
    monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_0DOT3_DENSE_SELECTOR", "1")
    assert P.limits()[P.FP8_DENSE].max_m == 16  # generic gate


def test_atoi_matches_c():
    assert [P._atoi(s) for s in ("64", " 12x", "-3", "", "abc", "+7")] == [
        64,
        12,
        -3,
        0,
        0,
        7,
    ]


def test_plan_ignores_unrelated_layers_and_fingerprint_is_stable():
    model = Model([torch.nn.Linear(2, 2), nvfp4_dense(2048, 2048, True)])
    cfg = vcfg(FLINT_CAPTURE, max_seqs=8, spec=4)
    a, b = P.build([model], cfg), P.build([flint_model()], cfg)
    assert len(a.items) == 1
    assert a.fingerprint() == P.build([model], cfg).fingerprint()
    assert a.fingerprint() != b.fingerprint()


def test_moe_routing_distinct_and_balanced():
    w, ids = kinds.moe_routing(5, 6, 256, "cpu")
    assert ids.shape == (5, 6) and ids.dtype == torch.int32
    assert w.dtype == torch.float32 and torch.allclose(w.sum(1), torch.ones(5))
    for row in ids.tolist():
        assert len(set(row)) == 6
    _, ids = kinds.moe_routing(4, 8, 8, "cpu")
    assert torch.bincount(ids.flatten(), minlength=8).tolist() == [4] * 8


# --------------------------------------------------------------------------
# Coordination on a threaded 2-rank group
# --------------------------------------------------------------------------
class ThreadGroup:
    """In-process Group: the collectives rendezvous on a shared barrier.

    A rank that never arrives breaks the barrier after `timeout` seconds, so a
    protocol deadlock fails the test instead of hanging it.
    """

    class _Shared:
        def __init__(self, n, timeout):
            self.barrier = threading.Barrier(n, timeout=timeout)
            self.slots = [None] * n
            self.calls = []

    def __init__(self, shared, rank):
        self.shared, self.rank = shared, rank
        self.world_size = len(shared.slots)

    @classmethod
    def make(cls, n=2, timeout=10.0):
        shared = cls._Shared(n, timeout)
        return [cls(shared, r) for r in range(n)]

    def all_gather_object(self, obj):
        self.shared.slots[self.rank] = obj
        self.shared.barrier.wait()
        out = list(self.shared.slots)
        self.shared.barrier.wait()
        if self.rank == 0:
            self.shared.calls.append("all_gather")
        return out

    def broadcast_object(self, obj, src):
        if self.rank == src:
            self.shared.slots[src] = obj
        self.shared.barrier.wait()
        out = self.shared.slots[src]
        self.shared.barrier.wait()
        if self.rank == 0:
            self.shared.calls.append("broadcast")
        return out


class FakeTM:
    """One rank's TurboMind dispatch cache.

    A call measures (a random, rank-specific spec) on its first sight of a
    shape, as kMeasure does; import is authoritative; Measure returns early
    on an exact cache hit.
    """

    def __init__(self, rank, seed, fail_on=None):
        self.rank = rank
        self.cache = {}
        self.measured = 0
        self.rng = random.Random(seed)
        self.fail_on = fail_on

    def __call__(self, item, m):
        if self.fail_on == "warm":
            raise ValueError(f"boom on rank {self.rank}")
        key = f"{item.kind}:{item.key}:{m}"
        if key not in self.cache:
            self.cache[key] = f"r{self.rank}-split{self.rng.randint(1, 16)}"
            self.measured += 1

    def export(self):
        return json.dumps(self.cache, sort_keys=True).encode() if self.cache else None

    def import_(self, payload):
        if self.fail_on == "import":
            raise OSError("import failed")
        table = json.loads(payload)
        self.cache.update(table)
        return len(table)

    def import_file(self):
        return 0


def _worker(model, cfg):
    return SimpleNamespace(
        vllm_config=cfg, device=torch.device("cpu"), get_model=lambda: model
    )


def _run_ranks(models, fakes, cfg=None, groups=None):
    cfg = cfg or vcfg(FLINT_CAPTURE, max_seqs=8, spec=4)
    groups = groups or ThreadGroup.make(len(models))
    results, errors = [None] * len(models), [None] * len(models)

    def body(r):
        try:
            results[r] = warmup.run(
                _worker(models[r], cfg), group=groups[r], executor=fakes[r]
            )
        except BaseException as exc:  # noqa: BLE001
            errors[r] = exc

    threads = [threading.Thread(target=body, args=(r,)) for r in range(len(models))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "warmup deadlocked"
    for e in errors:
        assert not isinstance(e, threading.BrokenBarrierError), "collective mismatch"
    return results, errors, groups[0].shared.calls


def test_coordinated_ranks_end_with_rank0_specs():
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, calls = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]
    assert fakes[0].cache == fakes[1].cache
    assert all(v.startswith("r0-") for v in fakes[1].cache.values())
    assert fakes[1].measured == 0  # the follower only replays cache hits
    assert fakes[0].measured == 6 * 8 + 4
    for res in results:
        assert res.mode == "coordinated" and res.source == 0 and res.agree
        assert res.counts[P.NVFP4_DENSE] == 48 and res.counts[P.NVFP4_MOE] == 4
    assert results[1].imported == 52
    assert calls == ["all_gather", "broadcast", "all_gather"]


def test_rank_without_a_layer_kind_does_not_deadlock():
    no_moe = Model([nvfp4_dense(k, n, True) for k, n in FLINT_DENSE])
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, calls = _run_ranks([flint_model(), no_moe], fakes)
    assert errors == [None, None]
    assert results[1].counts[P.NVFP4_MOE] == 0
    # Import carries rank 0's MoE specs too, so the caches still agree.
    assert fakes[0].cache == fakes[1].cache and results[0].agree
    assert calls == ["all_gather", "broadcast", "all_gather"]


def test_rank_without_any_layer_does_not_deadlock():
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, _ = _run_ranks([flint_model(), Model([])], fakes)
    assert errors == [None, None]
    assert fakes[1].cache == {} and results[1].counts == {}
    assert results[0].agree  # only ranks with work are compared


def test_source_is_lowest_rank_with_work():
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, _ = _run_ranks([Model([]), flint_model()], fakes)
    assert errors == [None, None]
    assert results[0].source == results[1].source == 1
    assert all(v.startswith("r1-") for v in fakes[1].cache.values())


def test_no_rank_has_work_is_one_collective():
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, calls = _run_ranks([Model([]), Model([])], fakes)
    assert errors == [None, None] and calls == ["all_gather"]
    assert results[0].mode == "none"


def test_source_failure_raises_on_every_rank():
    fakes = [FakeTM(0, 1, fail_on="warm"), FakeTM(1, 2)]
    _, errors, calls = _run_ranks([flint_model(), flint_model()], fakes)
    assert all(isinstance(e, coord.WarmupError) for e in errors)
    assert "boom on rank 0" in str(errors[1])
    assert calls == ["all_gather", "broadcast"]


def test_follower_failure_raises_on_every_rank():
    fakes = [FakeTM(0, 1), FakeTM(1, 2, fail_on="import")]
    _, errors, calls = _run_ranks([flint_model(), flint_model()], fakes)
    assert all(isinstance(e, coord.WarmupError) for e in errors)
    assert "rank 1: OSError" in str(errors[0])
    assert calls == ["all_gather", "broadcast", "all_gather"]


def test_plan_failure_on_one_rank_does_not_deadlock():
    class Broken(torch.nn.Module):
        def modules(self):
            raise AttributeError("bad layer")

    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, calls = _run_ranks([flint_model(), Broken()], fakes)
    assert errors[0] is None and results[0].mode == "coordinated"
    assert isinstance(errors[1], RuntimeError) and "bad layer" in str(errors[1])
    assert calls == ["all_gather", "broadcast", "all_gather"]


def test_missing_export_falls_back_without_deadlock():
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    for f in fakes:
        f.export = lambda: None
    results, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]
    assert results[1].mode == "fallback" and fakes[1].measured == 52
    assert not results[0].agree


def test_source_imports_cache_file_before_tuning():
    seen = []
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    fakes[0].import_file = lambda: seen.append(0) or 7
    fakes[1].import_file = lambda: seen.append(1) or 7
    results, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None] and seen == [0]
    assert results[0].imported == 7 + 52


def test_tp1_runs_locally_without_collectives():
    fake = FakeTM(0, 1)
    res = warmup.run(
        _worker(flint_model(), vcfg(FLINT_CAPTURE, 8, 4)), group=None, executor=fake
    )
    assert res.mode == "local" and fake.measured == 52


# --------------------------------------------------------------------------
# Kill switch, capture guard, platform hook
# --------------------------------------------------------------------------
class NoCollectives:
    rank, world_size = 0, 2

    def all_gather_object(self, obj):
        raise AssertionError("no collective expected")

    broadcast_object = all_gather_object


def test_kill_switch_skips_everything(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_TM_PRECAPTURE_WARMUP", "0")
    fake = FakeTM(0, 1)
    worker = _worker(flint_model(), vcfg(FLINT_CAPTURE, 8, 4))
    assert warmup.run(worker, group=NoCollectives(), executor=fake) is None
    assert fake.measured == 0


@pytest.mark.parametrize("value", [None, "1", "", "off"])
def test_kill_switch_only_zero_acts(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("VLLM_SM70_TM_PRECAPTURE_WARMUP", value)
    assert warmup.enabled() is True


def test_refuses_during_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fake = FakeTM(0, 1)
    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        warmup.run(
            _worker(flint_model(), vcfg(FLINT_CAPTURE, 8, 4)),
            group=NoCollectives(),
            executor=fake,
        )
    assert fake.measured == 0


def test_refuses_during_capture_even_when_switched_off(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_TM_PRECAPTURE_WARMUP", "0")
    monkeypatch.setattr(warmup, "is_capturing", lambda: True)
    with pytest.raises(RuntimeError):
        warmup.run(object())


def test_worker_without_config_is_a_no_op():
    assert warmup.run(object(), group=NoCollectives()) is None


def test_platform_kernel_warmup_calls_run_once(monkeypatch):
    import importlib
    import sys
    import types

    pytest.importorskip("vllm")
    try:  # as plugin/conftest.py stub_stable_libtorch
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    from vllm_sm70._cuda_platform import Sm70CudaPlatform

    calls = []
    monkeypatch.setattr(warmup, "run", lambda worker: calls.append(worker))
    sentinel = object()
    assert Sm70CudaPlatform.kernel_warmup(sentinel) is None
    assert calls == [sentinel]


def test_env_vars_registered_with_defaults():
    names = envs.environment_variables()
    assert "VLLM_SM70_TM_PRECAPTURE_WARMUP" in envs._SECTIONS["warmup"]
    assert "VLLM_SM70_GEMM_LUT_PATH" in names
    assert envs.VLLM_SM70_TM_PRECAPTURE_WARMUP is True
    assert envs.VLLM_SM70_GEMM_LUT_PATH is None


# --------------------------------------------------------------------------
# Cache file
# --------------------------------------------------------------------------
def test_legacy_table_raw_and_partitioned():
    raw = b"\x01\x02table"
    assert cache.legacy_table(raw) == raw
    packed = (
        cache.MAGIC
        + json.dumps({"legacy": base64.b64encode(raw).decode(), "abc": "eA=="}).encode()
    )
    assert cache.legacy_table(packed) == raw
    no_legacy = cache.MAGIC + json.dumps({"abc": "eA=="}).encode()
    assert cache.legacy_table(no_legacy) is None


def test_resolve_path_template():
    assert cache.resolve_path(None, 0, 0) is None
    assert (
        cache.resolve_path("/c/{arch}-{device}-{rank}.bin", 1, 0) == "/c/sm70-1-0.bin"
    )


def test_import_file_missing_and_bad(tmp_path, monkeypatch):
    assert cache.import_file("cpu", str(tmp_path / "absent.bin")) == 0
    bad = tmp_path / "bad.bin"
    bad.write_bytes(cache.MAGIC + b"{not json")
    assert cache.import_file("cpu", str(bad)) == 0  # warning, not an error
    good = tmp_path / "good.bin"
    good.write_bytes(b"table")
    got = []
    monkeypatch.setattr(cache, "import_bytes", lambda dev, t: got.append(t) or 3)
    assert cache.import_file("cpu", str(good)) == 3 and got == [b"table"]
