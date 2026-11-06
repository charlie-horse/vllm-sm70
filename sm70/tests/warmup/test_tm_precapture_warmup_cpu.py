# SPDX-License-Identifier: Apache-2.0
"""TurboMind pre-capture warmup (vllm_sm70/warmup): CPU tests.

Shape enumeration for fake Flint TP2 and Qwen configs, the TP coordination
protocol on a threaded 2-rank group with a fake per-rank dispatch cache, the
kill switch, the capture guard, and the cache-file format.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
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
    "VLLM_SM70_GEMM_LUT_EXPORT_PATH",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in TUNE_VARS:
        monkeypatch.delenv(name, raising=False)
    # Prod's tuned mode: small-shape tuning on (the C++ default is 0 since
    # #77, so the warmup and LUT only matter with these set). Tests that need
    # the default or a mix set them again.
    for name in (
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES",
        "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES",
    ):
        monkeypatch.setenv(name, "1")
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
# 6 dense shapes x 8 M + MoE tokens 3..21 (19)
FLINT_CALLS = 6 * 8 + 19


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
    # K12 owns M <= 2; the MoE gate counts slots: 21 * 6 = 126 <= 128. MoE
    # warms every token count (eager MoE has no imported-cache reuse).
    assert moe.ms == tuple(range(3, 22))
    assert plan.calls() == FLINT_CALLS
    # Every capture size within the limits is warmed for every dense shape.
    for size in FLINT_CAPTURE:
        if size > 2:
            assert all(size in i.ms for i in dense)


def test_flint_without_k12_and_gemv_warms_small_m():
    model = Model([nvfp4_dense(2048, 2048, False), nvfp4_moe(k12=False)])
    plan = P.build([model], vcfg(FLINT_CAPTURE, max_seqs=8, spec=4))
    dense, moe = plan.items
    assert dense.ms[:3] == (1, 2, 4)
    assert moe.ms == tuple(range(1, 22))


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
    assert moe.ms == tuple(range(1, 17))  # 16 * 8 = 128 slots


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
    on an exact cache hit. After an import the device is "imported": an
    uncovered NVFP4/FP8 dense call then takes kReuse (LowerBound) and is not
    measured, as in gemm_runtime.cpp.
    """

    DENSE = (P.NVFP4_DENSE, P.FP8_DENSE, P.FP8_QPN8_TM)

    def __init__(self, rank, seed, fail_on=None):
        self.rank = rank
        self.cache = {}
        self.measured = 0
        self.lowerbound = 0
        self.imported = False
        self.rng = random.Random(seed)
        self.fail_on = fail_on

    def __call__(self, item, m):
        if self.fail_on == "warm":
            raise ValueError(f"boom on rank {self.rank}")
        key = f"{item.kind}:{item.key}:{m}"
        if key in self.cache:
            return
        if self.imported and item.kind in self.DENSE:
            self.lowerbound += 1  # kReuse on a miss: nearest entry, no measure
            return
        self.cache[key] = f"r{self.rank}-split{self.rng.randint(1, 16)}"
        self.measured += 1

    def export(self):
        return json.dumps(self.cache, sort_keys=True).encode() if self.cache else None

    def import_(self, payload):
        if self.fail_on == "import":
            raise OSError("import failed")
        table = json.loads(payload)
        self.cache.update(table)
        self.imported = True
        return len(table)


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
    assert fakes[0].measured == FLINT_CALLS
    for res in results:
        assert res.mode == "coordinated" and res.source == 0 and res.agree
        assert res.counts[P.NVFP4_DENSE] == 48 and res.counts[P.NVFP4_MOE] == 19
    assert results[1].imported == FLINT_CALLS
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
    # No payload: each rank tunes locally (fallback), so the ranks cannot be
    # shown to agree; that is a digest mismatch and raises on every rank.
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    for f in fakes:
        f.export = lambda: None
    _, errors, calls = _run_ranks([flint_model(), flint_model()], fakes)
    assert fakes[1].measured == FLINT_CALLS  # the follower tuned locally
    assert calls == ["all_gather", "broadcast", "all_gather"]
    for e in errors:
        assert isinstance(e, RuntimeError) and "mode fallback" in str(e)


class DriftTM(FakeTM):
    """A rank whose import does not take the source's specs (e.g. a C++
    import that silently drops records): its final cache differs."""

    def import_(self, payload):
        n = super().import_(payload)
        key = min(self.cache)
        self.cache[key] = "drifted"
        return n


def test_digest_mismatch_raises_on_every_rank(monkeypatch):
    seen = []
    monkeypatch.setattr(
        warmup.logger, "info", lambda msg, *a: seen.append(msg % a if a else msg)
    )
    fakes = [FakeTM(0, 1), DriftTM(1, 2)]
    _, errors, calls = _run_ranks([flint_model(), flint_model()], fakes)
    assert calls == ["all_gather", "broadcast", "all_gather"]  # no extra wait
    for e in errors:
        assert isinstance(e, RuntimeError) and not isinstance(e, coord.WarmupError)
        msg = str(e)
        assert "different TurboMind dispatch caches" in msg
        assert "VLLM_SM70_TM_PRECAPTURE_WARMUP=0" in msg
    # The summary is still logged first, with the verdict.
    assert any("ranks DISAGREE" in m for m in seen), seen


def test_digest_mismatch_writes_no_lut(tmp_path, monkeypatch):
    path = tmp_path / "tm-{arch}.bin"
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_EXPORT_PATH", str(path))
    fakes = [FakeTM(0, 1), DriftTM(1, 2)]
    _, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert all(isinstance(e, RuntimeError) for e in errors)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# Pinning specs: LUT export and import
# --------------------------------------------------------------------------
@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(
        warmup.logger, "warning", lambda fmt, *a, **k: seen.append(fmt % a)
    )
    return seen


def _export_run(tmp_path, monkeypatch, model=None, seeds=(1, 2)):
    path = tmp_path / "lut" / "tm-{arch}.bin"
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_EXPORT_PATH", str(path))
    fakes = [FakeTM(0, seeds[0]), FakeTM(1, seeds[1])]
    model = model or flint_model
    _, errors, _ = _run_ranks([model(), model()], fakes)
    assert errors == [None, None]
    monkeypatch.delenv("VLLM_SM70_GEMM_LUT_EXPORT_PATH")
    return tmp_path / "lut" / "tm-sm70.bin", fakes


def test_export_writes_source_cache_atomically(tmp_path, monkeypatch):
    writes = []
    real = cache.write_lut
    monkeypatch.setattr(cache, "write_lut", lambda *a: writes.append(a) or real(*a))
    lut, fakes = _export_run(tmp_path, monkeypatch)
    assert len(writes) == 1  # the source rank only
    assert lut.read_bytes() == fakes[0].export()
    manifest = json.loads(pathlib.Path(cache.manifest_path(str(lut))).read_bytes())
    assert manifest["sha256"] == hashlib.sha256(lut.read_bytes()).hexdigest()
    assert len(manifest["calls"]) == FLINT_CALLS
    # Temp files were renamed into place, none left behind.
    assert sorted(p.name for p in lut.parent.iterdir()) == [
        "tm-sm70.bin",
        "tm-sm70.bin.plan.json",
    ]


def test_lut_files_are_0644_whatever_the_umask(tmp_path, monkeypatch):
    old = os.umask(0o077)
    try:
        lut, _ = _export_run(tmp_path, monkeypatch)
    finally:
        os.umask(old)
    for p in (lut, pathlib.Path(cache.manifest_path(str(lut)))):
        assert (p.stat().st_mode & 0o777) == 0o644, p
    # And with a permissive umask: still exactly 0644, not 0666.
    old = os.umask(0)
    try:
        cache.write_atomic(str(tmp_path / "x.bin"), b"x")
    finally:
        os.umask(old)
    assert ((tmp_path / "x.bin").stat().st_mode & 0o777) == 0o644


def test_manifest_records_identity(tmp_path, monkeypatch):
    lut, _ = _export_run(tmp_path, monkeypatch)
    manifest = json.loads(pathlib.Path(cache.manifest_path(str(lut))).read_bytes())
    assert manifest["version"] == 2
    ident = manifest["identity"]
    assert ident["ext"] == envs.VLLM_SM70_EXT_FINGERPRINT
    assert ident["gpu"] is None  # CPU device in this test
    tuning = ident["tuning"]
    assert tuning["VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES"] == "1"
    assert tuning["VLLM_SM70_FP8_TUNE_SMALL_SHAPES"] == "1"
    assert tuning["VLLM_SM70_AWQ_TUNE_SMALL_SHAPES"] == "1"
    assert "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M" in tuning
    assert "VLLM_SM70_PROFILE_TRACE" not in tuning  # diagnostics only
    # Every registered dispatch knob is snapshotted (a new knob in the "ops"
    # section is picked up without touching cache.py).
    assert set(tuning) == set(envs._SECTIONS["ops"]) - {"VLLM_SM70_PROFILE_TRACE"}


def _refuse(kind, monkeypatch, tmp_path):
    if kind == "knob":  # a dispatch knob the warmup plan does not read
        monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", "0")
        return "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL"
    if kind == "small_shapes":
        monkeypatch.setenv("VLLM_SM70_FP8_TUNE_SMALL_SHAPES", "0")
        return "VLLM_SM70_FP8_TUNE_SMALL_SHAPES"
    if kind == "ext":
        monkeypatch.setattr(cache, "ext_fingerprint", lambda: "abi=9;so=feed")
        return "extension"
    if kind == "gpu":
        monkeypatch.setattr(cache, "gpu_name", lambda device: "Other GPU")
        return "GPU"
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["knob", "small_shapes", "ext", "gpu"])
def test_lut_with_foreign_identity_is_refused(
    tmp_path, monkeypatch, warnings_seen, kind
):
    lut, first = _export_run(tmp_path, monkeypatch)
    what = _refuse(kind, monkeypatch, tmp_path)
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(lut))
    fakes = [FakeTM(0, 3), FakeTM(1, 4)]
    results, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]  # skipped, not fatal
    refusals = [w for w in warnings_seen if "refusing TurboMind LUT" in w]
    assert len(refusals) == 2 and what in refusals[0]  # once per rank
    # Not loaded: nothing imported from it, the source tuned everything.
    assert [r.lut_records for r in results] == [0, 0]
    if kind != "small_shapes":  # tuning off for FP8 only; Flint is NVFP4
        assert fakes[0].measured == FLINT_CALLS
        assert fakes[0].cache != first[0].cache
    assert fakes[0].cache == fakes[1].cache  # ranks still agree


def test_v1_manifest_without_identity_is_refused(tmp_path, monkeypatch, warnings_seen):
    lut, _ = _export_run(tmp_path, monkeypatch)
    mp = pathlib.Path(cache.manifest_path(str(lut)))
    manifest = json.loads(mp.read_bytes())
    manifest.pop("identity")
    manifest["version"] = 1
    mp.write_text(json.dumps(manifest))
    expected = cache.identity(torch.device("cpu"))
    assert cache.load_lut(str(lut), expected) is None
    assert any("before v2" in w for w in warnings_seen)
    # Without an expected identity (direct callers) it still loads.
    assert cache.load_lut(str(lut)) is not None


def test_lut_summary_says_not_loaded_when_refused(tmp_path, monkeypatch):
    lut, _ = _export_run(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M", "32")
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(lut))
    seen = []
    monkeypatch.setattr(
        warmup.logger, "info", lambda msg, *a: seen.append(msg % a if a else msg)
    )
    _, errors, _ = _run_ranks(
        [flint_model(), flint_model()], [FakeTM(0, 3), FakeTM(1, 4)]
    )
    assert errors == [None, None]
    summary = [m for m in seen if "pre-capture warmup done" in m]
    assert summary and all(f"LUT {lut}: not loaded" in m for m in summary)
    assert all("ranks agree" in m for m in summary)


def test_lut_pins_specs_on_every_rank_without_measuring(tmp_path, monkeypatch):
    lut, first = _export_run(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(lut))
    fakes = [FakeTM(0, 3), FakeTM(1, 4)]  # other seeds: tuning would differ
    results, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]
    assert fakes[0].measured == fakes[1].measured == 0
    assert fakes[0].cache == fakes[1].cache == first[0].cache
    assert [r.lut_records for r in results] == [
        FLINT_CALLS
    ] * 2  # imported on every rank
    assert all(r.agree for r in results)


def test_lut_missing_shapes_are_tuned_before_import_and_warned(
    tmp_path, monkeypatch, warnings_seen
):
    def partial():  # five of the six dense shapes, no MoE
        return Model([nvfp4_dense(k, n, True) for k, n in FLINT_DENSE[:5]])

    lut, first = _export_run(tmp_path, monkeypatch, model=partial)
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(lut))
    fakes = [FakeTM(0, 3), FakeTM(1, 4)]
    _, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]
    # The 8 calls of the sixth dense shape and the 19 MoE calls are measured
    # on the source, before the import, so none falls back to LowerBound.
    assert fakes[0].measured == 27 and fakes[0].lowerbound == 0
    assert fakes[1].measured == 0
    assert fakes[0].cache == fakes[1].cache
    for key, spec in first[0].cache.items():
        assert fakes[1].cache[key] == spec  # covered shapes keep the LUT spec
    msgs = [w for w in warnings_seen if "does not cover" in w]
    assert len(msgs) == 2 and msgs[0] == msgs[1]  # once per rank
    msg = msgs[0]
    assert f"27 of {FLINT_CALLS}" in msg and "nvfp4_moe" in msg and "[3072, 2048" in msg


def test_lut_without_manifest_tunes_everything_then_lut_wins(
    tmp_path, monkeypatch, warnings_seen
):
    lut, first = _export_run(tmp_path, monkeypatch)
    pathlib.Path(cache.manifest_path(str(lut))).unlink()
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(lut))
    fakes = [FakeTM(0, 3), FakeTM(1, 4)]
    _, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]
    assert fakes[0].measured == FLINT_CALLS  # coverage unknown: all tuned first
    assert fakes[0].cache == fakes[1].cache == first[0].cache  # then the LUT wins
    assert any("no manifest" in w for w in warnings_seen)


class PolicyTM(FakeTM):
    """FakeTM that picks the dispatch policy as gemm_runtime.cpp does (eager):

    dense: imported device and reuse_imported (NVFP4 always,
    FP8 per VLLM_SM70_FP8_REUSE_IMPORTED_CACHE, default 1) -> reuse; tuning
    off -> default; already tuned -> reuse; else measure. MoE: tuning off ->
    default; tuned -> reuse; else measure (no imported check). A "measure"
    times kernels only on a cache miss (Gemm::Impl::Measure returns early on
    an exact hit); `timed` counts those.
    """

    def __init__(self, rank, seed):
        super().__init__(rank, seed)
        self.tuned = set()
        self.policies = []
        self.timed = 0

    def __call__(self, item, m):
        key = f"{item.kind}:{item.key}:{m}"
        tune_on = P.limits()[item.kind].enabled
        if item.kind in self.DENSE:
            reuse_imported = item.kind == P.NVFP4_DENSE or (
                P.knob("VLLM_SM70_FP8_REUSE_IMPORTED_CACHE", 1) != 0
            )
            if self.imported and reuse_imported:
                policy = "reuse"
            elif not tune_on:
                policy = "default"
            elif key in self.tuned:
                policy = "reuse"
            else:
                policy = "measure"
        elif not tune_on:
            policy = "default"
        elif key in self.tuned:
            policy = "reuse"
        else:
            policy = "measure"
        if policy == "measure":
            self.tuned.add(key)
        self.policies.append((key, policy))
        if key in self.cache:
            return  # exact hit: reuse/measure/default all take the entry
        if policy == "measure":
            self.timed += 1
            self.cache[key] = f"r{self.rank}-split{self.rng.randint(1, 16)}"
        elif policy == "reuse":
            self.lowerbound += 1
        else:
            self.cache[key] = "heuristic"


def qwen_mixed_model():
    return Model([*qwen38_model().layers, fp8_moe(top_k=8)])


@pytest.mark.parametrize("fp8_reuse", [None, "0"])
def test_tuning_on_with_lut_never_times_covered_shapes(
    tmp_path, monkeypatch, fp8_reuse
):
    """Prod mode for T4c: small-shape tuning on + a LUT -> no timed kMeasure on
    any rank, and both instances hold the exported specs."""
    for name in (
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES",
    ):
        monkeypatch.setenv(name, "1")
    if fp8_reuse is not None:
        monkeypatch.setenv("VLLM_SM70_FP8_REUSE_IMPORTED_CACHE", fp8_reuse)

    def model():
        return Model([*flint_model().layers, *qwen_mixed_model().layers])

    path = tmp_path / "tm-{arch}.bin"
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_EXPORT_PATH", str(path))
    first = [PolicyTM(0, 1), PolicyTM(1, 2)]
    _, errors, _ = _run_ranks([model(), model()], first)
    assert errors == [None, None] and first[0].timed > 0  # run 1 tunes
    monkeypatch.delenv("VLLM_SM70_GEMM_LUT_EXPORT_PATH")
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(path))
    for seeds in ((3, 4), (5, 6)):  # two fresh instances
        fakes = [PolicyTM(0, seeds[0]), PolicyTM(1, seeds[1])]
        results, errors, _ = _run_ranks([model(), model()], fakes)
        assert errors == [None, None]
        assert [f.timed for f in fakes] == [0, 0]
        assert fakes[0].cache == fakes[1].cache == first[0].cache
        # Covered dense calls take kReuse after the import, never kMeasure.
        dense_measure = [
            k
            for f in fakes
            for k, p in f.policies
            if p == "measure" and k.split(":")[0] in PolicyTM.DENSE
        ]
        if fp8_reuse is None:
            assert dense_measure == []
        assert all(r.lut_records > 0 for r in results)


def test_tuning_off_needs_no_lut(tmp_path, monkeypatch):
    """#77 mode: tuning off -> nothing to warm, LUT not read, kDefault."""
    for name in (
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(tmp_path / "absent.bin"))
    fakes = [PolicyTM(0, 1), PolicyTM(1, 2)]
    results, errors, calls = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None] and calls == ["all_gather"]
    assert results[0].mode == "none" and [f.policies for f in fakes] == [[], []]


def test_absent_lut_tunes_as_usual(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(tmp_path / "none.bin"))
    fakes = [FakeTM(0, 1), FakeTM(1, 2)]
    results, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None] and fakes[0].measured == FLINT_CALLS
    assert results[0].lut_records == 0


def test_failed_export_warns_and_serves(tmp_path, monkeypatch, warnings_seen):
    path = tmp_path / "x.bin"
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_EXPORT_PATH", str(path))
    fake = FakeTM(0, 1)
    fake.export = lambda: None
    res = warmup.run(
        _worker(flint_model(), vcfg(FLINT_CAPTURE, 8, 4)), group=None, executor=fake
    )
    assert res.mode == "local" and not path.exists()
    assert any("not written" in w for w in warnings_seen)


def test_tp1_runs_locally_without_collectives():
    fake = FakeTM(0, 1)
    res = warmup.run(
        _worker(flint_model(), vcfg(FLINT_CAPTURE, 8, 4)), group=None, executor=fake
    )
    assert res.mode == "local" and fake.measured == FLINT_CALLS


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
    assert "VLLM_SM70_GEMM_LUT_EXPORT_PATH" in envs._SECTIONS["warmup"]
    assert envs.VLLM_SM70_GEMM_LUT_EXPORT_PATH is None


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


def test_write_atomic_keeps_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "lut.bin"
    cache.write_atomic(str(target), b"old")
    assert target.read_bytes() == b"old"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cache.write_atomic(str(target), b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["lut.bin"]  # tmp removed


def test_load_lut_absent_bad_and_stale_manifest(tmp_path):
    assert cache.load_lut(str(tmp_path / "absent.bin")) is None
    bad = tmp_path / "bad.bin"
    bad.write_bytes(cache.MAGIC + b"{not json")
    assert cache.load_lut(str(bad)) is None  # warning, not an error
    good = tmp_path / "good.bin"
    calls = [cache.call_id(P.NVFP4_DENSE, (2048, 2048, 16, True), 4)]
    cache.write_lut(str(good), b"table", calls)
    lut = cache.load_lut(str(good))
    assert lut.table == b"table" and lut.covers == set(calls)
    assert lut.missing(calls + [("x", (1,), 1)]) == [("x", (1,), 1)]
    good.write_bytes(b"other table")  # manifest no longer matches
    assert cache.load_lut(str(good)).covers is None


def test_default_tuning_off_plans_nothing(monkeypatch):
    """#77: with the C++ default (tuning off) the warmup has nothing to tune;
    every call is kDefault, deterministic without a LUT."""
    for name in (
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES",
        "VLLM_SM70_FP8_TUNE_SMALL_SHAPES",
        "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES",
    ):
        monkeypatch.delenv(name, raising=False)
    assert P.TUNE_SMALL_SHAPES_DEFAULT == 0
    limits = P.limits()
    assert not any(lim.enabled for lim in limits.values())


def test_lut_summary_phrases_are_stable(tmp_path, monkeypatch):
    """M5's runner (Mara) greps every engine log for these exact phrases; a
    rename must fail here, not silently in a gate. They are logged on a
    "vllm.sm70.*" logger so they reach the engine log (#76)."""
    from vllm_sm70 import warmup as W

    assert W.logger.name.startswith("vllm.sm70")
    lut, _ = _export_run(tmp_path, monkeypatch)
    monkeypatch.setenv("VLLM_SM70_GEMM_LUT_PATH", str(lut))
    seen = []
    monkeypatch.setattr(
        W.logger, "info", lambda msg, *a: seen.append(msg % a if a else msg)
    )
    fakes = [FakeTM(0, 3), FakeTM(1, 4)]
    _, errors, _ = _run_ranks([flint_model(), flint_model()], fakes)
    assert errors == [None, None]
    summary = [m for m in seen if "pre-capture warmup" in m]
    assert summary, seen
    for line in summary:
        assert "tuned (measured) 0 calls" in line
        assert f"{FLINT_CALLS}/{FLINT_CALLS} calls covered" in line
        assert "ranks agree" in line
