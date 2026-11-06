# SPDX-License-Identifier: Apache-2.0
"""TurboMind pre-capture warmup, TP-coordinated (Sm70CudaPlatform.kernel_warmup).

What: before CUDA graph capture, run every TurboMind GEMM shape a decode step
  can hit (dense NVFP4/FP8 and NVFP4/FP8 MoE, at every capture size, the
  spec-decode verify M and the eager-only decode sizes) once, so that each is
  tuned; tune on one TP rank and give every rank that rank's dispatch cache.
Why: TurboMind picks a launch spec per shape. In eager, a shape's first call
  is kMeasure, a timed autotune whose pick can change run to run and between
  TP ranks; under capture an untuned shape gets kDefault
  (csrc/turbomind/ops/gemm_runtime.cpp:441-475, MoE :537-560). Different
  split-K or kernel means different bits, so graph replay diverged from
  eager and rank 0 from rank 1. After this, capture reuses the tuned spec
  (kReuse) and both ranks hold the same specs.
How: plan.py (shapes and M values), kinds.py (one call per shape and M),
  coord.py (source rank tunes, cache broadcast, all ranks import and replay),
  cache.py (cache bytes; VLLM_SM70_GEMM_LUT_EXPORT_PATH writes a LUT and
  VLLM_SM70_GEMM_LUT_PATH imports one on every rank, so the calls it covers
  are not measured and their specs are the same on every start).
  Called from upstream's kernel_warmup via U1 (patches/0002), after the
  compile warmup and before capture_model (UP v1/worker/gpu_worker.py:982-994).
Kill-switch: VLLM_SM70_TM_PRECAPTURE_WARMUP=0 skips all of it (main's
  behaviour before this). Every TP rank must see the same value: the switch
  is read before the first collective.

ported from FORK cf9492a vllm/model_executor/warmup/awq_sm70_warmup.py:
1229-1336 (``sm70_awq_warmup``); provenance in
sm70/csrc/turbomind/PROVENANCE.md, "Pre-capture warmup".
"""

from __future__ import annotations

import logging
import time
from typing import Any

from vllm_sm70.warmup import plan as P
from vllm_sm70.warmup.coord import Result, coordinate, tp_group

logger = logging.getLogger("vllm.sm70.warmup")

KILL_SWITCH = "VLLM_SM70_TM_PRECAPTURE_WARMUP"
CACHE_PATH = "VLLM_SM70_GEMM_LUT_PATH"
EXPORT_PATH = "VLLM_SM70_GEMM_LUT_EXPORT_PATH"
# Ops whose presence means the build has TurboMind (K03).
REQUIRED_OPS = ("sm70_gemm_import_cache", "sm70_gemm_export_cache")


def enabled() -> bool:
    from vllm_sm70 import envs

    return bool(getattr(envs, KILL_SWITCH))


def is_capturing() -> bool:
    try:
        import torch

        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        return False


def _models(worker: Any) -> list[Any]:
    """The target model and, if any, the drafter's model (same M plan)."""
    models = []
    get_model = getattr(worker, "get_model", None)
    if callable(get_model):
        models.append(get_model())
    drafter = getattr(getattr(worker, "model_runner", None), "drafter", None)
    draft_model = getattr(drafter, "model", None)
    if draft_model is not None and hasattr(draft_model, "modules"):
        models.append(draft_model)
    return models


def _sm70_device(worker: Any) -> Any | None:
    """worker.device when it is a cc 7.0 CUDA device with TurboMind built."""
    device = getattr(worker, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    import torch

    if torch.cuda.get_device_capability(device) != (7, 0):
        return None
    try:
        from vllm_sm70.ops import tm
    except Exception:
        return None
    if not all(tm.has_op(op) for op in REQUIRED_OPS):
        return None
    return device


def run(worker: Any, *, group: Any = "tp", executor: Any = None) -> Result | None:
    """Warm and coordinate; returns the Result (None when skipped).

    ``group`` is the TP group ("tp", the default), a coord.Group, or None
    (local only). ``executor(item, m)`` runs one call; tests replace it.

    Raises:
        RuntimeError: called during CUDA graph capture (it would tune inside
            the graph), the warmup failed on some TP rank, or the TP ranks
            ended with different dispatch caches (digest mismatch).
    """
    if is_capturing():
        raise RuntimeError(
            "vllm_sm70: TurboMind pre-capture warmup called during CUDA graph "
            "capture; it must run before capture_model."
        )
    if not enabled():
        logger.info("vllm_sm70: TurboMind pre-capture warmup off (%s=0).", KILL_SWITCH)
        return None
    vllm_config = getattr(worker, "vllm_config", None)
    if vllm_config is None:
        return None
    if group == "tp":
        par = getattr(vllm_config, "parallel_config", None)
        # Elastic EP re-warms one process at a time (UP elastic_execute.py:698,
        # process_local_only): no collectives there. SM70 MoE refuses EP anyway.
        group = None if getattr(par, "enable_elastic_ep", False) else tp_group()

    device = (
        _sm70_device(worker) if executor is None else getattr(worker, "device", None)
    )
    # A local failure here must not skip the collectives below (the other
    # ranks would wait forever): join them without work, then raise.
    build_error: Exception | None = None
    plan = None
    try:
        if device is not None:
            plan = P.build(_models(worker), vllm_config)
    except Exception as exc:
        build_error = exc
    has_work = bool(plan and plan.items)

    from vllm_sm70.warmup import cache

    exported_records: list[int] = []  # filled by the real export op
    if executor is None:
        from vllm_sm70.warmup import kinds

        executor = kinds.run

        def export() -> bytes | None:
            table, records = cache.export_table(device)
            exported_records[:] = [records]
            return table

        def import_(payload: bytes) -> int:
            return cache.import_bytes(device, payload)
    else:
        export = getattr(executor, "export", lambda: None)
        import_ = getattr(executor, "import_", lambda payload: 0)

    rank = 0 if group is None else group.rank
    lut_path, export_path = _paths(device, rank)
    calls = _calls(plan)
    ident = cache.identity(device) if has_work and (lut_path or export_path) else None
    lut = cache.load_lut(lut_path, ident) if (has_work and lut_path) else None
    missing = lut.missing(calls) if lut is not None else []
    if lut is not None and missing:
        _warn_missing(lut, missing, len(calls))

    tuned: list[int] = [0]  # calls this rank ran to tune (source only)

    def warm() -> dict[str, int]:
        return _warm(plan, executor, device)

    def tune_all() -> dict[str, int]:
        counts = warm()
        tuned[0] = sum(counts.values())
        return counts

    def tune_uncovered() -> dict[str, int]:
        # With a LUT only the calls it does not cover are measured, and before
        # the import (see coord.coordinate). The rest come from the LUT.
        counts = _warm(plan, executor, device, only=set(missing))
        tuned[0] = sum(counts.values())
        return counts

    t0 = time.perf_counter()
    result = coordinate(
        group,
        has_work=has_work,
        warm=warm,
        export_bytes=export,
        import_bytes=import_,
        source_tune=tune_uncovered if lut is not None else tune_all,
        import_lut=(lambda: import_(lut.table)) if lut is not None else None,
    )
    if build_error is not None:
        raise RuntimeError(
            "vllm_sm70: TurboMind pre-capture warmup could not plan this rank's "
            f"layers ({type(build_error).__name__}: {build_error}). "
            f"{KILL_SWITCH}=0 skips the warmup."
        ) from build_error
    seconds = time.perf_counter() - t0
    lut_parts = [f"tuned (measured) {tuned[0]} calls on this rank"]
    if lut_path:
        lut_parts.append(
            f"LUT {lut.path}: {result.lut_records} records imported, "
            f"{len(calls) - len(missing)}/{len(calls)} calls covered"
            if lut is not None
            else f"LUT {lut_path}: not loaded"
        )
    disagree = result.mode != "local" and not result.agree
    if export_path and has_work and rank == result.source:
        lut_parts.append(
            f"LUT export {export_path}: not written (ranks disagree)"
            if disagree
            else _export_lut(export, export_path, calls, exported_records, ident)
        )
    _log_summary(plan, result, seconds, group, lut_parts)
    if disagree:
        # Every rank holds the same gathered digests, so every rank raises
        # here; none is left waiting in a collective.
        raise RuntimeError(
            "vllm_sm70: TP ranks hold different TurboMind dispatch caches after "
            "the pre-capture warmup (digests "
            f"{[d[:12] if d else None for d in result.digests]}, mode "
            f"{result.mode}); CUDA graph replay would differ between ranks. "
            f"{KILL_SWITCH}=0 skips the warmup."
        )
    return result


def _paths(device: Any, rank: int) -> tuple[str | None, str | None]:
    from vllm_sm70 import envs
    from vllm_sm70.warmup import cache

    idx = _index(device)
    return (
        cache.resolve_path(getattr(envs, CACHE_PATH), idx, rank),
        cache.resolve_path(getattr(envs, EXPORT_PATH), idx, rank),
    )


def _calls(plan: P.Plan | None) -> list[tuple]:
    from vllm_sm70.warmup.cache import call_id

    if plan is None:
        return []
    return [call_id(i.kind, i.key, m) for i in plan.items for m in i.ms]


def _warn_missing(lut: Any, missing: list[tuple], total: int) -> None:
    by_shape: dict[tuple, list[int]] = {}
    for kind, key, m in missing:
        by_shape.setdefault((kind, key), []).append(m)
    shapes = "; ".join(
        f"{kind} {list(key)} M{_ranges(sorted(ms))}"
        for (kind, key), ms in by_shape.items()
    )
    why = "it has no manifest" if lut.covers is None else "its manifest lacks them"
    logger.warning(
        "vllm_sm70: TurboMind LUT %s does not cover %d of %d warmup calls (%s); "
        "tuning them now, so they are not pinned: %s",
        lut.path,
        len(missing),
        total,
        why,
        shapes,
    )


def _export_lut(
    export: Any,
    path: str,
    calls: list[tuple],
    records: list[int],
    ident: dict[str, Any] | None = None,
) -> str:
    """Write the source rank's whole dispatch cache for a later run's
    VLLM_SM70_GEMM_LUT_PATH (fork _save_lut_cache, atomic here). Returns the
    summary fragment; failures are a WARNING, never fatal."""
    from vllm_sm70.warmup import cache

    try:
        table = export()
        if not table:
            logger.warning(
                "vllm_sm70: empty TurboMind dispatch cache; %s not written.", path
            )
            return f"LUT export {path}: not written (empty cache)"
        n = records[0] if records else None
        sha = cache.write_lut(path, table, calls, n, ident)
    except Exception as exc:
        logger.warning("vllm_sm70: TurboMind LUT export to %s failed (%s).", path, exc)
        return f"LUT export {path}: failed"
    return (
        f"LUT exported to {path}: {'?' if n is None else n} records, "
        f"{len(table)} bytes, sha256 {sha}, manifest {cache.manifest_path(path)}"
    )


def _index(device: Any) -> int:
    idx = getattr(device, "index", None)
    return 0 if idx is None else int(idx)


def _warm(
    plan: P.Plan | None, executor: Any, device: Any, only: set | None = None
) -> dict[str, int]:
    counts = {kind: 0 for kind in P.KINDS}
    if plan is None:
        return counts
    import torch

    from vllm_sm70.warmup.cache import call_id

    with torch.inference_mode():
        for item in plan.items:
            for m in item.ms:
                if only is not None and call_id(item.kind, item.key, m) not in only:
                    continue
                executor(item, m)
                counts[item.kind] += 1
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
    return counts


def _log_summary(
    plan: P.Plan | None,
    res: Result,
    seconds: float,
    group: Any,
    extra: list[str] | None = None,
) -> None:
    if plan is None or not plan.items:
        logger.info(
            "vllm_sm70: TurboMind pre-capture warmup: no TurboMind layers to "
            "tune on this rank."
        )
        return
    by_kind = plan.by_kind()
    parts = []
    for kind in P.KINDS:
        items = by_kind[kind]
        if items:
            ms = sorted({m for i in items for m in i.ms})
            parts.append(
                f"{kind} {len(items)} shapes x M{_ranges(ms)} = {res.counts.get(kind, 0)} calls"
            )
    for kind, why in plan.skipped.items():
        parts.append(f"{kind} skipped: {why}")
    mv = plan.m_values
    if res.mode == "local":
        tp = "TP 1"
    else:
        dg = res.digests[res.source] if res.digests else None
        tp = (
            f"TP {len(res.digests)} {res.mode}, source rank {res.source}, "
            f"{res.imported} records imported, ranks "
            + (f"agree (cache {dg[:12]})" if res.agree and dg else "DISAGREE")
        )
    logger.info(
        "vllm_sm70: TurboMind pre-capture warmup done in %.1f s: %s; capture "
        "sizes %s, verify M %d, eager decode M%s; %s; %s.",
        seconds,
        "; ".join(parts),
        list(mv.capture),
        mv.verify_m,
        _ranges(list(mv.eager)) if mv.eager else " none",
        tp,
        "; ".join(extra or []),
    )


def _ranges(ms: list[int]) -> str:
    """[1,2,3,5,8,9] -> " 1-3,5,8-9"."""
    if not ms:
        return " none"
    out, start, prev = [], ms[0], ms[0]
    for m in ms[1:] + [None]:  # type: ignore[list-item]
        if m is not None and m == prev + 1:
            prev = m
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if m is not None:
            start = prev = m
    return " " + ",".join(out)
