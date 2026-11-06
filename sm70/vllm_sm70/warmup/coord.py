# SPDX-License-Identifier: Apache-2.0
"""TP coordination of the pre-capture warmup: one rank measures, all replay.

ported from FORK cf9492a vllm/model_executor/warmup/awq_sm70_warmup.py:190-252
(``_run_coordinated_dense_warmup``). The mechanism is the fork's: the source
rank tunes, exports its whole TurboMind dispatch cache
(``sm70_gemm_export_cache``), the bytes are broadcast on the TP group's CPU
(gloo) group, and every rank, the source included, imports them
(``sm70_gemm_import_cache``) and then replays the same calls. Import replaces
a rank's entry for every shape in the payload (dispatch_cache.cu:212-239),
and a replayed kMeasure call returns early on an exact cache hit
(gemm.cu Measure), so every rank ends on the source's launch specs. Import
also marks the device "imported", so in capture any shape of a kind gets
kReuse (gemm_runtime.cpp:441-475, :548-559).

Changes from the fork:
- One phase covers every kind (NVFP4/FP8 dense and NVFP4/FP8 MoE). The fork
  coordinated only the dense routes and let each rank tune MoE on its own.
- The source is the lowest rank that has TurboMind work (rank 0 in practice)
  and is agreed by an all-gather first, so a rank without a layer of some
  kind, or without any, still makes the same collectives and never hangs.
- A failure on any rank is shared before anyone raises, so no rank is left
  waiting in a collective. The fork used a barrier and let errors escape.
- The ranks compare sha256 digests of their final caches, for the log.

Collective sequence, identical on every rank whatever its layers:
  1. all_gather(has_work)
  2. broadcast(payload or error) from the source
  3. all_gather(digest or error)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("vllm.sm70.warmup")


class Group(Protocol):
    """The two collectives the protocol needs, over a CPU process group."""

    rank: int
    world_size: int

    def all_gather_object(self, obj: Any) -> list[Any]: ...

    def broadcast_object(self, obj: Any, src: int) -> Any: ...


class TorchGroup:
    """Group over a torch.distributed CPU (gloo) group, e.g. the TP group's."""

    def __init__(self, cpu_group: Any, rank: int, ranks: list[int]) -> None:
        self.cpu_group = cpu_group
        self.rank = rank
        self.ranks = list(ranks)
        self.world_size = len(self.ranks)

    def all_gather_object(self, obj: Any) -> list[Any]:
        import torch.distributed as dist

        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj, group=self.cpu_group)
        return out

    def broadcast_object(self, obj: Any, src: int) -> Any:
        import torch.distributed as dist

        box = [obj if self.rank == src else None]
        dist.broadcast_object_list(box, src=self.ranks[src], group=self.cpu_group)
        return box[0]


def tp_group() -> TorchGroup | None:
    """The TP group's CPU side, or None for TP 1 / no distributed setup.

    Uses the gloo ``cpu_group`` directly, not GroupCoordinator.broadcast_object,
    which may go through the scheduler's message-queue broadcaster.
    """
    try:
        from vllm.distributed import parallel_state as ps
    except ImportError:
        return None
    if not ps.model_parallel_is_initialized():
        return None
    g = ps.get_tp_group()
    if g.world_size <= 1:
        return None
    return TorchGroup(g.cpu_group, g.rank_in_group, g.ranks)


@dataclass
class Result:
    counts: dict[str, int] = field(default_factory=dict)  # kind -> calls run
    mode: str = "local"  # "local" (TP 1), "coordinated", "fallback", "none"
    source: int = 0
    imported: int = 0  # records imported (payload and LUT file)
    lut_records: int = 0  # of which from the LUT file
    digests: list[str | None] = field(default_factory=list)
    agree: bool = True


def digest(payload: bytes | None) -> str | None:
    return None if payload is None else hashlib.sha256(payload).hexdigest()


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class WarmupError(RuntimeError):
    pass


def coordinate(
    group: Group | None,
    *,
    has_work: bool,
    warm: Callable[[], dict[str, int]],
    export_bytes: Callable[[], bytes | None],
    import_bytes: Callable[[bytes], int],
    source_tune: Callable[[], Any] | None = None,
    import_lut: Callable[[], int] | None = None,
) -> Result:
    """Run ``warm`` so that every rank ends with the same dispatch cache.

    ``warm`` makes this rank's TurboMind calls and returns calls per kind.
    ``export_bytes``/``import_bytes`` move the whole dispatch cache.
    ``source_tune`` is what the source runs to tune (default ``warm``); with a
    LUT file it runs only the calls the LUT does not cover. ``import_lut``
    (a LUT file is set and loaded) runs on every rank with work: on the
    source after ``source_tune`` and before the export, elsewhere before the
    payload import. Tuning before the LUT import matters: an import marks the
    device "imported", after which NVFP4/FP8 dense calls take kReuse
    (LowerBound) and no longer measure.
    """
    tune = source_tune if source_tune is not None else warm
    if group is None or group.world_size <= 1:
        res = Result(mode="local")
        if has_work:
            tune()
            if import_lut is not None:
                res.lut_records = int(import_lut())
                res.imported = res.lut_records
            res.counts = warm()
            res.digests = [digest(_safe_export(export_bytes))]
        return res

    rank = group.rank
    flags = group.all_gather_object(bool(has_work))  # 1
    workers = [r for r, f in enumerate(flags) if f]
    if not workers:
        return Result(mode="none", digests=[None] * group.world_size)
    src = workers[0]

    res = Result(mode="coordinated", source=src)
    sent: tuple[bytes | None, str | None] = (None, None)
    if rank == src:
        try:
            tune()
            if import_lut is not None:
                res.lut_records = int(import_lut())
                res.imported += res.lut_records
            sent = (export_bytes(), None)
        except Exception as exc:  # shared below, then raised on every rank
            sent = (None, _err(exc))
    payload, error = group.broadcast_object(sent, src)  # 2
    if error is not None:
        raise WarmupError(
            f"vllm_sm70: TurboMind pre-capture warmup failed on TP rank {src}: "
            f"{error}. VLLM_SM70_TM_PRECAPTURE_WARMUP=0 skips the warmup."
        )

    local_error: str | None = None
    mine: str | None = None
    if has_work:
        try:
            if import_lut is not None and rank != src:
                res.lut_records = int(import_lut())
                res.imported += res.lut_records
            if payload:
                res.imported += int(import_bytes(payload))
            else:
                # No export op or an empty cache: keep the collectives in
                # step and tune locally (the fork's fallback). Specs may differ.
                res.mode = "fallback"
                logger.warning(
                    "vllm_sm70: TP rank %d got no TurboMind dispatch cache from "
                    "rank %d; tuning locally, so ranks may pick different specs.",
                    rank,
                    src,
                )
            # Replay: a cache hit on every rank (the source measured these
            # shapes, the others imported them), and it records each shape as
            # tuned so capture reuses it.
            res.counts = warm()
            mine = digest(_safe_export(export_bytes))
        except Exception as exc:
            local_error = _err(exc)
    gathered = group.all_gather_object((mine, local_error))  # 3
    failed = [(r, e) for r, (_, e) in enumerate(gathered) if e is not None]
    if failed:
        detail = "; ".join(f"rank {r}: {e}" for r, e in failed)
        raise WarmupError(
            f"vllm_sm70: TurboMind pre-capture warmup failed ({detail}). "
            "VLLM_SM70_TM_PRECAPTURE_WARMUP=0 skips the warmup."
        )
    res.digests = [d for d, _ in gathered]
    worker_digests = {res.digests[r] for r in workers}
    res.agree = len(worker_digests) == 1 and None not in worker_digests
    return res


def _safe_export(export_bytes: Callable[[], bytes | None]) -> bytes | None:
    try:
        return export_bytes()
    except Exception:
        logger.debug("vllm_sm70: dispatch cache export failed", exc_info=True)
        return None
