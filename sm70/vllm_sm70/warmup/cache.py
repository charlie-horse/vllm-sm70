# SPDX-License-Identifier: Apache-2.0
"""TurboMind dispatch-cache moves: whole-cache bytes and the optional file.

ported from FORK cf9492a vllm/model_executor/warmup/sm70_native_cache.py
(``import_cache`` :64-95, ``packed_cache_bytes`` :125-127) and
awq_sm70_warmup.py:79-88 (``_resolve_lut_path``), :124-144
(``_load_lut_cache``), :147-165 (``_save_lut_cache``), :168-187 (``_export_lut_bytes``, ``_import_lut_bytes``).

The fork kept one table per native-policy partition (``SM70POLICY1`` files,
partition keys hashed from policy tokens). We have no policy tokens
(tm_config.h reads knobs from the environment once), so every table is the
fork's "legacy" partition: a raw file is imported as is, and a partitioned
file contributes only its "legacy" entry.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("vllm.sm70.warmup")

MAGIC = b"SM70POLICY1\n"  # FORK sm70_native_cache.py:16
LEGACY_KEY = "legacy"  # FORK sm70_native_cache.py:55-57 (_key(()))
IMPORT_OP = "sm70_gemm_import_cache"
EXPORT_OP = "sm70_gemm_export_cache"


def _tm():
    from vllm_sm70.ops import tm

    return tm


def has_cache_ops() -> bool:
    try:
        tm = _tm()
    except Exception:
        return False
    return tm.has_op(IMPORT_OP) and tm.has_op(EXPORT_OP)


def _hint(device: Any):
    import torch

    return torch.empty(0, dtype=torch.uint8, device=device)


def resolve_path(template: str | None, device_index: int, rank: int) -> str | None:
    """FORK awq_sm70_warmup.py:79-88; "{rank}" is the TP rank here (the fork
    substituted the device index for it)."""
    if not template:
        return None
    return (
        template.replace("{device}", str(device_index))
        .replace("{arch}", "sm70")
        .replace("{rank}", str(rank))
    )


def legacy_table(payload: bytes) -> bytes | None:
    """The raw TurboMind table inside a file's bytes (None: nothing usable).

    A raw table is returned unchanged; a ``SM70POLICY1`` file yields its
    "legacy" partition (FORK sm70_native_cache.py:67-95).
    """
    if not payload.startswith(MAGIC):
        return payload
    partitions = json.loads(payload[len(MAGIC) :])
    encoded = partitions.get(LEGACY_KEY)
    if encoded is None:
        return None
    return base64.b64decode(encoded, validate=True)


def import_bytes(device: Any, table: bytes) -> int:
    """Import a raw table into this device's cache; returns records imported."""
    tm = _tm()
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-gemm-import-") as tmp:
        path = Path(tmp) / "gemm-lut.bin"
        path.write_bytes(table)
        return int(tm.sm70_gemm_import_cache(_hint(device), str(path)))


def export_table(device: Any) -> tuple[bytes | None, int]:
    """This device's whole cache as (raw-table bytes or None if empty, records)."""
    tm = _tm()
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-gemm-export-") as tmp:
        path = Path(tmp) / "gemm-lut.bin"
        records = int(tm.sm70_gemm_export_cache(_hint(device), str(path)))
        if records <= 0 or not path.exists():
            return None, max(records, 0)
        return path.read_bytes(), records


def export_bytes(device: Any) -> bytes | None:
    return export_table(device)[0]


# --------------------------------------------------------------------------
# Pinning specs: LUT export (source rank) and coverage-aware import
# --------------------------------------------------------------------------
MANIFEST_SUFFIX = ".plan.json"
# 2: adds "identity" (extension, GPU, tuning knobs); a v1 manifest has none,
# so its LUT is refused (see check_identity).
MANIFEST_VERSION = 2
# LUT and manifest are read by every rank and by later runs, possibly as
# another user: world-readable, owner-writable, whatever the umask.
FILE_MODE = 0o644
# Registered "ops" knobs that never change a launch spec (diagnostics).
_NOT_DISPATCH = frozenset({"VLLM_SM70_PROFILE_TRACE"})


def write_atomic(path: str, data: bytes, mode: int = FILE_MODE) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory and
    os.replace, so a reader sees the old file or the whole new one. The file
    gets ``mode`` explicitly (mkstemp creates 0600; fchmod ignores umask)."""
    import os

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def manifest_path(path: str) -> str:
    return path + MANIFEST_SUFFIX


def call_id(kind: str, key: tuple, m: int) -> tuple:
    """One warmup call, as the manifest stores it (JSON-stable)."""
    return (kind, tuple(key), int(m))


def tuning_snapshot() -> dict[str, Any]:
    """Every registered knob that can change a TurboMind launch spec: the
    "ops" section of vllm_sm70.envs (tm_config.h's knobs, read once by the
    C++, plus the K09/K18 opt-ins), minus diagnostics. Values as envs
    returns them (raw strings, None when unset)."""
    from vllm_sm70 import envs

    names = sorted(set(envs._SECTIONS["ops"]) - _NOT_DISPATCH)
    return {name: getattr(envs, name) for name in names}


def gpu_name(device: Any) -> str | None:
    """torch's name for a CUDA ``device``; None for anything else."""
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    import torch

    return str(torch.cuda.get_device_name(device))


def ext_fingerprint() -> str:
    """The _sm70_C build: "abi=<n>;so=<sha256>" or "none" (vllm_sm70._ext,
    also what VLLM_SM70_EXT_FINGERPRINT puts into the compile cache key)."""
    from vllm_sm70 import envs

    return str(envs.VLLM_SM70_EXT_FINGERPRINT)


def identity(device: Any) -> dict[str, Any]:
    """What a dispatch cache is only valid for: the kernels it indexes (the
    extension build), the GPU it was timed on, and the knobs that steer
    dispatch and tuning. Recorded in the manifest; checked on import."""
    return {
        "ext": ext_fingerprint(),
        "gpu": gpu_name(device),
        "tuning": tuning_snapshot(),
    }


def check_identity(recorded: Any, expected: dict[str, Any]) -> list[str]:
    """Differences between a manifest's identity and this process's, as
    short phrases (empty: it matches)."""
    if not isinstance(recorded, dict):
        return ["no identity recorded (manifest from before v2)"]
    out = []
    if recorded.get("ext") != expected["ext"]:
        out.append(f"extension {recorded.get('ext')} != {expected['ext']}")
    if recorded.get("gpu") != expected["gpu"]:
        out.append(f"GPU {recorded.get('gpu')!r} != {expected['gpu']!r}")
    old = recorded.get("tuning")
    old = old if isinstance(old, dict) else {}
    new = expected["tuning"]
    knobs = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    if knobs:
        out.append(
            "tuning knobs "
            + ", ".join(f"{n}={old.get(n)!r}->{new.get(n)!r}" for n in knobs)
        )
    return out


def write_lut(
    path: str,
    table: bytes,
    calls: list[tuple],
    records: int | None = None,
    ident: dict[str, Any] | None = None,
) -> str:
    """Write the raw table and its manifest (both atomically, mode 0644);
    return the table's sha256. The manifest lists the warmup calls the table
    was tuned for, which is what the import checks coverage against, and
    ``ident`` (see identity()), which the import checks validity against.
    The manifest is written first, so a table never sits next to a stale
    manifest."""
    import hashlib

    manifest = {
        "version": MANIFEST_VERSION,
        "sha256": hashlib.sha256(table).hexdigest(),
        "records": records,
        "identity": ident,
        "calls": [[k, list(key), m] for k, key, m in calls],
    }
    write_atomic(manifest_path(path), json.dumps(manifest, sort_keys=True).encode())
    write_atomic(path, table)
    return manifest["sha256"]


class Lut:
    """A loaded LUT file: the raw table and the calls it covers (None:
    unknown, e.g. a fork file without a manifest)."""

    def __init__(self, path: str, table: bytes, covers: set[tuple] | None):
        self.path = path
        self.table = table
        self.covers = covers

    def missing(self, calls: list[tuple]) -> list[tuple]:
        if self.covers is None:
            return list(calls)
        return [c for c in calls if c not in self.covers]


class LutRefused(RuntimeError):
    """VLLM_SM70_GEMM_LUT_STRICT=1 and the LUT's manifest is missing, stale
    or written for another build or setup."""


def _refused(path: str, problems: list[str]) -> LutRefused:
    return LutRefused(
        f"vllm_sm70: TurboMind LUT {path} refused (VLLM_SM70_GEMM_LUT_STRICT=1): "
        + "; ".join(problems)
        + ". Re-export it with VLLM_SM70_GEMM_LUT_EXPORT_PATH on this build "
        "and setup, or unset VLLM_SM70_GEMM_LUT_STRICT."
    )


def load_lut(
    path: str, expected: dict[str, Any] | None = None, strict: bool = False
) -> Lut | None:
    """Read a LUT (raw or SM70POLICY1) and its manifest; None if absent or
    unusable (logged).

    With ``expected`` (identity() of this process), a manifest whose
    identity differs refuses the LUT: a WARNING naming each difference, None
    returned, so nothing is imported and the warmup tunes as without a LUT.
    A stale cache is an expected state after a rebuild or a knob change, so
    it is not fatal. A LUT without a manifest (fork files) cannot be
    checked and keeps the fork behaviour (every call tuned, then imported).

    ``strict`` (VLLM_SM70_GEMM_LUT_STRICT=1, gate runs): a missing or
    unreadable LUT file, or any manifest problem (missing, unreadable, not
    version MANIFEST_VERSION, sha256 not matching the table, identity
    differing) raises LutRefused listing them, instead of the INFO/WARNING
    and re-tune.
    """
    import hashlib

    p = Path(path)
    if not p.exists():
        if strict:
            raise _refused(path, ["no LUT file"])
        logger.info("vllm_sm70: no TurboMind LUT at %s; tuning as usual.", path)
        return None
    try:
        table = legacy_table(p.read_bytes())
    except Exception as exc:
        if strict:
            raise _refused(
                path, [f"cannot read the LUT ({type(exc).__name__}: {exc})"]
            ) from exc
        logger.warning("vllm_sm70: cannot read TurboMind LUT %s (%s).", path, exc)
        return None
    if not table:
        if strict:
            raise _refused(path, ["the LUT has no usable table"])
        logger.warning("vllm_sm70: TurboMind LUT %s has no usable table.", path)
        return None
    covers = None
    problems: list[str] = []  # for strict
    diffs: list[str] = []
    mp = Path(manifest_path(path))
    if not mp.exists():
        problems.append(f"no manifest {mp}")
    else:
        try:
            manifest = json.loads(mp.read_bytes())
            if manifest.get("version") != MANIFEST_VERSION:
                problems.append(
                    f"manifest version {manifest.get('version')!r}, "
                    f"expected {MANIFEST_VERSION}"
                )
            if manifest.get("sha256") != hashlib.sha256(table).hexdigest():
                problems.append("manifest sha256 does not match the table")
                if not strict:
                    logger.warning(
                        "vllm_sm70: %s does not match %s; treating every shape "
                        "as uncovered.",
                        mp,
                        path,
                    )
            else:
                covers = {call_id(k, tuple(key), m) for k, key, m in manifest["calls"]}
        except Exception as exc:
            problems.append(f"bad manifest {mp} ({type(exc).__name__}: {exc})")
            if not strict:
                logger.warning("vllm_sm70: bad LUT manifest %s (%s).", mp, exc)
        else:
            if expected is not None:
                diffs = check_identity(manifest.get("identity"), expected)
                problems += [d for d in diffs if d not in problems]
    if strict and problems:
        raise _refused(path, problems)
    if diffs:
        logger.warning(
            "vllm_sm70: refusing TurboMind LUT %s: its manifest %s was "
            "written for another build or setup (%s); not loading it, "
            "tuning as without a LUT.",
            path,
            mp,
            "; ".join(diffs),
        )
        return None
    return Lut(path, table, covers)
