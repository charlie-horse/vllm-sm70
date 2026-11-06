# SPDX-License-Identifier: Apache-2.0
"""TurboMind dispatch-cache moves: whole-cache bytes and the optional file.

ported from FORK cf9492a vllm/model_executor/warmup/sm70_native_cache.py
(``import_cache`` :64-95, ``packed_cache_bytes`` :125-127) and
awq_sm70_warmup.py:79-88 (``_resolve_lut_path``), :124-144
(``_load_lut_cache``), :168-187 (``_export_lut_bytes``, ``_import_lut_bytes``).

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


def export_bytes(device: Any) -> bytes | None:
    """This device's whole cache as raw-table bytes (None if empty)."""
    tm = _tm()
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-gemm-export-") as tmp:
        path = Path(tmp) / "gemm-lut.bin"
        records = int(tm.sm70_gemm_export_cache(_hint(device), str(path)))
        if records <= 0 or not path.exists():
            return None
        return path.read_bytes()


def import_file(device: Any, path: str) -> int:
    """Import a cache file (raw or SM70POLICY1); 0 when absent or unusable.

    FORK awq_sm70_warmup.py:124-144: a missing file is logged, a bad file is
    a warning, never an error.
    """
    p = Path(path)
    if not p.exists():
        logger.info("vllm_sm70: no TurboMind dispatch cache at %s.", path)
        return 0
    try:
        table = legacy_table(p.read_bytes())
        if table is None:
            logger.warning(
                "vllm_sm70: %s has no %r partition; nothing imported.",
                path,
                LEGACY_KEY,
            )
            return 0
        return import_bytes(device, table)
    except Exception as exc:
        logger.warning(
            "vllm_sm70: TurboMind dispatch cache import from %s failed (%s).",
            path,
            exc,
        )
        return 0
