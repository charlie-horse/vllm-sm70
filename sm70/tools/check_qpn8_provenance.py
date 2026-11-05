#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check sm70/csrc/qpn8/PROVENANCE.md against the tree and a fork checkout (K09).

Same rules as check_turbomind_provenance.py (K02):

For every row:
  * the new file exists and its `wc -l` matches the recorded count;
  * verbatim rows: sha256(new file) == recorded sha256, and, when --fork is
    given, sha256(fork file) == recorded sha256 too (byte-identical copy);
  * trimmed/rewritten rows with --fork: the fork file exists and its
    `wc -l` and sha256 match the recorded fork values (so the baseline that
    reviewers diff against is pinned).
It also fails if a file under sm70/csrc/qpn8 has no row.

Usage:
  sm70/tools/check_qpn8_provenance.py [--fork /path/to/fork@cf9492a]
Exit status: 0 = all good, 1 = mismatch, 2 = usage/parse error.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PROVENANCE = REPO / "sm70/csrc/qpn8/PROVENANCE.md"
TREE = REPO / "sm70/csrc/qpn8"
EXTRA: list[pathlib.Path] = []
COLUMNS = [
    "new path",
    "fork path",
    "status",
    "lines",
    "fork lines",
    "fork last-touch",
    "sha256",
    "fork sha256",
]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wc_l(path: pathlib.Path) -> int:
    return path.read_bytes().count(b"\n")


def parse_rows(text: str) -> list[dict[str, str]]:
    rows = []
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if header is None:
            if cells[: len(COLUMNS)] == COLUMNS:
                header = cells
            continue
        if set(line.replace("|", "").strip()) <= set("-: "):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork", type=pathlib.Path, help="fork checkout at cf9492a")
    args = ap.parse_args()

    rows = parse_rows(PROVENANCE.read_text())
    if not rows:
        print(f"no provenance rows parsed from {PROVENANCE}", file=sys.stderr)
        return 2

    errors: list[str] = []
    listed = set()
    for r in rows:
        new = REPO / r["new path"]
        listed.add(new)
        status = r["status"]
        if not new.is_file():
            errors.append(f"missing: {r['new path']}")
            continue
        if str(wc_l(new)) != r["lines"]:
            errors.append(f"lines {r['new path']}: {wc_l(new)} != {r['lines']}")
        if status == "verbatim" and sha256(new) != r["sha256"]:
            errors.append(f"sha256 {r['new path']}: tree differs from record")
        if status not in ("verbatim", "trimmed", "rewritten", "new", "external"):
            errors.append(f"bad status {status!r} for {r['new path']}")
        if args.fork and r["fork path"] not in ("", "-"):
            old = args.fork / r["fork path"]
            if status == "external":
                continue
            if not old.is_file():
                errors.append(f"fork file missing: {r['fork path']}")
                continue
            if str(wc_l(old)) != r["fork lines"]:
                errors.append(
                    f"fork lines {r['fork path']}: {wc_l(old)} != {r['fork lines']}"
                )
            if sha256(old) != r["fork sha256"]:
                errors.append(f"fork sha256 {r['fork path']}: differs from record")
            if status == "verbatim" and sha256(old) != sha256(new):
                errors.append(f"not verbatim: {r['new path']} vs {r['fork path']}")

    on_disk = {p for p in TREE.rglob("*") if p.is_file()} | {
        p for p in EXTRA if p.is_file()
    }
    on_disk.discard(PROVENANCE)
    for p in sorted(on_disk - listed):
        errors.append(f"no provenance row: {p.relative_to(REPO)}")

    for e in errors:
        print(e)
    verbatim = sum(r["status"] == "verbatim" for r in rows)
    print(
        f"{len(rows)} rows ({verbatim} verbatim), "
        f"{'fork checked' if args.fork else 'fork not checked'}: "
        f"{'OK' if not errors else f'{len(errors)} problem(s)'}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
