#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check sm70/csrc/flash_v100/PROVENANCE.md against the tree and a fork checkout.

Two tables are checked (same idea as check_turbomind_provenance.py, K02):

Files (`new path | fork path | status | lines | fork lines | fork last-touch |
sha256 | fork sha256`):
  * the new file exists, its `wc -l` and sha256 match the record;
  * with --fork: the fork file's `wc -l` and sha256 match the record;
    verbatim rows are byte-identical; trimmed rows are a line subsequence of
    the fork file (deletions only).
Definitions (`new file | definition | lines | fork lines | status | sha256`):
  * sha256 of the new file's lines `lines` (joined with "\\n") matches;
  * with --fork: verbatim rows equal the fork's `fork lines` text; trimmed
    rows are a line subsequence of it.
Every file under sm70/csrc/flash_v100 (except PROVENANCE.md) needs a file row.

Usage:
  sm70/tools/check_flash_v100_provenance.py [--fork /path/to/fork@cf9492a]
Exit status: 0 = all good, 1 = mismatch, 2 = usage/parse error.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TREE = REPO / "sm70/csrc/flash_v100"
PROVENANCE = TREE / "PROVENANCE.md"
FILE_COLUMNS = [
    "new path",
    "fork path",
    "status",
    "lines",
    "fork lines",
    "fork last-touch",
    "sha256",
    "fork sha256",
]
DEF_COLUMNS = ["new file", "definition", "lines", "fork lines", "status", "sha256"]
FILE_STATUSES = ("verbatim", "trimmed", "adapted", "new")
DEF_STATUSES = ("verbatim", "trimmed", "adapted", "new")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wc_l(path: pathlib.Path) -> int:
    return path.read_bytes().count(b"\n")


def parse_table(text: str, columns: list[str]) -> list[dict[str, str]]:
    rows = []
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if header is None:
            if cells[: len(columns)] == columns:
                header = cells
            continue
        if set(line.replace("|", "").strip()) <= set("-: "):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def line_block(path: pathlib.Path, span: str) -> list[str]:
    a, b = (int(x) for x in span.split("-"))
    return path.read_text().split("\n")[a - 1 : b]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork", type=pathlib.Path, help="fork checkout at cf9492a")
    args = ap.parse_args()

    text = PROVENANCE.read_text()
    files = parse_table(text, FILE_COLUMNS)
    defs = parse_table(text, DEF_COLUMNS)
    if not files or not defs:
        print(f"no provenance rows parsed from {PROVENANCE}", file=sys.stderr)
        return 2

    errors: list[str] = []
    fork_of: dict[str, str] = {}
    listed = set()
    for r in files:
        new = REPO / r["new path"]
        listed.add(new)
        status = r["status"]
        if status not in FILE_STATUSES:
            errors.append(f"bad status {status!r} for {r['new path']}")
        if not new.is_file():
            errors.append(f"missing: {r['new path']}")
            continue
        if str(wc_l(new)) != r["lines"]:
            errors.append(f"lines {r['new path']}: {wc_l(new)} != {r['lines']}")
        if sha256_bytes(new.read_bytes()) != r["sha256"]:
            errors.append(f"sha256 {r['new path']}: tree differs from record")
        fork_of[r["new path"]] = r["fork path"]
        if not args.fork or r["fork path"] in ("", "-"):
            continue
        old = args.fork / r["fork path"]
        if not old.is_file():
            errors.append(f"fork file missing: {r['fork path']}")
            continue
        if str(wc_l(old)) != r["fork lines"]:
            errors.append(
                f"fork lines {r['fork path']}: {wc_l(old)} != {r['fork lines']}"
            )
        if sha256_bytes(old.read_bytes()) != r["fork sha256"]:
            errors.append(f"fork sha256 {r['fork path']}: differs from record")
        if status == "verbatim" and old.read_bytes() != new.read_bytes():
            errors.append(f"not verbatim: {r['new path']} vs {r['fork path']}")
        if status == "trimmed" and not is_subsequence(
            new.read_text().split("\n"), old.read_text().split("\n")
        ):
            errors.append(f"not deletions-only: {r['new path']} vs {r['fork path']}")

    for r in defs:
        new = REPO / r["new file"]
        where = f"{r['new file']}:{r['definition']}"
        if r["status"] not in DEF_STATUSES:
            errors.append(f"bad status {r['status']!r} for {where}")
        if not new.is_file():
            errors.append(f"missing: {r['new file']}")
            continue
        block = line_block(new, r["lines"])
        if sha256_bytes("\n".join(block).encode()) != r["sha256"]:
            errors.append(f"sha256 {where} ({r['lines']}): tree differs from record")
        if not args.fork or r["fork lines"] in ("", "-"):
            continue
        fork_path = fork_of.get(r["new file"])
        if not fork_path or fork_path == "-":
            errors.append(f"{where}: no fork file in the files table")
            continue
        old_block = line_block(args.fork / fork_path, r["fork lines"])
        if r["status"] == "verbatim" and old_block != block:
            errors.append(f"not verbatim: {where} vs fork {r['fork lines']}")
        if r["status"] == "trimmed" and not is_subsequence(block, old_block):
            errors.append(f"not deletions-only: {where} vs fork {r['fork lines']}")

    on_disk = {p for p in TREE.rglob("*") if p.is_file()}
    on_disk.discard(PROVENANCE)
    for p in sorted(on_disk - listed):
        errors.append(f"no provenance row: {p.relative_to(REPO)}")

    for e in errors:
        print(e)
    verbatim = sum(r["status"] == "verbatim" for r in defs)
    print(
        f"{len(files)} files, {len(defs)} definitions ({verbatim} verbatim), "
        f"{'fork checked' if args.fork else 'fork not checked'}: "
        f"{'OK' if not errors else f'{len(errors)} problem(s)'}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
