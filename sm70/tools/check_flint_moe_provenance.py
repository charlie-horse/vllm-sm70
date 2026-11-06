#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check sm70/csrc/flint_moe/PROVENANCE.md against the tree and a fork checkout.

Same tables and rules as check_flash_v100_provenance.py (K05), applied to
the K12 area: file rows (lines, sha256; with --fork, verbatim = identical,
trimmed = deletions only) and definition rows (sha256 of a line span; with
--fork, verbatim/trimmed against the fork span). Every file under
sm70/csrc/flint_moe except PROVENANCE.md needs a file row.

Usage:
  sm70/tools/check_flint_moe_provenance.py [--fork /path/to/fork@cf9492a]
Exit status: 0 = all good, 1 = mismatch, 2 = usage/parse error.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_flash_v100_provenance as base  # noqa: E402

base.TREE = base.REPO / "sm70/csrc/flint_moe"
base.PROVENANCE = base.TREE / "PROVENANCE.md"

if __name__ == "__main__":
    sys.exit(base.main())
