# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the K18 tests (not a conftest: other test dirs have one
with the same module name)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ops"))
from _schemas import sm70_schemas  # noqa: E402


def dflash2_schemas() -> dict[str, str]:
    return sm70_schemas(area="dflash2")
