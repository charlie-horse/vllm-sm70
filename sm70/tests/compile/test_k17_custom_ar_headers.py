# SPDX-License-Identifier: Apache-2.0
"""K17: the upstream headers csrc/custom_ar is built against are the pinned ones.

sm70/csrc/custom_ar reads upstream's CustomAllreduce, which lives in vLLM's
_C_custom_ar .so, through custom_all_reduce.cuh and custom_collective_common.cuh
(plus type_convert.cuh for _f16Vec). The struct layout is pinned with
static_asserts at build time and the bytes with UPSTREAM_HEADERS.sha256
(checked by CMake too). This test checks the same pins against the vLLM
source tree on the CPU, so a rebase of patches/BASE or a patch in
patches/series that touches those headers fails here before anyone builds.

The tree is $VLLM_SM70_VLLM_CSRC_DIR or the csrc/ next to the imported vllm
package (PYTHONPATH=<upstream-sm70>, as the CPU suite runs).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

SM70 = Path(__file__).resolve().parents[2]
REPO = SM70.parent
AREA = SM70 / "csrc" / "custom_ar"
PINS = AREA / "UPSTREAM_HEADERS.sha256"
PINNED_BASE = "v0.33.0"
HEADERS = {"custom_all_reduce.cuh", "custom_collective_common.cuh", "type_convert.cuh"}


def _pins() -> dict[str, str]:
    pins = {}
    for line in PINS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        digest, name = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line).groups()
        pins[name] = digest
    return pins


def _csrc() -> Path | None:
    env = os.environ.get("VLLM_SM70_VLLM_CSRC_DIR")
    if env:
        return Path(env)
    try:
        import importlib.util

        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    csrc = Path(spec.origin).resolve().parent.parent / "csrc"
    return csrc if (csrc / "custom_all_reduce.cuh").exists() else None


def test_pin_file_covers_the_included_headers():
    assert set(_pins()) == HEADERS
    source = (AREA / "tp2_ar_fused_add_rms_norm.cu").read_text()
    included = set(re.findall(r'#include "([^"]+)"', source))
    # custom_collective_common.cuh comes in through custom_all_reduce.cuh.
    assert included | {"custom_collective_common.cuh"} == HEADERS


def test_pins_are_for_the_current_base():
    base = (REPO / "patches" / "BASE").read_text().strip()
    assert base == PINNED_BASE, (
        f"patches/BASE is {base}; re-pin csrc/custom_ar for it (static_asserts "
        "and UPSTREAM_HEADERS.sha256) after checking the CustomAllreduce layout"
    )
    for line in (REPO / "patches" / "series").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patch = (REPO / "patches" / line).read_text(errors="replace")
        for header in HEADERS:
            assert f"csrc/{header}" not in patch, (
                f"{line} edits csrc/{header}; re-pin csrc/custom_ar"
            )


def test_headers_match_the_pins():
    csrc = _csrc()
    if csrc is None:
        pytest.skip("no vLLM csrc/ tree (set VLLM_SM70_VLLM_CSRC_DIR)")
    for name, digest in _pins().items():
        got = hashlib.sha256((csrc / name).read_bytes()).hexdigest()
        assert got == digest, f"{csrc / name}: sha256 {got}, pinned {digest}"
