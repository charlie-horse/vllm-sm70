# SPDX-License-Identifier: Apache-2.0
"""Read the `_sm70_C` op schemas straight from the C++ sources (no build)."""

from __future__ import annotations

import pathlib
import re

CSRC = pathlib.Path(__file__).resolve().parents[2] / "csrc"
# K01's own registry (abi_version) is not a TurboMind op.
SKIP_FILES = {"torch_bindings_sm70.cpp"}

_BLOCK = re.compile(r"TORCH_LIBRARY(?:_FRAGMENT)?\(\s*(\w+)\s*,\s*\w+\s*\)\s*\{")
_DEF = re.compile(r'm\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')


def _block_end(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced registration block")


def sm70_schemas(awq: bool = True) -> dict[str, str]:
    """{op name: schema} for every m.def in a TORCH_LIBRARY*(_sm70_C) block.

    With awq=False, defs inside `#ifdef VLLM_SM70_TM_AWQ` are left out.
    """
    out: dict[str, str] = {}
    for path in sorted(CSRC.rglob("*")):
        if path.suffix not in (".cu", ".cpp", ".cc") or path.name in SKIP_FILES:
            continue
        text = path.read_text()
        for m in _BLOCK.finditer(text):
            if m.group(1) != "_sm70_C":
                continue
            end = _block_end(text, m.end() - 1)
            before = text[: m.start()]
            in_awq = before.rfind("#ifdef VLLM_SM70_TM_AWQ") > before.rfind("#endif")
            if in_awq and not awq:
                continue
            for d in _DEF.finditer(text, m.end(), end):
                schema = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', d.group(1)))
                name = schema.split("(", 1)[0].strip()
                assert name not in out, f"duplicate def {name} in {path.name}"
                out[name] = schema
    return out


def registration_sources() -> list[pathlib.Path]:
    return [
        p
        for p in sorted(CSRC.rglob("*"))
        if p.suffix in (".cu", ".cpp", ".cc", ".cuh", ".h")
        and "_sm70_C" in p.read_text(errors="replace")
    ]
