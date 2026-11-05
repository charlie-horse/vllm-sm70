# SPDX-License-Identifier: Apache-2.0
"""K11 CPU checks: the NVFP4 M<=2 GEMV shape set and its launch table.

What:
- The kernel's launch table (VLLM_TM_GEMV_TABLE in
  csrc/turbomind/ops/nvfp4_tm_gemv_sm70.cu) equals
  vllm_sm70.ops.tm.NVFP4_TM_GEMV_SHAPES and M3's MODELOPT_NVFP4_GEMV_SHAPES.
  An admitted shape the kernel can't run would silently fall back to
  TurboMind; one the kernel runs but M3 never admits is dead code.
- Every table entry passes the kernel's own validity rule
  (tm_gemv_config_valid), so no admitted Flint TP2 shape falls back.
- The Python mirror of the op's shape gate agrees with the table.
- The op schema matches M3's call
  (tm.nvfp4_tm_gemv_dispatch_sm70_out(out, x, w, s, 16, k_ld, q_ld, shift)).
Op-level parts of fork tests/quantization/test_sm70_modelopt_nvfp4_gemv.py
(0c9f3ee73): the Flint/non-Flint shape sets. The layer-admission parts are
M3's (tests/quant/test_m3_nvfp4_linear.py).
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _schemas import sm70_schemas  # noqa: E402

CU = (
    pathlib.Path(__file__).resolve().parents[2]
    / "csrc/turbomind/ops/nvfp4_tm_gemv_sm70.cu"
)
FLINT_TP2 = {
    "qkv": (2048, 2560),
    "o": (2048, 2048),
    "shared_gate_up": (2048, 1024),
    "shared_down": (512, 2048),
    "dense_gate_up": (2048, 6144),
    "dense_down": (3072, 2048),
}


def _launch_table():
    text = CU.read_text()
    body = text[text.index("#define VLLM_TM_GEMV_TABLE(X)") :]
    body = body[: body.index("\n\n")]
    rows = re.findall(
        r"X\(\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(true|false),\s*(\d+)\)",
        body,
    )
    return [
        (int(k), int(n), int(g), int(w), int(p), pl == "true", int(mb))
        for k, n, g, w, p, pl, mb in rows
    ]


def _config_valid(k, n, groups, warps, prefetch):
    # tm_gemv_config_valid() in nvfp4_tm_gemv_sm70.cu
    if k <= 0 or n <= 0 or n % 32:
        return False
    unit_k = 32 * groups
    return k % unit_k == 0 and (k // unit_k) % (warps * prefetch) == 0


def test_launch_table_is_the_flint_tp2_set():
    table = _launch_table()
    assert len(table) == 6
    assert {(k, n) for k, n, *_ in table} == set(FLINT_TP2.values())


@pytest.mark.parametrize("entry", _launch_table(), ids=lambda e: f"{e[0]}x{e[1]}")
def test_every_table_entry_is_launchable(entry):
    k, n, groups, warps, prefetch, _pipelined, _min_blocks = entry
    assert groups in (1, 2, 4)
    assert _config_valid(k, n, groups, warps, prefetch)
    # grid = N / (32 / G) column blocks; each covers whole 32-column tiles.
    assert n % (32 // groups) == 0


def test_python_mirror_matches_table():
    torch = pytest.importorskip("torch")  # noqa: F841
    from vllm_sm70.ops import tm

    assert tm.NVFP4_TM_GEMV_SHAPES == frozenset(FLINT_TP2.values())
    for k, n in FLINT_TP2.values():
        for m in (1, 2):
            assert tm.nvfp4_tm_gemv_takes_gemv(m, k, n, 16, 32 * k, n)
        for m in (0, 3, 8):
            assert not tm.nvfp4_tm_gemv_takes_gemv(m, k, n, 16, 32 * k, n)
        assert not tm.nvfp4_tm_gemv_takes_gemv(1, k, n, 32, 32 * k, n)
        assert not tm.nvfp4_tm_gemv_takes_gemv(1, k, n, 16, 0, n)
        assert not tm.nvfp4_tm_gemv_takes_gemv(1, k, n, 16, 32 * k, 0)
    # fork test_other_model_shapes_are_not_admitted: Qwen TP2, transposed
    # Flint, and near misses all keep TurboMind.
    for k, n in ((5120, 8704), (4096, 6144), (2560, 2048), (2048, 4096), (1024, 2048)):
        assert not tm.nvfp4_tm_gemv_takes_gemv(1, k, n, 16, 32 * k, n)


def test_m3_admits_exactly_the_kernel_shapes():
    pytest.importorskip("vllm")
    from vllm_sm70.ops import tm
    from vllm_sm70.quant import nvfp4_linear as nvl

    assert nvl.MODELOPT_NVFP4_GEMV_SHAPES == tm.NVFP4_TM_GEMV_SHAPES
    assert nvl.GEMV_OP == "nvfp4_tm_gemv_dispatch_sm70_out"


def test_schema_matches_m3_call():
    schema = sm70_schemas()["nvfp4_tm_gemv_dispatch_sm70_out"]
    assert schema == (
        "nvfp4_tm_gemv_dispatch_sm70_out(Tensor(a!) out, Tensor input, "
        "Tensor weight, Tensor scales, int group_size, int k_ld, int q_ld, "
        "int scale_shift=14) -> ()"
    )


def test_cu_has_no_fork_only_paths():
    text = CU.read_text()
    code = re.sub(r"//[^\n]*", "", text)
    for gone in ("VLLM_NVFP4_TM_GEMV_STANDALONE", "_tm_gemv_dev", "native_policy"):
        assert gone not in code
