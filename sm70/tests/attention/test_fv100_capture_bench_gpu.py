# SPDX-License-Identifier: Apache-2.0
"""K07 GPU benchmark: captured-decode envelope cost (sm70/tools/bench_fv100_capture.py).

Prints the table (run with -s to see it); asserts only that every number
was measured and that each captured workspace covers its grid in tokens,
the envelope one at least as many as the context one. Not in bytes: a
>= 32K envelope plans 1024-token partitions and a shorter context 256
(fv100._select_default_decode_partition_size), so the envelope workspace
can be the smaller (flint-tp2 at 32768, ctx 16384: 32 x 1024 vs 64 x 256
partitions, 0.13 vs 0.26 MiB at B=1).

    pytest -m "gpu and slow" -s sm70/tests/attention/test_fv100_capture_bench_gpu.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "bench_fv100_capture.py"


def _tool():
    spec = importlib.util.spec_from_file_location("bench_fv100_capture", TOOL)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves the tool's (string) annotations through
    # sys.modules[cls.__module__]; an unregistered module is None there.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("batch", [1, 8])
def test_capture_envelope_benchmark(fv100_gpu, batch):
    bench = _tool()
    rows = bench.run(batch=batch, iters=100)
    assert {r.shape for r in rows} == set(bench.SHAPES)
    for r in rows:
        print(
            f"{r.shape} mml={r.max_model_len} ctx={r.context} B={r.batch} {r.route}: "
            f"eager {r.eager_us:.1f}us, graph-env {r.graph_env_us:.1f}us, "
            f"graph-ctx {r.graph_ctx_us:.1f}us, ws-env {r.workspace_env_mib:.2f}MiB "
            f"p{r.partition_env}, ws-ctx {r.workspace_ctx_mib:.2f}MiB p{r.partition_ctx}"
        )
        assert min(r.eager_us, r.graph_env_us, r.graph_ctx_us) > 0
        assert r.tokens_ctx >= r.context, r
        assert r.tokens_env >= max(r.max_model_len, r.tokens_ctx), r
