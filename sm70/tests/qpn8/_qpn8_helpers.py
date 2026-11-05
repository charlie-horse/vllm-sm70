# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the K09 QPN8 tests (not a conftest: other test dirs
have conftest modules with the same name)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ops"))
from _schemas import sm70_schemas  # noqa: E402

# Dense block-FP8 projections per TP2 rank, as (N, K) = weight shape.
# From the public HF configs (not re-read on c4130): Qwen3.6-35B-A3B hidden
# 2048, 16 q / 2 kv heads x 256 with output gate, GDN 16 k / 32 v heads x 128,
# shared expert 512; Qwen3.8-27B hidden 5120, 24 q / 4 kv heads x 256 with
# output gate, GDN 16 k / 48 v heads x 128, MLP 17408. GDN in_proj_ba
# (N = 2 * Hv / TP = 32 or 48) is not 128-aligned and so never QPN8.
TP2_SHAPES = {
    "qwen3.6-qkv_proj": (4608, 2048),
    "qwen3.6-o_proj": (2048, 2048),
    "qwen3.6-gdn.in_proj_qkvz": (6144, 2048),
    "qwen3.6-gdn.out_proj": (2048, 2048),
    "qwen3.6-shared_expert.gate_up_proj": (512, 2048),
    "qwen3.6-shared_expert.down_proj": (2048, 256),
    "qwen3.8-qkv_proj": (7168, 5120),
    "qwen3.8-o_proj": (5120, 3072),
    "qwen3.8-gdn.in_proj_qkvz": (8192, 5120),
    "qwen3.8-gdn.out_proj": (5120, 3072),
    "qwen3.8-mlp.gate_up_proj": (17408, 5120),
    "qwen3.8-mlp.down_proj": (5120, 8704),
}


def qpn8_schemas() -> dict[str, str]:
    return sm70_schemas(area="qpn8")
