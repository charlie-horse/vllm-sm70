# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the K08 GDN tests (not a conftest: tests/ops has one
with the same module name)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ops"))
from _schemas import sm70_schemas  # noqa: E402


def gdn_schemas() -> dict[str, str]:
    return sm70_schemas(area="gdn")


def mock_ctor_env(monkeypatch):
    """Let upstream ChunkGatedDeltaRule() be built without a VllmConfig."""
    from vllm.config import CompilationConfig
    from vllm.model_executor import custom_op
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m

    cc = CompilationConfig(custom_ops=["none"])
    monkeypatch.setattr(custom_op, "get_cached_compilation_config", lambda: cc)
    monkeypatch.setattr(m, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        m, "_resolve_gdn_prefill_backend", lambda _c: ("auto", "triton")
    )
    monkeypatch.setattr(m, "_log_gdn_backend_decision", lambda *a, **k: None)
    return m
