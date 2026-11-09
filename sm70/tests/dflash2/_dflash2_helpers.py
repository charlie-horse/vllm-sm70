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


def qwen_vllm_config(*, arch="Qwen3_5ForConditionalGeneration", k=7, sm70=None):
    """A getattr-only VllmConfig stand-in. The defaults are the fork's Qwen3.8
    DFlash2 verifier contract (M1 dflash2_verifier_qualified): Qwen3_5 arch,
    fp16, hidden 5120, heads 24/4 x 256, dflash k=7, selector_top_k 16, PP1.
    ``arch="Qwen3_5MoeForConditionalGeneration"`` is Qwen3.6 (not qualified).
    """
    from types import SimpleNamespace

    import torch

    text = SimpleNamespace(
        hidden_size=5120, num_attention_heads=24, num_key_value_heads=4, head_dim=256
    )
    model = SimpleNamespace(
        architectures=[arch], dtype=torch.float16, hf_text_config=text
    )
    draft = SimpleNamespace(
        hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
    )
    spec = SimpleNamespace(
        method="dflash", num_speculative_tokens=k, draft_model_config=draft
    )
    parallel = SimpleNamespace(
        pipeline_parallel_size=1, enable_dbo=False, ubatch_size=0
    )
    additional = {"sm70": sm70} if sm70 is not None else {}
    return SimpleNamespace(
        model_config=model,
        speculative_config=spec,
        parallel_config=parallel,
        additional_config=additional,
    )
