# SPDX-License-Identifier: Apache-2.0
"""CPU checks: KV planning for Inkling 2 Mini's 96 cache layers at TP2.

Needs upstream vLLM's Python importable (skipped otherwise). The specs
come from the model's cache owners (models/inkling2_sm70/cache.py) built for
the vendored config.json (48 fp16 attention passes, 48 conv-state passes
with the hidden streams replicated per rank) and are fed to upstream's
planner.
Covers the group plan, page sizes, the scheduler and hash blocks, and the
per-request admission caps at the batched-token budgets the model serves.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inkling2_owners import build

TP = 2
PAGE = 65_536
LAYER_OUTER = "LBNHC"  # Flash-V100's layout
BLOCK_OUTER = "BLNHC"


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


requires_vllm = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")
pytestmark = requires_vllm


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


def _specs(block_size: int, tp: int = TP, dtype=torch.float16) -> dict:
    """The specs the model's cache owners publish."""
    owners, vllm_config = build(block_size, tp, dtype)
    return owners.kv_cache_specs(vllm_config)


def _vllm_config(block_size: int, layout: str, batched_tokens: int = 8192):
    from vllm.config import CacheConfig

    cache_config = CacheConfig(block_size=block_size, enable_prefix_caching=True)
    cache_config.kv_cache_layout = layout
    return SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=batched_tokens,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, tensor_parallel_size=TP
        ),
        model_config=SimpleNamespace(max_model_len=1_048_576),
        kv_transfer_config=None,
        speculative_config=None,
        # One batch in flight: --no-async-scheduling.
        max_in_flight_tokens=batched_tokens,
    )


def _plan(block_size: int, layout: str):
    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

    vllm_config = _vllm_config(block_size, layout)
    specs = _specs(block_size)
    return vllm_config, specs, get_kv_cache_groups(vllm_config, specs)


def _kinds(groups) -> list[tuple[str, int | None, int]]:
    """(spec type, sliding window, block size) per group, sorted."""
    return sorted(
        (
            type(g.kv_cache_spec).__name__,
            getattr(g.kv_cache_spec, "sliding_window", None),
            g.kv_cache_spec.block_size,
        )
        for g in groups
    )


UNIFIED_KINDS = sorted(
    [("FullAttentionSpec", None, 32)]
    + [("SlidingWindowSpec", 1024, 32)] * 5
    + [("SlidingWindowSpec", 4, 4)] * 6
)


def test_spec_counts_and_bytes_per_token():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    specs = _specs(32)
    attn = [s for n, s in specs.items() if ".attn" in n]
    conv = [s for n, s in specs.items() if ".conv_state" in n]
    full = [s for s in attn if isinstance(s, FullAttentionSpec)]
    assert (len(attn), len(full), len(conv)) == (48, 8, 48)
    assert {s.page_size_bytes // s.block_size for s in attn} == {2_048}
    assert {s.head_size for s in conv} == {2_048}
    assert {s.page_size_bytes // s.block_size for s in conv} == {16_384}
    # Only the 8 global passes grow with the context.
    assert sum(s.page_size_bytes // s.block_size for s in full) == 16_384


@pytest.mark.parametrize("layout", [LAYER_OUTER, BLOCK_OUTER])
def test_block_32_gives_12_groups_of_8_with_equal_pages(layout):
    from vllm.v1.core.kv_cache_planning import _get_kv_cache_bytes_per_block
    from vllm.v1.core.kv_cache_utils import (
        resolve_cache_hit_alignment_tokens,
        resolve_kv_cache_block_sizes,
    )

    vllm_config, specs, groups = _plan(32, layout)
    assert len(groups) == 12
    assert all(len(g.layer_names) == 8 for g in groups)
    # An exact partition of the 96 owners: no padding layers.
    assert sorted(n for g in groups for n in g.layer_names) == sorted(specs)
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {PAGE}
    assert {g.kv_cache_spec.page_size_padded for g in groups} == {None}
    assert _kinds(groups) == UNIFIED_KINDS
    assert (
        _get_kv_cache_bytes_per_block(
            groups, vllm_config.cache_config.get_resolved_kv_cache_layout()
        )
        == 8 * PAGE
    )

    kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    assert (scheduler_block, hash_block) == (32, 4)
    # Hashes are 4 tokens wide, but prefix hits still land on whole blocks.
    assert (
        resolve_cache_hit_alignment_tokens(
            kv_cache_config, vllm_config, scheduler_block, hash_block
        )
        == 32
    )


def test_block_16_layer_outer_is_lifted_to_the_block_32_plan():
    """The smaller attention page is unified up to the conv page."""
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    vllm_config, _, groups = _plan(16, LAYER_OUTER)
    assert len(groups) == 12
    assert all(len(g.layer_names) == 8 for g in groups)
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {PAGE}
    assert _kinds(groups) == UNIFIED_KINDS
    assert resolve_kv_cache_block_sizes(
        SimpleNamespace(kv_cache_groups=groups), vllm_config
    ) == (32, 4)


def test_block_16_block_outer_packs_18_groups():
    """Mixed pages are packed instead: 6 attention groups of 8 and 12 conv
    groups of 4, each filling a 262,144 B block."""
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    vllm_config, specs, groups = _plan(16, BLOCK_OUTER)
    assert sorted(len(g.layer_names) for g in groups) == [4] * 12 + [8] * 6
    assert sorted(n for g in groups for n in g.layer_names) == sorted(specs)
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {4 * PAGE}
    assert resolve_kv_cache_block_sizes(
        SimpleNamespace(kv_cache_groups=groups), vllm_config
    ) == (16, 4)


@pytest.mark.parametrize(
    "batched_tokens, conv_blocks, conv_gb, sliding_blocks, sliding_gb",
    [
        (512, 130, 0.41, 49, 0.13),
        (2048, 514, 1.62, 97, 0.25),
        (8192, 2050, 6.45, 289, 0.76),
    ],
)
def test_per_request_caps(
    batched_tokens, conv_blocks, conv_gb, sliding_blocks, sliding_gb
):
    """cdiv(window - 1 + B, block) + 1 blocks per layer, per GPU."""
    _, _, groups = _plan(32, LAYER_OUTER)
    vllm_config = _vllm_config(32, LAYER_OUTER, batched_tokens)
    total = {4: 0, 1024: 0}
    for group in groups:
        spec = group.kv_cache_spec
        window = getattr(spec, "sliding_window", None)
        if window is None:
            continue
        blocks = spec.max_admission_blocks_per_request(
            max_in_flight_tokens=batched_tokens, max_model_len=1_048_576
        )
        assert blocks == math.ceil((window - 1 + batched_tokens) / spec.block_size) + 1
        assert blocks == {4: conv_blocks, 1024: sliding_blocks}[window]
        assert spec.max_memory_usage_bytes(vllm_config) == blocks * PAGE
        total[window] += blocks * PAGE * len(group.layer_names)
    assert round(total[4] / 1e9, 2) == conv_gb
    assert round(total[1024] / 1e9, 2) == sliding_gb


@pytest.mark.parametrize(
    "tokens, gb", [(100_000, 1.64), (262_144, 4.29), (1_048_576, 17.18)]
)
def test_global_kv_per_request(tokens, gb):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    _, _, groups = _plan(32, LAYER_OUTER)
    (full,) = [g for g in groups if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    blocks = math.ceil(tokens / full.kv_cache_spec.block_size)
    assert round(blocks * PAGE * len(full.layer_names) / 1e9, 2) == gb


def test_tp1_bf16_bytes_per_token():
    """Upstream's TP1 layout: pages already equal at its default block 16."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    specs = _specs(16, tp=1, dtype=torch.bfloat16)
    assert {s.page_size_bytes for s in specs.values()} == {PAGE}
    full = [s for s in specs.values() if isinstance(s, FullAttentionSpec)]
    assert sum(s.page_size_bytes // s.block_size for s in full) == 32_768


# --- capacity: one batch in flight (sync) against two (async) -----------------------

MODEL_LEN = 8192


def _capacity(batched_tokens: int, batches: int, num_blocks: int, model_len: int):
    """Upstream's reported KV tokens and per-request worst-case blocks."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

    _, _, groups = _plan(32, LAYER_OUTER)
    vllm_config = _vllm_config(32, LAYER_OUTER, batched_tokens)
    vllm_config.max_in_flight_tokens = batches * batched_tokens
    vllm_config.model_config = SimpleNamespace(max_model_len=model_len)
    for group in groups:
        group.host_resident = False
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=groups, num_blocks=num_blocks, hisparse_host_num_blocks=None
    )
    tokens, _ = get_kv_cache_capacity(vllm_config, kv_cache_config)
    per_request = sum(
        math.ceil(
            g.kv_cache_spec.max_memory_usage_bytes(vllm_config)
            / g.kv_cache_spec.page_size_bytes
        )
        for g in groups
    )
    return tokens, per_request


@pytest.mark.parametrize(
    "batched_tokens, num_blocks, sync, async_",
    [
        # Pool blocks at an 8K model length; the pool is the same with one
        # or two batches in flight.
        (512, 31_692, (202_670, 1281), (121_944, 2129)),
        (2048, 31_608, (67_694, 3825), (35_878, 7217)),
    ],
)
def test_async_halves_the_reported_capacity_not_the_pool(
    batched_tokens, num_blocks, sync, async_
):
    """The worst case per request counts window - 1 plus every in-flight
    token in each conv and sliding-window group: 6 conv groups of 4-token
    blocks dominate it."""
    assert _capacity(batched_tokens, 1, num_blocks, MODEL_LEN) == sync
    assert _capacity(batched_tokens, 2, num_blocks, MODEL_LEN) == async_
