# SPDX-License-Identifier: Apache-2.0
"""CPU checks: the Inkling 2 cache owners (models/inkling2_sm70/cache.py).

Needs upstream vLLM's Python importable (skipped otherwise). Owners are
built for the vendored Inkling 2 Mini config.json at TP2 in fp16. Covers the
owner count and names, the visit-to-cache mapping, the published specs, the
conv stream layout, and the planner's grouping at block 32.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inkling2_owners import build, owner_vllm_config, text_config

TP = 2
BLOCK = 32
PAGE = 65_536
CORE = range(6, 18)


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


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


@pytest.fixture
def built():
    return build(BLOCK, TP, torch.float16)


def test_owner_counts_and_registration(built):
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    from vllm_sm70.models.inkling2_sm70.cache import (
        Inkling2AttentionCache,
        Inkling2ConvState,
    )

    owners, vllm_config = built
    assert (len(owners.attn), len(owners.conv)) == (48, 48)
    assert list(owners.attn) == list(owners.conv)
    assert sum(not o.is_local for o in owners.attn.values()) == 8
    assert all(isinstance(o, Inkling2AttentionCache) for o in owners.attn.values())
    assert all(isinstance(o, Inkling2ConvState) for o in owners.conv.values())
    named = owners.named_owners()
    assert len(named) == 96
    assert all(isinstance(o, AttentionLayerBase) for o in named.values())
    # The runner enumerates the forward context.
    context = vllm_config.compilation_config.static_forward_context
    assert sorted(context) == sorted(named)
    assert all(context[name] is owner for name, owner in named.items())
    assert [owners.passes(layer) for layer in (0, 5, 6, 17, 18, 23)] == [
        1,
        1,
        3,
        3,
        1,
        1,
    ]


def test_names_are_unique_with_one_integer(built):
    from vllm.model_executor.models.utils import extract_layer_index

    owners, _ = built
    for kind, table in (("attn", owners.attn), ("conv_state", owners.conv)):
        for (layer, pass_idx), owner in table.items():
            suffix = f".pass{pass_idx}" if pass_idx else ""
            assert owner.prefix == f"model.layers.{layer}.{kind}{suffix}"
            integers = [part for part in owner.prefix.split(".") if part.isdigit()]
            assert integers == [str(layer)]
            assert extract_layer_index(owner.prefix) == layer
            assert (owner.layer_idx, owner.pass_idx) == (layer, pass_idx)
    assert len(owners.named_owners()) == 96


def test_visit_to_cache_mapping(built):
    from vllm_sm70.models.inkling2_sm70.cache import cache_index, layer_passes

    owners, _ = built
    config = text_config()
    visits = layer_passes(config)
    assert visits == (
        [(layer, 0) for layer in range(24)]
        + [(layer, 1) for layer in CORE]
        + [(layer, 2) for layer in CORE]
    )
    assert list(owners.attn) == visits
    # Pass 0 -> 0-23, pass 1 -> 24-35, pass 2 -> 36-47.
    assert [cache_index(config, *visit) for visit in visits] == list(range(48))
    assert [cache_index(config, layer, 1) for layer in CORE] == list(range(24, 36))
    assert [cache_index(config, layer, 2) for layer in CORE] == list(range(36, 48))
    for table in (owners.attn, owners.conv):
        assert [owner.cache_index for owner in table.values()] == list(range(48))
    for layer, pass_idx in ((5, 1), (18, 2), (6, 3), (24, 0)):
        with pytest.raises(ValueError):
            cache_index(config, layer, pass_idx)


def test_attention_specs(built):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    owners, vllm_config = built
    local = set(text_config().local_layer_ids)
    kinds = {SlidingWindowSpec: 0, FullAttentionSpec: 0}
    for (layer, _), owner in owners.attn.items():
        spec = owner.get_kv_cache_spec(vllm_config)
        kinds[type(spec)] += 1
        assert (layer in local) == isinstance(spec, SlidingWindowSpec)
        if layer in local:
            assert spec.sliding_window == 1024
        assert (spec.block_size, spec.num_kv_heads, spec.head_size) == (BLOCK, 4, 128)
        assert spec.dtype == torch.float16
        assert spec.page_size_bytes == PAGE
    assert kinds == {SlidingWindowSpec: 40, FullAttentionSpec: 8}


def test_conv_specs_and_stream_layout(built):
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    from vllm_sm70.models.inkling2_sm70.cache import (
        STREAM_ATTN,
        STREAM_K,
        STREAM_MLP,
        STREAM_V,
    )

    owners, vllm_config = built
    for owner in owners.conv.values():
        spec = owner.get_kv_cache_spec(vllm_config)
        assert type(spec) is SlidingWindowSpec
        assert (spec.sliding_window, spec.block_size) == (4, 4)
        assert (spec.num_kv_heads, spec.head_size, spec.head_size_v) == (4, 2048, 0)
        assert spec.dtype == torch.float16
        assert spec.page_size_bytes == PAGE
        # Four streams side by side: K, V and the two full-width outputs.
        ranges = owner.stream_ranges
        assert ranges[STREAM_K] == (0, 128)
        assert ranges[STREAM_V] == (128, 128)
        assert ranges[STREAM_ATTN] == (256, 576)
        assert ranges[STREAM_MLP] == (832, 576)
        assert ranges[STREAM_MLP][0] + ranges[STREAM_MLP][1] == 1408
        assert owner.num_kv_heads * ranges[STREAM_ATTN][1] == 2304


def test_conv_cache_block_size_follows_the_bound_cache(built):
    owners, _ = built
    owner = owners.conv[6, 1]
    assert owner.cache_block_size == 4
    owner.bind_kv_cache(torch.zeros(3, 4, 8, 2048, dtype=torch.float16))
    assert owner.cache_block_size == 8


@pytest.mark.parametrize("layout", ["LBNHC", "BLNHC"])
def test_planner_gives_12_groups_of_8_with_equal_pages(built, layout):
    from vllm.config import CacheConfig
    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

    owners, owner_config = built
    specs = owners.kv_cache_specs(owner_config)
    assert len(specs) == 96
    assert {spec.page_size_bytes for spec in specs.values()} == {PAGE}

    cache_config = CacheConfig(block_size=BLOCK, enable_prefix_caching=True)
    cache_config.kv_cache_layout = layout
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_batched_tokens=8192
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, tensor_parallel_size=TP
        ),
        model_config=SimpleNamespace(max_model_len=1_048_576),
        kv_transfer_config=None,
        speculative_config=None,
        max_in_flight_tokens=8192,
    )
    groups = get_kv_cache_groups(vllm_config, specs)
    assert len(groups) == 12
    assert all(len(group.layer_names) == 8 for group in groups)
    assert sorted(n for g in groups for n in g.layer_names) == sorted(specs)
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {PAGE}
    assert {g.kv_cache_spec.page_size_padded for g in groups} == {None}
    # A group holds one kind of owner.
    for group in groups:
        assert len({".conv_state" in name for name in group.layer_names}) == 1


def test_duplicate_name_is_rejected(built):
    from vllm_sm70.models.inkling2_sm70.cache import build_cache_owners

    _, vllm_config = built
    with pytest.raises(ValueError, match="Duplicate layer name"):
        build_cache_owners(text_config(), vllm_config=vllm_config, tp_size=TP)


def test_tp_size_must_divide_the_kv_heads():
    from vllm_sm70.models.inkling2_sm70.cache import build_cache_owners

    with pytest.raises(ValueError, match="divides"):
        build_cache_owners(
            text_config(),
            vllm_config=owner_vllm_config(BLOCK, torch.float16),
            tp_size=3,
        )


def test_tp1_keeps_all_heads_and_halves_the_conv_width():
    owners, vllm_config = build(16, 1, torch.float16)
    conv = owners.conv[0, 0]
    assert (conv.num_kv_heads, conv.head_size) == (8, 1024)
    assert conv.stream_ranges[2:] == ((256, 288), (544, 288))
    assert owners.attn[0, 0].get_kv_cache_spec(vllm_config).num_kv_heads == 8
