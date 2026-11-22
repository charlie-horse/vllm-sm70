# SPDX-License-Identifier: Apache-2.0
"""M12 CPU checks: the Inkling 2 cache backends and their metadata builders
(models/inkling2_sm70/backend.py).

Needs upstream v0.33.0's Python importable (skipped otherwise). The batch is
two decode requests and one chunked-prefill request; every expected value is
written out by hand. Owners are built for the vendored Inkling 2 Mini
config.json at TP2 in fp16, block size 32 (conv blocks hold 4 tokens).
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
from _inkling2_owners import build

TP = 2
BLOCK = 32
CONV_BLOCK = 4

# Request A decodes position 39, B decodes position 6, C prefills positions
# 6-10 of its prompt (6 tokens are cached).
QUERY_LENS = [1, 1, 5]
SEQ_LENS = [40, 7, 11]
QUERY_START_LOC = [0, 1, 2, 7]
SEQ_IDX = [0, 1, 2, 2, 2, 2, 2]
QUERY_START = [0, 1, 2, 2, 2, 2, 2]
POSITIONS = [39, 6, 6, 7, 8, 9, 10]

# Attention: 32 tokens per block. A's position 39 is slot 7 of its 2nd block.
ATTN_TABLE = [[10, 11], [12, 0], [13, 0]]
ATTN_SLOTS = [11 * 32 + 7, 12 * 32 + 6, 422, 423, 424, 425, 426]
# Conv: 4 tokens per block; row r holds blocks 100 + 10 r + (0..9).
CONV_TABLE = [[100 + 10 * r + b for b in range(10)] for r in range(3)]
CONV_SLOTS = [109 * 4 + 3, 111 * 4 + 2, 486, 487, 488, 489, 490]


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


def _vllm_config(batched_tokens: int = 64):
    from vllm.config import CacheConfig

    cache_config = CacheConfig(block_size=BLOCK, enable_prefix_caching=True)
    cache_config.kv_cache_layout = "LBNHC"
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
        max_in_flight_tokens=batched_tokens,
    )


def _common(table, slots, *, pad_reqs=0, pad_tokens=0, positions=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    qsl = torch.tensor(
        QUERY_START_LOC + [QUERY_START_LOC[-1]] * pad_reqs, dtype=torch.int32
    )
    seq_lens = torch.tensor(SEQ_LENS + [0] * pad_reqs, dtype=torch.int32)
    width = len(table[0])
    return CommonAttentionMetadata(
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens,
        num_reqs=len(QUERY_LENS) + pad_reqs,
        num_actual_tokens=len(slots) + pad_tokens,
        max_query_len=max(QUERY_LENS),
        max_seq_len=max(SEQ_LENS),
        block_table_tensor=torch.tensor(
            table + [[0] * width] * pad_reqs, dtype=torch.int32
        ),
        slot_mapping=torch.tensor(slots + [-1] * pad_tokens, dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        positions=positions,
    )


def _builder(owner, owner_config, vllm_config=None):
    builder_cls = owner.get_attn_backend().get_builder_cls()
    return builder_cls(
        owner.get_kv_cache_spec(owner_config),
        [owner.prefix],
        vllm_config or _vllm_config(),
        torch.device("cpu"),
    )


def _check_tokens(md, table, slots, block_size):
    assert md.num_actual_tokens == 7
    assert md.block_size == block_size
    assert md.block_table.tolist() == table
    assert md.slot_mapping.tolist() == slots
    assert md.seq_idx.tolist() == SEQ_IDX
    assert md.query_start.tolist() == QUERY_START
    assert md.positions.tolist() == POSITIONS
    assert md.block_table.dtype == torch.int32
    assert md.slot_mapping.dtype == torch.int64
    assert md.seq_idx.dtype == torch.int32
    assert md.query_start.dtype == torch.int32
    assert md.positions.dtype == torch.int64
    # The slot of a token is where its position lands in its request's blocks.
    block = md.block_table[md.seq_idx.long(), md.positions // block_size]
    assert (block * block_size + md.positions % block_size).tolist() == slots


def test_owners_resolve_their_backends(built):
    from vllm.v1.attention.backend import AttentionBackend, MultipleOf

    from vllm_sm70.models.inkling2_sm70 import backend

    owners, owner_config = built
    attn, conv = owners.attn[6, 1], owners.conv[6, 1]
    assert attn.get_attn_backend() is backend.Inkling2AttentionBackend
    assert conv.get_attn_backend() is backend.Inkling2SconvBackend
    for owner, builder_cls, name in (
        (attn, backend.Inkling2AttentionMetadataBuilder, "INKLING2_SM70_ATTN"),
        (conv, backend.Inkling2SconvMetadataBuilder, "INKLING2_SM70_SCONV"),
    ):
        cls = owner.get_attn_backend()
        assert issubclass(cls, AttentionBackend)
        assert cls.get_name() == name
        assert cls.get_builder_cls() is builder_cls
        with pytest.raises(NotImplementedError, match="no attention impl"):
            cls.get_impl_cls()
        spec = owner.get_kv_cache_spec(owner_config)
        assert cls.customize_spec(spec) is spec
        assert cls.get_supported_kernel_block_sizes(spec) == [MultipleOf(1)]


def test_builder_flags(built):
    from vllm.v1.attention.backend import AttentionCGSupport

    owners, owner_config = built
    for owner in (owners.attn[0, 0], owners.attn[5, 0], owners.conv[0, 0]):
        builder = _builder(owner, owner_config)
        assert builder.reorder_batch_threshold == 1
        assert builder.supports_update_block_table is True
        spec = owner.get_kv_cache_spec(owner_config)
        assert type(builder).get_cudagraph_support(None, spec) is (
            AttentionCGSupport.UNIFORM_BATCH
        )


@pytest.mark.parametrize("layer", [0, 5], ids=["sliding", "global"])
def test_attention_metadata(built, layer):
    owners, owner_config = built
    md = _builder(owners.attn[layer, 0], owner_config).build(
        0, _common(ATTN_TABLE, ATTN_SLOTS)
    )
    _check_tokens(md, ATTN_TABLE, ATTN_SLOTS, BLOCK)
    assert md.sliding_window == (1024 if layer == 0 else None)
    assert md.num_reqs == 3
    assert md.max_query_len == 5
    assert md.max_seq_len == 40
    assert md.query_start_loc.tolist() == QUERY_START_LOC
    assert md.seq_lens.tolist() == SEQ_LENS
    assert md.causal is True
    # Decode-first split.
    assert (md.num_decodes, md.num_prefills) == (2, 1)
    assert (md.num_decode_tokens, md.num_prefill_tokens) == (2, 5)
    assert md.query_lens_cpu.tolist() == QUERY_LENS
    assert md.seq_lens_cpu.tolist() == SEQ_LENS
    assert md.query_lens_cpu.dtype == md.seq_lens_cpu.dtype == "int64"


def test_sconv_metadata(built):
    from vllm_sm70.models.inkling2_sm70.backend import Inkling2AttentionMetadata

    owners, owner_config = built
    md = _builder(owners.conv[0, 0], owner_config).build(
        0, _common(CONV_TABLE, CONV_SLOTS)
    )
    assert not isinstance(md, Inkling2AttentionMetadata)
    _check_tokens(md, CONV_TABLE, CONV_SLOTS, CONV_BLOCK)


def test_all_decode_batch(built):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    owners, owner_config = built
    qsl = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([40, 7], dtype=torch.int32)
    cm = CommonAttentionMetadata(
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=40,
        block_table_tensor=torch.tensor(ATTN_TABLE[:2], dtype=torch.int32),
        slot_mapping=torch.tensor(ATTN_SLOTS[:2], dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
    )
    md = _builder(owners.attn[0, 0], owner_config).build(0, cm)
    assert (md.num_decodes, md.num_prefills) == (2, 0)
    assert (md.num_decode_tokens, md.num_prefill_tokens) == (2, 0)
    assert md.seq_idx.tolist() == [0, 1]
    assert md.query_start.tolist() == [0, 1]
    assert md.positions.tolist() == [39, 6]


@pytest.mark.parametrize("kind", ["attn", "conv"])
def test_padding_tokens_and_requests_are_zeroed(built, kind):
    owners, owner_config = built
    owner = getattr(owners, kind)[0, 0]
    table, slots = (
        (ATTN_TABLE, ATTN_SLOTS)
        if kind == "attn"
        else (
            CONV_TABLE,
            CONV_SLOTS,
        )
    )
    builder = _builder(owner, owner_config)
    # A full batch first, so the padded rows of the buffers hold stale values.
    builder.build(0, _common(table, slots + slots))
    md = builder.build(0, _common(table, slots, pad_reqs=2, pad_tokens=3))
    assert md.num_actual_tokens == 7
    assert md.slot_mapping.tolist() == slots + [-1] * 3
    assert md.seq_idx.tolist() == SEQ_IDX + [0] * 3
    assert md.query_start.tolist() == QUERY_START + [0] * 3
    assert md.positions.tolist() == POSITIONS + [0] * 3
    assert md.block_table.shape[0] == 5


def test_runner_positions_are_passed_through(built):
    owners, owner_config = built
    positions = torch.tensor(POSITIONS, dtype=torch.int64)
    for owner in (owners.attn[0, 0], owners.conv[0, 0]):
        md = _builder(owner, owner_config).build(
            0, _common(ATTN_TABLE, ATTN_SLOTS, positions=positions)
        )
        assert md.positions.data_ptr() == positions.data_ptr()
        assert md.positions.tolist() == POSITIONS


def test_token_buffers_belong_to_one_builder(built):
    """Two builders (two ubatches) of one group keep separate buffers, sized
    once from max_num_batched_tokens."""
    owners, owner_config = built
    owner = owners.conv[0, 0]
    first = _builder(owner, owner_config, _vllm_config(8))
    second = _builder(owner, owner_config, _vllm_config(8))
    assert first.token_index.size == 8
    md = first.build(0, _common(CONV_TABLE, CONV_SLOTS))
    ptr = md.seq_idx.data_ptr()
    shifted = [[row[0] + 1] + row[1:] for row in CONV_TABLE]
    other = second.build(0, _common(shifted, CONV_SLOTS[:1] * 7))
    assert other.seq_idx.data_ptr() != ptr
    assert other.query_start.data_ptr() != md.query_start.data_ptr()
    _check_tokens(md, CONV_TABLE, CONV_SLOTS, CONV_BLOCK)
    assert first.build(0, _common(CONV_TABLE, CONV_SLOTS)).seq_idx.data_ptr() == ptr
    small = _builder(owner, owner_config, _vllm_config(4))
    with pytest.raises(ValueError, match="max_num_batched_tokens 4"):
        small.build(0, _common(CONV_TABLE, CONV_SLOTS))


def test_backends_declare_the_lbnhc_layout(built):
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
    from vllm.v1.kv_cache_layout import KVCacheLayout

    owners, _ = built
    backends = [
        owners.attn[0, 0].get_attn_backend(),
        owners.conv[0, 0].get_attn_backend(),
    ]
    assert get_supported_kv_cache_layouts(backends) == [KVCacheLayout.LBNHC]
    assert KVCacheLayout.LBNHC.is_block_compact


def test_slot_mapping_shorter_than_the_batch_is_rejected(built):
    owners, owner_config = built
    builder = _builder(owners.conv[0, 0], owner_config)
    with pytest.raises(ValueError, match="slot mapping"):
        builder.build(0, _common(CONV_TABLE, CONV_SLOTS[:5]))


def test_update_block_table_changes_the_table_and_slots_only(built):
    owners, owner_config = built
    builder = _builder(owners.attn[0, 0], owner_config)
    md = builder.build(0, _common(ATTN_TABLE, ATTN_SLOTS))
    table = torch.tensor(ATTN_TABLE, dtype=torch.int32) + 50
    slots = torch.tensor(ATTN_SLOTS, dtype=torch.int64) + 50 * BLOCK
    other = builder.update_block_table(md, table, slots)
    assert other is not md and type(other) is type(md)
    assert other.block_table is table
    assert other.slot_mapping.tolist() == slots.tolist()
    assert md.block_table.tolist() == ATTN_TABLE
    assert md.slot_mapping.tolist() == ATTN_SLOTS
    for name in ("seq_idx", "query_start", "positions", "seq_lens"):
        assert getattr(other, name) is getattr(md, name)
    assert other.num_decodes == 2 and other.sliding_window == 1024


def test_per_pass_metadata_keys_resolve_to_distinct_caches(built):
    """As the runner does it: one builder per KV group, one metadata object
    per owner name, one bound cache per owner."""
    from vllm.forward_context import override_forward_context
    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

    from vllm_sm70.models.inkling2_sm70.backend import (
        Inkling2AttentionMetadata,
        Inkling2SconvMetadata,
        cache_metadata,
    )

    owners, owner_config = built
    vllm_config = _vllm_config()
    named = owners.named_owners()
    groups = get_kv_cache_groups(vllm_config, owners.kv_cache_specs(owner_config))
    assert len(groups) == 12

    attn_metadata = {}
    group_of = {}
    for gid, group in enumerate(groups):
        first = named[group.layer_names[0]]
        is_conv = ".conv_state" in first.prefix
        builder = first.get_attn_backend().get_builder_cls()(
            group.kv_cache_spec, group.layer_names, vllm_config, torch.device("cpu")
        )
        table = torch.tensor(CONV_TABLE if is_conv else ATTN_TABLE) + 1000 * gid
        block = CONV_BLOCK if is_conv else BLOCK
        slots = torch.tensor(CONV_SLOTS if is_conv else ATTN_SLOTS) + 1000 * gid * block
        md = builder.build(0, _common(table.tolist(), slots.tolist()))
        for name in group.layer_names:
            attn_metadata[name] = md
            group_of[name] = gid

    for owner in named.values():
        owner.kv_cache = torch.zeros(1)

    visits = [(6, 0), (6, 1), (6, 2), (17, 2), (5, 0)]
    with override_forward_context(SimpleNamespace(attn_metadata=attn_metadata)):
        for kind, md_cls in (("attn", Inkling2AttentionMetadata), ("conv", None)):
            chosen = [getattr(owners, kind)[visit] for visit in visits]
            assert len({owner.prefix for owner in chosen}) == len(visits)
            assert len({id(owner.kv_cache) for owner in chosen}) == len(visits)
            for owner in chosen:
                md = cache_metadata(owner)
                assert md is attn_metadata[owner.prefix]
                if md_cls is None:
                    assert type(md) is Inkling2SconvMetadata
                    assert md.block_size == CONV_BLOCK
                else:
                    assert type(md) is md_cls
                    assert md.block_size == BLOCK
                    assert md.sliding_window == owner.sliding_window
                assert int(md.block_table[0, 0]) // 1000 == group_of[owner.prefix]
        # An attention key never resolves to conv metadata, nor the reverse.
        for visit in visits:
            assert cache_metadata(owners.attn[visit]) is not cache_metadata(
                owners.conv[visit]
            )
        # Every owner has a key of its own.
        assert set(attn_metadata) == set(named)
        owners.attn[6, 1].kv_cache = torch.tensor([])
        assert cache_metadata(owners.attn[6, 1]) is None
        assert cache_metadata(owners.attn[6, 2]) is not None
    with override_forward_context(SimpleNamespace(attn_metadata=None)):
        assert cache_metadata(owners.attn[6, 2]) is None
