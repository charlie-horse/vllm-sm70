# SPDX-License-Identifier: Apache-2.0
"""M12 CPU checks: on the V2 runner the Inkling 2 metadata builders build once
per (builder class, spec) and fill the token index once per step
(models/inkling2_sm70/backend.py ``_StepBuilds``); on the V1 runner every
build is its own.

Uses the batch of test_m12_inkling2_metadata_cpu.py. A step is the set of
builds that share one ``query_start_loc_cpu``, ``seq_lens`` and ``positions``
tensor, as in UP v1/worker/gpu/attn_utils.py ``build_attn_metadata``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inkling2_owners import build
from test_m12_inkling2_metadata_cpu import (
    ATTN_SLOTS,
    ATTN_TABLE,
    BLOCK,
    CONV_BLOCK,
    CONV_SLOTS,
    CONV_TABLE,
    _builder,
    _check_tokens,
    _common,
    _stub_stable_libtorch,  # noqa: F401
    _vllm_config,
    pytestmark,  # noqa: F401
)

TP = 2

TOKEN_FIELDS = ("seq_idx", "query_start", "positions")


def _config(v2: bool):
    config = _vllm_config()
    config.use_v2_model_runner = v2
    return config


def _group(cm, table, slots):
    """The common metadata of another KV group of the same step."""
    return replace(
        cm,
        block_table_tensor=torch.tensor(table, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, dtype=torch.int64),
    )


def _shifted(table, slots, block_size, by=50):
    return (
        [[b + by for b in row] for row in table],
        [s + by * block_size for s in slots],
    )


@pytest.fixture
def built():
    return build(BLOCK, TP, torch.float16)


@pytest.fixture
def fills(monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import backend

    calls = []
    fill = backend._TokenIndex.fill

    def counted(self, cm, decode_only):
        calls.append(self)
        return fill(self, cm, decode_only)

    monkeypatch.setattr(backend._TokenIndex, "fill", counted)
    return calls


def test_v2_groups_of_one_step_share_one_build(built, fills):
    owners, owner_config = built
    config = _config(True)
    attn = [_builder(owners.attn[layer, 0], owner_config, config) for layer in (0, 6)]
    conv = [_builder(owners.conv[layer, 0], owner_config, config) for layer in (0, 6)]
    cm_attn = _common(ATTN_TABLE, ATTN_SLOTS)
    cm_conv = _group(cm_attn, CONV_TABLE, CONV_SLOTS)
    table, slots = _shifted(ATTN_TABLE, ATTN_SLOTS, BLOCK)
    ctable, cslots = _shifted(CONV_TABLE, CONV_SLOTS, CONV_BLOCK)

    first = attn[0].build(0, cm_attn)
    second = attn[1].build(0, _group(cm_attn, table, slots))
    conv_first = conv[0].build(0, cm_conv)
    conv_second = conv[1].build(0, _group(cm_attn, ctable, cslots))

    assert len(fills) == 1
    _check_tokens(first, ATTN_TABLE, ATTN_SLOTS, BLOCK)
    _check_tokens(second, table, slots, BLOCK)
    _check_tokens(conv_first, CONV_TABLE, CONV_SLOTS, CONV_BLOCK)
    _check_tokens(conv_second, ctable, cslots, CONV_BLOCK)
    for md in (second, conv_first, conv_second):
        for name in TOKEN_FIELDS:
            assert getattr(md, name) is getattr(first, name)
    assert second.num_decodes == 2 and second.sliding_window == 1024
    assert type(conv_second) is type(conv_first) is not type(first)


def test_v2_a_new_step_or_padding_builds_again(built, fills):
    owners, owner_config = built
    builder = _builder(owners.attn[0, 0], owner_config, _config(True))
    cm = _common(ATTN_TABLE, ATTN_SLOTS)
    first = builder.build(0, cm)
    assert builder.build(0, _common(ATTN_TABLE, ATTN_SLOTS)) is not first
    assert len(fills) == 2
    padded = builder.build(0, _common(ATTN_TABLE, ATTN_SLOTS, pad_tokens=1))
    assert padded.slot_mapping.shape[0] == len(ATTN_SLOTS) + 1
    assert len(fills) == 3


def test_v2_another_spec_builds_its_own_metadata(built, fills):
    owners, owner_config = built
    config = _config(True)
    cm = _common(ATTN_TABLE, ATTN_SLOTS)
    sliding = _builder(owners.attn[0, 0], owner_config, config).build(0, cm)
    full = _builder(owners.attn[5, 0], owner_config, config).build(0, cm)
    assert (sliding.sliding_window, full.sliding_window) == (1024, None)
    assert len(fills) == 1
    assert full.seq_idx is sliding.seq_idx


@pytest.mark.parametrize("config", [_vllm_config, lambda: _config(False)])
def test_v1_builds_every_group(built, fills, config):
    owners, owner_config = built
    attn = [_builder(owners.attn[layer, 0], owner_config, config()) for layer in (0, 6)]
    cm = _common(ATTN_TABLE, ATTN_SLOTS)
    first = attn[0].build(0, cm)
    second = attn[1].build(0, cm)
    assert len(fills) == 2
    assert second is not first
    assert second.seq_idx.data_ptr() != first.seq_idx.data_ptr()
    _check_tokens(second, ATTN_TABLE, ATTN_SLOTS, BLOCK)
