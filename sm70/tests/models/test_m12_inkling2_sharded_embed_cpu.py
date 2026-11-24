# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 token embedding split over the TP ranks (weights.py,
loop.py ``embed_input_ids``), on CPU with the tiny checkpoint.

- the weight map: rank r holds rows [r V/2, (r + 1) V/2) of the padded
  embedding; VLLM_SM70_INKLING_SHARDED_EMBED=0 and TP1 hold all rows;
- each rank's lookup holds its own ids' rows and zeros elsewhere; summed by
  the harness's all-reduce, the normed embedding equals TP1's bit for bit,
  in fp32 and fp16, for ids at the shard edges and the padding rows;
- the DSpark draft's anchor embedding (dspark.py ``embed_block``, the
  target's ``embed_input_ids``) through the sharded embedding equals TP1's.

The whole TP2 model against TP1 is tests/models/test_m12_inkling2_model_tp2_cpu.py.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx

TINY = HERE / "fixtures" / "inkling2_tiny"
SWITCH = "VLLM_SM70_INKLING_SHARDED_EMBED"


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


def _config():
    from vllm_sm70.models.inkling2_sm70 import sm70_configs

    raw = json.loads((TINY / "config.json").read_text())
    return sm70_configs.Inkling2LoopSm70MMConfig(**raw).get_text_config()


def _native():
    from safetensors.torch import load_file

    return load_file(TINY / "model.safetensors")


def _body(config, tp_size, tp_rank, dtype):
    """The embedding part of a rank's Inkling2Sm70LoopModel, loaded by the
    rank's weight map."""
    from torch import nn

    from vllm_sm70.models.inkling2_sm70 import weights
    from vllm_sm70.models.inkling2_sm70.loop import Inkling2Sm70LoopModel

    wmap = weights.build_weight_map(
        config, tp_size=tp_size, tp_rank=tp_rank, dtype=dtype
    )
    params = wmap.convert(_native().items()).params
    body = Inkling2Sm70LoopModel.__new__(Inkling2Sm70LoopModel)
    nn.Module.__init__(body)
    body.tp_size, body.tp_rank = tp_size, tp_rank
    body.embed_sharded = wmap.embed_sharded
    body.eps = float(config.rms_norm_eps)
    body.embed_tokens = nn.Module()
    body.embed_tokens.weight = nn.Parameter(
        params["model.embed_tokens.weight"], requires_grad=False
    )
    body.embed_norm = nn.Module()
    body.embed_norm.weight = nn.Parameter(
        params["model.embed_norm.weight"], requires_grad=False
    )
    return body


def _ids(config):
    vocab = config.padded_vocab_size
    half = vocab // 2
    edges = [0, 1, half - 1, half, half + 1, vocab - 2, vocab - 1]
    return torch.tensor(edges + list(range(0, vocab, 7)))


def test_weight_map_splits_the_padded_rows(monkeypatch):
    from vllm_sm70.models.inkling2_sm70 import weights

    config = _config()
    embed = _native()["model.llm.embed.weight"].float()
    vocab = config.padded_vocab_size
    assert embed.shape[0] == vocab and vocab % 2 == 0
    for rank in (0, 1):
        body = _body(config, 2, rank, torch.float32)
        assert body.embed_sharded is True
        want = embed[rank * vocab // 2 : (rank + 1) * vocab // 2]
        assert torch.equal(body.embed_tokens.weight, want)
    assert _body(config, 1, 0, torch.float32).embed_sharded is False
    monkeypatch.setenv(SWITCH, "0")
    assert weights.EMBED_SWITCH == SWITCH and not weights.sharded_embed_enabled()
    for rank in (0, 1):
        body = _body(config, 2, rank, torch.float32)
        assert body.embed_sharded is False
        assert torch.equal(body.embed_tokens.weight, embed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_each_rank_holds_its_rows_and_zeros(monkeypatch, dtype):
    from vllm_sm70.models.inkling2_sm70 import loop

    config = _config()
    ids = _ids(config)
    whole = torch.nn.functional.embedding(
        ids, _native()["model.llm.embed.weight"].to(dtype)
    )
    sent = []
    monkeypatch.setattr(loop, "all_reduce", lambda x, tp: sent.append(x) or x)
    for rank in (0, 1):
        _body(config, 2, rank, dtype).embed_input_ids(ids)
    half = config.padded_vocab_size // 2
    for rank, rows in enumerate(sent):
        mine = (ids >= rank * half) & (ids < (rank + 1) * half)
        assert torch.equal(rows[mine], whole[mine])
        assert bool((rows[~mine] == 0).all())
    assert torch.equal(sent[0] + sent[1], whole)


def _tp2_embed(config, dtype, fn):
    harness = hx.PagedHarness(2, block_size=8, dtype=dtype)
    bodies = [_body(config, 2, rank, dtype) for rank in (0, 1)]
    with torch.no_grad():
        return harness.lockstep(lambda rank: fn(bodies[rank]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tp2_normed_embedding_equals_tp1(dtype):
    config = _config()
    ids = _ids(config)
    with torch.no_grad():
        want = _body(config, 1, 0, dtype).embed_input_ids(ids)
    for got in _tp2_embed(config, dtype, lambda body: body.embed_input_ids(ids)):
        assert got.dtype == dtype and torch.equal(got, want)


def test_dspark_anchor_embedding_through_the_sharded_embedding():
    """``embed_block`` writes the target's normed embedding of each block's
    anchor in row 0 and the mask embedding elsewhere."""
    from vllm_sm70.models.inkling2_sm70.dspark import Inkling2DSparkSm70Decoder

    config = _config()
    dtype = torch.float16
    ids = _ids(config)[:6].view(2, 3)
    mask = torch.randn(config.hidden_size).to(dtype)

    def draft(body):
        decoder = Inkling2DSparkSm70Decoder.__new__(Inkling2DSparkSm70Decoder)
        torch.nn.Module.__init__(decoder)
        decoder.embed, decoder.mask_embed = body.embed_input_ids, mask
        decoder.block, decoder.hidden = 3, config.hidden_size
        return decoder.embed_block(ids)

    with torch.no_grad():
        want = draft(_body(config, 1, 0, dtype))
    for got in _tp2_embed(config, dtype, draft):
        assert torch.equal(got, want)
