# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 merged projections on an SM70 GPU.

The tiny checkpoint (fixtures/inkling2_tiny) in fp16 at TP1 on the GPU,
built with the packs (``VLLM_SM70_INKLING_MERGED_PROJ`` unset) and without
(``=0``). The move to the GPU splits the packs; the post-load hook packs
them again. In each engine scenario, with steps padded as for a FULL decode
graph, every visit, the final state and the logits of the two agree within
the fp16 kernel tolerance: the stacked GEMMs and the merged gate GEMM sum in
another order than the single ones.

    pytest -m gpu sm70/tests/models/test_m12_inkling2_merged_proj_gpu.py
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx

pytestmark = pytest.mark.gpu

SWITCH = "VLLM_SM70_INKLING_MERGED_PROJ"
LEN_A, LEN_B, PREFIX = 23, 19, 8
TOL = 2e-3
SCENARIOS = ("prefill", "decode", "chunk3", "resume", "interleaved")
PAD = {"pad_tokens": 3, "pad_reqs": 1, "metadata": "full"}


def _engine(monkeypatch, merged: bool):
    from test_m12_inkling2_model_cpu import Engine

    if merged:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, "0")
    engine = Engine(torch.float16, device="cuda")
    engine.model.process_weights_after_loading()
    return engine


@pytest.fixture
def pair(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    return _engine(monkeypatch, True), _engine(monkeypatch, False)


@pytest.fixture
def sequences():
    from test_m12_inkling2_model_cpu import _raw_config

    generator = torch.Generator().manual_seed(12)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    a = torch.randint(0, vocab, (LEN_A,), generator=generator)
    b = torch.randint(0, vocab, (LEN_B,), generator=generator)
    b[:PREFIX] = a[:PREFIX]
    return {"a": a, "b": b}


def test_packs_are_on_the_gpu_after_the_hook(pair):
    from vllm_sm70.models.inkling2_sm70 import weights

    merged, plain = pair
    for pack in merged.model.model.weight_map.packs:
        members = [merged.model.get_parameter(name) for name in pack.params]
        view = weights.rows_view(*members)
        assert view is not None and view.is_cuda, pack
    attn = merged.model.model.layers[5].attn
    assert attn.q_scale_fp32.is_cuda and attn.q_scale_fp32.dtype == torch.float32
    assert plain.model.model.layers[5].attn.q_scale_fp32 is None


@pytest.mark.parametrize("name", SCENARIOS)
def test_merged_matches_unmerged_in_fp16(pair, sequences, name):
    merged, plain = pair
    plans = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)
    want = plain.run(plans[name], sequences, **PAD)
    got = merged.run(plans[name], sequences, **PAD)
    assert got.keys() == want.keys() and got
    for req_id in want:
        a, b = got[req_id].float(), want[req_id].float()
        scale = b.abs().amax().clamp(min=1.0)
        assert ((a - b).abs().amax() / scale) <= TOL, f"{name}/{req_id}"
