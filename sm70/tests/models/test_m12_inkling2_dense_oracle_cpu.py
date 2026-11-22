# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 dense stateless oracle (inkling2_ref.py).

CPU fp32, without the Hugging Face Inkling 2 classes: the oracle runs the tiny
checkpoint (fixtures/inkling2_tiny/) and is compared with what the reference
recorded in cases.safetensors: every visit's state, the router logits, the
logits, a greedy continuation and a text + image prefill.
"""

import json
import pathlib
import sys

import pytest

pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from inkling2_ref import Inkling2Oracle

TINY = pathlib.Path(__file__).resolve().parent / "fixtures" / "inkling2_tiny"
ATOL = 1e-4

CONFIG = json.loads((TINY / "config.json").read_text())
TEXT = CONFIG["text_config"]
MANIFEST = json.loads((TINY / "manifest.json").read_text())
CASES = safetensors_torch.load_file(TINY / "cases.safetensors")


@pytest.fixture(scope="module")
def oracle():
    return Inkling2Oracle.from_checkpoint(TINY)


@pytest.fixture(scope="module")
def text(oracle):
    return oracle.forward(CASES["text.input_ids"])


def max_diff(got: torch.Tensor, want: torch.Tensor) -> float:
    assert got.shape == want.shape and got.dtype == want.dtype
    return (got - want).abs().max().item()


def test_visit_order(oracle):
    assert oracle.visit_layers == MANIFEST["visit_layers"]
    assert len(oracle.visit_layers) == 48
    assert [oracle.is_full(i) for i in range(24)] == [
        i not in TEXT["local_layer_ids"] for i in range(24)
    ]


def test_text_case_is_past_the_window_and_the_log_scaling_floor():
    length = CASES["text.input_ids"].shape[1]
    assert length > TEXT["sliding_window_size"]
    assert length > TEXT["rel_extent"]
    assert length > TEXT["log_scaling_n_floor"]


def test_text_embeddings(text):
    assert max_diff(text.inputs_embeds[0], CASES["text.inputs_embeds"]) <= 1e-5


def test_text_visit_states(text):
    want = CASES["text.visit_states"]
    assert text.visit_states.shape == (48, 1, *want.shape[1:])
    diffs = [max_diff(text.visit_states[v, 0], want[v]) for v in range(48)]
    failed = [(v, d) for v, d in enumerate(diffs) if d > ATOL]
    assert not failed, f"first failing visit {failed[0]}"


def test_text_router_logits(oracle, text):
    got, want = text.router_logits[:, 0], CASES["text.router_logits"]
    assert max_diff(got, want) <= ATOL
    # The same experts as the reference at every visit and token.
    k = TEXT["num_experts_per_tok"]
    for v, layer in enumerate(oracle.visit_layers):
        bias = oracle.weights(f"layers.{layer}.mlp.gate.bias")
        chosen = [
            (torch.sigmoid(x) + bias).topk(k).indices.sort().values
            for x in (got[v], want[v])
        ]
        assert torch.equal(*chosen), v


def test_text_final_state_and_logits(text):
    assert (
        max_diff(text.final_hidden_state[0], CASES["text.final_hidden_state"]) <= ATOL
    )
    assert text.logits.dtype == torch.float32
    assert text.logits.shape[-1] == TEXT["unpadded_vocab_size"]
    assert text.logits.abs().max() < TEXT["final_logit_softcapping"]
    for name in ("logits", "logits_cached_decode", "logits_chunked_prefill"):
        assert max_diff(text.logits[0], CASES[f"text.{name}"]) <= ATOL, name


def test_stateless_prefix_equals_the_whole_sequence(oracle, text):
    ids = CASES["text.input_ids"]
    for length in (1, TEXT["sliding_window_size"], 13):
        prefix = oracle.forward(ids[:, :length])
        assert max_diff(prefix.visit_states, text.visit_states[:, :, :length]) <= 1e-5
        assert max_diff(prefix.logits, text.logits[:, :length]) <= 1e-5


def test_batch_rows_are_independent(oracle, text):
    ids = CASES["text.input_ids"]
    both = oracle.forward(torch.cat([ids, ids.flip(1)]))
    assert max_diff(both.visit_states[:, :1], text.visit_states) <= 1e-5
    assert max_diff(both.logits[:1], text.logits) <= 1e-5


def test_greedy_ids(oracle):
    want = CASES["greedy.output_ids"]
    got = oracle.greedy(CASES["greedy.prompt_ids"], want.shape[1])
    assert torch.equal(got, want)


def test_text_and_image_case(oracle):
    ids = CASES["image.input_ids"]
    image_rows = ids[0] == CONFIG["image_token_id"]
    features = CASES["image.features"]
    assert image_rows.sum().item() == features.shape[0]
    # Image features are written over the normed embeddings, un-normed.
    embeds = oracle.embed(ids)
    embeds[0, image_rows] = features
    assert max_diff(embeds[0], CASES["image.inputs_embeds"]) <= 1e-5
    out = oracle.forward(inputs_embeds=embeds)
    want = CASES["image.visit_states_last_token"]
    diffs = [max_diff(out.visit_states[v, 0, -1], want[v]) for v in range(48)]
    failed = [(v, d) for v, d in enumerate(diffs) if d > ATOL]
    assert not failed, f"first failing visit {failed[0]}"
    assert max_diff(out.logits[0], CASES["image.logits"]) <= ATOL
    # The features matter: the text-only embeddings give other logits.
    assert max_diff(oracle.forward(ids).logits[0], CASES["image.logits"]) > 1e-2


def test_forward_takes_exactly_one_input(oracle):
    ids = CASES["text.input_ids"]
    with pytest.raises(ValueError):
        oracle.forward()
    with pytest.raises(ValueError):
        oracle.forward(ids, inputs_embeds=oracle.embed(ids))


def test_half_precision_keeps_the_reference_dtypes():
    out = Inkling2Oracle.from_checkpoint(TINY, torch.float16).forward(
        CASES["text.input_ids"]
    )
    assert out.visit_states.dtype == torch.float16
    assert out.final_hidden_state.dtype == torch.float16
    assert out.router_logits.dtype == torch.float32
    assert out.logits.dtype == torch.float32
    assert out.visit_states.isfinite().all() and out.logits.isfinite().all()
    # Visit 0 has no router discontinuity upstream of its own experts' inputs.
    assert max_diff(out.router_logits[0, 0], CASES["text.router_logits"][0]) < 2e-2
