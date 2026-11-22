# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2: no host sync in a decode step.

A one-token decode step of the tiny checkpoint (fixtures/inkling2_tiny) runs
under ``inkling2_sync_guard.no_host_sync``, the CPU stand-in for
``torch.cuda.set_sync_debug_mode("error")``: any op that would make the host
wait for the GPU fails the test, the torch convs included. With
VLLM_SM70_INKLING_DECODE_GRAPH=0 the convs sync again. Also: the one-token
torch expert sum is bit-equal to the grouped one, and the guard catches what
it should.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx
from inkling2_sync_guard import HostSync, allowed, no_host_sync
from test_m12_inkling2_model_cpu import LEN_A, Engine, _have_vllm, _raw_config

pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

PROMPT = 9


@pytest.fixture
def sequences():
    generator = torch.Generator().manual_seed(5)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    return {"a": torch.randint(0, vocab, (LEN_A,), generator=generator)}


def _guard_decode_steps(engine, monkeypatch, guarded):
    """Run the model body of every one-token step under the guard; `guarded`
    counts those steps."""
    body = engine.model.model
    forward = body.forward

    def checked(positions, inputs_embeds):
        if inputs_embeds.shape[0] != 1:
            return forward(positions, inputs_embeds)
        guarded.append(inputs_embeds.dtype)
        with no_host_sync():
            return forward(positions, inputs_embeds)

    monkeypatch.setattr(body, "forward", checked)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_decode_step_has_no_host_sync(monkeypatch, sequences, dtype):
    engine = Engine(dtype, cudagraph_mode="FULL_DECODE_ONLY")
    want = engine.run(hx.decode_plan("a", LEN_A, PROMPT), sequences)["a"]
    guarded = []
    _guard_decode_steps(engine, monkeypatch, guarded)
    got = engine.run(hx.decode_plan("a", LEN_A, PROMPT), sequences)["a"]
    assert guarded == [dtype] * (LEN_A - PROMPT)
    assert torch.equal(got, want)


def test_the_guard_sees_the_convs_sync_without_the_decode_path(monkeypatch, sequences):
    """With VLLM_SM70_INKLING_DECODE_GRAPH=0 a decode step takes the ops of
    mixed batches, whose torch convs sync."""
    monkeypatch.setenv("VLLM_SM70_INKLING_DECODE_GRAPH", "0")
    engine = Engine(cudagraph_mode="FULL_DECODE_ONLY")
    body = engine.model.model
    forward = body.forward

    def checked(positions, inputs_embeds):
        if inputs_embeds.shape[0] != 1:
            return forward(positions, inputs_embeds)
        with no_host_sync():
            return forward(positions, inputs_embeds)

    monkeypatch.setattr(body, "forward", checked)
    with pytest.raises(HostSync):
        engine.run(hx.decode_plan("a", PROMPT + 1, PROMPT), sequences)


def _experts(dtype, layer):
    import json

    from safetensors.torch import load_file

    from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
    from vllm_sm70.models.inkling2_sm70 import moe, quant_config, sm70_configs

    tiny = HERE / "fixtures" / "inkling2_tiny"
    native = load_file(tiny / "model.safetensors")
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((tiny / "config.json").read_text())
    )
    formats = quant_config.expert_formats(
        json.loads((tiny / "hf_quant_config.json").read_text())
    )
    module = moe.Inkling2Sm70Experts(config, dtype=dtype)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SM70_INKLING_NVFP4_TM", "0")
        module.load_expert_layer(
            ew.load_expert_layers(native, config, formats=formats)[layer]
        )
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    module.gate.weight.data.copy_(native[prefix + "weight"])
    module.gate.bias.data.copy_(native[prefix + "bias"])
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("layer", [1, 23])  # NVFP4, MXFP8
def test_one_token_expert_sum_is_bit_equal_to_the_grouped_sum(dtype, layer):
    module = _experts(dtype, layer)
    generator = torch.Generator().manual_seed(layer)
    for trial in range(8):
        x = torch.randn(1, module.hidden_size, generator=generator).to(dtype)
        _, weights, indices = module.route(x, torch.tensor([trial == 7]))
        if trial in (5, 6):  # ids outside 0..E-1 select no expert
            indices = indices.clone()
            indices[0, 0] = -1 if trial == 5 else module.num_experts + 3
        grouped = torch.full_like(x, float("nan"))
        module._grouped_expert_sum(x, indices, weights, grouped)
        token = torch.full_like(x, float("nan"))
        with no_host_sync():
            module.torch_expert_sum(x, indices, weights, token)
        assert torch.equal(token, grouped), trial
    assert bool((token == 0).all())  # the padding row


def test_grouped_expert_sum_reads_the_host_once():
    module = _experts(torch.float32, 1)
    x = torch.randn(5, module.hidden_size, generator=torch.Generator().manual_seed(0))
    _, weights, indices = module.route(x)
    reads = []
    tolist = torch.Tensor.tolist

    def counted(self):
        reads.append(self.shape)
        return tolist(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", counted)
        patch.setattr(
            torch, "bincount", lambda *a, **k: pytest.fail("bincount syncs on CUDA")
        )
        module.torch_expert_sum(x, indices, weights, torch.empty_like(x))
    assert reads == [torch.Size([module.num_experts])]


@pytest.mark.parametrize(
    "op",
    [
        lambda t: t.sum().item(),
        lambda t: bool(t.any()),
        lambda t: int(t[0]),
        lambda t: t.tolist(),
        lambda t: t.cpu(),
        lambda t: t.nonzero(),
        lambda t: torch.where(t > 0),
        lambda t: t.unique(),
        lambda t: torch.bincount(t.long().abs()),
        lambda t: t[t > 0],
        lambda t: t.masked_select(t > 0),
        lambda t: t.repeat_interleave(t.long().abs()),
        lambda t: torch.as_tensor([1, 2]),
    ],
)
def test_the_guard_catches_host_syncs(op):
    t, rows = torch.tensor([1.0, -2.0, 3.0]), torch.tensor([0, 2])
    op(t)
    with no_host_sync(), pytest.raises(HostSync):
        op(t)
    with no_host_sync():
        allowed(op)(t)
        torch.where(t > 0, t, 0.0).masked_fill(t < 0, 1.0)
        t[rows]
        t.repeat_interleave(2)
