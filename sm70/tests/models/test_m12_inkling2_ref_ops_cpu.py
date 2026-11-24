# SPDX-License-Identifier: Apache-2.0
"""Tests for the Inkling 2 torch reference ops (ops/torch_ref.py).

CPU fp32, without the Hugging Face Inkling 2 classes: every op recorded in
fixtures/inkling2_tiny/ops.safetensors is recomputed from its recorded inputs
and the tiny checkpoint's weights and compared with the recorded outputs.
"""

import json
import math
import pathlib
import sys

import pytest

pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch
import torch.nn.functional as F

from vllm_sm70.models.inkling2_sm70.ops import torch_ref as ref

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from inkling2_ref import Inkling2Weights

TINY = pathlib.Path(__file__).resolve().parent / "fixtures" / "inkling2_tiny"
ATOL = 1e-5

TEXT = json.loads((TINY / "config.json").read_text())["text_config"]
EPS = TEXT["rms_norm_eps"]
QK_CLIP, REL_CLIP = TEXT["qk_l2_clip_threshold"], TEXT["rel_l2_clip_threshold"]
LOG_SCALING = (TEXT["log_scaling_alpha"], TEXT["log_scaling_n_floor"])
CAPS = (TEXT["mlp_gate_softcap"], TEXT["mlp_up_softcap"])
KV_HEADS = {False: TEXT["swa_num_key_value_heads"], True: TEXT["num_key_value_heads"]}

INDEX = json.loads((TINY / "manifest.json").read_text())["ops"]
OPS = safetensors_torch.load_file(TINY / "ops.safetensors")
NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
CHECKED: set[str] = set()


def arg(op: str, name: str):
    value = INDEX[op]["inputs"][name]
    return OPS[value] if isinstance(value, str) else value[0]


def close(op: str, *got: torch.Tensor) -> None:
    """Assert `got` equals the recorded tensor outputs of `op`."""
    want = [OPS[o] for o in INDEX[op]["outputs"] if isinstance(o, str)]
    assert len(got) == len(want), op
    for i, (g, w) in enumerate(zip(got, want)):
        assert g.dtype == w.dtype and g.shape == w.shape, (op, i)
        if w.is_floating_point():
            assert (g - w).abs().max().item() <= ATOL, (op, i)
        else:
            assert torch.equal(g, w), (op, i)
    CHECKED.add(op)


W = Inkling2Weights(NATIVE)
w = W
expert_weights = W.experts
attention_weights = W.attention
message_queries = W.message_queries


def test_rms_norm():
    op = "rmsnorm.0"
    close(
        op, ref.rms_norm(arg(op, "hidden_states"), w("layers.1.attn_norm.weight"), EPS)
    )
    op = "rmsnorm_unweighted.0"
    close(op, ref.rms_norm(arg(op, "hidden_states"), eps=EPS))
    op = "embed.0"
    embedded = F.embedding(arg(op, "input_ids"), w("embed.weight"))
    close(op, ref.rms_norm(embedded, w("embed_norm.weight"), EPS))


def test_l2_clip():
    op = "l2_clip.0"
    close(op, ref.l2_clip(arg(op, "hidden_states"), QK_CLIP))
    for op, layer, name, threshold in (
        ("scaled_l2_clip.0", 1, "q", QK_CLIP),
        ("rel_l2_clip_sliding.0", 1, "rel", REL_CLIP),
    ):
        scale = w(f"layers.{layer}.attn.{name}_l2_clip.scale")
        close(op, ref.l2_clip(arg(op, "hidden_states"), threshold, scale))


def test_log_scale():
    op = "log_scale.0"
    positions = arg(op, "position_ids")
    close(op, ref.log_scale(arg(op, "hidden_states"), positions, *LOG_SCALING))
    tau = ref.log_scale_tau(positions, *LOG_SCALING)[0]
    floor = TEXT["log_scaling_n_floor"]
    assert (tau[:floor] == 1).all() and (tau[floor:] > 1).all()
    assert tau[-1].item() == pytest.approx(1 + 0.1 * math.log(20 / floor), abs=1e-6)
    for op, name, threshold in (
        ("log_scaled_l2_clip.0", "q", QK_CLIP),
        ("rel_l2_clip_full.0", "rel", REL_CLIP),
    ):
        scale = w(f"layers.5.attn.{name}_l2_clip.scale")
        clipped = ref.l2_clip(arg(op, "hidden_states"), threshold, scale)
        close(op, ref.log_scale(clipped, positions, *LOG_SCALING))


def test_static_conv():
    op = "sconv_static_key.0"
    weight = w("layers.1.attn.k_sconv.weight")
    close(op, ref.static_conv(arg(op, "hidden_states"), weight))
    close(op, ref.static_conv(arg(op, "hidden_states"), weight.squeeze(1)))
    op = "sconv_static_hidden.0"
    close(
        op, ref.static_conv(arg(op, "hidden_states"), w("layers.1.attn_sconv.weight"))
    )


def test_dynamic_conv():
    for op, name in (("sconv_dynamic_key.0", "k"), ("sconv_dynamic_value.0", "v")):
        conv = f"layers.5.attn.{name}_sconv."
        taps = ref.dynamic_conv_taps(
            arg(op, "dynamic_input"), w(conv + "weight_proj"), KV_HEADS[True]
        )
        assert taps.shape == (1, 20, KV_HEADS[True], TEXT["sconv_kernel_size"])
        close(
            op, ref.dynamic_conv(arg(op, "hidden_states"), w(conv + "static_w"), taps)
        )


def test_conv_tap_order_static_oldest_first_dynamic_newest_first():
    channels, taps, length = 4, 4, 9
    x = torch.randn(1, length, channels, generator=torch.Generator().manual_seed(0))
    back = torch.cat([x.new_zeros(1, taps - 1, channels), x], dim=1)[:, :length]
    first_tap = torch.zeros(taps)
    first_tap[0] = 1.0
    # Tap 0 of a static conv reads the token 3 steps back, of a dynamic conv
    # the current token.
    static = ref.static_conv(x, first_tap.expand(channels, taps))
    assert torch.allclose(static, (back + x) / math.sqrt(5), atol=1e-6)
    no_taps = torch.zeros(1, length, 2, taps)
    dynamic = ref.dynamic_conv(x, first_tap[:, None].expand(taps, channels), no_taps)
    assert torch.allclose(dynamic, (x + x) / 3, atol=1e-6)
    assert not torch.allclose(static, dynamic, atol=1e-3)
    # The same weights read newest-first are the static conv's last tap.
    last_tap = first_tap.flip(0)
    assert torch.allclose(
        ref.static_conv(x, last_tap.expand(channels, taps)) * math.sqrt(5),
        dynamic * 3,
        atol=1e-6,
    )
    # Per-token taps are shared by the channels of a head, not across heads.
    one_head = no_taps.clone()
    one_head[0, :, 0, 1] = 1.0
    out = ref.dynamic_conv(x, torch.zeros(taps, channels), one_head) * 3 - x
    previous = torch.cat([x.new_zeros(1, 1, channels), x], dim=1)[:, :length]
    assert torch.allclose(out[..., :2], previous[..., :2], atol=1e-6)
    assert torch.allclose(out[..., 2:], torch.zeros(()), atol=1e-6)


def test_conv_history_and_valid_mask():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 12, 8, generator=g)
    valid = torch.rand(2, 12, generator=g) > 0.3
    static_weight = torch.randn(8, 4, generator=g)
    dynamic_weight = torch.randn(4, 8, generator=g)
    taps = torch.randn(2, 12, 2, 4, generator=g)
    masked = x * valid[..., None]
    for whole, tail in (
        (
            ref.static_conv(x, static_weight, valid),
            ref.static_conv(x[:, 7:], static_weight, valid[:, 7:], masked[:, 4:7]),
        ),
        (
            ref.dynamic_conv(x, dynamic_weight, taps, valid),
            ref.dynamic_conv(
                x[:, 7:], dynamic_weight, taps[:, 7:], valid[:, 7:], masked[:, 4:7]
            ),
        ),
    ):
        assert torch.allclose(whole[:, 7:], tail, atol=1e-6)
    # A masked token is no conv input but keeps its residual.
    weight = torch.zeros(8, 4)
    weight[:, 3] = 1.0
    out = ref.static_conv(x, weight, valid) * math.sqrt(5)
    assert torch.allclose(out, x + masked, atol=1e-6)


def test_conv_and_expert_sum_round_to_the_input_dtype():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 12, 8, generator=g).half()
    weight = torch.randn(8, 4, generator=g).half()
    out = ref.static_conv(x, weight)
    assert out.dtype == torch.float16
    assert torch.equal(out, ref.static_conv(x.float(), weight.float()).half())
    taps = torch.randn(1, 12, 2, 4, generator=g)
    out = ref.dynamic_conv(x, weight.T, taps)
    assert out.dtype == torch.float16
    assert torch.equal(out, ref.dynamic_conv(x.float(), weight.T.float(), taps).half())

    op = "experts_nvfp4.0"
    gate_up, down = (t.half() for t in expert_weights(1))
    hidden = arg(op, "hidden_states").half()
    index, weights = arg(op, "top_k_index"), arg(op, "top_k_weights")
    out = ref.experts(hidden, index, weights, gate_up, down, *CAPS)
    assert out.dtype == torch.float16
    # One rounding, after the fp32 sum of the fp32-weighted expert outputs.
    terms = torch.zeros(*index.shape, hidden.shape[-1])
    for token in range(index.shape[0]):
        for slot in range(index.shape[1]):
            e = index[token, slot]
            gate, up = F.linear(hidden[token], gate_up[e]).chunk(2)
            y = F.linear(ref.softcapped_swiglu(gate, up, *CAPS), down[e])
            assert y.dtype == torch.float16
            terms[token, slot] = y.float() * weights[token, slot]
    assert torch.allclose(out.float(), terms.sum(1).half().float(), atol=2e-3)
    assert (out.float() - OPS[INDEX[op]["outputs"][0]]).abs().max() < 2e-2


def test_attn_res():
    for op, query in (
        ("attn_res_one_query.0", w("attn_res.attn_queries.0")),
        ("attn_res_three_queries.0", w("attn_res.attn_queries.4")),
        ("attn_res_mlp.1", w("attn_res.mlp_queries.12")),
        ("attn_res_final.0", w("attn_res.output_query")),
    ):
        out = ref.attn_res(
            arg(op, "block_states"), arg(op, "partial_block_states"), query
        )
        assert out.shape[-2] == query.numel() // 64
        close(op, *out.unbind(-2))
    op = "attn_res_message.17"
    query = message_queries()[arg(op, "loop_layer_idx")]
    close(
        op,
        ref.attn_res(arg(op, "block_states"), arg(op, "partial_block_states"), query),
    )


def test_relative_position_bias():
    for op, layer, log_scaling in (
        ("rel_bias_sliding.0", 1, None),
        ("rel_bias_full.0", 5, LOG_SCALING),
    ):
        a = f"layers.{layer}.attn."
        distance = arg(op, "relative_distance")
        states = ref.query_projection(
            arg(op, "hidden_states"),
            w(a + "wr_du.weight"),
            w(a + "rel_l2_clip.scale"),
            REL_CLIP,
            arg(op, "position_ids"),
            log_scaling,
        )
        bias = ref.relative_position_bias(
            states, w(a + "rel_logits_proj.proj"), distance
        )
        close(op, bias)
        outside = (distance < 0) | (distance >= 8)
        assert (bias[..., outside] == 0).all() and (bias[..., ~outside] != 0).any()


def test_projections():
    op = "query_proj_full.0"
    a = "layers.5.attn."
    query = ref.query_projection(
        arg(op, "hidden_states"),
        w(a + "wq_du.weight"),
        w(a + "q_l2_clip.scale"),
        QK_CLIP,
        arg(op, "position_ids"),
        LOG_SCALING,
    )
    close(op, query.transpose(1, 2))
    op = "key_proj_sliding.0"
    a = "layers.1.attn."
    close(
        op,
        ref.key_value_projection(
            arg(op, "hidden_states"),
            w(a + "wk_dv.weight"),
            w(a + "k_sconv.weight"),
            KV_HEADS[False],
            clip_threshold=QK_CLIP,
        ),
    )
    op = "value_proj_full.0"
    a = "layers.5.attn."
    close(
        op,
        ref.key_value_projection(
            arg(op, "hidden_states"),
            w(a + "wv_dv.weight"),
            w(a + "v_sconv.static_w"),
            KV_HEADS[True],
            w(a + "v_sconv.weight_proj"),
            eps=EPS,
        ),
    )


def test_gated_output():
    op = "gated_output_proj.0"
    a = "layers.1.attn."
    close(
        op,
        ref.gated_output(
            arg(op, "attn_output"),
            arg(op, "hidden_states"),
            w(a + "gate_attn.weight"),
            w(a + "wo_ud.weight"),
        ),
    )


@pytest.mark.parametrize(
    "op,layer",
    [
        ("attn_sliding.0", 1),
        ("attn_full.0", 5),
        ("attn_sliding_loop.1", 6),
        ("attn_full_loop.2", 11),
    ],
)
def test_attention(op, layer):
    full = layer % 6 == 5
    hidden = arg(op, "hidden_states")
    inputs = INDEX[op]["inputs"]
    allowed = arg(op, "attention_mask")
    assert allowed.dtype == torch.bool
    out, weights = ref.attention(
        hidden,
        arg(op, "key_hidden_states") if "key_hidden_states" in inputs else hidden,
        arg(op, "value_hidden_states") if "value_hidden_states" in inputs else hidden,
        attention_weights(layer),
        num_kv_heads=KV_HEADS[full],
        allowed=allowed,
        relative_distance=arg(op, "relative_distance"),
        position_ids=arg(op, "position_ids"),
        qk_clip=QK_CLIP,
        rel_clip=REL_CLIP,
        eps=EPS,
        log_scaling=LOG_SCALING if full else None,
    )
    recorded = [o for o in INDEX[op]["outputs"] if isinstance(o, str)]
    close(op, *(out, weights)[: len(recorded)])
    # The sink takes probability mass and masked keys get none.
    assert (weights.sum(-1) < 1).all()
    assert (weights.masked_select(~allowed) == 0).all()


def test_dense_attention_sink_and_mask():
    g = torch.Generator().manual_seed(3)
    query = torch.randn(1, 4, 5, 4, generator=g)
    key, value = torch.randn(2, 1, 2, 5, 4, generator=g).unbind(0)
    allowed = torch.ones(5, 5).tril().bool()
    sinks = torch.randn(1, 4, 5, generator=g)
    out, weights = ref.dense_attention(query, key, value, allowed, 0.5, sinks=sinks)
    assert out.shape == (1, 5, 16) and weights.shape == (1, 4, 5, 5)
    # Heads 0-1 read key/value head 0, heads 2-3 head 1.
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key.repeat_interleave(2, 1)) * 0.5
    scores = scores.masked_fill(~allowed, float("-inf"))
    expect = torch.cat([scores, sinks[..., None]], -1).softmax(-1)[..., :-1]
    assert torch.allclose(weights, expect, atol=1e-6)
    # A very negative sink leaves a plain masked softmax.
    _, plain = ref.dense_attention(query, key, value, allowed, 0.5)
    _, no_sink = ref.dense_attention(
        query, key, value, allowed, 0.5, sinks=torch.full_like(sinks, -1e9)
    )
    assert torch.allclose(plain, no_sink, atol=1e-6)
    assert torch.allclose(plain.sum(-1), torch.ones(()), atol=1e-6)


def test_router():
    op = "router.0"
    logits, weights, indices = ref.router(
        arg(op, "hidden_states"),
        w("layers.1.mlp.gate.weight"),
        w("layers.1.mlp.gate.bias"),
        TEXT["num_experts_per_tok"],
        TEXT["route_scale"],
    )
    close(op, logits, weights, indices)
    assert torch.allclose(
        weights.sum(-1), torch.full((20,), TEXT["route_scale"]), atol=1e-5
    )
    # The bias picks the experts but is not part of the weights.
    by_score = torch.sigmoid(logits).topk(2, dim=-1).indices
    assert not torch.equal(by_score.sort(-1).values, indices.sort(-1).values)


def test_softcapped_swiglu():
    op = "swiglu.0"
    close(op, ref.softcapped_swiglu(arg(op, "gate"), arg(op, "up"), *CAPS))
    # The sigmoid reads the raw gate; the caps bound the other two factors.
    gate, up = torch.tensor([40.0, -40.0]), torch.tensor([1e3, 1e3])
    out = ref.softcapped_swiglu(gate, up, *CAPS)
    assert out[0].item() == pytest.approx(4.0 * 25.0, rel=1e-4)
    assert out[1].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("op,layer", [("experts_nvfp4.0", 1), ("experts_mxfp8.0", 23)])
def test_experts(op, layer):
    gate_up, down = expert_weights(layer)
    close(
        op,
        ref.experts(
            arg(op, "hidden_states"),
            arg(op, "top_k_index"),
            arg(op, "top_k_weights"),
            gate_up,
            down,
            *CAPS,
        ),
    )


def test_message_gate():
    for op, layer, name in (
        ("message_gate_query.0", 6, "attn"),
        ("message_gate_key.1", 11, "k"),
    ):
        c = f"layers.{layer}.controllers.{name}."
        out = ref.message_gate(
            arg(op, "hidden_states"),
            arg(op, "message"),
            w(c + "down_weight"),
            w(c + "forget_weight"),
            w(c + "gain_weight"),
            EPS,
        )
        close(op, out)
        swapped = ref.message_gate(
            arg(op, "hidden_states"),
            arg(op, "message"),
            w(c + "down_weight"),
            w(c + "gain_weight"),
            w(c + "forget_weight"),
            EPS,
        )
        assert (swapped - out).abs().max() > 1e-3


def test_logit_cap():
    op = "logit_softcap.0"
    close(op, ref.logit_cap(arg(op, "hidden_states"), TEXT["final_logit_softcapping"]))
    out = ref.logit_cap(torch.tensor([1e4, -1e4], dtype=torch.float16))
    assert out.dtype == torch.float32 and out.tolist() == [30.0, -30.0]


def test_every_recorded_op_is_checked(request):
    here = [i for i in request.session.items if i.module is request.module]
    if len(here) < sum(1 for i in request.module.__dict__ if i.startswith("test_")):
        pytest.skip("needs the whole module to have run")
    assert CHECKED == set(INDEX)
