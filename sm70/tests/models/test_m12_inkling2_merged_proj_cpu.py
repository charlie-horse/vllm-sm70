# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 merged projections (weights.py packs, layers.py), on CPU.

The weight map's packs, ``rows_view`` and ``pack_parameters``; the packed
projections and message gates against the per-projection ops; the loaded
model's packs and fp32 clip scales; and the registered model class on the
tiny checkpoint (fixtures/inkling2_tiny) with the packs against the same
class built with ``VLLM_SM70_INKLING_MERGED_PROJ=0``, at TP1 and TP2, in the
paged engine scenarios. In fp32 the packed ops agree with the separate ones
to 1e-6, and every visit, the final state and the logits of the model to
5e-6 of their scale. Also: fewer GEMMs per decode step, and the kill switch.
"""

from __future__ import annotations

import collections
import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx

LEN_A, LEN_B, PREFIX = 23, 19, 8
TOL = 1e-6
# One stacked GEMM vs separate ones: the summation order differs with the CPU's
# vector width (AVX2 vs AVX-512), about 1e-6 of the output's scale.
GEMM_TOL = 1e-5
# Over 48 visits: the stacked GEMMs' other summation order (about 1e-7 per op)
# grows to about 1.1e-6 of the stream's scale on the tiny checkpoint.
MODEL_TOL = 5e-6
SWITCH = "VLLM_SM70_INKLING_MERGED_PROJ"
SCENARIOS = ("prefill", "decode", "chunk3", "resume", "interleaved")
PADDING = {
    "exact": {},
    "full": {"pad_tokens": 3, "pad_reqs": 1, "metadata": "full"},
}


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


def _weights_module():
    from vllm_sm70.models.inkling2_sm70 import weights

    return weights


def _tiny_text_config():
    from test_m12_inkling2_model_cpu import _raw_config

    from vllm_sm70.models.inkling2_sm70.sm70_configs import Inkling2LoopSm70MMConfig

    return Inkling2LoopSm70MMConfig(**_raw_config()).get_text_config()


def _engines(tp_size: int, merged: bool):
    """TP1: the model-test engine; TP2: the TP2-test engine."""
    with pytest.MonkeyPatch.context() as patch:
        if merged:
            patch.delenv(SWITCH, raising=False)
        else:
            patch.setenv(SWITCH, "0")
        if tp_size == 1:
            from test_m12_inkling2_model_cpu import Engine

            return Engine()
        from test_m12_inkling2_model_tp2_cpu import Engine

        return Engine(tp_size)


def _models(engine):
    return getattr(engine, "models", None) or [engine.model]


@pytest.fixture(scope="module")
def pairs() -> dict[int, tuple]:
    return {tp: (_engines(tp, True), _engines(tp, False)) for tp in (1, 2)}


@pytest.fixture(scope="module")
def sequences() -> dict[str, torch.Tensor]:
    from test_m12_inkling2_model_cpu import _raw_config

    generator = torch.Generator().manual_seed(12)
    vocab = _raw_config()["text_config"]["unpadded_vocab_size"]
    a = torch.randint(0, vocab, (LEN_A,), generator=generator)
    b = torch.randint(0, vocab, (LEN_B,), generator=generator)
    b[:PREFIX] = a[:PREFIX]
    return {"a": a, "b": b}


# --- packs, rows_view, pack_parameters ----------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2])
def test_weight_map_packs(tp_size):
    weights = _weights_module()
    text = _tiny_text_config()
    weight_map = weights.build_weight_map(
        text, tp_size=tp_size, tp_rank=tp_size - 1, dtype=torch.float16
    )
    packs = {pack.params[0]: pack.params for pack in weight_map.packs}
    local = set(text.local_layer_ids)
    core = set(text.loop_core_layers)
    for layer in range(text.num_hidden_layers):
        p = f"model.layers.{layer}."
        attn = p + "attn.qkvr.weight"
        if layer in local:
            assert attn not in packs
        else:
            assert packs[attn] == (
                attn,
                p + "attn.k_sconv.weight_proj",
                p + "attn.v_sconv.weight_proj",
            )
            # Column-parallel: the rank's own rows of each member.
            rows = [weight_map.params[n].shape[0] for n in packs[attn]]
            full = weights.qkvr_sizes(text, False)
            assert rows[0] == sum(full) // tp_size
            assert rows[1] == rows[2] == text.num_key_value_heads * 4 // tp_size
            assert {weight_map.params[n].dtype for n in packs[attn]} == {torch.float16}
        gates = [f"{p}controllers.{s}." for s, _ in weights.MESSAGE_STREAMS]
        down = gates[0] + "down_weight"
        gain = gates[0] + "gain_weight"
        if layer not in core:
            assert down not in packs and gain not in packs
            continue
        assert packs[down] == tuple(g + "down_weight" for g in gates)
        assert packs[gain] == tuple(
            g + w for g in gates for w in ("gain_weight", "forget_weight")
        )
        # The reference keeps the gain and forget gates in fp32, down in fp16.
        assert {weight_map.params[n].dtype for n in packs[gain]} == {torch.float32}
        assert {weight_map.params[n].dtype for n in packs[down]} == {torch.float16}
    assert len(weight_map.packs) == len(set(range(24)) - local) + 2 * len(core)


def test_rows_view():
    weights = _weights_module()
    buffer = torch.arange(24.0).view(6, 4)
    a, b, c = buffer[:1], buffer[1:4], buffer[4:]
    assert weights.rows_view(a, b, c).data_ptr() == buffer.data_ptr()
    assert torch.equal(weights.rows_view(a, b, c), buffer)
    assert torch.equal(weights.rows_view(b, c), buffer[1:])
    assert weights.rows_view(a) is a
    assert weights.rows_view(a, c) is None  # a gap
    assert weights.rows_view(b, a) is None  # out of order
    assert weights.rows_view(a, b.clone()) is None  # another storage
    assert weights.rows_view(a, buffer[1:4, :2]) is None  # not contiguous
    assert weights.rows_view(buffer.view(-1)[:4].view(1, 4).half(), b) is None


def test_pack_parameters_copies_and_is_idempotent():
    weights = _weights_module()
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.randn(2, 3), requires_grad=False)
    model.b = torch.nn.Parameter(torch.randn(4, 3), requires_grad=False)
    want = (model.a.clone(), model.b.clone())
    ids = (id(model.a), id(model.b))
    pack = weights.Pack(("a", "b"))
    assert weights.pack_parameters(model, [pack]) == [pack]
    assert (id(model.a), id(model.b)) == ids
    assert torch.equal(model.a, want[0]) and torch.equal(model.b, want[1])
    view = weights.rows_view(model.a, model.b)
    assert view is not None and torch.equal(view, torch.cat(want))
    assert weights.pack_parameters(model, [pack]) == []
    assert weights.rows_view(model.a, model.b).data_ptr() == view.data_ptr()


# --- the packed ops against the per-projection ops -----------------------------


@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("taps", [0, 6])
@pytest.mark.parametrize("inputs", [1, 3])
def test_project_packed_matches_project_qkvr(tokens, taps, inputs):
    import torch.nn.functional as F

    from vllm_sm70.models.inkling2_sm70 import prologue

    gen = torch.Generator().manual_seed(tokens * 10 + taps + inputs)
    sizes = (24, 8, 24, 6, 8, 8)
    hidden = 32
    weight = torch.randn(sum(sizes) + 2 * taps, hidden, generator=gen)
    xs = [torch.randn(tokens, hidden, generator=gen) for _ in range(inputs)]
    x_q, x_k, x_v = (*xs, None, None)[:3] if inputs == 1 else xs
    got = prologue.project_packed(x_q, x_k, x_v, weight, sizes, taps)
    qkvr = weight[: sum(sizes)]
    want = list(prologue.project_qkvr(x_q, x_k, x_v, qkvr, sizes))
    if taps:
        k_proj, v_proj = weight[sum(sizes) :].split(taps)
        want += [
            F.linear(x_q if x_k is None else x_k, k_proj),
            F.linear(x_q if x_v is None else x_v, v_proj),
        ]
    else:
        want += [None, None]
        assert got[6] is None and got[7] is None
    for i, (g, w) in enumerate(zip(got, want)):
        if w is None:
            continue
        assert g.shape == w.shape, i
        torch.testing.assert_close(g, w, atol=GEMM_TOL, rtol=GEMM_TOL)


def test_project_packed_reads_adjacent_inputs_in_place():
    from vllm_sm70.models.inkling2_sm70 import prologue

    stacked = torch.randn(3, 2, 16)
    x_q, x_k, x_v = stacked.unbind(0)
    assert prologue._stacked(x_q, x_k, x_v).data_ptr() == stacked.data_ptr()
    apart = [x.clone() for x in (x_q, x_k, x_v)]
    assert torch.equal(prologue._stacked(*apart), stacked.flatten(0, 1))


def test_query_states_out_matches_the_rounded_result():
    from vllm_sm70.models.inkling2_sm70 import prologue
    from vllm_sm70.models.inkling2_sm70.ops import torch_ref

    gen = torch.Generator().manual_seed(3)
    x = (torch.randn(5, 4 * 8, generator=gen) * 3).half()
    scale = torch.randn(4, 8, generator=gen).half()
    tau = 1 + torch.rand(5, generator=gen)
    for t in (None, tau):
        # The former formula: cast, then scale, clip, round, tau, round.
        y = x.unflatten(-1, (4, 8)).float() * scale.float()
        norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        want = (y * (2.0 / (norm + 1e-6)).clamp(max=1.0)).half()
        if t is not None:
            want = (want.float() * t[:, None, None]).half()
        for s in (scale, scale.float()):
            out = torch.empty(5, 4, 8, dtype=torch.float16)
            assert prologue.query_states(x, s, 2.0, t, out=out) is out
            assert torch.equal(out, want)
            assert torch.equal(prologue.query_states(x, s, 2.0, t), want)
    clipped = torch_ref.l2_clip(x.unflatten(-1, (4, 8)), 2.0)
    assert clipped.dtype == torch.float16


def test_merged_message_gate_matches_message_gate():
    from vllm_sm70.models.inkling2_sm70.layers import Inkling2Sm70MessageGate
    from vllm_sm70.models.inkling2_sm70.ops.torch_ref import message_gate

    weights = _weights_module()
    hidden, rank = 32, 4
    specs = {
        "g.down_weight": weights.ParamSpec((rank, 2 * hidden), torch.float32),
        "g.gain_weight": weights.ParamSpec((hidden, rank), torch.float32),
        "g.forget_weight": weights.ParamSpec((hidden, rank), torch.float32),
    }
    config = type("C", (), {"rms_norm_eps": 1e-6})()
    gate = Inkling2Sm70MessageGate(config, specs, "g", merged=True)
    gen = torch.Generator().manual_seed(5)
    for p in gate.parameters():
        p.copy_(torch.randn(p.shape, generator=gen))
    state, message = (torch.randn(3, hidden, generator=gen) for _ in range(2))
    want = message_gate(
        state, message, gate.down_weight, gate.forget_weight, gate.gain_weight
    )
    assert weights.rows_view(gate.gain_weight, gate.forget_weight) is None
    torch.testing.assert_close(gate(state, message), want, atol=0, rtol=0)
    weights.pack_parameters(gate, [weights.Pack(("gain_weight", "forget_weight"))])
    assert weights.rows_view(gate.gain_weight, gate.forget_weight) is not None
    torch.testing.assert_close(gate(state, message), want, atol=TOL, rtol=TOL)


# --- the loaded model ----------------------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2])
def test_loaded_packs_alias_one_buffer_and_hold_the_same_values(pairs, tp_size):
    weights = _weights_module()
    merged, plain = pairs[tp_size]
    for model, reference in zip(_models(merged), _models(plain)):
        assert model.model.merged and not reference.model.merged
        params = dict(model.named_parameters())
        assert params.keys() == dict(reference.named_parameters()).keys()
        for pack in model.model.weight_map.packs:
            members = [params[name] for name in pack.params]
            assert weights.rows_view(*members) is not None, pack
            assert (
                weights.rows_view(*(reference.get_parameter(n) for n in pack.params))
                is None
            )
        for name, param in reference.named_parameters():
            assert torch.equal(params[name], param), name


def test_post_load_hook_repacks_and_keeps_fp32_scales(pairs):
    weights = _weights_module()
    model = pairs[1][0].model
    pack = model.model.weight_map.packs[0]
    members = [model.get_parameter(name) for name in pack.params]
    values = [p.clone() for p in members]
    members[1].data = members[1].data.clone()  # as a move to another device
    assert weights.rows_view(*members) is None
    model.process_weights_after_loading()
    assert weights.rows_view(*members) is not None
    assert all(torch.equal(p, v) for p, v in zip(members, values))
    attn = model.model.layers[5].attn
    kept = attn.q_scale_fp32
    assert kept.dtype == torch.float32 and torch.equal(kept, attn.q_scale.float())
    model.process_weights_after_loading()
    assert attn.q_scale_fp32 is kept
    assert attn.r_scale_fp32.dtype == torch.float32
    assert pairs[1][1].model.model.layers[5].attn.q_scale_fp32 is None


def _assert_rows(engine, got, want, where):
    """Each visit's state, the final state and the logits agree to MODEL_TOL
    of their own largest value (at least 1)."""
    assert got.shape == want.shape, where
    hidden = engine.hidden
    states = hidden * (engine.num_visits + 1)
    parts = [hidden] * (engine.num_visits + 1) + [want.shape[1] - states]
    for i, (g, w) in enumerate(zip(got.split(parts, 1), want.split(parts, 1))):
        bound = MODEL_TOL * max(1.0, float(w.abs().max()))
        assert float((g - w).abs().max()) <= bound, (where, i)


CASES = [(1, name, "exact") for name in SCENARIOS]
CASES += [(1, name, "full") for name in ("decode", "interleaved")]
CASES += [(2, name, "exact") for name in ("decode", "interleaved")]


@pytest.mark.parametrize(("tp_size", "name", "padding"), CASES)
def test_merged_matches_unmerged(pairs, sequences, tp_size, name, padding):
    merged, plain = pairs[tp_size]
    plans = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)
    want = plain.run(plans[name], sequences, **PADDING[padding])
    got = merged.run(plans[name], sequences, **PADDING[padding])
    assert got.keys() == want.keys() and got
    for req_id in want:
        _assert_rows(merged, got[req_id], want[req_id], f"{name}/{req_id}")


class _Gemms(collections.Counter):
    """Counts the GEMM ops dispatched while active."""

    NAMES = ("mm", "addmm", "bmm", "baddbmm", "mv", "addmv", "dot")

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        counter = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func.overloadpacket.__name__ in _Gemms.NAMES:
                    counter[func.overloadpacket.__name__] += 1
                return func(*args, **(kwargs or {}))

        self.mode = Mode()
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self.mode.__exit__(*exc)


def test_a_decode_step_runs_fewer_gemms(pairs, sequences):
    """One GEMM fewer per extra input of a visit, one per message gate."""
    text = _tiny_text_config()
    counts = []
    for engine in pairs[1]:
        steps = []
        forward = engine.forward

        def counted(rank, batch, forward=forward, steps=steps):
            with _Gemms() as gemms:
                out = forward(rank, batch)
            steps.append((batch.positions.shape[0], sum(gemms.values())))
            return out

        engine.forward = counted
        try:
            plans = hx.scenarios(LEN_A, LEN_B, prefix_len=PREFIX)
            engine.run(plans["decode"], sequences)
        finally:
            del engine.forward
        assert steps[-1][0] == 1
        counts.append(steps[-1][1])
    core = len(text.loop_core_layers)
    passes = text.loop_config["num_passes"]
    gated_visits = core * (passes - 1)
    local = set(text.local_layer_ids)
    global_visits = sum(
        passes if layer in text.loop_core_layers else 1
        for layer in range(text.num_hidden_layers)
        if layer not in local
    )
    local_gated = gated_visits - (passes - 1) * len(
        [layer for layer in text.loop_core_layers if layer not in local]
    )
    # 4 message gates per gated visit lose a GEMM; a gated local visit runs
    # 1 GEMM instead of 3, a global visit 1 instead of 5.
    saved = 4 * gated_visits + 2 * local_gated + 4 * global_visits
    assert counts[1] - counts[0] == saved, counts


def test_kill_switch_keeps_every_projection_apart(pairs):
    weights = _weights_module()
    plain = pairs[1][1].model
    assert not plain.model.merged
    for layer in plain.model.layers:
        assert not layer.attn.merged
        for gate in getattr(layer, "controllers", {}).values():
            assert not gate.merged
            assert weights.rows_view(gate.gain_weight, gate.forget_weight) is None
    plain.process_weights_after_loading()
    assert plain.model.layers[5].attn.q_scale_fp32 is None
    for pack in plain.model.weight_map.packs:
        members = [plain.get_parameter(name) for name in pack.params]
        assert weights.rows_view(*members) is None


def test_switch_name_is_registered():
    from vllm_sm70 import envs

    weights = _weights_module()
    assert weights.MERGED_SWITCH == SWITCH
    assert SWITCH in envs._SECTIONS["models"]
