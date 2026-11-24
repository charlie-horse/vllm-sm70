# SPDX-License-Identifier: Apache-2.0
"""M12 Inkling 2 DSpark draft (inkling2_sm70/dspark.py, speculator.py), on CPU.

A tiny random draft (3 layers, local layers 0 and 2 with a window of 6,
blocks of 8, Markov rank 8) is loaded from upstream-named tensors into one
module per TP rank. The paged engine harness (inkling2_harness.py) runs
draft rounds: a context step stores the K/V of the committed tokens, a
block step drafts 8 positions, then the 8 positions are taken back and the
next context step rewrites them. The speculator's sequential Markov
sampling runs on each block. Its fp32 logits are compared with a dense fp32
reference built from the unsharded tensors: within 1e-4 in fp32 and 1e-2
of the largest logit in fp16, with no inf/NaN, at TP1 and TP2, with
contexts past the window (the harness frees the blocks before it and
fills them with NaN).

Rounds that verify no drafts (a context step alone, as the speculator's
propose with num_speculative_tokens 0) leave the later rounds' drafts and
logits as in a run that drafted every round.

Also: the weight loader's checks, the fold, f_rnn after rejected tokens,
and the registered class names. The fp16 TP1 cases run again in a child
pytest with TRITON_INTERPRET=1, through the Triton kernels.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
H, HEADS, KV_HEADS, D, INTER = 64, 4, 2, 16, 128
LAYERS, LOCAL, WINDOW, BLOCK, RANK = 3, (0, 2), 6, 8, 8
VOCAB, PADDED_VOCAB, TAPS = 50, 56, 3
PAGE, MAX_LEN, CAP, EPS, THETA = 4, 256, 30.0, 1e-6, 10000.0
MASK_ID = 0
# Context tokens per request per round; the first round is the prompt.
ROUNDS = ({"a": 13, "b": 5}, {"a": 3, "b": 1}, {"a": 8, "b": 2}, {"a": 1, "b": 7})
TOLS = {torch.float32: 1e-4, torch.float16: 1e-2}


def _config():
    from vllm_sm70.models.inkling2_sm70.configs import Inkling2DSparkConfig

    return Inkling2DSparkConfig(
        block_size=BLOCK,
        target_layer_ids=list(range(1, TAPS + 1)),
        hidden_size=H,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=D,
        intermediate_size=INTER,
        max_position_embeddings=4 * MAX_LEN,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        local_layer_ids=list(LOCAL),
        sliding_window_size=WINDOW,
        markov_rank=RANK,
        markov_head_type="rnn_hidden",
        vocab_size=VOCAB,
        padded_vocab_size=PADDED_VOCAB,
        final_logit_softcapping=CAP,
    )


def _checkpoint(seed: int = 0) -> dict[str, torch.Tensor]:
    """fp32 tensors under the names of mtp.safetensors."""
    g = torch.Generator().manual_seed(seed)

    def lin(rows, cols):
        return torch.randn(rows, cols, generator=g) / math.sqrt(cols)

    def norm(n):
        return 1.0 + 0.1 * torch.randn(n, generator=g)

    w = {}
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = lin(HEADS * D, H)
        for name in ("k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"):
            w[p + f"self_attn.{name}.weight"] = lin(KV_HEADS * D, H)
        w[p + "self_attn.o_proj.weight"] = lin(H, HEADS * D)
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            w[p + f"self_attn.{name}.weight"] = norm(D)
        for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            w[p + f"{name}.weight"] = norm(H)
        w[p + "mlp.gate_proj.weight"] = lin(INTER, H)
        w[p + "mlp.up_proj.weight"] = lin(INTER, H)
        w[p + "mlp.down_proj.weight"] = lin(H, INTER)
    w["norm.weight"] = norm(H)
    w["fuse_weight"] = torch.randn(LAYERS + 1, TAPS, generator=g)
    w["mask_embed"] = torch.randn(H, generator=g)
    m = "markov_head."
    w[m + "markov_w1.weight"] = torch.randn(PADDED_VOCAB, RANK, generator=g)
    w[m + "rnn_proj.weight"] = lin(3 * RANK, 2 * RANK + 2 * H)
    w[m + "rnn_proj.bias"] = 0.1 * torch.randn(3 * RANK, generator=g)
    w[m + "hidden_out.weight"] = lin(H, RANK)
    w[m + "hidden_out.bias"] = 0.1 * torch.randn(H, generator=g)
    w[m + "dflare_norm.weight"] = norm(H)
    return w


def _target_weights(seed: int = 1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "embed": torch.randn(PADDED_VOCAB, H, generator=g),
        "embed_norm": 1.0 + 0.1 * torch.randn(H, generator=g),
        "lm_head": 2.0 * torch.randn(VOCAB, H, generator=g) / math.sqrt(H),
    }


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight


class _Body(torch.nn.Module):
    def __init__(self, t: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
        super().__init__()
        self.embed = t["embed"].to(dtype)
        self.norm = t["embed_norm"].to(dtype)

    def embed_input_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return _rms(self.embed[ids].float(), self.norm.float()).to(self.embed.dtype)


class _Target(torch.nn.Module):
    def __init__(self, t: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
        super().__init__()
        self.model = _Body(t, dtype)


# --- dense fp32 reference ----------------------------------------------------------


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """neox RoPE of x [T, heads, D]."""
    inv = THETA ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    angle = (positions.double()[:, None] * inv)[:, None, :]
    cos, sin = angle.cos().float(), angle.sin().float()
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def _heads(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.view(x.shape[0], n, D)


def reference(w, t, features, ctx_len, anchor, prevs):
    """fp32 logits [BLOCK - 1, VOCAB] of one request's block after ctx_len
    committed tokens with target features [>= ctx_len, TAPS * H]; the
    Markov chain is fed `prevs` (the anchor, then the drafted tokens)."""
    alpha = torch.softmax(w["fuse_weight"], -1)
    feats = features[:ctx_len].float().view(ctx_len, TAPS, H)
    ctx_pos = torch.arange(ctx_len)
    pos = torch.arange(ctx_len, ctx_len + BLOCK)
    embed = _rms(t["embed"][anchor], t["embed_norm"])
    res = torch.cat([embed[None], w["mask_embed"].expand(BLOCK - 1, H)])
    for i in range(LAYERS):
        p = f"layers.{i}."
        a = p + "self_attn."
        ctx = _rms((feats * alpha[i][:, None]).sum(1), w[p + "dflare_norm.weight"])
        kc = _rms(_heads(ctx @ w[a + "k_proj_ctx.weight"].T, KV_HEADS),
                  w[a + "k_norm_ctx.weight"])  # fmt: skip
        kc = _rope(kc, ctx_pos)
        vc = _heads(ctx @ w[a + "v_proj_ctx.weight"].T, KV_HEADS)
        h = _rms(res, w[p + "input_layernorm.weight"])
        q = _rope(_rms(_heads(h @ w[a + "q_proj.weight"].T, HEADS),
                       w[a + "q_norm.weight"]), pos)  # fmt: skip
        k = _rope(_rms(_heads(h @ w[a + "k_proj.weight"].T, KV_HEADS),
                       w[a + "k_norm.weight"]), pos)  # fmt: skip
        v = _heads(h @ w[a + "v_proj.weight"].T, KV_HEADS)
        keys, values = torch.cat([kc, k]), torch.cat([vc, v])
        key_pos = torch.cat([ctx_pos, pos])
        if i in LOCAL:
            keep = key_pos >= ctx_len - WINDOW
            keys, values = keys[keep], values[keep]
        keys = keys.repeat_interleave(HEADS // KV_HEADS, 1)
        values = values.repeat_interleave(HEADS // KV_HEADS, 1)
        scores = torch.einsum("qhd,khd->hqk", q, keys) / math.sqrt(D)
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
        res = res + out.reshape(BLOCK, -1) @ w[a + "o_proj.weight"].T
        h = _rms(res, w[p + "post_attention_layernorm.weight"])
        gate = h @ w[p + "mlp.gate_proj.weight"].T
        up = h @ w[p + "mlp.up_proj.weight"].T
        res = (
            res
            + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
        )
    hidden = _rms(res, w["norm.weight"])[1:]
    base = hidden @ t["lm_head"].T
    m = "markov_head."
    f_rnn = _rms((feats[-1] * alpha[-1][:, None]).sum(0), w[m + "dflare_norm.weight"])
    state = torch.zeros(RANK)
    logits = []
    for i, prev in enumerate(prevs):
        inputs = torch.cat([state, w[m + "markov_w1.weight"][prev], hidden[i], f_rnn])
        raw = w[m + "rnn_proj.weight"] @ inputs + w[m + "rnn_proj.bias"]
        gate, cand, out = raw.chunk(3)
        gate = torch.sigmoid(gate)
        state = gate * state + (1 - gate) * torch.tanh(cand)
        latent = torch.tanh(out)
        bias = t["lm_head"] @ (
            w[m + "hidden_out.weight"] @ latent + w[m + "hidden_out.bias"]
        )
        logits.append(CAP * torch.tanh((base[i] + bias) / CAP))
    return torch.stack(logits)


# --- the draft on the harness -------------------------------------------------------


def _vllm_config(harness: hx.PagedHarness, rank: int, hf_config) -> SimpleNamespace:
    config = harness.vllm_config(rank)
    config.model_config.max_model_len = MAX_LEN
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=BLOCK - 1,
        draft_model_config=SimpleNamespace(hf_config=hf_config),
    )
    return config


def _head(rank: int, tp_size: int, dtype: torch.dtype, target, sharded: bool):
    """The target's LM head as the model builds it: rank `rank`'s vocabulary
    rows when `sharded` (weights.py ``head_sharded``), else whole."""
    from vllm_sm70.models.inkling2_sm70.model import Inkling2Sm70LMHead

    head_tp = tp_size if sharded else 1
    head = Inkling2Sm70LMHead(
        VOCAB, H, dtype, tp_size=head_tp, tp_rank=rank if sharded else 0
    )
    head.weight.data.copy_(target["lm_head"].chunk(head_tp)[rank % head_tp])
    return head


def _build(
    tp_size: int, dtype: torch.dtype, checkpoint, target, *, sharded: bool = False
):
    from vllm_sm70.models.inkling2_sm70.dspark import Inkling2DSparkSm70Model

    harness = hx.PagedHarness(tp_size, block_size=PAGE, dtype=dtype, num_blocks=96)
    hf_config = _config()
    models = []
    for rank in range(tp_size):
        with harness.tp_rank(rank):
            model = Inkling2DSparkSm70Model(
                vllm_config=_vllm_config(harness, rank, hf_config)
            )
            loaded = model.load_weights(checkpoint.items())
            assert loaded == {name for name, _ in model.named_parameters()}
            model.lm_head = _head(rank, tp_size, dtype, target, sharded)
            models.append(model)
    # A sharded head's fold is a collective.
    harness.lockstep(lambda rank: models[rank].attach_target(_Target(target, dtype)))
    harness.bind()
    return harness, models


def _speculator(model, num_reqs: int, max_tokens: int, dtype: torch.dtype):
    """The speculator's state that its context-anchor and sampling steps read,
    without the v0.33 runner around it; it samples greedily."""
    from vllm_sm70.models.inkling2_sm70.speculator import Inkling2DSparkSm70Speculator

    spec = Inkling2DSparkSm70Speculator.__new__(Inkling2DSparkSm70Speculator)
    steps = BLOCK - 1
    spec.model = model
    spec.num_speculative_steps = steps
    spec.hidden_states = torch.zeros(max_tokens, TAPS * H, dtype=dtype)
    spec.f_rnn = torch.zeros(num_reqs, H, dtype=dtype)
    spec.draft_tokens = torch.zeros(num_reqs, steps, dtype=torch.int64)
    spec._anchor_idx = torch.arange(num_reqs) * BLOCK
    rows = torch.arange(num_reqs)[:, None] * BLOCK + torch.arange(1, BLOCK)
    spec.sample_indices = rows.flatten()
    spec.sample_idx_mapping = torch.arange(num_reqs).repeat_interleave(steps)
    spec.sample_pos = torch.zeros(num_reqs * steps, dtype=torch.int64)
    spec.logits = []

    def sample(logits, idx_map, sample_pos, step):
        assert logits.dtype == torch.float32 and logits.shape[-1] == VOCAB
        spec.logits.append(logits.clone())
        return logits.argmax(-1)

    spec._sample_logits = sample
    return spec


def run_rounds(
    tp_size: int,
    dtype: torch.dtype,
    seed: int = 0,
    *,
    sharded: bool = False,
    skip: frozenset[int] = frozenset(),
):
    """Per drafted round and request: (ctx_len, anchor, tokens, logits
    [7, V]) of every rank, and the fp32 features. Rounds in `skip` verify no
    drafts, as the speculator's propose with num_speculative_tokens 0: the
    context step stores the K/V only and there is no block step."""
    from vllm.forward_context import get_forward_context

    checkpoint, target = _checkpoint(seed), _target_weights(seed + 1)
    harness, models = _build(tp_size, dtype, checkpoint, target, sharded=sharded)
    g = torch.Generator().manual_seed(seed + 2)
    total = {r: sum(rnd[r] for rnd in ROUNDS) for r in ROUNDS[0]}
    features = {r: torch.randn(n, TAPS * H, generator=g) for r, n in total.items()}
    anchors = torch.randint(1, VOCAB, (len(ROUNDS), 2), generator=g)
    for req in ROUNDS[0]:
        harness.add_request(req)
    specs = [_speculator(m, 2, 64, dtype) for m in models]
    results = []
    for index, schedule in enumerate(ROUNDS):
        ctx = {r: harness.requests[r].computed + n for r, n in schedule.items()}

        def store(rank, batch, index=index):
            model, spec = models[rank], specs[rank]
            feats = batch.take({r: f.to(dtype) for r, f in features.items()})
            md = get_forward_context().attn_metadata
            names = model.get_draft_kv_cache_layer_names()
            slots = [md[name].slot_mapping for name in names]
            model.precompute_and_store_context_kv(feats, batch.positions, slots)
            spec.hidden_states[: feats.shape[0]] = feats
            if index in skip:
                return list(batch.req_ids)
            rejected = torch.zeros(len(batch.req_ids), dtype=torch.int32)
            spec.prepare_context_anchor(
                len(batch.req_ids), batch.query_start_loc, rejected
            )
            return list(batch.req_ids)

        order = harness.step(schedule, store)[0]
        if index in skip:
            continue
        f_rows = {r: i for i, r in enumerate(order)}

        def block(rank, batch, index=index, f_rows=f_rows):
            model, spec = models[rank], specs[rank]
            ids = torch.full((batch.num_tokens,), MASK_ID, dtype=torch.int64)
            for r in batch.req_ids:
                ids[batch.rows(r).start] = anchors[index, "ab".index(r)]
            hidden = model(ids, batch.positions)
            assert hidden.dtype == dtype and hidden.shape == (batch.num_tokens, H)
            # f_rnn rows follow the context step's request order.
            spec.f_rnn = spec.f_rnn[[f_rows[r] for r in batch.req_ids]]
            spec.input_buffers = SimpleNamespace(input_ids=ids)
            spec.logits = []
            spec._sample_sequential(len(batch.req_ids), hidden)
            logits = torch.stack(spec.logits, 1)
            return {
                r: (spec.draft_tokens[i].clone(), logits[i])
                for i, r in enumerate(batch.req_ids)
            }

        outs = harness.step(dict.fromkeys(schedule, BLOCK), block)
        for req in schedule:
            harness.requests[req].computed -= BLOCK
            anchor = int(anchors[index, "ab".index(req)])
            per_rank = [out[req] for out in outs]
            results.append((req, ctx[req], anchor, per_rank))
    return results, features, checkpoint, target


@pytest.fixture(scope="module")
def fp32_run():
    return run_rounds(1, torch.float32)


@pytest.mark.parametrize(
    ("tp_size", "dtype", "sharded"),
    [
        pytest.param(1, torch.float32, False, id="tp1-fp32"),
        pytest.param(1, torch.float16, False, id="tp1-fp16"),
        pytest.param(2, torch.float32, False, id="tp2-fp32-whole-head"),
        pytest.param(2, torch.float32, True, id="tp2-fp32-sharded-head"),
        pytest.param(2, torch.float16, True, id="tp2-fp16-sharded-head"),
    ],
)
def test_draft_logits_match_the_dense_reference(tp_size, dtype, sharded):
    if INTERPRETED and (tp_size, dtype) != (1, torch.float16):
        pytest.skip("the interpreted child runs the fp16 TP1 case")
    results, features, checkpoint, target = run_rounds(tp_size, dtype, sharded=sharded)
    assert max(ctx for _, ctx, _, _ in results) > WINDOW + BLOCK + PAGE
    worst = 0.0
    for req, ctx_len, anchor, per_rank in results:
        tokens, logits = per_rank[0]
        assert torch.isfinite(logits).all()
        for other_tokens, other_logits in per_rank[1:]:
            assert torch.equal(other_tokens, tokens)
            torch.testing.assert_close(other_logits, logits, atol=1e-6, rtol=0)
        prevs = [anchor, *tokens[:-1].tolist()]
        want = reference(checkpoint, target, features[req], ctx_len, anchor, prevs)
        err = (logits - want).abs().max().item() / max(1.0, want.abs().max().item())
        worst = max(worst, err)
        if dtype == torch.float32:
            assert torch.equal(tokens, want.argmax(-1))
    assert worst <= TOLS[dtype], worst


def test_the_fp16_draft_tokens_mostly_agree_with_fp32(fp32_run):
    if INTERPRETED:
        pytest.skip("covered by the parent run")
    fp16, *_ = run_rounds(1, torch.float16)
    same = total = 0
    for (_, _, _, a), (_, _, _, b) in zip(fp32_run[0], fp16, strict=True):
        same += int((a[0][0] == b[0][0]).sum())
        total += a[0][0].numel()
    assert same >= 0.9 * total, (same, total)


@pytest.mark.parametrize("skip", [frozenset({1, 2}), frozenset({0, 2})])
def test_rounds_without_drafts_keep_the_draft_in_sync(fp32_run, skip):
    """A step that verifies no drafts (spec_policy.py) stores the context
    K/V only; a later drafting round gives the drafts and logits of a run
    that drafted every round."""
    if INTERPRETED:
        pytest.skip("covered by the parent run")
    got, *_ = run_rounds(1, torch.float32, skip=skip)
    want = [
        result
        for index, result in enumerate(fp32_run[0])
        if index // len(ROUNDS[0]) not in skip
    ]
    assert len(got) == len(want) == len(ROUNDS[0]) * (len(ROUNDS) - len(skip))
    for (req, ctx, anchor, ranks), (w_req, w_ctx, w_anchor, w_ranks) in zip(
        got, want, strict=True
    ):
        assert (req, ctx, anchor) == (w_req, w_ctx, w_anchor)
        assert torch.equal(ranks[0][0], w_ranks[0][0])
        torch.testing.assert_close(ranks[0][1], w_ranks[0][1], atol=1e-6, rtol=0)


# --- pieces -------------------------------------------------------------------------


def test_interpreted_suite():
    """Run the fp16 TP1 case again with TRITON_INTERPRET=1: every draft op
    through its Triton kernel."""
    if INTERPRETED:
        pytest.skip("this is the interpreted child run")
    env = dict(os.environ, TRITON_INTERPRET="1")
    env.pop("VLLM_SM70_INKLING_DSPARK_KERNELS", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu",
            "-k",
            "tp1-fp16 or kernels_are_on",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    tail = proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    summary = proc.stdout.strip().splitlines()[-1]
    assert re.search(r"\b2 passed", summary), summary
    assert "skipped" not in summary, summary


def test_kernels_are_on_in_the_interpreter():
    from vllm_sm70.models.inkling2_sm70 import dspark

    assert dspark.kernels_on(torch.zeros(1)) is INTERPRETED


def test_prepare_context_anchor_reads_the_last_accepted_row():
    from vllm_sm70.models.inkling2_sm70.speculator import Inkling2DSparkSm70Speculator

    spec = Inkling2DSparkSm70Speculator.__new__(Inkling2DSparkSm70Speculator)
    spec.hidden_states = torch.arange(20.0)[:, None].expand(20, 3).clone()
    spec.f_rnn = torch.zeros(4, 3)
    spec.model = SimpleNamespace(compute_f_rnn=lambda rows: rows * 2)
    query_start_loc = torch.tensor([0, 4, 5, 12, 12], dtype=torch.int32)
    rejected = torch.tensor([2, 0, 7, 0, 9], dtype=torch.int32)
    spec.prepare_context_anchor(3, query_start_loc, rejected)
    assert spec.f_rnn[:, 0].tolist() == [2 * 1, 2 * 4, 2 * 4, 0]


def test_draft_logits_are_fp32():
    from vllm_sm70.models.inkling2_sm70.speculator import Inkling2DSparkSm70Speculator

    spec = Inkling2DSparkSm70Speculator.__new__(Inkling2DSparkSm70Speculator)
    assert spec.draft_logits_spec(None) == (torch.float32, 0.0)


def test_a_sharded_head_folds_and_gathers_the_whole_vocabulary():
    """Each rank holds half of the LM head rows; the fold and the base
    logits cover the whole vocabulary on every rank."""
    checkpoint, target = _checkpoint(), _target_weights()
    harness, models = _build(2, torch.float32, checkpoint, target, sharded=True)
    m = "markov_head."
    want = target["lm_head"] @ checkpoint[m + "hidden_out.weight"]
    hidden = torch.randn(3, H, generator=torch.Generator().manual_seed(5))
    base = harness.lockstep(lambda rank: models[rank].compute_base_logits(hidden))
    for model, logits in zip(models, base, strict=True):
        assert model.lm_head.weight.shape == (VOCAB // 2, H)
        torch.testing.assert_close(model.model.markov_head.fold_weight, want)
        torch.testing.assert_close(logits, hidden @ target["lm_head"].T)


def test_the_fold_is_fp64_and_refills_in_place():
    checkpoint, target = _checkpoint(), _target_weights()
    _, (model,) = _build(1, torch.float16, checkpoint, target)
    head = model.model.markov_head
    m = "markov_head."
    want = target["lm_head"].double() @ checkpoint[m + "hidden_out.weight"].double()
    assert head.fold_weight.dtype == torch.float16
    torch.testing.assert_close(head.fold_weight.double(), want, atol=2e-3, rtol=1e-3)
    pointer = head.fold_weight.data_ptr()
    model.lm_head.weight.data.mul_(2)
    model.refresh_derived_weights()
    assert head.fold_weight.data_ptr() == pointer
    torch.testing.assert_close(
        head.fold_weight.double(), 2 * want, atol=4e-3, rtol=1e-3
    )


@pytest.mark.parametrize(
    ("tensors", "match"),
    [
        ({"layers.0.self_attn.r_proj.weight": torch.zeros(4, H)}, "unexpected"),
        ({"norm.weight": torch.zeros(H + 1)}, "has shape"),
        ({"norm.weight": torch.full((H,), 7e4)}, "fp16 range"),
        ({"layers.0.mlp.gate_proj.weight": torch.zeros(INTER, H)}, "incomplete"),
    ],
    ids=["unexpected", "shape", "range", "incomplete"],
)
def test_the_loader_refuses_bad_tensors(tensors, match):
    checkpoint, target = _checkpoint(), _target_weights()
    _, (model,) = _build(1, torch.float16, checkpoint, target)
    with pytest.raises(ValueError, match=match):
        model.load_weights(tensors.items())


def test_registered_names_resolve():
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_sm70.models.inkling2_sm70 import draft
    from vllm_sm70.models.inkling2_sm70.dspark import Inkling2DSparkSm70Model
    from vllm_sm70.models.inkling2_sm70.speculator import Inkling2DSparkSm70Speculator

    module, name = draft.DRAFT_LAZY_TARGET.split(":")
    assert getattr(importlib.import_module(module), name) is Inkling2DSparkSm70Model
    spec_cls = resolve_obj_by_qualname(draft.SPECULATOR_TARGET)
    assert spec_cls is Inkling2DSparkSm70Speculator


def test_num_speculative_tokens_must_fill_the_block():
    from vllm_sm70.models.inkling2_sm70.dspark import Inkling2DSparkSm70Model

    harness = hx.PagedHarness(1, block_size=PAGE)
    config = _vllm_config(harness, 0, _config())
    config.speculative_config.num_speculative_tokens = BLOCK
    with harness.tp_rank(0), pytest.raises(ValueError, match="num_speculative"):
        Inkling2DSparkSm70Model(vllm_config=config)
