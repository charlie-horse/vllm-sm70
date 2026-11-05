# SPDX-License-Identifier: Apache-2.0
"""K08 GPU checks for the FlashQLA GDN kernels on SM70 (V100).

What:
- The GDN ops come from the built _sm70_C (port of fork
  tests/kernels/mamba/test_sm70_flashqla_extension.py: there is no pybind
  loader or JIT fallback any more, so the check is that the ops are
  registered with CUDA kernels in the installed .so, plus the column-group
  heuristic and torch.library.opcheck on the schemas and fakes).
- Prefill route selection (adapted from fork tests/model_executor/layers/
  test_gdn_prefill_backend_resolve.py): on SM70 the OOT ChunkGatedDeltaRule
  routes to FlashQLA; VLLM_SM70_GDN_FLASHQLA_PREFILL=0 or a non-fp16 /
  non-128 tensor keeps upstream Triton/FLA.
- Exactness at the Qwen3.6 TP2 (H8/HV16) and Qwen3.8 TP2 (H8/HV24) shapes,
  varlen with 2 sequences: FlashQLA prefill output and final state vs
  upstream Triton chunk_gated_delta_rule, and FlashQLA decode output and
  state vs upstream fused_recurrent_gated_delta_rule_packed_decode, each
  rel-L2 < 1e-3; both also against a float64 recurrence.

Run from the repo root: pytest -m gpu sm70/tests/gdn
"""

from __future__ import annotations

import pytest
from _gdn_helpers import mock_ctor_env

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

pytestmark = pytest.mark.gpu

D = 128
TOL = 1e-3
SHAPES = [
    pytest.param(8, 16, id="qwen3.6-tp2-H8-HV16"),
    pytest.param(8, 24, id="qwen3.8-tp2-H8-HV24"),
]
SEQLENS = (300, 517)  # varlen, 2 sequences; not multiples of the 64 chunk


def rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def ref_recurrence(q, k, v, g, beta, h0, scale):
    """float64 gated delta rule for one sequence, state [Hv, V, K].
    q, k [T, H, K]; v [T, Hv, V]; g (log decay), beta [T, Hv]."""
    rep = v.shape[1] // q.shape[1]
    q = q.double().repeat_interleave(rep, dim=1)
    k = k.double().repeat_interleave(rep, dim=1)
    v, g, beta = v.double(), g.double(), beta.double()
    state = h0.double().clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        state *= g[t].exp()[:, None, None]
        kv = torch.einsum("hvk,hk->hv", state, k[t])
        u = beta[t][:, None] * (v[t] - kv)
        state += u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", state, q[t]) * scale
    return out, state


def _prefill_inputs(h, hv, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    T = sum(SEQLENS)

    def rn(*s):
        return torch.randn(*s, generator=gen, device=dev)

    q = F.normalize(rn(1, T, h, D), dim=-1).half()
    k = F.normalize(rn(1, T, h, D), dim=-1).half()
    v = (0.5 * rn(1, T, hv, D)).half()
    g = -(0.01 + 0.5 * torch.rand(1, T, hv, generator=gen, device=dev))
    beta = torch.rand(1, T, hv, generator=gen, device=dev).sigmoid()
    h0 = 0.1 * rn(len(SEQLENS), hv, D, D)
    cu = torch.tensor(
        [0, SEQLENS[0], SEQLENS[0] + SEQLENS[1]], dtype=torch.int32, device=dev
    )
    return q, k, v, g, beta, h0, cu


# ------------------------------------------------------------- the ops


def test_ops_come_from_the_built_extension(gdn_gpu):
    from vllm_sm70 import _ext

    path = _ext.load().__file__
    assert path.endswith(".so"), path
    for name in (
        "gdn_forward_vlk_varlen_out",
        "gdn_decode_mixed_qkv_global_state",
        "gdn_resolve_column_groups_per_block",
    ):
        assert gdn_gpu.has_op(name)


@pytest.mark.parametrize(
    "tokens, h, hv, groups",
    [
        (4096, 8, 16, 2),
        (2, 8, 16, 2),
        (4096, 8, 24, 2),
        (2048, 4, 12, 2),
        (10, 4, 12, 1),
        (10, 4, 8, 1),
        (5000, 4, 8, 4),
    ],
)
def test_column_groups_heuristic(gdn_gpu, tokens, h, hv, groups):
    assert gdn_gpu.gdn_resolve_column_groups_per_block(tokens, h, hv) == groups


def test_opcheck(gdn_gpu):
    h, hv = 8, 16
    q, k, v, g, beta, h0, cu = _prefill_inputs(h, hv)
    out = torch.empty_like(v)
    state = torch.empty_like(h0)
    torch.library.opcheck(
        torch.ops._sm70_C.gdn_forward_vlk_varlen_out.default,
        (out, state, q, k, v, g, beta, h0, cu, D**-0.5, False),
        test_utils=("test_schema", "test_faketensor"),
    )
    args = _decode_inputs(h, hv, [3, 5])
    torch.library.opcheck(
        torch.ops._sm70_C.gdn_decode_mixed_qkv_global_state.default,
        (
            args["mixed_qkv"],
            args["a"],
            args["b"],
            args["A_log"],
            args["dt_bias"],
            args["ssm_state"],
            args["state_indices"],
            args["out"],
            D**-0.5,
            True,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )


# ------------------------------------------------- prefill route selection


@pytest.fixture
def oot_gpu(monkeypatch, gdn_gpu):
    pytest.importorskip("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    from vllm.model_executor.custom_op import op_registry_oot

    m = mock_ctor_env(monkeypatch)
    monkeypatch.setattr(gdn_gpu, "_oot_class", None)
    monkeypatch.delitem(op_registry_oot, "ChunkGatedDeltaRule", raising=False)
    gdn_gpu.register_chunk_gated_delta_rule()
    yield m
    op_registry_oot.pop("ChunkGatedDeltaRule", None)


def test_volta_routes_prefill_to_flashqla(oot_gpu, gdn_gpu):
    op = oot_gpu.ChunkGatedDeltaRule()
    assert op.gdn_prefill_backend == "flashqla_sm70"
    q, k, v, g, beta, h0, cu = _prefill_inputs(8, 16)
    o, ht = op(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
    )
    o2, ht2 = gdn_gpu.flashqla_chunk_gated_delta_rule(
        q, k, v, g, beta, h0, True, cu_seqlens=cu
    )
    torch.testing.assert_close(o, o2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ht, ht2, rtol=1e-4, atol=1e-5)


def test_prefill_kill_switch_keeps_triton(monkeypatch, oot_gpu):
    monkeypatch.setenv("VLLM_SM70_GDN_FLASHQLA_PREFILL", "0")
    op = oot_gpu.ChunkGatedDeltaRule()
    assert op._forward_method.__name__ == "forward_native"


@pytest.mark.parametrize(
    "dtype, head_dim, reason",
    [(torch.bfloat16, 128, "dtype"), (torch.float16, 64, "head_dim")],
)
def test_non_qualifying_tensors_fall_back(gdn_gpu, dtype, head_dim, reason):
    q = torch.zeros(1, 4, 8, head_dim, dtype=dtype, device="cuda")
    v = torch.zeros(1, 4, 16, head_dim, dtype=dtype, device="cuda")
    g = torch.zeros(1, 4, 16, device="cuda")
    assert gdn_gpu.prefill_rejection(q, q, v, g, g, None) == reason


# ---------------------------------------------------------- exactness


@pytest.mark.parametrize("h, hv", SHAPES)
def test_prefill_matches_triton_and_fp64(gdn_gpu, h, hv):
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    q, k, v, g, beta, h0, cu = _prefill_inputs(h, hv)
    o, ht = gdn_gpu.flashqla_chunk_gated_delta_rule(
        q, k, v, g, beta, h0, True, cu_seqlens=cu
    )
    o_tr, ht_tr = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=D**-0.5,
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False,
    )
    starts = cu.tolist()
    o_ref = torch.empty_like(v, dtype=torch.float64)
    ht_ref = torch.empty_like(h0, dtype=torch.float64)
    for n in range(len(SEQLENS)):
        s, e = starts[n], starts[n + 1]
        o_ref[0, s:e], ht_ref[n] = ref_recurrence(
            q[0, s:e], k[0, s:e], v[0, s:e], g[0, s:e], beta[0, s:e], h0[n], D**-0.5
        )
    errors = {
        "o vs triton": rel_l2(o, o_tr),
        "state vs triton": rel_l2(ht, ht_tr),
        "o vs fp64": rel_l2(o, o_ref),
        "state vs fp64": rel_l2(ht, ht_ref),
        "triton o vs fp64": rel_l2(o_tr, o_ref),  # context only
    }
    assert errors["o vs triton"] < TOL, errors
    assert errors["state vs triton"] < TOL, errors
    assert errors["o vs fp64"] < TOL, errors
    assert errors["state vs fp64"] < TOL, errors


def _decode_inputs(h, hv, indices, slots=6, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    T = len(indices)

    def rn(*s):
        return torch.randn(*s, generator=gen, device=dev)

    return dict(
        mixed_qkv=rn(T, 2 * h * D + hv * D).half(),
        a=rn(T, hv).half(),
        b=rn(T, hv).half(),
        A_log=torch.empty(hv, device=dev).uniform_(0, 2.7, generator=gen),
        dt_bias=rn(hv),
        ssm_state=0.1 * rn(slots, hv, D, D),
        state_indices=torch.tensor(indices, dtype=torch.int32, device=dev),
        out=torch.empty(T, hv, D, dtype=torch.float16, device=dev),
    )


def _ref_decode(args, h, hv):
    mixed = args["mixed_qkv"].double()
    q = F.normalize(mixed[:, : h * D].view(-1, h, D), dim=-1)
    k = F.normalize(mixed[:, h * D : 2 * h * D].view(-1, h, D), dim=-1)
    v = mixed[:, 2 * h * D :].view(-1, hv, D)
    x = args["a"].double() + args["dt_bias"].double()
    g = -args["A_log"].double().exp() * F.softplus(x)
    beta = args["b"].double().sigmoid()
    outs, states = [], []
    for t, slot in enumerate(args["state_indices"].tolist()):
        o, s = ref_recurrence(
            q[t : t + 1],
            k[t : t + 1],
            v[t : t + 1],
            g[t : t + 1],
            beta[t : t + 1],
            args["ssm_state"][slot],
            D**-0.5,
        )
        outs.append(o[0])
        states.append(s)
    return torch.stack(outs), torch.stack(states)


@pytest.mark.parametrize("h, hv", SHAPES)
def test_decode_matches_triton_and_fp64(gdn_gpu, h, hv):
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    indices = [3, 5]  # 2 requests
    args = _decode_inputs(h, hv, indices)
    tr_state = args["ssm_state"].clone()
    tr_out = torch.empty(len(indices), 1, hv, D, dtype=torch.float16, device="cuda")
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=args["mixed_qkv"],
        a=args["a"],
        b=args["b"],
        A_log=args["A_log"],
        dt_bias=args["dt_bias"],
        scale=D**-0.5,
        initial_state=tr_state,
        out=tr_out,
        ssm_state_indices=args["state_indices"],
        use_qk_l2norm_in_kernel=True,
    )
    o_ref, s_ref = _ref_decode(args, h, hv)
    before = args["ssm_state"].clone()
    assert gdn_gpu.flashqla_decode(**args) is True
    out, state = args["out"], args["ssm_state"]
    idx = torch.tensor(indices, device="cuda")
    errors = {
        "o vs triton": rel_l2(out, tr_out[:, 0]),
        "state vs triton": rel_l2(state[idx], tr_state[idx]),
        "o vs fp64": rel_l2(out, o_ref),
        "state vs fp64": rel_l2(state[idx], s_ref),
    }
    for key, err in errors.items():
        assert err < TOL, (key, errors)
    untouched = [i for i in range(state.shape[0]) if i not in indices]
    torch.testing.assert_close(state[untouched], before[untouched], rtol=0, atol=0)


def test_decode_null_and_pad_slots(gdn_gpu):
    """Slot 0 (NULL_BLOCK_ID) and -1 (PAD_SLOT_ID) write zeros, no state."""
    args = _decode_inputs(8, 16, [0, -1, 2])
    before = args["ssm_state"].clone()
    args["out"].fill_(1.0)
    assert gdn_gpu.flashqla_decode(**args) is True
    assert torch.count_nonzero(args["out"][:2]) == 0
    assert torch.count_nonzero(args["out"][2]) > 0
    keep = [0, 1, 3, 4, 5]
    torch.testing.assert_close(args["ssm_state"][keep], before[keep], rtol=0, atol=0)


def test_decode_kill_switch_runs_nothing(monkeypatch, gdn_gpu):
    monkeypatch.setenv("VLLM_SM70_GDN_DECODE_FLASHQLA", "0")
    args = _decode_inputs(8, 16, [3, 5])
    before = args["ssm_state"].clone()
    assert gdn_gpu.flashqla_decode(**args) is False
    torch.testing.assert_close(args["ssm_state"], before, rtol=0, atol=0)
