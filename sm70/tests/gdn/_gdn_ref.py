# SPDX-License-Identifier: Apache-2.0
"""K08 GPU test references: inputs, the float64 recurrence and the upstream
Triton chunk comparison that runs in a spawned child (see
test_k08_gdn_gpu.py::test_prefill_matches_triton_chunk).

Importable by a `spawn` child: no pytest, no fixtures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

D = 128
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


def errors_vs_fp64(q, k, v, g, beta, h0, cu, o, ht) -> dict[str, float]:
    starts = cu.tolist()
    o_ref = torch.empty_like(v, dtype=torch.float64)
    ht_ref = torch.empty_like(h0, dtype=torch.float64)
    for n in range(len(starts) - 1):
        s, e = starts[n], starts[n + 1]
        o_ref[0, s:e], ht_ref[n] = ref_recurrence(
            q[0, s:e], k[0, s:e], v[0, s:e], g[0, s:e], beta[0, s:e], h0[n], D**-0.5
        )
    return {"o vs fp64": rel_l2(o, o_ref), "state vs fp64": rel_l2(ht, ht_ref)}


def chunk_prefill_errors(h: int, hv: int) -> dict[str, float]:
    """FlashQLA prefill vs upstream Triton chunk_gated_delta_rule (spawn child).

    On sm_70 Triton 3.8 lowers the chunk kernels' tl.dot to FMA; their
    autotuned variants take ~10 min of ptxas per shape (c4130, 2026-11-06).
    """
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    from vllm_sm70.gdn import kernels

    q, k, v, g, beta, h0, cu = _prefill_inputs(h, hv)
    o, ht = kernels.flashqla_chunk_gated_delta_rule(
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
    torch.cuda.synchronize()
    return {
        "o vs triton chunk": rel_l2(o, o_tr),
        "state vs triton chunk": rel_l2(ht, ht_tr),
    }


def _child(h: int, hv: int, conn) -> None:
    try:
        conn.send(("ok", chunk_prefill_errors(h, hv)))
    except BaseException as exc:  # report, never hang the parent
        import traceback

        conn.send(("error", "".join(traceback.format_exception(exc))))
    finally:
        conn.close()
