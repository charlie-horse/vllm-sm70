# SPDX-License-Identifier: Apache-2.0
"""CPU stand-ins for the `_sm70_C::fv100_*` ops, for backend tests.

They follow the op contracts (out first, K/V strided views, decode row b's
query at position seq_lens[b] - 1, prefill q [B, H, M, D] ending at
seq_lens[b]) and compute the result with the SDPA reference, so the
backend's routing, batching and layout handling can be checked without a
GPU. `calls` records (op name, selected argument values).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch  # noqa: E402
from _fv100_ref import FULL, decode_reference, prefill_reference  # noqa: E402


class CpuOps:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, name: str):
        if "prefill" in name:
            return self._prefill(name)
        if name == "fv100_decode_verify":
            return self._verify(name)
        return self._decode(name)

    def _verify(self, name):
        # B2: same expanded-row contract as the XQA decode (row b's query at
        # position seq_lens[b] - 1), 8 rows per request, 128-key partitions.
        def op(out, q, k, v, table, lens, tmp, mx, es, scale, query_len, launch):
            assert query_len == 8 and q.shape[0] % 8 == 0
            assert launch * 128 >= int(lens.max()), (launch, int(lens.max()))
            assert tmp.shape[2] >= launch and mx.shape[2] >= launch
            for r in range(0, q.shape[0], 8):  # one table row per request
                assert torch.equal(table[r : r + 8], table[r : r + 1].expand(8, -1))
            self.calls.append(
                (name, {"rows": q.shape[0], "launch": launch, "query_len": query_len})
            )
            # Full attention: B2 takes no window and no sinks.
            ref = decode_reference(
                q, k, v, table, lens, scale=scale, window=FULL, sinks=None
            )
            out.copy_(ref.to(out.dtype))

        return op

    def _decode(self, name):
        def op(
            out,
            q,
            k,
            v,
            table,
            lens,
            tmp,
            mx,
            es,
            active,
            scale,
            partition,
            launch,
            kv_dtype,
            k_scale,
            v_scale,
            wl,
            wr,
            extra=None,
            counters=None,
        ):
            assert launch * partition >= int(lens.max()), (launch, partition)
            sinks = None if name == "fv100_decode_paged_xqa" else extra
            self.calls.append(
                (
                    name,
                    dict(
                        rows=q.shape[0],
                        window=(wl, wr),
                        sinks=sinks,
                        launch=launch,
                        partition=partition,
                        counters=counters,
                    ),
                )
            )
            ref = decode_reference(
                q, k, v, table, lens, scale=scale, window=(wl, wr), sinks=sinks
            )
            out.copy_(ref.to(out.dtype))

        return op

    def _prefill(self, name):
        def op(
            out,
            q,
            k,
            v,
            table,
            lens,
            scale,
            kv_dtype,
            k_scale,
            v_scale,
            causal,
            wl,
            wr,
            sinks=None,
        ):
            assert q.is_contiguous() and out.is_contiguous()
            self.calls.append(
                (
                    name,
                    dict(
                        rows=q.shape[0],
                        query_len=q.shape[2],
                        causal=causal,
                        window=(wl, wr),
                        sinks=sinks,
                    ),
                )
            )
            ref = prefill_reference(
                q.permute(0, 2, 1, 3),
                k,
                v,
                table,
                lens,
                scale=scale,
                causal=causal,
                window=(wl, wr),
                sinks=sinks,
            )
            out.copy_(ref.permute(0, 2, 1, 3).to(out.dtype))

        return op
