# Status: roadmap nodes vs evidence

Snapshot as of 2026-11-08, from merged PRs (#1–#88), PR comments, and Nov 5–7 GPU logs on c4130 (read-only). `~/vllm-sm70-work/...` paths are on c4130. The log paths are the evidence; they are not public links.

**Status terms**
- **passed**: a PR, log or junit file shows the exit test (or the named part of it) passing.
- **partial**: some of the exit test has evidence; the gap is in "what's left".
- **unverified**: a claim exists (for example, a PR body says "passed"), but no log was found, or the exit names a test I could not identify.
- **not run**: no GPU evidence in the PRs or on c4130.
- **failed**: a logged failure of the exit test or its direct proxy.
- "CPU evidence" is the count stated on the node's merged PR. I did not re-run CPU tests for this doc.
- The Nov 7 `gpu_sweep` ran from source `c7fd4ee`. Later merges may change its results.

## Node table

Roadmap order. Each node appears once.

| node | PR(s) | CPU evidence | GPU exit test | evidence link or path | what's left |
|---|---|---|---|---|---|
| N0 | #2 | nix eval in CI | **passed**: Phase 1 gate, TP1 and TP2, TRITON_ATTN, fp16 | PR #2 comment (Bea, Nov 5); `~/vllm-sm70-work/bea/phase1/{tp1,tp2}/` | none for the exit |
| N1 | #6, #7, #13 | CI unit + nix eval, green when merged | n/a (CI node) | PR #6 | main CI not re-checked here; PR #4/#10/#22/#24 record a red `test_no_vllm_envs` on main, fix not confirmed |
| K01 | #16 | `sm70/tests/plugin`: 31 passed, 22 skipped without vllm; 53 with (PR #16) | **not run**: no `_sm70_C` / `_C_stable_libtorch` co-import test found | none | run the co-import check on c4130 |
| N2 | #27, #61 | r440 `.#sm70-ext` builds (comments on #50–#63) | **unverified**: c4130 `import vllm_sm70` + `torch.ops._sm70_C` not in any log found | PR #26 comment (K03 build, 38 ops, CPU import) | c4130 import check |
| U1 | #16 | CPU tests in PR #16 | n/a (CPU exit test) | PR #16 | CPU exit not re-run here |
| M1 | #8 | CPU defaults and kill-switch tests (PR #8) | n/a (CPU exit test) | PR #8 | none |
| U2 | #33 | `sm70/tests/gdn` CPU | **unverified**: "upstream GDN tests pass on sm70" not identified. `tests.gdn` 21 passed in the Nov 7 sweep, but those are the repo's GDN tests | `bea/nov7/logs/gpu_sweep.junit.xml` | name the upstream GDN test; run it |
| K02 | #12, #78 | `check_turbomind_provenance.py --fork` OK at 7958a70 (PR #78) | n/a (build + provenance node) | PR #78 comment | none |
| M2 | #9 | mocked cc tests (PR #9) | **not run**: real Flint and Qwen config admission not tested on GPU | none | run real checkpoint configs through the capability check |
| M7 | #22, #45 | `test_m7_dflash_sm70.py` CPU | **unverified**: PR #34 says "M7 Triton 3/3 passed" (Bea); no log located; the exit ("no inf/NaN in fp16") is not clearly the same test | PR #34 comment | locate Bea's log or re-run with the no-inf/NaN check |
| K03 | #26, #28 | `sm70/tests/ops` CPU (PR #26) | **partial**: `tests.ops` 102 passed in the Nov 7 sweep. The exit's named cases (TM vs `torch.matmul` at M ∈ {1,2,8,64,512}; MoE vs reference loop) are not identified in the junit | `bea/nov7/logs/gpu_sweep.junit.xml` | confirm the named cases; MoE reference loop unverified |
| M3 | #24, #30, #42 | 25 CPU tests (PR #24) | **unverified**: `test_m3_nvfp4_linear.py` GPU tests sit inside `tests.quant` (154 passed in sweep), not broken out | `bea/nov7/logs/gpu_sweep.junit.xml` | per-file result; GEMV path depends on K11 |
| M4 | #31, #34, #52, #72, #79 | 22 CPU tests (PR #31) | **unverified** for `test_m4_nvfp4_moe.py`. `tests.flint_moe` (K12's suite) 15 passed, which is not M4's file | `bea/nov7/logs/flint_moe.log` | run M4's GPU file and report per test |
| K04 | none | n/a | **not run** (optional) | none | AWQ not built |
| M5 | #36, #41, #52 | 333 CPU passed (PR #36) | **failed in the calibration log**: `window.out` shows T4a, T4b, T5 FAIL; T1, T2, T3, T4c, T4d, T6, T7 PASS. PR #86 says a later candidate run passed every node; that log was not found. The quality suite (math 10/10, code 5/5, needle@32K 3/3, gibberish at c20, glm47, LMCache) has no log | `bea/m5-cal-nov7/window.out` | locate the PR #86 window log, or re-run; run the quality suite |
| M6 | #32, #41 | 25 CPU tests (PR #32) | **unverified**: `test_m6_qwen_fp8.py` not broken out of `tests.quant`; TP2 e2e and 32 GB/GPU fit not run | `bea/nov7/logs/gpu_sweep.junit.xml` | per-file result; TP2 e2e on Qwen3.6-35B-A3B and 3.8-27B |
| K05 | #37 | CPU import shows 5 fv100 ops (PR #37) | **unverified**: `test_sm70_flash_v100_sinks_prefill` and `test_sm70_flash_v100_splitkv_n` not identified in junit; the junit also lists `test_fv100_accuracy_gpu` failures I could not reconcile with `failures="9"` | `bea/nov7/logs/gpu_sweep.junit.xml` | identify the two tests; reconcile the fv100 accuracy failures |
| K06 | #39, #51, #83, #84 | 1099 CPU passed at e50db90 (PR #84) | **not run**: the exit (Flint TP2 4K greedy-identical over 256 tokens on 5 prompts; decode tok/s ≥ fork) has no evidence | `bea/nov7/k14.out` (throughput only) | greedy-identity run; tok/s vs fork |
| K07 | #51 | CPU (PR #51) | **not run**: smallq exactness and DFlash2 acceptance within 2% not found | none | run `test_fv100_spec_gpu.py` (36 tests) and acceptance check |
| K08 | #33 | `sm70/tests/gdn` CPU | **partial**: `tests.gdn` 21 passed in the sweep; the rel-L2 < 1e-3 check is not named in the junit | `bea/nov7/logs/gpu_sweep.junit.xml` | confirm the rel-L2 check |
| M11 | none | n/a | **not run**: no PR | none | write the layer subclass; GPU check vs upstream |
| K09 | #50, #65 | CPU (PR #50) | **failed**: `test_k09_qpn8_gpu.py::test_opcheck`: `mul_cuda` not implemented for `Float8_e4m3fn` (sweep); isolated run `rc=1, 1 failed`. The max-abs-rel < 1e-2 check not confirmed | `bea/nov7/logs/gpu_sweep.log`, `bea/nov7/logs/k09iso.log`, `bea/nov7/iso.out` | fix the FP8 dtype in opcheck; re-run the exit |
| K10 | none | n/a | **not run** (opt) | none | needs a Nix TileLang |
| K11 | #53 | CPU (PR #53) | **unverified**: PR #53 says "queued for the Nov 7 window"; `tests.ops` may include `test_k11_nvfp4_tm_gemv_gpu.py`, not confirmed | `bea/nov7/logs/gpu_sweep.junit.xml` | confirm the file ran |
| K12 | #55 | CPU 565 passed (PR #55) | **passed**: `tests.flint_moe` 15 passed, 0 failed (Nov 7 01:12). Exit is "fork tests in the carry map", not mapped one-to-one | `bea/nov7/logs/flint_moe.junit.xml` | map the carry-map tests |
| K13 | #57, #86 | 1296 CPU passed at 8480748 (PR #86) | **partial**: `test_m10_flint_router.py` 2 passed, 0 failed (Nov 7 02:13). "Greedy-identical with the switch off" not seen | `bea/nov7/logs/test_m10_flint_router.junit.xml` | greedy-identity with `VLLM_SM70_FLINT_FUSED_ROUTER=0` |
| K14 | #56, #79 | 535 CPU passed (PR #56) | **partial**: one-launch sink decode live (`k14.out`): 175.7 tok/s at n=1 with K14 on, 121.1 with `VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0`. Coherence sweep only to n=8; `test_decode_sinks_descending_capture_regression` inside `tests.attention`, not broken out | `bea/nov7/k14.out`, `bea/nov7/logs/gpu_sweep.junit.xml` | n=20 coherence sweep; confirm the regression test id |
| K15 | #54 | 533 CPU passed (PR #54) | **not run** as the named file; PR #54 says "queued for Nov 7" | none | run `test_k15_qk_norm_rope_kv_gpu.py` (50 tests) |
| K16 | #59, #66 | 1307 CPU passed at 0291f61 (PR #66) | **passed** for the K16 suite: 66 passed, 0 failed (`k16_full.junit.xml`). M10 sampler: G1, G3 pass, G2 xfail (`test_m10_flint_sampler.junit.xml`). "Greedy-identical with switch off" not shown | `bea/nov7/logs/k16_full.junit.xml`, `bea/nov7/logs/test_m10_flint_sampler.junit.xml` | greedy identity with `VLLM_SM70_FLINT_SAMPLER=0` |
| K17 | #63, #73, #88 | 576 CPU passed (PR #63) | **unverified**: GPU item 1 queued for Nov 7. `bea/nov7/logs/k17-default/` and `k17-off/` exist (e2e.out, unit.log) but I did not read them; `test_sm70_tp2_ar_fused_add_rms_norm` not in any junit I parsed | `bea/nov7/logs/k17-default/`, `bea/nov7/logs/k17-off/` | read those logs; bitwise equality and pass-firing |
| K18 | #58 | CPU (PR #58) | **failed**: `test_dynamic_prefill_trace_retains_m8_decode_dispatch` (sweep); isolated run `rc=1, 1 failed` | `bea/nov7/logs/gpu_sweep.log`, `bea/nov7/logs/k18iso.log`, `bea/nov7/iso.out` | fix the trace failure; the fp16 GEMM and acceptance checks are unverified |
| U3 | #43 | 505 CPU passed (PR #43) | n/a (CPU exit test) | PR #43 | none |
| U4 | #44 | 509 CPU passed (PR #44) | n/a (CPU exit test) | PR #44 | none |
| U5 | #60 | header-hash test (PR #73) | **not run**: R5 FULL-capture check with expandable segments has no evidence | none | run the R5 check |
| M8 | #65 | CPU (PR #65) | **not run**: the decode A/B that decides the default is not in any log. The qpn8 GPU suite (391 cases in the sweep) ran | `bea/nov7/logs/gpu_sweep.junit.xml` | run `m8_qpn8_ab.sh` |
| M9 | none | n/a | **not run**: "not in a PR until its hook point is named" | none | name the hook point; PR; GPU test |
| M10 | #66, #86 | 1296 CPU passed (PR #86) | **partial**: router and sampler GPU tests pass (see K13, K16). The e2e "greedy-identical with each switch off" is not seen | `bea/nov7/logs/` (above) | e2e identity runs |
| M10a | #86 | CPU (PR #86) | **partial**: router part of M10 | as above | as above |
| C1 | none | n/a | **not run** | none | side-by-side bench and quality run vs fork |
| C2 | #62 | `scripts/check-lmcache-cpu.sh`, 9 checks, passes on c4130 (PR #62 comment) | **partial**: GPU smoke (`import cupy,torch`) from Bea's window not in a log found; `kvtier_test.py` not seen | PR #62 comment | GPU smoke log; `kvtier_test.py` |
| C3 | none | n/a | **not run** | none | cutover |
| C4 | none | n/a | **not run** | none | archive the fork after C3 |

## Phase exits

**Phase 2 exit** (Flint and Qwen3.6/3.8 serve on c4130 TP2 with `--dtype half` on TRITON_ATTN; M5 and M6 quality suites pass)
- Serving on TRITON_ATTN at TP2: **unverified**. The Nov 7 runs served on Flash-V100 (`k14.out`).
- M5 quality: **failed in the calibration log** (T4a, T4b, T5 FAIL in `window.out`). A later run is reported in PR #86 but not found. The quality suite has no log.
- M6 quality and TP2 32 GB fit: **no evidence**.
- Verdict: no criterion met with evidence.

**Phase 3 exit** (Qwen3.6/3.8 decode and prefill at or above the fork's `bench_vllm.py` numbers, with Flash-V100, GDN FlashQLA and DFlash2 on)
- Fork baseline (Nov 4, `bea/baseline.md`): 176.8 tok/s at 4K, 162.8 at 32K.
- Ours: 175.7 tok/s at n=1 with K14 on (`k14.out`, Nov 7). Different conditions; `bench_vllm.py` not run as the exit defines it.
- DFlash2 spec decode ≥ fork (111 tok/s at 4K): **not run**.
- Verdict: **no evidence** that the criterion holds.

**Phase 4 exit** (Flint TP2 4K decode at least 170 tok/s with every fast path on; every kill switch at 0 greedy-identical)
- Throughput: 175.7 tok/s (n=1) is above 170, but K13, K16, K17 and M10 were not confirmed on in that run. **unverified**.
- Greedy-identical with each switch at 0: **no evidence** for any switch.
- Verdict: no criterion met with evidence.

**Phase 5 exit**: not started (C1–C4 not run).

## Nodes with no GPU exit-test evidence

Nothing below has a passing log for its exit test.

- **Not run:** K01, N2, M2, K04, K06, K07, K10, K15, M8, M9, M11, U5, C1, C3, C4
- **Failed:** K09 (opcheck FP8 dtype), K18 (trace test), M5 (T4a, T4b, T5 in the calibration log)
- **Unverified or partial:** U2, M7, M3, M4, M6, K03, K05, K08, K11, K13, K14, K16 (greedy identity missing), K17, M10, M10a, C2

**Passed (GPU exit):** N0 (Phase 1), K12 (suite), K16 (suite, greedy identity still missing).

## Not verified in this doc
- CPU counts beyond those stated in PR bodies; current main CI.
- Contents of `bea/nov7/logs/k17-*`, `k15` runs, and the PR #86 window log.
- The 9 `gpu_sweep` failures: 7 are `GPU memory not released after shutdown` between tests (test isolation, not a kernel result); the others are in the rows above.
