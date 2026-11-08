# Status: roadmap nodes vs evidence

Snapshot as of 2026-11-08, from merged PRs (#1-#88), PR comments, and Nov 5-7 GPU logs on c4130 (read-only). `~/vllm-sm70-work/...` paths are on c4130.

**Status terms**
- **passed**: a PR, log or junit file shows the exit test (or the named part of it) passing.
- **partial**: some of the exit test has evidence; the gap is in "what's left".
- **unverified**: a claim exists (for example, a PR body says "passed"), but no log was found, or the exit names a test I could not identify.
- **not run**: no GPU evidence in the PRs or on c4130.
- **failed**: a logged failure of the exit test or its direct proxy.
- "CPU evidence" is the count stated on the node's merged PR. CPU tests were not re-run for this doc.
- Per-file GPU counts come from `bea/nov7/logs/gpu_sweep.junit.xml`, parsed per testcase. The sweep ran from source `c7fd4ee`, which has the same tree as `e0c8146`. Later merges (for example #88, `bfbbf360`) may not be covered.
- K14 throughput numbers are the mean of two reps (`k14.out`). K17 numbers are the median of 3 runs (`k17-*/e2e.out`).

## Node table

Roadmap order. Each node appears once.

| node | PR(s) | CPU evidence | GPU exit test | evidence link or path | what's left |
|---|---|---|---|---|---|
| N0 | #2 | nix eval in CI | **passed**: Phase 1 gate, TP1 and TP2, TRITON_ATTN, fp16 | PR #2 comment (Bea, Nov 5); `~/vllm-sm70-work/bea/phase1/{tp1,tp2}/` | none for the exit |
| N1 | #6, #7, #13 | CI unit + nix eval, green when merged | n/a (CI node) | PR #6 | main CI not re-checked here; PR #4/#10/#22/#24 record a red `test_no_vllm_envs` on main, fix not confirmed |
| K01 | #16 | `sm70/tests/plugin`: 31 passed, 22 skipped without vllm; 53 with (PR #16) | **not run**: no `_sm70_C` / `_C_stable_libtorch` co-import test found | none | run the co-import check on c4130 |
| N2 | #27, #61 | r440 `.#sm70-ext` builds (comments on #50-#63) | **unverified**: c4130 `import vllm_sm70` + `torch.ops._sm70_C` not in any log found | PR #26 comment (K03 build, 38 ops, CPU import) | c4130 import check |
| U1 | #16 | CPU tests in PR #16 | n/a (CPU exit test) | PR #16 | CPU exit not re-run here |
| M1 | #8 | CPU defaults and kill-switch tests (PR #8) | n/a (CPU exit test) | PR #8 | none |
| U2 | #33 | `sm70/tests/gdn` CPU | **unverified**: "upstream GDN tests pass on sm70" not identified. `tests.gdn` (repo's own GDN tests) 19 pass, 2 skip in the sweep | `bea/nov7/logs/gpu_sweep.junit.xml` | name the upstream GDN test; run it |
| K02 | #12, #78 | `check_turbomind_provenance.py --fork` OK at 7958a70 (PR #78) | n/a (build + provenance node) | PR #78 comment | none |
| M2 | #9 | mocked cc tests (PR #9) | **not run**: real Flint and Qwen config admission not tested on GPU | none | run real checkpoint configs through the capability check |
| M7 | #22, #45 | `test_m7_dflash_sm70.py` CPU | **partial**: `tests.models.test_m7_dflash_sm70` 3 pass (`test_triton_matches_reference_gpu`, two bitwise), 1 skip (`test_e2e_real_draft_gpu`, the exit's fp16 draft forward). PR #34's "Triton 3/3" is not the same run | `bea/nov7/logs/gpu_sweep.junit.xml` | the skipped e2e draft forward; acceptance-rate e2e (needs M6 e2e) |
| K03 | #26, #28 | `sm70/tests/ops` CPU (PR #26) | **partial**: `tests.ops` 102 passed in the sweep. The exit's named cases (TM vs `torch.matmul` at M in {1,2,8,64,512}; MoE vs reference loop) are not identified in the junit | `bea/nov7/logs/gpu_sweep.junit.xml` | confirm the named cases; MoE reference loop unverified |
| M3 | #24, #30, #42 | 25 CPU tests (PR #24) | **passed**: `tests.quant.test_m3_nvfp4_linear` 99 of 99 pass in the sweep. Test contents confirmed from the test code. | `bea/nov7/logs/gpu_sweep.junit.xml` | GEMV path depends on K11 (now passed) |
| M4 | #31, #34, #52, #72, #79 | 22 CPU tests (PR #31) | **passed**: `tests.quant.test_m4_nvfp4_moe` 12 of 12 pass in the sweep. Test contents confirmed from the test code. | `bea/nov7/logs/gpu_sweep.junit.xml` | none for the exit as stated |
| K04 | none | n/a | **not run** (optional) | none | AWQ not built |
| M5 | #36, #41, #52 | 333 CPU passed (PR #36) | **partial**: the Nov 7 gate run passes t1, t2, t3, t4c, t4d, t4e, t5, t6, t7; t4b is skipped by design under Flash-V100 (T4e replaces it). The roadmap exit is the quality suite (math 10/10, code 5/5, needle@32K 3/3, gibberish at c20, glm47, LMCache), which has no log. The Nov 6 `window.out` shows t4a, t4b, t5 FAIL, but it predates the Nov 7 gate rulings and is superseded | `bea/nov7/logs/m5full.out`, `bea/m5-cal-nov7/window-full-n7/`, `bea/nov7/logs/mutation.out` (mutation check); superseded: `bea/m5-cal-nov7/window.out` | run and log the quality suite |
| M6 | #32, #41 | 25 CPU tests (PR #32) | **partial**: `tests.quant.test_m6_qwen_fp8` 20 of 20 unit pass in the sweep. The TP2 e2e (Qwen3.6-35B-A3B and 3.8-27B, within 32 GB/GPU) is not run | `bea/nov7/logs/gpu_sweep.junit.xml` | TP2 e2e and memory fit |
| K05 | #37 | CPU import shows 5 fv100 ops (PR #37) | **passed**: the exit's two tests were ported under new names and pass in the sweep: `test_fv100_prefill_sinks_gpu.py` 39/39 (port of `test_sm70_flash_v100_sinks_prefill`) and `test_fv100_d256_prefill_gpu.py` 24/24 (port of `test_sm70_flash_v100_splitkv_n`) | `bea/nov7/logs/gpu_sweep.junit.xml` | none for the exit |
| K06 | #39, #51, #83, #84 | 1099 CPU passed at e50db90 (PR #84) | **not run**: the exit (Flint TP2 4K greedy-identical over 256 tokens on 5 prompts; decode tok/s >= fork) has no evidence | `bea/nov7/logs/k14.out` (throughput only) | greedy-identity run; tok/s vs fork |
| K07 | #51 | CPU (PR #51) | **partial**: `tests.attention.test_fv100_spec_gpu` 36 of 36 pass, including smallq 12/12. DFlash2 acceptance A/B not run. Pass (Charlie, 2026-11-08) is `accept_len` with Flash-V100 spec verify on within 2% (relative) of `VLLM_SM70_FLASH_V100_SPEC=0`, same prompts, rows 2 and 4 of `sm70/scripts/qwen_phase2/qwen_matrix.sh` | `bea/nov7/logs/gpu_sweep.junit.xml` | run the A/B with `qwen_matrix.sh 2` and `qwen_matrix.sh 4` |
| K08 | #33 | `sm70/tests/gdn` CPU | **partial**: `tests.gdn.test_k08_gdn_gpu` 19 pass (the rel-L2 cases pass), 2 skip (they need `SM70_TEST_TRITON_CHUNK=1`) | `bea/nov7/logs/gpu_sweep.junit.xml` | run the 2 skipped cases with `SM70_TEST_TRITON_CHUNK=1` |
| M11 | none | n/a | **not run**: no PR | none | write the layer subclass; GPU check vs upstream |
| K09 | #50, #65 | CPU (PR #50) | **failed (probably test-side, not confirmed)**: `tests.qpn8.test_k09_qpn8_gpu` 390 pass, 1 fail (`test_opcheck`: `mul_cuda` not implemented for `Float8_e4m3fn`); isolated run `rc=1, 1 failed`. The max-abs-rel < 1e-2 check is not confirmed | `bea/nov7/logs/gpu_sweep.log`, `bea/nov7/logs/k09iso.log`, `bea/nov7/iso.out` | fix the FP8 dtype in opcheck (test side is the likely cause); re-run the exit |
| K10 | none | n/a | **not run** (opt) | none | needs a Nix TileLang |
| K11 | #53 | CPU (PR #53) | **passed**: `tests.ops.test_k11_nvfp4_tm_gemv_gpu` 28 of 28 pass in the sweep. The exit is "fork tests listed in the carry map"; that mapping is not done | `bea/nov7/logs/gpu_sweep.junit.xml` | map the carry-map tests |
| K12 | #55 | CPU 565 passed (PR #55) | **passed**: `tests.flint_moe` 15 passed, 0 failed (Nov 7 01:12). Exit is "fork tests in the carry map", not mapped one-to-one | `bea/nov7/logs/flint_moe.junit.xml` | map the carry-map tests |
| K13 | #57, #86 | 1296 CPU passed at 8480748 (PR #86) | **partial**: `tests.router.test_k13_router_gpu` 12 of 12 pass in the sweep. "Greedy-identical with the switch off" not seen | `bea/nov7/logs/gpu_sweep.junit.xml` | greedy-identity with `VLLM_SM70_FLINT_FUSED_ROUTER=0` |
| K14 | #56, #79 | 535 CPU passed (PR #56) | **partial**: one-launch sink decode is live. Mean of two reps, default arm: 175.5 tok/s at n=1, 262.3 at n=2, 37.5 at n=4, 56.0 at n=8. Switch-off arm (`VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0`): 121.0, 196.3, 36.6, 55.0. The n=4 drop is in both arms, so it is not K14. The regression test `test_decode_sinks_descending_capture_regression` is reported 2/2 in the sweep (per-file parse by Quinn) | `bea/nov7/logs/k14.out`, `bea/nov7/logs/gpu_sweep.junit.xml` | root-cause the n=4 drop (both arms); n=20 coherence sweep |
| K15 | #54 | 533 CPU passed (PR #54) | **passed**: `tests.flint.test_k15_qk_norm_rope_kv_gpu` 50 of 50 pass in the sweep. Exit is "vs unfused ops" | `bea/nov7/logs/gpu_sweep.junit.xml` | none for the exit |
| K16 | #59, #66 | 1307 CPU passed at 0291f61 (PR #66) | **passed** for the K16 suite: 66 of 66 (`k16_full.junit.xml`). M10 sampler: G1 and G3 pass; G2 xfail; G4 XPASS (`test_m10_flint_sampler.junit.xml`). "Greedy-identical with switch off" not shown | `bea/nov7/logs/k16_full.junit.xml`, `bea/nov7/logs/test_m10_flint_sampler.junit.xml` | greedy identity with `VLLM_SM70_FLINT_SAMPLER=0`; decide whether the G4 XPASS needs its marker updated |
| K17 | #63, #73, #88 | 576 CPU passed (PR #63) | **partial**: `tests.distributed.test_sm70_tp2_ar_fused_add_rms_norm` 2 of 2 pass (bitwise test). Pass-firing on `bfbbf360` (#88) and switch-off greedy identity are missing. Throughput A/B only: median of 3 runs, `k17-default/e2e.out` 175.7 tok/s vs `k17-off/e2e.out` 174.7 tok/s at ctx 4096 greedy (JSON write failed; numbers from the printed line) | `bea/nov7/logs/gpu_sweep.junit.xml`, `bea/nov7/logs/k17-default/`, `bea/nov7/logs/k17-off/` | pass-firing log on `bfbbf360`; switch-off greedy identity |
| K18 | #58 | CPU (PR #58) | **failed (probably test-side, not confirmed)**: `tests.dflash2.test_k18_dflash2_gpu` 29 pass, 1 fail (`test_dynamic_prefill_trace_retains_m8_decode_dispatch`); isolated run `rc=1, 1 failed` | `bea/nov7/logs/gpu_sweep.log`, `bea/nov7/logs/k18iso.log`, `bea/nov7/iso.out` | fix or confirm the trace test; the fp16 GEMM and acceptance checks are unverified |
| U3 | #43 | 505 CPU passed (PR #43) | n/a (CPU exit test) | PR #43 | none |
| U4 | #44 | 509 CPU passed (PR #44) | n/a (CPU exit test) | PR #44 | none |
| U5 | #60 | header-hash test (PR #73) | **not run**: R5 FULL-capture check with expandable segments has no evidence | none | run the R5 check |
| M8 | #65 | CPU (PR #65) | **not run**: the decode A/B that decides the default is not in any log. The qpn8 GPU suite (390 of 391 pass) ran | `bea/nov7/logs/gpu_sweep.junit.xml` (qpn8 tests) | run `m8_qpn8_ab.sh` |
| M9 | none | n/a | **not run**: "not in a PR until its hook point is named" | none | name the hook point; PR; GPU test |
| M10 | #66, #86 | 1296 CPU passed (PR #86) | **partial**: router and sampler GPU tests pass in the isolated runs (see K13, K16). The e2e "greedy-identical with each switch off" is not seen | `bea/nov7/logs/` (above) | e2e identity runs |
| M10a | #86 | CPU (PR #86) | **partial**: router part of M10 | as above | as above |
| C1 | none | n/a | **not run** | none | side-by-side bench and quality run vs fork |
| C2 | #62 | `scripts/check-lmcache-cpu.sh`, 9 checks, passes on c4130 (PR #62 comment) | **partial**: GPU smoke (`import cupy,torch`) from Bea's window not in a log found; `kvtier_test.py` not seen | PR #62 comment | GPU smoke log; `kvtier_test.py` |
| C3 | none | n/a | **not run** | none | cutover |
| C4 | none | n/a | **not run** | none | archive the fork after C3 |

**Sweep failures (9):** 7 are `GPU memory not released after shutdown` between tests (test isolation, not a kernel result). The other two are K09 (opcheck) and K18 (trace), in the rows above.

## Phase exits

**Phase 2 exit** (Flint and Qwen3.6/3.8 serve on c4130 TP2 with `--dtype half` on TRITON_ATTN; M5 and M6 quality suites pass)
- Serving on TRITON_ATTN at TP2: **unverified**. The Nov 7 runs served on Flash-V100 (`k14.out`).
- M5: numerical gates pass in `m5full` (Nov 7). The quality suite has no log. **partial**.
- M6: unit 20/20; TP2 e2e and quality suite not run. **partial**.
- Verdict: no criterion met in full.

**Phase 3 exit** (Qwen3.6/3.8 4K decode at or above the documented 1Cat 1.5.1 numbers, as a 3-run median, with Flash-V100, GDN FlashQLA and DFlash2 on; Charlie, 2026-11-08: no fork or 1Cat baseline runs)
- Bar: Qwen3.6 row 1 (no DFlash2) >= 66 tok/s; row 2 (with DFlash2) >= 111 tok/s; Qwen3.8 row 4 >= 80 tok/s (floor). Row 3 (Qwen3.8 without DFlash2) has no documented number and is record-only. Qwen3.8 cold 100K TTFT <= 66 s is also part of the bar.
- Qwen decode and prefill numbers: **none**. The Nov 7 throughput numbers are Flint, not Qwen, so they do not count toward this exit.
- Verdict: **no evidence** for the criterion.

**Phase 4 exit** (Flint TP2 4K decode at least 170 tok/s with every fast path on; every kill switch at 0 greedy-identical)
- Throughput: 175.5 tok/s mean at n=1 with K14 on (`k14.out`) is above 170, but K13, K16, K17 and M10 were not all confirmed on in that run. **unverified**.
- Greedy-identical with each switch at 0: **no evidence** for any switch.
- Verdict: no criterion met with evidence.

**Phase 5 exit**: not started (C1-C4 not run).

## Nodes with no GPU exit-test evidence

- **Not run:** K01, N2, M2, K04, K06, K10, M8, M9, M11, U5, C1, C3, C4
- **Failed:** K09 (opcheck FP8 dtype; probably test-side), K18 (trace test; probably test-side)
- **Unverified or partial:** U2, M5 (quality suite missing), M6 (TP2 e2e), M7, K03, K07 (acceptance), K08, K13 (greedy identity), K14 (n=4 drop; n=20 sweep), K16 (greedy identity), K17 (pass-firing, switch-off identity), M10, M10a, C2

**Passed (GPU exit):** N0 (Phase 1), K05, K11 (suite), K12 (suite), K15, K16 (suite), M3, M4.

## Not verified in this doc
- CPU counts beyond those stated in PR bodies; current main CI.
- The Nov 6 `window.out` is treated as superseded by the Nov 7 `m5full` run, per the Nov 7 gate rulings.
- `gpu_sweep` ran from `c7fd4ee`; later merges may change its results.
