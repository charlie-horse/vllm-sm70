# Status: roadmap nodes vs evidence

Snapshot as of 2026-11-08, from merged PRs (#1-#99, with #92 and #97 merged after the Nov 7 sweep), PR comments, Nov 5-7 GPU logs on c4130 (read-only), and Quinn's Flint step 1 verification (`quinn-step1-verify.md`). `~/vllm-sm70-work/...` paths are on c4130.

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
| M7 | #22, #45, #101 | `test_m7_dflash_sm70.py` CPU | **passed**: e2e at #101 (merged `d83288f`) with the real draft, at 37c1b65: triton 3/3 and e2e 1/1. The Nov 8 run before the fix (`20261108T2035`) failed at `get_metrics()` ("Stat logging disabled"), a test-side failure. Earlier GPU sweep: `tests.models.test_m7_dflash_sm70` 3 pass, 1 skip (`test_e2e_real_draft_gpu`) | `bea/qwen-phase2/20261108T2257-m7-101/m7/{triton,e2e}.junit.xml`; `bea/qwen-phase2/20261108T2035/m7/e2e.junit.xml` (pre-fix failure); `bea/nov7/logs/gpu_sweep.junit.xml` | none for the exit |
| K03 | #26, #28 | `sm70/tests/ops` CPU (PR #26) | **partial**: `tests.ops` 102 passed in the sweep. The exit's named cases (TM vs `torch.matmul` at M in {1,2,8,64,512}; MoE vs reference loop) are not identified in the junit | `bea/nov7/logs/gpu_sweep.junit.xml` | confirm the named cases; MoE reference loop unverified |
| M3 | #24, #30, #42 | 25 CPU tests (PR #24) | **passed**: `tests.quant.test_m3_nvfp4_linear` 99 of 99 pass in the sweep. Test contents confirmed from the test code. | `bea/nov7/logs/gpu_sweep.junit.xml` | GEMV path depends on K11 (now passed) |
| M4 | #31, #34, #52, #72, #79, #97 | 22 CPU tests (PR #31) | **passed**: `tests.quant.test_m4_nvfp4_moe` 12 of 12 pass in the sweep. Test contents confirmed from the test code. #97 (compact slot groups, M*top_k <= 80; merged after the sweep) passes its GPU tests at a3dc3b2 (24/24 incl. all-padding routes; G2c teacher-forced gate), per Quinn's review | `bea/nov7/logs/gpu_sweep.junit.xml`; `nov8/logs/ab97b/test_m4.junit.xml` (24/24 at a3dc3b2, incl. all-padding routes); `ab97b/g2c.log` (teacher-forced gate pass); `nov8/logs/kai3/p100-m4.log` (28/28, #100 open) | none for the exit as stated. `dbd41ba` has the same M4 code and tests as a3dc3b2 (empty diff). Paths under `~/vllm-sm70-work/bea/`; from Quinn's PR #103 review, not re-read here |
| K04 | none | n/a | **not run** (optional) | none | AWQ not built |
| M5 | #36, #41, #52, #98 | 333 CPU passed (PR #36) | **partial**: the Nov 7 gate run passes t1, t2, t3, t4c, t4d, t4e, t5, t6, t7; t4b is skipped by design under Flash-V100 (T4e replaces it). The quality suite ran on `a3dc3b2` (1d): math 10/10, code 5/5, needle@32K 3/3, concurrency 20 clean, parse checks 4/4. LMCache prefix reuse is not shown (see Flint step 1). #98 (`glm47_sm70`) merged. The Nov 6 `window.out` shows t4a, t4b, t5 FAIL, superseded by the Nov 7 gate rulings | `bea/nov7/logs/m5full.out`, `bea/m5-cal-nov7/window-full-n7/`, `bea/nov7/logs/mutation.out`; quality: `bea/nov8/logs/S1d-main/`, `S1d-check5/`; superseded: `bea/m5-cal-nov7/window.out` | LMCache reuse proof (prefix-cache reset between sends; external-source tokens ≈ 11008) |
| M6 | #32, #41 | 25 CPU tests (PR #32) | **partial**: `tests.quant.test_m6_qwen_fp8` 20 of 20 unit pass in the sweep. The TP2 e2e (Qwen3.6-35B-A3B and 3.8-27B, within 32 GB/GPU) is not run | `bea/nov7/logs/gpu_sweep.junit.xml` | TP2 e2e and memory fit |
| K05 | #37 | CPU import shows 5 fv100 ops (PR #37) | **passed**: the exit's two tests were ported under new names and pass in the sweep: `test_fv100_prefill_sinks_gpu.py` 39/39 (port of `test_sm70_flash_v100_sinks_prefill`) and `test_fv100_d256_prefill_gpu.py` 24/24 (port of `test_sm70_flash_v100_splitkv_n`) | `bea/nov7/logs/gpu_sweep.junit.xml` | none for the exit |
| K06 | #39, #51, #83, #84 | 1099 CPU passed at e50db90 (PR #84) | **not run**: the exit (Flint TP2 4K greedy-identical over 256 tokens on 5 prompts; decode tok/s >= fork) has no evidence | `bea/nov7/logs/k14.out` (throughput only) | greedy-identity run; tok/s vs fork |
| K07 | #51 | CPU (PR #51) | **passed**: Bea's D2 A/B on main 6355538 (CUSTOM backend, same prompts, k=7). `accept_len` with Flash-V100 spec verify on vs `VLLM_SM70_FLASH_V100_SPEC=0`: row 2 accept_len −1.43%, row 4 +0.30% (recomputed from results.json; the note gives 3.56 vs 3.61 and 3.94 vs 3.93), both within the 2% bar; all four runs pass quality. INFO: row 2 decode-phase accept_len is −2.16%; K07 logs no route line, because the off arm is set by the script env. Exit (Charlie, 2026-11-08) per `sm70/scripts/qwen_phase2/qwen_matrix.sh` rows 2 and 4. `tests.attention.test_fv100_spec_gpu` 36 of 36 pass, smallq 12/12 | `bea/qwen-phase2/20261108T2035/d2-{on,off}/row{2,4}/results.json` (build k3jsgqgg = main 6355538); `side-session-notes/local_f53bb309…/bea-qwen-blocks.md` §D2; `bea/nov7/logs/gpu_sweep.junit.xml` (fv100 spec tests) | none for the exit |
| K08 | #33 | `sm70/tests/gdn` CPU | **partial**: `tests.gdn.test_k08_gdn_gpu` 19 pass (the rel-L2 cases pass), 2 skip (they need `SM70_TEST_TRITON_CHUNK=1`) | `bea/nov7/logs/gpu_sweep.junit.xml` | run the 2 skipped cases with `SM70_TEST_TRITON_CHUNK=1` |
| M11 | #96 (merged, `a700639`) | n/a | **passed**: exit test `test_m11_gdn_layer_gpu` 5 passed (repo-main @6355538, build k3jsgqgg). INFO: C3 e2e at util 0.92 (the off arm cannot start at 0.85), quality pass in both arms, FlashQLA prefill line 60× on / 0× off. K08 decode4k ties switch-off (89.5/89.6 tok/s row 1, 22.9/22.9 row 3), so there is no batch-1 decode speedup; none is claimed | `bea/qwen-phase2/20261108T2035/C2-m11-gdn-layer.log`; INFO: `bea/qwen-phase2/20261108T2257-m7-101/c3-{on2,off2}/row{1,3}` | none for the exit as stated |
| K09 | #50, #65, #95 | CPU (PR #50) | **partial**: Nov 7 sweep had 1 fail, `test_opcheck` (`mul_cuda` not implemented for `Float8_e4m3fn`; test-side). #95 (merged) fixes it; rerun `-k opcheck`: 1 passed (`nov8/logs/kai3/p95.log`). The max-abs-rel < 1e-2 check is not confirmed | `bea/nov7/logs/gpu_sweep.log`, `bea/nov7/logs/k09iso.log`, `bea/nov8/logs/kai3/p95.log` | confirm the max-abs-rel < 1e-2 check; re-run the full exit |
| K10 | none | n/a | **not run** (opt) | none | needs a Nix TileLang |
| K11 | #53 | CPU (PR #53) | **passed**: `tests.ops.test_k11_nvfp4_tm_gemv_gpu` 28 of 28 pass in the sweep. The exit is "fork tests listed in the carry map"; that mapping is not done | `bea/nov7/logs/gpu_sweep.junit.xml` | map the carry-map tests |
| K12 | #55 | CPU 565 passed (PR #55) | **passed**: `tests.flint_moe` 15 passed, 0 failed (Nov 7 01:12). Exit is "fork tests in the carry map", not mapped one-to-one | `bea/nov7/logs/flint_moe.junit.xml` | map the carry-map tests |
| K13 | #57, #86 | 1296 CPU passed at 8480748 (PR #86) | **partial**: `tests.router.test_k13_router_gpu` 12 of 12 pass in the sweep. "Greedy-identical with the switch off" not seen | `bea/nov7/logs/gpu_sweep.junit.xml` | greedy-identity with `VLLM_SM70_FLINT_FUSED_ROUTER=0` |
| K14 | #56, #79, #97 | 535 CPU passed (PR #56) | **partial**: one-launch sink decode is live. Nov 7, aggregate tok/s (mean of two reps), default arm: 175.5 at n=1, 262.3 at n=2, 37.5 at n=4, 56.0 at n=8. Switch-off arm (`VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0`): 121.0, 196.3, 36.6, 55.0. The n=4 drop was in both arms, so it was not K14. #97 (merged after the sweep) is per-stream: with compact on, n=3/4/8 = 72.5/59.1/33.3 tok/s/stream vs 10.3/9.4/7.0 off (`nov8/logs/AB97b-A-default/conc.out`, `AB97b-B-compact0/conc.out`). Aggregate at n=4 is 233.9 vs 37.5. The regression test `test_decode_sinks_descending_capture_regression` is reported 2/2 in the sweep (per-file parse by Quinn) | `bea/nov7/logs/k14.out`, `bea/nov7/logs/gpu_sweep.junit.xml`, `bea/nov8/logs/AB97b-*/conc.out` | n=20 coherence sweep |
| K15 | #54 | 533 CPU passed (PR #54) | **passed**: `tests.flint.test_k15_qk_norm_rope_kv_gpu` 50 of 50 pass in the sweep. Exit is "vs unfused ops" | `bea/nov7/logs/gpu_sweep.junit.xml` | none for the exit |
| K16 | #59, #66 | 1307 CPU passed at 0291f61 (PR #66) | **passed** for the K16 suite: 66 of 66 (`k16_full.junit.xml`). M10 sampler: G1 and G3 pass; G2 xfail; G4 XPASS (`test_m10_flint_sampler.junit.xml`). "Greedy-identical with switch off" not shown | `bea/nov7/logs/k16_full.junit.xml`, `bea/nov7/logs/test_m10_flint_sampler.junit.xml` | greedy identity with `VLLM_SM70_FLINT_SAMPLER=0`; decide whether the G4 XPASS needs its marker updated |
| K17 | #63, #73, #88 | 576 CPU passed (PR #63) | **partial**: `tests.distributed.test_sm70_tp2_ar_fused_add_rms_norm` 2 of 2 pass (bitwise test). Pass-firing on `bfbbf360` (#88) is shown by the S1b serve log ("fused 2 pairs in the first compiled graph", `bea/nov8/logs/S1b-bfbbf36-default/serve.log`). Switch-off greedy identity is 5/5 in Quinn's 1c matrix (AR_RMSNORM, greedy-identical per the criterion). Throughput A/B only: median of 3 runs, `k17-default/e2e.out` 175.7 tok/s vs `k17-off/e2e.out` 174.7 tok/s at ctx 4096 greedy (JSON write failed; numbers from the printed line) | `bea/nov7/logs/gpu_sweep.junit.xml`, `bea/nov7/logs/k17-default/`, `bea/nov7/logs/k17-off/` | none for the exit; the `k17-*/` throughput runs are not an identity test |
| K18 | #58, #94 | CPU (PR #58) | **partial**: Nov 7 sweep had 1 fail, `test_dynamic_prefill_trace_retains_m8_decode_dispatch` (test-side). #94 (merged) fixes it; rerun: 1 passed, one recompile, rows=1 size-1 guard (`nov8/logs/kai3/p94.log`). The fp16 GEMM and acceptance checks are unverified | `bea/nov7/logs/gpu_sweep.log`, `bea/nov7/logs/k18iso.log`, `bea/nov8/logs/kai3/p94.log` | the fp16 GEMM and acceptance checks |
| U3 | #43 | 505 CPU passed (PR #43) | n/a (CPU exit test) | PR #43 | none |
| U4 | #44 | 509 CPU passed (PR #44) | n/a (CPU exit test) | PR #44 | none |
| U5 | #60 | header-hash test (PR #73) | **not run**: R5 FULL-capture check with expandable segments has no evidence | none | run the R5 check |
| M8 | #65, #102 (open warning PR) | CPU (PR #65) | **closed by decision (Astro, 2026-11-08): wired, no-op at prod defaults; M32 opt-in not recommended.** A/B on main 6355538 (Bea, opt-in on): QPN8/M6 = 0.994 / 0.985 / 0.733 / 0.669 at c=1..4; greedy 5/5. Shipped default (max_m = 8) admits no layer, so prod equals M6. Fork-parity copy run skipped by decision. The qpn8 GPU suite (390 of 391 pass) ran in Nov 7 | `bea/qwen-phase2/20261108T2035/m8/` (`summary.txt`, `ab_summary.json`), `B3-m8-ab.log`; verdict table in `side-session-notes/local_f53bb309…/mara-m8-ab-verdict.md`; `bea/nov7/logs/gpu_sweep.junit.xml` (qpn8 tests) | M32 warning lands in #102 (open, head 4551890); no further M8 work |
| M9 | #91, #92 | #91 names the hook points (U1 via K18, PluggableLayer OOT), 0 upstream lines; #92 (M9a) merged at 2fe63e6, channel-FP8 M8 projections via U1/K18 | **partial**: M9a GPU op check passed at 0ecee81: rel_l2 0.0259–0.0269 for 5 shapes × M 8..32 (`nov8/logs/kai3/p92.log`). The check is an ad-hoc snippet, not a committed GPU test; `2fe63e6` has the same ops code. M9b not started | `bea/nov8/logs/kai3/p92.log` | commit the M9a GPU test; M9b; M9b's gate is independent of M8 (Mara) |
| M10 | #66, #86 | 1296 CPU passed (PR #86) | **partial**: router and sampler GPU tests pass in the isolated runs (see K13, K16). The e2e "greedy-identical with each switch off" is not seen | `bea/nov7/logs/` (above) | e2e identity runs |
| M10a | #86 | CPU (PR #86) | **partial**: router part of M10 | as above | as above |
| C1 | none | n/a | **not run** | none | side-by-side bench and quality run vs fork |
| C2 | #62 | `scripts/check-lmcache-cpu.sh`, 9 checks, passes on c4130 (PR #62 comment) | **partial**: GPU smoke (`import cupy,torch`) from Bea's window not in a log found; `kvtier_test.py` not seen | PR #62 comment | GPU smoke log; `kvtier_test.py` |
| C3 | none | n/a | **not run** | none | cutover |
| C4 | none | n/a | **not run** | none | archive the fork after C3 |

**Sweep failures (9):** 7 are `GPU memory not released after shutdown` between tests (test isolation, not a kernel result). The other two are K09 (opcheck) and K18 (trace); both reruns pass after #95 and #94 (see rows).

## Phase exits

**Phase 2 exit** (Flint and Qwen3.6/3.8 serve on c4130 TP2 with `--dtype half` on TRITON_ATTN; M5 and M6 quality suites pass)
- Serving on TRITON_ATTN at TP2: **unverified**. The Nov 7 runs served on Flash-V100 (`k14.out`).
- M5: numerical gates pass in `m5full` (Nov 7). The quality suite ran on `a3dc3b2` (5 of 6 checks confirmed; LMCache reuse unproven). **partial**.
- M6: unit 20/20; TP2 e2e and quality suite not run. **partial**.
- Verdict: no criterion met in full.

**Phase 3 exit** (Qwen3.6/3.8 4K decode at or above the documented 1Cat 1.5.1 numbers, as a 3-run median, with Flash-V100, GDN FlashQLA and DFlash2 on; Charlie, 2026-11-08: no fork or 1Cat baseline runs)
- Bar: Qwen3.6 row 1 (no DFlash2) >= 66 tok/s; row 2 (with DFlash2) >= 111 tok/s; Qwen3.8 row 4 >= 80 tok/s (floor). Row 3 (Qwen3.8 without DFlash2) has no documented number and is record-only. Qwen3.8 cold 100K TTFT <= 66 s is also part of the bar.
- Qwen decode and prefill numbers: **none**. The Nov 7 throughput numbers are Flint, not Qwen, so they do not count toward this exit. Block F (the speed run) is on hold behind the 1Cat parity slot (Mara, `mara-1cat-parity.md`).
- Verdict: **no evidence** for the criterion.

**Phase 4 exit** (Flint TP2 4K decode at least 170 tok/s with every fast path on; kill switches per Charlie's 2026-11-08 criterion, see below)
- Kill-switch criterion: Charlie, in the Astro lead session, 2026-11-08, choosing "Teacher-forced envelope". The 5 kernel-changing Flint switches (`MODELOPT_NVFP4_GEMV`, `FLINT_MOE_QPN_DECODE`, `FLINT_QK_ROPE_KV`, `FLINT_FUSED_ROUTER`, `FLINT_ATTN_FUSED_DECODE`) pass if switch=0 vs default, teacher-forced on the M5 sequences, stays within the fork envelope (as #86 for K13). `MOE_COMPACT` (Astro, 2026-11-08) joins the teacher-forced group, since it changes kernels and #97 was gated teacher-forced (G2c, #100). `SAMPLER` and `AR_RMSNORM` stay greedy-identical. `docs/roadmap.md` Phase 4 exit and Settled list updated to match in this PR.
- Throughput: **met**. 4K default 173.3 tok/s and 4K greedy 176.9 tok/s (Quinn's 1b recompute) are above 170. The S1b serve log (`bea/nov8/logs/S1b-bfbbf36-default/serve.log`) shows every fast path on: fused router on 47 layers, AR+RMSNorm fusion registered and "fused 2 pairs", sampler enabled, K12 admitted, GEMV enabled, fused q/k/RoPE, FLASH_ATTN_V100. 1b ran on `hmrk01b1` (bfbbf360) only; `l56dj370` is 1d.
- Fork 32K comparison (no roadmap gate): fork 162.8 default / 162.7 greedy (Bea, Nov 5); ours 156.3 default (−4.0%) and 148.1 greedy (−9.0%). Greedy is slower than default on our build, unexplained; a 5-run rerun is open.
- Kill switches at 0 (greedy counts from Quinn's 1c, not the criterion Charlie chose): AR_RMSNORM 5/5 identical, and its fusion is logged in base and absent in the arm. SAMPLER 5/5 is vacuous (greedy never reaches the sampler). The other five diverge after a shared prefix.
- Verdict: throughput met; kill-switch criterion not yet met (teacher-forced evidence pending for three switches; QK_ROPE_KV and FUSED_ROUTER have on/off tests, whose latest GPU logs are not checked here).

**Flint step 1 (Quinn's verification, `quinn-step1-verify.md`, 2026-11-08)**
- 1a Flash-V100 vs TRITON_ATTN: **confirmed**. Backends correct; 18 of 37 greedy outputs identical; runs repeat.
- 1b single-stream decode: **confirmed** (numbers above). Label nit: the e2e JSON carries the hard-coded fork label `bea-baseline-e66530b`.
- 1c kill-switch matrix: **arms and numbers confirmed**; criterion owner is Charlie (see Phase 4). Greedy identity per switch: GEMV 1/5, QPN 1/5, QK_ROPE 1/5, ROUTER 1/5, SAMPLER 5/5 (vacuous: greedy never reaches the sampler), ATTN 2/5, AR_RMSNORM 5/5 (fusion logged in base, absent in the arm). Model-level teacher-forced evidence is missing for MODELOPT_NVFP4_GEMV, FLINT_MOE_QPN_DECODE and FLINT_ATTN_FUSED_DECODE. `ATTN_FUSED_DECODE` logs nothing, so its route is not confirmed by a log line.
- 1d M5 quality on `a3dc3b2`: 5 of 6 **confirmed** (math 10/10, code 5/5, needle@32K 3/3, concurrency 20 clean, parse checks 4/4). **LMCache prefix reuse not shown**: the second-send speedup came from vLLM's GPU prefix cache, not LMCache retrieve. Needs a prefix-cache reset between sends and external-source token counts.
- Still pending: 1c for the three uncovered switches (teacher-forced), 1d LMCache reuse. Bea's cutover finding stands (prod `vllm-lmcache.service` cannot register the new build's KV).

**Phase 5 exit**: not started (C1-C4 not run).

## Nodes with no GPU exit-test evidence

- **Not run:** K01, N2, M2, K04, K06, K10, U5, C1, C3, C4 (M8 closed by decision, see table)
- **Failed in Nov 7 sweep, fixed by merged test PRs and rerun passing:** K09 (#95), K18 (#94); the exit parts that remain are in their rows
- **Unverified or partial:** U2, M5 (LMCache reuse), M6 (TP2 e2e), M9 (M9a test, M9b), K03, K08, K09 (max-abs-rel), K13 (greedy identity), K14 (n=20 sweep), K16 (greedy identity), K17 (pass-firing, switch-off identity), K18 (fp16 GEMM, acceptance), M10, M10a, C2

**Passed (GPU exit):** N0 (Phase 1), K05, K07, K11 (suite), K12 (suite), K15, K16 (suite), M3, M4, M7, M11.

## Not verified in this doc
- CPU counts beyond those stated in PR bodies; current main CI.
- The Nov 6 `window.out` is treated as superseded by the Nov 7 `m5full` run, per the Nov 7 gate rulings.
- `gpu_sweep` ran from `c7fd4ee`; later merges may change its results.
