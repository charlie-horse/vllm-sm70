# Releases: the `latest` branch

Hosts consume this flake as `github:charlie-horse/vllm-sm70/latest`, never `main`. The host flake re-locks its inputs daily, so whatever `latest` points at reaches c4130 on its next switch.

## Moving `latest`

`latest` only fast-forwards, to a commit that is already on `main`. Penny (release) moves it, and only once all four of these hold for **that exact commit**:

1. `main` CI is green on it (patch series, both Python unit jobs, Nix eval).
2. C1 passes on it: the side-by-side bench and quality run against the fork on c4130 (see `docs/roadmap.md`, Phase 5).
3. Bea's GPU gate passed on it, in a downtime window (22:00–06:00 ET, via flake#5). "Bea's GPU gate" is the following list, run on that exact commit at prod args:
   - `pytest -m gpu` passes.
   - M5 T1–T7 pass (the T-gates are defined in `docs/m5-calibration-nov7.md`).
   - The 6-switch envelope: each of the 6 kernel-changing switches (`MODELOPT_NVFP4_GEMV`, `FLINT_MOE_QPN_DECODE`, `FLINT_QK_ROPE_KV`, `FLINT_FUSED_ROUTER`, `FLINT_ATTN_FUSED_DECODE`, `FLINT_MOE_COMPACT`) stays within the fork envelope when set to 0, teacher-forced.
   - 4K decode ≥ 170 tok/s, and 32K decode measured at prod args, recorded against C1's fork numbers at prod args (`C1-fork/e2e.out`): 4K 177.8 tok/s, 32K 163.0 default and 165.6 greedy. No gate at 32K yet.
   - Precondition: no leftover servers on the test ports. Check `ss -ltnp` for the vLLM and LMCache ports before the gate. After start, confirm the fresh `lmcache-server.log` shows the server bound (C1R's LMCache once failed to bind and used a stale server).
   - The LMCache reuse check (`docs/cutover-runbook.md` section 4.3) and the check5 and quality probes.
4. Astro says OK.

A gate run on another commit doesn't count, even if only docs changed in between. Never force-push `latest` or move it backwards. To roll back, revert on `main`, run the gates again, and fast-forward.

To move it:

```
git fetch origin main latest
git merge-base --is-ancestor origin/latest <commit>   # must succeed: fast-forward only
git push origin <commit>:refs/heads/latest
```

Record each move in the table below, in the same PR or straight after it.

## History

| date (UTC) | commit | CI | C1 | GPU gate | OK |
|---|---|---|---|---|---|
| 2026-11-05 | `3869cd6` | green | – | – | Astro (bootstrap only) |
| 2026-11-12 | `cef0f9e` | green (four jobs) | PASS (2026-11-09, after the `glm47_sm70` rerun; review 5833861019) | PASS with accepted deviations (see below; Quinn's verdict) | granted by Charlie; Astro's OK (gate 4), 2026-11-12 |

**Gate 3 record, `cef0f9e` (build `aczb9zwk`), 2026-11-12.** Accepted deviations, all under Quinn's verdict:

1. **`pytest -m gpu` ran per file, not as one process.** The single process hung at 662 of 1412 (a harness problem: "GPU memory not released after shutdown" in the model tests' `_shutdown`, the same as Nov 7's sweep). It also collected an untracked copy, `test_bea_switch_g2.py` (3 tests, not in `cef0f9e`). Coverage was completed one process per file, on the clean `cef0f9e` tree: all 1409 `cef0f9e` tests ran, and all passed except the 2 `capture_bench` cases (see 4). Logs: `nov8/logs/GATE/{pytest-gpu.partial.log,pyfiles,pyfiles2}`.
2. **Early window.** Prod was stopped and the gate started at 21:35 ET, before the 22:00 window start, by an intentional early `gpu-window-start` (Astro and Charlie).
3. **LMCache §4.3.** Tonight's run showed store only, because prefix caching was on. Retrieval is taken from the Nov 9 prefix-cache-off rerun on the same build `aczb9zwk`: `external_kv_transfer` 11008 tokens.
4. **Waiver for one `capture_bench` assert.** `ws-env MiB >= ws-ctx MiB` fails in both cases (B=1, B=8) of `sm70/tests/attention/test_fv100_capture_bench_gpu.py` at flint-tp2, mml 32768, ctx 16384. The cause is a test bug: it compares bytes across different partition sizes (1024 vs 256). Replacement evidence: #118 at `9c96f0e` (now in main `32250ff`), 2/2 on `aczb9zwk` (`GATE/capb/capb.log`). Granted by Charlie; Astro's OK (gate 4), 2026-11-12.

**Numbers gate 3 asks for, `cef0f9e` at prod args:** 4K decode 174.6 tok/s (≥ 170). 32K decode 157.1 tok/s default and 157.5 greedy (rerun 158.0 / 153.7 / 157.8), against C1's fork numbers at prod args: 163.0 default and 165.6 greedy.

The first entry is a **bootstrap**. It created the branch so the host flake has something to lock, and it has not passed C1 or the GPU gate. Don't deploy it: the cutover (C3) waits until `latest` has been fast-forwarded to a commit that passed all four gates.
