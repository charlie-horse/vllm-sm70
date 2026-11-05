# Qwen Phase 2 GPU runs on c4130 (handoff to Bea)

Window: 00:00–06:00 ET, prod Flint stopped by Bea first (her runbook). The
scripts below never touch systemd, the `vllm` service, `/etc/nixos` or
`activeModel`; they only start their own `vllm serve` on `127.0.0.1:8100` and
write under `$OUT`. Scripts: `sm70/scripts/qwen_phase2/`.

## Prereqs (before the window)
- A checkout of this repo at the PR's commit on c4130 (any directory; the
  scripts find the repo from their own path).
- Weights staged by Mara, each dir with a `config.json`:
  `$QWEN_STAGE/{Qwen3.6-35B-A3B-FP8,Qwen3.6-35B-A3B-DFlash2,Qwen3.8-27B-FP8,Qwen3.8-27B-DFlash2}`,
  default `QWEN_STAGE=$HOME/vllm-sm70-work/mara/qwen-weights`.
- The Felix-built main venv: `VLLM_BIN` (its `vllm`) and `VLLM_PY` (its
  python, with `pytest` for M7). Both are required; the scripts exit 2 if unset.
- Run inside `tmux`: a dropped SSH session kills the run.

Env the scripts set if unset (from `docs/phase1-gpu-runbook.md`):
`HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0`.
The attention backend is passed as `--attention-backend TRITON_ATTN`
(override with `QP2_ATTN_BACKEND`); v0.33.0 ignores `VLLM_ATTENTION_BACKEND`. If TRITON_ATTN fails in ptxas, the
`TRITON_PTXAS_PATH` fallback from that runbook applies here too.

## Commands (inside the window, after Flint is stopped)
```sh
cd <repo checkout>
export VLLM_BIN=<main venv>/bin/vllm VLLM_PY=<main venv>/bin/python
export OUT=$HOME/vllm-sm70-work/bea/qwen-phase2/$(date +%Y%m%dT%H%M)   # one dir for the whole window

sm70/scripts/qwen_phase2/m7_e2e.sh            # M7 DFlash2: triton parity + e2e
sm70/scripts/qwen_phase2/qwen_matrix.sh all   # rows 1-4 in order
# or one row at a time: qwen_matrix.sh 1 | 2 | 3 | 4
sm70/scripts/qwen_phase2/qwen_matrix.sh 4-noemu   # M7 A/B (optional, INFO only)
sm70/scripts/qwen_phase2/qwen_matrix.sh summary   # reprint all rows in $OUT
sm70/scripts/qwen_phase2/m8_qpn8_ab.sh        # M8 QPN8 A/B (section below)
```
Optional overrides: `MAX_MODEL_LEN` (default 32768; lower it if startup says
"estimated maximum model length is N", the needle test then scales down with
it), `HEALTH_TIMEOUT` (default 1500 s), `GPU_MEM_UTIL` (default 0.85),
`DFLASH2_E2E_TARGET` / `DFLASH2_E2E_DRAFT` for M7.

| row | target | draft / spec | gated |
|---|---|---|---|
| 1 | Qwen3.6-35B-A3B-FP8 | none | yes |
| 2 | Qwen3.6-35B-A3B-FP8 | Qwen3.6-35B-A3B-DFlash2, `dflash`, k=7, `draft_sample_method=probabilistic` | yes |
| 3 | Qwen3.8-27B-FP8 | none | yes |
| 4 | Qwen3.8-27B-FP8 | Qwen3.8-27B-DFlash2, same spec settings | yes |
| 2-noemu / 4-noemu | as row 2 / 4 | same, with `VLLM_SM70_DFLASH2_EMULATION=0` | no (INFO) |

Every row: TP2, `--dtype half`, `--gpu-memory-utilization 0.85`,
`--max-model-len 32768`, `--host 127.0.0.1 --port 8100`, `CUDA_VISIBLE_DEVICES=0,1`.

## Refusals (exit 2, nothing started)
- `VLLM_BIN` / `VLLM_PY` unset, or a model dir without `config.json`.
- Something already listens on port 8100.
- `nvidia-smi` shows more than 1 GiB used on either GPU: prod is still up, or
  LMCache has not yet dropped back to idle (~328 MiB/GPU, about 2 min after
  Flint stops). Wait and rerun.

## What the scripts do
**`m7_e2e.sh`** runs, by exact node ID:
- `sm70/tests/models/test_m7_dflash_sm70.py::test_triton_matches_reference_gpu` (3 params, GPU0)
- `sm70/tests/models/test_m7_dflash_sm70.py::test_e2e_real_draft_gpu` (TP2,
  `DFLASH2_E2E_TARGET`/`DFLASH2_E2E_DRAFT`, default the Qwen3.8-27B pair)

PASS = every selected test passed; a skip counts as FAIL. The e2e test only
asserts `vllm:spec_decode_num_accepted_tokens > 0` and that the text is not
garbled; it does not print an acceptance number. The A/B with
`VLLM_SM70_DFLASH2_EMULATION=0` can't run through this test: its autouse
fixture deletes the variable and the test asserts `m7.register()` is True,
which is False with the switch off. The A/B is therefore row 4 vs `4-noemu`
(same server, same probe), which records acceptance length both ways.

**`qwen_matrix.sh ROW`** starts the server, waits for `/health`, runs
`probe.py`, then stops the server's whole process group (TERM, then KILL after
60 s) on any exit, and waits for both GPUs to go idle before the next row.
The probe sends greedy requests (temperature 0) with
`chat_template_kwargs: {"enable_thinking": false}`:

| check | what | gate |
|---|---|---|
| math | 10 fixed problems, numeric answer on an `ANSWER:` line | ≥ 9/10 |
| code | 5 small functions, asserts run in a subprocess (10 s timeout) | 5/5 |
| needle | passphrase in ~31K tokens of filler (max-model-len − 768), depths 10/50/90 % | 3/3 |
| gibberish | 20 parallel requests; flags empty output, U+FFFD, > 15 % non-ASCII, repeated 4-grams, long word/char runs | 0 flagged |
| speed | single stream, ~4K-token prompt, 512 tokens (`ignore_eos`), median of 3 decode tok/s and TTFT | recorded only |
| spec (rows 2/4) | deltas of `vllm:spec_decode_*` in `/metrics` per phase and overall: mean acceptance length (1 + accepted/drafts), acceptance rate, per-position rate | must show > 0 drafts and > 0 accepted tokens; numbers recorded |

A row PASSes when every gate holds. `-noemu` rows are always INFO.

## Expected duration
| step | time |
|---|---|
| `m7_e2e.sh` (triton parity < 1 min, e2e load + compile + generate) | ~8–12 min |
| one matrix row (load + JIT/compile 3–8 min, probe ~5 min, teardown ~1 min) | ~10–15 min |
| `qwen_matrix.sh all` | ~45–60 min |
| each `-noemu` A/B row | ~10–15 min (may fail fast at startup) |

First starts include Triton JIT and torch.compile; later rows reuse the caches.

## Results
`$OUT/m7/{triton,e2e}.log` and `.junit.xml`.
`$OUT/row<N>/`: `results.json` (all answers, verdict, spec-decode deltas),
`server.log`, `probe.log`, `cmd.txt` (exact command, startup seconds),
`nvidia-smi.csv` (memory after load). The script ends with one line per row (numbers below are illustrative only):
```
row2 PASS math 10/10 code 5/5 needle 3/3 gibberish 0/20 decode4k 98.3 tok/s accept_len 3.41 (decode-phase 3.62) accept_rate 0.35
```
Send Mara the summary lines and the `$OUT` directory (server logs included).
On a FAIL, the bracketed reasons at the end of the line name the failed gates.

## M8 QPN8 A/B
Run this after the M8 GPU tests pass, in this order:
```sh
cd <repo checkout>   # PR #65 head
$VLLM_PY -m pytest -q -p no:cacheprovider -m gpu sm70/tests/qpn8 -v                      # K09 ops
$VLLM_PY -m pytest -q -p no:cacheprovider -m gpu sm70/tests/quant/test_m8_qpn8_wiring.py -v  # M8 wiring (23)
sm70/scripts/qwen_phase2/m8_qpn8_ab.sh
```
`m8_qpn8_ab.sh` serves row 4 (Qwen3.8-27B-FP8 + Qwen3.8-27B-DFlash2, `dflash`
k=7 probabilistic) with `--max-num-seqs 4`, twice, with the same server
settings and teardown as `qwen_matrix.sh`:

| arm | env | expected M8 route |
|---|---|---|
| `on` | `VLLM_SM70_FP8_BLOCK_QPN8=1 VLLM_SM70_FP8_QPN8_M32=$QP2_QPN8_M32` (default 1) | block QPN8 for M ≤ max_m (32 with M32=1) |
| `off` | `VLLM_SM70_FP8_BLOCK_QPN8=0 VLLM_SM70_FP8_QPN8_M32=$QP2_QPN8_M32` | M6 TurboMind |

Both env values and the copy policy are recorded in each arm's `cmd.txt`.
`VLLM_SM70_FP8_QPN8` / `fp8_qpn8` is the fork's separate sm70_native selector,
and M8 does not read it. max_m is K09's `qpn8_max_m(block_scaled=True)`:
8 by default, and 32 with `VLLM_SM70_FP8_QPN8_M32=1` (K09's opt-in for
block-scaled M9–32, not yet validated on V100). The script sets it to 1 by
default so that the `on` arm measures that path.

**Copy policy.** `QP2_QPN8_COPY=auto|true|false` (default `auto`, which is
`None`) passes
`--additional-config '{"sm70": {"fp8_block_qpn8_turbomind_copy": ...}}'` to
both arms. Here rows = 4 × (7 + 1) = 32.
- `auto` (safe default): admits QPN8 with no copy iff rows ≤ max_m.
  Otherwise the model stays on M6, so it never holds two weight copies.
  With `QP2_QPN8_M32=1` (default), 32 ≤ 32 and QPN8 is admitted with no copy.
  **With `QP2_QPN8_M32=0`, auto leaves the `on` arm on M6 (no QPN8).** Use
  M32=1, or set the copy policy explicitly.
- `true`: admits QPN8 and keeps a second, TurboMind-packed copy for rows above
  max_m. That is about 2× the dense FP8 bytes per GPU, roughly +12 GiB on
  Qwen3.8-27B at TP2. With `QP2_QPN8_M32=0` and max_num_seqs 4 it may OOM;
  use `QP2_QPN8_COPY=false` then.
- `false`: admits QPN8 with no copy. Rows above max_m dequantize to fp16 and
  run torch.mm.

Record which settings the PASS used.

`probe.py ab-run` sends a fixed ~1K-token prompt at concurrency 1, 2, 3 and 4.
Each stream decodes 256 tokens (`ignore_eos`, temperature 0), and each
concurrency runs 3 rounds. The aggregate decode tok/s (the sum of the
per-stream rates) is recorded as the median of those rounds. The probe also
records the greedy first 32 tokens of 5 fixed prompts.
`probe.py ab-compare` prints one line per concurrency
(`c=N qpn8 X m6 Y tok/s ratio R`) plus the greedy match count.
**PASS** = ratio ≥ 0.98 (noise band) at every concurrency, and all 5 greedy outputs match.

The script also greps `server.log` for `SM70 FP8 QPN8` into
`$OUT/m8/<arm>/m8_log.txt`. In `on`, expect
"SM70 FP8 QPN8 decode (M<=N) enabled ...: rows=32, max_m=N, keep_copy=...",
and, with a copy, "TurboMind copy on cuda:X: G GiB so far" warnings (one per GiB).
"SM70 FP8 QPN8 not used: ... rows=32 ... > max_m=8" means auto refused, so
the A/B compared M6 with M6.
`off` has no line, because the kill switch stops M8 before it logs.

Results: `$OUT/m8/{on,off}/` hold `ab.json`, `server.log`, `probe.log`,
`cmd.txt`, `nvidia-smi.csv` and `m8_log.txt`. `$OUT/m8/summary.txt` and
`ab_summary.json` hold the comparison. Duration: about 2 × (10 min load + 5 min probe).
Exit codes: 0 PASS, 1 FAIL, 2 refused (same refusals as above).
