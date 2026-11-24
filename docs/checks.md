# Testing the code on a Volta GPU

These checks show that a build of this repo runs correctly and fast enough
on Volta GPUs (V100, `sm_70`).
- **Unit tests** (pytest) check single kernels and code paths.
- **Numeric gates** check that the model's numbers stay within a known
  tolerance of a reference.
- **Server checks** start a real vLLM server and test its speed and answers.
- **LMCache checks** test LMCache, the separate server that stores computed
  prompt data (KV cache) so vLLM can reuse it.

Run everything from a checkout of this repo at the commit under test.

## Build

`$OUT` is the built `.#vllm` and `$GT` the built `.#gpu-tests` (the same
python with pytest added).

```sh
OUT=$(nix build .#vllm --no-link --print-out-paths)
GT=$(nix build .#gpu-tests --no-link --print-out-paths)
CUDA_VISIBLE_DEVICES= $OUT/bin/python -c "import torch, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print(torch.version.cuda, torch.cuda.get_arch_list(), vllm.__version__)"
```

The import check needs no GPU. `sm_70` must be in the printed list. Don't
pass `-I` to `$OUT/bin/python`: it drops vLLM from the module path.

## GPU rules

- Nothing else may use the GPUs during a GPU check. The scripts refuse
  (exit 2) if a GPU has more than 1 GiB in use.
- Keep `gpu_memory_utilization` at or below 0.85 in the M5 gates.
- A skipped test counts as a failure, except where a section says otherwise.

## Test server

The server checks run against a server started from `$OUT` with the
arguments and environment that `nix/module.nix` (vLLM) and `nix/lmcache.nix`
(LMCache) give the model. Start LMCache first; both run as the same user.

```sh
$OUT/bin/lmcache server <arguments from nix/lmcache.nix> &
$OUT/bin/vllm serve <model path> <arguments from nix/module.nix> &
until curl -sf 127.0.0.1:<port>/health; do sleep 10; done
```

Before starting, make sure nothing else listens on those ports
(`ss -ltnp`), so the checks can't talk to an old server.

## GPU gate

The gate passes for one exact commit. A run on any other commit doesn't
count, even if only docs changed. Everything uses `$OUT` and `$GT` built
from that commit. All of these must pass:

1. [Build](#build) and the [LMCache build check](#lmcache-build-check).
2. [GPU pytest](#gpu-pytest): every file passes.
3. [M5 gates](#m5-gates): every node PASS, plus the
   [mutation check](#mutation-check) and the
   [Flash-V100 check](#flash-v100-check).
4. [Switch envelope](#switch-envelope): every switch stays within the
   envelope.
5. On the [test server](#test-server):
   - [Decode speed](#decode-speed): 4K ≥ 170 tok/s; 32K measured and
     recorded.
   - [Greedy identity](#greedy-identity): every prompt matches the base.
   - [Quality probe](#quality-probe): PASS.
   - [LMCache reuse](#lmcache-reuse): both parts pass.

Keep every log of the run together with the commit.

## GPU pytest

Run one pytest process per test file, never one process for the whole tree.

```sh
$GT/bin/pytest --collect-only -q -m gpu sm70/tests | grep '::' | cut -d: -f1 | sort -u > gpu-files
while read -r f; do $GT/bin/pytest -q -p no:cacheprovider -m gpu "$f" || echo "FAIL $f"; done < gpu-files 2>&1 | tee pytest-gpu.log
```

The model tests need the environment of the [M5 gates](#m5-gates); export
it first, or they skip. Without GPUs, `--collect-only` still works.

## M5 gates

Numeric gates T1–T7 for the Flint model. Most are **teacher-forced**: the
model is fed a fixed token sequence and must score each next token, so two
runs can be compared token by token without one early difference changing
everything after it. Each gate compares our numbers with the **envelope**
(how far two valid runs of the reference build differ from each other):
ours may differ by at most 1.5× that. **Greedy** means always picking the
most likely token, so the output should repeat exactly.

Rules:
- TurboMind small-shape tuning (picking kernels by timing at startup) is
  off. The scripts switch it off and print `tuning: ...=0` in every log;
  check it. `M5_ALLOW_TUNING=1` is for diagnostics only, and then needs
  `VLLM_SM70_GEMM_LUT_PATH` and `VLLM_SM70_GEMM_LUT_STRICT=1`.
- T4c (two runs of our own build) must PASS, or T4a/T4d/T4e/T7 are
  reported `NOT COUNTED`.
- t4b skips by design under Flash-V100 (`SKIPPED by design`); that counts
  as a pass.
- Never raise `POOLED_FACTOR` in `sm70/scripts/m5_teacher_forced.py`.

```sh
W=$PWD/gate-$(git rev-parse --short HEAD); mkdir -p $W
TF=sm70/scripts/m5_teacher_forced.py
ARCH='{"architectures": ["FlintSm70ForCausalLM"]}'
export VLLM_PY=$GT/bin/python
export FLINT_SM70_CHECKPOINT=<Flint checkpoint dir>
export FLINT_SM70_GOLDEN=<reference greedy outputs, JSON>
export FLINT_SM70_T4_SEQS_DIR=<token sequences: setK.json, at least 3>
export FLINT_SM70_T4_ENVELOPE=<envelope JSON>
export FLINT_SM70_FORK_SCORES_DIR=<reference scores: setK/{eager,eager2,graphs,unfused,graphsonly}.json>
export FLINT_SM70_T5_FORK_GOLDEN_SCORES=<reference scores of the golden sequences>
OUT=$W/m5 bash sm70/scripts/m5_window.sh
```

`M5_NODES="t4c t4a ..."` runs a subset. Exact T4b runs only on the Triton
attention backend: `VLLM_SM70_FLASH_V100=0 M5_NODES=t4b bash sm70/scripts/m5_window.sh`.

The reference files come from `m5_teacher_forced.py`: `gen` makes the
sequences, `score` scores them on a build, `seqs-from-golden` turns the
golden outputs into sequences, and `envelope` builds the envelope from the
scores (CPU only). An envelope that lacks a field the gates need fails
closed; rebuild it from the same scores:

```sh
$VLLM_PY $TF envelope --scores-dir $FLINT_SM70_FORK_SCORES_DIR --out $FLINT_SM70_T4_ENVELOPE
```

### Mutation check

Proves the gates can catch bugs: it plants small bugs and T4d and T7 must
FAIL. Run it after T4d and T7 passed, or add `M5_BASELINE=1`. Same
environment as above.

```sh
OUT=$W/mutation bash sm70/scripts/m5_mutation_check.sh
```

| planted bug | must FAIL |
|---|---|
| `rope_pos_decode`, `kv_slot_decode` | T4d and T7 |
| `moe_decode_scale` | T4d |
| `moe_decode_scale_half`, `rope_theta_1p01` | info only, but must be applied and fire |

Every required run must fail the gate itself (not crash or skip), and its
log must show `M5_MUTATION <name> installed` and `fired`.

### Flash-V100 check

Flash-V100 is this repo's attention kernel for Volta.

1. The M5 logs show it was used:
   ```sh
   grep -h "Flint SM70: using\|Triton forward\|grouped sink decode\|sink decode is off" $W/m5/*.log
   ```
   Pass: `Flint SM70: using FLASH_ATTN_V100 (CUSTOM)` and
   `... N one-launch, 0 two-launch`. Fail on `using TRITON_ATTN`,
   `using the Triton forward` or `one-launch sink decode is off`.
2. Its kernel tests pass:
   ```sh
   $GT/bin/pytest -q -m gpu sm70/tests/attention/test_fv100_fused_decode_gpu.py \
     sm70/tests/attention/test_fv100_decode_sinks_gpu.py sm70/tests/attention/test_fv100_prefill_sinks_gpu.py
   ```
3. Batched greedy output matches the Triton backend:
   ```sh
   FV=sm70/scripts/m5_fv100_check.py
   $VLLM_PY $FV gen --model $FLINT_SM70_CHECKPOINT --hf-overrides "$ARCH" --backend fv100  --out $W/fv100.json
   $VLLM_PY $FV gen --model $FLINT_SM70_CHECKPOINT --hf-overrides "$ARCH" --backend triton --out $W/triton.json
   $VLLM_PY $FV compare $W/fv100.json $W/triton.json --show
   ```
   `compare` fails on run-to-run differences, repetitive output or a
   difference at the first token. Text that reads as gibberish also fails.

## Switch envelope

A **kill switch** is an environment variable that turns one of our changes
off (set to `0`). With each kernel-changing switch off, the model's numbers
must stay within the envelope. A new kill switch that changes kernels joins
this list:

`VLLM_SM70_MODELOPT_NVFP4_GEMV`, `VLLM_SM70_FLINT_MOE_QPN_DECODE`,
`VLLM_SM70_FLINT_QK_ROPE_KV`, `VLLM_SM70_FLINT_FUSED_ROUTER`,
`VLLM_SM70_FLINT_ATTN_FUSED_DECODE`, `VLLM_SM70_FLINT_MOE_COMPACT`.

```sh
S=(--model "$FLINT_SM70_CHECKPOINT" --hf-overrides "$ARCH" --seqs-dir $FLINT_SM70_T4_SEQS_DIR --out-dir $W/sw --mode graphs --decode --prefill)
$VLLM_PY $TF score "${S[@]}" --name base
for sw in MODELOPT_NVFP4_GEMV FLINT_MOE_QPN_DECODE FLINT_QK_ROPE_KV FLINT_FUSED_ROUTER FLINT_ATTN_FUSED_DECODE FLINT_MOE_COMPACT; do
  $VLLM_PY $TF score "${S[@]}" --name off-$sw --env VLLM_SM70_$sw=0
  for k in 0 1 2; do for m in decode prefill; do
    echo "$sw set$k $m $($VLLM_PY $TF compare $W/sw/set$k/base.json $W/sw/set$k/off-$sw.json --metric $m)"
  done; done
done | tee $W/switch-envelope.log
```

Pass, for every switch, set and metric, using the `eager_vs_graphs` ANCHOR
and GATE lines that `envelope` prints: mean ≤ 1.5 × the anchor mean, max ≤
1.5 × the anchor max, and the count over 1.0 within the GATE line's limit.
`VLLM_SM70_FLINT_MOE_COMPACT` only acts at batch 3–4; its gate is
`sm70/tests/models/test_m4_compact_g2.py` in the [GPU pytest](#gpu-pytest).

## Decode speed

**Decode tok/s** is how many output tokens per second one request gets.
One warm-up request, then three measured, one at a time:

```sh
for n in 4096 32768; do
  $OUT/bin/vllm bench serve --base-url http://127.0.0.1:<port> \
    --model <served model name> --tokenizer <model path> \
    --dataset-name random --random-input-len $n --random-output-len 256 \
    --ignore-eos --max-concurrency 1 --num-warmups 1 --num-prompts 3 | tee bench-$n.log
done
```

Decode tok/s = 1000 / `Median TPOT (ms)`. Gate: 4K ≥ 170 tok/s; 32K is
recorded.

## Greedy identity

Sends fixed prompts greedily and checks the answers match a base recorded
on a gated build (e.g. `scripts/checks/greedy/base-prod-cef0f9e.json`).

```sh
G=scripts/checks/greedy
python3 $G/greedy_run.py http://127.0.0.1:<port> $G/greedy-prompts.json greedy.json 2 <base.json> --model <served model name>
```

Pass: every prompt prints `vs ref ... match`. Compare like with like. The
**prefix cache** keeps already-computed prompt data, and a first ("cold")
send can differ from later ("cached") sends on near-tie tokens. So compare
cold with cold (fresh server) or cached with cached.

`scripts/checks/greedy/teacher_forced.py` (`capture`, then `compare REF
CAND`) compares two server setups teacher-forced; it exits 1 on failure.

## Quality probe

Asks the server math, code, find-a-fact-in-a-long-text ("needle") and 20
parallel questions. Gates: math ≥ 9/10, code 5/5, needle 3/3, no gibberish.

```sh
python3 sm70/scripts/qwen_phase2/probe.py run --url http://127.0.0.1:<port> \
  --model <served model name> --row gate --max-model-len 32768 --out quality.json
```

Pass: exit 0 and `PASS` in the summary line.

## LMCache reuse

Checks that LMCache stores a long prompt and gives it back.

1. Within one vLLM run:
   ```sh
   python3 scripts/check6-lmcache-reaper.py run --dry-run \
     --base-url http://127.0.0.1:<port> --admin-url http://127.0.0.1:<lmcache http port>
   ```
   Pass: `SMOKE-PASS`, exit 0. It sends a long prompt, pushes it out of
   vLLM's own prefix cache, sends it again, and the second request must be
   served from LMCache.
2. Across a vLLM restart: send a prompt of at least two 256-token chunks,
   restart only vLLM (LMCache keeps running), send it again. Pass:
   `vllm:prompt_tokens_cached_by_source{source="external_kv_transfer"}` in
   `/metrics` grows by about the prompt length minus its last partial
   chunk, and the answer matches.

## LMCache build check

Checks the LMCache inside `$OUT`, with no GPU and no running services.

```sh
scripts/check-lmcache-cpu.sh $OUT        # exit 0 = all PASS
```

| # | check | catches |
|---|---|---|
| 1 | import `lmcache.cuda_ops`, `lmcache_native`, `lmcache_redis`, `lmcache_fs` | a missing extension |
| 2 | `cuda_ops*.so` links `libcudart.so.12`, not `.so.13` | a CUDA 13 build |
| 3 | `$OUT/bin/lmcache server --help` | a broken entry point |
| 4 | import the MP connector from LMCache and from vLLM | a missing dependency |
| 5 | patch 0001's markers in the installed adapter | a dropped patch |
| 6 | `importlib.metadata.version("lmcache")` is the pinned version | wrong metadata |
| 7 | dashboard files and app import | missing package data |
| 8 | cupy's CUDA runtime and NVRTC are 12.x | a CUDA 13 cupy |
| 9 | no `libcudart.so.13` in `$OUT`'s closure | a second CUDA runtime |

On a GPU, torch and cupy must share a CUDA stream:

```sh
$OUT/bin/python -c "import cupy,torch; s=torch.cuda.Stream(); cupy.cuda.ExternalStream(s.cuda_stream).use(); print('ok')"
```

## Check 6: LMCache after 1 h idle

Tests `nix/patches/lmcache/0001-mp-worker-heartbeat-from-registration.patch`:
after a vLLM worker has had no long prompt for over an hour, LMCache must
still store and serve for it.

`scripts/check6-lmcache-reaper.py` talks to vLLM and LMCache on loopback
and reads their start time and logs from systemd. Pass
`--vllm-unit`/`--lmcache-unit` if your services have other names. Nothing
else may send prompts during the hour: any prompt over 200 tokens makes the
run INVALID. A run takes over an hour.

```sh
# restart vLLM, then:
python3 scripts/check6-lmcache-reaper.py run 2>&1 | tee check6.log
```

`run` is `start` + wait + `finish`; you can also run `start` right after
the restart and `finish` later (state in `./check6-state.json`). If the
script can't read the LMCache log, it ends INCOMPLETE and prints the
command to save it; then:

```sh
python3 scripts/check6-lmcache-reaper.py judge --journal-file <saved log>
```

Results: PASS/FAIL/INVALID per criterion, `./check6-summary.json`, exit 0
PASS, 2 INVALID, 1 FAIL or error. `--dry-run` skips the wait. See `--help`
for the other options.

PASS, all of:
1. LMCache's log has no `Reaped GPU instance` line since vLLM started.
2. LMCache's `/status` lists every GPU in `registered_gpu_ids`, after the
   wait and after the requests.
3. The second long request is served from LMCache:
   `vllm:external_prefix_cache_hits` grows and
   `usage.prompt_tokens_details.cached_tokens` is above 0.

INVALID means the run didn't test the bug: vLLM restarted, a long prompt
arrived during the hour, the wait was too short, or the long prompt stayed
in vLLM's own prefix cache (raise `--max-flood-tokens`). A build without the
patch fails all three.

## Qwen matrix, M7 and M8

Scripts in `sm70/scripts/qwen_phase2/` start their own server on port 8100
and write results under `$OUT` (here a results folder, not the build).

```sh
export VLLM_BIN=<built .#vllm>/bin/vllm VLLM_PY=$GT/bin/python
export QWEN_STAGE=<folder with the Qwen model folders> OUT=$PWD/qwen-results
sm70/scripts/qwen_phase2/m7_e2e.sh              # M7 DFlash2 tests
sm70/scripts/qwen_phase2/qwen_matrix.sh all     # rows 1-4, or one row: 1 | 2 | 3 | 4
sm70/scripts/qwen_phase2/qwen_matrix.sh summary
```

Exit 0 PASS, 1 FAIL, 2 refused to start. A row passes when the [quality
probe](#quality-probe) gates hold; rows 2 and 4 (with a draft model) must
also show accepted draft tokens. `2-noemu`/`4-noemu` are info only.

M8 QPN8 A/B: after `$GT/bin/pytest -q -m gpu sm70/tests/qpn8 sm70/tests/quant/test_m8_qpn8_wiring.py`
passes, run `sm70/scripts/qwen_phase2/m8_qpn8_ab.sh`. Pass: QPN8 speed ≥
0.98 × the M6 speed at every concurrency, all 5 greedy outputs match, and
`$OUT/m8/on/m8_log.txt` shows `SM70 FP8 QPN8 decode (M<=N) enabled`. With
`QP2_QPN8_M32=0`, set `QP2_QPN8_COPY` explicitly, or the `on` arm stays on
M6.
