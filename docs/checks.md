# GPU checks

Every command runs on the serving host, from a checkout of this repo at
the commit under test, inside `tmux`. Commands marked sudo run with sudo.

## GPU runs on the serving host

- Production vLLM serves on the GPUs all the time. A GPU run stops it first
  and restores it at the end, also after a failure:
  ```sh
  sudo systemctl stop vllm vllm-lmcache
  nvidia-smi --query-gpu=index,memory.used --format=csv   # every GPU near 0 MiB
  # ... the run ...
  sudo systemctl start vllm                    # starts vllm-lmcache first
  systemctl is-active vllm-lmcache vllm        # both active
  curl -sf 127.0.0.1:8000/v1/models            # the unit's --port
  ```
- After `vllm` stops, LMCache can keep GPU memory for a few minutes. The
  runners refuse (exit 2) while any GPU has more than 1 GiB in use: wait, or
  stop `vllm-lmcache` as above.
- Before starting a server, nothing listens on its ports:
  `ss -ltnp | grep -E ':(<vllm port>|<lmcache ports>)\b'` prints nothing.
- A test run never edits `/etc/nixos`, never switches the system and never
  changes the `vllm` unit.

## Candidate build

`$OUT` is the outPath of `.#vllm`, `$GT` that of `.#gpu-tests` (the same
python plus pytest), both built from the commit under test, on a build host
whose binary cache the serving host uses.

```sh
# build host, checkout at the commit
nix build .#vllm .#gpu-tests --no-link --print-out-paths
# serving host
nix-store --realise $OUT --add-root ~/vsm70/result-vllm --indirect
nix-store --realise $GT --add-root ~/vsm70/result-gpu-tests --indirect
cd /var/tmp && CUDA_VISIBLE_DEVICES= $OUT/bin/python -c "import torch, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), vllm.__version__)"
```

The import check needs no GPU; `sm_70` must be in the arch list. Don't pass
`-I` to the wrapper's python: it drops the vLLM tree from `PYTHONPATH`.
Then run the [LMCache build check](#lmcache-build-check).

## Candidate server at prod args

Runs `$OUT` with the production units' arguments and environment, as
transient units, after production is stopped.

1. Read both units: `systemctl cat vllm-lmcache vllm` (`ExecStart`,
   `Environment`, `User`, `SupplementaryGroups`).
2. Check the ports are free (see above).
3. Start LMCache, then vLLM, each with every `Environment=` of its unit as
   `--setenv=` and `$OUT/bin/...` in place of the unit's binary (sudo):
   ```sh
   sudo systemd-run --unit=vsm70-test-lmcache -p User=vllm -p Group=vllm \
     --setenv=... $OUT/bin/lmcache <the vllm-lmcache unit's arguments>
   sudo systemd-run --unit=vsm70-test -p User=vllm -p Group=vllm -p SupplementaryGroups=video \
     --setenv=... $OUT/bin/vllm serve <the vllm unit's arguments>
   until curl -sf 127.0.0.1:<port>/health; do sleep 10; done
   ```
   Both run as the same user.
4. Confirm the LMCache server is this run's: `journalctl -u vsm70-test-lmcache`
   shows it starting, and `ss -ltnp` shows its PID on the LMCache ports.
5. Stop with `sudo systemctl stop vsm70-test vsm70-test-lmcache`.

## GPU gate

The gate passes for one exact commit. A run on any other commit doesn't
count, even if only docs changed in between. Everything runs on `$OUT` and
`$GT` built from that commit; the server checks run at prod args. Steps:

1. [Candidate build](#candidate-build) and the [LMCache build
   check](#lmcache-build-check) pass.
2. Production stopped, GPUs idle, no server on the test ports.
3. [GPU pytest](#gpu-pytest): every file passes.
4. [M5 T-gates](#m5-t-gates): `m5_window.sh` reports every node PASS (t4b
   `SKIPPED by design` under Flash-V100 counts as pass), the [mutation
   check](#mutation-check) passes and the [Flash-V100 check](#flash-v100-check)
   passes.
5. [Switch envelope](#switch-envelope): every switch stays within the fork
   envelope.
6. [Candidate server at prod args](#candidate-server-at-prod-args), then on it:
   - [Decode bench](#decode-bench): 4K decode ≥ 170 tok/s; 32K decode
     measured and recorded.
   - [Greedy identity](#greedy-identity): all prompts match the base.
   - [Quality probe](#quality-probe): PASS.
   - [LMCache reuse](#lmcache-reuse): both parts pass.
7. Stop the candidate units and restore production
   ([GPU runs on the serving host](#gpu-runs-on-the-serving-host)).

Keep every log of the run with the commit and `$OUT`.

## GPU pytest

Run one pytest process per test file, never one process over the whole tree.

```sh
PT=$GT/bin/pytest
$PT --collect-only -q -m gpu sm70/tests | grep '::' | cut -d: -f1 | sort -u > gpu-files
while read -r f; do $PT -q -p no:cacheprovider -m gpu "$f" || echo "FAIL $f"; done < gpu-files 2>&1 | tee pytest-gpu.log
```

Model tests read their inputs from the env of the [M5 T-gates](#m5-t-gates);
export it first, or they skip. Without GPUs, `--collect-only` still collects.

## M5 T-gates

Teacher-forced and greedy gates for Flint, T1–T7, against the fork's
numeric envelope. Scripts: `sm70/scripts/m5_window.sh`,
`sm70/scripts/m5_teacher_forced.py`.

Rules:
- Gates run with TurboMind small-shape tuning off. The scripts set it and
  print a `tuning: ...=0` line in every log; check it. `M5_ALLOW_TUNING=1` is
  for diagnostics, and then only with `VLLM_SM70_GEMM_LUT_PATH` and
  `VLLM_SM70_GEMM_LUT_STRICT=1` (the scripts exit otherwise).
- Keep `gpu_memory_utilization` at or below 0.85.
- T4c must PASS; otherwise T4a/T4d/T4e/T7 are reported `NOT COUNTED`.
- A skip is a FAIL, except t4b's by-design skip under Flash-V100.
- Never raise `POOLED_FACTOR`. A pooled ratio between the limit and
  `POOLED_INVESTIGATE` is a real numeric change to investigate.

Inputs (from the [fork envelope](#fork-envelope)):

```sh
W=~/gpu-gate/$(git rev-parse --short HEAD); mkdir -p $W
TF=sm70/scripts/m5_teacher_forced.py
ARCH='{"architectures": ["FlintSm70ForCausalLM"]}'   # our model class
export VLLM_PY=$GT/bin/python
export FLINT_SM70_CHECKPOINT=<ModelOpt NVFP4 Flint checkpoint dir>
export FLINT_SM70_GOLDEN=<fork greedy golden JSON>
export FLINT_SM70_T4_SEQS_DIR=$W/seqs                  # setK.json, >= 3 sets
export FLINT_SM70_T4_ENVELOPE=$W/fork_envelope.json
export FLINT_SM70_FORK_SCORES_DIR=$W/fork
export FLINT_SM70_T5_FORK_GOLDEN_SCORES=$W/golden/fork/set9/fork_graphs.json
export OUT_M5=$W/m5
OUT=$OUT_M5 bash sm70/scripts/m5_window.sh
```

`M5_NODES="t4c t4a ..."` runs a subset; the default order is
`t4c t4a t4b t4e t4d t5 t7 t2 t3 t1 t6`. Exact T4b runs only on TRITON:
`VLLM_SM70_FLASH_V100=0 M5_NODES=t4b bash sm70/scripts/m5_window.sh`.
The runner prints one gate line per set and metric, one pooled line per
gate, and PASS/FAIL per node. An envelope without `over_1_0`,
`seq_mean_max`, `pooled_mean` or the `eager_vs_graphsonly` pair fails
closed: rebuild it ([fork envelope](#fork-envelope), step 3).

### Mutation check

T4d and T7 must FAIL on planted bugs. Run after T4d and T7 passed unmutated,
or add `M5_BASELINE=1`. Same env as above.

```sh
OUT=$W/mutation bash sm70/scripts/m5_mutation_check.sh
```

| mutation | must FAIL |
|---|---|
| `rope_pos_decode`, `kv_slot_decode` | T4d and T7 |
| `moe_decode_scale` | T4d (pooled-mean gate) |
| `moe_decode_scale_half`, `rope_theta_1p01` | INFO only; must still be applied and fire |

The check passes only if every required run fails the gate itself (not a
crash or skip) and its log shows `M5_MUTATION <name> installed` and `fired`.
`NOT FIRED` on `moe_decode_scale` means K12 was not admitted: look for
`K12 M<=2 decode admitted` in the log.

### Flash-V100 check

1. Route, from the M5 logs:
   ```sh
   grep -h "Flint SM70: using\|Triton forward\|grouped sink decode\|sink decode is off" $OUT_M5/*.log
   grep -h "K12 decode skipped\|K12 M<=2 decode" $OUT_M5/*.log | sort | uniq -c
   ```
   Pass: `Flint SM70: using FLASH_ATTN_V100 (CUSTOM)` and, in graphs mode,
   `grouped sink decode ... N one-launch, 0 two-launch`. Fail on
   `Flint SM70: using TRITON_ATTN`, `Flash-V100: using the Triton forward` or
   `one-launch sink decode is off`.
2. Kernel tests:
   ```sh
   $GT/bin/pytest -q -m gpu sm70/tests/attention/test_fv100_fused_decode_gpu.py \
     sm70/tests/attention/test_fv100_decode_sinks_gpu.py \
     sm70/tests/attention/test_fv100_prefill_sinks_gpu.py
   ```
3. Batched greedy, Flash-V100 against Triton:
   ```sh
   FV=sm70/scripts/m5_fv100_check.py
   $VLLM_PY $FV gen --model "$FLINT_SM70_CHECKPOINT" --hf-overrides "$ARCH" --backend fv100  --out $W/fv100.json
   $VLLM_PY $FV gen --model "$FLINT_SM70_CHECKPOINT" --hf-overrides "$ARCH" --backend triton --out $W/triton.json
   $VLLM_PY $FV compare $W/fv100.json $W/triton.json --show
   ```
   `compare` fails on run-to-run differences, degenerate output or a
   first-token divergence. Divergence after a shared prefix is allowed; text
   that reads as gibberish is a FAIL.

### Fork envelope

The gate inputs come from the fork's build (`$FORK_PY`, its python) on the
same checkpoint. Each command is a separate engine instance; don't merge
them. On our stack pass `--hf-overrides "$ARCH"`; on the fork pass none.

```sh
CKPT=$FLINT_SM70_CHECKPOINT
FORK_KW='{"kernel_config": {"sm70_skinny_moe": false}}'
FORK_GO_KW='{"kernel_config": {"sm70_skinny_moe": false}, "compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}}'
# 1. sequences, on ours
$VLLM_PY $TF gen --model $CKPT --hf-overrides "$ARCH" --set 0 --set 1 --set 2 --out-dir $W/seqs
# 2. fork scores
S="--model $CKPT --seqs-dir $W/seqs --out-dir $W/fork --decode --prefill"
for n in eager eager2; do $FORK_PY $TF score $S --name $n --mode eager --llm-kwargs "$FORK_KW"; done
$FORK_PY $TF score $S --name graphs --mode graphs --llm-kwargs "$FORK_KW"
$FORK_PY $TF score $S --name unfused --mode eager --llm-kwargs "$FORK_KW" --env VLLM_SM70_FLINT_QK_ROPE_KV=0
$FORK_PY $TF score $S --name graphsonly --mode graphs --llm-kwargs "$FORK_GO_KW"
# 2b. fork forced decode of the golden sequences (T5 near-tie rule)
$FORK_PY $TF seqs-from-golden --golden $FLINT_SM70_GOLDEN --out $W/golden/set9.json
$FORK_PY $TF score --model $CKPT --seqs $W/golden/set9.json --out-dir $W/golden/fork \
  --name fork_graphs --mode graphs --decode --llm-kwargs "$FORK_KW"
# 3. envelope (CPU, any python; rerun to rebuild from existing scores)
$VLLM_PY $TF envelope --scores-dir $W/fork --out $W/fork_envelope.json
```

A score run that raises `ForcingError` means the stack bypassed the forcing
processor: stop.

## Switch envelope

Each kernel-changing kill switch, set to `0`, stays within the fork
envelope, teacher-forced. A new kill switch that changes kernels joins this
list:

`VLLM_SM70_MODELOPT_NVFP4_GEMV`, `VLLM_SM70_FLINT_MOE_QPN_DECODE`,
`VLLM_SM70_FLINT_QK_ROPE_KV`, `VLLM_SM70_FLINT_FUSED_ROUTER`,
`VLLM_SM70_FLINT_ATTN_FUSED_DECODE`, `VLLM_SM70_FLINT_MOE_COMPACT`.

```sh
S=(--model "$CKPT" --hf-overrides "$ARCH" --seqs-dir $W/seqs --out-dir $W/sw --mode graphs --decode --prefill)
$VLLM_PY $TF score "${S[@]}" --name base
for sw in MODELOPT_NVFP4_GEMV FLINT_MOE_QPN_DECODE FLINT_QK_ROPE_KV FLINT_FUSED_ROUTER FLINT_ATTN_FUSED_DECODE FLINT_MOE_COMPACT; do
  $VLLM_PY $TF score "${S[@]}" --name off-$sw --env VLLM_SM70_$sw=0
  for k in 0 1 2; do for m in decode prefill; do
    echo "$sw set$k $m $($VLLM_PY $TF compare $W/sw/set$k/base.json $W/sw/set$k/off-$sw.json --metric $m)"
  done; done
done | tee $W/switch-envelope.log
```

Pass, per switch, set and metric, with the `eager_vs_graphs` ANCHOR and GATE
lines that `envelope` prints: mean ≤ 1.5 × the anchor mean, max ≤ 1.5 × the
pair's anchor max over both metrics, and the count over 1.0 nats ≤ the GATE
line's limit. `VLLM_SM70_FLINT_MOE_COMPACT` only acts at batch 3–4: its gate
is `sm70/tests/models/test_m4_compact_g2.py`, run in the
[GPU pytest](#gpu-pytest).

## Decode bench

On the [candidate server at prod args](#candidate-server-at-prod-args):

```sh
for in_len in 4096 32768; do
  $OUT/bin/vllm bench serve --base-url http://127.0.0.1:<port> \
    --model <served model name> --tokenizer <model path> \
    --dataset-name random --random-input-len $in_len --random-output-len 256 \
    --ignore-eos --max-concurrency 1 --num-warmups 1 --num-prompts 3 \
    | tee bench-$in_len.log
done
```

Decode tok/s = 1000 / `Median TPOT (ms)`. Gate: 4K ≥ 170 tok/s. 32K is
measured and recorded.

## Greedy identity

`scripts/checks/greedy/greedy_run.py` sends each prompt of
`greedy-prompts.json` `repeats` times and compares the first send with the
reference's first send. The reference is a base recorded on the gated build
at prod args, e.g. `scripts/checks/greedy/base-prod-cef0f9e.json`.

```sh
G=scripts/checks/greedy
python3 $G/greedy_run.py http://127.0.0.1:<port> $G/greedy-prompts.json greedy.json 2 <base.json> --model <served model name>
```

Pass: every prompt prints `vs ref ... match`. Compare like with like: cold
vs cold (fresh server or a unique prefix) or prefix-cached vs prefix-cached.
A cold send may diverge from cached sends on near-tie tokens.

For a teacher-forced comparison of two served configurations (cold vs
cached, two TP sizes, served vs HF), use `scripts/checks/greedy/teacher_forced.py`
(`capture`, then `compare REF CAND`; exits 1 on failure).

## Quality probe

`sm70/scripts/qwen_phase2/probe.py run` against a running server: math
(≥ 9/10), code (5/5), needle near max-model-len (3/3), no gibberish at
concurrency 20 (0 flagged), plus a 4K decode speed that is recorded only.

```sh
python3 sm70/scripts/qwen_phase2/probe.py run --url http://127.0.0.1:<port> \
  --model <served model name> --row gate --max-model-len 32768 --out quality.json
```

Pass: exit 0 and the summary line shows `PASS`.

## LMCache reuse

On the [candidate server at prod args](#candidate-server-at-prod-args)
(prefix caching on):

1. Store and retrieve within one vLLM instance:
   ```sh
   python3 scripts/check6-lmcache-reaper.py run --dry-run \
     --base-url http://127.0.0.1:<port> --admin-url http://127.0.0.1:<lmcache http port> \
     --vllm-unit vsm70-test --lmcache-unit vsm70-test-lmcache
   ```
   Pass: `SMOKE-PASS`, exit 0. It sends a ~11K-token prompt, evicts it from
   vLLM's GPU prefix cache and resends it; the second request must get an
   external hit.
2. Across a vLLM restart: send a prompt P of at least two 256-token chunks,
   restart only `vsm70-test` (sudo; LMCache keeps its store), resend P.
   Pass: `vllm:prompt_tokens_cached_by_source{source="external_kv_transfer"}`
   in `/metrics` grows by about P's length minus its last partial chunk, and
   the answer matches the first send.

## LMCache build check

Checks the LMCache inside `$OUT` without GPUs (`CUDA_VISIBLE_DEVICES` is
empty), sudo or services. Run it after realising a new `$OUT` and before
any switch to it.

```sh
scripts/check-lmcache-cpu.sh $OUT        # exit 0 = all PASS
```

| # | check | guards against |
|---|---|---|
| 1 | import `lmcache.cuda_ops`, `lmcache_native`, `lmcache_redis`, `lmcache_fs` | a missing extension (`import lmcache` only logs it) |
| 2 | `ldd cuda_ops*.so`: `libcudart.so.12`, no `.so.13` | a CUDA 13 build |
| 3 | `$OUT/bin/lmcache server --help` | a broken service entry point |
| 4 | import the MP connector from LMCache and from vLLM | a missing runtime dep on vLLM's path |
| 5 | patch 0001's markers in the installed adapter | a silently dropped patch |
| 6 | `importlib.metadata.version("lmcache")` is the pinned version | wrong metadata (vLLM reads the version) |
| 7 | `lmcache_frontend/static/index.html`, import the dashboard app | missing package data for `vllm-lmcache-ui` |
| 8 | cupy's CUDA runtime and NVRTC are 12.x | a CUDA 13 cupy |
| 9 | no `libcudart.so.13` in `$OUT`'s closure | a second, CUDA 13 runtime |

GPU half, with production stopped: torch's and cupy's CUDA runtimes share a
stream.

```sh
$OUT/bin/python -c "import cupy,torch; s=torch.cuda.Stream(); cupy.cuda.ExternalStream(s.cuda_stream).use(); print('ok')"
```

## Check 6: LMCache after 1 h idle

Guards `nix/patches/lmcache/0001-mp-worker-heartbeat-from-registration.patch`
(the bug is in its header): LMCache must still
store and serve for a vLLM worker that got no prompt of a 256-token chunk
for over an hour after registering.

Script: `scripts/check6-lmcache-reaper.py` (stdlib only, no sudo; it talks
to the vLLM and LMCache admin APIs on loopback and reads `systemctl show`
and `journalctl`). It runs against the production units, with the GPUs
serving and no other clients: any prompt over 200 tokens during the quiet
hour makes the run INVALID. A run takes over an hour.

```sh
sudo systemctl restart vllm                      # the script never uses sudo
python3 scripts/check6-lmcache-reaper.py run 2>&1 | tee ~/check6.log
```

`run` = `start` + wait + `finish`. In two steps (state in
`./check6-state.json`):

```sh
python3 scripts/check6-lmcache-reaper.py start     # right after the restart
python3 scripts/check6-lmcache-reaper.py finish    # any time later; waits out the rest
```

If the LMCache journal is not readable as the running user, the run ends INCOMPLETE and
prints the `journalctl` command with its `--since` time. Save the journal
(sudo) and re-judge:

```sh
sudo journalctl -u vllm-lmcache --since @<vllm start> -o short-unix --no-pager > ~/check6-lmcache.journal
python3 scripts/check6-lmcache-reaper.py judge --journal-file ~/check6-lmcache.journal
```

Output: PASS/FAIL/INVALID per criterion on stdout, the JSON summary in
`./check6-summary.json`, exit 0 PASS, 2 INVALID, 1 FAIL/INCOMPLETE/error.
`--dry-run` skips the wait and the quiet-period checks (verdicts
`SMOKE-PASS`/`SMOKE-INVALID`/`SMOKE-FAIL`, same codes). Other options:
`--wait-min`, `--base-url`, `--admin-url`, `--vllm-unit`/`--lmcache-unit`,
`--no-probe`, `--evict`, `--max-flood-tokens`; see `--help`.

PASS, all of:
1. No `Reaped GPU instance` line in `journalctl -u vllm-lmcache` since the
   vllm start (`pinged=False` or `True`).
2. `/status` `registered_gpu_ids` lists every GPU after the wait and after
   the requests.
3. Request 2 gets an external hit: `vllm:external_prefix_cache_hits` grows
   across it and `usage.prompt_tokens_details.cached_tokens` is above 0.

INVALID (exit 2) means the run did not test the bug: vLLM restarted, a
prompt over 200 tokens or an LMCache store arrived during the quiet hour,
the wait was shorter than the registration grace plus one reaper interval,
or the eviction did not take request 1 out of vLLM's GPU prefix cache
(raise `--max-flood-tokens` if the flood was capped). A reap or a wrong GPU
count is a FAIL even when the eviction failed. Against a build without the
patch, all three criteria FAIL.

## Qwen matrix, M7 and M8

Scripts: `sm70/scripts/qwen_phase2/`. They start their own `vllm serve` on
`127.0.0.1:8100`, write under `$OUT`, and never touch systemd or
the `vllm` unit. Production stopped first.

```sh
export VLLM_BIN=$OUT_VLLM/bin/vllm VLLM_PY=$GT/bin/python    # both required
export QWEN_STAGE=<dir with the Qwen target and DFlash2 draft dirs>
export OUT=~/qwen-phase2/$(date +%Y%m%dT%H%M)
sm70/scripts/qwen_phase2/m7_e2e.sh              # M7 DFlash2: triton parity + e2e
sm70/scripts/qwen_phase2/qwen_matrix.sh all     # rows 1-4; or 1 | 2 | 3 | 4
sm70/scripts/qwen_phase2/qwen_matrix.sh 4-noemu # M7 A/B, INFO only (also 2-noemu)
sm70/scripts/qwen_phase2/qwen_matrix.sh summary
```

`$OUT_VLLM` is the `.#vllm` outPath (the scripts use `$OUT` for results).
Overrides: `MAX_MODEL_LEN`, `HEALTH_TIMEOUT`, `GPU_MEM_UTIL`,
`QP2_ATTN_BACKEND`, `DFLASH2_E2E_TARGET`/`DFLASH2_E2E_DRAFT`. Exit 0 PASS, 1
FAIL, 2 refused (unset `VLLM_BIN`/`VLLM_PY`, a model dir without
`config.json`, port 8100 in use, or a GPU with more than 1 GiB used).

- `m7_e2e.sh` passes when every selected test passes; a skip is a FAIL.
- A matrix row passes when the probe's gates hold (see [quality
  probe](#quality-probe)); spec rows (2, 4) must also show more than 0 drafts
  and accepted tokens. `-noemu` rows are INFO.
- Results: `$OUT/m7/`, `$OUT/row<N>/` (`results.json`, `server.log`,
  `probe.log`, `cmd.txt`, `nvidia-smi.csv`); the last line per row names the
  failed gates.

M8 QPN8 A/B, after the M8 GPU tests pass:

```sh
$GT/bin/pytest -q -p no:cacheprovider -m gpu sm70/tests/qpn8
$GT/bin/pytest -q -p no:cacheprovider -m gpu sm70/tests/quant/test_m8_qpn8_wiring.py
sm70/scripts/qwen_phase2/m8_qpn8_ab.sh
```

It serves row 4 with `--max-num-seqs 4` twice: `on`
(`VLLM_SM70_FP8_BLOCK_QPN8=1`) and `off` (`=0`), with `QP2_QPN8_M32`
(default 1) and `QP2_QPN8_COPY=auto|true|false` on both arms. With
`QP2_QPN8_M32=0` and `auto`, the `on` arm stays on M6: set M32=1 or the copy
policy explicitly. `true` keeps a second weight copy and may OOM. Pass:
QPN8/M6 decode tok/s ≥ 0.98 at every concurrency and all 5 greedy outputs
match. In `on`, `$OUT/m8/on/m8_log.txt` must show
`SM70 FP8 QPN8 decode (M<=N) enabled`; `SM70 FP8 QPN8 not used` means the
A/B compared M6 with M6. Results: `$OUT/m8/`.
