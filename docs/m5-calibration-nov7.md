# M5 teacher-forced calibration and window run (c4130, Nov 7, 00:00 ET)

Bea runs everything below inside the GPU window. It calibrates the fork's
own numeric envelope with `sm70/scripts/m5_teacher_forced.py`, then runs our
M5 gates against it, then the mutation check. **Thresholds are provisional**:
Astro holds the final factors until Bea's no-autotune run reports. They are
constants at the top of `m5_teacher_forced.py` (`GATE_MEAN_FACTOR`,
`GATE_MAX_FACTOR`, `GATE_COUNT_FACTOR`, `ANCHOR_REDUCE`, `GATE_ANCHOR_PAIR`),
1.5x / 1.5x / 1.5x / max over sets today. The gate changed on Nov 7; see
[Gate changes](#gate-changes-quinn-nov-7).

## What is measured
- **Fixed sequences.** `gen` runs our eager greedy decode (64 tokens, one
  prompt at a time) on three disjoint sets of 8 short prompts (set 0 is the
  old T4/T5 prompt list). Every later score uses these token ids.
- **decode metric (new).** A V1 logits processor (`ForceRefLogitsProcessor`,
  passed by FQCN `m5_teacher_forced:ForceRefLogitsProcessor`) reads the
  reference continuation from `SamplingParams.extra_args["m5_force"]` and,
  at each step, masks the logits to -inf except the reference token. With
  `logprobs_mode="raw_logprobs"` and `logprobs=1`, the returned logprob of
  the forced token is the model's raw value, taken before any processor
  (vllm/v1/sample/sampler.py, `Sampler.forward`, on both stacks), and the
  raw top-1 is recorded too. So every continuation token is a real decode
  step (eager, or CUDA-graph replay in graphs mode). The script fails
  loudly (`ForcingError`) if the generated ids are not the reference.
- **prefill metric (old T4a).** `prompt_logprobs` over the whole sequence.
  It exercises prefill only.
- **One sequence at a time.** `max_num_seqs=1` and one `generate()` per
  sequence, so every decode step has M=1 in eager and in replay.
  `--max-num-seqs N` exists for batched experiments. Prefix caching is off.
- **One engine instance per score.** Each `score` run builds its own LLM
  (new engine core, new TP workers), scores, and shuts down. Each named
  score file (`eager`, `eager2`, `graphs`, `unfused`) therefore comes from a
  separate instance. The tests do the same: one LLM build per scored side.
- **Envelope.** For each pair and metric (decode, prefill): per set n, mean,
  max, count > 0.1 and count > 1.0 nats; anchor = max over sets of each.
  Pairs: `eager_vs_graphs` (eager vs graphs), `fused_vs_unfused` (eager
  vs eager with `VLLM_SM70_FLINT_QK_ROPE_KV=0`), `eager_vs_eager` (two
  separate eager instances: engine-instance nondeterminism, reported, not
  gated), `eager_vs_graphsonly` (eager vs mode NONE + FULL graphs, T4e's
  anchor; built when `setK/graphsonly.json` exists).
- **Gate.** For each set and metric: ours mean <= 1.5x that metric's anchor
  mean, ours max <= 1.5x the pair's anchor max over decode AND prefill, and
  ours count > 1.0 nats <= max(1, ceil(1.5x the pair's anchor count)). Each
  gate line also prints count > 0.1, our own eager-vs-eager instance noise
  for that set (`ours_eager_vs_eager`, written by T4c), the fork's
  (`fork_eager_vs_eager` anchor), the gate anchor and the effective limits.

| test | compares | metrics | anchor (fork) |
|---|---|---|---|
| T4a | our eager vs our graphs | decode, prefill | eager_vs_graphs |
| T4b | greedy 256 tok, eager vs mode NONE + cudagraph FULL, max_num_seqs 1 and 2; **TRITON only** (`VLLM_SM70_FLASH_V100=0`) | tokens identical | none |
| T4c | our eager vs our eager, two separate engine instances | decode bitwise (+ prefill stats, free-run first divergence) | fork eager_vs_eager printed |
| T4d | **our graphs vs the fork's graphs** (cross-stack) | decode | eager_vs_graphs |
| T4e | our eager vs our graphs-only (mode NONE + cudagraph FULL), forced | decode | eager_vs_graphsonly |
| T5 | free greedy vs the fork golden; near-tie divergence: our graphs vs the fork's graphs, forced on the golden sequence | first 32 tokens; decode | eager_vs_graphs |
| T7 | our fused vs our unfused, both eager | decode, prefill | fused_vs_unfused |

T4d exists because T4a compares our two modes with each other: a bug in
code both modes share (say a wrong RoPE position or KV slot in the fused
op) cancels out there. T4d puts our replayed decode next to the fork's on
the same sequences. The rule is that ours may differ from the fork by no more
than 1.5x what the fork's own eager and graphs differ by.

**T4c is expected to FAIL on today's build.** Bea saw (Nov 6) that two
separate eager instances give different free-run tokens on 7 of 8 prompts,
with real logprob shifts. The suspect is startup autotuning, such as
TurboMind kernel selection by timing. A T4c failure means "engine-instance
nondeterminism". It is a finding about the build, not a bug in the test.
T4c still writes `ours_eager_vs_eager` before its gate.

The fork stack is unaffected by the forcing processor in one respect worth
knowing. Its runner disables its greedy-token and top-k fast paths whenever
a non-builtin non-argmax logits processor is loaded, and whenever logprobs
are requested (`_non_argmax_logits_processors_inactive`,
`_can_use_sm70_compact_topk20_tokens`, `_sample_greedy_token_fastpath`). Its
scores therefore come from the full-logits sampler path, the same path ours
takes.

## Gate changes (Quinn, Nov 7)
The Nov 7 run raised three gate issues. Quinn's rulings:

1. **Max over decode and prefill, plus a > 1.0-nat count** (T4a, T4d, T4e,
   T7 and #86's G2). The single-token max is one fragile MoE routing
   position, and it lands in whichever metric it happens to hit (fork set1:
   decode max 2.616 at one token, prefill max 0.895; ours had a prefill set1
   max of 2.394). `effective_anchor(env, pair, metric)` now gives each
   (pair, set, metric) gate: the mean is unchanged (1.5x that metric's anchor
   mean); the max uses the pair's anchor max over **both** metrics (x1.5);
   and ours' count of tokens with |dlogprob| > 1.0 nats must be <= max(1,
   ceil(1.5x the pair's fork count over sets and metrics)). `stats()` and
   the anchors carry `over_1_0` next to `over_0_1`. **An envelope built
   before this change has no `over_1_0` and fails closed**: the test FAILs
   before loading a model and says to rebuild it (step 3, CPU only, from the
   existing score files). T4c's bitwise check is unchanged.
2. **T4b is exact only on TRITON; T4e replaces it under Flash-V100.** Exact
   greedy T4b assumes both arms run identical kernels. Under Flash-V100 the
   capture path uses envelope decode hints and the K14 one-launch decode, so
   they don't. T4b now runs only with `VLLM_SM70_FLASH_V100=0`; otherwise
   it skips, pointing at T4e, and the runner reports `t4b SKIPPED by
   design`. Mara is confirming this premise with data (the fork's own
   `greedy` eager vs graphs-only). **T4e** forced-decodes our eager vs our
   graphs-only (`compilation_config={"mode": 0, "cudagraph_mode": "FULL"}`;
   decode metric, max_num_seqs 1) and gates each set like T4a against the
   fork's `eager_vs_graphsonly` pair (`setK/eager.json` vs
   `setK/graphsonly.json`). If the envelope lacks that pair, T4e FAILs and
   names the missing file.
3. **T5 near-tie rule.** A golden-stable prompt may diverge before token 32
   only if both (a) the golden token at the first mismatch is our #2 within
   0.1 nats, and (b) forced decode of the golden sequence (prompt + golden
   continuation) with our graphs passes the T4d gate against the fork's
   forced decode of the same sequence (1.5x the `eager_vs_graphs` effective
   anchor). The fork file is `FLINT_SM70_T5_FORK_GOLDEN_SCORES` (step 2b),
   matched by prompt or `seqs_sha256`. If (a) holds but the file is unset
   or lacks the prompt, T5 FAILs with what to run. Our side costs one extra
   load, only when a near-tie needs it. One line per near-tie prompt:
   ```
   T5 near-tie '<prompt>': golden #2 gap 0.0300; forced decode ours vs fork mean=… max=… >1.0=0 (limits …) -> PASS
   ```

## Before you start
- **GPUs must be free.** Bea stops prod `vllm` and the LMCache server (it
  holds ~2.2 GiB per GPU). The runners refuse to start if any GPU has more
  than 1 GiB in use. Every LLM here uses `gpu_memory_utilization=0.85`; keep
  it at or below 0.85.
- Work from the repo checkout at `models/m5-t4-near-tie` (this commit or
  later). All commands below run from the repo root.
- Every step is one process; each model load takes about 2–3 min, and a
  graphs-mode load adds Inductor compile and capture (a few minutes more on
  a cold cache).

```sh
cd ~/vllm-sm70-work/bea/vllm-sm70          # checkout of models/m5-t4-near-tie
TF=sm70/scripts/m5_teacher_forced.py
W=~/vllm-sm70-work/bea/m5-cal-nov7; mkdir -p "$W"
# fork stack (Theo; read-only store paths)
FORK_PY=/nix/store/6sf22d6xfqx7787i163dz9g0rcvdqz3g-vllm-flint-c4130/bin/python
CKPT=/nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt
FORK_KW='{"kernel_config": {"sm70_skinny_moe": false}}'  # = prod --kernel-config.sm70_skinny_moe false
# our stack
VLLM_PY=...                                  # our venv/wrapper python with pytest
OURS_ARCH='{"architectures": ["FlintSm70ForCausalLM"]}'
export FLINT_SM70_CHECKPOINT=$CKPT           # same checkpoint on both stacks
nvidia-smi --query-gpu=memory.used --format=csv   # both GPUs ~0 MiB
```

On the fork, pass no `--hf-overrides`: its architecture is
`FlintForCausalLM`, the checkpoint default. The fork's unfused switch has
the same name, `VLLM_SM70_FLINT_QK_ROPE_KV=0`. `sm70_skinny_moe` is a field
of the fork's `KernelConfig` (vllm/config/kernel.py), and `LLM(kernel_config={...})`
turns the dict into a `KernelConfig` in `EngineArgs.__post_init__`.

**Deterministic kernels (REQUIRED, both stacks; Quinn/Astro Nov 6).**
TurboMind's small-shape autotune times kernels at engine start, so each
engine instance may pick different kernels and produce different bits (Bea:
two eager instances split on 7/8 prompts; identical with tuning off). Until
Kai's persisted tuning table lands, every step below runs with it off. The
fork reads the same variable names. `m5_teacher_forced.py`,
`m5_window.sh`, `m5_mutation_check.sh` and the test module all set these to
0 themselves (and print a `tuning:` line). Exporting them in the shell as
well makes it explicit:
```sh
export VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=0 VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0 \
       VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=0 VLLM_SM70_MXFP4_TUNE_SMALL_SHAPES=0
```
`M5_ALLOW_TUNING=1` turns the scripts' default off (diagnostics only, never
for gates). **T4c must PASS under this setting before T4a/T4d/T4e/T7 count.**
The runner reports them `NOT COUNTED` if T4c fails. Check that every log
has the `tuning: ...=0` line, the fork's included.

## 1. Sequences, on ours (1 load, ~5 min)
```sh
$VLLM_PY $TF gen --model "$CKPT" --hf-overrides "$OURS_ARCH" \
  --set 0 --set 1 --set 2 --out-dir "$W/seqs"
# -> $W/seqs/set{0,1,2}.json (8 sequences each, prompt + 64 eager greedy tokens)
```

## 2. Fork scores (5 loads, ~25–30 min)
Each command is a separate engine instance; do not merge them.
```sh
for name in eager eager2; do
  $FORK_PY $TF score --model "$CKPT" --seqs-dir "$W/seqs" --out-dir "$W/fork" \
    --name $name --mode eager --decode --prefill --llm-kwargs "$FORK_KW"
done
$FORK_PY $TF score --model "$CKPT" --seqs-dir "$W/seqs" --out-dir "$W/fork" \
  --name graphs --mode graphs --decode --prefill --llm-kwargs "$FORK_KW"
VLLM_SM70_FLINT_QK_ROPE_KV=0 $FORK_PY $TF score --model "$CKPT" --seqs-dir "$W/seqs" \
  --out-dir "$W/fork" --name unfused --mode eager --decode --prefill \
  --llm-kwargs "$FORK_KW" --env VLLM_SM70_FLINT_QK_ROPE_KV=0
# T4e's anchor: graphs-only (mode NONE + FULL CUDA graphs), a 5th instance
FORK_GO_KW='{"kernel_config": {"sm70_skinny_moe": false}, "compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}}'
$FORK_PY $TF score --model "$CKPT" --seqs-dir "$W/seqs" --out-dir "$W/fork" \
  --name graphsonly --mode graphs --decode --prefill --llm-kwargs "$FORK_GO_KW"
```
Each run prints `wrote .../setK/NAME.json: 8 sequences, decode forced OK, raw
top-1 != forced at x/512 steps`. A `ForcingError` means the stack bypassed
the processor: stop and report.

## 2b. Fork golden scores for T5 (1 load, ~5 min)
Forced decode of every golden sequence on the fork's graphs, for the T5
near-tie rule (Quinn, Nov 7). `--match <substring>` restricts it to one
prompt; T5 accepts either file.
```sh
GOLDEN=~/vllm-sm70-work/bea/golden/flint_cf9492a_greedy64.json
$FORK_PY $TF seqs-from-golden --golden "$GOLDEN" --out "$W/golden/set9.json"
$FORK_PY $TF score --model "$CKPT" --seqs "$W/golden/set9.json" --out-dir "$W/golden/fork" \
  --name fork_graphs --mode graphs --decode --llm-kwargs "$FORK_KW"
# -> $W/golden/fork/set9/fork_graphs.json (FLINT_SM70_T5_FORK_GOLDEN_SCORES, step 4)
```

## 3. Fork envelope (CPU, seconds)
```sh
$FORK_PY $TF envelope --scores-dir "$W/fork" --out "$W/fork_envelope.json"
```
This prints per pair, metric and set the mean, max, count > 0.1 and count >
1.0, then the ANCHOR lines and one GATE line per pair (max over both
metrics, the > 1.0 limit). **An envelope built before Nov 7 lacks
`over_1_0`, and every gate FAILs on it: rerun this same command to rebuild
it** (no GPU; any python works, `$VLLM_PY` too). With
`setK/graphsonly.json` present it adds `eager_vs_graphsonly`; without, it
prints `eager_vs_graphsonly: skipped` and T4e FAILs. Keep the output for Astro. The `eager_vs_eager` lines answer
whether the fork shares our engine-instance nondeterminism.

Optional, standalone (T4c in step 4 produces the same file). Our own
eager-vs-eager noise, for instance with the autotune-off knob, uses 2 loads:
```sh
for name in eager eager2; do
  $VLLM_PY $TF score --model "$CKPT" --hf-overrides "$OURS_ARCH" --seqs-dir "$W/seqs" \
    --out-dir "$W/ours" --name $name --mode eager --decode --prefill
done
$VLLM_PY $TF envelope --scores-dir "$W/ours" --pairs eager_vs_eager --out "$W/ours_eager_vs_eager.json"
```

## 4. Our window runner (12–13 model loads + T1/T2, ~50–55 min)
```sh
export VLLM_PY FLINT_SM70_GOLDEN=~/vllm-sm70-work/bea/golden/flint_cf9492a_greedy64.json
export FLINT_SM70_T4_SEQS_DIR="$W/seqs"
export FLINT_SM70_T4_ENVELOPE="$W/fork_envelope.json"
export FLINT_SM70_FORK_SCORES_DIR="$W/fork"
export FLINT_SM70_T5_FORK_GOLDEN_SCORES="$W/golden/fork/set9/fork_graphs.json"
export OUT="$W/window"        # FLINT_SM70_OURS_NOISE defaults to $OUT/ours_eager_vs_eager.json
bash sm70/scripts/m5_window.sh
```
Node order: `t4c t4a t4b t4e t4d t5 t7 t2 t3 t1 t6`. t4c runs first because
it writes `ours_eager_vs_eager`, which T4a, T4d, T4e and T7 print. Loads:
t4c 2, t4a 2, t4b 4 (TRITON only, 0 under Flash-V100), t4e 2, t4d 1, t5 1
(+1 when a near-tie needs rule (b)), t7 2, t3 1, t6 1: 12–13 by default. A
skip counts as FAIL, as before, except t4b's by-design skip under
Flash-V100 (`t4b SKIPPED by design`). Exact T4b on TRITON, separately (4
loads): `VLLM_SM70_FLASH_V100=0 M5_NODES=t4b bash sm70/scripts/m5_window.sh`.
The verdict section greps `T4a/T4b/T4c/T4d/T4e/T5/T7` lines. A gate line
looks like:
```
T4a eager_vs_graphs decode set1: ours mean=… max=… >0.1=3/512 >1.0=0 | ours_eager_vs_eager mean=… | fork_eager_vs_eager mean=… | anchor mean=… max=… >0.1=… >1.0=… | limits mean<=… (1.5x), max<=… (1.5x max over decode+prefill), >1.0<=1 (max(1, ceil(1.5x0))) -> PASS
```
Expected today: t4c FAIL (engine-instance nondeterminism, see above). Report
every other FAIL.

## 5. Mutation check (11 loads, ~30–40 min)
Run this only after T4d and T7 have passed unmutated in step 4, or add
`M5_BASELINE=1`, which costs 3 more loads.
```sh
bash sm70/scripts/m5_mutation_check.sh       # same env as step 4; OUT defaults under m5-mutation/
```
Planted bugs and the gates that MUST catch them (Quinn/Astro, Nov 6):

| mutation | what it breaks | must FAIL |
|---|---|---|
| `rope_pos_decode` | positions + 1 on fused-op calls of ≤ 8 tokens | T4d and T7 |
| `kv_slot_decode` | slot_mapping + 1 on fused-op calls of ≤ 8 tokens | T4d and T7 |
| `moe_decode_scale` | K12 M≤2 MoE decode output × 1.01 | T4d (T7 can't see it: both arms run the same MoE) |
| `rope_theta_1p01` | rope_theta × 1.01 via hf_overrides | INFO only (too weak at ~70-token positions) |

The check PASSes only if every required run fails the gate itself (not a
crash, not a skip) and shows the mutation applied: `M5_MUTATION <name>
installed in pid N` from at least 3 processes (2 for `moe_decode_scale`,
which only the TP workers import), or the `hf_overrides` line for
rope_theta. INFO runs are reported with an `INFO` prefix and never fail the
check. Hook mutations must also print `M5_MUTATION <name> fired in pid N`
(the first call they actually changed); a run that installed but never
fired is reported `NOT FIRED`, not "not caught". For `moe_decode_scale`
that usually means K12 was not admitted: look for the "K12 M<=2 decode
admitted" line (#76). `moe_decode_scale` only bites if the build has K12 and M4 admits the
decode: with forced decode at max_num_seqs=1 every step is M=1, and
since #76 the engine log shows "K12 M<=2 decode admitted".
`m5_mutations/sitecustomize.py` runs any shadowed sitecustomize first (the
Nix wrapper ships one), so the environment stays intact.

## 6. Flash-V100 for Flint (Kai's checks; 2 loads + GPU tests + bench, ~25 min)
Since this branch, Flint picks Flash-V100 (`AttentionBackendEnum.CUSTOM`,
sink prefill and decode) when no backend is given; `VLLM_SM70_FLASH_V100=0`
selects TRITON_ATTN. Every M5 run above already uses it, so check the route
before trusting their numbers.

1. **Route, from any engine log of steps 1–5.** Expect
   `Flint SM70: using FLASH_ATTN_V100 (CUSTOM)` and, in graphs mode,
   `Flash-V100 grouped sink decode in captured CUDA graphs so far: N
   one-launch, 0 two-launch` (K14 fused decode). FAIL on
   `Flint SM70: using TRITON_ATTN ... because <reason>`, on
   `vllm_sm70 Flash-V100: using the Triton forward`, or on
   `one-launch sink decode is off`.
   ```sh
   grep -h "Flint SM70: using\|Triton forward\|grouped sink decode\|sink decode is off" "$W"/window/*.log
   ```
   Also send Mara/Kai every `K12 decode skipped for an M=… call (<reason>)`
   line (#79; names the MoE layer that misses K12 at decode):
   ```sh
   grep -h "K12 decode skipped\|K12 M<=2 decode" "$W"/window/*.log | sort | uniq -c
   ```
2. **Kernel tests.**
   ```sh
   $VLLM_PY -m pytest -m gpu -q sm70/tests/attention/test_fv100_fused_decode_gpu.py \
     sm70/tests/attention/test_fv100_decode_sinks_gpu.py \
     sm70/tests/attention/test_fv100_prefill_sinks_gpu.py
   ```
3. **Batched greedy, Flash-V100 vs Triton** (graphs mode, max_num_seqs 20,
   batches of 2/3/4/8/20, each run twice):
   ```sh
   FV=sm70/scripts/m5_fv100_check.py
   $VLLM_PY $FV gen --model "$CKPT" --hf-overrides "$OURS_ARCH" --backend fv100  --out "$W/fv100.json"
   $VLLM_PY $FV gen --model "$CKPT" --hf-overrides "$OURS_ARCH" --backend triton --out "$W/triton.json"
   $VLLM_PY $FV compare "$W/fv100.json" "$W/triton.json" --show
   ```
   Divergence after a shared prefix is expected (MoE routing flips on last
   bits). FAIL on run-to-run differences (zero with tuning off; on
   Flash-V100 a diff points at the K14 fused decode), a degenerate or first-token
   divergence (the script flags these), or text that reads as gibberish.
4. **Bench.** Start `vllm serve` on the window stack with prod flags (no
   `--attention-backend`) and run `~/archive/scripts/bench_vllm.py` at 4K
   and 32K. Fork today: 177 tok/s (4K), 163 tok/s (32K). Ours on
   TRITON_ATTN was 47 tok/s at 32K, TTFT 31.5 s. Report decode tok/s and
   TTFT next to the fork's.

## Total
About 2 h 20 min of GPU time with no retries: step 1 ~5 min, step 2 ~30 min,
step 2b ~5 min, step 4 ~55 min, step 5 ~35 min, step 6 ~25 min. If the
window is short, run steps 1–3 and then `M5_NODES="t4c t4a t4e t4d t7"` (9
loads) before the full runner.
