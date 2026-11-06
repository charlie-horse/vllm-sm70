# M5 teacher-forced calibration and window run (c4130, Nov 7, 00:00 ET)

Bea runs everything below inside the GPU window. It calibrates the fork's
own numeric envelope with `sm70/scripts/m5_teacher_forced.py`, then runs our
M5 gates against it, then the mutation check. **Thresholds are provisional**:
Astro holds the final factors until Bea's no-autotune run reports. They are
constants at the top of `m5_teacher_forced.py` (`GATE_MEAN_FACTOR`,
`GATE_MAX_FACTOR`, `ANCHOR_REDUCE`, `GATE_ANCHOR_PAIR`), 1.5x / 1.5x / max
over sets today.

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
  max and count > 0.1 nats; anchor = max over sets of the mean and of the
  max. Pairs: `eager_vs_graphs` (eager vs graphs), `fused_vs_unfused` (eager
  vs eager with `VLLM_SM70_FLINT_QK_ROPE_KV=0`), `eager_vs_eager` (two
  separate eager instances: engine-instance nondeterminism, reported, not
  gated).
- **Gate.** For each set: ours mean <= 1.5x anchor mean AND ours max <= 1.5x
  anchor max. Each gate line also prints count > 0.1, our own eager-vs-eager
  instance noise for that set (`ours_eager_vs_eager`, written by T4c) and
  the fork's (`fork_eager_vs_eager` anchor), next to the gate anchor.

| test | compares | metrics | anchor (fork) |
|---|---|---|---|
| T4a | our eager vs our graphs | decode, prefill | eager_vs_graphs |
| T4b | greedy 256 tok, eager vs mode NONE + cudagraph FULL, max_num_seqs 1 and 2 | tokens identical | none |
| T4c | our eager vs our eager, two separate engine instances | decode bitwise (+ prefill stats, free-run first divergence) | fork eager_vs_eager printed |
| T4d | **our graphs vs the fork's graphs** (cross-stack) | decode | eager_vs_graphs |
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

Optional autotune-off knob (Kai; name TBD). If Kai provides it, set it in
the shell **and** record it with `--env` on every `score` command:
```sh
AUTOTUNE_OFF_ENV=<from Kai>                  # e.g. NAME=0
# then:  env "$AUTOTUNE_OFF_ENV" $PY $TF score ... --env "$AUTOTUNE_OFF_ENV"
```
`--env K=V` sets the variable in the script process before the LLM is built
(engine core and workers inherit it) and records it in the score JSON.
Exporting it in the shell as well covers anything read at import time.

## 1. Sequences, on ours (1 load, ~5 min)
```sh
$VLLM_PY $TF gen --model "$CKPT" --hf-overrides "$OURS_ARCH" \
  --set 0 --set 1 --set 2 --out-dir "$W/seqs"
# -> $W/seqs/set{0,1,2}.json (8 sequences each, prompt + 64 eager greedy tokens)
```

## 2. Fork scores (4 loads, ~20–25 min)
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
```
Each run prints `wrote .../setK/NAME.json: 8 sequences, decode forced OK, raw
top-1 != forced at x/512 steps`. A `ForcingError` means the stack bypassed
the processor: stop and report.

## 3. Fork envelope (CPU, seconds)
```sh
$FORK_PY $TF envelope --scores-dir "$W/fork" --out "$W/fork_envelope.json"
```
This prints per pair, metric and set the mean, max and count > 0.1, then the
ANCHOR lines. Keep the output for Astro. The `eager_vs_eager` lines answer
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

## 4. Our window runner (14 model loads + T1/T2, ~50–60 min)
```sh
export VLLM_PY FLINT_SM70_GOLDEN=~/vllm-sm70-work/bea/golden/flint_cf9492a_greedy64.json
export FLINT_SM70_T4_SEQS_DIR="$W/seqs"
export FLINT_SM70_T4_ENVELOPE="$W/fork_envelope.json"
export FLINT_SM70_FORK_SCORES_DIR="$W/fork"
export OUT="$W/window"        # FLINT_SM70_OURS_NOISE defaults to $OUT/ours_eager_vs_eager.json
bash sm70/scripts/m5_window.sh
```
Node order: `t4c t4a t4b t4d t5 t7 t2 t3 t1 t6`. t4c runs first because it
writes `ours_eager_vs_eager`, which T4a, T4d and T7 print. Loads: t4c 2, t4a
2, t4b 4, t4d 1, t5 1, t7 2, t3 1, t6 1. A skip counts as FAIL, as before.
The verdict section greps `T4a/T4b/T4c/T4d/T5/T7` lines. A gate line looks
like:
```
T4a eager_vs_graphs decode set1: ours mean=… max=… >0.1=3/512 | ours_eager_vs_eager mean=… | fork_eager_vs_eager mean=… | anchor mean=… max=… | limit 1.5x mean, 1.5x max -> PASS
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
check. `moe_decode_scale` only bites if the build has K12 and M4 admits the
decode: with forced decode at max_num_seqs=1 every step is M=1, and
since #76 the engine log shows "K12 M<=2 decode admitted".
`m5_mutations/sitecustomize.py` runs any shadowed sitecustomize first (the
Nix wrapper ships one), so the environment stays intact.

## Total
About 1 h 45 min of GPU time with no retries: step 1 ~5 min, step 2 ~25 min,
step 4 ~60 min, step 5 ~35 min. If the window is short, run steps 1–3 and
then `M5_NODES="t4c t4a t4d t7"` (7 loads) before the full runner.
