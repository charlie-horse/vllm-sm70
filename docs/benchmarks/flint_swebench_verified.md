# Flint on SWE-bench Verified (2x V100, NVFP4 W4A16)

Status: FINAL. Numbers from `scripts/analyze.py` on the completed run.

## Summary

- Flint, served by this vLLM fork on 2x V100 (SM70, TP2), NVFP4 W4A16
  weights, FP16 activations.
- SWE-bench Verified, 500 instances x 1 attempt, mini-swe-agent 2.4.6 with
  the official SWE-bench config, official swebench 5.0.2 evaluation.
- pass@1: **74.2%** (371 / 500 scored instances),
  95% Wilson CI 70.2% – 77.8%.
- 0 instances excluded as infra failures (not counted against
  the model; listed in Results).
- Model card claim: 74.2% (BF16). Difference: 0.0 pp.
  The NVFP4 W4A16 build on V100 matches the claim: the point estimate is identical and the CI contains 74.2%.

## Claim vs result

| | Model card | This run |
|---|---|---|
| Checkpoint | BF16 | NVFP4 W4A16, FP16 activations |
| Hardware | not stated | 2x V100 32 GB, TP2 |
| Agent | mini-swe-agent (version, config, step limit not stated) | mini-swe-agent 2.4.6, official `swebench.yaml`, step limit 250 |
| Attempts | average pass@1 over >= 5 attempts | 1 attempt per instance |
| Instances | 500 | 500 (500 scored, 0 infra-excluded) |
| Internet in task env | none | none (`--network=none`) |
| Agent timeout | up to 5 h | 5 h container timeout |
| pass@1 | 74.2% | 74.2% (95% CI 70.2% – 77.8%) |

Sources: huggingface.co/RecursiveSI/flint (model card) and the accompanying
blog post. The blog states NVFP4 is within 0.7 pts of 16-bit across their
benchmarks, so the expected NVFP4 result is roughly 73.5–74.2% before
sampling noise.

## Method

- Instances: all 500 of `SWE-bench/SWE-bench_Verified` (test split), run in
  a fixed shuffled order (`all_ids_shuffled.txt`). One attempt per instance.
- Why one attempt: a single attempt over all 500 instances is an unbiased
  estimate of mean pass@1. The CI is dominated by instance variance, not
  attempt variance: the pilot-based estimate is about ±4 pp for 500x1 vs
  about ±3.5 pp for 500x5, at 5x the cost. Approved by the user before the
  run.
- Sampling (client): temperature 1.0, top_p 0.95, max_tokens 65536 (the
  NVFP4 card's client budget; the card's minimum is 32768). top_k 20 comes
  from the server default. Context length 262144.
- Server: this fork, Nix flake input locked at `b7b1753` (= main `c52fd8d`
  plus `nix/` packaging), built for sm_70 / CUDA 12.8. 2x Tesla
  V100-PCIe-32GB, no NVLink, TP2, host c4130. Same config for the whole
  run (the Sat 12:44 EDT restart changed only the LMCache disk cap).
- Weights: RecursiveSI/flint NVFP4 GGUF, repacked losslessly to a ModelOpt
  NVFP4 W4A16 checkpoint (`vllm.models.flint.convert_gguf` at `f66728d`):
  E2M1 weights and E4M3 block scales unchanged, BF16 tensors cast to FP16
  (range-checked). FP16 overflow is avoided with exact power-of-two shifts
  from layer 46 (MoE intermediate /16, residual stream /8).
- Serve args: `--tensor-parallel-size 2 --dtype half --max-model-len 262144
  --max-num-seqs 2 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90 --enable-prefix-caching
  --enable-auto-tool-choice --kv-cache-dtype auto
  --kernel-config.sm70_skinny_moe false --enable-prompt-tokens-details
  --tool-call-parser glm47 --reasoning-parser glm47
  --exclude-tools-when-tool-choice-none --override-generation-config
  '{"temperature":1.0,"top_p":0.95,"top_k":20}' --served-model-name flint`,
  async scheduling on. KV offload via LMCache (MP connector, chunk 256,
  64 GB host RAM L1, disk L2 180 GB then 100 GB, recompute on load
  failure). Decode: 178 tok/s single stream at 4K context.
- Reasoning round-trip (card requirement: keep reasoning in the history).
  The server returns `message.reasoning` (not `reasoning_content`). Stock
  mini-swe-agent 2.4.6 + litellm 1.106.6 already preserve it. The adapter
  `FlintModel` (`adapter/flint_model.py`) pins this by copying it into
  `reasoning_content` on the assistant message, and the chat template
  renders both. Check: a two-turn probe grew from 23 to 131 prompt tokens
  when reasoning was carried over. Adapter unit tests: 6/6 pass.
- Per instance (`r440/scripts/run_one.sh`): rate-limited image pre-pull,
  agent run, official `run_evaluation` on the produced `preds.json`, image
  cleanup. Two instances run in parallel (`run_all.sh`, `xargs -P2`),
  matching the server's two sequences.
- Scoring (`scripts/analyze.py`): each instance gets exactly one category.
  pass@1 = resolved / (done - infra). Infra categories (runner, API, eval,
  agent sandbox) are excluded from the denominator and listed separately.
  By default an eval-side test timeout is also treated as infra; the
  alternative count with `--timeout-is-model` is reported below.
- Retries that are not counted as attempts:
  - API failures: each model call retries for about 20 min (tenacity, 25
    attempts); if the agent still exits on an API error, the instance is
    re-run from scratch, up to 3 tries total. The previous trajectory is moved
    to `agent_retired/`.
  - Memory watchdog: if host MemAvailable drops below 8 GB, the newest task
    container is killed and the instance is re-run in a later pass (up to 3
    passes).
  - A failed image pull backs off up to about 2 h before the instance is
    marked `pull_failed` (infra).

## Harness versions

| Component | Version / source |
|---|---|
| mini-swe-agent | 2.4.6, official `config/benchmarks/swebench.yaml` |
| Step limit | 250 (official config) |
| Cost limit | not effective (`cost_tracking: ignore_errors`) |
| litellm | 1.106.6 |
| swebench | 5.0.2, official `swebench.harness.run_evaluation` |
| Dataset | `SWE-bench/SWE-bench_Verified`, test split |
| Eval images | `docker.io/swebench/sweb.eval.x86_64.*:latest` |
| Python | 3.12 (uv venv) |
| Container runtime | rootless podman (nix) on r440, Docker API socket |
| Overlay config | `benchmarks/flint_swebench/r440/flint.yaml` |

`SWE-bench/SWE-bench_Verified` has the same 500 instance IDs as
`princeton-nlp/SWE-bench_Verified`; swebench 5.x needs its `image` column.

## Deviations from the reference setup

The card does not publish its mini-swe-agent config, so "reference" here
means the official mini-swe-agent SWE-bench config.

| Deviation | Reason | Expected effect |
|---|---|---|
| Task container `--network=none` | Card: no internet for SWE-bench | Matches the card |
| `container_timeout: 5h` | Card blog: agent timeouts up to 5 h | Matches the card |
| Rootless podman instead of Docker | No Docker daemon on r440 | None on results |
| overlay `ignore_chown_errors=true` | matplotlib images have files owned by UIDs above the subuid range; those files map to root | None: tests run as root |
| nice 19 / ionice idle, `--cpus=6` per task container | Shared host; 2 parallel tasks use at most 12 cores | Slower tool calls only |
| 2 parallel agents | Server `max-num-seqs 2` | None on results |
| NVFP4 W4A16 on V100 instead of BF16 | Hardware | Card blog: within 0.7 pts |
| 1 attempt instead of >= 5 | Cost; see Method | Wider CI, same expectation |

## Validity checks

Audit by Nadia:

- Gold-patch sanity eval through the r440 podman setup
  (`astropy__astropy-12907`, run id `gold_r440`): resolved.
- Negative control (patch that only edits the README): not resolved.
- Resolved reports were checked to have actually run their FAIL_TO_PASS
  and PASS_TO_PASS tests.
- Model patches touch source files only and differ from the gold patches.
- Leakage attempts seen in trajectories (`pip download` of newer releases,
  `git log --all`) were blocked: no network in the task container, and the
  repos in the images contain no fix commits.
- All 6 context overflows hit at 191–196K prompt tokens: prompt + `max_tokens` 65536 exceeded the 262144 window. These are model failures (the agent's history grew too long); none were re-run.
- No instance hit the 250-step limit; 494 of 500 agents submitted.
- The 8 wrong patches with non-standard analyzer labels (no tests
  collected, missing module, patch did not apply) were audited by hand.
  All 8 patches applied cleanly and all 8 are model failures:
  - 4 ran normally and failed the target test (the label was wrong).
  - 3 hit import errors because the hidden tests import helpers or
    dependencies the model's patch didn't add (pylint 4551, 4604, 4661).
  - 1 crashed the test run with the original bug (django 10554).
  None showed a missing image dependency, network error, timeout, OOM or
  killed container.

## Results

pass@1: 74.2% (371 / 500), 95% Wilson CI
70.2% – 77.8%.
With eval test timeouts counted as model failures (`--timeout-is-model`):
74.2% (unchanged: no eval test timeouts).

### Per repo

| Repo | Resolved | Scored | Rate | Infra-excluded |
|---|---|---|---|---|
| astropy/astropy | 15 | 22 | 68.2% | 0 |
| django/django | 180 | 231 | 77.9% | 0 |
| matplotlib/matplotlib | 23 | 34 | 67.6% | 0 |
| mwaskom/seaborn | 1 | 2 | 50.0% | 0 |
| pallets/flask | 1 | 1 | 100.0% | 0 |
| psf/requests | 3 | 8 | 37.5% | 0 |
| pydata/xarray | 19 | 22 | 86.4% | 0 |
| pylint-dev/pylint | 3 | 10 | 30.0% | 0 |
| pytest-dev/pytest | 17 | 19 | 89.5% | 0 |
| scikit-learn/scikit-learn | 28 | 32 | 87.5% | 0 |
| sphinx-doc/sphinx | 27 | 44 | 61.4% | 0 |
| sympy/sympy | 54 | 75 | 72.0% | 0 |
| **Total** | 371 | 500 | 74.2% | 0 |

### Failure breakdown

| Category | Count | Share of scored |
|---|---|---|
| Resolved | 371 | 74.2% |
| Wrong patch (eval ran, not resolved) | 123 | 24.6% |
| No patch or step/time limit | 0 | 0.0% |
| Context overflow | 6 | 1.2% |
| Format / tool errors | 0 | 0.0% |
| Other (inspected manually) | 0 | 0.0% |
| Infra, excluded | 0 | n/a |

Infra-excluded instances and reasons: none.

Of the wrong patches, 0 failed to apply. `django__django-10554` is reported
as not applied only because its test run crashed (audit above).

## Cost

Per scored instance (mean / median / p90):

| Metric | Mean | Median | p90 |
|---|---|---|---|
| Agent turns | 58.4 | 51 | 109 |
| Prompt tokens (sum over calls) | 2.58M | 1.38M | 7.01M |
| Completion tokens | 31.7K | 24.8K | 71.3K |
| Completion tokens per call | 487 | 442 | 824 |
| Peak context (prompt tokens) | 57.4K | 47.5K | 121.8K |
| Agent wall time | 12.2 min | 8.6 min | 28.0 min |
| Eval wall time | 50 s | 32 s | 110 s |

- Largest peak context in the run: 196,438 of 262144.
- Total run wall time: 63.1 h (start 2026-10-31 10:53 EDT, end
  2026-11-03 01:02 EST).
- Total tokens: 1.29B prompt, 15.8M
  completion.

Pilot (cloud sandbox, 10 instances): about 100 tok/s per stream decode with
2 streams; model time was about 60% of agent wall time. 7 resolved, 3 not
scored (Docker Hub 429 on image pulls).

Agent wall time includes API retry waits during the incidents below.

## Infra incidents

None of these were counted against the model. Instances whose agent hit an
API error were re-run from scratch (see Method); instances that still failed
are in the infra-excluded list.

| When (EDT) | Event | Handling |
|---|---|---|
| Oct 31 ~10:38 | vLLM restart (switch window) | Ridden out by per-call retries (~20 min), then up to 2 re-runs |
| Oct 31 12:44–12:47 | vLLM restart (LMCache cap change) | Same |
| Oct 31 ~13:20–14:15 | k8s rollout on r440 (cri-o to containerd, firewall reloads) | No effect: rootless podman is independent of cri-o; 0 infra errors |
| Oct 31 (smoke run) | Watchdog inherited the driver's lock fd and blocked the next driver start | Fixed: fd 8 closed for the watchdog (`1107909`) |
| Oct 31 13:36 | matplotlib image pulls failed on high-UID files | Fixed: overlay `ignore_chown_errors=true` (`b20d887`) |
| Oct 31 17:03 | `pytest-dev__pytest-7432`: transient layer error matched as missing image | Fixed match (`c1bd2a1`); instance re-queued |
| Nov 1 01:03 | Image prune raced a concurrent pull (layer not found) | Fixed: cleanup holds the pull lock (`3eb821e`) |

Fix commits on `claude/flint-swebench`:

```
23e15aa flint_swebench: add wire_probe helper used by the adapter test
3eb821e flint_swebench: serialize image cleanup with pulls
c1bd2a1 flint_swebench: don't treat transient layer errors as missing images
b20d887 flint_swebench: tolerate high-UID files in rootless podman pulls
1107909 flint_swebench: keep watchdog from inheriting the driver lock
f27f57b flint_swebench: r440 podman runner and infra-aware analysis
fb1e808 Add SWE-bench Verified harness for Flint (mini-swe-agent)
```

## Follow-up: quality gate for `--max-num-seqs 20` (Nov 3–4)

The server moved from `--max-num-seqs 2` to 20, and that triggered a
regression: gibberish after about 64 tokens, set off by concurrent and
prefix-cache-hit requests. After the fix, two gates checked it.

### Gate A: coherence (fixed build, hand-run, seqs 20)

Suite: `benchmarks/flint_coherence/` (`gate_a.py`, `near_tie.py`).

- 24 greedy prompts (code, math, prose, 8K–32K context), run at
  concurrency 1, 4 and 16, plus a repeat pass that re-sends every prompt
  (prefix-cache hits). Every output passes the gibberish detectors (zlib
  ratio, 8-gram repetition, non-ASCII ratio, Python parse), with 0 request
  errors.
- Greedy output is not bit-reproducible: it diverges from the c1 reference
  after 30–600 tokens, even on a c1 resend. Two checks show these are
  near-ties, not corruption:
  - A review panel judged all 96 divergences benign.
  - At the divergence index, the top-1/top-2 logprob gap is < 0.5 nats for
    79/83 checked points (median 0.11, max 1.09, the larger ones synonym
    swaps). Only 7% of all positions have a gap that small.
- `prompt_logprobs` (teacher-forced scoring) ran out of GPU memory and took
  the server down at a 2.1K-token context with 0.90 GPU utilization. This
  was already known and is tracked separately. That check is dropped for
  this box.
- `chat_template_kwargs.enable_thinking=false` is ignored: the template
  renders the same prompt either way. Tracked separately.

Verdict: PASS. The gibberish regression is fixed.

### Gate B: 50-task SWE-bench slice (production unit, seqs 20, LMCache)

Same agent and eval config as the 500-instance run. The slice is 50
instances, stratified by repo with seed 0 (`r440/slice50_ids.txt`). Run at
P=12 on production fork `e66530b` (fix `7b2bace`), Nov 4 06:47–12:01 EST.

| | Baseline, same 50 ids (seqs 2, P2) | Gate B (seqs 20, P12) |
|---|---|---|
| Resolved | 37/50 (74.0%) | 32/50 (64.0%), Wilson 50.1–75.9% |
| Discordant | 7 baseline-only | 2 run-only |
| Paired test | | McNemar exact p = 0.18 |
| Infra-excluded | 0 | 0 |
| Mean turns / completion tokens | 65.4 / 39.2K | 74.8 / 40.7K |
| Median agent wall per task | 606 s | 2,870 s |
| Aggregate completion tok/s | 70.7 (full run) | 108 (server counter: ~110) |
| Wall for 50 tasks | ~6.1 h (estimated, at P2) | 5.2 h |

Audit:

- All 9 discordant instances were read side by side. Every one is a
  coherent trajectory with a different fix, so they are reasoning variance.
  Example: sklearn-14629 patched `_validation` instead of adding
  `MultiOutputClassifier.classes_`.
- A mechanical scan of all trajectories in both runs finds the same rates
  in each:
  - messages with an 8-gram repeated ≥5×: 48/2537 vs 50/2356
  - zlib outliers: 2 vs 3
  - format errors: 0 vs 1
  - `finish_reason=length`: 0 vs 0
  - non-ASCII: 0.31% vs 0.26%, all ordinary punctuation
- There is no sign of the regression in agent output.

Confounds:

- Task containers got `--cpus=1` (12 in parallel) instead of 6. That gave
  157 vs 37 tool-command timeouts and longer trajectories. For example,
  django-14792 hit 28 timeouts, all after its fix was already chosen.
- One run-only win, matplotlib-21568, copied the fix from a newer matplotlib
  wheel cached in the image's pip cache. That is leakage. Without it the
  slice is 31/50.

Verdict:

- No detectable quality regression at this sample size. The 10-point gap is
  not significant (p = 0.18), and no failure traces to the server.
- 50 tasks cannot rule out a drop of a few points; that would need a larger
  rerun with `--cpus` matched.
- Throughput: about 1.5× aggregate generation per wall hour. Per-task
  latency is about 4.7× higher, so end-to-end wall for a fixed batch
  improved only about 1.2×.

## Reproduction

Files: `benchmarks/flint_swebench/` (r440 copies in `r440/`, installed on
r440 as `~/flint-swebench/{env.sh,flint.yaml,conf/,scripts/,adapter/}`).

1. Serve Flint (c4130, 2x V100) with the serve args under Method. The runner expects
   `$FLINT_API_BASE/models` to list `flint`.
2. Environment (r440):

   ```bash
   . ~/flint-swebench/env.sh
   cd $R
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python \
     mini-swe-agent==2.4.6 litellm==1.106.6 swebench==5.0.2
   ```

3. Adapter tests:

   ```bash
   cd $R && .venv/bin/python -m pytest adapter -q
   ```

4. Gold sanity eval:

   ```bash
   cd $R/gold && .venv/bin/python -m swebench.harness.run_evaluation \
     -d SWE-bench/SWE-bench_Verified -s test -p gold \
     -i astropy__astropy-12907 --max_workers 1 -id gold_r440
   ```

5. Full run (detaches; log in `runs/driver.log`; resumable):

   ```bash
   $R/scripts/run_all.sh $R/all_ids_shuffled.txt 1
   ```

   Single instance: `$R/scripts/run_one.sh <INSTANCE_ID> 1`. The agent
   command inside `run_one.sh` is:

   ```bash
   PYTHONPATH=$R/adapter timeout 6h nice -n 19 ionice -c3 \
     $R/.venv/bin/mini-extra swebench --subset verified --split test \
     --filter "^${ID}\$" -c swebench.yaml -c $R/flint.yaml \
     -c "model.model_kwargs.api_base=$FLINT_API_BASE" \
     -w 1 -o $R/runs/att1/agent/$ID
   ```

   and the evaluation:

   ```bash
   nice -n 19 ionice -c3 $R/.venv/bin/python -m swebench.harness.run_evaluation \
     -d SWE-bench/SWE-bench_Verified -s test -i $ID \
     -p $R/runs/att1/agent/$ID/preds.json --max_workers 1 -id att1_$ID \
     --timeout 1800
   ```

6. Analysis:

   ```bash
   $R/.venv/bin/python $R/scripts/analyze.py --runs $R/runs
   $R/.venv/bin/python $R/scripts/analyze.py --runs $R/runs \
     --timeout-is-model --out $R/runs/analysis_timeout_is_model
   ```

## Raw data

On r440 at `~/flint-swebench/runs` (not committed):

- `att1/agent/<ID>/`: trajectories, `preds.json`, exit statuses;
  `att1/agent_retired/`: trajectories replaced by API re-runs.
- `att1/eval/`: `done_<ID>` markers, swebench reports and logs.
- `att1/log/`: per-instance agent, eval and runner logs.
- `driver.log`, `events.log`: runner and incident logs.
- `analysis.md`, `analysis.json`: output of `analyze.py`.
- `~/flint-swebench/gold/`: gold sanity eval.
