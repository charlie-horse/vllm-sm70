# Flint on SWE-bench Verified (mini-swe-agent)

Harness used for `docs/benchmarks/flint_swebench_verified.md`.

- mini-swe-agent 2.4.6, official `config/benchmarks/swebench.yaml`, overlaid by
  `flint.yaml` (endpoint, sampling, no network in task container, 5 h timeout).
- `adapter/flint_model.py`: litellm model subclass that keeps the server's
  `reasoning` / `reasoning_content` on assistant messages in the history
  (model card requirement). Test: `adapter/test_flint_model.py`.
- swebench 5.0.2 official evaluation (`SWE-bench/SWE-bench_Verified`, docker
  images `swebench/sweb.eval.x86_64.*`).
- `scripts/run_one.sh ID ATTEMPT`: agent run, official eval, image cleanup.
  Resumable (skips instances with a `done_` marker).
- `scripts/analyze.py`: pass@1, CI, failure categories, token/time stats.
