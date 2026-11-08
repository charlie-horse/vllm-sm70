# Hardware regression checks

## GPU pytest suite (needs the GPUs; run in a window)

```sh
nix build .#gpu-tests -o ~/vsm70/result-gpu-tests && ~/vsm70/result-gpu-tests/bin/pytest -q -m gpu sm70/tests/<dir-or-file>
```

`.#gpu-tests` is the `.#vllm` python plus pytest (same Python, from the pinned nixpkgs); `bin/python` is the same wrapper without pytest's flags. Without the GPUs, `bin/pytest --collect-only -m gpu sm70/tests` still collects.

## LMCache build check (CPU only, any time)

Checks the LMCache that `nix/lmcache-build.nix` builds into `.#vllm`, on the
serving host, without the GPUs (`CUDA_VISIBLE_DEVICES` is empty), sudo or
services. Run it after realising a new `$OUT` and before any switch to it.

Script: `scripts/check-lmcache-cpu.sh` (bash; needs only `$OUT` and `nix-store`).

```sh
nix-store --realise $OUT --add-root ~/vsm70/result-vllm --indirect
scripts/check-lmcache-cpu.sh $OUT        # exit 0 = all PASS
```

| # | check | guards against |
|---|---|---|
| 1 | import `lmcache.cuda_ops`, `lmcache_native`, `lmcache_redis`, `lmcache_fs` | a missing extension (`import lmcache` only logs it) |
| 2 | `ldd cuda_ops*.so`: `libcudart.so.12`, no `.so.13` | a CUDA 13 build |
| 3 | `$OUT/bin/lmcache server --help` | the service entry point (a missing `.so` there prints "requires the full lmcache installation" and exits 1) |
| 4 | import the MP connector from LMCache and from vLLM | a missing runtime dep on vLLM's path |
| 5 | patch 0001's markers in the installed adapter | a silently dropped patch |
| 6 | `importlib.metadata.version("lmcache") == "0.5.6"` | wrong metadata (vLLM reads the version) |
| 7 | `lmcache_frontend/static/index.html`, import the dashboard app | missing package data for `vllm-lmcache-ui` |
| 8 | cupy's CUDA runtime and NVRTC are 12.x | a CUDA 13 cupy (it runs in `register_kv_caches`) |
| 9 | no `libcudart.so.13` in `$OUT`'s closure | a second, CUDA 13 runtime |

The GPU half, in a window: `$OUT/bin/python -c "import cupy,torch; s=torch.cuda.Stream(); cupy.cuda.ExternalStream(s.cuda_stream).use(); print('ok')"`
(torch's CUDA 12.6 runtime and cupy's static 12.9 runtime share a stream).

## Check 6: LMCache still stores and serves after 1 h of quiet

Guards `nix/patches/lmcache/0001-mp-worker-heartbeat-from-registration.patch`
(the bug: `docs/upstream/lmcache-heartbeat-issue.md`). Without the patch, a
vLLM worker that gets no prompt of at least one 256-token chunk within 3600 s
of registering is reaped by the LMCache server and never re-registers, so
every later STORE is rejected and nothing is reused until vLLM restarts.

Script: `scripts/check6-lmcache-reaper.py` (Python 3 stdlib only, no sudo; it
only talks to `127.0.0.1:8000` and `127.0.0.1:8080` and runs `systemctl show`
and `journalctl`).

**When:** in the 00:00–06:00 ET window. It needs the GPUs serving and no other
clients: any prompt over 200 tokens during the quiet hour makes the run
INVALID.

**Duration:** ~65–80 min after the restart: vLLM startup (Flint 3–5 min), 61
min of quiet counted from the LMCache registration, then the two ~11K-token
requests and the eviction flood (a few minutes).

### Commands (on c4130, as charlie, from a checkout of this repo)

Every sudo command takes the password on stdin from `$SERVER_PASSWORD`
(`sudo -S -p ''`), so it is never typed, printed or written out.

```sh
tmux new -s check6                     # or nohup; the run takes over an hour
sudo -S -p '' systemctl restart vllm <<<"$SERVER_PASSWORD"   # the script never uses sudo
python3 scripts/check6-lmcache-reaper.py run 2>&1 | tee ~/check6.log
```

`run` = `start` + wait + `finish`. The same in two steps (e.g. if the session
may drop; the state is in `./check6-state.json`):

```sh
python3 scripts/check6-lmcache-reaper.py start     # right after the restart
python3 scripts/check6-lmcache-reaper.py finish    # any time later; waits out the rest
```

If the LMCache journal is not readable as charlie (the run then ends
INCOMPLETE and prints the exact command, including the `--since` time), save it
and re-judge without re-running anything:

```sh
sudo -S -p '' journalctl -u vllm-lmcache --since @<vllm start> -o short-unix --no-pager <<<"$SERVER_PASSWORD" > ~/check6-lmcache.journal
python3 scripts/check6-lmcache-reaper.py judge --journal-file ~/check6-lmcache.journal
```

Results: per-criterion PASS/FAIL/INVALID on stdout, the exit status (0 PASS,
2 INVALID, 1 FAIL/INCOMPLETE/error; with `--dry-run` the verdicts are
`SMOKE-PASS`/`SMOKE-INVALID`/`SMOKE-FAIL` with the same codes) and the JSON
summary in `./check6-summary.json` (`verdict`, `criteria`, TTFTs, eviction,
the `/status` timeline during the wait).

Options worth knowing: `--dry-run` (no wait and no quiet-period checks, for
checking the plumbing at any time), `--wait-min`, `--base-url`,
`--admin-url`, `--vllm-unit`/`--lmcache-unit` (if vLLM runs under another unit,
e.g. `systemd-run --unit=vsm70-test`), `--no-probe`, `--evict`.

### What it does

1. `start`: reads the vllm unit's `ActiveEnterTimestamp`, waits until
   `/v1/models` answers and `/status` lists 2 `registered_gpu_ids`, and takes
   the anchor from the last `Registered KV cache for GPU ID` journal line
   (else the moment it saw the registration). It aborts if vLLM already served
   a prompt over 200 tokens since it started (`vllm:request_prompt_tokens`
   histogram), then sends one tiny prompt (under 200 tokens; `--no-probe` to
   skip).
2. Waits until anchor + 61 min, polling `/status` every 5 min.
3. `finish`: sends a ~11K-token synthetic prompt (fixed text after a random
   per-run nonce, so earlier runs' cache can't match), drops it from vLLM's own
   GPU prefix cache, and sends it again. The drop is needed because vLLM only
   asks LMCache for tokens its GPU cache doesn't already hold. It uses
   `POST /reset_prefix_cache` if vLLM runs with `VLLM_SERVER_DEV_MODE=1`, else
   it floods distinct filler prompts totalling 1.25× the GPU KV capacity
   (`num_gpu_blocks × block_size` from `vllm:cache_config_info`). The flood is
   also stored in LMCache (L1 64 GB, L2 32 GB).

### PASS criteria (all must hold)

1. No `Reaped GPU instance` line in `journalctl -u vllm-lmcache` since the
   vllm start: none at 1 h (`pinged=False`) and none during startup or CUDA
   graph capture (`pinged=True`). `Rejecting STORE` lines are counted too.
2. `/status` `registered_gpu_ids` has 2 ids after the wait and after the
   requests.
3. Request 2 gets an external hit: `vllm:external_prefix_cache_hits` grows
   across it, and `usage.prompt_tokens_details.cached_tokens` is above 0
   (Flint runs with `--enable-prompt-tokens-details`). The summary also
   reports `vllm:prefix_cache_hits` (GPU) and
   `vllm:prompt_tokens_cached_by_source`.

TTFT of request 1 vs 2 is reported as supporting information only.

### INVALID (exit 2): no verdict on the fix

- The run did not test the bug: vllm restarted during it, a prompt over 200
  tokens or an LMCache store happened during the quiet hour, or the wait was
  shorter than `/status` `worker_liveness.registration_grace_seconds` plus one
  reaper interval (`reap_timeout_seconds / 4`).
- The eviction did not take request 1 out of vLLM's GPU prefix cache, so
  criterion 3 was not tested: the eviction errored, the flood was capped below
  1.25× the KV capacity (raise `--max-flood-tokens`), or request 2 got GPU
  prefix-cache hits and no external ones.

A reap (criterion 1) or a wrong `registered_gpu_ids` count (criterion 2) is
still a FAIL when the eviction failed. So is no external hit with no GPU hit.

Against an LMCache build without the patch, the expected result is FAIL on
all three criteria (the bug reproduced).
