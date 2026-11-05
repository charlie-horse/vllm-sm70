#!/usr/bin/env python3
"""Regression check 6: LMCache still stores and serves after 1 h of quiet.

Guards nix/patches/lmcache/0001-mp-worker-heartbeat-from-registration.patch.
Without it, a vLLM worker that sees no prompt of at least one 256-token chunk
within --worker-registration-grace-seconds (3600 s) of registering is reaped
by the LMCache MP server while alive and never re-registers, so every later
STORE is rejected (docs/upstream/lmcache-heartbeat-issue.md).

Procedure (docs/checks.md has the operator steps):

  start   Run right after the operator's restart,
          `sudo -S -p '' systemctl restart vllm <<<"$SERVER_PASSWORD"`
          (this script never uses sudo). Waits until the API is up and both GPU
          instances are registered, records the anchor (the last
          "Registered KV cache" journal line, else the moment registration was
          observed), checks that no prompt over 200 tokens was served since
          the vllm start and optionally sends one tiny prompt.
  finish  Waits until anchor + --wait-min, then sends the same ~11K-token
          prompt twice (unique per run via a random nonce), evicting vLLM's
          own GPU prefix cache in between so the second request can only hit
          through LMCache, and judges.
  run     start + wait + finish, for tmux/nohup.
  judge   Re-judges a finished run, e.g. with --journal-file when the
          journal was not readable without sudo.

PASS iff all of:
  1. no "Reaped GPU instance" line in `journalctl -u vllm-lmcache` since the
     vllm start (pinged=True or False);
  2. /status `registered_gpu_ids` lists --expect-gpus ids after the wait and
     after the requests;
  3. request 2 gets an external (KV connector) prefix-cache hit:
     the vllm:external_prefix_cache_hits counter grows across it, and
     usage.prompt_tokens_details.cached_tokens > 0 when vLLM reports it.
TTFT of request 1 vs 2 is printed as supporting information.

The result is INVALID (exit 2, no verdict on the fix) if
  - vllm restarted, a prompt over 200 tokens or an LMCache store arrived
    during the quiet period, or the wait was shorter than the server's
    registration grace plus one reaper interval (the run did not test the
    bug; ignored with --dry-run); or
  - the eviction did not take request 1 out of vLLM's GPU prefix cache:
    the eviction errored, the flood was capped below --flood-factor x the
    KV capacity, or request 2 got GPU prefix-cache hits and no external
    ones (criterion 3 was not tested). A reap or a wrong
    registered_gpu_ids count is still a FAIL.

Exit status: 0 PASS (or SMOKE-PASS with --dry-run), 2 INVALID (or
SMOKE-INVALID), 1 anything else (FAIL, INCOMPLETE, errors).
Python 3 stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

CHUNK_TOKENS = 256
# vllm:request_prompt_tokens uses 1-2-5 buckets; 200 is the largest bound
# below one 256-token chunk, so "prompt > 200 tokens" is the quiet-period test.
QUIET_MAX_PROMPT = 200

REGISTERED = "Registered KV cache for GPU ID"
REAPED = "Reaped GPU instance"
REJECTED = "Rejecting STORE for unregistered GPU instance"

# localhost only: never route through http(s)_proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_ts(t: float | None) -> str:
    if t is None:
        return "?"
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t))


# ---------------------------------------------------------------- HTTP


def http(method: str, url: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with _OPENER.open(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def get_json(url: str, timeout: float = 30):
    return json.loads(http("GET", url, timeout=timeout)[1])


def post_json(url: str, body: dict, timeout: float = 600):
    return json.loads(http("POST", url, body, timeout=timeout)[1])


# ---------------------------------------------------------------- Prometheus

_SAMPLE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom(text: str) -> list[tuple[str, dict, float]]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        out.append((m.group(1), dict(_LABEL_RE.findall(m.group(2) or "")), value))
    return out


def prom_sum(samples, base: str, **labels) -> float | None:
    """Sum a counter over engines; None if vLLM does not export it."""
    names = (base, base + "_total")
    vals = [
        v
        for n, lab, v in samples
        if n in names and all(lab.get(k) == x for k, x in labels.items())
    ]
    return sum(vals) if vals else None


def vllm_metrics(base_url: str) -> dict:
    """The vLLM counters this check uses (names verified against v0.33.0)."""
    try:
        text = http("GET", base_url + "/metrics", timeout=30)[1].decode()
    except (OSError, urllib.error.URLError) as e:
        return {"error": str(e)}
    s = parse_prom(text)
    by_source: dict[str, float] = {}
    for n, lab, v in s:
        if n in ("vllm:prompt_tokens_cached_by_source",
                 "vllm:prompt_tokens_cached_by_source_total"):
            src = lab.get("source", "?")
            by_source[src] = by_source.get(src, 0.0) + v
    # Requests with more than QUIET_MAX_PROMPT prompt tokens (histogram).
    count = prom_sum(s, "vllm:request_prompt_tokens_count")
    le_vals = [
        v
        for n, lab, v in s
        if n == "vllm:request_prompt_tokens_bucket"
        and _le(lab) is not None
        and abs(_le(lab) - QUIET_MAX_PROMPT) < 1e-6
    ]
    long_prompts = None
    if count is not None and le_vals:
        long_prompts = count - sum(le_vals)
    blocks = block_size = None
    for n, lab, _v in s:
        if n == "vllm:cache_config_info":
            blocks = _int(lab.get("num_gpu_blocks"))
            block_size = _int(lab.get("block_size"))
            break
    return {
        "external_prefix_cache_hits": prom_sum(s, "vllm:external_prefix_cache_hits"),
        "external_prefix_cache_queries": prom_sum(s, "vllm:external_prefix_cache_queries"),
        "prefix_cache_hits": prom_sum(s, "vllm:prefix_cache_hits"),
        "prefix_cache_queries": prom_sum(s, "vllm:prefix_cache_queries"),
        "prompt_tokens_cached_by_source": by_source or None,
        "requests_total": count,
        "requests_prompt_over_200": long_prompts,
        "num_gpu_blocks": blocks,
        "block_size": block_size,
    }


def _le(lab: dict) -> float | None:
    try:
        return float(lab.get("le", ""))
    except ValueError:
        return None


def _int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def delta(after: dict, before: dict, key: str):
    a, b = after.get(key), before.get(key)
    if a is None or b is None:
        return None
    return a - b


def delta_by_source(after: dict, before: dict) -> dict | None:
    a = after.get("prompt_tokens_cached_by_source")
    if a is None:
        return None
    b = before.get("prompt_tokens_cached_by_source") or {}
    return {k: v - b.get(k, 0.0) for k, v in a.items() if v - b.get(k, 0.0)}


# ---------------------------------------------------------------- LMCache


def lmcache_status(admin_url: str) -> dict:
    """Fields of the MP server's GET /status (verified against LMCache 0.5.6)."""
    try:
        st = get_json(admin_url + "/status")
    except (OSError, urllib.error.URLError, ValueError) as e:
        return {"error": str(e), "registered_gpu_ids": None}
    sm = st.get("storage_manager") or {}
    l1s = sm.get("l1_managers") or {}
    if "l1_manager" in sm:
        l1s = {"l1": sm["l1_manager"]}
    objs = [m.get("total_object_count") for m in l1s.values()]
    usage = sm.get("l1_usage")
    return {
        "registered_gpu_ids": st.get("registered_gpu_ids"),
        "is_healthy": st.get("is_healthy"),
        "chunk_size": st.get("chunk_size"),
        "worker_liveness": st.get("worker_liveness"),
        "l1_objects": sum(objs) if objs and None not in objs else None,
        "l1_used_bytes": usage[0] if isinstance(usage, list) and usage else None,
    }


def n_ids(st: dict) -> int | None:
    ids = st.get("registered_gpu_ids")
    return len(ids) if isinstance(ids, list) else None


# ---------------------------------------------------------------- systemd / journal


def unit_info(unit: str) -> dict:
    props = ["ActiveState", "MainPID", "InvocationID", "ActiveEnterTimestamp"]
    cmd = ["systemctl", "show", unit] + [f"-p{p}" for p in props]
    info: dict = {}
    for extra in (["--timestamp=unix"], []):
        try:
            r = subprocess.run(cmd + extra, capture_output=True, text=True, timeout=30,
                               check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"error": str(e)}
        if r.returncode == 0:
            break
    else:
        return {"error": r.stderr.strip()}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        info[k] = v
    info["active_enter"] = _parse_systemd_ts(info.get("ActiveEnterTimestamp", ""))
    return info


def _parse_systemd_ts(v: str) -> float | None:
    v = v.strip()
    if not v:
        return None
    if v.startswith("@"):
        return float(v[1:])
    try:
        r = subprocess.run(["date", "-d", v, "+%s"], capture_output=True,
                           text=True, timeout=10, check=False)
        return float(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def journal_cmd(unit: str, since: float) -> list[str]:
    return ["journalctl", "-u", unit, "--since", f"@{int(since)}",
            "-o", "short-unix", "--no-pager"]


def read_journal(unit: str, since: float, expect: int, path: str | None) -> dict:
    """Scan the LMCache server's journal since the vllm start.

    Readable means it shows at least `expect` registrations since `since`:
    without them the window is incomplete (e.g. no permission to read the
    system journal, which journalctl reports only as a hint).
    """
    if path:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text, err, source = f.read(), "", f"file {path}"
        except OSError as e:
            return {"readable": False, "source": path, "why": str(e)}
    else:
        try:
            r = subprocess.run(journal_cmd(unit, since), capture_output=True,
                               text=True, timeout=120, check=False)
            text, err, source = r.stdout, r.stderr, "journalctl (no sudo)"
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"readable": False, "source": "journalctl", "why": str(e)}
    lines = text.splitlines()
    reg = [ln for ln in lines if REGISTERED in ln]
    reaped = [ln for ln in lines if REAPED in ln]
    rejected = sum(REJECTED in ln for ln in lines)
    reg_times = []
    for ln in reg:
        try:
            reg_times.append(float(ln.split(None, 1)[0]))
        except (ValueError, IndexError):
            pass
    readable = len(reg) >= expect
    why = ""
    if not readable:
        why = (f"{len(reg)} '{REGISTERED}' line(s) since the vllm start, "
               f"expected >= {expect}")
        if err.strip():
            why += f"; journalctl: {err.strip().splitlines()[-1]}"
    return {
        "readable": readable,
        "source": source,
        "why": why,
        "registered_lines": len(reg),
        "registered_times": reg_times,
        "reaped_lines": reaped,
        "rejected_store_lines": rejected,
    }


# ---------------------------------------------------------------- prompts

_WORDS = [
    "amber", "falcon", "river", "quiet", "harbor", "seven", "copper",
    "lantern", "meadow", "orbit", "granite", "willow", "signal", "ember",
    "canyon", "velvet", "thunder", "pebble", "marble", "cedar", "glacier",
    "compass", "beacon", "saffron", "tide", "prism", "ledger", "hollow",
    "summit", "rivet", "cobalt", "fern", "quartz", "harvest", "maple", "drift",
    "anchor", "ripple", "basalt", "orchard", "spindle", "juniper", "nectar",
    "falcon", "gravel", "lattice", "monsoon", "tundra",
]


def synth_text(nonce: str, sentences: int) -> str:
    """Deterministic filler; only the leading nonce differs between runs."""
    rng = random.Random(6)
    out = [f"Regression check 6, run {nonce}. Read the log below."]
    for i in range(sentences):
        out.append(f"{i:05d} " + " ".join(rng.choices(_WORDS, k=12)) + ".")
    out.append("Reply with the single word OK.")
    return "\n".join(out)


def messages_for(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def count_tokens(cfg, messages: list[dict]) -> int | None:
    try:
        r = post_json(cfg.base_url + "/tokenize",
                      {"model": cfg.model, "messages": messages}, timeout=120)
        return int(r["count"])
    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError):
        return None


class Sizer:
    """Maps a token target to a sentence count, calibrated with /tokenize."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.base, self.per = 30.0, 20.0  # fallback guess without /tokenize
        self.calibrated = False
        probe_n = 400
        c0 = count_tokens(cfg, messages_for(synth_text("calibrate", 0)))
        c1 = count_tokens(cfg, messages_for(synth_text("calibrate", probe_n)))
        if c0 is not None and c1 is not None and c1 > c0:
            self.base, self.per = float(c0), (c1 - c0) / probe_n
            self.calibrated = True

    def sentences(self, tokens: int) -> int:
        return max(1, int((tokens - self.base) / self.per))


# ---------------------------------------------------------------- requests


def chat_stream(cfg, messages: list[dict], max_tokens: int) -> dict:
    """One streamed chat request: TTFT, latency, usage incl. cached_tokens."""
    body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        cfg.base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = first_chunk = None
    usage = None
    with _OPENER.open(req, timeout=cfg.request_timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            now = time.time()
            if first_chunk is None:
                first_chunk = now - t0
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if ttft is None and any(v for k, v in d.items() if k != "role"):
                    ttft = now - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    total = time.time() - t0
    details = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "ttft_s": round(ttft if ttft is not None else (first_chunk or total), 3),
        "latency_s": round(total, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "usage": usage,
    }


def chat_once(cfg, messages: list[dict], max_tokens: int) -> dict:
    return post_json(
        cfg.base_url + "/v1/chat/completions",
        {"model": cfg.model, "messages": messages, "max_tokens": max_tokens,
         "temperature": 0},
        timeout=cfg.request_timeout,
    )


def wait_l1_settled(cfg, before: int | None) -> dict:
    """Stores are asynchronous: wait until the L1 object count stops moving."""
    deadline = time.time() + cfg.store_settle_timeout
    last = lmcache_status(cfg.admin_url)
    while time.time() < deadline:
        time.sleep(cfg.settle_s)
        cur = lmcache_status(cfg.admin_url)
        if cur.get("l1_objects") == last.get("l1_objects") and (
            before is None or cur.get("l1_objects") != before
        ):
            return cur
        last = cur
    return last


def evict_gpu_prefix_cache(cfg, sizer: Sizer, metrics: dict, max_len: int,
                           run_nonce: str) -> dict:
    """Push request 1 out of vLLM's own GPU prefix cache.

    Otherwise request 2 is served from GPU blocks and never asks the connector
    (vLLM only queries LMCache for tokens beyond the local hit). `reset` uses
    POST /reset_prefix_cache (only with VLLM_SERVER_DEV_MODE=1; LMCache is
    not reset). `flood` sends distinct filler prompts totalling
    --flood-factor x the GPU KV capacity (num_gpu_blocks x block_size from
    vllm:cache_config_info, an upper bound for hybrid models).
    """
    res: dict = {"mode": cfg.evict}
    if cfg.evict == "none":
        return res
    if cfg.evict in ("auto", "reset"):
        try:
            st, _ = http("POST", cfg.base_url + "/reset_prefix_cache", {}, timeout=60)
            res.update(method="reset_prefix_cache", http_status=st)
            return res
        except urllib.error.HTTPError as e:
            res["reset_http_status"] = e.code
            if cfg.evict == "reset":
                res["error"] = f"/reset_prefix_cache returned {e.code}"
                return res
        except (OSError, urllib.error.URLError) as e:
            res["reset_error"] = str(e)
            if cfg.evict == "reset":
                res["error"] = str(e)
                return res
    cap = cfg.kv_capacity_tokens
    if cap is None and metrics.get("num_gpu_blocks") and metrics.get("block_size"):
        cap = metrics["num_gpu_blocks"] * metrics["block_size"]
    if not cap:
        res["error"] = ("GPU KV capacity unknown (no vllm:cache_config_info); "
                        "pass --kv-capacity-tokens")
        return res
    target = int(cap * cfg.flood_factor)
    if target > cfg.max_flood_tokens:
        res["capped_from"] = target
        target = cfg.max_flood_tokens
    per = max(CHUNK_TOKENS * 4, min(cfg.flood_prompt_tokens, max_len - 64))
    n = math.ceil(target / per)
    res.update(method="flood", kv_capacity_tokens=cap, target_tokens=target,
               prompts=n, tokens_per_prompt=per)
    log(f"evict: flooding {n} x ~{per} tokens (GPU KV capacity {cap} tokens)")
    t0 = time.time()
    sent = 0
    for k in range(n):
        text = synth_text(f"{run_nonce}-flood-{k}-{secrets.token_hex(4)}",
                          sizer.sentences(per))
        r = chat_once(cfg, messages_for(text), 1)
        sent += int((r.get("usage") or {}).get("prompt_tokens") or 0)
    res.update(flood_prompt_tokens_sent=sent, seconds=round(time.time() - t0, 1))
    return res


# ---------------------------------------------------------------- state


def save_state(cfg, state: dict) -> None:
    tmp = cfg.state + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, cfg.state)


def load_state(cfg) -> dict:
    try:
        with open(cfg.state, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        sys.exit(f"cannot read state {cfg.state}: {e} (run `start` first)")


def resolve_model(cfg) -> str:
    models = get_json(cfg.base_url + "/v1/models")
    data = models.get("data") or []
    if not data:
        raise RuntimeError("/v1/models lists no model")
    if cfg.model:
        return cfg.model
    return data[0]["id"]


def model_max_len(cfg) -> int:
    for m in get_json(cfg.base_url + "/v1/models").get("data") or []:
        if m.get("id") == cfg.model and m.get("max_model_len"):
            return int(m["max_model_len"])
    return 32768


def print_journal_hint(cfg, since: float) -> None:
    cmd = " ".join(journal_cmd(cfg.lmcache_unit, since))
    print(
        "  The LMCache journal is not readable as this user. Have the operator run\n"
        "  (password on stdin from $SERVER_PASSWORD, never typed on the command line):\n"
        f"    sudo -S -p '' {cmd} <<<\"$SERVER_PASSWORD\" > ~/check6-lmcache.journal\n"
        "  then:\n"
        f"    python3 {sys.argv[0]} judge --state {cfg.state} "
        "--journal-file ~/check6-lmcache.journal",
        flush=True,
    )


# ---------------------------------------------------------------- phases


def phase_start(cfg) -> dict:
    if cfg.vllm_started_at is not None:
        unit = {"active_enter": cfg.vllm_started_at, "source": "--vllm-started-at"}
    else:
        unit = unit_info(cfg.vllm_unit)
        if unit.get("error") or not unit.get("active_enter"):
            sys.exit(f"cannot read {cfg.vllm_unit} start time: {unit} "
                     "(pass --vllm-started-at EPOCH)")
        if unit.get("ActiveState") != "active":
            sys.exit(f"{cfg.vllm_unit} is {unit.get('ActiveState')}, not active")
    started = unit["active_enter"]
    log(f"{cfg.vllm_unit} active since {fmt_ts(started)}")

    # Wait for the API and the LMCache registrations.
    deadline = time.time() + cfg.ready_timeout_min * 60
    saw_unregistered = False
    first_registered = None
    while True:
        st = lmcache_status(cfg.admin_url)
        try:
            cfg.model = resolve_model(cfg)
            api_up = True
        except (OSError, urllib.error.URLError, RuntimeError, ValueError):
            api_up = False
        k = n_ids(st)
        if k is not None and k < cfg.expect_gpus:
            saw_unregistered = True
        if api_up and k is not None and k >= cfg.expect_gpus:
            first_registered = time.time()
            break
        if time.time() > deadline:
            sys.exit(f"not ready after {cfg.ready_timeout_min} min: api_up={api_up} "
                     f"registered_gpu_ids={st.get('registered_gpu_ids')} "
                     f"{st.get('error', '')}")
        time.sleep(10)
    log(f"API up (model {cfg.model}); registered_gpu_ids={st['registered_gpu_ids']}")

    jr = read_journal(cfg.lmcache_unit, started, cfg.expect_gpus, cfg.journal_file)
    reg_times = [t for t in jr.get("registered_times", []) if t >= started - 1]
    if len(reg_times) >= cfg.expect_gpus:
        anchor, anchor_src = max(reg_times), "journal: last registration"
    elif saw_unregistered:
        anchor, anchor_src = first_registered, "observed: registration seen by polling"
    else:
        anchor = first_registered
        anchor_src = ("conservative: already registered at first poll, "
                      "anchored at that poll")
    if jr.get("reaped_lines"):
        log(f"WARNING: already {len(jr['reaped_lines'])} reap line(s) in the journal")

    metrics0 = vllm_metrics(cfg.base_url)
    long0 = metrics0.get("requests_prompt_over_200")
    if long0:
        msg = (f"vLLM already served {int(long0)} prompt(s) over "
               f"{QUIET_MAX_PROMPT} tokens since it started; the quiet period is "
               "broken. Restart vllm (operator: sudo -S -p '' systemctl restart "
               "vllm <<<\"$SERVER_PASSWORD\") and run `start` again.")
        if not cfg.dry_run:
            sys.exit(msg)
        log("dry run, continuing anyway: " + msg)
    elif long0 is None:
        log("WARNING: vllm:request_prompt_tokens histogram not found; the quiet "
            "period cannot be verified from metrics")

    probe = None
    if cfg.probe:
        msgs = messages_for("Reply with the single word ready.")
        n = count_tokens(cfg, msgs)
        if n is not None and n < QUIET_MAX_PROMPT:
            r = chat_once(cfg, msgs, cfg.max_tokens)
            probe = {"prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                     "at": time.time()}
            log(f"tiny probe sent ({probe['prompt_tokens']} prompt tokens)")
        else:
            log(f"tiny probe skipped (tokenized length {n})")

    st0 = lmcache_status(cfg.admin_url)
    state = {
        "check": 6,
        "dry_run": cfg.dry_run,
        "run_nonce": secrets.token_hex(8),
        "base_url": cfg.base_url,
        "admin_url": cfg.admin_url,
        "model": cfg.model,
        "vllm_unit": cfg.vllm_unit,
        "lmcache_unit": cfg.lmcache_unit,
        "expect_gpus": cfg.expect_gpus,
        "vllm_unit_info": unit,
        "vllm_started_at": started,
        "anchor": anchor,
        "anchor_source": anchor_src,
        "wait_min": cfg.wait_min,
        "deadline": anchor + cfg.wait_min * 60,
        "start": {"at": time.time(), "status": st0, "metrics": metrics0,
                  "journal_readable": jr.get("readable"), "probe": probe},
        "timeline": [],
    }
    save_state(cfg, state)
    log(f"anchor {fmt_ts(anchor)} ({anchor_src}); "
        f"finish at {fmt_ts(state['deadline'])}")
    log(f"state saved to {cfg.state}")
    if not jr.get("readable"):
        print_journal_hint(cfg, started)
    return state


def wait_phase(cfg, state: dict) -> None:
    while True:
        remaining = state["deadline"] - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, cfg.poll_min * 60))
        st = lmcache_status(cfg.admin_url)
        m = vllm_metrics(cfg.base_url)
        state["timeline"].append({
            "at": round(time.time(), 1),
            "registered_gpu_ids": st.get("registered_gpu_ids"),
            "l1_objects": st.get("l1_objects"),
            "requests_prompt_over_200": m.get("requests_prompt_over_200"),
        })
        save_state(cfg, state)
        mins = (time.time() - state["anchor"]) / 60
        log(f"t+{mins:.1f} min: registered_gpu_ids={st.get('registered_gpu_ids')} "
            f"l1_objects={st.get('l1_objects')} "
            f"long_prompts={m.get('requests_prompt_over_200')}")


def phase_finish(cfg, state: dict) -> dict:
    cfg.model = state["model"]
    if time.time() < state["deadline"]:
        log(f"waiting until {fmt_ts(state['deadline'])} "
            f"({(state['deadline'] - time.time()) / 60:.1f} min)")
        wait_phase(cfg, state)

    res: dict = {"at": time.time()}
    if cfg.vllm_started_at is None and state["vllm_unit_info"].get("InvocationID"):
        now_unit = unit_info(state["vllm_unit"])
        res["vllm_unit_info"] = now_unit
        res["vllm_restarted"] = (
            now_unit.get("InvocationID") != state["vllm_unit_info"].get("InvocationID")
        )
    res["status_after_wait"] = lmcache_status(cfg.admin_url)
    res["metrics_after_wait"] = vllm_metrics(cfg.base_url)
    log(f"after wait: registered_gpu_ids="
        f"{res['status_after_wait'].get('registered_gpu_ids')}")

    max_len = model_max_len(cfg)
    sizer = Sizer(cfg)
    target = min(cfg.prompt_tokens, max_len - cfg.max_tokens - 64)
    text = synth_text(state["run_nonce"], sizer.sentences(target))
    msgs = messages_for(text)
    res["prompt"] = {"target_tokens": target, "tokenize_calibrated": sizer.calibrated,
                     "tokens": count_tokens(cfg, msgs), "nonce": state["run_nonce"]}
    log(f"prompt: {res['prompt']['tokens']} tokens (target {target})")

    m_a = vllm_metrics(cfg.base_url)
    l1_before = res["status_after_wait"].get("l1_objects")
    r1 = chat_stream(cfg, msgs, cfg.max_tokens)
    time.sleep(cfg.settle_s)
    st1 = wait_l1_settled(cfg, l1_before)
    m_b = vllm_metrics(cfg.base_url)
    r1["external_hits_delta"] = delta(m_b, m_a, "external_prefix_cache_hits")
    r1["l1_objects_delta"] = (
        None if st1.get("l1_objects") is None or l1_before is None
        else st1["l1_objects"] - l1_before
    )
    res["request1"] = r1
    log(f"request 1: ttft {r1['ttft_s']} s, cached_tokens {r1['cached_tokens']}, "
        f"LMCache L1 objects +{r1['l1_objects_delta']}")

    res["evict"] = evict_gpu_prefix_cache(cfg, sizer, m_b, max_len, state["run_nonce"])
    if res["evict"].get("error"):
        log(f"WARNING: eviction: {res['evict']['error']}")

    m_c = vllm_metrics(cfg.base_url)
    r2 = chat_stream(cfg, msgs, cfg.max_tokens)
    time.sleep(cfg.settle_s)
    m_d = vllm_metrics(cfg.base_url)
    r2["external_hits_delta"] = delta(m_d, m_c, "external_prefix_cache_hits")
    r2["external_queries_delta"] = delta(m_d, m_c, "external_prefix_cache_queries")
    r2["device_hits_delta"] = delta(m_d, m_c, "prefix_cache_hits")
    r2["cached_by_source_delta"] = delta_by_source(m_d, m_c)
    res["request2"] = r2
    res["status_after_requests"] = lmcache_status(cfg.admin_url)
    res["metrics_after_requests"] = m_d
    log(f"request 2: ttft {r2['ttft_s']} s, cached_tokens {r2['cached_tokens']}, "
        f"external hits +{r2['external_hits_delta']}, "
        f"device hits +{r2['device_hits_delta']}")
    state["finish"] = res
    save_state(cfg, state)
    return state


# ---------------------------------------------------------------- verdict


def judge(cfg, state: dict) -> int:
    fin = state.get("finish")
    if not fin:
        sys.exit("no finish results in the state file; run `finish` first")
    expect = state["expect_gpus"]
    started = state["vllm_started_at"]
    jr = read_journal(state["lmcache_unit"], started, expect, cfg.journal_file)

    crit: dict = {}
    # 1. no reaps
    if jr.get("readable"):
        reaped = jr["reaped_lines"]
        crit["no_reaped_gpu_instances"] = {
            "result": "FAIL" if reaped else "PASS",
            "reaped_lines": reaped[:10],
            "rejected_store_lines": jr["rejected_store_lines"],
            "source": jr["source"],
        }
    else:
        crit["no_reaped_gpu_instances"] = {"result": "UNKNOWN", "why": jr.get("why"),
                                           "source": jr.get("source")}
    # 2. both instances registered after the wait and after the requests
    a = fin["status_after_wait"].get("registered_gpu_ids")
    b = fin["status_after_requests"].get("registered_gpu_ids")
    ok = n_ids(fin["status_after_wait"]) == expect and \
        n_ids(fin["status_after_requests"]) == expect
    crit["gpu_ids_registered"] = {"result": "PASS" if ok else "FAIL",
                                  "after_wait": a, "after_requests": b,
                                  "expected": expect}
    # 3. external hit on request 2
    r2 = fin["request2"]
    ext, cached = r2.get("external_hits_delta"), r2.get("cached_tokens")
    ev = fin.get("evict") or {}
    evict_invalid = []
    if ev.get("error"):
        evict_invalid.append(f"eviction failed: {ev['error']}")
    if ev.get("capped_from"):
        evict_invalid.append(
            f"flood capped at {ev.get('target_tokens')} tokens, below "
            f"{ev['capped_from']} (--flood-factor x KV capacity); raise "
            "--max-flood-tokens")
    if ext is not None and ext <= 0 and (r2.get("device_hits_delta") or 0) > 0:
        evict_invalid.append(
            "request 2 was served from vLLM's GPU prefix cache, so LMCache was "
            "not asked: the eviction did not work")
    why = "; ".join(evict_invalid)
    if evict_invalid:
        res = "INVALID"
    elif ext is None:
        res = "UNKNOWN"
        why = "vllm:external_prefix_cache_hits not exported"
    else:
        res = "PASS" if ext > 0 and (cached is None or cached > 0) else "FAIL"
    crit["external_hit_request2"] = {
        "result": res, "why": why,
        "external_prefix_cache_hits_delta": ext,
        "external_prefix_cache_queries_delta": r2.get("external_queries_delta"),
        "cached_tokens": cached,
        "device_prefix_cache_hits_delta": r2.get("device_hits_delta"),
        "cached_by_source_delta": r2.get("cached_by_source_delta"),
        "prompt_tokens": r2.get("prompt_tokens"),
    }

    # Validity of the run itself.
    invalid = []
    if fin.get("vllm_restarted"):
        invalid.append("vllm restarted during the run")
    long_start = state["start"]["metrics"].get("requests_prompt_over_200")
    long_end = fin["metrics_after_wait"].get("requests_prompt_over_200")
    if long_start is not None and long_end is not None and long_end > long_start:
        invalid.append(f"{int(long_end - long_start)} prompt(s) over "
                       f"{QUIET_MAX_PROMPT} tokens during the quiet period")
    l1s, l1w = state["start"]["status"].get("l1_objects"), \
        fin["status_after_wait"].get("l1_objects")
    if l1s is not None and l1w is not None and l1w > l1s:
        invalid.append(f"LMCache stored {l1w - l1s} object(s) during the quiet period")
    waited = fin["at"] - state["anchor"]
    wl = state["start"]["status"].get("worker_liveness") or {}
    need = None
    if wl.get("registration_grace_seconds"):
        need = wl["registration_grace_seconds"] + wl.get("reap_timeout_seconds", 0) / 4
    if waited < state["wait_min"] * 60 - 1:
        invalid.append(f"waited {waited / 60:.1f} min < --wait-min")
    if need is not None and waited < need:
        invalid.append(f"waited {waited:.0f} s < registration grace + one reaper "
                       f"interval ({need:.0f} s)")

    # Order: an untested run says nothing; a reap or a lost registration is a
    # FAIL of the fix even if the eviction failed; a failed eviction voids
    # only criterion 3.
    dry = bool(state.get("dry_run"))
    results = [c["result"] for c in crit.values()]
    if invalid and not dry:
        verdict = "INVALID"
    elif "FAIL" in results:
        verdict = "FAIL"
    elif "INVALID" in results:
        verdict = "INVALID"
    elif "UNKNOWN" in results:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    if dry:
        verdict = "SMOKE-" + verdict

    r1 = fin["request1"]
    summary = {
        "check": 6,
        "verdict": verdict,
        "dry_run": state.get("dry_run", False),
        "invalid_reasons": invalid + evict_invalid,
        "criteria": crit,
        "ttft_s": {"request1": r1.get("ttft_s"), "request2": r2.get("ttft_s")},
        "request1": {k: r1.get(k) for k in ("prompt_tokens", "cached_tokens",
                                            "external_hits_delta", "l1_objects_delta",
                                            "ttft_s", "latency_s")},
        "evict": fin.get("evict"),
        "model": state["model"],
        "vllm_started_at": fmt_ts(started),
        "anchor": fmt_ts(state["anchor"]),
        "anchor_source": state["anchor_source"],
        "waited_min_since_anchor": round(waited / 60, 1),
        "required_wait_s": need,
        "prompt": fin.get("prompt"),
        "timeline": state.get("timeline"),
        "journal": {k: jr.get(k) for k in ("readable", "source", "why",
                                           "registered_lines", "rejected_store_lines")},
        "judged_at": fmt_ts(time.time()),
    }
    with open(cfg.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n=== Regression check 6: LMCache after 1 h of quiet ===")
    for name, c in crit.items():
        extra = {k: v for k, v in c.items() if k != "result" and v not in (None, "")}
        print(f"  {c['result']:<8} {name}: {json.dumps(extra, sort_keys=True)}")
    print(f"  info     TTFT request 1 {r1.get('ttft_s')} s -> request 2 "
          f"{r2.get('ttft_s')} s; request 1 stored +{r1.get('l1_objects_delta')} "
          f"L1 objects; evict {fin.get('evict', {}).get('method')}")
    for reason in invalid:
        print(f"  {'note' if dry else 'INVALID':<8} {reason}")
    print(f"VERDICT: {verdict}   (summary: {cfg.summary})")
    if crit["no_reaped_gpu_instances"]["result"] == "UNKNOWN":
        print_journal_hint(cfg, started)
    return exit_code(verdict)


def exit_code(verdict: str) -> int:
    v = verdict.removeprefix("SMOKE-")
    return {"PASS": 0, "INVALID": 2}.get(v, 1)


# ---------------------------------------------------------------- CLI


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="vLLM OpenAI API (default %(default)s)")
    common.add_argument("--admin-url", default="http://127.0.0.1:8080",
                        help="LMCache MP server admin API (default %(default)s)")
    common.add_argument("--model", default=None,
                        help="served model name (default: first of /v1/models)")
    common.add_argument("--state", default="check6-state.json")
    common.add_argument("--summary", default="check6-summary.json")
    common.add_argument("--journal-file", default=None,
                        help="saved `journalctl -u vllm-lmcache` output to use "
                             "instead of running journalctl")
    common.add_argument("--vllm-unit", default="vllm")
    common.add_argument("--lmcache-unit", default="vllm-lmcache")
    common.add_argument("--vllm-started-at", type=float, default=None,
                        help="vllm start as a Unix time, instead of systemctl")
    common.add_argument("--expect-gpus", type=int, default=2)
    common.add_argument("--wait-min", type=float, default=61.0,
                        help="quiet minutes after registration (default 61)")
    common.add_argument("--dry-run", action="store_true",
                        help="smoke test: --wait-min 0, no quiet-period abort, "
                             "verdict SMOKE-*")
    common.add_argument("--no-probe", dest="probe", action="store_false",
                        help="do not send the tiny (<200-token) prompt at start")
    common.add_argument("--prompt-tokens", type=int, default=11000)
    common.add_argument("--max-tokens", type=int, default=16)
    common.add_argument("--evict", choices=["auto", "reset", "flood", "none"],
                        default="auto",
                        help="how to drop request 1 from vLLM's GPU prefix cache "
                             "(auto: /reset_prefix_cache if dev mode, else flood)")
    common.add_argument("--flood-factor", type=float, default=1.25)
    common.add_argument("--flood-prompt-tokens", type=int, default=16384)
    common.add_argument("--max-flood-tokens", type=int, default=4_000_000)
    common.add_argument("--kv-capacity-tokens", type=int, default=None)
    common.add_argument("--poll-min", type=float, default=5.0)
    common.add_argument("--ready-timeout-min", type=float, default=30.0)
    common.add_argument("--request-timeout", type=float, default=900.0)
    common.add_argument("--settle-s", type=float, default=3.0)
    common.add_argument("--store-settle-timeout", type=float, default=60.0,
                        help="max seconds to wait for request 1's LMCache store")

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, h in (("start", "record the vllm start and the anchor"),
                    ("finish", "wait out the quiet period, send the requests, judge"),
                    ("run", "start + wait + finish"),
                    ("judge", "re-judge a finished run (e.g. with --journal-file)")):
        sub.add_parser(name, parents=[common], help=h)
    cfg = p.parse_args(argv)
    if cfg.dry_run:
        cfg.wait_min = 0.0
    cfg.base_url = cfg.base_url.rstrip("/")
    cfg.admin_url = cfg.admin_url.rstrip("/")

    if cfg.cmd == "start":
        phase_start(cfg)
        return 0
    if cfg.cmd == "run":
        state = phase_start(cfg)
        return judge(cfg, phase_finish(cfg, state))
    state = load_state(cfg)
    # Later phases talk to the endpoints `start` used.
    cfg.base_url, cfg.admin_url = state["base_url"], state["admin_url"]
    if cfg.cmd == "finish":
        cfg.dry_run = state.get("dry_run", False)
        return judge(cfg, phase_finish(cfg, state))
    return judge(cfg, state)


if __name__ == "__main__":
    sys.exit(main())
