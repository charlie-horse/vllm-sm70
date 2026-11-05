# SPDX-License-Identifier: Apache-2.0
"""Verdict logic of scripts/check6-lmcache-reaper.py against a fake server.

An in-process fake of vLLM's OpenAI API (/v1/models, /tokenize,
/v1/chat/completions, /metrics) and the LMCache MP admin API (/status) on
127.0.0.1 ephemeral ports; the script runs as a subprocess with a fake journal
file and a sub-second wait. Stdlib only, no GPU, a few seconds in all.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check6-lmcache-reaper.py"
KV_CAPACITY = 16000  # num_gpu_blocks 1000 x block_size 16


class FakeServer:
    """mode "ok": LMCache stores and serves. mode "reap": both GPU instances
    are dropped once request 1 has been stored, so request 2 cannot hit."""

    def __init__(self, mode="ok", grace_s=0.2, reap_s=0.4):
        self.mode, self.grace_s, self.reap_s = mode, grace_s, reap_s
        self.lock = threading.Lock()
        self.ids = [111, 222]
        self.c = {"ext_hits": 0, "ext_q": 0, "dev_hits": 0, "dev_q": 0, "req": 0, "req_le200": 0}
        self.l1_objects = 5
        self.gpu = []  # LRU of (hash, tokens)
        self.lmc = set()
        self.servers = []

    @staticmethod
    def ntok(messages):
        return sum(len(m["content"]) for m in messages) // 4 + 8

    def serve_prompt(self, messages):
        t = self.ntok(messages)
        h = hashlib.sha1(json.dumps(messages).encode()).hexdigest()
        c = self.c
        with self.lock:
            c["req"] += 1
            c["req_le200"] += t <= 200
            c["dev_q"] += t
            cached = 0
            if any(x == h for x, _ in self.gpu):
                cached = t - 1
                c["dev_hits"] += cached
                self.gpu = [g for g in self.gpu if g[0] != h]
            else:
                c["ext_q"] += t
                if h in self.lmc and self.ids:
                    cached = (t // 256) * 256
                    c["ext_hits"] += cached
            self.gpu.append((h, t))
            while sum(x for _, x in self.gpu) > KV_CAPACITY:
                self.gpu.pop(0)
            if t >= 256 and self.ids:
                if h not in self.lmc:
                    self.l1_objects += t // 256
                self.lmc.add(h)
                if self.mode == "reap":
                    self.ids = []
        return t, cached

    def status(self):
        return {
            "is_healthy": True,
            "chunk_size": 256,
            "registered_gpu_ids": list(self.ids),
            "worker_liveness": {"enabled": True, "reap_timeout_seconds": self.reap_s,
                                "registration_grace_seconds": self.grace_s,
                                "tracked_instances": len(self.ids)},
            "storage_manager": {"is_healthy": True,
                                "l1_usage": [self.l1_objects * 1000, 10**9],
                                "l1_manager": {"total_object_count": self.l1_objects}},
        }

    def metrics(self):
        c, lab = self.c, 'engine="0",model_name="fake"'
        rows = [
            f"vllm:external_prefix_cache_hits_total{{{lab}}} {c['ext_hits']}.0",
            f"vllm:external_prefix_cache_queries_total{{{lab}}} {c['ext_q']}.0",
            f"vllm:prefix_cache_hits_total{{{lab}}} {c['dev_hits']}.0",
            f"vllm:prefix_cache_queries_total{{{lab}}} {c['dev_q']}.0",
            (f'vllm:prompt_tokens_cached_by_source_total{{{lab},source="device"}} '
             f"{c['dev_hits']}.0"),
            f'vllm:request_prompt_tokens_bucket{{{lab},le="200.0"}} {c["req_le200"]}.0',
            f'vllm:request_prompt_tokens_bucket{{{lab},le="+Inf"}} {c["req"]}.0',
            f"vllm:request_prompt_tokens_count{{{lab}}} {c['req']}.0",
            'vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="1000"} 1.0',
        ]
        return ("# TYPE vllm:external_prefix_cache_hits_total counter\n"
                + "\n".join(rows) + "\n").encode()

    def handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, obj, ctype="application/json"):
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/status":
                    return self.send(200, fake.status())
                if self.path == "/v1/models":
                    return self.send(200, {"data": [{"id": "fake", "max_model_len": 65536}]})
                if self.path == "/metrics":
                    return self.send(200, fake.metrics(), "text/plain")
                self.send(404, {"detail": "Not Found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                b = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/tokenize":
                    return self.send(200, {"count": fake.ntok(b["messages"]),
                                           "max_model_len": 65536, "tokens": []})
                if self.path != "/v1/chat/completions":
                    return self.send(404, {"detail": "Not Found"})  # /reset_prefix_cache
                t, cached = fake.serve_prompt(b["messages"])
                usage = {"prompt_tokens": t, "completion_tokens": 1, "total_tokens": t + 1,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                if not b.get("stream"):
                    return self.send(200, {"choices": [{"message": {"content": "OK"}}],
                                           "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in ({"choices": [{"delta": {"role": "assistant"}}]},
                              {"choices": [{"delta": {"reasoning": "Hm"}}]},
                              {"choices": [], "usage": usage}):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        return H

    def __enter__(self):
        for _ in range(2):
            srv = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.servers.append(srv)
        return self

    def __exit__(self, *exc):
        for srv in self.servers:
            srv.shutdown()
            srv.server_close()

    def url(self, i):
        return f"http://127.0.0.1:{self.servers[i].server_address[1]}"


class Check6Verdicts(unittest.TestCase):
    def run_check(self, fake, *extra, reaped=False):
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            journal = [f"{now - 1:.6f} host lmcache[1]: LMCache INFO: "
                       f"Registered KV cache for GPU ID {i} with 40 layers" for i in (111, 222)]
            if reaped:
                journal.append(f"{now:.6f} host lmcache[1]: LMCache WARNING: Reaped GPU "
                               "instance 111: silent for 3600.1s (pinged=False)")
            (d / "journal").write_text("\n".join(journal) + "\n")
            args = [
                sys.executable, str(SCRIPT), "run",
                "--base-url", fake.url(0), "--admin-url", fake.url(1),
                "--vllm-started-at", str(now - 5),
                "--journal-file", str(d / "journal"),
                "--state", str(d / "state.json"), "--summary", str(d / "summary.json"),
                "--settle-s", "0.05", "--store-settle-timeout", "1",
                "--poll-min", "0.01", "--no-probe", *extra,
            ]
            p = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
            summary_path = d / "summary.json"
            self.assertTrue(summary_path.exists(), p.stdout + p.stderr)
            return p.returncode, json.loads(summary_path.read_text()), p.stdout

    def test_pass(self):
        with FakeServer() as fake:
            rc, s, out = self.run_check(fake, "--wait-min", "0.01")
        self.assertEqual((rc, s["verdict"]), (0, "PASS"), out)
        c3 = s["criteria"]["external_hit_request2"]
        self.assertGreater(c3["external_prefix_cache_hits_delta"], 0)
        self.assertGreater(c3["cached_tokens"], 0)
        self.assertEqual(s["evict"]["method"], "flood")

    def test_reaped_is_fail(self):
        with FakeServer(mode="reap") as fake:
            rc, s, out = self.run_check(fake, "--wait-min", "0.01", reaped=True)
        self.assertEqual((rc, s["verdict"]), (1, "FAIL"), out)
        crit = s["criteria"]
        self.assertEqual(crit["no_reaped_gpu_instances"]["result"], "FAIL")
        self.assertEqual(crit["gpu_ids_registered"]["result"], "FAIL")
        self.assertEqual(crit["external_hit_request2"]["result"], "FAIL")

    def test_eviction_failed_is_invalid(self):
        # Request 2 is served from the GPU prefix cache: criterion 3 untested.
        with FakeServer() as fake:
            rc, s, out = self.run_check(fake, "--wait-min", "0.01", "--evict", "none")
        self.assertEqual((rc, s["verdict"]), (2, "INVALID"), out)
        c3 = s["criteria"]["external_hit_request2"]
        self.assertEqual(c3["result"], "INVALID")
        self.assertGreater(c3["device_prefix_cache_hits_delta"], 0)

    def test_capped_flood_is_invalid(self):
        with FakeServer() as fake:
            rc, s, out = self.run_check(fake, "--wait-min", "0.01",
                                        "--max-flood-tokens", "2000")
        self.assertEqual((rc, s["verdict"]), (2, "INVALID"), out)
        self.assertIn("capped_from", s["evict"])

    def test_reap_beats_failed_eviction(self):
        with FakeServer(mode="reap") as fake:
            rc, s, out = self.run_check(fake, "--wait-min", "0.01", "--evict", "none",
                                        reaped=True)
        self.assertEqual((rc, s["verdict"]), (1, "FAIL"), out)

    def test_short_wait_is_invalid(self):
        with FakeServer(grace_s=30, reap_s=60) as fake:
            rc, s, out = self.run_check(fake, "--wait-min", "0")
        self.assertEqual((rc, s["verdict"]), (2, "INVALID"), out)
        self.assertTrue(any("registration grace" in r for r in s["invalid_reasons"]), s)

    def test_dry_run_pass(self):
        with FakeServer(grace_s=30, reap_s=60) as fake:
            rc, s, out = self.run_check(fake, "--dry-run")
        self.assertEqual((rc, s["verdict"]), (0, "SMOKE-PASS"), out)


if __name__ == "__main__":
    unittest.main()
