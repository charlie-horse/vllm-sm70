"""Probe: does reasoning_content survive mini-swe-agent's LitellmModel + litellm
on the wire? Starts a fake OpenAI-compatible server, runs two turns, and prints
the assistant message litellm actually sent in request #2.

Usage: .venv/bin/python adapter/wire_probe.py [model_class_path]
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

REQS: list[dict] = []


def _resp(field: str) -> dict:
    msg = {
        "role": "assistant",
        "content": "Let me look.",
        field: "THINK-1: inspect repo first",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
    }
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "flint",
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class H(BaseHTTPRequestHandler):
    field = "reasoning_content"

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQS.append(body)
        data = json.dumps(_resp(H.field)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def run(prefix: str, field: str, model_class: str) -> dict | None:
    from minisweagent.models import get_model

    REQS.clear()
    H.field = field
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = {
        "model_name": f"{prefix}/flint",
        "cost_tracking": "ignore_errors",
        "model_kwargs": {
            "api_base": f"http://127.0.0.1:{srv.server_port}/v1",
            "api_key": "dummy",
            "drop_params": True,
        },
    }
    if model_class:
        cfg["model_class"] = model_class
    model = get_model(config=cfg)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    a1 = model.query(msgs)
    obs = model.format_observation_messages(
        a1, [{"output": "f", "returncode": 0, "exception_info": ""}]
    )
    model.query(msgs + [a1] + obs)
    srv.shutdown()
    return next(m for m in REQS[1]["messages"] if m["role"] == "assistant")


if __name__ == "__main__":
    mc = sys.argv[1] if len(sys.argv) > 1 else ""
    for prefix in ("openai", "hosted_vllm"):
        for field in ("reasoning_content", "reasoning"):
            sent = run(prefix, field, mc)
            keys = sorted(k for k in sent if "reason" in k or "think" in k)
            print(f"{prefix:12s} server-field={field:18s} sent keys={keys} "
                  f"value={sent.get('reasoning_content')!r}")
