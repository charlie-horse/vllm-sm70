#!/usr/bin/env python3
"""Tiny stand-in for the vLLM OpenAI server, for dry-running gate_a.py.

Implements ``/v1/models``, ``/version``, ``/tokenize`` (chat and prompt),
``/v1/completions`` (token-id prompts, ``prompt_logprobs``) and
``/v1/chat/completions`` (``return_token_ids``). Outputs are deterministic.

``--bug`` emulates the max-num-seqs regression: when more than one request
is in flight, logprobs past 64 tokens into the final assistant turn get much
worse and greedy output turns into gibberish after 64 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_WORDS_TEXT = (
    "the a of to in is that for it as with was on be by this are from or "
    "which an at not have has can will we they their its more also into "
    "cache memory request batch token sequence model layer kernel thread "
    "value result function return state queue block page buffer index "
    "first second next last each every other same different large small "
    "because therefore however while when after before during between "
    "simple careful precise general specific useful important common "
    "compute measure store load write read update check verify explain "
)
WORDS = _WORDS_TEXT.split()
SPLIT = re.compile(r"\w+|\s+|[^\w\s]")


class Vocab:
    def __init__(self):
        self.lock = threading.Lock()
        self.s2i: dict[str, int] = {}
        self.i2s: dict[int, str] = {}
        for special in (
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|observation|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "\n",
        ):
            self.id(special)

    def id(self, s: str) -> int:
        with self.lock:
            if s not in self.s2i:  # stable across processes
                i = _h("tok", s) % (1 << 31)
                assert i not in self.i2s, "mock vocab collision"
                self.s2i[s] = i
                self.i2s[i] = s
            return self.s2i[s]

    def encode(self, text: str) -> list[int]:
        out, pos = [], 0
        for m in re.finditer(r"<\|\w+\|>|</?think>|</?tool_call>", text):
            out += [self.id(t) for t in SPLIT.findall(text[pos : m.start()])]
            out.append(self.id(m.group()))
            pos = m.end()
        out += [self.id(t) for t in SPLIT.findall(text[pos:])]
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.i2s[i] for i in ids)


def render(messages, tools, add_generation_prompt) -> str:
    s = ""
    if tools:
        s += "<|system|>\n# Tools\n" + json.dumps(tools) + "\n"
    for m in messages:
        role = "observation" if m["role"] == "tool" else m["role"]
        s += f"<|{role}|>\n"
        if m["role"] == "assistant":
            r = m.get("reasoning") or m.get("reasoning_content")
            s += f"<think>{r or ''}</think>"
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c)
        s += c or ""
        for tc in m.get("tool_calls") or []:
            f = tc["function"]
            s += f"<tool_call>{f['name']}\n{f['arguments']}</tool_call>"
    if add_generation_prompt:
        s += "<|assistant|>\n"
    return s


def _h(*xs) -> int:
    return int.from_bytes(
        hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(), "little"
    )


VOCAB = Vocab()


def gen_text(prompt: str, n_tokens: int, code: bool) -> str:
    rng = random.Random(_h(prompt))
    parts = []
    if code:
        lines = ["Here is the implementation.", "", "```python"]
        for i in range(max(3, n_tokens // 40)):
            a, b = rng.randint(2, 99), rng.choice(WORDS)
            lines += [
                f"def {b}_{i}(x, y={a}):",
                f'    """Return {b} {i} of x."""',
                f"    return x * {a} + y - {rng.randint(0, 9)}",
                "",
            ]
        lines += ["```", "", "This covers the requested behaviour."]
        return "\n".join(lines)
    while len(VOCAB.encode(" ".join(parts))) < n_tokens:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 16)))
        parts.append(sent.capitalize() + ".")
    return " ".join(parts)


class State:
    bug = False
    lock = threading.Lock()
    inflight = 0
    peak = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send({"data": [{"id": "flint", "object": "model"}]})
        if self.path == "/version":
            return self._send({"version": "mock-0"})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with State.lock:
            State.inflight += 1
            busy = State.inflight > 1
            State.peak = max(State.peak, State.inflight)
        try:
            time.sleep(0.05)
            with State.lock:
                busy = busy or State.inflight > 1
            if self.path == "/tokenize":
                return self._tokenize(body)
            if self.path == "/v1/completions":
                return self._completions(body, busy)
            if self.path == "/v1/chat/completions":
                return self._chat(body, busy)
            self._send({"error": "not found"}, 404)
        finally:
            with State.lock:
                State.inflight -= 1

    def _tokenize(self, b):
        if "messages" in b:
            text = render(
                b["messages"], b.get("tools"), b.get("add_generation_prompt", True)
            )
        else:
            text = b["prompt"]
        ids = VOCAB.encode(text)
        self._send({"count": len(ids), "tokens": ids, "max_model_len": 1 << 18})

    def _completions(self, b, busy):
        ids = b["prompt"]
        asst = VOCAB.id("<|assistant|>")
        last = max((i for i, t in enumerate(ids) if t == asst), default=0)
        plp = [None]
        for j in range(1, len(ids)):
            lp = -(0.1 + (_h(ids[j - 1], ids[j]) % 1000) / 1000.0)
            if State.bug and busy and j > last + 2 + 64:
                lp -= 3.0
            plp.append(
                {
                    str(ids[j]): {
                        "logprob": lp,
                        "rank": 1,
                        "decoded_token": VOCAB.i2s[ids[j]],
                    }
                }
            )
        self._send(
            {
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": " the",
                        "finish_reason": "length",
                        "prompt_logprobs": plp if b.get("prompt_logprobs") else None,
                    }
                ],
            }
        )

    def _chat(self, b, busy):
        prompt = b["messages"][-1]["content"]
        code = "```python" in prompt
        n = min(b.get("max_tokens") or 512, 4000)
        text = gen_text(prompt, int(n * 0.8), code)
        ids = VOCAB.encode(text)[:n]
        if State.bug and busy and len(ids) > 64:
            rng = random.Random(_h(prompt, "bug"))
            junk = "".join(
                chr(rng.randint(0x4E00, 0x9FFF)) + rng.choice(["", " ", "的"])
                for _ in range(n)
            )
            ids = ids[:64] + VOCAB.encode(junk)[: n - 64]
        text = VOCAB.decode(ids)
        finish = "length" if len(ids) >= n else "stop"
        self._send(
            {
                "object": "chat.completion",
                "model": "flint",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "reasoning": None,
                        },
                        "token_ids": ids if b.get("return_token_ids") else None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(VOCAB.encode(prompt)),
                    "completion_tokens": len(ids),
                },
            }
        )


def serve(port: int = 0, bug: bool = False) -> ThreadingHTTPServer:
    State.bug = bug
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--bug", action="store_true")
    a = ap.parse_args()
    s = serve(a.port, a.bug)
    print(f"mock on http://127.0.0.1:{s.server_port}/v1 bug={a.bug}", flush=True)
    threading.Event().wait()
