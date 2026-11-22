#!/usr/bin/env python3
"""Send each prompt N times, record reasoning+content, report identity.

usage: greedy_run.py BASE_URL prompts.json out.json [repeats=2] [ref.json]
           [--model flint] [--reasoning-effort VALUE] [--keep-special-tokens]

A prompt entry may carry "images" (a list of URLs or data URIs); they are sent
as image_url parts before the text. With no options the request bodies are
{"model": "flint", "messages": [...], **params}.
"""
import argparse, json, sys, urllib.request


def build_body(prompt, params, model="flint", reasoning_effort=None, keep_special_tokens=False):
    content = prompt["content"]
    if prompt.get("images"):
        content = [{"type": "image_url", "image_url": {"url": u}} for u in prompt["images"]]
        content.append({"type": "text", "text": prompt["content"]})
    body = {"model": model, "messages": [{"role": "user", "content": content}], **params}
    if reasoning_effort is not None:
        body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "reasoning_effort": reasoning_effort}
    if keep_special_tokens:
        body["skip_special_tokens"] = False
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("repeats", nargs="?", type=int, default=2)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("--model", default="flint")
    ap.add_argument("--reasoning-effort", help="sent as chat_template_kwargs.reasoning_effort")
    ap.add_argument("--keep-special-tokens", action="store_true", help="send skip_special_tokens=false")
    args = ap.parse_args(argv)
    base = args.base.rstrip("/")
    P = json.load(open(args.prompts)); res = {}
    for p in P["prompts"]:
        runs = []
        for _ in range(args.repeats):
            body = build_body(p, P["params"], args.model, args.reasoning_effort, args.keep_special_tokens)
            r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            m = r["choices"][0]["message"]
            runs.append({"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "", "usage": r["usage"]})
        same = all(x["reasoning"] == runs[0]["reasoning"] and x["content"] == runs[0]["content"] for x in runs)
        res[p["id"]] = {"identical_within_run": same, "runs": runs}
        print(p["id"], "identical" if same else "DIFFERS", runs[0]["usage"]["prompt_tokens"], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    if args.ref:
        ref = json.load(open(args.ref))
        for k in res:
            a, b = res[k]["runs"][0], ref[k]["runs"][0]
            print("vs ref", k, "match" if (a["reasoning"], a["content"]) == (b["reasoning"], b["content"]) else "MISMATCH")


if __name__ == "__main__":
    sys.exit(main())
