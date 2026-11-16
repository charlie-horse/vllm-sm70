#!/usr/bin/env python3
"""usage: greedy_run.py BASE_URL prompts.json out.json [repeats=2]. Sends each prompt N times, records reasoning+content, reports identity."""
import json, sys, urllib.request
base, pf, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
P = json.load(open(pf)); res = {}
for p in P["prompts"]:
    runs = []
    for _ in range(reps):
        body = {"model": "flint", "messages": [{"role": "user", "content": p["content"]}], **P["params"]}
        r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        m = r["choices"][0]["message"]
        runs.append({"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "", "usage": r["usage"]})
    same = all(x["reasoning"] == runs[0]["reasoning"] and x["content"] == runs[0]["content"] for x in runs)
    res[p["id"]] = {"identical_within_run": same, "runs": runs}
    print(p["id"], "identical" if same else "DIFFERS", runs[0]["usage"]["prompt_tokens"], flush=True)
json.dump(res, open(out, "w"), indent=1)
if len(sys.argv) > 5:
    ref = json.load(open(sys.argv[5]))
    for k in res:
        a, b = res[k]["runs"][0], ref[k]["runs"][0]
        print("vs ref", k, "match" if (a["reasoning"], a["content"]) == (b["reasoning"], b["content"]) else "MISMATCH")
