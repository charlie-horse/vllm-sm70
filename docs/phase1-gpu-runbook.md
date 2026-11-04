# Phase 1 GPU test on c4130 (handoff to Bea)

Bea runs the window with her runbook (`systemd-run --unit=vsm70-test` as user
`vllm` with the vllm unit's env, Flint stop/restore). This file covers what is
specific to the vllm-sm70 build: the closure, the test commands, and the
checks with their expected output.

`$OUT` = outPath of `.#vllm` (r440, `~/vllm-sm70-work/felix/result-vllm`).
Expected (nixpkgs 7dcdd87, unchanged flake):
`/nix/store/zlbyyjzl4kifkdv1hl7rf08cqk78yvdr-vllm-sm70-0.33.0`.
Confirm it after the build.

## 1. Closure to c4130 (before the window, no service impact, no sudo)
c4130 has the harmonia binary cache on r440 as a substituter and trusts
`r440-cache-1`, so realising the outPath as charlie pulls it:
```sh
# on r440
nix path-info -Sh $OUT          # closure size; c4130 has 177G free
# on c4130
nix-store --realise $OUT
```
Fallback (only if harmonia fails): `nix copy --to ssh-ng://c4130 $OUT` from r440
needs a trusted user on c4130, so it's a sudo step.

## 2. Model (before the window, CPU/network only, ungated, no token)
Qwen/Qwen3-0.6B (BF16 safetensors, served as FP16 via `--dtype half`), in a
scratch dir the `vllm` user can read:
```sh
sudo -S -p '' install -d -m 0755 -o charlie /var/tmp/vsm70-test <<<"$SERVER_PASSWORD"
mkdir -p /var/tmp/vsm70-test/models
HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/var/tmp/vsm70-test/hf $OUT/bin/python -I -c \
 "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen3-0.6B', local_dir='/var/tmp/vsm70-test/models/Qwen3-0.6B'))"
chmod -R a+rX /var/tmp/vsm70-test/models
```

## 3. Static checks (before the window; one small CUDA context, no load)
```sh
cd /var/tmp && $OUT/bin/python -c "import torch, vllm, vllm._C; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), vllm.__version__, torch.cuda.get_device_capability(0))"
```
Expected: `2.14.0+cu126 12.6 [..., 'sm_70', ...] 0.33.0 (7, 0)`.
(No `-I` here: the wrapper puts the vLLM tree on PYTHONPATH, and `-I` would drop it.)

## 4. Test commands (inside the window, Flint stopped)
Unit env on top of the vllm unit's env:
```
VLLM_ATTENTION_BACKEND=TRITON_ATTN
HF_HUB_OFFLINE=1
VLLM_NO_USAGE_STATS=1
HOME=/var/tmp/vsm70-test/home-vllm         # writable by vllm (create it 0700 owned by vllm)
VLLM_CACHE_ROOT=/var/tmp/vsm70-test/home-vllm/vllm
TRITON_CACHE_DIR=/var/tmp/vsm70-test/home-vllm/triton
```
TP1 (GPU0):
```sh
CUDA_VISIBLE_DEVICES=0  $OUT/bin/vllm serve /var/tmp/vsm70-test/models/Qwen3-0.6B \
  --served-model-name t --dtype half --tensor-parallel-size 1 \
  --max-model-len 4096 --gpu-memory-utilization 0.80 \
  --host 127.0.0.1 --port 8001
```
TP2: the same with `CUDA_VISIBLE_DEVICES=0,1 --tensor-parallel-size 2`.
LMCache stays running (~328 MiB/GPU); 0.80 leaves room for it.
Use `RuntimeMaxSec=900` on the transient unit; first start includes Triton JIT
and torch.compile (expect 2–4 min to /health).

Client (any python, e.g. the old env `.../c4130-vllm-env/bin/python3.12 -I`, or curl):
```sh
until curl -sf localhost:8001/health; do sleep 5; done
curl -s localhost:8001/v1/models
curl -s localhost:8001/v1/chat/completions -H 'content-type: application/json' -d \
 '{"model":"t","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl -s localhost:8001/v1/completions -H 'content-type: application/json' -d \
 '{"model":"t","prompt":"The three primary colors are","max_tokens":32,"temperature":0}'
```

## 5. Checks and expected output
| check | expected |
|---|---|
| `/v1/models` | `"id":"t"`, `max_model_len` 4096 |
| chat completion | content contains `Paris` |
| text completion | a coherent English continuation (e.g. "red, yellow and blue") |
| unit log: `journalctl -u vsm70-test \| grep -iE "attention backend\|TRITON_ATTN"` | a line selecting `TRITON_ATTN` (e.g. `Using AttentionBackendEnum.TRITON_ATTN backend`) |
| unit log: `grep -iE "dtype\|float16"` | `dtype=torch.float16` |
| unit log errors: `grep -iE "Traceback\|no kernel image\|illegal"` | nothing |
| TP2 log | 2 workers (`TP0`, `TP1`), NCCL init OK, same completions |
| `nvidia-smi` during the run | the test process on GPU0 (TP1), or GPU0+1 (TP2) |

Send me (Felix) the log excerpts and the JSON responses; I attach them as the
exit evidence.

## 6. Outage estimate
| step | time |
|---|---|
| stop Flint (kill + stop) | ~1 min |
| TP1: start (JIT/compile) + checks | ~5 min |
| TP2: start + checks | ~6 min |
| restore Flint to `/v1/models` OK (Bea's number) | 3–4.5 min |
| **total** | **~15–17 min; ask for 25 min** |

If a test start fails, collect `journalctl -u vsm70-test -n 300` and go
straight to the restore; I debug offline on r440.

Scratch cleanup (`/var/tmp/vsm70-test`) only with Astro's go-ahead.
