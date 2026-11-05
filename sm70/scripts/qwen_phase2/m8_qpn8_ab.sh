#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# M8 QPN8 A/B on c4130 (2x V100-32GB, TP2, --dtype half, TRITON_ATTN).
#
#   m8_qpn8_ab.sh
#
# Serves Qwen3.8-27B-FP8 + Qwen3.8-27B-DFlash2 (qwen_matrix.sh row-4
# settings: dflash, k=7, probabilistic) with --max-num-seqs 4, twice:
#   on   default (block QPN8, Sm70Config.fp8_block_qpn8=True)
#   off  VLLM_SM70_FP8_BLOCK_QPN8=0 (plain M6 TurboMind)
# QP2_QPN8_COPY=auto|true|false (default auto) sets
# Sm70Config.fp8_block_qpn8_turbomind_copy via --additional-config, on both
# arms. auto keeps a second TurboMind copy of each weight iff
# 4 x (7 + 1) = 32 > K09 QPN8_MAX_M (no copy at QPN8_MAX_M 32, a copy at 8:
# ~2x dense FP8 per GPU). Use false if the "on" arm runs out of memory.
# Each: 127.0.0.1:8100 -> /health -> probe.py ab-run (decode tok/s at
# concurrency 1-4: ~1K-token prompt, 256 forced tokens with ignore_eos,
# temperature 0, median of 3; greedy first 32 tokens of 5 fixed prompts)
# -> server killed (trap on EXIT). Then probe.py ab-compare.
# PASS = QPN8/M6 tok/s >= 0.98 at every concurrency and greedy outputs match.
# The M8 admission/rejection log lines are copied to $OUT/m8/<arm>/m8_log.txt.
# Never touches systemd, the vllm service, /etc/nixos or activeModel.
#
# Env: VLLM_BIN, VLLM_PY (required), QWEN_STAGE, OUT, MAX_MODEL_LEN (32768),
#      HEALTH_TIMEOUT (seconds, 1500), GPU_MEM_UTIL (0.85), QP2_QPN8_COPY (auto).
# Exit: 0 PASS, 1 FAIL, 2 refused to start.
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PORT=8100
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1500}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS=4
SERVED_NAME=qwen
TARGET="$QWEN_STAGE/Qwen3.8-27B-FP8"
DRAFT="$QWEN_STAGE/Qwen3.8-27B-DFlash2"
M8_LOG_RE='SM70 FP8 QPN8'
QP2_QPN8_COPY="${QP2_QPN8_COPY:-auto}"

[[ $# -eq 0 ]] || { sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
case "$QP2_QPN8_COPY" in
  auto) EXTRA_CFG=() ;;
  true|false) EXTRA_CFG=(--additional-config "{\"sm70\": {\"fp8_block_qpn8_turbomind_copy\": $QP2_QPN8_COPY}}") ;;
  *) qp2_refuse "QP2_QPN8_COPY=$QP2_QPN8_COPY (want auto, true or false)" ;;
esac

# ---------------------------------------------------------------- preflight
qp2_require_exec VLLM_BIN
qp2_require_exec VLLM_PY
qp2_require_model_dir "target" "$TARGET"
qp2_require_model_dir "draft" "$DRAFT"
if qp2_port_in_use "$PORT"; then qp2_refuse "something already listens on port $PORT"; fi
qp2_check_gpus_idle
AB="$OUT/m8"
mkdir -p "$AB"
qp2_log "OUT=$AB max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN qpn8_copy=$QP2_QPN8_COPY"

# ---------------------------------------------------------------- one arm
run_arm() {  # ARM (on|off)  (runs in a subshell: its EXIT trap stops the server)
  local arm="$1" dir="$AB/$1" kill_env=()
  [[ "$arm" == off ]] && kill_env=(VLLM_SM70_FP8_BLOCK_QPN8=0)
  mkdir -p "$dir" || { qp2_log "$arm: cannot create $dir"; return 1; }
  rm -f "$dir/ab.json" "$dir/m8_log.txt"
  local spec="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}"
  local cmd=("$VLLM_BIN" serve "$TARGET" --served-model-name "$SERVED_NAME"
    --tensor-parallel-size 2 --dtype half --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --attention-backend "$QP2_ATTN_BACKEND"
    --host 127.0.0.1 --port "$PORT" --speculative-config "$spec" "${EXTRA_CFG[@]}")
  {
    printf 'arm=%s\n' "$arm"
    printf 'env: CUDA_VISIBLE_DEVICES=0,1 %s\n' "${kill_env[*]:-}"
    printf '%q ' "${cmd[@]}"; echo
  } >"$dir/cmd.txt" || { qp2_log "$arm: cannot write $dir/cmd.txt"; return 1; }

  qp2_log "$arm: starting server${kill_env[*]:+ (${kill_env[*]})}"
  env CUDA_VISIBLE_DEVICES=0,1 "${kill_env[@]}" setsid "${cmd[@]}" >"$dir/server.log" 2>&1 &
  SERVER_PID=$!
  trap stop_server EXIT
  trap 'exit 130' INT TERM HUP

  local url="http://127.0.0.1:$PORT" deadline=$((SECONDS + HEALTH_TIMEOUT)) t0=$SECONDS
  until "$VLLM_PY" "$QP2_DIR/probe.py" health --url "$url"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      qp2_log "$arm: server exited before /health; last log lines:"
      tail -n 40 "$dir/server.log" || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      qp2_log "$arm: no /health within ${HEALTH_TIMEOUT}s"
      return 1
    fi
    sleep 5
  done
  qp2_log "$arm: healthy after $((SECONDS - t0))s; probing"
  echo "startup_seconds=$((SECONDS - t0))" >>"$dir/cmd.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >"$dir/nvidia-smi.csv" || true
  grep -F "$M8_LOG_RE" "$dir/server.log" >"$dir/m8_log.txt" || echo "(no M8 log line)" >"$dir/m8_log.txt"
  qp2_log "$arm: M8 log: $(head -n 1 "$dir/m8_log.txt")"

  "$VLLM_PY" "$QP2_DIR/probe.py" ab-run --url "$url" --model "$SERVED_NAME" \
    --label "$arm" --out "$dir/ab.json" 2>&1 | tee "$dir/probe.log"
  local st=("${PIPESTATUS[@]}")
  return "${st[0]}"
}

# shellcheck disable=SC2329  # invoked via trap
stop_server() {
  [[ -n "${SERVER_PID:-}" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    qp2_log "stopping server (pgid $SERVER_PID)"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  fi
  # SIGKILL whatever is left of the group (TP workers included).
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  SERVER_PID=
}

# ---------------------------------------------------------------- main
rc=0
for arm in on off; do
  ( run_arm "$arm" ) || rc=1
  if ! qp2_wait_gpus_idle 180 || qp2_port_in_use "$PORT"; then
    qp2_log "GPUs not idle / port $PORT still busy 180 s after arm $arm; stopping"
    # ps|grep, not pgrep -f: over SSH pgrep matches its own bash -c line.
    # shellcheck disable=SC2009
    ps -eo pid,args | grep "[-]-port $PORT" || true
    exit 1
  fi
done

echo
echo "== M8 QPN8 A/B ($AB) =="
for arm in on off; do echo "$arm M8 log: $(head -n 1 "$AB/$arm/m8_log.txt" 2>/dev/null || echo n/a)"; done
"$VLLM_PY" "$QP2_DIR/probe.py" ab-compare --on "$AB/on/ab.json" --off "$AB/off/ab.json" \
  --out "$AB/ab_summary.json" | tee "$AB/summary.txt" || rc=1
exit "$rc"
