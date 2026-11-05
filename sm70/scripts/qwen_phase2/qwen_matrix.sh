#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qwen Phase 2 matrix on c4130 (2x V100-32GB, TP2, --dtype half, TRITON_ATTN).
#
#   qwen_matrix.sh ROW      ROW = 1 | 2 | 3 | 4 | all | 2-noemu | 4-noemu | summary
#
#   1  Qwen3.6-35B-A3B-FP8                          no spec decode
#   2  Qwen3.6-35B-A3B-FP8 + Qwen3.6-35B-A3B-DFlash2  dflash, k=7, probabilistic
#   3  Qwen3.8-27B-FP8                              no spec decode
#   4  Qwen3.8-27B-FP8 + Qwen3.8-27B-DFlash2          dflash, k=7, probabilistic
#   2-noemu / 4-noemu: row 2 / 4 with VLLM_SM70_DFLASH2_EMULATION=0 (M7 A/B;
#      recorded as INFO, never gated, not part of `all`).
#
# Each row: vllm serve on 127.0.0.1:8100 -> /health -> probe.py -> results.json
# + server.log in $OUT/rowN/ -> server killed (trap on EXIT). Never touches
# systemd, the vllm service, /etc/nixos or activeModel.
#
# Env: VLLM_BIN, VLLM_PY (required), QWEN_STAGE, OUT, MAX_MODEL_LEN (32768),
#      HEALTH_TIMEOUT (seconds, 1500), GPU_MEM_UTIL (0.85).
# Exit: 0 all gated rows of this run PASS, 1 a row FAILed, 2 refused to start.
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PORT=8100
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1500}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVED_NAME=qwen

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[[ $# -eq 1 ]] || usage
ROW_ARG="$1"

# row -> "target_subdir draft_subdir(or -) gated(1/0) emulation_kill(1/0)"
row_def() {
  case "$1" in
    1) echo "Qwen3.6-35B-A3B-FP8 - 1 0" ;;
    2) echo "Qwen3.6-35B-A3B-FP8 Qwen3.6-35B-A3B-DFlash2 1 0" ;;
    3) echo "Qwen3.8-27B-FP8 - 1 0" ;;
    4) echo "Qwen3.8-27B-FP8 Qwen3.8-27B-DFlash2 1 0" ;;
    2-noemu) echo "Qwen3.6-35B-A3B-FP8 Qwen3.6-35B-A3B-DFlash2 0 1" ;;
    4-noemu) echo "Qwen3.8-27B-FP8 Qwen3.8-27B-DFlash2 0 1" ;;
    *) return 1 ;;
  esac
}

case "$ROW_ARG" in
  all) ROWS=(1 2 3 4) ;;
  summary)
    [[ -n "${VLLM_PY:-}" ]] || qp2_refuse "VLLM_PY is not set"
    exec "$VLLM_PY" "$QP2_DIR/probe.py" summary --out-dir "$OUT" ;;
  *) row_def "$ROW_ARG" >/dev/null || usage; ROWS=("$ROW_ARG") ;;
esac

# ---------------------------------------------------------------- preflight
qp2_require_exec VLLM_BIN
qp2_require_exec VLLM_PY
for r in "${ROWS[@]}"; do
  read -r tgt drf _ _ <<<"$(row_def "$r")"
  qp2_require_model_dir "row $r target" "$QWEN_STAGE/$tgt"
  [[ "$drf" == - ]] || qp2_require_model_dir "row $r draft" "$QWEN_STAGE/$drf"
done
if qp2_port_in_use "$PORT"; then qp2_refuse "something already listens on port $PORT"; fi
qp2_check_gpus_idle
mkdir -p "$OUT"
qp2_log "OUT=$OUT rows=${ROWS[*]} max_model_len=$MAX_MODEL_LEN"

# ---------------------------------------------------------------- one row
fail_json() {  # FILE ROW GATED MSG
  "$VLLM_PY" -c 'import json,sys; json.dump({"row":sys.argv[2],"gated":sys.argv[3]=="1","status":"FAIL" if sys.argv[3]=="1" else "INFO","error":sys.argv[4],"phases":{}},open(sys.argv[1],"w"),indent=2)' \
    "$1" "$2" "$3" "$4"
}

run_row() {  # ROW  (runs in a subshell: its EXIT trap stops this row's server)
  local row="$1" tgt drf gated nokill
  read -r tgt drf gated nokill <<<"$(row_def "$row")"
  local dir="$OUT/row$row" spec=()
  mkdir -p "$dir" || { qp2_log "row $row: cannot create $dir"; return 1; }
  rm -f "$dir/results.json" || { qp2_log "row $row: cannot clear $dir/results.json"; return 1; }
  if [[ "$drf" != - ]]; then
    spec=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$QWEN_STAGE/$drf\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}")
  fi
  local cmd=("$VLLM_BIN" serve "$QWEN_STAGE/$tgt" --served-model-name "$SERVED_NAME"
    --tensor-parallel-size 2 --dtype half --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN" --attention-backend "$QP2_ATTN_BACKEND"
    --host 127.0.0.1 --port "$PORT" "${spec[@]}")
  local kill_env=()
  [[ "$nokill" == 1 ]] && kill_env=(VLLM_SM70_DFLASH2_EMULATION=0)
  {
    printf 'row=%s gated=%s\n' "$row" "$gated"
    printf 'env: CUDA_VISIBLE_DEVICES=0,1 %s\n' "${kill_env[*]:-}"
    printf '%q ' "${cmd[@]}"; echo
  } >"$dir/cmd.txt" || { qp2_log "row $row: cannot write $dir/cmd.txt"; return 1; }

  local label="$tgt"
  [[ "$drf" != - ]] && label+=" + $drf"
  [[ "$nokill" == 1 ]] && label+=", VLLM_SM70_DFLASH2_EMULATION=0"
  qp2_log "row $row: starting server ($label)"
  env CUDA_VISIBLE_DEVICES=0,1 "${kill_env[@]}" setsid "${cmd[@]}" >"$dir/server.log" 2>&1 &
  SERVER_PID=$!
  trap stop_server EXIT
  trap 'exit 130' INT TERM HUP

  local url="http://127.0.0.1:$PORT" deadline=$((SECONDS + HEALTH_TIMEOUT)) t0=$SECONDS
  until "$VLLM_PY" "$QP2_DIR/probe.py" health --url "$url"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      qp2_log "row $row: server exited before /health; last log lines:"
      tail -n 40 "$dir/server.log" || true
      fail_json "$dir/results.json" "$row" "$gated" "server exited before /health (see server.log)"
      return "$gated"
    fi
    if (( SECONDS >= deadline )); then
      fail_json "$dir/results.json" "$row" "$gated" "no /health within ${HEALTH_TIMEOUT}s"
      qp2_log "row $row: no /health within ${HEALTH_TIMEOUT}s"
      return "$gated"
    fi
    sleep 5
  done
  qp2_log "row $row: healthy after $((SECONDS - t0))s; probing"
  echo "startup_seconds=$((SECONDS - t0))" >>"$dir/cmd.txt" \
    || { qp2_log "row $row: cannot write $dir/cmd.txt"; return 1; }
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >"$dir/nvidia-smi.csv" || true

  local probe=("$VLLM_PY" "$QP2_DIR/probe.py" run --url "$url" --model "$SERVED_NAME"
    --row "$row" --out "$dir/results.json" --max-model-len "$MAX_MODEL_LEN")
  [[ "$drf" != - ]] && probe+=(--spec)
  [[ "$gated" == 0 ]] && probe+=(--no-gate)
  "${probe[@]}" 2>&1 | tee "$dir/probe.log"
  local st=("${PIPESTATUS[@]}")
  (( st[1] == 0 )) || { qp2_log "row $row: cannot write $dir/probe.log"; return 1; }
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
for r in "${ROWS[@]}"; do
  ( run_row "$r" ) || rc=1
  if ! qp2_wait_gpus_idle 180 || qp2_port_in_use "$PORT"; then
    qp2_log "GPUs not idle / port $PORT still busy 180 s after row $r; stopping the matrix"
    # ps|grep, not pgrep -f: over SSH pgrep matches its own bash -c line.
    # shellcheck disable=SC2009
    ps -eo pid,args | grep "[-]-port $PORT" || true
    rc=1
    break
  fi
done

echo
echo "== summary ($OUT) =="
# Lists every row in $OUT (earlier runs too); the exit code covers this run only.
"$VLLM_PY" "$QP2_DIR/probe.py" summary --out-dir "$OUT" || true
exit "$rc"
