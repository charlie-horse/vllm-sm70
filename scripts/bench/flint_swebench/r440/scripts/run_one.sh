#!/usr/bin/env bash
# Usage: run_one.sh INSTANCE_ID ATTEMPT
#   pre-pull (global rate limit) -> agent (podman, niced, API retry/re-run)
#   -> official eval (podman socket) -> image cleanup.
# Output layout (unchanged, read by scripts/analyze.py):
#   runs/attA/{agent/ID, eval/done_ID, log/ID.{agent,eval}.log}
# done_ID may carry "infra": "pull_failed" | "api_failed".
# Exit 3 = killed by the memory watchdog (no done_ marker -> re-run later).
# Env knobs (unset = previous behavior):
#   RUN_DIR=$R/runs      run root; output goes to $RUN_DIR/attA
#   TASK_CPUS=N          task container --cpus (flint.yaml default: 6)
#   KEEP_IMAGES=1        skip cleanup_image (images were pre-pulled)
#   START_FLOOR_GB=12    wait for this much MemAvailable before starting
set -u
. "$HOME/flint-swebench/env.sh"
: "${FLINT_API_BASE:?}"
RUN_DIR=${RUN_DIR:-$R/runs}
ID=$1; A=$2; OUT=$RUN_DIR/att$A; ST=$R/state
START_FLOOR_GB=${START_FLOOR_GB:-12}
mkdir -p "$OUT/agent" "$OUT/eval" "$OUT/log" "$ST/active"
REPORT=$OUT/eval/done_$ID
[ -f "$REPORT" ] && exit 0
LOG=$OUT/log/$ID.run.log
TAG=$(echo "$ID" | sed 's/__/_1776_/' | tr 'A-Z' 'a-z')
IMG=docker.io/swebench/sweb.eval.x86_64.$TAG:latest
API_RE='^(APIConnectionError|APIError|Timeout|InternalServerError'
API_RE+='|ServiceUnavailableError|BadGatewayError|APIResponseValidationError)$'
agent_s=0; eval_s=0

log() {
  echo "$(date '+%F %T') [$ID a$A] $*" | tee -a "$LOG" >> "$RUN_DIR/driver.log"
}
memavail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
done_marker() {  # $1 = infra reason or empty
  local infra=""; [ -n "${1:-}" ] && infra=",\"infra\":\"$1\""
  echo "{\"id\":\"$ID\",\"attempt\":$A,\"agent_s\":$agent_s,\"eval_s\":$eval_s$infra}" \
    > "$REPORT"
}
cleanup_image() {
  [ "${KEEP_IMAGES:-0}" = 1 ] && return 0
  # Hold the pull lock so pruning cannot drop shared layers mid-pull.
  (
    flock 9
    podman rmi -f "$IMG" >/dev/null 2>&1
    podman image prune -f >/dev/null 2>&1
  ) 9>"$ST/pull.lock"
}
memkilled() { [ -f "$ST/memkill_$ID" ]; }
finish_memkill() {
  log "memory watchdog killed this instance; no done_ marker, will re-run"
  rm -f "$ST/memkill_$ID" "$ST/active/$ID" "$ST/pid_$ID"
  cleanup_image; exit 3
}

# --- memory guard: wait for headroom before starting
while [ "$(memavail_gb)" -lt "$START_FLOOR_GB" ]; do
  log "waiting: MemAvailable $(memavail_gb)G < ${START_FLOOR_GB}G"; sleep 30
done
"$R/scripts/podman_service.sh" || { log "podman service down"; exit 4; }
echo "$IMG" > "$ST/active/$ID"; rm -f "$ST/memkill_$ID"

# --- a. pre-pull, globally rate limited, with backoff
. "$R/scripts/pull_lib.sh"
pulled=0
pull_image && pulled=1
if [ $pulled = 0 ]; then
  log "pull_failed"; done_marker pull_failed; rm -f "$ST/active/$ID"; exit 0
fi

# --- b/c. agent; on API/connection failure re-run from scratch (max 3 tries)
wait_endpoint() {  # up to 30 min; the agent itself retries ~20 min per call
  for _ in $(seq 1 60); do
    curl -s -m 10 "$FLINT_API_BASE/models" | grep -q flint && return 0
    log "endpoint $FLINT_API_BASE down, waiting"; sleep 30
  done
  return 1
}
exit_status() {
  "$R/.venv/bin/python" -I - "$OUT/agent/$ID" "$ID" <<'PY'
import glob, sys, yaml
d, iid = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{d}/exit_statuses_*.yaml"), reverse=True):
    data = yaml.safe_load(open(f)) or {}
    for st, ids in (data.get("instances_by_exit_status") or {}).items():
        if iid in (ids or []):
            print(st); sys.exit()
PY
}
# watchdog kills the PID in $ST/pid_$ID (no pkill -f); run in bg + wait
run_tracked() {
  local rc; "$@" & echo $! > "$ST/pid_$ID"; wait $!; rc=$?
  rm -f "$ST/pid_$ID"; return $rc
}
CPU_ARGS=()
if [ -n "${TASK_CPUS:-}" ]; then  # flint.yaml run_args with --cpus replaced
  CPU_ARGS=(-c "environment.run_args=$("$R/.venv/bin/python" -I - \
    "$R/flint.yaml" "$TASK_CPUS" <<'PY'
import json, sys, yaml
args = yaml.safe_load(open(sys.argv[1]))["environment"]["run_args"]
args = [a for a in args if not a.startswith("--cpus")]
print(json.dumps(args + [f"--cpus={sys.argv[2]}"]))
PY
)")
fi
cd "$R"
status=""
for try in 1 2 3; do
  if [ -d "$OUT/agent/$ID" ]; then
    mkdir -p "$OUT/agent_retired"
    mv "$OUT/agent/$ID" "$OUT/agent_retired/$ID.$(date +%s)"
  fi
  wait_endpoint
  log "agent start (try $try)"
  t0=$(date +%s)
  PYTHONPATH=$R/adapter run_tracked timeout 6h nice -n 19 ionice -c3 \
    "$R/.venv/bin/mini-extra" swebench --subset verified --split test \
    --filter "^${ID}\$" -c swebench.yaml -c "$R/flint.yaml" "${CPU_ARGS[@]}" \
    -c "model.model_kwargs.api_base=$FLINT_API_BASE" \
    -w 1 -o "$OUT/agent/$ID" > "$OUT/log/$ID.agent.log" 2>&1
  agent_s=$(( agent_s + $(date +%s) - t0 ))
  memkilled && finish_memkill
  status=$(exit_status)
  log "agent exit_status=${status:-?} (${agent_s}s total)"
  [[ "$status" =~ $API_RE ]] || break
  [ $try -lt 3 ] && log "API failure, re-running from scratch"
done
if [[ "$status" =~ $API_RE ]]; then
  log "api_failed"; done_marker api_failed
  rm -f "$ST/active/$ID"; cleanup_image; exit 0
fi

# --- d. official eval through the podman docker-compatible socket
t1=$(date +%s)
cd "$OUT/eval"
if [ -s "$OUT/agent/$ID/preds.json" ]; then
  run_tracked nice -n 19 ionice -c3 \
    "$R/.venv/bin/python" -m swebench.harness.run_evaluation \
    -d SWE-bench/SWE-bench_Verified -s test -i "$ID" \
    -p "$OUT/agent/$ID/preds.json" --max_workers 1 -id "att${A}_$ID" \
    --timeout 1800 > "$OUT/log/$ID.eval.log" 2>&1
fi
eval_s=$(( $(date +%s) - t1 ))
memkilled && finish_memkill
done_marker ""
log "done (agent ${agent_s}s, eval ${eval_s}s)"
rm -f "$ST/active/$ID"
cleanup_image
