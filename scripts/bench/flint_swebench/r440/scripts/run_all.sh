#!/usr/bin/env bash
# Usage: [env knobs] run_all.sh IDS_FILE ATTEMPT [P]
# Detaches itself (nohup setsid); log: $RUN_DIR/driver.log.
# P instances in parallel (default 2, or env P); resumable (skips done_
# markers); up to 3 passes so instances killed by the memory watchdog are
# re-run after the others.
# Env knobs (unset = previous behavior; also read by run_one.sh/watchdog.sh):
#   RUN_DIR=$R/runs  run root (attA/ and driver.log live here)
#   P=2              parallel instances
#   CPUSET=0-15      pin driver, agents and the podman service (taskset -c)
#   TASK_CPUS, KEEP_IMAGES, START_FLOOR_GB   see run_one.sh
#   WD_FLOOR_GB, WD_INTERVAL                 see watchdog.sh
set -u
. "$HOME/flint-swebench/env.sh"
IDS=$(readlink -f "$1"); A=$2; P=${3:-${P:-2}}; ST=$R/state
export RUN_DIR=${RUN_DIR:-$R/runs}
mkdir -p "$RUN_DIR" "$ST"
if [ "${_FLINT_RUN_ALL_CHILD:-}" != 1 ]; then
  PIN=(); [ -n "${CPUSET:-}" ] && PIN=(taskset -c "$CPUSET")
  _FLINT_RUN_ALL_CHILD=1 nohup setsid "${PIN[@]}" "$0" "$IDS" "$A" "$P" \
    >> "$RUN_DIR/driver.log" 2>&1 < /dev/null &
  echo "driver started (pid $!), log: $RUN_DIR/driver.log"
  exit 0
fi
# lock per run dir; the default run dir keeps the old lock name
LOCK=$ST/driver_att$A.lock
if [ "$RUN_DIR" != "$R/runs" ]; then
  LOCK=$ST/driver_$(echo "${RUN_DIR#"$R"/}" | tr / _)_att$A.lock
fi
exec 8>"$LOCK"
flock -n 8 || { echo "$(date '+%F %T') driver att$A already running"; exit 1; }
log() { echo "$(date '+%F %T') [driver att$A] $*"; }
"$R/scripts/podman_service.sh" || { log "podman service failed"; exit 1; }
if [ -n "${CPUSET:-}" ]; then  # eval containers are children of the service
  for pid in $(pgrep -f "podman system service .*$XDG_RUNTIME_DIR/podman/"); do
    taskset -a -p -c "$CPUSET" "$pid" > /dev/null && log "pinned service $pid"
  done
fi
if ! pgrep -f "$R/scripts/watchdog.sh" >/dev/null; then
  nohup setsid "$R/scripts/watchdog.sh" > /dev/null 2>&1 < /dev/null 8>&- &
  WD=$!
fi
log "start: $(grep -c . "$IDS") ids from $IDS, P=$P, run dir $RUN_DIR"
for pass in 1 2 3; do
  pending=$(grep -v '^\s*$' "$IDS" | while read -r id; do
    [ -f "$RUN_DIR/att$A/eval/done_$id" ] || echo "$id"; done)
  [ -z "$pending" ] && break
  log "pass $pass: $(echo "$pending" | wc -l) pending"
  echo "$pending" | xargs -P"$P" -I{} "$R/scripts/run_one.sh" {} "$A"
done
left=$(grep -v '^\s*$' "$IDS" | while read -r id; do
  [ -f "$RUN_DIR/att$A/eval/done_$id" ] || echo "$id"; done | wc -l)
log "finished; $left without done_ marker"
[ -n "${WD:-}" ] && kill "$WD" 2>/dev/null
