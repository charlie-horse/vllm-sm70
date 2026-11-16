#!/usr/bin/env bash
# Usage: prepull.sh IDS_FILE
# Pre-pull the eval images of IDS_FILE with run_one.sh's global rate limit
# (scripts/pull_lib.sh: >= 4 min apart under state/pull.lock, backoff on
# failures incl. 429). Detaches itself (nohup setsid, nice 19, ionice idle).
# Log: $R/runs/prepull.log. Never talks to the model endpoint.
# Pinned to PREPULL_CPUSET (default 12-15, inside run_all's CPUSET=0-15).
set -u
. "$HOME/flint-swebench/env.sh"
IDS=$(readlink -f "$1"); ST=$R/state; LOG=$R/runs/prepull.log
PREPULL_CPUSET=${PREPULL_CPUSET:-12-15}
mkdir -p "$ST" "$R/runs"
if [ "${_FLINT_PREPULL_CHILD:-}" != 1 ]; then
  _FLINT_PREPULL_CHILD=1 nohup setsid taskset -c "$PREPULL_CPUSET" \
    nice -n 19 ionice -c3 "$0" "$IDS" \
    >> "$LOG" 2>&1 < /dev/null &
  echo "prepull started (pid $!), log: $LOG"
  exit 0
fi
exec 8>"$ST/prepull.lock"
flock -n 8 || { echo "$(date '+%F %T') prepull already running"; exit 1; }
log() { echo "$(date '+%F %T') [prepull] $*" >> "$LOG"; }
. "$R/scripts/pull_lib.sh"
ids=$(grep -v '^\s*$' "$IDS"); n=$(echo "$ids" | wc -l); i=0; failed=0
log "start: $n ids from $IDS (pid $$)"
for ID in $ids; do
  i=$((i + 1))
  TAG=$(echo "$ID" | sed 's/__/_1776_/' | tr 'A-Z' 'a-z')
  IMG=docker.io/swebench/sweb.eval.x86_64.$TAG:latest
  if podman image exists "$IMG"; then log "$i/$n $ID present"; continue; fi
  t0=$(date +%s)
  if pull_image; then
    log "$i/$n $ID pulled ($(( $(date +%s) - t0 ))s incl. rate-limit wait)"
  else
    failed=$((failed + 1)); log "$i/$n $ID FAILED"
  fi
done
log "finished: $failed failed"
