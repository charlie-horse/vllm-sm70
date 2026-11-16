#!/usr/bin/env bash
# Every WD_INTERVAL s (default 20): if MemAvailable < WD_FLOOR_GB (default 8),
# kill the most recently started minisweagent-* / sweb.eval.* container and
# its runner (PID from state/pid_ID, written by run_one.sh); run_one.sh then
# exits without a done_ marker and run_all.sh re-runs it in a later pass.
. "$HOME/flint-swebench/env.sh"
ST=$R/state; WLOG=${RUN_DIR:-$R/runs}/driver.log
FLOOR=${WD_FLOOR_GB:-8}; IVL=${WD_INTERVAL:-20}
while true; do
  kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$kb" -lt $((FLOOR * 1048576)) ]; then
    line=$(podman ps --format json | "$R/.venv/bin/python" -I -c '
import json, sys
cs = [c for c in json.load(sys.stdin)
      if any(n.startswith(("minisweagent-", "sweb.eval.")) for n in c["Names"])]
if cs:
    c = max(cs, key=lambda c: c["Created"])
    print(c["Id"], c["Names"][0], c["Image"])')
    if [ -n "$line" ]; then
      read -r cid name image <<< "$line"
      key=${image##*/}   # sweb.eval.x86_64.<tag>:latest, registry-agnostic
      iid=$(grep -lF "$key" "$ST"/active/* 2>/dev/null | head -1 | xargs -r basename)
      echo "$(date '+%F %T') [watchdog] MemAvailable $((kb / 1024))M < ${FLOOR}G:" \
        "killing $name ($image) instance=${iid:-?}" >> "$WLOG"
      if [ -n "$iid" ]; then
        touch "$ST/memkill_$iid"
        pid=$(cat "$ST/pid_$iid" 2>/dev/null)
        [ -n "$pid" ] && kill "$pid" 2>/dev/null
      fi
      podman rm -f -t 0 "$cid" >/dev/null 2>&1
    else
      echo "$(date '+%F %T') [watchdog] MemAvailable low, no container" >> "$WLOG"
    fi
  fi
  sleep "$IVL"
done
