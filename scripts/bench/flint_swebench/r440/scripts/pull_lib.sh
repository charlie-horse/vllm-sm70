# Sourced by run_one.sh and prepull.sh: globally rate-limited image pulls.
# Needs: ST (state dir), IMG, LOG (pull output is appended here), log().
# pull_image returns 0 = image present, 1 = gave up, 2 = missing upstream.
PULL_GAP=240                                # >= 4 min between pulls (global)
PULL_BACKOFF=(0 120 240 480 960 1920 3600)  # s: then 2,4,8,16,32,60 min

pull_once() {
  (
    flock 9
    last=$(cat "$ST/last_pull" 2>/dev/null || echo 0)
    gap=$(( last + PULL_GAP - $(date +%s) ))
    [ "$gap" -gt 0 ] && sleep "$gap"
    date +%s > "$ST/last_pull"
    nice -n 19 ionice -c3 timeout 3600 podman pull -q "$IMG"
  ) 9>"$ST/pull.lock" >> "$LOG" 2>&1
}

pull_image() {  # retries with backoff on any failure (429 included)
  local d
  for d in "${PULL_BACKOFF[@]}"; do
    if [ "$d" -gt 0 ]; then
      log "pull failed, retry in $((d / 60)) min"; sleep "$d"
    fi
    if podman image exists "$IMG" || pull_once; then return 0; fi
    if tail -5 "$LOG" |
        grep -qiE 'manifest unknown|name unknown|repository does not exist'; then
      log "image does not exist upstream"; return 2
    fi
  done
  return 1
}
