#!/usr/bin/env bash
# (re)start the rootless podman API socket used by the swebench docker SDK
# CPUSET set (run_all.sh): start the service pinned with taskset -c.
. $HOME/flint-swebench/env.sh
SOCK=$XDG_RUNTIME_DIR/podman/podman.sock
ping() { curl -s -m "$1" --unix-socket $SOCK http://d/_ping | grep -q OK; }
mkdir -p $XDG_RUNTIME_DIR/podman
if ping 5; then exit 0; fi
rm -f $SOCK
# After a host reboot podman refuses to start until its runtime state is
# cleared ("boot ID differs"). That state is under our own runroot only;
# images live in graphroot ($R/storage) and are untouched.
if $PODMAN info 2>&1 | grep -q "boot ID differs"; then
  rm -rf "$R/run/containers" "$R/run/libpod/tmp"
fi
PIN=(); [ -n "${CPUSET:-}" ] && PIN=(taskset -c "$CPUSET")
nohup setsid "${PIN[@]}" nice -n 19 ionice -c3 $PODMAN system service --time=0 \
  unix://$SOCK > $R/run/podman-service.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  ping 2 && exit 0
  sleep 0.5
done
echo "podman service failed to start" >&2; exit 1
