#!/usr/bin/env bash
# Keep tailscale + ssh forward to the model endpoint alive.
while true; do
  if ! curl -s -m 10 http://127.0.0.1:18000/v1/models | grep -q flint; then
    echo "$(date) endpoint down, restarting" 
    bash "${TS_RESTART:?set TS_RESTART to tailscale restart script}"
    pkill -f 'ssh -fN -L 18000' ; sleep 2
    ssh -fN -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 18000:127.0.0.1:8000 c4130
  fi
  sleep 60
done
