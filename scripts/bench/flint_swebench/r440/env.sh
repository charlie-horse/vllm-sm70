# source ~/flint-swebench/env.sh  -- r440 SWE-bench harness environment
export R=$HOME/flint-swebench
export FLINT_API_BASE=http://<c4130-lan-ip>:8000/v1   # c4130 vLLM over LAN (tailnet ACL blocks <c4130-tailnet-ip>)  [IPs redacted on import]
# podman (nix, rootless, cgroupfs; runtime dir under $R so it survives logout w/o linger)
export XDG_RUNTIME_DIR=$R/run
export CONTAINERS_CONF=$R/conf/containers.conf
export CONTAINERS_STORAGE_CONF=$R/conf/storage.conf
export PODMAN=$R/podman-pkg/bin/podman
export PATH=$R/podman-pkg/bin:$R/bin:/run/wrappers/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock
export MSWEA_DOCKER_EXECUTABLE=$PODMAN
# manylinux wheels on NixOS
export LD_LIBRARY_PATH=$(readlink -f $R/lib-cc-lib)/lib:$(readlink -f $R/lib-z)/lib
# mini-swe-agent
export MSWEA_GLOBAL_CONFIG_DIR=$R/.mswea
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_SILENT_STARTUP=1
# tenacity wait_exponential(min=4,max=60): 4+4+8+16+32 then 60s each -> 25 attempts ~= 20 min
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=25
export UV_CACHE_DIR=$R/.uv-cache UV_PYTHON_DOWNLOADS=never
export HF_HOME=$R/.hf
