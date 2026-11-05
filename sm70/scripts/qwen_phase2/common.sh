# SPDX-License-Identifier: Apache-2.0
# Shared preflight for the Qwen Phase 2 window scripts. Sourced, not run.
# Touches nothing outside $OUT: no systemd, no vllm service, no /etc/nixos.
# shellcheck shell=bash

QP2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2034  # used by m7_e2e.sh
QP2_REPO="$(cd "$QP2_DIR/../../.." && pwd)"
QWEN_STAGE="${QWEN_STAGE:-$HOME/vllm-sm70-work/mara/qwen-weights}"
OUT="${OUT:-$HOME/vllm-sm70-work/bea/qwen-phase2/$(date +%Y%m%dT%H%M)}"
QP2_GPU_IDLE_MIB="${QP2_GPU_IDLE_MIB:-1024}"   # >1 GiB used => prod still up

# Env defaults from docs/phase1-gpu-runbook.md section 4 (only if unset).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

qp2_log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# exit 2 = refused to start (precondition), exit 1 = a test failed.
qp2_refuse() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }

qp2_require_exec() {  # NAME
  local name="$1" val="${!1:-}"
  [[ -n "$val" ]] || qp2_refuse "$name is not set (use the Felix-built main venv, e.g. $name=<venv>/bin/...)"
  [[ -x "$val" ]] || qp2_refuse "$name=$val is not an executable file"
}

qp2_require_model_dir() {  # LABEL DIR
  [[ -f "$2/config.json" ]] || qp2_refuse "$1: no config.json in $2 (set QWEN_STAGE or the per-model override)"
}

# Refuse if either GPU has more than QP2_GPU_IDLE_MIB MiB in use.
qp2_check_gpus_idle() {
  command -v nvidia-smi >/dev/null || qp2_refuse "nvidia-smi not found"
  local out idx used n=0 busy=""
  out="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)" \
    || qp2_refuse "nvidia-smi failed"
  while IFS=', ' read -r idx used; do
    [[ -n "$idx" ]] || continue
    n=$((n + 1))
    if (( used > QP2_GPU_IDLE_MIB )); then busy+=" GPU$idx=${used}MiB"; fi
  done <<<"$out"
  (( n >= 2 )) || qp2_refuse "need 2 GPUs, nvidia-smi shows $n"
  [[ -z "$busy" ]] || qp2_refuse "GPU memory in use:$busy (> ${QP2_GPU_IDLE_MIB} MiB). Is prod (vllm/Flint) still up, or LMCache not yet idle? Wait ~2 min after stopping Flint."
}

# Wait up to SECS for both GPUs to drop to idle; return 1 on timeout.
qp2_wait_gpus_idle() {  # SECS
  local deadline=$((SECONDS + $1))
  while (( SECONDS < deadline )); do
    ( qp2_check_gpus_idle ) 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

qp2_port_in_use() {  # PORT
  if command -v ss >/dev/null && [[ -n "$(ss -Htln "sport = :$1" 2>/dev/null)" ]]; then
    return 0
  fi
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}
