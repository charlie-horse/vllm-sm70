#!/usr/bin/env bash
# M5 window runner (Bea, c4130, inside the downtime window, Flint stopped).
# Runs the M5 GPU gates, each test node in its own pytest process (so one
# engine's GPU state cannot leak into the next), tees every log, and prints
# the per-prompt verdict lines (T2/T4/T5/T7) plus PASS/FAIL per node.
#
# Required env: VLLM_PY (venv python with pytest), FLINT_SM70_CHECKPOINT,
# FLINT_SM70_GOLDEN. Optional: OUT (log dir), M5_NODES (space-separated
# subset of the names below, default: all, in this order).
set -uo pipefail

: "${VLLM_PY:?set VLLM_PY to the venv python}"
: "${FLINT_SM70_CHECKPOINT:?set FLINT_SM70_CHECKPOINT}"
: "${FLINT_SM70_GOLDEN:?set FLINT_SM70_GOLDEN}"
OUT="${OUT:-$HOME/vllm-sm70-work/bea/m5-window/$(date +%Y%m%dT%H%M)}"
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 2; }
F=sm70/tests/models/test_m5_flint_sm70.py
[[ -f "$F" ]] || { echo "run from the repo root ($F not found)"; exit 2; }

declare -A NODE=(
  [t4]="$F::test_t4_compiled_graphs_match_eager"
  [t5]="$F::test_t5_golden_tokens_vs_fork"
  [t7]="$F::test_t7_unfused_path_matches"
  [t2]="$F::test_t2_triton_kernel_bitwise_vs_reference"
  [t3]="$F::test_t3_checkpoint_loads"
)
read -r -a ORDER <<< "${M5_NODES:-t4 t5 t7 t2 t3}"

used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
if used_mib | awk '$1 > 1024 {busy=1} END {exit !busy}'; then
  echo "a GPU has >1 GiB in use (is prod still up?):"; used_mib; exit 2
fi

declare -A RC
for name in "${ORDER[@]}"; do
  node="${NODE[$name]:-}"
  [[ -n "$node" ]] || { echo "unknown node name: $name"; exit 2; }
  echo "== $name: $node"
  "$VLLM_PY" -m pytest -q -p no:cacheprovider -s -rA "$node" 2>&1 | tee "$OUT/$name.log"
  RC[$name]=${PIPESTATUS[0]}
done

echo
echo "== verdicts ($OUT)"
grep -hE '^(T2 n=|T4 |T5 |T7 )' "$OUT"/*.log || true
echo
echo "== summary"
rc=0
for name in "${ORDER[@]}"; do
  if [[ ${RC[$name]} -eq 0 ]]; then echo "$name PASS"; else echo "$name FAIL (rc ${RC[$name]})"; rc=1; fi
done
exit "$rc"
