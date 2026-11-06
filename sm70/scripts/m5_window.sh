#!/usr/bin/env bash
# M5 window runner (Bea, c4130, inside the downtime window, Flint stopped).
# Runs every M5 GPU gate (T1-T7), each test node in its own pytest process
# (so one engine's GPU state cannot leak into the next), tees every log, and
# prints the per-prompt verdict lines (T2/T4/T5/T7) plus PASS/FAIL per node.
# A node PASSes only if pytest exits 0 AND its junit XML shows >0 tests and
# no failures, errors or skips (a skip, e.g. a missing env var, is a FAIL).
# One exception (Quinn, Nov 7): exact T4b runs only on TRITON attention
# (VLLM_SM70_FLASH_V100=0); under Flash-V100 every t4b case skips with
# "covered by T4e", reported SKIPPED (by design), and T4e gates replay.
#
# Required env: VLLM_PY (venv python with pytest), FLINT_SM70_CHECKPOINT,
# FLINT_SM70_GOLDEN (T5), and for the teacher-forced gates (T4a/T4c/T4d/T4e/T7,
# from m5_teacher_forced.py, see docs/m5-calibration-nov7.md):
#   FLINT_SM70_T4_SEQS_DIR      setK.json, >= 3 prompt sets (gen, on ours)
#   FLINT_SM70_T4_ENVELOPE      the fork's envelope JSON (envelope; built on or
#                               after Nov 7, with over_1_0 and eager_vs_graphsonly)
#   FLINT_SM70_FORK_SCORES_DIR  the fork's setK/{eager,eager2,graphs,unfused,graphsonly}.json
# Needed by T5 only if a golden-stable prompt diverges early at a near-tie:
#   FLINT_SM70_T5_FORK_GOLDEN_SCORES  the fork's forced decode of the golden
#                               sequences (seqs-from-golden + score --mode graphs)
# FLINT_SM70_OURS_NOISE defaults to $OUT/ours_eager_vs_eager.json: T4c (two
# separate eager engine instances) writes it and T4a/T4d/T4e/T7 print it on
# every gate line, so t4c runs first. T4c must PASS (with the tuning-off
# setting below) before T4a/T4d/T4e/T7 count; if it fails they are reported
# NOT COUNTED.
# Model loads per node: t4c 2, t4a 2, t4b 4 (TRITON only; 0 under
# Flash-V100), t4e 2, t4d 1, t5 1 (+1 to score near-tie golden sequences),
# t7 2, t3 1, t6 1 (t1/t2: tiny model / kernels only). Default (Flash-V100):
# 12 loads, 13 with a T5 near-tie.
# Optional: OUT (log dir), M5_NODES (space-separated subset of the names
# below, default: all, in this order), M5_MUTATION (mutation check only; see
# m5_mutation_check.sh).
set -uo pipefail

# Deterministic kernel choice (Quinn/Astro, Nov 6): TurboMind's small-shape
# autotune times kernels at engine start, so each instance may pick different
# kernels (Bea: 7/8 prompts split between two eager instances; identical with
# tuning off). Until Kai's persisted tuning table lands, every M5 gate runs
# with it off, on both stacks. M5_ALLOW_TUNING=1 opts out (diagnostics only).
if [[ "${M5_ALLOW_TUNING:-0}" != 1 ]]; then
  for q in NVFP4 FP8 AWQ MXFP4; do export "VLLM_SM70_${q}_TUNE_SMALL_SHAPES=0"; done
fi
echo "tuning: $(env | grep -E '^VLLM_SM70_[A-Z0-9]+_TUNE_SMALL_SHAPES=' | sort | tr '\n' ' ')"

: "${VLLM_PY:?set VLLM_PY to the venv python}"
: "${FLINT_SM70_CHECKPOINT:?set FLINT_SM70_CHECKPOINT}"
: "${FLINT_SM70_GOLDEN:?set FLINT_SM70_GOLDEN}"
: "${FLINT_SM70_T4_SEQS_DIR:?set FLINT_SM70_T4_SEQS_DIR (dir of setK.json)}"
: "${FLINT_SM70_T4_ENVELOPE:?set FLINT_SM70_T4_ENVELOPE (fork envelope JSON)}"
: "${FLINT_SM70_FORK_SCORES_DIR:?set FLINT_SM70_FORK_SCORES_DIR (fork scores)}"
for d in "$FLINT_SM70_T4_SEQS_DIR" "$FLINT_SM70_FORK_SCORES_DIR"; do
  [[ -d "$d" ]] || { echo "not a directory: $d"; exit 2; }
done
[[ -f "$FLINT_SM70_T4_ENVELOPE" ]] || { echo "missing $FLINT_SM70_T4_ENVELOPE"; exit 2; }
OUT="${OUT:-$HOME/vllm-sm70-work/bea/m5-window/$(date +%Y%m%dT%H%M)}"
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 2; }
export FLINT_SM70_OURS_NOISE="${FLINT_SM70_OURS_NOISE:-$OUT/ours_eager_vs_eager.json}"
F=sm70/tests/models/test_m5_flint_sm70.py
[[ -f "$F" ]] || { echo "run from the repo root ($F not found)"; exit 2; }

declare -A NODE=(
  [t1]="$F::test_t1_tiny_random_forward"
  [t6]="$F::test_t6_fp16_headroom_8k"
  [t4a]="$F::test_t4a_teacher_forced_eager_vs_graphs"
  [t4b]="$F::test_t4b_graph_replay_exact"
  [t4e]="$F::test_t4e_forced_decode_eager_vs_graphs_only"
  [t4c]="$F::test_t4c_eager_repeatable_across_engine_instances"
  [t4d]="$F::test_t4d_graphs_vs_fork_graphs"
  [t5]="$F::test_t5_golden_tokens_vs_fork"
  [t7]="$F::test_t7_unfused_path_matches"
  [t2]="$F::test_t2_triton_kernel_bitwise_vs_reference"
  [t3]="$F::test_t3_checkpoint_loads"
)
read -r -a ORDER <<< "${M5_NODES:-t4c t4a t4b t4e t4d t5 t7 t2 t3 t1 t6}"

# Nodes whose skip is by design when every skip message carries the marker.
declare -A SKIP_OK=([t4b]="covered by T4e")

check_junit() {  # XML [skip marker]; exit 0 pass, 5 all skipped by design, 1 else
  "$VLLM_PY" - "$1" "${2:-}" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
marker = sys.argv[2]
s = r if r.tag == "testsuite" else r.find("testsuite")
n, f, e, k = (int(s.get(a, 0)) for a in ("tests", "failures", "errors", "skipped"))
print(f"junit: tests={n} failures={f} errors={e} skipped={k}")
skips = []
for c in s.iter("testcase"):
    for x in c:
        if x.tag in ("failure", "error", "skipped"):
            print(f"  {x.tag}: {c.get('name')}: {(x.get('message') or '')[:300]}")
        if x.tag == "skipped":
            skips.append(x.get("message") or "")
if n > 0 and f == e == k == 0:
    sys.exit(0)
if marker and n > 0 and f == e == 0 and k == n == len(skips) and all(marker in m for m in skips):
    print(f"junit: every case skipped by design ({marker})")
    sys.exit(5)
sys.exit(1)
PY
}

used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
if used_mib | awk '$1 > 1024 {busy=1} END {exit !busy}'; then
  echo "a GPU has >1 GiB in use (is prod still up?):"; used_mib; exit 2
fi

declare -A RC SKIPPED
for name in "${ORDER[@]}"; do
  node="${NODE[$name]:-}"
  [[ -n "$node" ]] || { echo "unknown node name: $name"; exit 2; }
  echo "== $name: $node"
  "$VLLM_PY" -m pytest -q -p no:cacheprovider -s -rA \
    --junitxml="$OUT/$name.junit.xml" "$node" 2>&1 | tee "$OUT/$name.log"
  RC[$name]=${PIPESTATUS[0]}
  if [[ -f "$OUT/$name.junit.xml" ]]; then
    check_junit "$OUT/$name.junit.xml" "${SKIP_OK[$name]:-}" | tee -a "$OUT/$name.log"
    jrc=${PIPESTATUS[0]}
    if [[ $jrc -eq 5 && ${RC[$name]} -eq 0 ]]; then
      SKIPPED[$name]=1
    elif [[ $jrc -ne 0 ]]; then
      [[ ${RC[$name]} -ne 0 ]] || RC[$name]=3
    fi
  else
    echo "no junit XML for $name" | tee -a "$OUT/$name.log"
    [[ ${RC[$name]} -ne 0 ]] || RC[$name]=4
  fi
done

echo
echo "== verdicts ($OUT)"
grep -hE '^(T2 n=|T4[abcde] |T5 |T7 |ours_eager_vs_eager|M5_MUTATION |junit: )' "$OUT"/*.log || true
echo
echo "== summary"
rc=0
t4c_ok=1
[[ -v "RC[t4c]" && ${RC[t4c]} -ne 0 ]] && t4c_ok=0
for name in "${ORDER[@]}"; do
  if [[ ${RC[$name]} -ne 0 ]]; then
    echo "$name FAIL (rc ${RC[$name]})"; rc=1
  elif [[ -v "SKIPPED[$name]" ]]; then
    echo "$name SKIPPED by design (${SKIP_OK[$name]}; exact T4b needs VLLM_SM70_FLASH_V100=0)"
  elif [[ $t4c_ok -eq 0 && " t4a t4d t4e t7 " == *" $name "* ]]; then
    # Teacher-forced gates only count when eager is reproducible (T4c).
    echo "$name NOT COUNTED (t4c failed: engine instances not reproducible)"; rc=1
  else
    echo "$name PASS"
  fi
done
exit "$rc"
