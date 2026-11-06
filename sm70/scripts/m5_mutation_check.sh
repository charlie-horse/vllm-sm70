#!/usr/bin/env bash
# M5 mutation check (Quinn): T4d and T7 must FAIL on planted bugs.
# Runs each GPU gate node with each mutation (one pytest process per run,
# M5_MUTATION set, sm70/tests/models/m5_mutations first on PYTHONPATH so its
# sitecustomize patches the fused q/k-norm+RoPE+KV op in every process) and
# PASSes only if EVERY mutated run fails the GATE:
#   - pytest exit != 0, junit: tests > 0, failures > 0, errors == 0,
#     skipped == 0 (a crash or a skip does not count as catching the bug);
#   - the failure message is the gate's ("outside ... the fork");
#   - the mutation demonstrably took effect: hook mutations print
#     "M5_MUTATION <name> installed in pid N" from >= 3 processes (pytest +
#     2 TP workers); rope_theta_1p01 prints its hf_overrides line.
# Mutations (sm70/tests/models/m5_mutations/m5_mutation.py):
#   rope_pos_decode  positions + 1 on <= 8-token calls of the fused op
#   kv_slot_decode   slot_mapping + 1 on <= 8-token calls of the fused op
#   rope_theta_1p01  rope_theta x 1.01 via hf_overrides on the mutated side
# The unmutated T4d/T7 must PASS first (m5_window.sh, or M5_BASELINE=1 here).
#
# Required env: as m5_window.sh (VLLM_PY, FLINT_SM70_CHECKPOINT,
# FLINT_SM70_T4_SEQS_DIR, FLINT_SM70_T4_ENVELOPE, FLINT_SM70_FORK_SCORES_DIR).
# Optional: OUT, M5_MUTATIONS (subset, default all three), M5_MUTATION_NODES
# (default "t4d t7"), M5_BASELINE=1 (also run both nodes unmutated, must PASS).
set -uo pipefail

: "${VLLM_PY:?set VLLM_PY to the venv python}"
: "${FLINT_SM70_CHECKPOINT:?set FLINT_SM70_CHECKPOINT}"
: "${FLINT_SM70_T4_SEQS_DIR:?set FLINT_SM70_T4_SEQS_DIR}"
: "${FLINT_SM70_T4_ENVELOPE:?set FLINT_SM70_T4_ENVELOPE}"
: "${FLINT_SM70_FORK_SCORES_DIR:?set FLINT_SM70_FORK_SCORES_DIR}"
F=sm70/tests/models/test_m5_flint_sm70.py
MUT_DIR=sm70/tests/models/m5_mutations
[[ -f "$F" && -f "$MUT_DIR/sitecustomize.py" ]] || { echo "run from the repo root"; exit 2; }
MUT_DIR="$(cd "$MUT_DIR" && pwd)"
OUT="${OUT:-$HOME/vllm-sm70-work/bea/m5-mutation/$(date +%Y%m%dT%H%M)}"
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 2; }

declare -A NODE=(
  [t4d]="$F::test_t4d_graphs_vs_fork_graphs"
  [t7]="$F::test_t7_unfused_path_matches"
)
read -r -a MUTS <<< "${M5_MUTATIONS:-rope_pos_decode kv_slot_decode rope_theta_1p01}"
read -r -a NODES <<< "${M5_MUTATION_NODES:-t4d t7}"

used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
if used_mib | awk '$1 > 1024 {busy=1} END {exit !busy}'; then
  echo "a GPU has >1 GiB in use (is prod still up?):"; used_mib; exit 2
fi

junit_counts() {  # XML -> "tests failures errors skipped first-failure-message"
  "$VLLM_PY" - "$1" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
s = r if r.tag == "testsuite" else r.find("testsuite")
n, f, e, k = (int(s.get(a, 0)) for a in ("tests", "failures", "errors", "skipped"))
msg = ""
for c in s.iter("testcase"):
    for x in c:
        if x.tag == "failure" and not msg:
            msg = " ".join((x.get("message") or "").split())[:400]
print(n, f, e, k, msg)
PY
}

run_node() {  # name mutation(or "none") -> log path; sets RUN_RC
  local name=$1 mut=$2 tag="$1.$2"
  local -a envs=()
  if [[ $mut != none ]]; then
    envs=(M5_MUTATION="$mut" PYTHONPATH="$MUT_DIR${PYTHONPATH:+:$PYTHONPATH}")
  fi
  echo "== $tag: ${NODE[$name]}"
  env "${envs[@]}" "$VLLM_PY" -m pytest -q -p no:cacheprovider -s -rA \
    --junitxml="$OUT/$tag.junit.xml" "${NODE[$name]}" 2>&1 | tee "$OUT/$tag.log"
  RUN_RC=${PIPESTATUS[0]}
}

declare -A VERDICT
rc=0
if [[ "${M5_BASELINE:-0}" == 1 ]]; then
  for name in "${NODES[@]}"; do
    run_node "$name" none
    read -r n f e k _ < <(junit_counts "$OUT/$name.none.junit.xml" 2>/dev/null || echo "0 0 0 0")
    if [[ $RUN_RC -eq 0 && $n -gt 0 && $f -eq 0 && $e -eq 0 && $k -eq 0 ]]; then
      VERDICT[$name.none]="baseline PASS"
    else
      VERDICT[$name.none]="baseline FAIL (rc $RUN_RC, tests=$n failures=$f errors=$e skipped=$k): mutation results are meaningless"
      rc=1
    fi
  done
fi

for mut in "${MUTS[@]}"; do
  case $mut in rope_pos_decode|kv_slot_decode|rope_theta_1p01) ;;
    *) echo "unknown mutation: $mut"; exit 2 ;; esac
  for name in "${NODES[@]}"; do
    [[ -n "${NODE[$name]:-}" ]] || { echo "unknown node name: $name"; exit 2; }
    run_node "$name" "$mut"
    log="$OUT/$name.$mut.log" xml="$OUT/$name.$mut.junit.xml"
    if [[ ! -f $xml ]]; then
      VERDICT[$name.$mut]="NOT CAUGHT (no junit XML, rc $RUN_RC)"; rc=1; continue
    fi
    read -r n f e k msg < <(junit_counts "$xml")
    if [[ $mut == rope_theta_1p01 ]]; then
      applied=$(grep -c '^M5_MUTATION rope_theta_1p01: hf_overrides' "$log")
      need=1
    else
      applied=$(grep -oE "M5_MUTATION $mut installed in pid [0-9]+" "$log" | sort -u | wc -l)
      need=3
    fi
    if [[ $applied -lt $need ]]; then
      VERDICT[$name.$mut]="NOT APPLIED (marker seen $applied, need >= $need)"; rc=1
    elif [[ $RUN_RC -ne 0 && $n -gt 0 && $f -gt 0 && $e -eq 0 && $k -eq 0 \
            && $msg == *outside*fork* ]]; then
      VERDICT[$name.$mut]="CAUGHT: $msg"
    else
      VERDICT[$name.$mut]="NOT CAUGHT (rc $RUN_RC, tests=$n failures=$f errors=$e skipped=$k): $msg"
      rc=1
    fi
  done
done

echo
echo "== gate lines ($OUT)"
grep -hE '^(T4d |T7 |M5_MUTATION .*hf_overrides|junit: )' "$OUT"/*.log || true
echo
echo "== mutation check"
for key in "${!VERDICT[@]}"; do echo "$key: ${VERDICT[$key]}"; done | sort
if [[ $rc -eq 0 ]]; then echo "mutation check PASS"; else echo "mutation check FAIL"; fi
exit "$rc"
