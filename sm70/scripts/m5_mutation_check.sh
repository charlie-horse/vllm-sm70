#!/usr/bin/env bash
# M5 mutation check (Quinn): T4d and T7 must FAIL on planted bugs.
# Runs each GPU gate node with each mutation (one pytest process per run,
# M5_MUTATION set, sm70/tests/models/m5_mutations first on PYTHONPATH so its
# sitecustomize patches the fused q/k-norm+RoPE+KV op in every process) and
# PASSes only if every REQUIRED mutated run fails the GATE:
#   - pytest exit != 0, junit: tests > 0, failures > 0, errors == 0,
#     skipped == 0 (a crash or a skip does not count as catching the bug);
#   - the failure message is the gate's ("outside ... the fork");
#   - the mutation demonstrably took effect: hook mutations print
#     "M5_MUTATION <name> installed in pid N" from >= 3 processes (pytest +
#     2 TP workers; 2 for moe_decode_scale) AND "M5_MUTATION <name> fired in
#     pid N" at least once (it actually changed a call); rope_theta_1p01
#     prints its hf_overrides line.
# Mutations (sm70/tests/models/m5_mutations/m5_mutation.py) and the nodes
# that MUST catch them (Quinn/Astro, Nov 6):
#   rope_pos_decode  positions + 1 on <= 8-token fused-op calls   -> t4d t7
#   kv_slot_decode   slot_mapping + 1 on <= 8-token fused-op calls -> t4d t7
#   moe_decode_scale K12 M<=2 MoE decode output x 1.01            -> t4d
#                    (both T7 arms run the same MoE, so T7 cannot see it)
#   rope_theta_1p01  rope_theta x 1.01 via hf_overrides            -> INFO only
#                    (too weak at ~70-token positions; reported, not gated)
# The unmutated T4d/T7 must PASS first (m5_window.sh, or M5_BASELINE=1 here).
#
# Required env: as m5_window.sh (VLLM_PY, FLINT_SM70_CHECKPOINT,
# FLINT_SM70_T4_SEQS_DIR, FLINT_SM70_T4_ENVELOPE, FLINT_SM70_FORK_SCORES_DIR).
# Optional: OUT, M5_MUTATIONS (subset, default all three), M5_MUTATION_NODES
# (default "t4d t7"), M5_BASELINE=1 (also run both nodes unmutated, must PASS).
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
declare -A MUST=(
  [rope_pos_decode]="t4d t7"
  [kv_slot_decode]="t4d t7"
  [moe_decode_scale]="t4d"
  [rope_theta_1p01]=""
)
declare -A INFO_ONLY=([rope_theta_1p01]="t4d t7")
read -r -a MUTS <<< "${M5_MUTATIONS:-rope_pos_decode kv_slot_decode moe_decode_scale rope_theta_1p01}"
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
  [[ -v "MUST[$mut]" ]] || { echo "unknown mutation: $mut"; exit 2; }
  read -r -a targets <<< "${MUST[$mut]} ${INFO_ONLY[$mut]:-}"
  for name in "${NODES[@]}"; do
    [[ -n "${NODE[$name]:-}" ]] || { echo "unknown node name: $name"; exit 2; }
    [[ " ${targets[*]} " == *" $name "* ]] || continue
    info=0
    [[ " ${MUST[$mut]} " == *" $name "* ]] || info=1
    run_node "$name" "$mut"
    log="$OUT/$name.$mut.log" xml="$OUT/$name.$mut.junit.xml"
    if [[ ! -f $xml ]]; then
      VERDICT[$name.$mut]="NOT CAUGHT (no junit XML, rc $RUN_RC)"; rc=1; continue
    fi
    read -r n f e k msg < <(junit_counts "$xml")
    if [[ $mut == rope_theta_1p01 ]]; then
      applied=$(grep -c '^M5_MUTATION rope_theta_1p01: hf_overrides' "$log")
      need=1
    elif [[ $mut == moe_decode_scale ]]; then
      # the two TP workers run the MoE; the pytest process may not import it
      applied=$(grep -oE "M5_MUTATION $mut installed in pid [0-9]+" "$log" | sort -u | wc -l)
      need=2
    else
      applied=$(grep -oE "M5_MUTATION $mut installed in pid [0-9]+" "$log" | sort -u | wc -l)
      need=3
    fi
    tag=""; [[ $info -eq 1 ]] && tag="INFO "
    # Fired, not just installed (Quinn): the wrapper prints "fired" the first
    # time it changes a call in a process; the TP workers run the op.
    fired=1
    if [[ $mut != rope_theta_1p01 ]]; then
      fired=$(grep -oE "M5_MUTATION $mut fired in pid [0-9]+" "$log" | sort -u | wc -l)
    fi
    if [[ $applied -lt $need ]]; then
      VERDICT[$name.$mut]="${tag}NOT APPLIED (marker seen $applied, need >= $need)"
    elif [[ $fired -lt 1 ]]; then
      extra=""
      [[ $mut == moe_decode_scale ]] && ! grep -q "K12 M<=2 decode admitted" "$log" \
        && extra=" (no 'K12 M<=2 decode admitted' line: decode not admitted in this build/config)"
      VERDICT[$name.$mut]="${tag}NOT FIRED (installed but never changed a call)$extra"
    elif [[ $RUN_RC -ne 0 && $n -gt 0 && $f -gt 0 && $e -eq 0 && $k -eq 0 \
            && $msg == *outside*fork* ]]; then
      VERDICT[$name.$mut]="${tag}CAUGHT: $msg"; continue
    else
      VERDICT[$name.$mut]="${tag}NOT CAUGHT (rc $RUN_RC, tests=$n failures=$f errors=$e skipped=$k): $msg"
    fi
    [[ $info -eq 1 ]] || rc=1
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
