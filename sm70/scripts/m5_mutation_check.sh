#!/usr/bin/env bash
# Mutation check: T4d and T7 must FAIL on planted bugs. Runs each gate node
# once per mutation (sm70/tests/models/m5_mutations first on PYTHONPATH, so
# its sitecustomize patches the fused q/k-norm+RoPE+KV op in every process).
# A required run counts as caught only if pytest fails with failures > 0,
# errors == skipped == 0, the gate's own failure message, and the mutation
# printed "installed" (from every process that runs the op) and "fired".
# Required catches (MUST below): rope_pos_decode and kv_slot_decode by t4d
# and t7, moe_decode_scale by t4d (both T7 arms run the same MoE).
# moe_decode_scale_half and rope_theta_1p01 are INFO only: reported, never
# required (moe_decode_scale_half must still apply and fire).
# The unmutated T4d/T7 must PASS first (m5_window.sh, or M5_BASELINE=1 here).
# Required env: as m5_window.sh. Optional: OUT, M5_MUTATIONS (subset),
# M5_MUTATION_NODES (default "t4d t7"), M5_BASELINE=1.
set -uo pipefail

# TurboMind's small-shape autotune picks kernels per engine start, so engine
# instances can differ; the gates run with it off unless M5_ALLOW_TUNING=1.
if [[ "${M5_ALLOW_TUNING:-0}" != 1 ]]; then
  for q in NVFP4 FP8 AWQ MXFP4; do export "VLLM_SM70_${q}_TUNE_SMALL_SHAPES=0"; done
fi
# Tuning on is allowed only with a persisted TurboMind LUT in strict mode, so
# a missing or stale LUT fails instead of re-tuning per engine.
if [[ "${M5_ALLOW_TUNING:-0}" == 1 && -n "${VLLM_SM70_GEMM_LUT_PATH:-}" \
      && "${VLLM_SM70_GEMM_LUT_STRICT:-}" != 1 ]]; then
  echo "M5_ALLOW_TUNING=1 with VLLM_SM70_GEMM_LUT_PATH set needs VLLM_SM70_GEMM_LUT_STRICT=1"
  exit 2
fi
echo "tuning: $(env | grep -E '^VLLM_SM70_[A-Z0-9]+_TUNE_SMALL_SHAPES=' | sort | tr '\n' ' ')LUT: path=${VLLM_SM70_GEMM_LUT_PATH:-<unset>} strict=${VLLM_SM70_GEMM_LUT_STRICT:-<unset>}"

: "${VLLM_PY:?set VLLM_PY to the venv python}"
: "${FLINT_SM70_CHECKPOINT:?set FLINT_SM70_CHECKPOINT}"
: "${FLINT_SM70_T4_SEQS_DIR:?set FLINT_SM70_T4_SEQS_DIR}"
: "${FLINT_SM70_T4_ENVELOPE:?set FLINT_SM70_T4_ENVELOPE}"
: "${FLINT_SM70_FORK_SCORES_DIR:?set FLINT_SM70_FORK_SCORES_DIR}"
F=sm70/tests/models/test_m5_flint_sm70.py
MUT_DIR=sm70/tests/models/m5_mutations
[[ -f "$F" && -f "$MUT_DIR/sitecustomize.py" ]] || { echo "run from the repo root"; exit 2; }
MUT_DIR="$(cd "$MUT_DIR" && pwd)"
OUT="${OUT:-./out/m5-mutation/$(date +%Y%m%dT%H%M)}"
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 2; }

declare -A NODE=(
  [t4d]="$F::test_t4d_graphs_vs_fork_graphs"
  [t7]="$F::test_t7_unfused_path_matches"
)
declare -A MUST=(
  [rope_pos_decode]="t4d t7"
  [kv_slot_decode]="t4d t7"
  [moe_decode_scale]="t4d"
  [moe_decode_scale_half]=""
  [rope_theta_1p01]=""
)
declare -A INFO_ONLY=([rope_theta_1p01]="t4d t7" [moe_decode_scale_half]="t4d")
read -r -a MUTS <<< "${M5_MUTATIONS:-rope_pos_decode kv_slot_decode moe_decode_scale moe_decode_scale_half rope_theta_1p01}"
read -r -a NODES <<< "${M5_MUTATION_NODES:-t4d t7}"

used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
if used_mib | awk '$1 > 1024 {busy=1} END {exit !busy}'; then
  echo "a GPU has >1 GiB in use (is a server still running?):"; used_mib; exit 2
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
    elif [[ $mut == moe_decode_scale* ]]; then
      # the two TP workers run the MoE; the pytest process may not import it
      applied=$(grep -oE "M5_MUTATION $mut installed in pid [0-9]+" "$log" | sort -u | wc -l)
      need=2
    else
      applied=$(grep -oE "M5_MUTATION $mut installed in pid [0-9]+" "$log" | sort -u | wc -l)
      need=3
    fi
    tag=""; [[ $info -eq 1 ]] && tag="INFO "
    # Fired, not just installed: the wrapper prints "fired" the first time it
    # changes a call in a process; the TP workers run the op.
    fired=1
    if [[ $mut != rope_theta_1p01 ]]; then
      fired=$(grep -oE "M5_MUTATION $mut fired in pid [0-9]+" "$log" | sort -u | wc -l)
    fi
    if [[ $applied -lt $need ]]; then
      VERDICT[$name.$mut]="${tag}NOT APPLIED (marker seen $applied, need >= $need)"
    elif [[ $fired -lt 1 ]]; then
      extra=""
      [[ $mut == moe_decode_scale* ]] && ! grep -q "K12 M<=2 decode admitted" "$log" \
        && extra=" (no 'K12 M<=2 decode admitted' line: decode not admitted in this build/config)"
      VERDICT[$name.$mut]="${tag}NOT FIRED (installed but never changed a call)$extra"
    elif [[ $RUN_RC -ne 0 && $n -gt 0 && $f -gt 0 && $e -eq 0 && $k -eq 0 \
            && $msg == *outside*fork* ]]; then
      VERDICT[$name.$mut]="${tag}CAUGHT: $msg"; continue
    else
      VERDICT[$name.$mut]="${tag}NOT CAUGHT (rc $RUN_RC, tests=$n failures=$f errors=$e skipped=$k): $msg"
    fi
    # The INFO probe moe_decode_scale_half need not be caught, but must still
    # be applied and fire: evidence that the M<=2 MoE decode path runs.
    v=${VERDICT[$name.$mut]}
    if [[ $mut == moe_decode_scale_half && ( $v == *"NOT APPLIED"* || $v == *"NOT FIRED"* ) ]]; then
      rc=1
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
