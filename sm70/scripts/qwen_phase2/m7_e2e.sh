#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# M7 DFlash2 GPU tests on c4130 (header of sm70/tests/models/test_m7_dflash_sm70.py).
#
#   m7_e2e.sh
#
# Runs, by exact node ID (no -k):
#   1. test_triton_matches_reference_gpu   (all 3 params, GPU0 only)
#   2. test_e2e_real_draft_gpu             (TP2, real Qwen3.8-27B-FP8 + DFlash2)
# A skipped test counts as a failure (e.g. DFLASH2_E2E_* not picked up).
#
# The M7 A/B (VLLM_SM70_DFLASH2_EMULATION=0) is NOT run here: the test file's
# autouse fixture deletes that variable and the e2e test asserts m7.register()
# is True, which is False with the switch off. Use qwen_matrix.sh 4 vs 4-noemu.
#
# Env: VLLM_PY (required; needs pytest), QWEN_STAGE, OUT,
#      DFLASH2_E2E_TARGET (default $QWEN_STAGE/Qwen3.8-27B-FP8),
#      DFLASH2_E2E_DRAFT  (default $QWEN_STAGE/Qwen3.8-27B-DFlash2).
# Logs: $OUT/m7/. Exit: 0 pass, 1 fail/skip, 2 refused to start.
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TEST_FILE=sm70/tests/models/test_m7_dflash_sm70.py
NODE_TRITON="$TEST_FILE::test_triton_matches_reference_gpu"
NODE_E2E="$TEST_FILE::test_e2e_real_draft_gpu"

export DFLASH2_E2E_TARGET="${DFLASH2_E2E_TARGET:-$QWEN_STAGE/Qwen3.8-27B-FP8}"
export DFLASH2_E2E_DRAFT="${DFLASH2_E2E_DRAFT:-$QWEN_STAGE/Qwen3.8-27B-DFlash2}"

qp2_require_exec VLLM_PY
"$VLLM_PY" -c 'import pytest' 2>/dev/null || qp2_refuse "pytest is not importable from VLLM_PY=$VLLM_PY"
qp2_require_model_dir "DFLASH2_E2E_TARGET" "$DFLASH2_E2E_TARGET"
qp2_require_model_dir "DFLASH2_E2E_DRAFT" "$DFLASH2_E2E_DRAFT"
[[ -f "$QP2_REPO/$TEST_FILE" ]] || qp2_refuse "$TEST_FILE not found under $QP2_REPO"
qp2_check_gpus_idle

LOG="$OUT/m7"
mkdir -p "$LOG"
cd "$QP2_REPO"
qp2_log "OUT=$LOG target=$DFLASH2_E2E_TARGET draft=$DFLASH2_E2E_DRAFT"

# Every selected test must pass; skips and empty selections fail.
check_junit() {  # XML
  "$VLLM_PY" - "$1" <<'EOF'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
s = r if r.tag == "testsuite" else r.find("testsuite")
n, f, e, k = (int(s.get(a, 0)) for a in ("tests", "failures", "errors", "skipped"))
print(f"junit: tests={n} failures={f} errors={e} skipped={k}")
for c in s.iter("testcase"):
    for x in c:
        if x.tag in ("failure", "error", "skipped"):
            print(f"  {x.tag}: {c.get('name')}: {(x.get('message') or '')[:300]}")
sys.exit(0 if n > 0 and f == e == k == 0 else 1)
EOF
}

run_node() {  # NAME CUDA_VISIBLE_DEVICES NODE
  local name="$1" cvd="$2" node="$3" t0=$SECONDS rc=0
  qp2_log "$name: $node (CUDA_VISIBLE_DEVICES=$cvd)"
  CUDA_VISIBLE_DEVICES="$cvd" "$VLLM_PY" -m pytest -q -p no:cacheprovider -rA -s \
    --junitxml="$LOG/$name.junit.xml" "$node" >"$LOG/$name.log" 2>&1 || rc=$?
  tail -n 15 "$LOG/$name.log"
  check_junit "$LOG/$name.junit.xml" || rc=1
  qp2_log "$name: $([[ $rc == 0 ]] && echo PASS || echo FAIL) in $((SECONDS - t0))s (log $LOG/$name.log)"
  return "$rc"
}

rc=0
run_node triton 0 "$NODE_TRITON" || rc=1
qp2_wait_gpus_idle 120 || { qp2_log "GPU0 not idle after the triton test"; exit 1; }
run_node e2e 0,1 "$NODE_E2E" || rc=1

# The e2e test asserts accepted > 0 but does not print acceptance numbers.
grep -E "spec_decode|[Aa]cceptance|[Aa]ccepted" "$LOG/e2e.log" | tail -n 5 || true
echo
echo "M7 $([[ $rc == 0 ]] && echo PASS || echo FAIL)  (logs: $LOG)"
exit "$rc"
