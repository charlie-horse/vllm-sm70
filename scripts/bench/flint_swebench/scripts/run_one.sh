#!/usr/bin/env bash
# Usage: run_one.sh INSTANCE_ID ATTEMPT  -> agent run + official eval + image cleanup
set -u
ID=$1; A=$2; ROOT=/home/user/swe; OUT=$ROOT/runs/att$A
mkdir -p $OUT/agent $OUT/eval $OUT/log
REPORT=$OUT/eval/done_$ID
[ -f "$REPORT" ] && exit 0
# keep MemAvailable / disk sane
cd $ROOT
t0=$(date +%s)
PYTHONPATH=$ROOT/adapter MSWEA_COST_TRACKING=ignore_errors timeout 6h \
  .venv/bin/mini-extra swebench --subset verified --split test --filter "^${ID}\$" \
  -c swebench.yaml -c $ROOT/flint.yaml -w 1 -o $OUT/agent/$ID > $OUT/log/$ID.agent.log 2>&1
t1=$(date +%s)
cd $OUT/eval
if [ -s $OUT/agent/$ID/preds.json ]; then
  $ROOT/.venv/bin/python -m swebench.harness.run_evaluation -d SWE-bench/SWE-bench_Verified \
    -s test -i $ID -p $OUT/agent/$ID/preds.json --max_workers 1 -id att${A}_$ID \
    --timeout 1800 > $OUT/log/$ID.eval.log 2>&1
fi
t2=$(date +%s)
echo "{\"id\":\"$ID\",\"attempt\":$A,\"agent_s\":$((t1-t0)),\"eval_s\":$((t2-t1))}" > $REPORT
# free disk: drop this instance's image
img=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i "$(echo $ID | sed 's/__/_1776_/')" || true)
[ -n "$img" ] && docker rmi -f $img >/dev/null 2>&1
docker image prune -f >/dev/null 2>&1
