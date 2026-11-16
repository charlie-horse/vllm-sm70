#!/usr/bin/env bash
# Gate B: slice50_ids.txt (repo-stratified, seed 0) x1 at P=12 against a
# --max-num-seqs 20 server. Same agent/eval config as the 500x1 baseline;
# only the task container --cpus (1) and the parallelism differ.
# 12 task containers x 1 CPU, everything pinned to cores 0-15 (<= 16 cores).
# Images are pre-pulled (scripts/prepull.sh), so they are kept afterwards.
# Results: runs/slice50_s20/att1; compare with
#   .venv/bin/python -I scripts/throughput.py --run runs/slice50_s20 \
#     --baseline runs --ids slice50_ids.txt
set -u
. "$HOME/flint-swebench/env.sh"
RUN_DIR=$R/runs/slice50_s20 P=12 TASK_CPUS=1 CPUSET=0-15 KEEP_IMAGES=1 \
  START_FLOOR_GB=16 WD_FLOOR_GB=10 WD_INTERVAL=10 \
  exec "$R/scripts/run_all.sh" "$R/slice50_ids.txt" 1
