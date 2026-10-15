#!/bin/bash
# Run one arm of a paired closed-loop test split over several worker processes (episode i goes to worker i % N).
# usage: eval/run_paired_parallel.sh <mode> <episodes> <workers>   (env: HARD_MASS, CEM_VERSION, PAIRED_TAG pass through)
cd "$(dirname "$0")/.."
MODE=$1; N=$2; W=${3:-4}
for k in $(seq 0 $((W-1))); do
  SHARD="$k/$W" ${PY:-/home/corey/wksp/dp_venv/bin/python} eval/policy_rollout.py paired "$MODE" "$N" > "results/program_b/paired_${MODE}${PAIRED_TAG}_w$k.log" 2>&1 &
done
wait
