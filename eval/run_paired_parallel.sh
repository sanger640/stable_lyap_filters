#!/bin/bash
# Run one arm of a paired closed-loop test split over worker processes: worker k of TOTAL takes episodes i % TOTAL == k.
# usage: eval/run_paired_parallel.sh <mode> <episodes> <first_worker> <last_worker> <total_workers>
# (env: HARD_MASS, HARD_OFF, CEM_VERSION, PAIRED_TAG, PY pass through). Keep each worker on the same machine in every arm.
cd "$(dirname "$0")/.."
MODE=$1; N=$2; K0=${3:-0}; K1=${4:-3}; TOT=${5:-4}
for k in $(seq $K0 $K1); do
  SHARD="$k/$TOT" ${PY:-/home/corey/wksp/dp_venv/bin/python} eval/policy_rollout.py paired "$MODE" "$N" > "results/program_b/paired_${MODE}${PAIRED_TAG}_w$k.log" 2>&1 &
done
wait
