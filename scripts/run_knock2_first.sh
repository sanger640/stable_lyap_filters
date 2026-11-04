#!/usr/bin/env bash
# One-off order (user 2026-11-04): wait for the running t105_2x workers, run KNOCK stage 2, then restart SEED2/TAU3.
# usage (repo root): nohup bash scripts/run_knock2_first.sh <worker pids...> > results/knock/knock2_first.log 2>&1 &
cd "$(dirname "$0")/.."
echo "waiting for t105_2x workers $* $(date)"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
echo "t105_2x workers done: $(ls results/program_b/policy_rollouts/paired_lfc000_st3_t105_2x/ep*.json | wc -l) episodes $(date)"
NO_WAIT=1 bash scripts/run_knock2.sh > results/knock/knock2_chain.log 2>&1
echo "knock2 finished $(date); restarting SEED2/TAU3"
echo "restart after KNOCK stage 2 $(date)" >> results/jenga/seed2tau3_chain.log
nohup bash scripts/run_seed2tau3.sh >> results/jenga/seed2tau3_chain.log 2>&1 &
echo "SEED2/TAU3 restarted pid $! $(date)"
