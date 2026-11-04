#!/usr/bin/env bash
# RCTL1 (results/rctl1/protocol.json): waits until the KNOCK1b chain and NOISE1 have finished (scheduling amendment),
# then trains E_fast and grades it. usage (repo root): nohup bash scripts/run_rctl1.sh > results/rctl1/chain.log 2>&1 &
cd "$(dirname "$0")/.."
echo "waiting $(date)"
until grep -q "KNOCK1B graded" results/knock/knock1b_chain.log || [ "$(grep -c stop results/knock/knock1b_chain.log)" -ge 2 ]; do sleep 300; done
until grep -qE '"verdict"|Traceback' results/noise1/run.log; do sleep 120; done
echo "start $(date)"
FAST_TRAIN_Z_ON_CPU=1 PYTHONUNBUFFERED=1 /home/corey/wksp/dp_venv/bin/python eval/recipe_control.py > results/rctl1/run.log 2>&1
echo "RCTL1 done $(date)"; tail -12 results/rctl1/run.log
