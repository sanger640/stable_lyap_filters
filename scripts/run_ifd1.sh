#!/usr/bin/env bash
# IFD1 (results/jenga/ifd1_protocol.json): waits for the SOS1 chain (one GPU job at a time), then run -> grade.
# usage: nohup bash scripts/run_ifd1.sh <SOS1 chain pid> > results/jenga/ifd1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 $1 2>/dev/null; do sleep 300; done
echo "start $(date)"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 $PY eval/ifd1.py run > logs/ifd1_run.out 2>&1 || { echo "run failed $(date)"; exit 1; }
$PY eval/ifd1.py grade > results/jenga/ifd1_grade.log 2>&1; echo "graded $(date)"
grep -A3 '"H_exempt"' results/jenga/ifd1_result.json
