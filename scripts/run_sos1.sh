#!/usr/bin/env bash
# SOS1 stage S (results/jenga/sos1_protocol.json): waits for the RLAB2 chain (one GPU pool at a time), then
# calibration (2 shards) -> CF3 evaluation (2 shards) -> grade.
# usage: nohup bash scripts/run_sos1.sh <RLAB2 chain pid> > results/jenga/sos1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 $1 2>/dev/null; do sleep 300; done
echo "start $(date)"
for w in calib cf3; do
  for i in 0 1; do SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/sos1.py $w > logs/sos1_${w}_$i.out 2>&1 & done
  wait; echo "$w done $(date)"; ls results/jenga/sos1_${w}_*.npz || { echo "missing $w, stop"; exit 1; }
done
$PY eval/sos1.py grade > results/jenga/sos1_grade.log 2>&1; echo "graded $(date)"
grep -E '"verdict"|"selected"|"reproduced"' results/jenga/sos1_result.json
