# LSF-pub (results/jenga/lsf_pub_protocol.json): train both faithful values, gate, then dev sweeps after the IRRM a4 chain
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
for m in paper code; do
  echo "train $m start $(date)"
  LSF_VALUE_NAME=value_pub_$m.pt LSF_PUB_MODE=$m PYTHONUNBUFFERED=1 $PY eval/lsf.py ddpg_pub > results/program_b/lsf_v1/value_pub_${m}_train.log 2>&1
  ls -la results/program_b/lsf_v1/value_pub_$m.pt || exit 1
  LSF_VALUE_NAME=value_pub_$m.pt $PY eval/lsf_audit.py > results/jenga/lsf_pub_${m}_audit.log 2>&1
  LSF_VALUE_NAME=value_pub_$m.pt $PY eval/lsf_diag.py > results/jenga/lsf_pub_${m}_diag.log 2>&1
done
$PY eval/lsf_pub_gate.py value_pub_paper value_pub_code > results/jenga/lsf_pub_gate.log 2>&1
echo "gated $(date)"; cat results/jenga/lsf_pub_gate.json
while kill -0 2260208 2>/dev/null; do sleep 120; done                      # one closed-loop pool at a time
for m in paper code; do
  python3 -c "import json,sys; sys.exit(0 if json.load(open('results/jenga/lsf_pub_gate.json'))['value_pub_$m']['pass'] else 1)" || { echo "$m gate FAIL: no dev"; continue; }
  T=_pub${m:0:1}dev
  for e in 010 020 030 040 050; do
    for i in 0 1 2 3 4 5; do
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$T EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      LSF_VALUE_NAME=value_pub_$m.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY eval/policy_rollout.py paired lsfp$e 100 >> logs/lsfpub${T}_$i.out 2>&1 &
    done
    wait; echo "$m eps $e done $(date)"
  done
  RUNS=$(for e in 010 020 030 040 050; do echo -n "paired_lsfp${e}$T "; done)
  $PY cluster/merge_paired.py $RUNS
  AUDIT_OUT=results/jenga/outcome_audit$T.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $RUNS > results/jenga/outcome_audit$T.log 2>&1
  echo "$m dev audited $(date)"; tail -5 results/jenga/outcome_audit$T.log
done
