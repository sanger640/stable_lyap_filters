# LSF-pub amendment 1: exploratory dev sweep of value_pub_code.pt after the IRRM a4 chain and the lsfpub chain
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 2260208 2>/dev/null || kill -0 2276333 2>/dev/null; do sleep 120; done
T=_pubcdev
for e in 010 020 030 040 050; do
  for i in 0 1 2 3 4 5; do
    MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$T EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
    LSF_VALUE_NAME=value_pub_code.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
    $PY eval/policy_rollout.py paired lsfp$e 100 >> logs/lsfpub${T}_$i.out 2>&1 &
  done
  wait; echo "eps $e done $(date)"
done
RUNS=$(for e in 010 020 030 040 050; do echo -n "paired_lsfp${e}$T "; done)
$PY cluster/merge_paired.py $RUNS
AUDIT_OUT=results/jenga/outcome_audit$T.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $RUNS > results/jenga/outcome_audit$T.log 2>&1
echo "audited $(date)"; tail -5 results/jenga/outcome_audit$T.log
