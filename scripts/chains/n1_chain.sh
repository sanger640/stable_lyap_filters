# policy alone at 1x execution noise on the 100 dev episodes (exploratory reference; same setup otherwise), after the gate3 chain
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 3760372 2>/dev/null; do sleep 60; done
echo "n1 start $(date)"
for i in 0 1 2 3 4 5; do
  MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_n1dev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=1 HARD_OFF=1 \
  CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
  $PY eval/policy_rollout.py paired record 100 >> logs/n1dev_$i.out 2>&1 &
done
wait
$PY cluster/merge_paired.py paired_record_n1dev
AUDIT_OUT=results/jenga/outcome_audit_n1dev.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=1 $PY eval/outcome_audit.py runs paired_record_n1dev > results/jenga/outcome_audit_n1dev.log 2>&1
echo "n1 audited $(date)"
