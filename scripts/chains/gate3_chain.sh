cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
echo "gate3 dev start $(date)"
for i in 0 1 2 3 4 5; do
  MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_gate3dev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
  CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 $PY eval/irr_rollout.py lfcj000 100 >> logs/gate3dev_$i.out 2>&1 &
done
wait
echo "gate3 dev sweep done $(date)"
$PY cluster/merge_paired.py paired_lfcj000_gate3dev
AUDIT_OUT=results/jenga/outcome_audit_gate3dev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs paired_lfcj000_gate3dev > results/jenga/outcome_audit_gate3dev.log 2>&1
echo "gate3 dev audited $(date)"
