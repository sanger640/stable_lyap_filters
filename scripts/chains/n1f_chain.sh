# N1F (results/jenga/n1f_protocol.json): lfck000 and lfc000 at 1x execution noise, dev 15000-15099, after the edit-limit check
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 2160179 2>/dev/null; do sleep 60; done
echo "n1f start $(date)"
for i in 0 1 2 3 4 5; do
  ( MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_n1dev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=1 HARD_OFF=1 \
    CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
    $PY eval/irr_rollout.py lfck000 100
    MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_n1dev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=1 HARD_OFF=1 \
    CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
    $PY eval/policy_rollout.py paired lfc000 100 ) >> logs/n1f_$i.out 2>&1 &
done
wait
echo "n1f sweep done $(date)"
for a in lfck000 lfc000; do echo paired_${a}_n1dev $(ls results/program_b/policy_rollouts/paired_${a}_n1dev/ep*.json | wc -l); done
$PY cluster/merge_paired.py paired_lfck000_n1dev paired_lfc000_n1dev
AUDIT_OUT=results/jenga/outcome_audit_n1f.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=1 $PY eval/outcome_audit.py runs paired_lfck000_n1dev paired_lfc000_n1dev > results/jenga/outcome_audit_n1f.log 2>&1
echo "n1f audited $(date)"
