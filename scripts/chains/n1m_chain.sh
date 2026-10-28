# N1M (results/jenga/n1m_protocol.json): lfck000 and lfc000 at 1x noise, imagined noise matched (IRR_S=1, LSF_ALEA_SCALE=1)
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
echo "n1m start $(date)"
for i in 0 1 2 3 4 5; do
  ( export MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_n1mdev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=1 HARD_OFF=1 \
      IRR_S=1 LSF_ALEA_SCALE=1 CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1
    $PY eval/irr_rollout.py lfck000 100
    $PY eval/policy_rollout.py paired lfc000 100 ) >> logs/n1m_$i.out 2>&1 &
done
wait
echo "n1m sweep done $(date)"
for a in lfck000 lfc000; do echo paired_${a}_n1mdev $(ls results/program_b/policy_rollouts/paired_${a}_n1mdev/ep*.json | wc -l); done
$PY cluster/merge_paired.py paired_lfck000_n1mdev paired_lfc000_n1mdev
AUDIT_OUT=results/jenga/outcome_audit_n1m.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=1 $PY eval/outcome_audit.py runs paired_lfck000_n1mdev paired_lfc000_n1mdev > results/jenga/outcome_audit_n1m.log 2>&1
echo "n1m audited $(date)"
