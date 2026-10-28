# TO2x local arms (results/jenga/to2x_protocol.json): time-out episodes of h2h4 re-run with a 120-chunk cap, after lfck000
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 294754 2>/dev/null; do sleep 60; done
echo "to2x local start $(date)"
for i in 0 1 2 3 4 5; do
  ( for arm in record lfci000 lfc000; do
      SCRIPT=eval/policy_rollout.py; ARGS="paired $arm 600"
      [ "$arm" = "lfci000" ] && { SCRIPT=eval/irr_rollout.py; ARGS="lfci000 600"; }
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_to2x EP_OFFSET=25600 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 PAIRED_EPS=$PWD/results/jenga/to2x/eps_$arm.json \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY $SCRIPT $ARGS
    done ) >> logs/to2x_$i.out 2>&1 &
done
wait
echo "to2x local sweep done $(date)"
for a in record lfci000 lfc000; do echo paired_${a}_to2x $(ls results/program_b/policy_rollouts/paired_${a}_to2x/ep*.json | wc -l); done
$PY cluster/merge_paired.py paired_record_to2x paired_lfci000_to2x paired_lfc000_to2x
AUDIT_OUT=results/jenga/outcome_audit_to2x_local.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs paired_record_to2x paired_lfci000_to2x paired_lfc000_to2x > results/jenga/outcome_audit_to2x_local.log 2>&1
echo "to2x local audited $(date)"
