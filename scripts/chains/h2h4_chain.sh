# head-to-head 4 (results/jenga/lsf_h2h4_protocol.json), local RTX 3090, one pool of 6 shards
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
echo "h2h4 start $(date)"
for i in 0 1 2 3 4 5; do
  ( for arm in record lfci000 lfc000; do
      SCRIPT=eval/policy_rollout.py; ARGS="paired $arm 600"
      [ "$arm" = "lfci000" ] && { SCRIPT=eval/irr_rollout.py; ARGS="lfci000 600"; }
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_h2h4 EP_OFFSET=25600 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY $SCRIPT $ARGS
    done ) >> logs/h2h4_$i.out 2>&1 &
done
wait
echo "h2h4 sweep done $(date)"
for a in record lfci000 lfc000; do echo paired_${a}_h2h4 $(ls results/program_b/policy_rollouts/paired_${a}_h2h4/ep*.json | wc -l); done
$PY cluster/merge_paired.py paired_record_h2h4 paired_lfci000_h2h4 paired_lfc000_h2h4
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_h2h4.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs paired_record_h2h4 paired_lfci000_h2h4 paired_lfc000_h2h4 > results/jenga/outcome_audit_h2h4.log 2>&1
$PY eval/h2h4_analysis.py > results/jenga/h2h4_analysis.log 2>&1
echo "h2h4 analysed $(date)"
