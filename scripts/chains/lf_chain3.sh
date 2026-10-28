# LF3 sweep only (LF2 gate already passed, results/jenga/lflsf_LF2.json); amendments 1 + 2 arms
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
echo "LF3 dev sweep start $(date)"
mkdir -p logs
for i in 0 1 2; do
  ( for arm in record lfa-15 lfa-25 lfa000 lfa010 lfa030 lfa050 lfr010 lfr030 lfr050; do
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lflsfdev EP_OFFSET=15000 SHARD=$i/3 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY eval/policy_rollout.py paired $arm 100
    done ) >> logs/lflsfdev_$i.out 2>&1 &
done
wait
echo "LF3 dev sweep done $(date)"
ARMS="record lfa-25 lfa-15 lfa000 lfa010 lfa030 lfa050 lfr010 lfr030 lfr050"
$PY cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lflsfdev; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_lflsfdev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lflsfdev; done) > results/jenga/outcome_audit_lflsfdev.log 2>&1
$PY eval/lflsf.py dev | tee results/jenga/lflsf_LF3_dev.log
echo "LF3 analysed $(date)"
