# lflsf LF6 (results/jenga/lflsf_LF6_protocol.json): side-effect margin -> gate -> SAC value (after LF5's) -> gate -> dev arms (after LF5 chain)
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
export LF_TARGET=se
while kill -0 3631881 2>/dev/null; do sleep 30; done
grep -q '"GATE_LF6a_pass": true' results/jenga/lflsf_LF1_se.json || { echo "LF6a FAIL, stop $(date)"; exit 0; }
echo "LF6a pass $(date)"
while [ ! -f results/program_b/lsf_v1/value_lf8_alea.pt ] || pgrep -f "[l]sf.py sac" >/dev/null; do sleep 120; done
echo "LF6b training start $(date)"
export LSF_CLF=margin_lfse.pt LSF_VALUE_NAME=value_lfse_alea.pt LSF_ALEA=8
$PY eval/lsf.py sac > results/program_b/lsf_v1/value_lfse_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_lfse_alea.pt || exit 1
$PY eval/lsf_audit.py > results/jenga/lflsf_LF6b_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/lflsf_LF6b_diag.log 2>&1
$PY eval/lflsf.py gate2 | tee results/jenga/lflsf_LF6b.log
grep -q '"GATE_LF2_pass": true' results/jenga/lflsf_LF2_se.json || { echo "LF6b FAIL, stop $(date)"; exit 0; }
unset LSF_CLF LSF_ALEA
ARMS="lfa010 lfa030 lfa050 lfc030"; echo "LF6 arms $ARMS; waiting for the LF5 chain $(date)"
while kill -0 $1 2>/dev/null; do sleep 120; done
echo "LF6 sweep start $(date)"
for i in 0 1 2; do
  ( for arm in $ARMS; do
      LSF_VALUE_NAME=value_lfse_alea.pt MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lfsedev EP_OFFSET=15000 SHARD=$i/3 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY eval/policy_rollout.py paired $arm 100
    done ) >> logs/lfsedev_$i.out 2>&1 &
done
wait
echo "LF6 sweep done $(date)"
$PY cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lfsedev; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_lfsedev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lfsedev; done) > results/jenga/outcome_audit_lfsedev.log 2>&1
echo "LF6 audited $(date)"
