# lflsf LF2 gate -> LF3 local dev sweep (results/jenga/lflsf_protocol.json). Replaces lf_chain.sh 2026-10-25 17:15:
# user deprioritised psel, so prune_check / stepbench are dropped from the local queue; LF3 waits only for the GPU.
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
S=scripts/chains
true
$PY eval/cert_recheck.py analyse > results/jenga/deep_dive/cert_recheck_analyse.log 2>&1
echo "LF2 trained $(date)"; ls -la results/program_b/lsf_v1/value_lf_alea.pt || exit 1
export LSF_VALUE_NAME=value_lf_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/lflsf_LF2_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/lflsf_LF2_diag.log 2>&1
$PY eval/lflsf.py gate2 | tee results/jenga/lflsf_LF2.log
grep -q '"GATE_LF2_pass": true' results/jenga/lflsf_LF2.json || { echo "LF2 FAIL, stop"; exit 0; }
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
until [ -e $S/lf3_go ]; do sleep 30; done      # code freeze flag: set once any edit to policy_rollout / cbf_filter is done
echo "LF3 dev sweep start $(date)"
mkdir -p logs
for i in 0 1; do
  ( for arm in record lfa-15 lfa-25 lfa000 lfa010 lfa030 lfa050 lfr010 lfr030 lfr050; do
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lflsfdev EP_OFFSET=15000 SHARD=$i/2 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY eval/policy_rollout.py paired $arm 100
    done ) > logs/lflsfdev_$i.out 2>&1 &
done
wait
echo "LF3 dev sweep done $(date)"
ARMS="record lfa-25 lfa-15 lfa000 lfa010 lfa030 lfa050 lfr010 lfr030 lfr050"
$PY cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lflsfdev; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_lflsfdev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lflsfdev; done) > results/jenga/outcome_audit_lflsfdev.log 2>&1
$PY eval/lflsf.py dev | tee results/jenga/lflsf_LF3_dev.log
echo "LF3 analysed $(date)"
