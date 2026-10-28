# lflsf LF4 (results/jenga/lflsf_LF4_protocol.json): edits lfm / lfg / lfc at the LF3-selected eps, same dev episodes
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
until [ -s results/jenga/lflsf_LF3_dev.json ] && ! kill -0 $1 2>/dev/null; do sleep 120; done
E=$($PY -c "import json;s=json.load(open('results/jenga/lflsf_LF3_dev.json'))['selected'];print(s[3:] if s else '000')")
ARMS="lfm$E lfg$E lfc$E"; echo "LF4 start eps=$E $(date)"
for i in 0 1; do
  ( for arm in $ARMS; do
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lflsfdev EP_OFFSET=15000 SHARD=$i/2 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY eval/policy_rollout.py paired $arm 100
    done ) > logs/lflsfdev4_$i.out 2>&1 &
done
wait
echo "LF4 sweep done $(date)"
$PY cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lflsfdev; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_lflsfdev4.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lflsfdev; done) > results/jenga/outcome_audit_lflsfdev4.log 2>&1
$PY eval/lflsf.py dev4 | tee results/jenga/lflsf_LF4_dev.log
echo "LF4 analysed $(date)"
