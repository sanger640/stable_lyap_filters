# lflsf LF5 (results/jenga/lflsf_LF5_protocol.json): x8 margin -> SAC value -> gate -> eps -> dev arms -> overall selection
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
export LF_SCALE=8
$PY eval/lflsf.py margin > results/jenga/lflsf_LF1_x8.log 2>&1
grep -q '"GATE_LF1_pass": true' results/jenga/lflsf_LF1_x8.json || { echo "LF5a FAIL, stop"; exit 0; }
echo "LF5a pass $(date)"
export LSF_CLF=margin_lf8.pt LSF_VALUE_NAME=value_lf8_alea.pt LSF_ALEA=8
$PY eval/lsf.py sac > results/program_b/lsf_v1/value_lf8_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_lf8_alea.pt || exit 1
$PY eval/lsf_audit.py > results/jenga/lflsf_LF2_x8_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/lflsf_LF2_x8_diag.log 2>&1
$PY eval/lflsf.py gate2 | tee results/jenga/lflsf_LF2_x8.log
grep -q '"GATE_LF2_pass": true' results/jenga/lflsf_LF2_x8.json || { echo "LF5b FAIL, stop"; exit 0; }
unset LSF_CLF LSF_ALEA
$PY eval/lflsf.py eps5
E1=$($PY -c "import json;d=json.load(open('results/jenga/lflsf_LF5_eps.json'));print(f\"{round(d['labelled_eps_0.10']['x8_eps']*100):03d}\")")
E2=$($PY -c "import json;d=json.load(open('results/jenga/lflsf_LF5_eps.json'));print(f\"{round(d['labelled_eps_0.30']['x8_eps']*100):03d}\")")
ARMS="lfa$E1 lfa$E2 lfc$E2"; echo "LF5 arms $ARMS; waiting for LF4 $(date)"
while kill -0 $1 2>/dev/null; do sleep 120; done
echo "LF5 sweep start $(date)"
for i in 0 1 2; do
  ( for arm in $ARMS; do
      MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lf8dev EP_OFFSET=15000 SHARD=$i/3 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      $PY eval/policy_rollout.py paired $arm 100
    done ) >> logs/lf8dev_$i.out 2>&1 &
done
wait
echo "LF5 sweep done $(date)"
$PY cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lf8dev; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_lf8dev.json MUJOCO_GL=egl $PY eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lf8dev; done) > results/jenga/outcome_audit_lf8dev.log 2>&1
$PY eval/lflsf.py dev_all | tee results/jenga/lflsf_dev_all.log
echo "LF5 analysed $(date)"
