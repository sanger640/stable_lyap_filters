# IRRM amendment 1 (option c): B (fragility value at 1x) then A (irreversibility value at 2x), one training job at a time
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
echo "B train start $(date)"
LSF_VALUE_NAME=value_lf_alea_n1.pt LSF_CLF=margin_lf.pt LSF_ALEA=8 LSF_ALEA_SCALE=1 PYTHONUNBUFFERED=1 $PY eval/lsf.py sac > results/program_b/lsf_v1/value_lf_alea_n1_train.log 2>&1
echo "B trained $(date)"; ls -la results/program_b/lsf_v1/value_lf_alea_n1.pt || exit 1
export LSF_VALUE_NAME=value_lf_alea_n1.pt LSF_CLF=margin_lf.pt LSF_ALEA=8 LSF_ALEA_SCALE=1
$PY eval/lsf_audit.py > results/jenga/irrm_B_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_B_diag.log 2>&1
$PY eval/irrm_gate.py B > results/jenga/irrm_B_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA LSF_ALEA_SCALE
echo "B gated $(date)"
if grep -q '"B_pass": true' results/jenga/irrm_B_gate.json; then
  for i in 0 1 2 3 4 5; do
    MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_v1xdev EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=1 HARD_OFF=1 \
    LSF_VALUE_NAME=value_lf_alea_n1.pt LSF_ALEA_SCALE=1 CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json \
    PYTHONUNBUFFERED=1 $PY eval/policy_rollout.py paired lfc000 100 >> logs/v1xdev_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000_v1xdev
  AUDIT_OUT=results/jenga/outcome_audit_v1xdev.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=1 $PY eval/outcome_audit.py runs paired_lfc000_v1xdev > results/jenga/outcome_audit_v1xdev.log 2>&1
  echo "B closed loop audited $(date)"
else
  echo "B gate FAIL: no closed loop $(date)"
fi
# A: margin net on the nl8 labels (computed overnight), then the value
while [ ! -f results/jenga/irrm2_held_nl8.npz ]; do sleep 120; done
while pgrep -f "[i]rrm2.py labels" >/dev/null || pgrep -f "[i]rrm2.py held" >/dev/null; do sleep 120; done
echo "A margin start $(date)"
IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nl8.log 2>&1
$PY eval/irrm_gate.py M > results/jenga/irrm_M_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_M_gate.json || { echo "A margin gate FAIL, stop $(date)"; exit 0; }
echo "A train start $(date)"
LSF_VALUE_NAME=value_irr8_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_irr_nl8.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 $PY eval/lsf.py sac > results/program_b/lsf_v1/value_irr8_alea_train.log 2>&1
echo "A trained $(date)"; ls -la results/program_b/lsf_v1/value_irr8_alea.pt || exit 1
export LSF_VALUE_NAME=value_irr8_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_A_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_A_diag.log 2>&1
$PY eval/irrm_gate.py A > results/jenga/irrm_A_gate.log 2>&1
echo "A gated $(date)"
