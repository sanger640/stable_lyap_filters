# IRRM amendment 2 (results/jenga/irrm_protocol.json): I0 / I1 at 2x and 1x on dev 15000-15099, one at a time
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
run() {  # tag exec_scale alea_scale irr_check
  echo "$1 start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$1 EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$2 HARD_OFF=1 \
      LSF_VALUE_NAME=value_irr8_alea.pt LSF_ALEA_SCALE=$3 ${4:+LSF_IRR_CHECK=$4} CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt \
      CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 $PY eval/policy_rollout.py paired lfc000 100 >> logs/irrmcl$1_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000$1
  AUDIT_OUT=results/jenga/outcome_audit$1.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$2 $PY eval/outcome_audit.py runs paired_lfc000$1 > results/jenga/outcome_audit$1.log 2>&1
  echo "$1 audited $(date)"; tail -1 results/jenga/outcome_audit$1.log
}
run _i0dev2x 2 2 ""
run _i1dev2x 2 2 margin_irr_nl8.pt
run _i0dev1x 1 1 ""
run _i1dev1x 1 1 margin_irr_nl8.pt
