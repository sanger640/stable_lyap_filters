cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
cl() {  # tag exec_scale alea_scale value gamma
  echo "$1 start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$1 EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$2 HARD_OFF=1 \
      LSF_VALUE_NAME=$4 LSF_ALEA_SCALE=$3 LSF_CBF_GAMMA=$5 CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt \
      CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 $PY eval/policy_rollout.py paired lfc000 100 >> logs/irrmcl$1_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000$1
  AUDIT_OUT=results/jenga/outcome_audit$1.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$2 $PY eval/outcome_audit.py runs paired_lfc000$1 > results/jenga/outcome_audit$1.log 2>&1
  echo "$1 audited $(date)"; tail -1 results/jenga/outcome_audit$1.log
}
