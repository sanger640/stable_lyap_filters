#!/usr/bin/env bash
# H2H5 decisive head-to-head (results/jenga/h2h5_protocol.json). DO NOT LAUNCH without the user's go: it consumes the
# fresh episodes 26200-26799. Runs every arm on 6 local shards, one arm at a time (never two DINOv2 pools at once),
# 2x then 1x, audits each condition, then eval/h2h5_analysis.py.  usage (from the repo root):  nohup bash scripts/run_h2h5.sh > results/jenga/h2h5_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
arm() {  # name noise(1|2) mode tag extra-env...
  local name=$1 nz=$2 mode=$3 tag=$4; shift 4
  echo "$name ${nz}x start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=26200 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 "$@" \
      $PY eval/policy_rollout.py paired $mode 600 >> logs/h2h5${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name ${nz}x done $(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json | wc -l) episodes $(date)"
}
for nz in 2 1; do
  arm policy  $nz record  _h5_${nz}x
  arm ours90  $nz lfc000  _h5o90_${nz}x LSF_VALUE_NAME=value_irr8_alea.pt LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=$nz
  arm ours80  $nz lfc000  _h5o80_${nz}x LSF_VALUE_NAME=value_irr8_alea.pt LSF_CBF_GAMMA=0.2 LSF_ALEA_SCALE=$nz
  arm unisafe $nz lsfa030 _h5_${nz}x    LSF_ALEA_SCALE=$nz
  arm lsf_pub $nz lsfp020 _h5_${nz}x    LSF_VALUE_NAME=value_pub_code.pt
  RUNS="paired_record_h5_${nz}x paired_lfc000_h5o90_${nz}x paired_lfc000_h5o80_${nz}x paired_lsfa030_h5_${nz}x paired_lsfp020_h5_${nz}x"
  $PY cluster/merge_paired.py $RUNS
  AUDIT_OUT=results/jenga/outcome_audit_h5_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs $RUNS \
    > results/jenga/outcome_audit_h5_${nz}x.log 2>&1
  echo "${nz}x audited $(date)"
done
$PY eval/h2h5_analysis.py > results/jenga/h2h5_analysis.log 2>&1
echo "H2H5 analysed $(date)"
