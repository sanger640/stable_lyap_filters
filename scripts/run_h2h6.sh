#!/usr/bin/env bash
# H2H6 confirmatory fresh head-to-head (results/jenga/h2h6_protocol.json): episodes 26800-27399, arms policy / ours90 /
# a16g20 / unisafe, 6 local shards, one arm at a time, 2x then 1x, audits, then eval/h2h6_analysis.py.
# usage (repo root): nohup bash scripts/run_h2h6.sh > results/jenga/h2h6_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
arm() {  # name noise mode tag extra-env...
  local name=$1 nz=$2 mode=$3 tag=$4; shift 4
  echo "$name ${nz}x start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=26800 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 "$@" \
      $PY eval/policy_rollout.py paired $mode 600 >> logs/h2h6${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name ${nz}x done $(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json | wc -l) episodes $(date)"
}
for nz in 2 1; do
  arm policy  $nz record  _h6_${nz}x
  arm ours90  $nz lfc000  _h6o90_${nz}x LSF_VALUE_NAME=value_irr8_alea.pt  LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=$nz
  arm a16g20  $nz lfc000  _h6a16_${nz}x LSF_VALUE_NAME=value_irr16_alea.pt LSF_CBF_GAMMA=0.2 LSF_ALEA_SCALE=$nz
  arm unisafe $nz lsfa030 _h6_${nz}x    LSF_ALEA_SCALE=$nz
  RUNS="paired_record_h6_${nz}x paired_lfc000_h6o90_${nz}x paired_lfc000_h6a16_${nz}x paired_lsfa030_h6_${nz}x"
  $PY cluster/merge_paired.py $RUNS
  AUDIT_OUT=results/jenga/outcome_audit_h6_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs $RUNS \
    > results/jenga/outcome_audit_h6_${nz}x.log 2>&1
  echo "${nz}x audited $(date)"
done
$PY eval/h2h6_analysis.py > results/jenga/h2h6_analysis.log 2>&1
echo "H2H6 analysed $(date)"
