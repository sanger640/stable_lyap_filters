#!/usr/bin/env bash
# ABL1 (results/jenga/abl1_protocol.json): level-set vs CBF ablation + 1.6 s value at 80%, dev episodes, after the NSI chain.
# usage (repo root): nohup bash scripts/run_abl1.sh <nsi chain pid> > results/jenga/abl1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
arm() {  # tag mode noise extra-env...
  local tag=$1 mode=$2 nz=$3; shift 3
  echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 "$@" \
      $PY eval/policy_rollout.py paired $mode 100 >> logs/abl1${tag}_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_$mode$tag
  AUDIT_OUT=results/jenga/outcome_audit$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_$mode$tag \
    > results/jenga/outcome_audit$tag.log 2>&1
  echo "$tag audited $(date)"; tail -1 results/jenga/outcome_audit$tag.log
}
for nz in 2 1; do
  arm _lv0dev${nz}x    lfc000 $nz LSF_VALUE_NAME=value_irr8_alea.pt  LSF_CBF_GAMMA=1.0
  arm _lv30dev${nz}x   lfc030 $nz LSF_VALUE_NAME=value_irr8_alea.pt  LSF_CBF_GAMMA=1.0
  arm _lsr30dev${nz}x  lfa030 $nz LSF_VALUE_NAME=value_irr8_alea.pt
  arm _a16g20dev${nz}x lfc000 $nz LSF_VALUE_NAME=value_irr16_alea.pt LSF_CBF_GAMMA=0.2
done
$PY eval/abl1_analysis.py > results/jenga/abl1_analysis.log 2>&1
echo "ABL1 analysed $(date)"
