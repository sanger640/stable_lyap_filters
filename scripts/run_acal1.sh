#!/usr/bin/env bash
# ACAL1 (results/jenga/acal1_protocol.json): offline calibration of each value's barrier -> gate N -> dev arm of the
# irreversible-fragility value at its calibrated gamma (gate D) -> fresh paired arms (ours90, fxn) on 27400-27999, 2x then 1x
# -> audits -> analysis. One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_acal1.sh > results/jenga/acal1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
echo "start $(date)"

# ---- calibration (offline)
MUJOCO_GL=egl PYTHONUNBUFFERED=1 $PY eval/acal1.py run > logs/acal1_run.out 2>&1 || { echo "run failed $(date)"; exit 1; }
$PY eval/acal1.py grade > results/jenga/acal1_grade.log 2>&1; echo "graded $(date)"
eval "$(grep -E '^(GATE_N|FX_G2|FX_G1)=' results/jenga/acal1_grade.log)"
echo "GATE_N=$GATE_N FX_G2=$FX_G2 FX_G1=$FX_G1"
[ "$GATE_N" = "1" ] || { echo "STOP at gate N $(date)"; exit 0; }
declare -A G=([2]=$FX_G2 [1]=$FX_G1)

# ---- dev (gate D)
for nz in 2 1; do
  tag=ac1fxndev${nz}x
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_nsif16_alea.pt LSF_CBF_GAMMA=${G[$nz]} \
      $PY eval/policy_rollout.py paired lfc000 100 >> logs/acal1_${tag}_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000_$tag
  AUDIT_OUT=results/jenga/outcome_audit_$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000_$tag \
    > results/jenga/outcome_audit_$tag.log 2>&1
  echo "$tag $(date) $(tail -1 results/jenga/outcome_audit_$tag.log)"
done
$PY eval/acal1.py dev > results/jenga/acal1_dev.log 2>&1; cat results/jenga/acal1_dev.log
grep -q "DEV_PASS=1" results/jenga/acal1_dev.log || { echo "STOP at gate D $(date)"; exit 0; }

# ---- fresh paired arms
fresh() {  # tag value gamma noise
  local tag=$1 value=$2 gam=$3 nz=$4
  echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_$tag EP_OFFSET=27400 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=$value LSF_CBF_GAMMA=$gam LSF_ALEA_SCALE=$nz \
      $PY eval/policy_rollout.py paired lfc000 600 >> logs/acal1_${tag}_$i.out 2>&1 &
  done
  wait
  echo "$tag done $(ls results/program_b/policy_rollouts/paired_lfc000_$tag/ep*.json | wc -l) episodes $(date)"
  $PY cluster/merge_paired.py paired_lfc000_$tag
  AUDIT_OUT=results/jenga/outcome_audit_$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz \
    $PY eval/outcome_audit.py runs paired_lfc000_$tag > results/jenga/outcome_audit_$tag.log 2>&1
  tail -1 results/jenga/outcome_audit_$tag.log
}
for nz in 2 1; do
  fresh ac1o90_${nz}x value_irr8_alea.pt 0.1 $nz
  fresh ac1fxn_${nz}x value_nsif16_alea.pt ${G[$nz]} $nz
done
$PY eval/acal1.py test > results/jenga/acal1_analysis.log 2>&1; echo "ACAL1 analysed $(date)"
head -12 results/jenga/acal1_analysis.log
