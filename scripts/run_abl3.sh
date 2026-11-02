#!/usr/bin/env bash
# ABL3 (results/jenga/abl3_protocol.json): gated edit vs always-on (2x, 1x) and ABL2's key ablations at 1x, episodes 26800-27099.
# One GPU pool at a time; arms are resumable (skipped when 300 episodes exist).
# usage (repo root): nohup bash scripts/run_abl3.sh > results/jenga/abl3_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
echo "start $(date)"
arm() {  # name script mode tag noise extra-env...  (ours90 settings unless overridden; later env entries win)
  local name=$1 scr=$2 mode=$3 tag=$4 nz=$5; shift 5
  [ "$(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json 2>/dev/null | wc -l)" -ge 300 ] && { echo "$name already done, skipped $(date)"; return; }
  echo "$name start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=26800 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_irr8_alea.pt LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=$nz IRR_S=$nz "$@" \
      $PY eval/$scr $([ $scr = policy_rollout.py ] && echo paired) $mode 300 >> logs/abl3${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name done $(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json | wc -l) episodes $(date)"
}
P=policy_rollout.py; G=irr_rollout.py
arm o90_1x  $P lfc000  _ab3_o90_1x  1
arm b50_1x  $P lfc000  _ab3_b50_1x  1 LSF_CBF_GAMMA=0.5
arm b80_1x  $P lfc000  _ab3_b80_1x  1 LSF_CBF_GAMMA=0.2
arm lv0_1x  $P lfc000  _ab3_lv0_1x  1 LSF_CBF_GAMMA=1.0
arm det_1x  $P lfc000  _ab3_det_1x  1 LSF_DET=1
arm mean_1x $P lfc000  _ab3_mean_1x 1 LSF_AGG=mean
arm g90_2x  $G lfcg090 _ab3_g90_2x  2 IRR_MASK=none
arm gk_2x   $G lfck000 _ab3_gk_2x   2
arm g90_1x  $G lfcg090 _ab3_g90_1x  1 IRR_MASK=none
arm gk_1x   $G lfck000 _ab3_gk_1x   1
for nz in 1 2; do
  R=$(ls -d results/program_b/policy_rollouts/paired_*_ab3_*_${nz}x | xargs -n1 basename | tr '\n' ' ')
  $PY cluster/merge_paired.py $R
  AUDIT_OUT=results/jenga/outcome_audit_ab3_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs $R \
    > results/jenga/outcome_audit_ab3_${nz}x.log 2>&1
  echo "audited ${nz}x $(date)"; tail -n $(echo $R | wc -w) results/jenga/outcome_audit_ab3_${nz}x.log
done
$PY eval/abl3_analysis.py > results/jenga/abl3_analysis.log 2>&1; echo "ABL3 analysed $(date)"
