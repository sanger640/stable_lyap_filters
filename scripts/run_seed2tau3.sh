#!/usr/bin/env bash
# SEED2 + TAU3 (results/jenga/seed2tau3_protocol.json): value-seed spread of ours90 (2x, 1x) and the tau x1.05 value (2x),
# episodes 26800-27099. One GPU pool at a time; arms are resumable (skipped when 300 episodes exist).
# usage (repo root): nohup bash scripts/run_seed2tau3.sh > results/jenga/seed2tau3_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
echo "start $(date)"
arm() {  # name tag noise extra-env...  (ours90 settings unless overridden; later env entries win)
  local name=$1 tag=$2 nz=$3; shift 3
  [ "$(ls results/program_b/policy_rollouts/paired_lfc000$tag/ep*.json 2>/dev/null | wc -l)" -ge 300 ] && { echo "$name already done, skipped $(date)"; return; }
  echo "$name start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=26800 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_irr8_alea.pt LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=$nz IRR_S=$nz "$@" \
      $PY eval/policy_rollout.py paired lfc000 300 >> logs/st3${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name done $(ls results/program_b/policy_rollouts/paired_lfc000$tag/ep*.json | wc -l) episodes $(date)"
}
value() {  # value-name margin-name   (value_irr8_alea.pt's recipe in model_OIL; extra env before the call)
  [ -f results/program_b/lsf_v1/$1 ] && { echo "$1 exists, skipped $(date)"; return; }
  LSF_VALUE_NAME=$1 LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=$2 LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
    $PY eval/lsf.py sac > results/program_b/lsf_v1/${1%.pt}_train.log 2>&1
  ls results/program_b/lsf_v1/$1 || { echo "$1 missing, stop $(date)"; exit 1; }
  LSF_VALUE_NAME=$1 LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_audit.py > results/jenga/${1%.pt}_audit.log 2>&1
  LSF_VALUE_NAME=$1 LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_diag.py > results/jenga/${1%.pt}_diag.log 2>&1
}
arm s2_2x _st3_s2_2x 2 LSF_VALUE_NAME=value_irr8s2_alea.pt
arm s2_1x _st3_s2_1x 1 LSF_VALUE_NAME=value_irr8s2_alea.pt
echo "TAU3 margin start $(date)"
[ -f results/program_b/lsf_v1/margin_irr_nl8_t105.pt ] || IRRM_NL=8 IRRM_THR_SCALE=1.05 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nl8_t105.log 2>&1
ls results/program_b/lsf_v1/margin_irr_nl8_t105.pt || { echo "margin missing, stop $(date)"; exit 1; }
value value_irr8t105_alea.pt margin_irr_nl8_t105.pt
$PY eval/irrm_gate.py AT105 > results/jenga/irrm_AT105_gate.log 2>&1; echo "TAU3 value graded $(date)"
arm t105_2x _st3_t105_2x 2 LSF_VALUE_NAME=value_irr8t105_alea.pt
echo "SEED3 value start $(date)"
LSF_SEED=20261104 value value_irr8s3_alea.pt margin_irr_nl8.pt
$PY eval/irrm_gate.py AS3 > results/jenga/irrm_AS3_gate.log 2>&1; echo "SEED3 value graded $(date)"
arm s3_2x _st3_s3_2x 2 LSF_VALUE_NAME=value_irr8s3_alea.pt
arm s3_1x _st3_s3_1x 1 LSF_VALUE_NAME=value_irr8s3_alea.pt
for nz in 1 2; do
  R=$(ls -d results/program_b/policy_rollouts/paired_lfc000_st3_*_${nz}x | xargs -n1 basename | tr '\n' ' ')
  $PY cluster/merge_paired.py $R
  AUDIT_OUT=results/jenga/outcome_audit_st3_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs $R \
    > results/jenga/outcome_audit_st3_${nz}x.log 2>&1
  echo "audited ${nz}x $(date)"; tail -n $(echo $R | wc -w) results/jenga/outcome_audit_st3_${nz}x.log
done
$PY eval/seed2tau3_analysis.py > results/jenga/seed2tau3_analysis.log 2>&1; echo "SEED2TAU3 analysed $(date)"
