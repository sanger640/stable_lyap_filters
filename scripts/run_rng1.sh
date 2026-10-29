#!/usr/bin/env bash
# RNG1 (results/jenga/rng1_protocol.json): area / mass undo score with a range, calibrated on undo-vs-hold scores.
# R (offline: calibration + CF3 decisions -> gate R) -> labels -> margin (gate MR) -> value (gate AR) -> dev closed loop
# r20 / r10 at 2x then 1x -> audits -> eval/rng1_dev_analysis.py. One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_rng1.sh > results/jenga/rng1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
echo "R start $(date)"
for w in calib cf3; do
  for i in 0 1; do SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/rng1.py $w > logs/rng1_${w}_$i.out 2>&1 & done
  wait
done
$PY eval/rng1.py grade > results/jenga/rng1_grade.log 2>&1
grep -q '"verdict": "PASS' results/jenga/rng1_result.json || { echo "gate R STOP $(date)"; exit 0; }
echo "labels start $(date)"
for i in 0 1; do IRRM_STAT=rng IRRM_NL=8 SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_rng8_$i.out 2>&1 & done
wait
IRRM_STAT=rng IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_rng8.out 2>&1
ls results/jenga/irrm2_labels_rng8_0.npz results/jenga/irrm2_labels_rng8_1.npz results/jenga/irrm2_held_rng8.npz || { echo "labels missing, stop $(date)"; exit 1; }
echo "margin start $(date)"
IRRM_STAT=rng IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_rng8.log 2>&1
$PY eval/irrm_gate.py MR > results/jenga/irrm_MR_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_MR_gate.json || { echo "MR gate FAIL, stop $(date)"; exit 0; }
echo "value start $(date)"
LSF_VALUE_NAME=value_rng8_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_rng_nl8.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
  $PY eval/lsf.py sac > results/program_b/lsf_v1/value_rng8_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_rng8_alea.pt || { echo "value missing, stop $(date)"; exit 1; }
export LSF_VALUE_NAME=value_rng8_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_AR_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_AR_diag.log 2>&1
$PY eval/irrm_gate.py AR > results/jenga/irrm_AR_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
grep -q '"A_pass": true' results/jenga/irrm_AR_gate.json || { echo "AR gate FAIL, stop $(date)"; exit 0; }
echo "value gated $(date)"
arm() {  # tag gamma noise
  local tag=$1 gam=$2 nz=$3
  echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_rng8_alea.pt LSF_CBF_GAMMA=$gam \
      $PY eval/policy_rollout.py paired lfc000 100 >> logs/rng1${tag}_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000$tag
  AUDIT_OUT=results/jenga/outcome_audit$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000$tag \
    > results/jenga/outcome_audit$tag.log 2>&1
  echo "$tag audited $(date)"; tail -1 results/jenga/outcome_audit$tag.log
}
for nz in 2 1; do
  arm _r20dev${nz}x 0.2 $nz
  arm _r10dev${nz}x 0.1 $nz
done
$PY eval/rng1_dev_analysis.py > results/jenga/rng1_dev_analysis.log 2>&1
echo "RNG1 analysed $(date)"
