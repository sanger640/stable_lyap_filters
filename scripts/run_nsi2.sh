#!/usr/bin/env bash
# NSI2 (results/jenga/nsi2_protocol.json): NSI target on RNG1's labels -> margin (MN2) -> value (AN2) -> dev n20 / n10 at
# 2x then 1x -> audits -> eval/nsi2_dev_analysis.py. Waits for the RNG1 chain. One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_nsi2.sh <rng1 chain pid> > results/jenga/nsi2_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
ls results/jenga/irrm2_labels_rng8_0.npz results/jenga/irrm2_labels_rng8_1.npz results/jenga/irrm2_held_rng8.npz || { echo "RNG1 labels missing, NSI2 not run $(date)"; exit 0; }
echo "margin start $(date)"
IRRM_STAT=rng IRRM_NSI=1 IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_rngnsi8.log 2>&1
$PY eval/irrm_gate.py MN2 > results/jenga/irrm_MN2_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_MN2_gate.json || { echo "MN2 gate FAIL, stop $(date)"; exit 0; }
echo "value start $(date)"
LSF_VALUE_NAME=value_rngnsi8_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_rngnsi_nl8.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
  $PY eval/lsf.py sac > results/program_b/lsf_v1/value_rngnsi8_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_rngnsi8_alea.pt || { echo "value missing, stop $(date)"; exit 1; }
export LSF_VALUE_NAME=value_rngnsi8_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_AN2_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_AN2_diag.log 2>&1
$PY eval/irrm_gate.py AN2 > results/jenga/irrm_AN2_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
grep -q '"A_pass": true' results/jenga/irrm_AN2_gate.json || { echo "AN2 gate FAIL, stop $(date)"; exit 0; }
echo "value gated $(date)"
arm() {  # tag gamma noise
  local tag=$1 gam=$2 nz=$3
  echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_rngnsi8_alea.pt LSF_CBF_GAMMA=$gam \
      $PY eval/policy_rollout.py paired lfc000 100 >> logs/nsi2${tag}_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000$tag
  AUDIT_OUT=results/jenga/outcome_audit$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000$tag \
    > results/jenga/outcome_audit$tag.log 2>&1
  echo "$tag audited $(date)"; tail -1 results/jenga/outcome_audit$tag.log
}
for nz in 2 1; do
  arm _n20dev${nz}x 0.2 $nz
  arm _n10dev${nz}x 0.1 $nz
done
$PY eval/nsi2_dev_analysis.py > results/jenga/nsi2_dev_analysis.log 2>&1
echo "NSI2 analysed $(date)"
