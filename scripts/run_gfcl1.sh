#!/usr/bin/env bash
# GFCL1 (results/jenga/gfcl1_protocol.json): nm90 (no-mask value) and ns90 (0.8 s irreversible-fragility value) at the 90% barrier
# on H2H5's episodes 26200-26799, 2x then 1x, as H2H5's ours90 with only the value swapped; audits, analysis, then the 2x videos.
# Waits for the XFER1 chain (one GPU pool at a time).
# usage (repo root): nohup bash scripts/run_gfcl1.sh <XFER1 chain pid> > results/jenga/gfcl1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
arm() {  # name noise(1|2) tag value
  local name=$1 nz=$2 tag=$3 value=$4
  echo "$name ${nz}x start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=26200 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=$value LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=$nz \
      $PY eval/policy_rollout.py paired lfc000 600 >> logs/gfcl1${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name ${nz}x done $(ls results/program_b/policy_rollouts/paired_lfc000$tag/ep*.json | wc -l) episodes $(date)"
  $PY cluster/merge_paired.py paired_lfc000$tag
}
for nz in 2 1; do
  arm nm90 $nz _h5nm90_${nz}x value_irrnm8_alea.pt
  arm ns90 $nz _h5ns90_${nz}x value_nsi8_alea.pt
  for t in nm ns; do
    AUDIT_OUT=results/jenga/outcome_audit_h5${t}_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz \
      $PY eval/outcome_audit.py runs paired_lfc000_h5${t}90_${nz}x > results/jenga/outcome_audit_h5${t}_${nz}x.log 2>&1
    tail -1 results/jenga/outcome_audit_h5${t}_${nz}x.log
  done
  echo "${nz}x audited $(date)"
done
$PY eval/gfcl1_analysis.py > results/jenga/gfcl1_analysis.log 2>&1
echo "GFCL1 analysed $(date)"
VID_LAYOUT=row NO_SLICE=1 NPROC=4 nice -n 10 $PY eval/filter_videos.py H5N > results/program_b/filter_videos/h5n_videos.log 2>&1
echo "GFCL1 videos $(date)"
