#!/usr/bin/env bash
# XFER1 (results/jenga/xfer1_protocol.json): value transfer under the s2s physics shift. Labels + margin + value in model_S2S's
# imagination -> 4 arms x 300 fresh shifted-physics episodes (30400-30699, 2x) -> shifted-physics audit -> eval/xfer1_analysis.py.
# Waits for the NM1 chain. One GPU pool at a time.  usage (repo root): nohup bash scripts/run_xfer1.sh <nm1 chain pid> > results/jenga/xfer1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
S2S=$PWD/results/program_b/s2s_v1/model/model_S2S.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "labels start $(date)"
for i in 0 1; do IRRM_WM=$S2S IRRM_WM_TAG=s2s IRRM_NL=8 SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_nl8_s2s_$i.out 2>&1 & done
wait
IRRM_WM=$S2S IRRM_WM_TAG=s2s IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_nl8_s2s.out 2>&1
ls results/jenga/irrm2_labels_nl8_s2s_0.npz results/jenga/irrm2_labels_nl8_s2s_1.npz results/jenga/irrm2_held_nl8_s2s.npz || { echo "labels missing, stop $(date)"; exit 1; }
echo "margin start $(date)"
IRRM_WM=$S2S IRRM_WM_TAG=s2s IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nl8_s2s.log 2>&1
ls results/program_b/lsf_v1/margin_irr_nl8_s2s.pt || { echo "margin missing, stop $(date)"; exit 1; }
echo "value start $(date)"
LSF_WM=$S2S LSF_VALUE_NAME=value_irr8_s2s_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_irr_nl8_s2s.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
  $PY eval/lsf.py sac > results/program_b/lsf_v1/value_irr8_s2s_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_irr8_s2s_alea.pt || { echo "value missing, stop $(date)"; exit 1; }
LSF_WM=$S2S LSF_VALUE_NAME=value_irr8_s2s_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_audit.py > results/jenga/xfer1_value_audit.log 2>&1
LSF_WM=$S2S LSF_VALUE_NAME=value_irr8_s2s_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_diag.py > results/jenga/xfer1_value_diag.log 2>&1
echo "value done $(date)"
arm() {  # name mode tag extra-env...
  local name=$1 mode=$2 tag=$3; shift 3
  echo "$name start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=30400 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      MASS_ONLY=0.3 PAIRED_MAX_CHUNKS=120 PYTHONUNBUFFERED=1 "$@" \
      $PY eval/policy_rollout.py paired $mode 300 >> logs/xfer1${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name done $(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json | wc -l) episodes $(date)"
}
arm policy record _x1  CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json
arm A      lfc000 _x1a CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json LSF_VALUE_NAME=value_irr8_alea.pt     LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=2
arm B      lfc000 _x1b CBF_WM=$S2S CBF_CALIB=results/jenga/cbf_calibration_S2S.json LSF_VALUE_NAME=value_irr8_alea.pt     LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=2
arm C      lfc000 _x1c CBF_WM=$S2S CBF_CALIB=results/jenga/cbf_calibration_S2S.json LSF_VALUE_NAME=value_irr8_s2s_alea.pt LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=2
RUNS="paired_record_x1 paired_lfc000_x1a paired_lfc000_x1b paired_lfc000_x1c"
$PY cluster/merge_paired.py $RUNS
AUDIT_OUT=results/jenga/outcome_audit_x1.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=2 $PY eval/xfer1_audit.py $RUNS > results/jenga/outcome_audit_x1.log 2>&1
echo "audited $(date)"
$PY eval/xfer1_analysis.py > results/jenga/xfer1_analysis.log 2>&1
echo "XFER1 analysed $(date)"
