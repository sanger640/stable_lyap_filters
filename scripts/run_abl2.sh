#!/usr/bin/env bash
# ABL2 + VIS1 + SEED1 + TAU2 (results/jenga/abl2_protocol.json): the paper's ablations of the 90% filter on episodes 26800-27099 (2x),
# the filter under three new camera looks, a value-seed control for XFER1, and the tau x0.85 value. Waits for the RLAB1 chain.
# usage (repo root): nohup bash scripts/run_abl2.sh <RLAB1 chain pid> > results/jenga/abl2_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
[ "$1" = - ] || while kill -0 "$1" 2>/dev/null; do sleep 300; done   # '-': no wait (GFX1 resumes this chain)
echo "start $(date)"
arm() {  # name tag offset extra-env...  (ours90 settings unless overridden; later env entries win)
  local name=$1 tag=$2 off=$3; shift 3
  [ "$(ls results/program_b/policy_rollouts/paired_lfc000$tag/ep*.json 2>/dev/null | wc -l)" -ge 300 ] && { echo "$name already done, skipped $(date)"; return; }
  echo "$name start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=$off SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_irr8_alea.pt LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=2 "$@" \
      $PY eval/policy_rollout.py paired lfc000 300 >> logs/abl2${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name done $(ls results/program_b/policy_rollouts/paired_lfc000$tag/ep*.json | wc -l) episodes $(date)"
}
audit() {  # out-json script runs...
  local out=$1 scr=$2; shift 2
  $PY cluster/merge_paired.py "$@"
  if [ "$scr" = xfer1 ]; then AUDIT_OUT=results/jenga/$out MUJOCO_GL=egl AUDIT_NOISE_SCALE=2 $PY eval/xfer1_audit.py "$@" > results/jenga/${out%.json}.log 2>&1
  else AUDIT_OUT=results/jenga/$out MUJOCO_GL=egl AUDIT_NOISE_SCALE=2 $PY eval/outcome_audit.py runs "$@" > results/jenga/${out%.json}.log 2>&1; fi
  tail -n $# results/jenga/${out%.json}.log
}
value() {  # value-name margin-name   (value_irr8_alea.pt's recipe in model_OIL; extra env before the call)
  LSF_VALUE_NAME=$1 LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=$2 LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
    $PY eval/lsf.py sac > results/program_b/lsf_v1/${1%.pt}_train.log 2>&1
  ls results/program_b/lsf_v1/$1 || { echo "$1 missing, stop $(date)"; exit 1; }
  LSF_VALUE_NAME=$1 LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_audit.py > results/jenga/${1%.pt}_audit.log 2>&1
  LSF_VALUE_NAME=$1 LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_diag.py > results/jenga/${1%.pt}_diag.log 2>&1
}

# ---- ABL2
A2="o90 b50 b80 lv0 h16 frag nofb act proj det mean soft"
arm o90  _ab2_o90  26800
arm b50  _ab2_b50  26800 LSF_CBF_GAMMA=0.5
arm b80  _ab2_b80  26800 LSF_CBF_GAMMA=0.2
arm lv0  _ab2_lv0  26800 LSF_CBF_GAMMA=1.0
arm h16  _ab2_h16  26800 LSF_VALUE_NAME=value_irr16_alea.pt
arm frag _ab2_frag 26800 LSF_VALUE_NAME=value_lf_alea.pt
arm nofb _ab2_nofb 26800 LSF_NO_FALLBACK=1
arm act  _ab2_act  26800 LSF_EDIT_ALT=actor
arm proj _ab2_proj 26800 LSF_EDIT_ALT=proj
arm det  _ab2_det  26800 LSF_DET=1
arm mean _ab2_mean 26800 LSF_AGG=mean
arm soft _ab2_soft 26800 LSF_AGG=softmin
audit outcome_audit_ab2.json plain $(for a in $A2; do echo -n "paired_lfc000_ab2_$a "; done)
$PY eval/abl2_analysis.py abl2 > results/jenga/abl2_analysis.log 2>&1; echo "ABL2 analysed $(date)"

# ---- VIS1
$PY eval/vis1_offline.py > results/jenga/vis1_offline.log 2>&1; echo "VIS1 offline $(date)"; grep -E "^(none|light|colour|camera) " results/jenga/vis1_offline.log
for lk in light colour camera; do arm vis1_$lk _vis1_$lk 26800 VIS_SHIFT=$lk; done
audit outcome_audit_vis1.json plain paired_lfc000_vis1_light paired_lfc000_vis1_colour paired_lfc000_vis1_camera
$PY eval/abl2_analysis.py vis1 > results/jenga/vis1_analysis.log 2>&1; echo "VIS1 analysed $(date)"

# ---- SEED1
echo "SEED1 value start $(date)"
LSF_SEED=20261031 value value_irr8s2_alea.pt margin_irr_nl8.pt
$PY eval/irrm_gate.py AS2 > results/jenga/irrm_AS2_gate.log 2>&1; echo "SEED1 value graded $(date)"
arm seed1_a2 _seed1_a2 30400 LSF_VALUE_NAME=value_irr8s2_alea.pt MASS_ONLY=0.3
audit outcome_audit_seed1.json xfer1 paired_lfc000_seed1_a2
$PY eval/abl2_analysis.py seed1 > results/jenga/seed1_analysis.log 2>&1; echo "SEED1 analysed $(date)"

# ---- TAU2
echo "TAU2 margin start $(date)"
IRRM_NL=8 IRRM_THR_SCALE=0.85 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nl8_t085.log 2>&1
ls results/program_b/lsf_v1/margin_irr_nl8_t085.pt || { echo "margin missing, stop $(date)"; exit 1; }
value value_irr8t085_alea.pt margin_irr_nl8_t085.pt
$PY eval/irrm_gate.py AT085 > results/jenga/irrm_AT085_gate.log 2>&1; echo "TAU2 value graded $(date)"
arm tau2_t085 _tau2_t085 26800 LSF_VALUE_NAME=value_irr8t085_alea.pt
audit outcome_audit_tau2.json plain paired_lfc000_tau2_t085
$PY eval/abl2_analysis.py tau2 > results/jenga/tau2_analysis.log 2>&1; echo "TAU2 analysed $(date)"
echo "ABL2 chain done $(date)"
