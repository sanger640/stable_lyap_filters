#!/usr/bin/env bash
# GFX1 (results/jenga/gfx1_protocol.json): G0 gripper check -> F margins (flat, chance) -> F values -> gates -> dev barrier arms
# (nm, ff, fq x gamma 0.5 / 0.2 / 0.1 x 2x, 1x) -> selection -> fresh arms on H2H5's episodes -> analysis -> resume the ABL2 chain.
# Waits for the given PIDs (ABL2's act workers). One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_gfx1.sh <pid> [pid ...] > results/jenga/gfx1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done
echo "start $(date)"

# ---- G0 (offline gripper check)
for i in 0 1; do SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/gfx1_grip.py run > logs/gfx1_grip_$i.out 2>&1 & done
wait
$PY eval/gfx1_grip.py grade > results/jenga/gfx1_grip.log 2>&1; echo "G0 graded $(date)"; grep -A12 '"gate_G"' results/jenga/gfx1_grip.json

# ---- F margins + values
for mode in flat chance; do
  IRRM_NSI=1 IRRM_NL=16 IRRM_NSI_MODE=$mode PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_${mode}16.log 2>&1
done
ls results/program_b/lsf_v1/margin_nsif_nl16.pt results/program_b/lsf_v1/margin_nsiq_nl16.pt || { echo "margin missing, stop $(date)"; exit 1; }
$PY eval/irrm_gate.py MNF > results/jenga/irrm_MNF_gate.log 2>&1; $PY eval/irrm_gate.py MNQ > results/jenga/irrm_MNQ_gate.log 2>&1
echo "F margins $(date)"
for pair in "nsif16 nsif MNF ANF" "nsiq16 nsiq MNQ ANQ"; do
  set -- $pair; v=value_$1_alea.pt
  echo "$v start $(date)"
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_$2_nl16.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
    $PY eval/lsf.py sac > results/program_b/lsf_v1/value_$1_alea_train.log 2>&1
  ls -la results/program_b/lsf_v1/$v || { echo "$v missing, stop $(date)"; exit 1; }
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_audit.py > results/jenga/irrm_$4_audit.log 2>&1
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_diag.py > results/jenga/irrm_$4_diag.log 2>&1
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/irrm_gate.py $4 > results/jenga/irrm_$4_gate.log 2>&1
  echo "$v graded $(date)"
done

# ---- S: dev barrier arms
dev() {  # tag value gamma noise
  local tag=$1 value=$2 gam=$3 nz=$4
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=$value LSF_CBF_GAMMA=$gam \
      $PY eval/policy_rollout.py paired lfc000 100 >> logs/gfx1_${tag}_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000_$tag
  AUDIT_OUT=results/jenga/outcome_audit_$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000_$tag \
    > results/jenga/outcome_audit_$tag.log 2>&1
  echo "$tag $(date) $(tail -1 results/jenga/outcome_audit_$tag.log)"
}
for v in "nm value_irrnm8_alea.pt" "ff value_nsif16_alea.pt" "fq value_nsiq16_alea.pt"; do
  set -- $v
  for g in "50 0.5" "20 0.2" "10 0.1"; do
    set -- $v $g
    for nz in 2 1; do dev $1$3dev${nz}x $2 $4 $nz; done
  done
done
$PY eval/gfx1_analysis.py select > results/jenga/gfx1_select.log 2>&1; echo "selected $(date)"; cat results/jenga/gfx1_select.log | grep -E "^(F_|NM_)"
eval "$(grep -E '^(F_TAG|F_VALUE|F_GAMMA|NM_GAMMA)=' results/jenga/gfx1_select.log)"

# ---- T: fresh arms on H2H5's episodes
fresh() {  # tag value gamma noise
  local tag=$1 value=$2 gam=$3 nz=$4
  echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_$tag EP_OFFSET=26200 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=$value LSF_CBF_GAMMA=$gam LSF_ALEA_SCALE=$nz \
      $PY eval/policy_rollout.py paired lfc000 600 >> logs/gfx1_${tag}_$i.out 2>&1 &
  done
  wait
  echo "$tag done $(ls results/program_b/policy_rollouts/paired_lfc000_$tag/ep*.json | wc -l) episodes $(date)"
  $PY cluster/merge_paired.py paired_lfc000_$tag
}
if [ -z "$F_TAG" ]; then echo "no F arm eligible, no fresh test $(date)"; else
  for nz in 2 1; do
    fresh h5fx_${nz}x $F_VALUE $F_GAMMA $nz
    [ -n "$NM_GAMMA" ] && fresh h5nmx_${nz}x value_irrnm8_alea.pt $NM_GAMMA $nz
    for t in fx $([ -n "$NM_GAMMA" ] && echo nmx); do
      AUDIT_OUT=results/jenga/outcome_audit_h5${t}_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz \
        $PY eval/outcome_audit.py runs paired_lfc000_h5${t}_${nz}x > results/jenga/outcome_audit_h5${t}_${nz}x.log 2>&1
      tail -1 results/jenga/outcome_audit_h5${t}_${nz}x.log
    done
    echo "${nz}x audited $(date)"
  done
  $PY eval/gfx1_analysis.py test > results/jenga/gfx1_analysis.log 2>&1; echo "GFX1 analysed $(date)"
fi

# ---- resume the paused ABL2 chain (finished arms are skipped)
echo "ABL2 resume $(date)"
bash scripts/run_abl2.sh - >> results/jenga/abl2_chain.log 2>&1
echo "GFX1 chain done $(date)"
