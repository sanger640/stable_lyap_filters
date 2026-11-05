#!/usr/bin/env bash
# KNOCK stage 5 + HZN1 pool chain (results/jenga/knock_protocol.json stage5, results/jenga/hzn1_protocol.json). One worker pool
# at a time; values come from scripts/run_train_k5.sh (single processes, run alongside). Arms are resumable.
# usage (repo root): nohup bash scripts/run_knock5_hzn1.sh > results/knock/knock5_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
KNK=$PWD/results/program_b/knock_v1/model/model_KNK.pt
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
V1=results/program_b/lsf_v1
LAB=$PWD/results/program_b/lsf_v3_red/value_unisafe_alea.pt
echo "start $(date)"
gam() {  # key value -> prints "G2 G1" (ACAL1's label-free rule)
  [ -f results/jenga/acalx_$1.json ] || MUJOCO_GL=egl PYTHONUNBUFFERED=1 $PY eval/acal_extra.py $1 $2 > results/jenga/acalx_$1.log 2>&1
  $PY -c "import json; s=json.load(open('results/jenga/acalx_$1.json'))['scales']; print(s['2x']['gamma'], s['1x']['gamma'])"
}
knock() {  # tag mode extra-env...   (stage 4's z_fx settings unless overridden)
  local tag=$1 mode=$2; shift 2
  [ "$(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json 2>/dev/null | wc -l)" -ge 200 ] && { echo "$tag already done $(date)"; return; }
  echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env -u LSF_OUT MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=45000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=15 CBF_WM=$KNK LSF_FILTER_WM=$KNK CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_nsif16_alea.pt LSF_CBF_GAMMA=0.218 LSF_ALEA_SCALE=2 IRR_S=2 "$@" \
      $PY eval/knock_rollout.py $mode 200 >> logs/knock5${tag}_$i.out 2>&1 &
  done
  wait
  echo "$tag done $(ls results/program_b/policy_rollouts/paired_$mode$tag/ep*.json | wc -l) episodes $(date)"
}
jenga() {  # tag value gamma noise extra-env...   (ACAL1's fresh() settings)
  local tag=$1 value=$2 g=$3 nz=$4; shift 4
  if [ "$(ls results/program_b/policy_rollouts/paired_lfc000_$tag/ep*.json 2>/dev/null | wc -l)" -lt 600 ]; then
    echo "$tag start $(date)"
    for i in 0 1 2 3 4 5; do
      env -u LSF_OUT MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_$tag EP_OFFSET=27400 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
        PAIRED_MAX_CHUNKS=120 CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
        LSF_VALUE_NAME=$value LSF_CBF_GAMMA=$g LSF_ALEA_SCALE=$nz "$@" \
        $PY eval/policy_rollout.py paired lfc000 600 >> logs/hzn1_${tag}_$i.out 2>&1 &
    done
    wait
    echo "$tag done $(ls results/program_b/policy_rollouts/paired_lfc000_$tag/ep*.json | wc -l) episodes $(date)"
  fi
  [ -f results/jenga/outcome_audit_$tag.json ] && return
  $PY cluster/merge_paired.py paired_lfc000_$tag
  AUDIT_OUT=results/jenga/outcome_audit_$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz \
    $PY eval/outcome_audit.py runs paired_lfc000_$tag > results/jenga/outcome_audit_$tag.log 2>&1
  echo "$tag audited $(date) $(tail -1 results/jenga/outcome_audit_$tag.log)"
}
waitfor() {  # file [skip-marker]
  while [ ! -f "$1" ]; do [ -n "$2" ] && [ -f "$2" ] && return 1; sleep 120; done; sleep 30; return 0
}

# ---- KNOCK stage 5: labelled baselines and no fallback
read LG2 LG1 <<< "$(gam lcbf $LAB)"; echo "lcbf gamma 2x $LG2 $(date)"
knock _knock5_unir lsfa030 LSF_OUT=$PWD/results/program_b/lsf_v3_red   # env -u runs before the assignments
knock _knock5_uni lsfa030
knock _knock5_lcbf lfc000 LSF_VALUE_NAME=$LAB LSF_CBF_GAMMA=$LG2
knock _knock5_zfxnf lfc000 LSF_NO_FALLBACK=1
$PY eval/knock_analysis.py stage5 > results/knock/stage5_analysis.log 2>&1; grep -E "^H_" results/knock/stage5_analysis.log
NOFB=$(grep -oP '^H_nofb=\K\S+' results/knock/stage5_analysis.log)

# ---- HZN1 Jenga: plain 1.6 s irreversibility at its matched barrier; fx without the fallback if H_nofb passed
for nz in 2 1; do
  jenga hz1i16_${nz}x value_irr16_alea.pt $([ $nz = 2 ] && echo 0.207 || echo 0.259) $nz
done
if [ "$NOFB" = "1" ]; then
  for nz in 2 1; do jenga hz1fxnf_${nz}x value_nsif16_alea.pt $([ $nz = 2 ] && echo 0.218 || echo 0.216) $nz LSF_NO_FALLBACK=1; done
else
  echo "H_nofb=$NOFB: fxnf Jenga arms not run $(date)"
fi
$PY eval/hzn1_analysis.py > results/jenga/hzn1_analysis.log 2>&1; grep -E "^(H_|J_)" results/jenga/hzn1_analysis.log

# ---- KNOCK stage 5: value seed 2
waitfor $V1/value_nsif16s2_alea.pt && { read FG2 FG1 <<< "$(gam fx2 value_nsif16s2_alea.pt)"; echo "fx2 gamma 2x $FG2 $(date)"
  knock _knock5_zfx2 lfc000 LSF_VALUE_NAME=value_nsif16s2_alea.pt LSF_CBF_GAMMA=$FG2; }

# ---- irreversible fragility at 0.8 s: knock-down and Jenga (only if the margin gate passed)
if waitfor $V1/value_nsif8_alea.pt results/jenga/hzn1_f8_skipped; then
  read EG2 EG1 <<< "$(gam f8 value_nsif8_alea.pt)"; echo "f8 gamma 2x $EG2 1x $EG1 $(date)"
  knock _knock5_zf8 lfc000 LSF_VALUE_NAME=value_nsif8_alea.pt LSF_CBF_GAMMA=$EG2
  for nz in 2 1; do jenga hz1f8_${nz}x value_nsif8_alea.pt $([ $nz = 2 ] && echo $EG2 || echo $EG1) $nz; done
else
  echo "f8 skipped (margin gate) $(date)"
fi
$PY eval/knock_analysis.py stage5 > results/knock/stage5_analysis.log 2>&1; echo "KNOCK5 analysed $(date)"; grep -E "^H_" results/knock/stage5_analysis.log
$PY eval/hzn1_analysis.py > results/jenga/hzn1_analysis.log 2>&1; echo "HZN1 analysed $(date)"; grep -E "^(H_|J_)" results/jenga/hzn1_analysis.log
echo "done $(date)"
