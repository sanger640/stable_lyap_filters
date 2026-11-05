#!/usr/bin/env bash
# KNOCK stage 3 (results/jenga/knock_protocol.json stage3): values trained on the knock-down task's label-free data.
# Prep runs as single GPU processes beside the SEED2/TAU3 chain; values and closed loop wait for it (one pool at a time).
# Every step is resumable (skipped when its output exists).
# usage (repo root): nohup bash scripts/run_knock3.sh > results/knock/knock3_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
KNK=$PWD/results/program_b/knock_v1/model/model_KNK.pt
export LSF_OUT=$PWD/results/program_b/lsf_knock
mkdir -p $LSF_OUT
echo "start $(date)"
[ -f $LSF_OUT/train_reset_z.npy ] || LSF_SOURCES=knock_v1 $PY eval/lsf.py data > $LSF_OUT/data.log 2>&1
[ -f $LSF_OUT/train_reset_plan.npy ] || $PY eval/knock_values.py plans >> $LSF_OUT/data.log 2>&1
ls $LSF_OUT/train_reset_plan.npy || { echo "resets missing, stop $(date)"; exit 1; }
echo "resets $(date)"; tail -3 $LSF_OUT/data.log
HJ_RUNS=paired_record_knock_wm HJ_SCALES=4 HJ_HOLD_GRIP=1 CBF_WM=$KNK PYTHONUNBUFFERED=1 $PY eval/hj_data2.py > results/knock/hj_data_knk.log 2>&1
echo "fork statistics $(ls results/jenga/hj2/data/paired_record_knock_wm/ep*.npz | wc -l) episodes $(date)"
IRR="IRRM_NL=8 IRRM_MASK=none IRRM_WM=$KNK IRRM_WM_TAG=knk IRRM_HELD_RUN=paired_record_knock_wm IRRM_HELD_MIN_EP=44130 PYTHONUNBUFFERED=1"
[ -f results/jenga/irrm2_labels_nm8_knk_0.npz ] || env $IRR $PY eval/irrm2.py labels > results/knock/irrm2_labels_knk.log 2>&1
[ -f results/jenga/irrm2_held_nm8_knk.npz ] || env $IRR $PY eval/irrm2.py held > results/knock/irrm2_held_knk.log 2>&1
# amendment 1: Jenga's margin update counts (epochs x Jenga labels / knock labels: 40 x 59690 / 8000, 30 x 69247 / 1950)
[ -f $LSF_OUT/margin_irrnm_nl8_knk.pt ] || env $IRR IRRM_EPOCHS=298 $PY eval/irrm2.py margin > results/knock/irrm2_margin_knk.log 2>&1
ls $LSF_OUT/margin_irrnm_nl8_knk.pt || { echo "irr margin missing, stop $(date)"; exit 1; }
[ -f $LSF_OUT/margin_lf.pt ] || LF_EPOCHS=1065 LF_RUN=paired_record_knock_wm LF_SPLIT_EP=44130 LF_RESULT=results/knock/lflsf_LF1_knk.json PYTHONUNBUFFERED=1 \
  $PY eval/lflsf.py margin > results/knock/lflsf_margin_knk.log 2>&1
ls $LSF_OUT/margin_lf.pt || { echo "fork margin missing, stop $(date)"; exit 1; }
echo "margins $(date)"; grep -E '"held_(spearman|auc)' results/jenga/irrm2_result_nm8_knk.json results/knock/lflsf_LF1_knk.json
FIT=""   # M3 (amendment 1): margins whose in-sample recall reaches half the Jenga margin's
for m in irr frag; do
  LF_RUN=paired_record_knock_wm LF_SPLIT_EP=44130 $PY eval/knock_values.py fit $m >> results/knock/stage3_margin_fit.log 2>&1 && FIT="$FIT $m"
  tail -1 results/knock/stage3_margin_fit.log
done
echo "M3 passed:$FIT $(date)"
echo "waiting for the SEED2/TAU3 chain $(date)"
until grep -qE "analysed|stop|Traceback|missing" results/jenga/seed2tau3_chain.log; do sleep 300; done
run() {  # tag offset n chunks value-path   (stage 2's ours90 settings, the filter imagining with model_KNK)
  local tag=$1 off=$2 n=$3 ch=$4 val=$5 sh=$6
  for i in $(seq 0 $((sh - 1))); do
    env -u LSF_OUT MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=$off SHARD=$i/$sh EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=$ch CBF_WM=$KNK LSF_FILTER_WM=$KNK CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=$val LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=2 IRR_S=2 \
      $PY eval/knock_rollout.py lfc000 $n >> logs/knock3${tag}_$i.out 2>&1 &
  done
  wait
}
arm() {  # name tag value-path
  [ "$(ls results/program_b/policy_rollouts/paired_lfc000$2/ep*.json 2>/dev/null | wc -l)" -ge 200 ] && { echo "$1 done, skipped"; return; }
  echo "$1 start $(date)"; run $2 45000 200 15 $3 6
  echo "$1 done $(ls results/program_b/policy_rollouts/paired_lfc000$2/ep*.json | wc -l) episodes $(date)"
}
value() {  # name file extra-env...
  local name=$1 f=$2; shift 2
  [ -f $LSF_OUT/$f ] && { echo "$f exists, skipped"; return; }
  echo "$name value start $(date)"
  env LSF_WM=$KNK LSF_VALUE_NAME=$f LSF_ALEA=8 PYTHONUNBUFFERED=1 "$@" $PY eval/lsf.py sac > $LSF_OUT/${f%.pt}_train.log 2>&1
  ls $LSF_OUT/$f || { echo "$f missing, stop $(date)"; exit 1; }
  echo "$name value done $(date)"
}
gated() {  # name tag value-path
  [ -f results/program_b/policy_rollouts/paired_lfc000_knock3g_$1/ep44149.json ] || run _knock3g_$1 44130 20 1 $3 2
  if $PY eval/knock_values.py gate $1 >> results/knock/stage3_gate.log 2>&1; then echo "$1 G3 pass $(date)"; arm $1 $2 $3
  else echo "$1 G3 FAIL, arm not run $(date)"; fi; tail -1 results/knock/stage3_gate.log
}
arm z_o90 _knock3_zo90 value_irr8_alea.pt
arm z_frag _knock3_zfrag value_lf_alea.pt
if [[ "$FIT" == *frag* ]]; then
  value k_frag value_lf_knk.pt LSF_CLF=margin_lf.pt
  gated k_frag _knock3_kfrag $LSF_OUT/value_lf_knk.pt
else echo "k_frag not run (M3 fail)"; fi
if [[ "$FIT" == *irr* ]]; then
  value k_o90 value_irrnm8_knk.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_irrnm_nl8_knk.pt LSF_MARGIN_MODE=irr
  gated k_o90 _knock3_ko90 $LSF_OUT/value_irrnm8_knk.pt
else echo "k_o90 not run (M3 fail)"; fi
$PY eval/knock_analysis.py stage3 > results/knock/stage3_analysis.log 2>&1; echo "KNOCK3 analysed $(date)"; tail -60 results/knock/stage3_analysis.log
