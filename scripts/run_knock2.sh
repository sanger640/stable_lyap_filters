#!/usr/bin/env bash
# KNOCK stage 2 (results/jenga/knock_protocol.json stage2): zero-shot closed loop on seeds 45000-45199, policy vs ours90 vs
# frag, model_KNK. Waits for the SEED2/TAU3 chain (one GPU worker pool at a time); arms resumable.
# usage (repo root): nohup bash scripts/run_knock2.sh > results/knock/knock2_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
KNK=$PWD/results/program_b/knock_v1/model/model_KNK.pt
echo "waiting $(date)"
until grep -qE "analysed|stop|Traceback|missing" results/jenga/seed2tau3_chain.log; do sleep 300; done
arm() {  # name mode tag extra-env...
  local name=$1 mode=$2 tag=$3; shift 3
  [ "$(ls results/program_b/policy_rollouts/paired_${mode}${tag}/ep*.json 2>/dev/null | wc -l)" -ge 200 ] && { echo "$name done, skipped"; return; }
  echo "$name start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=45000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=15 CBF_WM=$KNK CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_irr8_alea.pt LSF_CBF_GAMMA=0.1 LSF_ALEA_SCALE=2 IRR_S=2 "$@" \
      $PY eval/knock_rollout.py $mode 200 >> logs/knock2${tag}_$i.out 2>&1 &
  done
  wait
  echo "$name done $(ls results/program_b/policy_rollouts/paired_${mode}${tag}/ep*.json | wc -l) episodes $(date)"
}
arm policy record _knock2
arm ours90 lfc000 _knock2_o90
arm frag lfc000 _knock2_frag LSF_VALUE_NAME=value_lf_alea.pt
$PY eval/knock_analysis.py > results/knock/stage2_analysis.log 2>&1; echo "KNOCK2 analysed $(date)"; tail -40 results/knock/stage2_analysis.log
