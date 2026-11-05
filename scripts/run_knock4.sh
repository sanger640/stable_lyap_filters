#!/usr/bin/env bash
# KNOCK stage 4 (results/jenga/knock_protocol.json stage4): the Jenga-trained irreversible-fragility value (value_nsif16_alea.pt,
# ACAL1's matched barrier gamma 0.218 at 2x) zero-shot with model_KNK, on stage 3's 200 paired episodes. Resumable.
# usage (repo root): nohup bash scripts/run_knock4.sh > results/knock/knock4_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
KNK=$PWD/results/program_b/knock_v1/model/model_KNK.pt
TAG=_knock4_zfx
echo "start $(date)"
if [ "$(ls results/program_b/policy_rollouts/paired_lfc000$TAG/ep*.json 2>/dev/null | wc -l)" -lt 200 ]; then
  for i in 0 1 2 3 4 5; do
    env -u LSF_OUT MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$TAG EP_OFFSET=45000 SHARD=$i/6 EXEC_NOISE_SCALE=2 HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=15 CBF_WM=$KNK LSF_FILTER_WM=$KNK CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_nsif16_alea.pt LSF_CBF_GAMMA=0.218 LSF_ALEA_SCALE=2 IRR_S=2 \
      $PY eval/knock_rollout.py lfc000 200 >> logs/knock4${TAG}_$i.out 2>&1 &
  done
  wait
fi
echo "z_fx done $(ls results/program_b/policy_rollouts/paired_lfc000$TAG/ep*.json | wc -l) episodes $(date)"
$PY eval/knock_analysis.py stage4 > results/knock/stage4_analysis.log 2>&1; echo "KNOCK4 analysed $(date)"; tail -45 results/knock/stage4_analysis.log
