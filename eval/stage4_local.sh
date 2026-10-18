#!/bin/bash
# Stage 4 locally (results/jenga/stage4_protocol.json): W workers (one pool), each loops over blocks of 25 episodes and
# runs all 4 arms for its shard of the block, so partial results stay paired. Resumable (per-episode files).
# usage: bash eval/stage4_local.sh W
W=${1:-5}
cd /home/corey/wksp/stable_lyap_filters
export MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_s4 EXEC_NOISE_SCALE=2 HARD_OFF=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export CBF_CALIB=results/jenga/cbf_calibration_OIL.json
mkdir -p logs
for w in $(seq 0 $((W - 1))); do
  (for blk in $(seq 0 11); do
     for arm in record cbfmsoil cbfh cbfg; do
       SHARD=$w/$W EP_OFFSET=$((14000 + 25 * blk)) /home/corey/wksp/dp_venv/bin/python eval/policy_rollout.py paired $arm 25
     done
   done) > logs/stage4_w$w.log 2>&1 &
done
wait
