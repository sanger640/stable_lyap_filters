#!/bin/bash
# 2a-masked rollouts: 4 tasks x seeds 31, 32 (2 episodes each); at most 8 simulator processes on the GPU at once.
cd ~/wksp/safemanip/SafeManip/eval/models/openpi
for sd in 31 32; do for t in StackBowlsCabinet RecycleBottlesByType LoadDishwasher StoreLeftoversInBowl; do
  while [ $(pgrep -fc "[s]afemanip_(fork_check|video)") -ge 8 ]; do sleep 30; done
  MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 nohup .venv/bin/python /home/corey/wksp/stable_lyap_filters/eval/safemanip_fork_check.py --tasks $t --episodes 2 --seed $sd --masked --out ~/wksp/safemanip/dd_masked_s$sd > ~/wksp/safemanip/dd_masked_s${sd}_$t.log 2>&1 &
  echo $! >> ~/wksp/safemanip/masked_pids.txt; sleep 25
done; done
