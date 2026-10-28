#!/bin/bash
cd ~/wksp/safemanip/SafeManip/eval/models/openpi
for t in LoadDishwasher StoreLeftoversInBowl; do
  while [ $(pgrep -fc "[s]afemanip_(fork_check|video)") -ge 6 ]; do sleep 30; done
  MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 nohup .venv/bin/python /home/corey/wksp/stable_lyap_filters/eval/safemanip_fork_check.py --tasks $t --episodes 2 --seed 32 --masked --out ~/wksp/safemanip/dd_masked_s32 > ~/wksp/safemanip/dd_masked_s32_$t.log 2>&1 &
  echo $! >> ~/wksp/safemanip/masked_pids.txt; sleep 60
done
