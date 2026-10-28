#!/bin/bash
# Run the seed-8 SafeManip 1c tasks as slots free up (GPU fits ~8 simulator processes next to the pi0.5 server).
cd ~/wksp/safemanip/SafeManip/eval/models/openpi
for t in StoreLeftoversInBowl LoadDishwasher RecycleBottlesByType PrepareCoffee SetUpCuttingStation DeliverStraw; do
  while [ $(pgrep -fc "eval/safemanip_fork_check") -ge 8 ]; do sleep 30; done
  MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 nohup .venv/bin/python /home/corey/wksp/stable_lyap_filters/eval/safemanip_fork_check.py --tasks $t --episodes 2 --seed 8 --out ~/wksp/safemanip/dd_check1c_s8 > ~/wksp/safemanip/dd_check1c_s8_$t.log 2>&1 &
  sleep 60
done
wait; echo done > ~/wksp/safemanip/dd_check1c.queue_done
