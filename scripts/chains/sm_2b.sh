#!/bin/bash
cd ~/wksp/safemanip/SafeManip/eval/models/openpi
: > ~/wksp/safemanip/sm2b_pids.txt
for sd in 41 42; do for t in StackBowlsCabinet RecycleBottlesByType; do for rep in a b; do
  while [ $(pgrep -fc "[s]afemanip_fork_check") -ge 6 ]; do sleep 30; done
  MUJOCO_GL=egl NUMBA_DISABLE_JIT=1 nohup .venv/bin/python /home/corey/wksp/stable_lyap_filters/eval/safemanip_fork_check.py --tasks $t --episodes 1 --seed $sd$([ $rep = b ] && echo 0) --masked --n-fut 33 --out ~/wksp/safemanip/dd_2b_s$sd$([ $rep = b ] && echo 0) > ~/wksp/safemanip/dd_2b_${sd}${rep}_$t.log 2>&1 &
  echo $! >> ~/wksp/safemanip/sm2b_pids.txt; sleep 40
done; done; done
touch ~/wksp/safemanip/sm2b_launched
