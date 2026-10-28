#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
LOG=results/libero/wm/chain.log
( source ~/wksp/safelibero/vlsa-aegis/main/.venv/bin/activate
  export LIBERO_CONFIG_PATH=$HOME/wksp/safelibero/libero_config PYTHONPATH=$HOME/wksp/safelibero/vlsa-aegis/safelibero:$PWD/libero MUJOCO_GL=egl
  for i in $(seq 0 9); do python libero/arm_links.py --shard $i 10 > results/libero/arm2_$i.log 2>&1 & done; wait )
echo "arm2 done $(date) $(ls results/libero/arm/*.npy | wc -l)" >> $LOG
python3 libero/wm_libero.py calib > results/libero/wm/calib.log 2>&1 || echo "calib FAILED" >> $LOG
python3 libero/wm_libero.py l3 > results/libero/wm/l3.log 2>&1 || echo "l3 FAILED" >> $LOG
echo "l3 done $(date)" >> $LOG
