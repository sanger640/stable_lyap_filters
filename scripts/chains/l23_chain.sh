#!/bin/bash
# After L2 collection: encode all, arm links, train base model, 4 ensemble members, L3 offline check.
cd /home/corey/wksp/stable_lyap_filters
LOG=results/libero/wm/chain.log
while pgrep -f "l2_run.sh|l2_long.sh" >/dev/null; do python3 libero/wm_libero.py encode >> results/libero/wm/encode.log 2>&1; sleep 300; done
sleep 130
python3 libero/wm_libero.py encode >> results/libero/wm/encode.log 2>&1; echo "encode done $(date) $(ls results/libero/wm/enc/*.npz | wc -l)" >> $LOG
( source ~/wksp/safelibero/vlsa-aegis/main/.venv/bin/activate
  export LIBERO_CONFIG_PATH=$HOME/wksp/safelibero/libero_config PYTHONPATH=$HOME/wksp/safelibero/vlsa-aegis/safelibero:$PWD/libero MUJOCO_GL=egl
  for i in 0 1 2 3 4 5; do python libero/arm_links.py --shard $i 6 > results/libero/arm_$i.log 2>&1 & done; wait; echo "arm done $(date) $(ls results/libero/arm/*.npy | wc -l)" >> $LOG ) &
ARM=$!
echo "arm done $(date) $(ls results/libero/arm/*.npy | wc -l)" >> $LOG
export FAST_TRAIN_Z_ON_CPU=1
python3 libero/wm_libero.py train > results/libero/wm/train.log 2>&1 || { echo "train FAILED" >> $LOG; exit 1; }
echo "train done $(date)" >> $LOG
# ensemble deferred (user, 2026-10-19)

wait $ARM
python3 libero/wm_libero.py calib > results/libero/wm/calib.log 2>&1 || echo "calib FAILED" >> $LOG
python3 libero/wm_libero.py l3 > results/libero/wm/l3.log 2>&1 || echo "l3 FAILED" >> $LOG
echo "l3 done $(date)" >> $LOG
