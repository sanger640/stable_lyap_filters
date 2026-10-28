#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters; P=~/wksp/physion/dd; L=results/physion_check2b
until [ -f $P/gen_train2.done ]; do sleep 60; done
export CHECK2_TAG=b FAST_TRAIN_Z_ON_CPU=1
for st in encode train eval; do
  python3 eval/physion_check2.py $st > $L/$st.log 2>&1 || { echo "$st FAILED" >> $L/chain.log; exit 1; }
  echo "$st done $(date +%T)" >> $L/chain.log
done
