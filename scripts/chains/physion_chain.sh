#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters; P=~/wksp/physion/dd; L=results/physion_check2; mkdir -p $L
until [ -f $P/gen_train.done ] && [ -f $P/gen_test.done ] && [ -f $P/gen_gte.done ]; do sleep 60; done
~/wksp/physion/.venv/bin/python ~/wksp/physion/dd/labels.py > $L/labels.log 2>&1 || { echo "labels FAILED" >> $L/chain.log; exit 1; }
for st in pca encode gte train eval; do
  FAST_TRAIN_Z_ON_CPU=1 python3 eval/physion_check2.py $st > $L/$st.log 2>&1 || { echo "$st FAILED" >> $L/chain.log; exit 1; }
  echo "$st done $(date +%T)" >> $L/chain.log
done
