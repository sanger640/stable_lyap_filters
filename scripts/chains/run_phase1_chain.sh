#!/bin/bash
# Sequential Phase 1: wait for data, then one GPU job at a time.
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b
until grep -q -E "^done|failures|Traceback" $L/phase1_data.log; do sleep 120; done
grep -q "^done" $L/phase1_data.log || exit 1
for c in A B C D E; do
  python3 eval/phase1_models.py train --cell $c > $L/phase1_train_$c.log 2>&1 || exit 1
done
python3 eval/phase1_models.py select > $L/phase1_select.log 2>&1 || exit 1
python3 eval/phase1_models.py train_f > $L/phase1_train_F.log 2>&1
