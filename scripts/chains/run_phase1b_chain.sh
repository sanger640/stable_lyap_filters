#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b
until grep -q -E "^done|failures|Traceback" $L/phase1b_data.log; do sleep 120; done
grep -q "^done" $L/phase1b_data.log || exit 1
for a in c b d a1 a2 combined; do
  python3 eval/phase1b_models.py $a > $L/phase1b_$a.log 2>&1 || exit 1
done
