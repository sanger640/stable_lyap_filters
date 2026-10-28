#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
while pgrep -f run_phase1b_chain.sh >/dev/null; do sleep 120; done
for a in conv3 vit3; do
  python3 eval/phase1c_models.py $a > results/program_b/phase1c_$a.log 2>&1 || exit 1
done
