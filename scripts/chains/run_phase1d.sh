#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
for a in P R PR; do
  python3 eval/phase1d_models.py $a > results/program_b/phase1d_$a.log 2>&1 || exit 1
done
