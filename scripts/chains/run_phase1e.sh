#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
for a in U US R_fast; do
  python3 eval/phase1e_models.py $a > results/program_b/phase1e_$a.log 2>&1 || exit 1
done
