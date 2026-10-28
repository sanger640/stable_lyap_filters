#!/bin/bash
# Phase 2 chain: H2 screen (CPU, now) -> wait for training-data render -> H2 render -> train US2 -> grade on H2
cd /home/corey/wksp/stable_lyap_filters
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=results/program_b/phase2_models.log
python3 eval/phase2_models.py h2screen --screen-workers 12 >> $L 2>&1
until grep -q "^done" results/program_b/phase2_data.log; do sleep 60; done
for a in 1 2 3; do python3 eval/phase2_models.py h2 >> $L 2>&1 && break; sleep 20; done
python3 eval/phase2_models.py train >> $L 2>&1 && python3 eval/phase2_models.py grade >> $L 2>&1
