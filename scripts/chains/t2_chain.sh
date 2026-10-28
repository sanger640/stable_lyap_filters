#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=results/program_b/t2_run.log
for a in 1 2 3; do python3 eval/t2_data.py train_data >> $L 2>&1 && break; sleep 30; done
for a in 1 2 3; do python3 eval/t2_data.py h >> $L 2>&1 && break; sleep 30; done
while pgrep -f "t1_filter.py" >/dev/null; do sleep 60; done
python3 eval/t2_data.py model >> $L 2>&1 && python3 eval/t2_data.py grade >> $L 2>&1 && git add results/jenga/t2_h_result.json && git commit -qm "T2: US monitor trained and graded on H_T2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
