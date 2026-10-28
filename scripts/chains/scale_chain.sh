#!/bin/bash
# After the hybrid is graded and the first data pass ends: refill failed renders alone on the GPU, then ship to the 5080 and train.
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_overnight.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f results/jenga/phase2_hybrid_result.json ] && ! pgrep -f "phase2_hybrid.py" >/dev/null && ! pgrep -f "phase2_scale.py data" >/dev/null; do sleep 60; done
for a in 1 2 3; do python3 eval/phase2_scale.py data >> results/program_b/phase2_scale_data.log 2>&1 && break; sleep 30; done
grep -q "data done" results/program_b/phase2_scale_data.log || { echo "[$(date '+%F %T')] scale data incomplete" >> $L; exit 1; }
echo "[$(date '+%F %T')] scale data done" >> $L
rsync -a eval/ gpu5090:wksp/stable_lyap_filters/eval/
rsync -aR results/jenga/phase2_generality_protocol.json gpu5090:wksp/stable_lyap_filters/
rsync -a results/program_b/phase2/x gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/
echo "[$(date '+%F %T')] scale data copied" >> $L
ssh gpu5090 'cd ~/wksp/stable_lyap_filters && . .venv/bin/activate && python eval/phase2_scale.py pack > results/program_b/phase2_scale_pack.log 2>&1 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval/phase2_scale.py train > results/program_b/phase2_scale_train.log 2>&1 && python eval/phase2_scale.py grade > results/program_b/phase2_scale_grade.log 2>&1' < /dev/null
rsync -a gpu5090:wksp/stable_lyap_filters/results/jenga/phase2_scale_result.json results/jenga/ && git add results/jenga/phase2_scale_result.json && git commit -qm "Phase 2 amendment 11: 3x data scale-up graded on H2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[$(date '+%F %T')] scale-up graded" >> $L
