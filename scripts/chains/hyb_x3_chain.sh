#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_overnight.log
until grep -q "scale-up graded" $L; do sleep 120; done
rsync -a eval/ gpu5090:wksp/stable_lyap_filters/eval/
ssh gpu5090 'cd ~/wksp/stable_lyap_filters && . .venv/bin/activate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval/phase2_hybrid.py train_x3 > results/program_b/phase2_hybrid_x3.log 2>&1 && python eval/phase2_hybrid.py grade_x3 >> results/program_b/phase2_hybrid_x3.log 2>&1' < /dev/null
rsync -a gpu5090:wksp/stable_lyap_filters/results/jenga/phase2_hybrid_x3_result.json results/jenga/ && git add results/jenga/phase2_hybrid_x3_result.json && git commit -qm "Phase 2 amendment 12: hybrid on 3x data graded on H2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[$(date '+%F %T')] hybrid x3 graded" >> $L
