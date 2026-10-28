#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
until grep -q copied results/program_b/joint_copy.log 2>/dev/null; do sleep 60; done
rsync -a eval/ gpu5090:wksp/stable_lyap_filters/eval/
rsync -aR results/jenga/phase2_generality_protocol.json gpu5090:wksp/stable_lyap_filters/
ssh gpu5090 'cd ~/wksp/stable_lyap_filters && . .venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python eval/joint_hybrid.py pack > results/program_b/joint.log 2>&1 && python eval/joint_hybrid.py det >> results/program_b/joint.log 2>&1 && python eval/joint_hybrid.py head >> results/program_b/joint.log 2>&1 && python eval/joint_hybrid.py grade >> results/program_b/joint.log 2>&1' < /dev/null
rsync -a gpu5090:wksp/stable_lyap_filters/results/jenga/joint_hybrid_result.json results/jenga/ && git add results/jenga/joint_hybrid_result.json && git commit -qm "Amendment 13: joint three-task model graded on all three selection sets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "[$(date '+%F %T')] joint graded" >> results/program_b/phase2_overnight.log
