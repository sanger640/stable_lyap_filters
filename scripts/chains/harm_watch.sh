#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_overnight.log
until ssh -o BatchMode=yes gpu5090 'grep -q "trained US2_harm" ~/wksp/stable_lyap_filters/results/program_b/phase2_harm.log' 2>/dev/null; do sleep 120; done
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_US2_harm.pt gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_US2_harm_log.json results/program_b/phase2/
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2_harm.log results/program_b/
python3 eval/phase2_amend4.py grade_harm >> results/program_b/phase2_grade_harm.log 2>&1 && echo "[$(date '+%F %T')] US2_harm graded" >> $L
git add results/jenga/phase2_amend4_result.json && git commit -qm "Phase 2 diagnostic A: harm-label rebalancing US2_harm graded on H2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
