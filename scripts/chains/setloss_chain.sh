#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_overnight.log
until ssh -o BatchMode=yes gpu5090 'grep -q "trained SL_high" ~/wksp/stable_lyap_filters/results/program_b/phase2_SL_high.log' 2>/dev/null; do sleep 120; done
for a in SL_low SL_high; do rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_$a.pt gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_${a}_log.json results/program_b/phase2/; rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2_$a.log results/program_b/; done
until grep -q "trained C0" results/program_b/phase2_C0.log; do sleep 60; done
python3 eval/phase2_setloss.py grade > results/program_b/phase2_setloss_grade.log 2>&1 && echo "[$(date '+%F %T')] set-loss arms graded" >> $L
git add results/jenga/phase2_setloss_result.json && git commit -qm "Phase 2 amendment 9: set-loss arms C0 / SL_low / SL_high graded on H2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
