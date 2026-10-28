#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_overnight.log
until ssh -o BatchMode=yes gpu5090 'grep -q "trained T_pix" ~/wksp/stable_lyap_filters/results/program_b/phase2_T_pix.log' 2>/dev/null; do sleep 120; done
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_T_pix.pt gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_T_pix_log.json results/program_b/phase2/
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2_T_pix.log results/program_b/
echo "[$(date '+%F %T')] T_pix copied" >> $L
until grep -q "trained T" results/program_b/phase2_T.log; do sleep 120; done
python3 eval/phase2_transformer.py grade > results/program_b/phase2_transformer_grade.log 2>&1 && echo "[$(date '+%F %T')] transformer arms graded" >> $L
git add results/jenga/phase2_transformer_result.json && git commit -qm "Phase 2 amendment 8: transformer arms T and T_pix graded on H2 (BOTH)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
