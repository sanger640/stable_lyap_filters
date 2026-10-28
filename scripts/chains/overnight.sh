#!/bin/bash
# Overnight Phase 2 orchestration (user approved 1-3 on 2026-10-10).
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_overnight.log
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
commit(){ git add "$@" >/dev/null 2>&1; git commit -qm "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1 && log "committed: $MSG"; }
# 1) seed-1 replicate on the 5080 -> copy back -> grade on H2
until ssh -o BatchMode=yes gpu5090 'grep -q "trained US2_long_s1" ~/wksp/stable_lyap_filters/results/program_b/phase2_long_s1.log' 2>/dev/null; do sleep 120; done
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_US2_long_s1.pt gpu5090:wksp/stable_lyap_filters/results/program_b/phase2/model_US2_long_s1_log.json results/program_b/phase2/
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/phase2_long_s1.log results/program_b/
log "seed-1 model copied"
python3 eval/phase2_amend4.py grade_s1 >> results/program_b/phase2_grade_s1.log 2>&1 && log "seed-1 graded"
MSG="Phase 2 amendment 5 addendum: US2_long seed-1 replicate graded on H2"; commit results/jenga/phase2_amend4_result.json
# 2) wait for xlong + its H2 grade
until python3 -c "import json,sys;sys.exit(0 if 'amendment5_grade_H2' in json.load(open('results/jenga/phase2_amend4_result.json')) else 1)" 2>/dev/null; do sleep 120; done
log "xlong graded"
MSG="Phase 2 amendment 5: US2_xlong graded on H2 (BOTH)"; commit results/jenga/phase2_amend4_result.json
# re-grade the seed-1 replicate beside xlong (same table)
python3 eval/phase2_amend4.py grade_s1 >> results/program_b/phase2_grade_s1.log 2>&1
MSG="Phase 2: seed-1 replicate re-graded beside US2_xlong"; commit results/jenga/phase2_amend4_result.json
# 3) panels only if the rule passes
if python3 -c "import json,sys;sys.exit(0 if json.load(open('results/jenga/phase2_amend4_result.json'))['amendment5_grade_H2']['best_passes_in_family_bar_on_H2'] else 1)"; then
  log "rule passed: panels"
  while pgrep -f "phase2_models.py panelscreen" >/dev/null; do sleep 120; done
  python3 eval/phase2_panels.py >> results/program_b/phase2_panels.log 2>&1 && log "panels graded"
  MSG="Phase 2 fresh panels (in-family, held-out F3) graded"; commit results/jenga/phase2_panels_result.json
else
  log "rule not met: stop (no panels)"
fi
log "overnight done"
