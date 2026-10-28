#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
until [ -f results/jenga/vjepa_ac_zeroshot_result.json ] && ! pgrep -f "vjepa_ac_zeroshot.py eval" >/dev/null; do sleep 30; done
python3 eval/phase2_encoder_diag.py > results/program_b/phase2_encoder_diag.log 2>&1
git add results/jenga/phase2_encoder_diag_result.json && git commit -qm "Phase 2 amendment 7: DINOv2 vs iBOT encoder diagnostic on H2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
