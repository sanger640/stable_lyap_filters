#!/bin/bash
# Sequential restart after the 2026-10-03 13:02 power loss: GPU training alone, then panel 5 with 3 workers.
cd /home/corey/wksp/stable_lyap_filters
python3 eval/step3b_models.py > results/program_b/step3b_models.log 2>&1 || exit 1
python3 eval/step3b_panel5.py generate --workers 3 > results/program_b/panel5_gen.log 2>&1 || exit 1
if python3 -c "import json,sys; sys.exit(0 if json.load(open('results/jenga/step3b_p1_result.json'))['P1_pass'] else 1)"; then
  python3 eval/step3b_panel5.py grade > results/program_b/panel5_grade.log 2>&1
fi
