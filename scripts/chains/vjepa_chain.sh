#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
until [ "$(stat -c %s ~/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt 2>/dev/null)" = "11760743310" ]; do sleep 30; done
until ! pgrep -f "vjepa_ac_zeroshot.py render" >/dev/null; do sleep 30; done
[ "$(ls results/program_b/vjepa_ac/frames | grep -c 'npz$')" -ge 160 ] || { echo "render incomplete" >> results/program_b/vjepa_ac_eval.log; exit 1; }
python3 eval/vjepa_ac_zeroshot.py eval > results/program_b/vjepa_ac_eval.log 2>&1
git add results/jenga/vjepa_ac_zeroshot_result.json && git commit -qm "V-JEPA 2-AC zero-shot vs US2_xlong on H2 (same rendered perturbed starts, BOTH readout)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
