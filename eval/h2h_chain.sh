#!/bin/bash
# After the head-to-head (Fir 66521959): copy back, merge, replay-audit, analyse (results/jenga/lsf_h2h_protocol.json).
cd /home/corey/wksp/stable_lyap_filters
ARMS="record cbfg cbfh lsfh000"
for a in $ARMS; do
  rsync -a fir:/scratch/ali313/slf/repo/results/program_b/policy_rollouts/paired_${a}_h2h/ results/program_b/policy_rollouts/paired_${a}_h2h/
done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_h2h; done)
AUDIT_OUT=results/jenga/outcome_audit_h2h.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_h2h; done) 2>&1 | grep -v -i warn
python3 eval/h2h_analysis.py
