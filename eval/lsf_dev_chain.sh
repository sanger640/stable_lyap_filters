#!/bin/bash
# After the Nibi LSF dev sweep (job 24705559): copy back, merge, replay-audit, select eps (docs/LATENT_SAFETY_FILTER.md).
cd /home/corey/wksp/stable_lyap_filters
ARMS="record lsfe000 lsfe005 lsfe010 lsfe015 lsfe020 lsfe030"
for a in $ARMS; do
  rsync -a nibi:projects/def-soojeon/ali313/slf/repo/results/program_b/policy_rollouts/paired_${a}_lsfdev/ results/program_b/policy_rollouts/paired_${a}_lsfdev/
done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lsfdev; done)
AUDIT_OUT=results/jenga/outcome_audit_lsfdev.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lsfdev; done) 2>&1 | grep -v -i warn
python3 eval/lsf_dev_analysis.py
