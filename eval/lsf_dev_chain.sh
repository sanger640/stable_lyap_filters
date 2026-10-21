#!/bin/bash
# After the LSF dev sweep (Nibi 24709409 tasks 0-69, Fir 66192391 tasks 70-99): copy back, merge, replay-audit, select eps (docs/LATENT_SAFETY_FILTER.md).
cd /home/corey/wksp/stable_lyap_filters
ARMS="record lsfe000 lsfe005 lsfe010 lsfe015 lsfe020 lsfe030"
for a in $ARMS; do
  rsync -a nibi:projects/def-soojeon/ali313/slf/repo/results/program_b/policy_rollouts/paired_${a}_lsfdev/ results/program_b/policy_rollouts/paired_${a}_lsfdev/
  rsync -a fir:/scratch/ali313/slf/repo/results/program_b/policy_rollouts/paired_${a}_lsfdev/ results/program_b/policy_rollouts/paired_${a}_lsfdev/
done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lsfdev; done)
AUDIT_OUT=results/jenga/outcome_audit_lsfdev.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lsfdev; done) 2>&1 | grep -v -i warn
python3 eval/lsf_dev_analysis.py
