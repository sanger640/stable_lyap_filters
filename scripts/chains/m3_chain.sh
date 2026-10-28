cd /home/corey/wksp/stable_lyap_filters
until [ "$(timeout 30 ssh -o BatchMode=yes rorqual "squeue -u ali313 -h -n m3 | wc -l" 2>/dev/null || echo 1)" = "0" ]; do sleep 300; done
ARMS="record cbfg cbfg3m cbfo3m"
for a in $ARMS; do rsync -a -e "ssh -o BatchMode=yes" rorqual:/scratch/ali313/slf/repo/results/program_b/policy_rollouts/paired_${a}_m3 results/program_b/policy_rollouts/; done
for a in $ARMS; do echo $a $(ls results/program_b/policy_rollouts/paired_${a}_m3/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_m3; done)
AUDIT_OUT=results/jenga/outcome_audit_m3.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_m3; done) > results/jenga/outcome_audit_m3.log 2>&1
CL_TAG=m3 CL_ARMS='record cbfg cbfg3m cbfo3m' CL_PRIMARY=cbfg3m,cbfg python3 eval/ocl_analysis.py > results/jenga/m3_analysis.log 2>&1
echo done $(date)
