cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes $1 "squeue -u ali313 -h -n tr | wc -l" 2>/dev/null || echo 1; }
until [ "$(n fir)" = "0" ] && [ "$(n nibi)" = "0" ]; do sleep 300; done
ARMS="record cbfg cbfg3m cbfgtr"; P=results/program_b/policy_rollouts
for a in $ARMS; do
  mkdir -p $P/paired_${a}_tr
  rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_tr/ $P/paired_${a}_tr/
  rsync -a --exclude '*.tmp' nibi:projects/def-soojeon/ali313/slf/repo/$P/paired_${a}_tr/ $P/paired_${a}_tr/
done
for a in $ARMS; do echo $a $(ls $P/paired_${a}_tr/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_tr; done)
AUDIT_OUT=results/jenga/outcome_audit_tr.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_tr; done) > results/jenga/outcome_audit_tr.log 2>&1
CL_TAG=tr CL_ARMS='record cbfg cbfg3m cbfgtr' CL_PRIMARY=cbfgtr,cbfg python3 eval/ocl_analysis.py > results/jenga/tr_analysis.log 2>&1
echo done $(date)
