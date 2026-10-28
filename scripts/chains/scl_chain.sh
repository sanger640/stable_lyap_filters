cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes $1 "squeue -u ali313 -h -n scl | wc -l" 2>/dev/null || echo 1; }
until [ "$(n fir)" = "0" ] && true; do sleep 300; done
ARMS="record cbfg3m cbfgs"; P=results/program_b/policy_rollouts
for a in $ARMS; do
  mkdir -p $P/paired_${a}_scl
  rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_scl/ $P/paired_${a}_scl/
done
for a in $ARMS; do echo $a $(ls $P/paired_${a}_scl/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_scl; done)
AUDIT_OUT=results/jenga/outcome_audit_scl.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_scl; done) > results/jenga/outcome_audit_scl.log 2>&1
CL_TAG=scl CL_ARMS='record cbfg3m cbfgs' CL_PRIMARY=cbfgs,cbfg3m python3 eval/ocl_analysis.py > results/jenga/scl_analysis.log 2>&1
echo done $(date)
