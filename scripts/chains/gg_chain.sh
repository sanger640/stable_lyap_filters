cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes $1 "squeue -u ali313 -h -n gg | wc -l" 2>/dev/null || echo 1; }
until [ "$(n fir)" = "0" ] && true; do sleep 300; done
ARMS="record cbfg3m cbfgg3m"; P=results/program_b/policy_rollouts
for a in $ARMS; do
  mkdir -p $P/paired_${a}_gg
  rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_gg/ $P/paired_${a}_gg/
done
for a in $ARMS; do echo $a $(ls $P/paired_${a}_gg/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_gg; done)
AUDIT_OUT=results/jenga/outcome_audit_gg.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_gg; done) > results/jenga/outcome_audit_gg.log 2>&1
CL_TAG=gg CL_ARMS='record cbfg3m cbfgg3m' CL_PRIMARY=cbfgg3m,cbfg3m python3 eval/ocl_analysis.py > results/jenga/gg_analysis.log 2>&1
echo done $(date)
