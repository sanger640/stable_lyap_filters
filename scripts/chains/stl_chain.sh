cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes $1 "squeue -u ali313 -h -n stl | wc -l" 2>/dev/null || echo 1; }
until [ "$(n fir)" = "0" ] && true; do sleep 300; done
ARMS="record cbfg3m cbfgst3m"; P=results/program_b/policy_rollouts
for a in $ARMS; do
  mkdir -p $P/paired_${a}_stl
  rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_stl/ $P/paired_${a}_stl/
done
for a in $ARMS; do echo $a $(ls $P/paired_${a}_stl/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_stl; done)
AUDIT_OUT=results/jenga/outcome_audit_stl.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_stl; done) > results/jenga/outcome_audit_stl.log 2>&1
CL_TAG=stl python3 eval/steer_analysis.py > results/jenga/stl_analysis.log 2>&1
echo done $(date)
