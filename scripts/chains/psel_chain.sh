cd /home/corey/wksp/stable_lyap_filters
until grep -q "trial ok" results/jenga/psel_launch.log; do grep -q "trial failed" results/jenga/psel_launch.log && exit 1; sleep 300; done; sleep 600
n() { timeout 30 ssh -o BatchMode=yes $1 "squeue -u ali313 -h -n psel | wc -l" 2>/dev/null || echo 1; }
until [ "$(n fir)" = "0" ] && true; do sleep 300; done
ARMS="cbfg3m cbfps3m"; P=results/program_b/policy_rollouts
for a in $ARMS; do
  mkdir -p $P/paired_${a}_psel
  rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_psel/ $P/paired_${a}_psel/
done
for a in $ARMS; do echo $a $(ls $P/paired_${a}_psel/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_psel; done)
AUDIT_OUT=results/jenga/outcome_audit_psel.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_psel; done) > results/jenga/outcome_audit_psel.log 2>&1
CL_TAG=psel CL_ARMS="cbfg3m cbfps3m" python3 eval/steer_analysis.py > results/jenga/psel_analysis.log 2>&1
echo done $(date)
