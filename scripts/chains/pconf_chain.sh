cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n 353507)" = "0" ]; do sleep 600; done
ARMS="record cbfps3m cbfps4m"; P=results/program_b/policy_rollouts
for a in $ARMS; do mkdir -p $P/paired_${a}_pconf; rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_pconf/ $P/paired_${a}_pconf/; echo $a $(ls $P/paired_${a}_pconf/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_pconf; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_pconf.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_pconf; done) > results/jenga/outcome_audit_pconf.log 2>&1
/home/corey/wksp/dp_venv/bin/python eval/pconf_analysis.py > results/jenga/pconf_analysis.log 2>&1
echo "pconf analysed $(date)"
