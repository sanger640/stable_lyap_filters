# usage: bash h2h3_chain.sh <array job id>
cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n $1)" = "0" ]; do sleep 600; done
ARMS="record cbfps3m lsfa030"; P=results/program_b/policy_rollouts
for a in $ARMS; do mkdir -p $P/paired_${a}_h2h3; rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_h2h3/ $P/paired_${a}_h2h3/; echo $a $(ls $P/paired_${a}_h2h3/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_h2h3; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done
AUDIT_OUT=results/jenga/outcome_audit_h2h3.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_h2h3; done) > results/jenga/outcome_audit_h2h3.log 2>&1
H2H_STAGE=h2h3 /home/corey/wksp/dp_venv/bin/python eval/h2h_analysis.py > results/jenga/h2h3_analysis.log 2>&1
echo "h2h3 analysed $(date)"
