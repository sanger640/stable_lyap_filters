cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
# fix bench first (shorter)
until [ "$(n 257435,257436,257437)" = "0" ]; do sleep 300; done
mkdir -p results/jenga/deep_dive/fixbench
rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/fixbench/ results/jenga/deep_dive/fixbench/
wc -l results/jenga/deep_dive/fixbench/*.jsonl | tail -1
python3 eval/fix_bench.py analyse > results/jenga/fixbench_analysis.log 2>&1; echo "fixbench analysed $(date)"
until [ "$(n 257438)" = "0" ]; do sleep 300; done
ARMS="lsfs010 lsfs030 lsfs050 lsfs060 lsfs065 lsfs070"; P=results/program_b/policy_rollouts
for a in $ARMS; do mkdir -p $P/paired_${a}_lsfsdev; rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_lsfsdev/ $P/paired_${a}_lsfsdev/; echo $a $(ls $P/paired_${a}_lsfsdev/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lsfsdev; done)
while pgrep -f "[o]utcome_audit.py runs paired_cbfg3m_gtv" >/dev/null; do sleep 120; done     # one audit at a time
AUDIT_OUT=results/jenga/outcome_audit_lsfsdev.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lsfsdev; done) > results/jenga/outcome_audit_lsfsdev.log 2>&1
LSF_FAMILY=lsfs LSF_TAG=lsfsdev LSF_GRID="10 30 50 60 65 70" python3 eval/lsf_dev_analysis.py > results/jenga/lsfsdev_analysis.log 2>&1
echo "lsfsdev analysed $(date)"
