cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n 337330)" = "0" ]; do sleep 300; done
ARMS="lsfa010 lsfa030 lsfa050 lsfa060 lsfa065 lsfa070"; P=results/program_b/policy_rollouts
for a in $ARMS; do mkdir -p $P/paired_${a}_lsfadev; rsync -a --exclude '*.tmp' fir:/scratch/ali313/slf/repo/$P/paired_${a}_lsfadev/ $P/paired_${a}_lsfadev/; echo $a $(ls $P/paired_${a}_lsfadev/ep*.json | wc -l); done
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_lsfadev; done)
while pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 120; done     # one audit at a time
AUDIT_OUT=results/jenga/outcome_audit_lsfadev.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_lsfadev; done) > results/jenga/outcome_audit_lsfadev.log 2>&1
LSF_FAMILY=lsfa LSF_TAG=lsfadev LSF_GRID="10 30 50 60 65 70" python3 eval/lsf_dev_analysis.py > results/jenga/lsfadev_analysis.log 2>&1
echo "lsfadev analysed $(date)"
