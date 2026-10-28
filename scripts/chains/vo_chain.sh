cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n 272847)" = "0" ]; do sleep 300; done
mkdir -p results/jenga/deep_dive/vo_scl; rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/vo_scl/ results/jenga/deep_dive/vo_scl/
cat results/jenga/deep_dive/vo_scl/rows_*.jsonl | wc -l
python3 eval/vo_gate.py > results/jenga/vo_gate.log 2>&1; echo "vo done $(date)"
