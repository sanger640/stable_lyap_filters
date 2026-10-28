cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n 274905)" = "0" ]; do sleep 300; done
mkdir -p results/jenga/deep_dive/vo_scl_l15_s4; rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/vo_scl_l15_s4/ results/jenga/deep_dive/vo_scl_l15_s4/
cat results/jenga/deep_dive/vo_scl_l15_s4/rows_*.jsonl | wc -l
python3 eval/vo_gate.py vo_scl_l15_s4 > results/jenga/vo15_gate.log 2>&1; echo "vo done $(date)"
