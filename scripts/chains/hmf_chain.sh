cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n 260589,260590)" = "0" ]; do sleep 300; done
for s in ocl scl; do mkdir -p results/jenga/deep_dive/hmf_$s; rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/hmf_$s/ results/jenga/deep_dive/hmf_$s/; cat results/jenga/deep_dive/hmf_$s/margin_rows_*.jsonl | wc -l; done
python3 eval/hmf_gate.py ocl > results/jenga/hmf_ocl.log 2>&1
python3 eval/hmf_gate.py scl > results/jenga/hmf_scl.log 2>&1
echo "hmf done $(date)"
